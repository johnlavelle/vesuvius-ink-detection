import copy
import json
import multiprocessing as mp
import os
import pprint
import time
from abc import ABC, abstractmethod
from itertools import islice
from timeit import default_timer as timer
from typing import Generator, Any, Iterator
import gc

import dask
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from xarray import DataArray

import tensorboard_access
from vesuvius.config import Configuration
from vesuvius.data_io import SaveModel
from vesuvius.dataloader import get_train_loader, get_test_loader
from vesuvius.datapoints import Datapoint, DatapointTuple
from vesuvius.fragment_dataset import BaseDataset
from vesuvius.sampler import CropBoxSobol, CropBoxRegular
from vesuvius.trackers import TrackerAvg, Incrementer

# If READ_EXISTING_CONFIG is False, config is specified in Configuration (below)
# else config is read from CONFIG_PATH.
READ_CONFIG_FILE = False
CONFIG_PATH = 'configs/config.json'


# Data Processing

def centre_pixel(da: DataArray) -> DataArray:
    return da.isel(x=len(da.x) // 2, y=len(da.y) // 2).astype(np.float32)


class SampleXYZ(BaseDataset):

    def get_datapoint(self, index: int) -> DatapointTuple:
        rnd_slice = self.get_volume_slice(index)
        slice_xy = {key: value for key, value in rnd_slice.items() if key in ('x', 'y')}
        label = self.ds.labels.sel(**slice_xy).transpose('x', 'y')
        voxels = self.ds.images.sel(**rnd_slice).transpose('z', 'x', 'y')
        voxels = voxels.expand_dims('Cin')
        # voxels = self.normalise_voxels(voxels)
        # voxels = voxels.expand_dims('Cin')
        s = rnd_slice
        dp = Datapoint(voxels, int(self.label_operation(label)), self.ds.fragment,
                       s['x'].start, s['x'].stop, s['y'].start, s['y'].stop, s['z'].start, s['z'].stop)
        return dp.to_namedtuple()


# Models


class CNN1(nn.Module):

    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.LazyLinear(128)
        self.relu = nn.ReLU()  # Use a single ReLU instance
        self.fc2 = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def cnn1_sequential():
    return nn.Sequential(
        nn.Conv3d(1, 16, 3, 1, 1),
        nn.MaxPool3d(2, 2),
        nn.Conv3d(16, 32, 3, 1, 1),
        nn.MaxPool3d(2, 2),
        nn.Conv3d(32, 64, 3, 1, 1),
        nn.MaxPool3d(2, 2),
        nn.Flatten(start_dim=1),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.LazyLinear(1),
        nn.Sigmoid())


class HybridModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(HybridModel, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(16)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.MaxPool3d(2, 2)

        self.conv2 = nn.Conv3d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.pool2 = nn.MaxPool3d(2, 2)

        self.conv3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(64)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.pool3 = nn.AdaptiveMaxPool3d((8, 8, 8))

        self.flatten = nn.Flatten(start_dim=1)

        # FCN part for scalar input
        self.fc_scalar = nn.Linear(1, 16)
        self.bn_scalar = nn.BatchNorm1d(16)
        self.dropout_scalar = nn.Dropout(dropout_rate)

        # Combined layers (initialized later)
        self.fc_combined1 = nn.Linear(64 * 8 * 8 * 8 + 16, 128)
        self.bn_combined = nn.BatchNorm1d(128)
        self.dropout_combined = nn.Dropout(dropout_rate)

        self.fc_combined2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, scalar_input: torch.Tensor):
        # CNN part
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        # FCN part
        x_scalar = self.fc_scalar(scalar_input)
        x_scalar = self.bn_scalar(x_scalar)
        scalar_out = F.relu(x_scalar)

        # Combine CNN and FCN outputs
        combined = torch.cat((x, scalar_out), dim=1)

        # Combined layers
        x = self.fc_combined1(combined)
        x = self.bn_combined(x)
        x = F.relu(x)
        x = self.dropout_combined(x)
        x = self.fc_combined2(x)

        return self.sigmoid(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def get_focal_weights(self, prob: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss weights."""
        return (1 - prob) ** self.gamma

    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        # Apply the sigmoid function to get probabilities
        prob = torch.sigmoid(input)

        # Compute the binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # Get the focal loss weights
        focal_weights = self.get_focal_weights(prob)

        # Apply the focal weights
        focal_loss = self.alpha * focal_weights * bce_loss

        # Reduce the loss (mean, sum or none)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Configuration


# def convert_config(cfg: dataclass) -> dataclass:
#     """Convert the string representation of non-basic data types
#     in a dataclass object to their corresponding objects."""
#     for key, type_ in config1.__annotations__.items():
#         if type_ not in (str, int, float, bool, list, tuple, dict):
#             value = getattr(cfg, key)
#             if isinstance(value, str):
#                 setattr(cfg, key, eval(value))
#     return cfg


# def get_config_model(config_path: str, model_path: str) -> Tuple[Configuration, torch.nn.Module]:
#     lm = LoadModel(config_path, model_path)
#     return convert_config(lm.config()), lm.model()


# if READ_CONFIG_FILE:
#     loader = LoadModel('output/runs/2023-04-14_17-37-44/', 1)
#     model1 = loader.model()
#     config1 = loader.config
# else:
#     # Hold back data test box for fragment
#     XL, YL = 2048, 7168  # lower left corner of the test box
#     WIDTH, HEIGHT = 2045, 2048
#
#     print('Getting config from Configuration instantiation...\n')
#     config1 = Configuration(info="",
#                             model=HybridModel,
#                             volume_dataset_cls=SampleXYZ,
#                             crop_box_cls=CropBoxSobol,
#                             label_fn=centre_pixel,
#                             training_steps=32 * (40_000 // 32) - 1,  # This should be small enough to fit on disk
#                             batch_size=32,
#                             fragments=[1, 2, 3],
#                             test_box=(XL, YL, XL + WIDTH, YL + HEIGHT),  # Hold back rectangle
#                             test_box_fragment=2,  # Hold back fragment
#                             box_width_xy=91,
#                             box_width_z=6,
#                             balance_ink=True,
#                             shuffle=True,
#                             group_pixels=False,
#                             num_workers=min(1, mp.cpu_count() - 1),
#                             prefix='/data/kaggle/input/vesuvius-challenge-ink-detection/train/',
#                             suffix_cache='sobol',
#                             collate_fn=None)


def get_train2_config(config) -> Configuration:
    cfg = copy.copy(config)
    cfg.suffix_cache = 'regular'
    cfg.crop_box_cls = CropBoxRegular
    # Keep shuffle = False, so the dataloader does not shuffle, to ensure you get all the z bins for each (x, y).
    # The data will already be shuffled w.r.t. (x, y), per fragment. The cached dataset will be completely shuffled.
    cfg.shuffle = False
    cfg.group_pixels = True
    cfg.balance_ink = True
    cfg.sampling = 5  # TODO: delete this?
    cfg.stride_xy = 61
    cfg.stride_z = 6
    return cfg


# Training


def train_loaders(cfg) -> Generator[Datapoint, None, None]:
    for epoch in range(EPOCHS):
        reset_cache = CACHED_DATA and (epoch != 0 and epoch % RESET_CACHE_EPOCH_INTERVAL == 0)
        reset_cache = reset_cache or (epoch == 0 and FORCE_CACHE_RESET)
        yield from get_train_loader(cfg, cached=CACHED_DATA, reset_cache=reset_cache)


class BaseTrainer(ABC):

    def __init__(self,
                 epochs: int = 1,
                 learning_rate: float = 0.03,
                 l1_lambda: float = 0.001,
                 criterion_kwargs: dict = None,
                 model_kwargs: dict = None,
                 config_kwargs: dict = None) -> None:

        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        if criterion_kwargs is None:
            criterion_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if config_kwargs is None:
            config_kwargs = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_subdir = os.path.join("output/runs", current_time)
        self.saver = SaveModel(log_subdir, 1)
        os.makedirs(log_subdir, exist_ok=True)
        self.writer = SummaryWriter(log_subdir, flush_secs=60)
        self.incrementer = Incrementer()
        self.logger_loss = TrackerAvg('loss/train', self.writer)
        self.logger_test_loss = TrackerAvg('loss/test', self.writer)
        self.logger_lr = TrackerAvg('stats/lr', self.writer)

        self.config = self.get_config(**config_kwargs)
        self.model = self.config.model(**model_kwargs).to(self.device)
        self.batch_size = self.config.batch_size
        self.total = epochs * self.config.training_steps
        self.loops = self.total // self.batch_size
        self.check_model()
        pp.pprint(self.config)
        print()

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer, self.loops)
        self.criterion = self.get_criterion(**criterion_kwargs)
        self.criterion_val = nn.BCEWithLogitsLoss()

        self.train_loader_iter = self.train_loaders()
        self.test_loader_iter = get_test_loader(self.config)

    @abstractmethod
    def get_criterion(self, **kwargs) -> nn.Module:
        ...

    @abstractmethod
    def get_scheduler(self, optimizer, total) -> Any:
        ...

    @abstractmethod
    def get_optimizer(self) -> Optimizer:
        ...

    @abstractmethod
    def process_model_output(self, datapoint) -> torch.Tensor:
        ...

    @abstractmethod
    def check_model(self) -> None:
        ...

    @staticmethod
    @abstractmethod
    def get_config(**kwargs) -> Configuration:
        ...

    @abstractmethod
    def train_loaders(self) -> Generator[Datapoint, None, None]:
        ...

    @abstractmethod
    def validate(self, i) -> None:
        ...

    @abstractmethod
    def get_loss(self) -> float:
        ...

    def loss_generator(self) -> Iterator[float]:
        while self.incrementer.loop < self.loops:
            yield self.get_loss()

    def __next__(self) -> float:
        if self.incrementer.loop >= self.loops:
            raise StopIteration
        return self.get_loss()

    def __iter__(self):
        tqdm_kwargs = dict(total=self.loops, disable=False, desc='Training', position=0)
        yield from tqdm(self.loss_generator(), **tqdm_kwargs)

    def __enter__(self):
        self.time_start = timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.config.update_nn_kwargs(self.optimizer, self.scheduler, self.criterion, self.learning_rate, EPOCHS)
        self.config['performance_dict'] = {'loss/train': self.logger_loss.average,
                                           'loss/test': self.logger_test_loss.average,
                                           'steps': self.incrementer.count}

        self.saver.model(self.model)
        json_file_path = self.saver.config(self.config)
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        config_json = json.dumps(data, indent=4)
        self.writer.add_text('config', config_json)
        self.writer.flush()
        self.writer.close()

        torch.cuda.empty_cache()
        gc.collect()

    def __str__(self) -> str:
        if self.time_start is not None:
            time_end = timer()
            time_spent = time_end - self.time_start
            return f"Training took {time_spent:.1f} seconds"
        else:
            return "Training not started"


class TrainerScalar1(BaseTrainer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_config(**kwargs):
        xl, yl = 2048, 7168  # lower left corner of the test box
        width, height = 2045, 2048
        default_config = dict(info="",
                              model=HybridModel,
                              volume_dataset_cls=SampleXYZ,
                              crop_box_cls=CropBoxSobol,
                              label_fn=centre_pixel,
                              training_steps=32 * (40_000 // 32) - 1,  # This should be small enough to fit on disk
                              batch_size=32,
                              fragments=[1, 2, 3],
                              test_box=(xl, yl, xl + width, yl + height),  # Hold back rectangle
                              test_box_fragment=2,  # Hold back fragment
                              box_width_xy=91,
                              box_width_z=6,
                              balance_ink=True,
                              shuffle=True,
                              group_pixels=False,
                              num_workers=min(1, mp.cpu_count() - 1),
                              prefix='/data/kaggle/input/vesuvius-challenge-ink-detection/train/',
                              suffix_cache='sobol',
                              collate_fn=None)
        return Configuration(**{**default_config, **kwargs})

    def train_loaders(self) -> Generator[Datapoint, None, None]:
        yield from train_loaders(self.config)

    def validate(self, i) -> None:
        for datapoint_test in islice(self.train_loaders(), 20):
            outputs = self.process_model_output(datapoint_test)
            val_loss = self.criterion_val(outputs, datapoint_test.label.float().to(self.device))
            self.logger_test_loss.update(val_loss.item(), len(datapoint_test.voxels))
        self.logger_test_loss.log(i)

    def get_loss(self) -> float:
        if self.incrementer.loop == self.loops:
            raise StopIteration

        datapoint = next(self.train_loader_iter)

        self.model.train()

        self.optimizer.zero_grad()

        outputs = self.process_model_output(datapoint)
        target = datapoint.label.float().to(self.device)
        base_loss = self.criterion(outputs, target)
        l1_regularization = torch.norm(self.model.fc_scalar.weight, p=1)
        loss = base_loss + (self.l1_lambda * l1_regularization)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.logger_loss.update(loss.item(), len(datapoint.voxels))
        self.logger_lr.update(self.scheduler.get_last_lr()[0], 1)
        self.incrementer.increment(len(outputs))

        if self.incrementer.loop != 0:
            if self.incrementer.loop % SAVE_INTERVAL == 0:
                self.saver.model(self.model)
                self.saver.config(self.config)

            if self.incrementer.loop % LOG_INTERVAL == 0:
                self.logger_loss.log(self.incrementer.count)
                self.logger_lr.log(self.incrementer.count)

            if self.incrementer.loop % VALIDATE_INTERVAL == 0:
                try:
                    self.model.eval()
                    with torch.no_grad():
                        self.validate(self.incrementer.count)
                    self.model.train()
                except AttributeError:
                    print('Validation code not compatible with this version of Python')

        return loss.item()

    def get_criterion(self, **kwargs) -> nn.Module:
        return FocalLoss(**kwargs)

    def get_scheduler(self, optimizer, total) -> Any:
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=total)

    def get_optimizer(self) -> Optimizer:
        return optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def check_model(self) -> None:
        rnd_vals = torch.randn(self.config.batch_size,
                               1,
                               self.config.box_width_z,
                               self.config.box_width_xy,
                               self.config.box_width_xy).to(self.device)
        print(f'Shape of sample batches: {rnd_vals.shape}', '\n')
        try:
            scalar_input = torch.tensor([2.0] * self.config.batch_size).to(self.device).view(-1, 1).to(self.device)
            self.model.eval()
            with torch.no_grad():
                output = self.model.forward(rnd_vals, scalar_input)
                assert output.shape == (self.config.batch_size, 1)
        except RuntimeError as e:
            raise RuntimeError(f'{e}\n'
                               'Model is not compatible with the input data size')

    def process_model_output(self, datapoint) -> torch.Tensor:
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model(datapoint.voxels.to(self.device), scalar.to(self.device))


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    dask.config.set(scheduler='synchronous')

    CACHED_DATA = True
    FORCE_CACHE_RESET = False  # Deletes cache. Only used if CACHED_DATA is True.
    EPOCHS = 30
    RESET_CACHE_EPOCH_INTERVAL = EPOCHS
    SAVE_INTERVAL = 1_000_000
    VALIDATE_INTERVAL = 200
    LOG_INTERVAL = 10

    print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    for alpha, gamma in [(1, 0), (0.25, 2), (0.5, 2), (0.75, 2)]:
        print(f'alpha={alpha}, gamma={gamma}')
        with TrainerScalar1(EPOCHS,
                            l1_lambda=0,
                            criterion_kwargs=dict(alpha=alpha, gamma=gamma),
                            config_kwargs=dict(training_steps=32 * (1_00 // 32) - 1)) as trainer1:
            list(trainer1)
    print(trainer1)
