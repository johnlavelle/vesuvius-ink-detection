import copy
import json
import multiprocessing as mp
import os
import pprint
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from itertools import islice
from timeit import default_timer as timer
from typing import Generator, Tuple, Any

import dask
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as func
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from xarray import DataArray

from vesuvius.config import Configuration
from vesuvius.data_io import SaveModel, LoadModel
from vesuvius.dataloader import get_train_loader, get_test_loader
from vesuvius.datapoints import Datapoint, DatapointTuple
from vesuvius.sampler import CropBoxSobol, CropBoxRegular
from vesuvius.fragment_dataset import BaseDataset
from vesuvius.trackers import TrackerAvg
from vesuvius.utils import Incrementer
import tensorboard_access


pp = pprint.PrettyPrinter(indent=4)
dask.config.set(scheduler='synchronous')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using the {DEVICE}\n')

# If READ_EXISTING_CONFIG is False, config is specified in Configuration (below)
# else config is read from CONFIG_PATH.
READ_CONFIG_FILE = False
CONFIG_PATH = 'configs/config.json'

CACHED_DATA = True
FORCE_CACHE_RESET = False  # Deletes cache. Only used if CACHED_DATA is True.
EPOCHS = 200
RESET_CACHE_EPOCH_INTERVAL = 4
SAVE_INTERVAL = 1_000_000
VALIDATE_INTERVAL = 1_000
LOG_INTERVAL = 100


# Data Processing

def centre_pixel(da: DataArray) -> DataArray:
    return da.isel(x=len(da.x) // 2, y=len(da.y) // 2).astype(np.float32)


class SampleXYZ(BaseDataset):

    def get_datapoint(self, index: int) -> DatapointTuple:
        rnd_slice = self.get_volume_slice(index)
        slice_xy = {key: value for key, value in rnd_slice.items() if key in ('x', 'y')}
        label = self.ds.labels.sel(**slice_xy).transpose('x', 'y')
        voxels = self.ds.images.sel(**rnd_slice).transpose('z', 'x', 'y')
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


class HybridModel1(nn.Module):
    def __init__(self, channels):
        super(HybridModel1, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3 = nn.AdaptiveMaxPool2d((8, 8))
        self.flatten = nn.Flatten(start_dim=1)

        # FCN part for scalar input
        self.fc_scalar = nn.Linear(1, 16)

        # Combined layers (initialized later)
        self.fc_combined1 = nn.Linear(64 * 8 * 8 + 16, 128)
        self.relu = nn.ReLU()
        self.fc_combined2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, scalar_input: torch.Tensor):
        # CNN part
        x = self.pool1(func.relu(self.conv1(x)))
        x = self.pool2(func.relu(self.conv2(x)))
        x = self.pool3(func.relu(self.conv3(x)))
        x = self.flatten(x)

        # FCN part
        scalar_out = func.relu(self.fc_scalar(scalar_input))

        # Combine CNN and FCN outputs
        combined = torch.cat((x, scalar_out), dim=1)

        # Combined layers
        x = self.relu(self.fc_combined1(combined))
        x = self.sigmoid(self.fc_combined2(x))

        return x


# Configuration


def convert_config(cfg: dataclass) -> dataclass:
    """Convert the string representation of non-basic data types
    in a dataclass object to their corresponding objects."""
    for key, type_ in config1.__annotations__.items():
        if type_ not in (str, int, float, bool, list, tuple, dict):
            value = getattr(cfg, key)
            if isinstance(value, str):
                setattr(cfg, key, eval(value))
    return cfg


def get_config_model(config_path: str, model_path: str) -> Tuple[Configuration, torch.nn.Module]:
    lm = LoadModel(config_path, model_path)
    return convert_config(lm.config()), lm.model()


if READ_CONFIG_FILE:
    loader = LoadModel('output/runs/2023-04-14_17-37-44/', 1)
    model1 = loader.model()
    config1 = loader.config
else:
    # Hold back data test box for fragment
    XL, YL = 1100, 3500  # lower left corner of the test box
    WIDTH, HEIGHT = 2045, 2048

    print('Getting config from Configuration instantiation...\n')
    config1 = Configuration(info="",
                            model=HybridModel1,
                            volume_dataset_cls=SampleXYZ,
                            crop_box_cls=CropBoxSobol,
                            label_fn=centre_pixel,
                            training_steps=100_000,  # If caching, this should be small enough to fit on disk
                            batch_size=32,
                            fragments=[1, 2, 3],
                            test_box=(XL, YL, XL + WIDTH, YL + HEIGHT),  # Hold back rectangle
                            test_box_fragment=1,  # Hold back fragment
                            box_width_xy=61,
                            box_width_z=6,
                            balance_ink=True,
                            shuffle=True,
                            group_pixels=False,
                            num_workers=min(1, mp.cpu_count() - 1),
                            prefix='/data/kaggle/vesuvius/train/',
                            suffix_cache='sobol',
                            collate_fn=None)


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

    def __init__(self, config, model, learning_rate, device):
        self.config = config
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.device = device
        self.incrementer = Incrementer()

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_subdir = os.path.join("output/runs", current_time)
        self.saver = SaveModel(log_subdir, 1)
        os.makedirs(log_subdir, exist_ok=True)
        self.writer = SummaryWriter(log_subdir, flush_secs=60)
        self.logger_loss = TrackerAvg('loss/train', self.writer)
        self.logger_test_loss = TrackerAvg('loss/test', self.writer)
        self.logger_lr = TrackerAvg('stats/lr', self.writer)

        self.time_start = None
        self.total = EPOCHS * self.config.training_steps // self.config.batch_size

    @abstractmethod
    def get_criterion(self) -> nn.Module:
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

    def train_loaders(self) -> Generator[Datapoint, None, None]:
        yield from train_loaders(self.config)

    def validate(self, dataloader, criterion, i) -> None:
        self.model.eval()
        with torch.no_grad():
            for datapoint_test in islice(dataloader, 20):
                outputs = self.process_model_output(datapoint_test)
                val_loss = criterion(outputs, datapoint_test.label.float().to(self.device))
                self.logger_test_loss.update(val_loss.item(), len(datapoint_test.voxels))
        self.logger_test_loss.update(val_loss.item(), len(datapoint_test.voxels))
        self.logger_test_loss.log(i)

    def train(self, dataloader, criterion, optimizer, scheduler, validate_fn) -> None:
        self.model.train()
        tqdm_kwargs = dict(total=self.total, disable=False, desc='Training', position=0)
        for datapoint in tqdm(dataloader, **tqdm_kwargs):
            if self.incrementer == self.total:
                break

            optimizer.zero_grad()
            outputs = self.process_model_output(datapoint)
            loss = criterion(outputs, datapoint.label.float().to(self.device))
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.logger_loss.update(loss.item(), len(datapoint.voxels))
            self.logger_lr.update(scheduler.get_last_lr()[0], 1)
            self.incrementer.increment(len(outputs))

            if self.incrementer.value == 0:
                continue

            if self.incrementer.value % SAVE_INTERVAL == 0:
                self.saver.model(model1)
                self.saver.config(config1)

            if self.incrementer.value % LOG_INTERVAL == 0:
                self.logger_loss.log(self.incrementer.value)
                self.logger_lr.log(self.incrementer.value)

            if self.incrementer.value % VALIDATE_INTERVAL == 0:
                validate_fn(self.incrementer.value)

    def run(self) -> None:
        train_loader_sobol = self.train_loaders()
        test_loader_sobol = get_test_loader(self.config)

        total = EPOCHS * self.config.training_steps // self.config.batch_size

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer, total)
        criterion = self.get_criterion()

        validate_i = partial(self.validate, test_loader_sobol, criterion)
        self.train(train_loader_sobol, criterion, optimizer, scheduler, validate_i)

        self.config.update_nn_kwargs(optimizer, scheduler, criterion, self.learning_rate, EPOCHS)
        self.config['performance_dict'] = {'loss/train': self.logger_loss.average,
                                           'loss/test': self.logger_test_loss.average,
                                           'steps': self.incrementer.value}

    def validate_model(self) -> None:
        rnd_vals = torch.randn(self.config.batch_size,
                               1,
                               self.config.box_width_z,
                               self.config.box_width_xy,
                               self.config.box_width_xy).to(DEVICE)
        print(f'Shape of sample batches: {rnd_vals.shape}', '\n')
        try:
            self.model.eval()
            with torch.no_grad():
                self.model.forward(rnd_vals)
        except RuntimeError as e:
            raise RuntimeError(f'{e}\n'
                               'Model is not compatible with the input data size')

    def __enter__(self):
        pp.pprint(self.config)
        print()
        print(self.model)
        print()
        self.validate_model()
        self.time_start = timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.saver.model(self.model)
        json_file_path = self.saver.config(self.config)
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        config_json = json.dumps(data, indent=4)
        self.writer.add_text('config', config_json)
        self.writer.flush()
        self.writer.close()

    def __str__(self) -> str:
        if self.time_start is not None:
            time_end = timer()
            time_spent = time_end - self.time_start
            return f"Training took {time_spent:.1f} seconds"
        else:
            return "Training not started"


class Trainer1(BaseTrainer):

    def get_criterion(self) -> nn.Module:
        return nn.BCELoss()

    def get_scheduler(self, optimizer, total) -> Any:
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=total)

    def get_optimizer(self) -> Optimizer:
        return optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def process_model_output(self, datapoint) -> torch.Tensor:
        return self.model(datapoint.voxels.to(self.device))


class TrainerScalar1(Trainer1):

    def validate_model(self) -> None:
        rnd_vals = torch.randn(self.config.batch_size,
                               self.config.box_width_z,
                               self.config.box_width_xy,
                               self.config.box_width_xy).to(DEVICE)
        print(f'Shape of sample batches: {rnd_vals.shape}', '\n')
        try:
            scalar_input = torch.tensor([2.0] * self.config.batch_size).to(self.device).view(-1, 1).to(self.device)
            self.model.eval()
            with torch.no_grad():
                self.model.forward(rnd_vals, scalar_input)
        except RuntimeError as e:
            raise RuntimeError(f'{e}\n'
                               'Model is not compatible with the input data size')

    def process_model_output(self, datapoint) -> torch.Tensor:
        return self.model(datapoint.voxels.to(self.device),
                          datapoint.z_start.view(-1, 1).float().to(self.device))


if __name__ == '__main__':
    print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    model1 = config1.model(channels=config1.box_width_z)
    with TrainerScalar1(config1, model1, 0.03, DEVICE) as trainer1:
        trainer1.run()
    print(trainer1)
