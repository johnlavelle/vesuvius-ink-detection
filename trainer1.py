import pprint
from functools import partial
from itertools import islice

import dask
import torch

from src import tensorboard_access
from vesuvius.ann.criterions import FocalLoss
from vesuvius.ann.models import HybridModel
from vesuvius.ann.optimisers import SGDOneCycleLR
from vesuvius.dataloader import test_loader, get_train_loaders
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxSobol
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer
from vesuvius.utils import timer


# If READ_EXISTING_CONFIG is False, config is specified in Configuration (below)
# else config is read from CONFIG_PATH.
READ_CONFIG_FILE = False
CONFIG_PATH = 'configs/config.json'

pp = pprint.PrettyPrinter(indent=4)
dask.config.set(scheduler='synchronous')


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
#
#
# def get_config_model(config_path: str, model_path: str) -> Tuple[Configuration, torch.nn.Module]:
#     lm = LoadModel(config_path, model_path)
#     return convert_config(lm.config()), lm.model()
#
#
# if READ_CONFIG_FILE:
#     loader = LoadModel('output/runs/2023-04-14_17-37-44/', 1)
#     model1 = loader.model()
#     config1 = loader.config
# else:
#     # Hold back data test box for fragment
#     XL, YL = 2048, 7168  # lower left corner of the test box
#     WIDTH, HEIGHT = 2045, 2048
#
# def get_train2_config(config) -> Configuration1:
#     cfg = copy.copy(config)
#     cfg.suffix_cache = 'regular'
#     cfg.crop_box_cls = CropBoxRegular
#     # Keep shuffle = False, so the dataloader does not shuffle, to ensure you get all the z bins for each (x, y).
#     # The data will already be shuffled w.r.t. (x, y), per fragment. The cached dataset will be completely shuffled.
#     cfg.shuffle = False
#     cfg.group_pixels = True
#     cfg.balance_ink = True
#     cfg.sampling = 5  # TODO: delete this?
#     cfg.stride_xy = 61
#     cfg.stride_z = 6
#     return cfg


# Training


# Data Processing


class Trainer1(BaseTrainer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_train_test_loaders(self) -> None:
        self.test_loader_iter = lambda length: islice(self.test_loader(self.config), length)
        self.train_loader_iter = self.train_loader(self.config, self.epochs)
        self.train_loader_iter = islice(self.train_loader_iter, self.config.training_steps)

    def _apply_forward(self, datapoint) -> torch.Tensor:
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model(datapoint.voxels.to(self.device), scalar.to(self.device))

    def validate(self) -> 'BaseTrainer':
        self.model.eval()
        with torch.no_grad():
            for datapoint_test in self.test_loader_iter(self.validation_steps):
                outputs = self._apply_forward(datapoint_test)
                val_loss = self.criterion_validate(outputs, datapoint_test.label.float().to(self.device))
                batch_size = len(datapoint_test.label)
                self.trackers.logger_test_loss.update(val_loss.item(), batch_size)
        self.trackers.log_test()
        return self

    def forward(self) -> 'BaseTrainer':
        self.model.train()
        self.optimizer.zero_grad()
        self.outputs = self._apply_forward(self.datapoint)
        return self

    def loss(self) -> 'BaseTrainer':
        target = self.datapoint.label.float().to(self.device)

        base_loss = self.criterion(self.outputs, target)
        l1_regularization = torch.norm(self.model.fc_scalar.weight, p=1)
        loss = base_loss + (self.l1_lambda * l1_regularization)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        batch_size = len(self.datapoint.voxels)
        self.trackers.logger_loss.update(loss.item(), batch_size)
        self.trackers.logger_lr.update(self.scheduler.get_last_lr()[0], batch_size)
        return self


xl, yl = 2048, 7168  # lower left corner of the test box
width, height = 2045, 2048
config_kwargs = dict(training_steps=32 * (40000 // 32) - 1,
                     test_box=(xl, yl, xl + width, yl + height),  # Hold back rectangle
                     test_box_fragment=2,  # Hold back fragment
                     model=HybridModel,
                     volume_dataset_cls=SampleXYZ,
                     crop_box_cls=CropBoxSobol)

if __name__ == '__main__':

    pp = pprint.PrettyPrinter(indent=4)
    dask.config.set(scheduler='synchronous')
    print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')

    FORCE_CACHE_RESET = False  # Deletes cache. Only used if CACHED_DATA is True.
    EPOCHS = 2
    RESET_CACHE_EPOCH_INTERVAL = EPOCHS
    SAVE_INTERVAL = 1_000_000
    VALIDATE_INTERVAL = 1000
    LOG_INTERVAL = 100

    train_loaders = partial(
        get_train_loaders,
        cached_data=True,
        force_cache_reset=FORCE_CACHE_RESET,
        reset_cache_epoch_interval=RESET_CACHE_EPOCH_INTERVAL)

    for alpha, gamma in [(1, 0), (0.25, 2), (0.5, 2), (0.75, 2)]:

        criterion = FocalLoss(alpha=alpha, gamma=gamma)

        with Track() as track, timer("Training"):

            trainer1 = Trainer1(train_loaders,
                                test_loader,
                                track,
                                SGDOneCycleLR,
                                criterion,
                                config_kwargs=config_kwargs,
                                learning_rate=0.03,
                                l1_lambda=0,
                                epochs=EPOCHS)

            for i, train in enumerate(trainer1):
                train.forward().loss()

                if i == 0:
                    continue
                if i % LOG_INTERVAL == 0:
                    train.trackers.log_train()
                if i % VALIDATE_INTERVAL == 0:
                    train.validate()

            train.save_model_output()
