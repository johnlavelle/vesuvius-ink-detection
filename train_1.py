import pprint
from functools import partial
from itertools import islice

import dask
import torch
from torch.nn import BCEWithLogitsLoss

from src import tensorboard_access
from vesuvius import ann
from vesuvius.ann import models
from vesuvius.ann.criterions import FocalLoss
from vesuvius.config import Configuration
from vesuvius.config import ConfigurationModel
from vesuvius.dataloader import test_loader, get_train_loaders
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxRegular
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer
from vesuvius.trainer import centre_pixel
from vesuvius.utils import timer

# If READ_EXISTING_CONFIG is False, config is specified in Configuration (below)
# else config is read from CONFIG_PATH.
READ_CONFIG_FILE = False
CONFIG_PATH = 'configs/config.json'

pp = pprint.PrettyPrinter(indent=4)
dask.config.set(scheduler='synchronous')


class Trainer1(BaseTrainer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_train_test_loaders(self) -> None:
        self.test_loader_iter = lambda length: islice(self.test_loader(self.config), length)
        self.train_loader_iter = self.train_loader(self.config, self.epochs)
        self.train_loader_iter = islice(self.train_loader_iter, self.config.model0.total_steps)

    def _apply_forward(self, datapoint) -> torch.Tensor:
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model0(datapoint.voxels.to(self.device), scalar.to(self.device))

    def validate(self) -> 'BaseTrainer':
        self.model0.eval()
        with torch.no_grad():
            for datapoint_test in self.test_loader_iter(self.config.validation_steps):
                outputs = self._apply_forward(datapoint_test)
                val_loss = self.criterion0(outputs, datapoint_test.label.float().to(self.device))
                batch_size = len(datapoint_test.label)
                self.trackers.logger_test_loss.update(val_loss.item(), batch_size)
        self.trackers.log_test()
        return self

    def forward(self) -> 'BaseTrainer':
        self.model0.train()
        self.optimizer0.zero_grad()
        self.outputs = self._apply_forward(self.datapoint)
        return self

    def loss(self) -> 'BaseTrainer':
        target = self.datapoint.label.float().to(self.device)

        base_loss = self.criterion0(self.outputs, target)
        l1_regularization = torch.norm(self.model0.fc_scalar.weight, p=1)
        loss = base_loss + (self.config.model0.l1_lambda * l1_regularization)
        loss.backward()
        self.optimizer0.step()
        self.scheduler0.step()

        batch_size = len(self.datapoint.voxels)
        self.trackers.logger_loss.update(loss.item(), batch_size)
        self.trackers.logger_lr.update(self.scheduler0.get_last_lr()[0], batch_size)
        return self


xl, yl = 2048, 7168  # lower left corner of the test box
width, height = 2045, 2048

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

    config_model0 = ConfigurationModel(
        model=models.HybridModel(),
        criterion=BCEWithLogitsLoss(),
        learning_rate=0.03,
        total_steps=10_000_000,
        epochs=EPOCHS)

    config = Configuration(
        volume_dataset_cls=SampleXYZ,
        crop_box_cls=CropBoxRegular,
        suffix_cache='regular',
        label_fn=centre_pixel,
        transformers=ann.transforms.transform1,
        shuffle=False,
        group_pixels=True,
        balance_ink=True,
        batch_size=32,
        stride_xy=91,
        stride_z=6,
        num_workers=0,
        model0=config_model0)

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
                                config)

            for i, train in enumerate(trainer1):
                train.forward().loss()

                if i == 0:
                    continue
                if i % LOG_INTERVAL == 0:
                    train.trackers.log_train()
                if i % VALIDATE_INTERVAL == 0:
                    train.validate()

            train.save_model()
