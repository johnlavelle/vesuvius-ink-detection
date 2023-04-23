import pprint
from functools import partial
from itertools import islice
from typing import Tuple

import dask
import torch
from torch.utils.data import DataLoader

import tensorboard_access
from trainer1 import SampleXYZ
from vesuvius.ann.criterions import FocalLoss
from vesuvius.ann.models import HybridModel
from vesuvius.ann.optimisers import SGDOneCycleLR
from vesuvius.dataloader import get_train_loader_regular_z
from vesuvius.dataloader import get_train_loaders
from vesuvius.sampler import CropBoxRegular
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer
from vesuvius.utils import timer

# If READ_EXISTING_CONFIG is False, config is specified in Configuration (below)
# else config is read from CONFIG_PATH.
READ_CONFIG_FILE = False
CONFIG_PATH = 'configs/config.json'

pp = pprint.PrettyPrinter(indent=4)
dask.config.set(scheduler='synchronous')


class JointTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.outputs_collected = []
        self.test_loader_iter = lambda n: islice(iter(self.test_loader(self.config, False)), n)
        batch_size = round(self.config.batch_size / ((65 - self.config.stride_z) / self.config.stride_z + 1))
        self.train_loader_iter = iter(DataLoader(self.train_loader(self.config, False),
                                                 batch_size=batch_size,
                                                 num_workers=self.config.num_workers))

        # def __next__(self):
        #     if self.trackers.incrementer.loop >= self.loops:
        #         raise StopIteration
        #     datapoints = []
        #     while True:
        #         dp = next(self.train_loader_iter)
        #         length = self.batch_size // len(dp.label)
        #         if len(datapoints) >= length:
        #             break
        #         else:
        #             datapoints.append(dp)
        #     self.datapoint = default_collate(datapoints)
        #     self.trackers.increment(len(self.datapoint.label))
        #     return self

    def _apply_forward(self, datapoint) -> torch.Tensor:
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model(datapoint.voxels.to(self.device), scalar.to(self.device))

    def _forward(self) -> 'BaseTrainer':
        self.model.train()
        self.outputs_collected.append(self._apply_forward(self.datapoint))
        return self

    def forward(self) -> 'BaseTrainer':
        train.optimizer.zero_grad()
        for _ in range(1):
            self._forward()
        train.outputs = torch.cat(train.outputs_collected, dim=0)
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

    def dummy_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_input = torch.randn(self.config.batch_size,
                                   1,
                                   self.config.box_width_z,
                                   self.config.box_width_xy,
                                   self.config.box_width_xy).to(self.device)
        scalar_input = torch.tensor([2.0] * self.config.batch_size).to(self.device).view(-1, 1).to(self.device)
        return tensor_input, scalar_input


# get_train_loader_regular_z = partial(get_train_loader_regular_z, force_cache_reset=False)

if __name__ == '__main__':
    print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')

    CACHED_DATA = True
    FORCE_CACHE_RESET = False  # Deletes cache. Only used if CACHED_DATA is True.
    EPOCHS = 2
    RESET_CACHE_EPOCH_INTERVAL = EPOCHS
    VALIDATE_INTERVAL = 1000
    LOG_INTERVAL = 100

    train_dataset = partial(
        get_train_loaders,
        cached_data=CACHED_DATA,
        force_cache_reset=FORCE_CACHE_RESET,
        reset_cache_epoch_interval=RESET_CACHE_EPOCH_INTERVAL)

    for alpha, gamma in [(1, 0), (0.25, 2), (0.5, 2), (0.75, 2)]:
        # with alpha=1, gamma=0, the loss is the same as the standard torch.nn.BCEWithLogitsLoss
        criterion = FocalLoss(alpha=alpha, gamma=gamma)

        with Track() as track, Track() as trackers2, timer("Training"):

            config_kwargs = dict(training_steps=32 * (4000 // 32) - 1,
                                 model=HybridModel,
                                 volume_dataset_cls=SampleXYZ,
                                 crop_box_cls=CropBoxRegular,
                                 suffix_cache='regular',
                                 shuffle=False,
                                 group_pixels=True,
                                 balance_ink=True,
                                 stride_xy=61,
                                 stride_z=6)
            trainer = JointTrainer(get_train_loader_regular_z,
                                   get_train_loader_regular_z,
                                   track,
                                   SGDOneCycleLR,
                                   criterion,
                                   config_kwargs=config_kwargs,
                                   learning_rate=0.03,
                                   l1_lambda=0,
                                   epochs=EPOCHS)

            for i, train in enumerate(trainer):
                pass
                # print(train.datapoint.voxels.shape)
                # train.forward()
            #     train.outputs
            #     print(i)
            #
            #     if i == 0:
            #         continue
            #     if i % LOG_INTERVAL == 0:
            #         train.trackers1.log_train()
            #     if i % VALIDATE_INTERVAL == 0:
            #         train.validate()
            #
            # train.save_model_output()
