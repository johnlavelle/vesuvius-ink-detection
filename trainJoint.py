import copy
import dataclasses
import pprint
from itertools import repeat, chain, cycle, islice
from typing import Any

import dask
import torch
from torch import autograd
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from src import tensorboard_access
from vesuvius import ann
from vesuvius.ann import models
from vesuvius.config import Configuration
from vesuvius.config import ConfigurationModel
from vesuvius.dataloader import get_dataset_regular_z
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxRegular
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer, centre_pixel
from vesuvius.utils import timer

# If READ_EXISTING_CONFIG is False, config is specified in Configuration (below)
# else config is read from CONFIG_PATH.
READ_CONFIG_FILE = False
CONFIG_PATH = 'configs/config.json'

pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)
dask.config.set(scheduler='synchronous')


def pretty_print_dataclass(obj: Any, indent: int = 4) -> str:
    if not dataclasses.is_dataclass(obj):
        return str(obj)

    indent_str = ' ' * indent
    result = obj.__class__.__name__ + "(\n"

    for field in dataclasses.fields(obj):
        field_value = getattr(obj, field.name)
        field_value_str = pretty_print_dataclass(field_value, indent + 4)
        result += f"{indent_str}{field.name}={field_value_str},\n"

    result += ")"
    print(result)


class JointTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output0 = None
        self.output1 = None
        self.config_test = None
        self.test_loader_len = None
        self.inputs = None
        self._loss = None
        self.labels = None
        self.outputs_collected = None
        self.labels_collected = None
        self.train_loader_iter = None
        self.last_model = self.config.model1

        self.get_train_test_loaders()

        pretty_print_dataclass(config)

        self.model0, self.optimizer0, self.scheduler0, self.criterion0 = self.setup_model(self.config.model0)
        self.model1, self.optimizer1, self.scheduler1, self.criterion1 = self.setup_model(self.config.model1)

    def _apply_forward(self, datapoint) -> torch.Tensor:
        voxels = datapoint.voxels
        dim0, dim1 = voxels.shape[:2]
        voxels = voxels.reshape(dim0 * dim1, *voxels.shape[2:])
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model0(voxels.to(self.device), scalar.to(self.device))

    def forward(self) -> 'JointTrainer':
        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()

        self.model0.train()
        self.outputs_collected = []
        self.labels_collected = []
        for _ in range(config.accumulation_steps):
            output = self._apply_forward(self.datapoint)
            self.outputs_collected.append(output.reshape(self.datapoint.label.shape[:2]))
            self.labels_collected.append(self.datapoint.label.float().mean(dim=1))
        self.output0 = torch.cat(self.outputs_collected, dim=0)
        self.labels = torch.cat(self.labels_collected, dim=0)
        self.labels = self.labels.to(self.device)
        return self

    def forward2(self) -> 'JointTrainer':
        self.model1.train()
        self.output1 = self.model1(self.output0)
        return self

    def loss(self) -> 'JointTrainer':
        base_loss = self.criterion1(self.output1, self.labels)
        l1_regularization = torch.norm(getattr(self.model0, 'module', self.model0).fc_scalar.weight, p=1)
        self._loss = base_loss + (self.config.model1.l1_lambda * l1_regularization)
        self.trackers.update_train(self._loss.item(), self.labels.shape[0])
        return self

    def backward(self) -> 'JointTrainer':
        self._loss.backward()
        return self

    def step(self) -> 'JointTrainer':
        self.optimizer0.step()
        self.scheduler0.step()
        self.optimizer1.step()
        self.scheduler1.step()
        return self

    def validate(self) -> 'JointTrainer':
        self.model0.eval()
        self.model1.eval()
        iterations = self.config.validation_steps
        iterations = round(self.config.batch_size * (iterations / self.config.batch_size))
        with torch.no_grad():
            for datapoint_test in islice(self.test_loader_iter, iterations):
                output0 = self._apply_forward(datapoint_test)
                output0 = output0.reshape(datapoint_test.label.shape[:2])
                output1 = self.model1(output0)
                labels = datapoint_test.label.float().mean(dim=1)
                labels = labels.to(self.device)
                val_loss = self.criterion1(output1, labels)
                self.trackers.update_test(val_loss.item(), len(labels))
            self.trackers.update_lr(self.scheduler1.get_last_lr()[0])
        self.model0.train()
        self.model1.train()
        return self

    def get_train_test_loaders(self) -> None:
        self.batch_size = round(self.config.batch_size / ((65 - self.config.stride_z) / self.config.stride_z + 1))

        dataloader_train = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.config.num_workers,
                                      drop_last=True)

        self.total_loops = min(len(dataloader_train), config.total_steps_max) * config.epochs
        config.loops_per_epoch = min(len(dataloader_train), config.total_steps_max)
        self.total_loops = config.loops_per_epoch * config.epochs
        self.train_loader_iter = chain.from_iterable(repeat(dataloader_train, config.epochs))
        self.train_loader_iter = islice(self.train_loader_iter, config.total_steps_max)

        self.config_test = copy.copy(self.config)
        self.config_test.transformers = None
        test_loader = DataLoader(self.train_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.config.num_workers,
                                 drop_last=True)

        self.test_loader_len = len(test_loader)
        self.test_loader_iter = cycle(iter(test_loader))

    def save_model(self):
        self._save_model(self.model0, suffix='0')
        self._save_model(self.model1, suffix='1')


if __name__ == '__main__':
    try:
        print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    except RuntimeError:
        print('Failed to get public tensorboard URL')

    EPOCHS = 600
    TOTAL_STEPS = 10_000_000
    VALIDATE_INTERVAL = 5000
    LOG_INTERVAL = 100

    config_model0 = ConfigurationModel(
        model=models.HybridModel(),
        learning_rate=0.03)

    config_model1 = ConfigurationModel(
        model=models.SimpleBinaryClassifier(0.5),
        learning_rate=config_model0.learning_rate,
        criterion=BCEWithLogitsLoss())

    config = Configuration(
        total_steps_max=TOTAL_STEPS,
        epochs=EPOCHS,
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
        accumulation_steps=5,
        model0=config_model0,
        model1=config_model1)

    train_dataset = get_dataset_regular_z(config, False, test_data=False)
    test_dataset = get_dataset_regular_z(config, False, test_data=True)

    with Track() as track, autograd.detect_anomaly(check_nan=False), timer("Training"):
        trainer = JointTrainer(train_dataset, test_dataset, track, config)

        for i, train in enumerate(trainer):
            train.forward().forward2().loss().backward().step()

            if i == 0:
                continue
            if i % LOG_INTERVAL == 0:
                train.trackers.log_train()
            if i % VALIDATE_INTERVAL == 0:
                train.validate()
                train.trackers.log_test()
        train.save_model()
