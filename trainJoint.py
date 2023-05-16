import copy
import time
from itertools import repeat, chain, islice
from typing import Tuple

import dask
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from src import tensorboard_access
from vesuvius import ann
from vesuvius.ann import models
from vesuvius.config import Configuration
from vesuvius.config import ConfigurationModel
from vesuvius.dataloader import get_dataset_regular_z
from vesuvius.datapoints import DatapointTuple
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxRegular
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer, centre_pixel
from vesuvius.utils import timer, pretty_print_dataclass


class JointTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output0 = None
        self.output1 = None
        self.config_test = None
        self.inputs = None
        self._loss = None
        self.labels = None
        self.outputs_collected = None
        self.labels_collected = None
        self.last_model = self.config.model1
        self.batch_size = self.config.batch_size
        self.input1_length = config.box_width_z // config.box_sub_width_z

        pretty_print_dataclass(config)

        self.model0, self.optimizer0, self.scheduler0, self.criterion0 = self.setup_model(self.config.model0)
        self.model1, self.optimizer1, self.scheduler1, self.criterion1 = self.setup_model(self.config.model1)

    def _apply_forward(self, datapoint) -> Tuple[torch.Tensor, torch.Tensor]:
        voxels = datapoint.voxels
        scalar = datapoint.label
        return scalar, self.model0(voxels.to(self.device), scalar.to(self.device))

    @staticmethod
    def _assert_all_values_are_one_or_zero(arr:  torch.Tensor):
        assert torch.all(torch.logical_or(arr == 0.0, arr == 1.0)), "Not all values are 1.0 or 0.0"

    def forward(self) -> 'JointTrainer':
        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()

        self.model0.train()
        self.outputs_collected, self.labels_collected = [], []
        for s in range(config.accumulation_steps):
            if s != 0:
                self.__next__()
            label, output = self._apply_forward(self.datapoint)
            output, label = self.reshape_output0(output),  self.reshape_output0(label)
            self.outputs_collected.append(output)
            self.labels_collected.append(label)

        self.output0 = torch.cat(self.outputs_collected, dim=0)
        self.labels = torch.cat(self.labels_collected, dim=0)

        self.labels = self.labels.mean(dim=1).to(self.device)
        self._assert_all_values_are_one_or_zero(self.labels)

        self.labels = self.labels.unsqueeze_(1)
        return self

    def reshape_output0(self, arr: torch.Tensor):
        return arr.reshape([-1, self.input1_length])

    def forward2(self) -> 'JointTrainer':
        self.model1.train()
        self.output1 = self.model1(self.output0.squeeze())
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

        with torch.no_grad():
            for datapoint_test in self.test_loader_iter:
                datapoint_test = self.reshape_datapoint(datapoint_test)
                labels, output0 = self._apply_forward(datapoint_test)

                output0 = self.reshape_output0(output0).squeeze()
                output1 = self.model1(output0)

                labels = self.reshape_output0(datapoint_test.label, 1).mean(dim=1)
                labels = labels.to(self.device)

                val_loss = self.criterion1(output1, labels)
                self.trackers.update_test(val_loss.item(), len(labels))
            self.trackers.update_lr(self.scheduler1.get_last_lr()[0])
        self.model0.train()
        self.model1.train()
        return self

    def save_model(self):
        self._save_model(self.model0, suffix='0')
        self._save_model(self.model1, suffix='1')

    def reshape_datapoint(self, datapoint):
        bsw = config.box_sub_width_z
        kwargs = {k: v.repeat_interleave(65 // bsw, dim=0) for k, v in datapoint._asdict().items() if
                  k != 'voxels'}
        kwargs['label'] = kwargs['label'].float()
        kwargs['voxels'] = self.datapoint.voxels.reshape(self.batch_size * 65 // bsw, 1, bsw, 91, 91)
        return DatapointTuple(**kwargs)

    def __next__(self):
        super().__next__()
        self.datapoint = self.reshape_datapoint(self.datapoint)


if __name__ == '__main__':
    dask.config.set(scheduler='synchronous')

    start_time = time.time()

    try:
        print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    except RuntimeError:
        print('Failed to get public tensorboard URL')

    EPOCHS = 1
    TOTAL_STEPS = 1_000_000
    SAVE_INTERVAL_MINUTES = 30
    VALIDATE_INTERVAL = 10
    LOG_INTERVAL = 10
    PRETRAINED_MODEL0 = False
    BOX_SUB_WIDTH_Z = 5

    save_interval_seconds = SAVE_INTERVAL_MINUTES * 60

    if PRETRAINED_MODEL0:
        config0 = Configuration.from_dict('configs/trainXYZ/')
        assert config0.box_width_z == BOX_SUB_WIDTH_Z
        config_model0 = config0.model0
        config_model0.model.requires_grad = False
    else:
        config_model0 = ConfigurationModel(
            model=models.HybridBinaryClassifier(dropout_rate=0.3, width_multiplier=1),
            learning_rate=0.03
        )

    config_model1 = ConfigurationModel(
        model=models.StackingClassifier(13),
        learning_rate=0.03,
        criterion=BCEWithLogitsLoss()
    )

    config = Configuration(
        samples_max=TOTAL_STEPS,
        epochs=EPOCHS,
        volume_dataset_cls=SampleXYZ,
        crop_box_cls=CropBoxRegular,
        suffix_cache='regular',
        label_fn=centre_pixel,
        transformers=ann.transforms.transform1,
        shuffle=False,
        group_pixels=False,
        balance_ink=True,
        batch_size=4,
        box_width_z=65,
        box_sub_width_z=BOX_SUB_WIDTH_Z,
        stride_xy=91,
        stride_z=65,
        num_workers=8,
        validation_steps=100,
        accumulation_steps=8,
        model0=config_model0,
        model1=config_model1
    )

    train_dataset = get_dataset_regular_z(config, False, test_data=False)
    dataloader_train = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  drop_last=True,
                                  pin_memory=True)

    config_val = copy.copy(config)
    config_val.transformers = None
    test_dataset = get_dataset_regular_z(config_val, False, test_data=True)

    with Track() as track, timer("Training"):
        trainer = JointTrainer(train_dataset, test_dataset, track, config)

        for i, train in enumerate(trainer):
            train.forward().forward2().loss().backward().step()

            if i == 0:
                continue
            if time.time() - start_time >= save_interval_seconds:
                train.save_model()  # save the model
                start_time = time.time()  # reset the start_time for the next save interval
            if i % LOG_INTERVAL == 0:
                train.trackers.log_train()
            if i % VALIDATE_INTERVAL == 0:
                train.validate()
                train.trackers.log_test()
        train.save_model()
