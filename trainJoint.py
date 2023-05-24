import copy
import time
from typing import Tuple, Iterable

import numpy as np
import dask
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from src import tensorboard_access
from vesuvius import ann
from vesuvius.ann import models
from vesuvius.config import Configuration, ConfigurationModel
from vesuvius.dataloader import get_dataset_regular_z
from vesuvius.datapoints import DatapointTuple
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxRegular
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer, centre_pixel
from vesuvius.utils import timer, pretty_print_dataclass
from vesuvius.metric import f0_5_score


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
        self.num_z_vols = 65 // config.box_sub_width_z

        pretty_print_dataclass(config)

        self.model0, self.optimizer0, self.scheduler0, self.criterion0 = self.setup_model(self.config.model0)
        self.model1, self.optimizer1, self.scheduler1, self.criterion1 = self.setup_model(self.config.model1)

    def _apply_forward(self, datapoint) -> Tuple[torch.Tensor, torch.Tensor]:
        voxels = datapoint.voxels
        scalar = datapoint.label
        try:
            return scalar, self.model0(voxels.to(self.device), scalar.to(self.device))
        except RuntimeError:
            pass

    @staticmethod
    def _assert_all_values_are_one_or_zero(arr: torch.Tensor):
        assert torch.all(torch.logical_or(arr == 0.0, arr == 1.0)), "Not all values are 1.0 or 0.0"

    def zero_grad(self):
        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()
        return self

    def forward(self) -> 'JointTrainer':
        self.model0.train()
        self.outputs_collected, self.labels_collected = [], []
        for s in range(config.accumulation_steps):
            if s != 0:
                self.__next__()
            label, output = self._apply_forward(self.datapoint)
            output, label = self.reshape_output0(output), self.reshape_output0(label)
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
        l1_regularization0 = torch.norm(getattr(self.model0, 'module', self.model0).fc_scalar.weight, p=1)
        l1_regularization1 = torch.norm(getattr(self.model1, 'module', self.model1).fc2.weight, p=1)
        l10 = self.config.model0.l1_lambda * l1_regularization0
        l11 = self.config.model1.l1_lambda * l1_regularization1
        self._loss = base_loss + l10 + l11
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

    def prediction(self, data_iter: Iterable) -> Tuple[np.ndarray, np.ndarray]:
        self.model0.eval()
        self.model1.eval()

        with torch.no_grad():
            labels, outputs = [], []
            for datapoint in data_iter:
                datapoint = self.reshape_datapoint(datapoint)
                label, output0 = self._apply_forward(datapoint)
                label, output0 = self.reshape_output0(label), self.reshape_output0(output0)
                output1 = self.model1(output0.squeeze())
                label = label.mean(dim=1).unsqueeze(1).to(self.device)
                _loss = self.criterion1(output1, label)
                self.trackers.update_test(_loss.item(), len(label))
                outputs.append(output1.flatten().detach().cpu().numpy())
                labels.append(label.flatten().detach().cpu().numpy())
            outputs = np.concatenate(outputs)
            labels = np.concatenate(labels)

        self.model0.train()
        self.model1.train()
        return outputs, labels

    def validate(self) -> 'JointTrainer':
        outputs, labels = self.prediction(self.test_loader_iter)
        outputs_int = (outputs >= 0.5).astype(float)
        score = f0_5_score(outputs_int, labels)
        self.trackers.log_score(score)
        self.trackers.update_lr(self.scheduler1.get_last_lr()[0])
        return self

    def save_model(self):
        self._save_model(self.model0, suffix='0')
        self._save_model(self.model1, suffix='1')

    def reshape_datapoint(self, datapoint):
        kwargs = {k: v.repeat_interleave(self.num_z_vols, dim=0) for k, v in datapoint._asdict().items() if
                  k != 'voxels'}
        kwargs['label'] = kwargs['label'].float()
        kwargs['voxels'] = self.datapoint.voxels.reshape(self.batch_size * self.num_z_vols,
                                                         1, config.box_sub_width_z, 91, 91)
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

    EPOCHS = 100
    TOTAL_STEPS = 1_000_000
    SAVE_INTERVAL_MINUTES = 30
    VALIDATE_INTERVAL = 1000
    LOG_INTERVAL = 100
    PRETRAINED_MODEL0 = False
    BOX_SUB_WIDTH_Z = 5
    LEARNING_RATE = 0.01

    save_interval_seconds = SAVE_INTERVAL_MINUTES * 60

    if PRETRAINED_MODEL0:
        config0 = Configuration.from_dict('configs/XYZ/')
        assert config0.box_width_z == BOX_SUB_WIDTH_Z
        config_model0 = config0.model0
        config_model0.model.requires_grad = False
    else:
        config_model0 = ConfigurationModel(
            model=models.HybridBinaryClassifierShallow(dropout_rate=0.2, width=1),
            optimizer_scheduler_cls=ann.optimisers.AdamOneCycleLR,
            learning_rate=LEARNING_RATE,
            l1_lambda=0.01
        )

    config_model1 = ConfigurationModel(
        model=models.StackingClassifierShallow(13, 1),
        optimizer_scheduler_cls=ann.optimisers.AdamOneCycleLR,
        learning_rate=LEARNING_RATE,
        l1_lambda=0.01,
        criterion=BCEWithLogitsLoss()
    )

    config = Configuration(
        info='nn.Conv3d(1, self.width, 5, 1, 2); nn.AvgPool3d(5, 5)',
        samples_max=TOTAL_STEPS,
        epochs=EPOCHS,
        volume_dataset_cls=SampleXYZ,
        crop_box_cls=CropBoxRegular,
        suffix_cache='regular',
        label_fn=centre_pixel,
        transformers=ann.transforms.transform_train,
        shuffle=False,
        balance_ink=True,
        batch_size=32,
        box_width_z=65,
        box_sub_width_z=BOX_SUB_WIDTH_Z,
        stride_xy=91,
        stride_z=65,
        num_workers=5,
        validation_steps=100,
        accumulation_steps=1,
        model0=config_model0,
        model1=config_model1
    )

    train_dataset = get_dataset_regular_z(config, False, test_data=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1,
                                  num_workers=config.num_workers,
                                  drop_last=True,
                                  pin_memory=False)

    config_val = copy.copy(config)
    config_val.transformers = ann.transforms.transform_val
    test_dataset = get_dataset_regular_z(config_val, False, test_data=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=config.num_workers,
                                 drop_last=True,
                                 pin_memory=False)

    with Track() as track, timer("Training"):
        trainer = JointTrainer(train_dataset, test_dataset, track, config)

        for i, train in enumerate(trainer):
            train.zero_grad().forward().forward2().loss().backward().step()

            if i == 0:
                continue
            if time.time() - start_time >= save_interval_seconds:
                train.save_model()
                start_time = time.time()
            if i % LOG_INTERVAL == 0:
                train.trackers.log_train()
            if i % VALIDATE_INTERVAL == 0:
                train.validate()
                train.trackers.log_test()
        train.save_model()
