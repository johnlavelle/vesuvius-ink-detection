import copy
import time
import random

from typing import Tuple, Iterable, List

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
import pandas as pd
import matplotlib.pyplot as plt

from src import tensorboard_access
from vesuvius import ann
from vesuvius.ann import models
from vesuvius.config import Configuration, ConfigurationModel
from vesuvius.dataloader import get_dataset_regular_z
from vesuvius.datapoints import DatapointTuple
from vesuvius.labels import centre_pixel
from vesuvius.metric import f0_5_score
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxRegular
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer
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
        self.num_z_vols = 65 // config.box_sub_width_z

        pretty_print_dataclass(config)

        self.model0, self.optimizer_scheduler0, self.criterion0 = self.setup_model(self.config.model0)
        self.model1, self.optimizer_scheduler1, self.criterion1 = self.setup_model(self.config.model1)

    def _apply_forward(self, datapoint) -> Tuple[torch.Tensor, torch.Tensor]:
        voxels = datapoint.voxels
        scalar = (datapoint.z_start / (65 - self.config.box_sub_width_z)).view(-1, 1).float()
        try:
            return datapoint.label, self.model0(voxels.to(self.device), scalar.to(self.device))
        except RuntimeError as err:
            raise err

    @staticmethod
    def _assert_all_values_are_one_or_zero(arr: torch.Tensor):
        assert torch.all(torch.logical_or(arr == 0.0, arr == 1.0)), "Not all values are 1.0 or 0.0"

    def forward0(self) -> 'JointTrainer':
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

    def forward1(self) -> 'JointTrainer':
        self.model1.train()
        self.output1 = self.model1(self.output0.squeeze())
        return self

    def loss(self) -> 'JointTrainer':
        base_loss = self.criterion1(self.output1, self.labels)
        if self.config.model0.l1_lambda:
            l1_regularization0 = torch.norm(getattr(self.model0, 'module', self.model0).fc_scalar.weight, p=1)
            l10 = self.config.model0.l1_lambda * l1_regularization0
        else:
            l10 = 0
        if self.config.model1.l1_lambda:
            l1_regularization1 = torch.norm(getattr(self.model1, 'module', self.model1).fc2.weight, p=1)
            l11 = self.config.model1.l1_lambda * l1_regularization1
        else:
            l11 = 0
        self._loss = base_loss + l10 + l11

        self.trackers.update_train(self._loss.item(), self.labels.shape[0])
        return self

    def backward(self) -> 'JointTrainer':
        self._loss.backward()
        return self

    def step(self) -> 'JointTrainer':
        self.optimizer_scheduler0.step()
        self.optimizer_scheduler1.step()
        return self

    def prediction(self, data_iter: Iterable) -> Tuple[np.ndarray, np.ndarray, List, List]:
        self.model0.eval()
        self.model1.eval()

        with torch.no_grad():
            _labels, _outputs, _xs, _ys, _fragments = [], [], [], [], []
            for datapoint in data_iter:
                x_start = datapoint.x_start
                x_stop = datapoint.x_stop
                y_start = datapoint.y_start
                y_stop = datapoint.y_stop
                fragment = datapoint.fragment

                datapoint = self.reshape_datapoint(datapoint)
                label, output0 = self._apply_forward(datapoint)
                label, output0 = self.reshape_output0(label), self.reshape_output0(output0)
                output1 = self.model1(output0.squeeze())

                label = label.mean(dim=1).unsqueeze(1).to(self.device)
                _loss = self.criterion1(output1, label)
                self.trackers.update_test(_loss.item(), len(label))
                _outputs.append(output1.flatten().detach().cpu().numpy())
                _labels.append(label.flatten().detach().cpu().numpy())

                x_centre = (x_start + x_stop) // 2
                y_centre = (y_start + y_stop) // 2
                _xs.extend(x_centre.detach().numpy())
                _ys.extend(y_centre.detach().numpy())
                _fragments.extend(fragment.detach().numpy())

        _outputs = np.concatenate(_outputs)
        _labels = np.concatenate(_labels)

        self.model0.train()
        self.model1.train()

        return _outputs, _labels, _fragments, zip(_xs, _ys)

    def validate(self) -> 'JointTrainer':
        _outputs, _labels, _fragments, _coords = self.prediction(self.val_loader_iter)

        predicted_labels_int = (_outputs >= 0.5).astype(float)
        score = f0_5_score(predicted_labels_int, _labels)
        self.trackers.log_score(score)
        self.trackers.update_lr(self.optimizer_scheduler0.optimizer.param_groups[0]['lr'])
        return self

    def inference(self):
        _outputs, _labels, _fragments, _coords = self.prediction(self.test_loader_iter)
        plt.hist(_outputs, color='r')
        plt.show()
        predicted_labels_int = (_outputs >= 0.5).astype(float)
        return _outputs, predicted_labels_int, _coords, _fragments

    def save_model(self):
        self._save_model(self.model0, suffix='0')
        self._save_model(self.model1, suffix='1')

    def reshape_datapoint(self, datapoint):

        datapoint_dict = datapoint._asdict()

        kwargs = {k: v.repeat_interleave(self.num_z_vols, dim=0) for k, v in datapoint_dict.items() if
                  k != 'voxels'}

        kwargs['z_start'] = torch.tensor(np.arange(0, self.config.box_width_z, self.config.box_sub_width_z)).repeat(self.config.batch_size)
        kwargs['z_stop'] = torch.tensor(np.array(list(min(e + 13 - 1, 64) for e in datapoint_dict['z_start']))).repeat(self.config.batch_size)

        kwargs['label'] = kwargs['label'].float()
        kwargs['voxels'] = datapoint.voxels.reshape(self.batch_size * self.num_z_vols,
                                                    1, config.box_sub_width_z, config.box_width_xy, config.box_width_xy)
        # rnd = np.random.uniform(-5, 5)
        # kwargs['z_start'] = kwargs['z_start'] + rnd)
        # kwargs['z_stop'] = 0*(kwargs['z_stop'] + rnd)
        return DatapointTuple(**kwargs)

    def __next__(self):
        super().__next__()
        self.datapoint = self.reshape_datapoint(self.datapoint)


if __name__ == '__main__':
    # dask.config.set(scheduler='synchronous')

    try:
        print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    except RuntimeError:
        print('Failed to get public tensorboard URL')

    TRAIN = True
    INFERENCE = True
    STORED_CONFIG = True

    EPOCHS = 3
    TOTAL_STEPS = 10_000_000
    SAVE_INTERVAL_MINUTES = 30
    VALIDATE_INTERVAL = 1_000
    LOG_INTERVAL = 250
    PRETRAINED_MODEL0 = True
    BOX_SUB_WIDTH_Z = 13
    LEARNING_RATE = 0.3

    save_interval_seconds = SAVE_INTERVAL_MINUTES * 60

    if STORED_CONFIG:
        _config = Configuration.from_dict('output/runs/2023-06-08_16-45-30/')
        # assert _config.box_sub_width_z == BOX_SUB_WIDTH_Z
        config_model0 = _config.model0
        config_model0.model.requires_grad = False
        # config_model1 = _config.model1
        # config_model1.model.requires_grad = False
    else:
        config_model0 = ConfigurationModel(
            model=models.HybridBinaryClassifier(dropout_rate=0.1),
            optimizer_scheduler_cls=ann.optimisers.AdamOneCycleLR,
            learning_rate=LEARNING_RATE,
            l1_lambda=0.0
        )

    config_model1 = ConfigurationModel(
        model=models.StackingClassifierShallow(5, 1),
        optimizer_scheduler_cls=ann.optimisers.AdamOneCycleLR,
        learning_rate=LEARNING_RATE,
        l1_lambda=0.00001,
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
        batch_size=16,
        box_width_z=65,
        box_width_xy=65,
        box_sub_width_z=BOX_SUB_WIDTH_Z,
        stride_xy=3 * 65 // 4,
        stride_z=65,
        num_workers=1,
        validation_steps=100,
        accumulation_steps=1,
        model0=config_model0,
        model1=config_model1,
        fragments=(1, 2, 3)
    )

    random.seed(config.seed)  # Set the seed here

    config_val = copy.copy(config)
    config_val.transformers = ann.transforms.transform_val

    config_inference = copy.copy(config)
    config_inference.prefix = "/data/kaggle/input/vesuvius-challenge-ink-detection/test/"
    config_inference.fragments = ('a', 'b')
    # config_inference.prefix = "/data/kaggle/input/vesuvius-challenge-ink-detection/train/"
    # config_inference.fragments = (1, 2)
    config_inference.balance_ink = False
    config_inference.validation_steps = 1_000_000

    train_dataset = get_dataset_regular_z(config, False, validation=False)
    val_dataset = get_dataset_regular_z(config_val, False, validation=True)
    test_dataset = get_dataset_regular_z(config_inference, False, validation=False)

    start_time = time.time()
    with Track() as track, timer("Training"):
        trainer = JointTrainer(config,
                               track,
                               train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               test_dataset=test_dataset)

        if TRAIN:
            for i, train in enumerate(trainer):
                train.forward0().forward1().loss().backward().step()

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

    if INFERENCE:
        outputs, labels, coords, fragments = trainer.inference()
        # Convert list of tuples into two lists
        x_coord, y_coord = zip(*coords)

        # Create DataFrame
        df = pd.DataFrame({
            'X': x_coord,
            'Y': y_coord,
            'outputs': outputs,
            'fragments': fragments
        })

        # Aggregate duplicates by taking the mean
        # df_agg = df.groupby(['X', 'Y']).mean().reset_index()
        for frag in (1, 2):
            df_agg = df[df['fragments'] == frag]
            # Pivot DataFrame to create grid
            grid = df_agg.pivot(index='X', columns='Y', values='outputs')

            grid.fillna(-1, inplace=True)

            # Assuming 'grid' is your 2D array or DataFrame
            plt.figure(figsize=(10, 10))  # Adjust size as needed
            plt.imshow(grid.transpose(), cmap='hot',
                       interpolation='none')  # Change colormap as needed. Other options: 'cool', 'coolwarm', 'Greys', etc.
            plt.colorbar(label='labels')  # Shows a color scale
            plt.show()

            df_agg['outputs'].plot.hist(bins=50)
            plt.show()
