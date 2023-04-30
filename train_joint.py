import copy
import pprint
from itertools import repeat, chain, cycle, islice

import dask
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

from src import tensorboard_access
from vesuvius import ann
from vesuvius.config import Configuration
from vesuvius.config import ConfigurationModel
from vesuvius.dataloader import get_train_loader_regular_z
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxRegular
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer, centre_pixel
from vesuvius.utils import timer
from vesuvius.ann import models

# If READ_EXISTING_CONFIG is False, config is specified in Configuration (below)
# else config is read from CONFIG_PATH.
READ_CONFIG_FILE = False
CONFIG_PATH = 'configs/config.json'

pp = pprint.PrettyPrinter(indent=4)
dask.config.set(scheduler='synchronous')


class JointTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output0 = None
        self.output1 = None
        self.config_test = None
        self.test_loader_len = None
        self.inputs = None
        self._loss = None
        self._loss_joint = None
        self.labels = None
        self.outputs_collected = None
        self.labels_collected = None
        self.model1, self.optimizer1, self.scheduler1, self.criterion1 = self.setup_model(self.config.model1)

    def setup_model(self, model_object):
        args = super().setup_model(model_object)
        if torch.cuda.device_count() >= 2:
            self.model0 = nn.DataParallel(self.model0)
            self.model1 = nn.DataParallel(self.model1)
            print('Using DataParallel for training.')
        return args

    def _apply_forward(self, datapoint) -> torch.Tensor:
        voxels = datapoint.voxels
        dim0, dim1 = voxels.shape[:2]
        voxels = voxels.reshape(dim0 * dim1, *voxels.shape[2:])
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model0(voxels.to(self.device), scalar.to(self.device))

    def forward(self) -> 'JointTrainer':
        self.model0.train()
        self.outputs_collected = []
        self.labels_collected = []
        for _ in range(5):
            output = self._apply_forward(self.datapoint)
            self.outputs_collected.append(output.reshape(self.datapoint.label.shape[:2]))
            self.labels_collected.append(self.datapoint.label.float().mean(dim=1))
        self.outputs = torch.cat(self.outputs_collected, dim=0)
        self.labels = torch.cat(self.labels_collected, dim=0)
        self.labels = self.labels.to(self.device)
        return self

    def forward2(self) -> 'JointTrainer':
        self.model1.train()
        self.output1 = self.model1(self.outputs)
        return self

    def loss(self) -> 'JointTrainer':
        base_loss = self.criterion1(self.output1, self.labels)
        l1_regularization = torch.norm(self.model0.fc_scalar.weight, p=1)
        self._loss_joint = base_loss + (self.config.model1.l1_lambda * l1_regularization)
        self.trackers.update_train(self._loss_joint.item(), self.labels.shape[0])
        return self

    def backward(self) -> 'JointTrainer':
        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()
        self._loss_joint.backward()
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

        dataloader_train = DataLoader(self.train_loader(self.config, False, test_data=False),
                                      batch_size=self.batch_size,
                                      num_workers=self.config.num_workers,
                                      drop_last=True)
        self.total = self.epochs * len(dataloader_train) * self.batch_size
        self.loops = self.epochs * len(dataloader_train)
        self.train_loader_iter = chain.from_iterable(repeat(dataloader_train, self.epochs))

        self.config_test = copy.copy(self.config)
        self.config_test.transformers = None
        test_loader = DataLoader(self.train_loader(self.config_test, False, test_data=True),
                                 batch_size=self.batch_size,
                                 num_workers=self.config.num_workers,
                                 drop_last=True)

        self.test_loader_len = len(test_loader)
        self.test_loader_iter = cycle(iter(test_loader))

    def __str__(self) -> str:
        loop = self.trackers.incrementer.loop
        lr = self.config.model1.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"Current Loop: {loop}, Learning Rate: {lr}, Epochs: {epochs}, Batch Size: {batch_size}"

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        loop = self.trackers.incrementer.loop
        lr = self.config.model1.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"{classname}(current_loop={loop}, learning_rate={lr}, epochs={epochs}, batch_size={batch_size})"


# get_train_loader_regular_z = partial(get_train_loader_regular_z, force_cache_reset=False)

if __name__ == '__main__':
    try:
        print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    except RuntimeError:
        print('Failed to get public tensorboard URL')

    EPOCHS = 30
    TOTAL_STEPS = 10_000_000
    VALIDATE_INTERVAL = 500
    LOG_INTERVAL = 50

    config_model0 = ConfigurationModel(
        model=models.HybridModel(),
        learning_rate=0.03,
        total_steps=TOTAL_STEPS,
        epochs=EPOCHS)

    config_model1 = ConfigurationModel(
        model=models.SimpleBinaryClassifier(),
        learning_rate=config_model0.learning_rate,
        total_steps=config_model0.total_steps,
        epochs=config_model0.epochs,
        criterion=BCEWithLogitsLoss())

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
        model0=config_model0,
        model1=config_model1)

    with Track() as track, timer("Training"):

        trainer = JointTrainer(get_train_loader_regular_z,
                               get_train_loader_regular_z,
                               track,
                               config=config)

        for i, train in enumerate(trainer):
            train.forward().forward2().loss().backward().step()

            if i == 0:
                continue
            if i % LOG_INTERVAL == 0:
                train.trackers.log_train()
            if i % VALIDATE_INTERVAL == 0:
                train.validate()
                train.trackers.log_test()
        train.save_model_output()
