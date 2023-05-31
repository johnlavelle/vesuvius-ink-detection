import pprint
import copy

import dask
import torch
from torch.utils.data import DataLoader

from vesuvius import ann
from vesuvius.ann import models
from vesuvius.ann.criterions import FocalLoss
from vesuvius.config import Configuration
from vesuvius.config import ConfigurationModel
from vesuvius.dataloader import get_test_loader, get_train_dataset
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxSobol
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer
from vesuvius.labels import centre_pixel
from vesuvius.utils import timer

pp = pprint.PrettyPrinter(indent=4)
dask.config.set(scheduler='synchronous')


class TrainerXYZ(BaseTrainer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        self.outputs = None
        self.loss_value = None

        self.last_model = self.config.model0
        self.model0, self.optimizer0, self.scheduler0, self.criterion0 = self.setup_model(self.last_model)

    def _apply_forward(self, datapoint) -> torch.Tensor:
        # Rescale z to be between 0 and 1
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model0(datapoint.voxels.to(self.device), scalar.to(self.device))

    def validate(self) -> 'TrainerXYZ':
        self.model0.eval()
        with torch.no_grad():
            for datapoint_test in self.val_loader_iter:
                outputs = self._apply_forward(datapoint_test)
                val_loss = self.criterion0(outputs, datapoint_test.label.float().to(self.device))
                batch_size = len(datapoint_test.label)
                self.trackers.logger_test_loss.update(val_loss.item(), batch_size)
        self.trackers.log_test()
        return self

    def forward0(self) -> 'TrainerXYZ':
        self.model0.train()
        self.optimizer0.zero_grad()
        self.outputs = self._apply_forward(self.datapoint)
        return self

    def loss(self) -> 'TrainerXYZ':
        target = self.datapoint.label.float().to(self.device)

        base_loss = self.criterion0(self.outputs, target)
        l1_regularization = torch.norm(self.model0.fc_scalar.weight, p=1)
        self.loss_value = base_loss + (self.config.model0.l1_lambda * l1_regularization)

        batch_size = len(self.datapoint.voxels)
        self.trackers.logger_loss.update(self.loss_value.item(), batch_size)
        self.trackers.logger_lr.update(self.scheduler0.get_last_lr()[0], batch_size)
        return self

    def backward(self) -> 'TrainerXYZ':
        self.loss_value.backward()
        return self

    def step(self) -> 'TrainerXYZ':
        self.optimizer0.step()
        self.scheduler0.step()
        return self

    def save_model(self):
        self._save_model(self.model0, suffix='0')


xl, yl = 2048, 7168  # lower left corner of the test box
width, height = 2045, 2048

if __name__ == '__main__':

    pp = pprint.PrettyPrinter(indent=4)
    dask.config.set(scheduler='synchronous')
    # print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')

    EPOCHS = 100
    SAVE_INTERVAL = 1_000_000
    VALIDATE_INTERVAL = 1000
    LOG_INTERVAL = 100

    config_model0 = ConfigurationModel(
        model=models.HybridBinaryClassifier(dropout_rate=0.1),
        criterion=FocalLoss(),
        learning_rate=0.03)

    config = Configuration(
        epochs=EPOCHS,
        samples_max=160_000,
        volume_dataset_cls=SampleXYZ,
        crop_box_cls=CropBoxSobol,
        suffix_cache='sobol',
        label_fn=centre_pixel,
        transformers=ann.transforms.transform_train,
        shuffle=True,
        balance_ink=True,
        box_width_z=5,
        batch_size=32,
        num_workers=10,
        model0=config_model0)

    print(config)
    # train_loaders = partial(
    #     get_train_datasets,
    #     cached_data=True,
    #     force_cache_reset=FORCE_CACHE_RESET,
    #     reset_cache_epoch_interval=RESET_CACHE_EPOCH_INTERVAL)

    for alpha, gamma in [
        (0.25, 2),
        # (0.5, 2),
        # (0.75, 2),
        # (1, 0)
    ]:

        train_dataset = get_train_dataset(config, cached=True, reset_cache=False)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      drop_last=True,
                                      pin_memory=True)

        config_val = copy.copy(config)
        config_val.transformers = None
        test_dataloader = get_test_loader(config)

        criterion = FocalLoss(alpha=alpha, gamma=gamma)

        with Track() as track, timer("Training"):

            trainer = TrainerXYZ(train_dataloader, test_dataloader, track, config)

            for i, train in enumerate(trainer):
                pass
                train.forward0().loss().backward().step()

                if i == 0:
                    continue
                if i % LOG_INTERVAL == 0:
                    train.trackers.log_train()
                if i % VALIDATE_INTERVAL == 0:
                    train.validate()
            train.save_model()
