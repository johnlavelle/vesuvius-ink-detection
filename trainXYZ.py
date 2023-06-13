import copy
import pprint

import dask
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
import numpy as np
import gc

from src import tensorboard_access
from vesuvius import ann
from vesuvius.ann import models
from vesuvius.config import Configuration
from vesuvius.config import ConfigurationModel
from vesuvius.dataloader import get_train_dataset
from vesuvius.labels import centre_pixel
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxSobol
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer
from vesuvius.utils import timer
from vesuvius.ann.criterions import FocalLoss
from torch.nn import BCEWithLogitsLoss

pp = pprint.PrettyPrinter(indent=4)
dask.config.set(scheduler='synchronous')


def gaussian(x, mu, sigma):
    y = torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return len(y) * y / torch.sum(y)


def interpolate(start, end, ratio):
    """
    This function interpolates between 'start' and 'end'
    based on the given ratio.

    Args:
    start (float): The starting value.
    end (float): The ending value.
    ratio (float): The ratio to interpolate between start and end.
                   Should be between 0 and 1.

    Returns:
    float: The interpolated value.
    """
    if ratio < 0 or ratio > 1:
        raise ValueError("The ratio must be between 0 and 1.")

    return start + (end - start) * ratio


class LossError(ValueError):
    pass


class TrainerXYZ(BaseTrainer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.outputs = None
        self.loss_value = None
        # self.sigma = self.train_dataset.sigma

        self.criterion_val = BCEWithLogitsLoss()

        self.last_model = self.config.model0
        self.model0, self.optimizer_scheduler0, self.criterion0 = self.setup_model(self.last_model)

    def _apply_forward(self, datapoint) -> torch.Tensor:
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model0(datapoint.voxels.to(self.device), scalar.to(self.device))

    def get_loss_weight(self, datapoint):
        weight = gaussian(datapoint.z_start, mu=10, sigma=self.sigma)
        return weight.view(-1, 1).to(self.device)

    def validate(self) -> 'TrainerXYZ':
        self.model0.eval()
        with torch.no_grad():
            for datapoint_test in self.val_loader_iter:
                outputs = self._apply_forward(datapoint_test)
                # weight = self.get_loss_weight(datapoint_test)
                val_loss = self.criterion_val(outputs, datapoint_test.label.float().to(self.device))
                batch_size = len(datapoint_test.label)
                self.trackers.logger_test_loss.update(val_loss.item(), batch_size)

        if self.trackers.logger_test_loss.average > 1:
            raise LossError

        self.trackers.log_test()
        self.model0.train()
        return self

    def forward0(self) -> 'TrainerXYZ':
        self.model0.train()
        self.outputs = self._apply_forward(self.datapoint)
        return self

    def loss(self) -> 'TrainerXYZ':
        target = self.datapoint.label.float().to(self.device)
        # weight = self.get_loss_weight(self.datapoint)
        base_loss = self.criterion0(self.outputs, target)
        if self.config.model0.l1_lambda:
            parameters0 = [p.flatten() for name, p in getattr(self.model0, 'module', self.model0).named_parameters() if
                           'weight' in name]
            concatenated_parameters0 = torch.cat(parameters0)
            l1_regularization0 = torch.norm(concatenated_parameters0, p=1)
        else:
            l1_regularization0 = 0
        self.loss_value = base_loss + (self.config.model0.l1_lambda * l1_regularization0)

        batch_size = len(self.datapoint.voxels)
        self.trackers.logger_loss.update(self.loss_value.item(), batch_size)
        self.trackers.logger_lr.update(self.optimizer_scheduler0.optimizer.param_groups[0]['lr'], batch_size)
        if self.loss_value.item() > 1:
            raise LossError

        if torch.isnan(self.loss_value):
            raise LossError

        return self

    def backward(self) -> 'TrainerXYZ':
        self.loss_value.backward()
        return self

    def step(self) -> 'TrainerXYZ':
        self.optimizer_scheduler0.step()
        return self

    def save_model(self):
        self._save_model(self.model0, suffix='0')


xl, yl = 2048, 7168  # lower left corner of the test box
width, height = 2045, 2048

if __name__ == '__main__':

    pp = pprint.PrettyPrinter(indent=4)
    dask.config.set(scheduler='synchronous')
    try:
        print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    except RuntimeError:
        pass

    EPOCHS = 30
    SAVE_INTERVAL = 1_000_000
    VALIDATE_INTERVAL = 500
    LOG_INTERVAL = 200

    # train_loaders = partial(
    #     get_train_datasets,
    #     cached_data=True,
    #     force_cache_reset=FORCE_CACHE_RESET,
    #     reset_cache_epoch_interval=RESET_CACHE_EPOCH_INTERVAL)

    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    l1_lambda_range = [0.0001, 0.001, 0.01, 0.1]
    dropout_rate_range = np.arange(0, 0.81, 0.1)
    alpha_range = np.arange(0, 1.01, 0.1)
    gamma_range = np.arange(0, 1.01, 0.1)

    # Define the number of iterations for the random search
    n_iterations = 10_000

    # Perform the random search
    for i in range(n_iterations):
        try:
            l1_lambda0 = np.random.choice(l1_lambda_range)
            dropout_rate0 = np.random.choice(dropout_rate_range)
            alpha = np.random.uniform(0, 1)  # uniformly sample from [0, 1]
            gamma = np.random.uniform(0, 1)  # uniformly sample from [0, 1]
            lr = np.random.choice(learning_rate)

            config_model0 = ConfigurationModel(
                model=models.EncoderDecoderZ(dropout_rate0),
                criterion=FocalLoss(alpha, gamma),
                l1_lambda=l1_lambda0,
                learning_rate=lr)

            config = Configuration(
                epochs=EPOCHS,
                samples_max=20_000,
                volume_dataset_cls=SampleXYZ,
                crop_box_cls=CropBoxSobol,
                suffix_cache='sobol',
                label_fn=centre_pixel,
                transformers=ann.transforms.transform_train,
                shuffle=True,
                balance_ink=True,
                box_width_xy=61,
                box_width_z=20,
                batch_size=32,
                num_workers=10,
                fragments=[1, 2, 3],
                in_memory_dataset=False,
                model0=config_model0)

            config_inference = copy.copy(config)
            config_inference.prefix = "/data/kaggle/input/vesuvius-challenge-ink-detection/test/"
            config_inference.fragments = ('a', 'b')
            # config_inference.prefix = "/data/kaggle/input/vesuvius-challenge-ink-detection/train/"
            # config_inference.fragments = (1, 2)
            config_inference.balance_ink = False
            config_inference.validation_steps = 1_000_000

            print(config)

            train_dataset = get_train_dataset(config, cached=True, reset_cache=False, val_data=False)
            val_dataset = get_train_dataset(config, cached=True, reset_cache=False, val_data=True)

            with Track() as track, timer("Training"):

                trainer = TrainerXYZ(config, track, train_dataset, val_dataset)

                for i, train in enumerate(trainer):

                    # if i % 2 ** 11 == 0:
                    #     fxy_idx_old = set(train.train_dataset.ds['fxy_idx'].values)
                    #
                    #     sigma = interpolate(10, 10, train.trackers.incrementer.loop / train.total_loops)
                    #     train.sigma = sigma
                    #     train.val_loader_iter = train.get_val_loader_iter(sigma)
                    #
                    #     fxy_idx_new = set(train.train_dataset.ds['fxy_idx'].values)
                    #
                    #     print('length', len(fxy_idx_new), len(fxy_idx_new.intersection(fxy_idx_old)))

                    train.forward0().loss().backward().step()

                    if i == 0:
                        continue
                    if i % LOG_INTERVAL == 0:
                        train.trackers.log_train()
                    if i % VALIDATE_INTERVAL == 0:
                        train.validate()
        except LossError:
            pass

        train.save_model()

        del train_dataset
        del val_dataset
        del train
        gc.collect()
        # break
