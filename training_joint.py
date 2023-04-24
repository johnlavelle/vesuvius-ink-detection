import pprint
from functools import partial
from itertools import islice, repeat, chain

import dask
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

import tensorboard_access
from vesuvius import ann
from vesuvius.ann.optimisers import SGDOneCycleLR
from vesuvius.dataloader import get_train_loader_regular_z
from vesuvius.dataloader import get_train_loaders
from vesuvius.sampler import CropBoxRegular, SampleXYZ
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

        self.model2 = None
        self.optimizer_scheduler2 = None
        self.optimizer2 = None
        self.scheduler2 = None
        self.criterion2 = None
        self.train_loader_iter = None
        self.test_loader_iter = None
        self.outputs_collected = []
        self.labels_collected = []
        self.labels = None
        self.output2 = None

        self.setup_model2()
        self.get_train_test_loaders()

    def setup_model2(self):
        self.model2 = ann.models.BinaryClassifier().to(self.device)
        self.optimizer_scheduler2 = self.optimizer_scheduler_cls(self.model2, self.learning_rate, self.total)
        self.optimizer2 = self.optimizer_scheduler2.optimizer()
        self.scheduler2 = self.optimizer_scheduler2.scheduler(self.optimizer)
        self.criterion2 = BCEWithLogitsLoss()
        print(self.optimizer_scheduler2, '\n')

    def get_train_test_loaders(self) -> None:
        self.batch_size = round(self.config.batch_size / ((65 - self.config.stride_z) / self.config.stride_z + 1))

        dataloader = DataLoader(self.train_loader(self.config, False),
                                batch_size=self.batch_size,
                                num_workers=self.config.num_workers)

        self.total = self.epochs * len(dataloader) * self.batch_size
        self.loops = self.epochs * len(dataloader)
        self.train_loader_iter = chain.from_iterable(repeat(dataloader, self.epochs))
        self.test_loader_iter = lambda n: islice(iter(self.test_loader(self.config, False)), n)

    def _apply_forward(self, datapoint) -> torch.Tensor:
        voxels = datapoint.voxels
        dim0, dim1 = voxels.shape[:2]
        voxels = voxels.reshape(dim0 * dim1, *voxels.shape[2:])
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model(voxels.to(self.device), scalar.to(self.device))

    def forward(self) -> 'BaseTrainer':
        self.model.train()
        self.outputs_collected = []
        self.labels_collected = []
        for _ in range(5):
            output = self._apply_forward(self.datapoint)
            self.outputs_collected.append(output.reshape(self.datapoint.label.shape[:2]))
            self.labels_collected.append(self.datapoint.label.float().mean(dim=1))
        self.outputs = torch.cat(train.outputs_collected, dim=0)
        self.labels = torch.cat(train.labels_collected, dim=0)
        return self

    def forward2(self):
        self.output2 = self.model2(self.outputs)

    def loss(self) -> 'BaseTrainer':
        self.labels = self.labels.to(self.device)
        loss2 = self.criterion2(self.output2, self.labels)
        self.trackers.logger_loss.update(loss2.item(), self.labels.shape[0])
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer.step()
        self.optimizer2.step()
        return self

    def validate(self) -> 'BaseTrainer':
        self.model.eval()
        with torch.no_grad():
            for datapoint_test in self.test_loader_iter(self.validation_steps):
                voxels = datapoint_test
                voxels = voxels.reshape(self.datapoint.label.shape[:2])
                outputs = self._apply_forward(datapoint_test)
                val_loss = self.criterion_validate(outputs, datapoint_test.label.float().to(self.device))
                batch_size = len(datapoint_test.label)
                self.trackers.logger_test_loss.update(val_loss.item(), batch_size)
        self.trackers.log_test()
        return self


# get_train_loader_regular_z = partial(get_train_loader_regular_z, force_cache_reset=False)

if __name__ == '__main__':
    print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')

    CACHED_DATA = True
    FORCE_CACHE_RESET = False  # Deletes cache. Only used if CACHED_DATA is True.
    EPOCHS = 1000
    RESET_CACHE_EPOCH_INTERVAL = EPOCHS
    VALIDATE_INTERVAL = 1000
    LOG_INTERVAL = 100

    train_dataset = partial(
        get_train_loaders,
        cached_data=CACHED_DATA,
        force_cache_reset=FORCE_CACHE_RESET,
        reset_cache_epoch_interval=RESET_CACHE_EPOCH_INTERVAL)

    with Track() as track, Track() as trackers2, timer("Training"):

        config_kwargs = dict(training_steps=1000,
                             model=ann.models.HybridModel,
                             volume_dataset_cls=SampleXYZ,
                             crop_box_cls=CropBoxRegular,
                             suffix_cache='regular',
                             shuffle=False,
                             group_pixels=True,
                             balance_ink=True,
                             stride_xy=61,
                             stride_z=6,
                             num_workers=6)
        trainer = JointTrainer(get_train_loader_regular_z,
                               get_train_loader_regular_z,
                               track,
                               SGDOneCycleLR,
                               BCEWithLogitsLoss,
                               config_kwargs=config_kwargs,
                               learning_rate=0.03,
                               l1_lambda=0,
                               epochs=EPOCHS)

        for i, train in enumerate(trainer):
            train.forward()
            train.forward2()
            train.loss()
            train.validate()

            if i == 0:
                continue
            if i % LOG_INTERVAL == 0:
                train.trackers.logger_loss.log(i)
            #     if i % VALIDATE_INTERVAL == 0:
            #         train.validate()
            #
            # train.save_model_output()
