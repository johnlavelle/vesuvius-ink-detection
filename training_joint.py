import pprint
from itertools import repeat, chain, cycle, islice

import dask
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from src import tensorboard_access
from vesuvius import ann
from vesuvius.ann.optimisers import SGDOneCycleLR
from vesuvius.dataloader import get_train_loader_regular_z
from vesuvius.sample_processors import SampleXYZ
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

        self.model2 = None
        self.os2 = None
        self.optimizer2 = None
        self.scheduler2 = None
        self.criterion2 = None
        self.train_loader_iter = None
        self.test_loader_iter = None
        self.labels = None
        self.output2 = None
        self.test_loader_len = None
        self.outputs_collected = []
        self.labels_collected = []

        self.setup_model2()
        self.get_train_test_loaders()

    def setup_model2(self) -> None:
        self.model2 = ann.models.BinaryClassifier().to(self.device)
        self.os2 = self.optimizer_scheduler_cls(self.model2, self.learning_rate, self.total)
        self.optimizer2 = self.os2.optimizer()
        self.scheduler2 = self.os2.scheduler()
        self.criterion2 = BCEWithLogitsLoss()
        print(self.os2, '\n')

    def get_train_test_loaders(self) -> None:
        self.batch_size = round(self.config.batch_size / ((65 - self.config.stride_z) / self.config.stride_z + 1))

        dataloader_train = DataLoader(self.train_loader(self.config, False, test_data=False),
                                      batch_size=self.batch_size,
                                      num_workers=self.config.num_workers)
        self.total = self.epochs * len(dataloader_train) * self.batch_size
        self.loops = self.epochs * len(dataloader_train)
        self.train_loader_iter = chain.from_iterable(repeat(dataloader_train, self.epochs))

        test_loader = DataLoader(self.train_loader(self.config, False, test_data=True),
                                 batch_size=self.batch_size,
                                 num_workers=self.config.num_workers)

        self.test_loader_len = len(test_loader)
        self.test_loader_iter = cycle(iter(test_loader))

    def _apply_forward(self, datapoint) -> torch.Tensor:
        voxels = datapoint.voxels
        dim0, dim1 = voxels.shape[:2]
        voxels = voxels.reshape(dim0 * dim1, *voxels.shape[2:])
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model(voxels.to(self.device), scalar.to(self.device))

    def forward(self) -> 'BaseTrainer':
        self.model.train()
        self.trackers.increment(train.datapoint.label.shape[0])

        self.outputs_collected = []
        self.labels_collected = []
        for _ in range(5):
            output = self._apply_forward(self.datapoint)
            self.outputs_collected.append(output.reshape(self.datapoint.label.shape[:2]))
            self.labels_collected.append(self.datapoint.label.float().mean(dim=1))
        self.outputs = torch.cat(train.outputs_collected, dim=0)
        self.labels = torch.cat(train.labels_collected, dim=0)
        return self

    def forward2(self) -> 'BaseTrainer':
        self.model2.train()
        self.output2 = self.model2(self.outputs)
        return self

    def loss(self) -> 'BaseTrainer':
        self.labels = self.labels.to(self.device)
        loss2 = self.criterion2(self.output2, self.labels)
        self.trackers.update_train(loss2.item(), self.labels.shape[0])
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer.step()
        self.optimizer2.step()
        return self

    def validate(self) -> 'BaseTrainer':
        self.model.eval()
        self.model2.eval()
        with torch.no_grad():
            for datapoint_test in islice(self.test_loader_iter, 200):
                output = self._apply_forward(datapoint_test)
                output = output.reshape(self.datapoint.label.shape[:2])
                output2 = self.model2(output)
                labels = datapoint_test.label.float().mean(dim=1)
                labels = labels.to(self.device)
                val_loss = self.criterion2(output2, labels)
                self.trackers.update_test(val_loss.item(), len(labels))
        self.trackers.update_lr(self.optimizer.param_groups[0]['lr'])
        return self


# get_train_loader_regular_z = partial(get_train_loader_regular_z, force_cache_reset=False)

if __name__ == '__main__':
    try:
        print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    except RuntimeError:
        print('Failed to get public tensorboard URL')

    EPOCHS = 50
    VALIDATE_INTERVAL = 5000
    LOG_INTERVAL = 100

    xl, yl = 2048, 7168  # lower left corner of the test box
    width, height = 2045, 2048
    config_kwargs = dict(training_steps=10_000_000,
                         model=ann.models.HybridModel,
                         volume_dataset_cls=SampleXYZ,
                         crop_box_cls=CropBoxRegular,
                         suffix_cache='regular',
                         test_box=(xl, yl, xl + width, yl + height),  # Hold back rectangle
                         test_box_fragment=2,  # Hold back fragment
                         shuffle=False,
                         group_pixels=True,
                         balance_ink=True,
                         stride_xy=61,
                         stride_z=6,
                         num_workers=0)

    with Track() as track, timer("Training"):

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
            train.forward().forward2().loss()
            if i == 0:
                continue
            if i % LOG_INTERVAL == 0:
                train.trackers.log_train()
            if i % VALIDATE_INTERVAL == 0:
                train.validate()
                train.trackers.log_test()
        train.save_model_output()
