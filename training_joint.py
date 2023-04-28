import copy
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model2 = None
        self.criterion2 = None
        self.output1 = None
        self.output2 = None
        self.optimizer2 = None
        self.scheduler2 = None
        self.config_test = None
        self.test_loader_len = None
        self.inputs = None
        self.output1 = None
        self.output2 = None
        self.os2 = None
        self.loss2 = None
        self.labels = None
        self.outputs_collected = None
        self.labels_collected = None
        self.setup_model2()

    def setup_model2(self) -> None:
        self.model2 = ann.models.BinaryClassifier().to(self.device)
        self.os2 = self.optimizer_scheduler_cls(self.model2, self.learning_rate, self.total)
        self.optimizer2 = self.os2.optimizer()
        self.scheduler2 = self.os2.scheduler()
        self.criterion2 = BCEWithLogitsLoss()
        print(self.os2, '\n')

    def _apply_forward(self, datapoint) -> torch.Tensor:
        voxels = datapoint.voxels
        dim0, dim1 = voxels.shape[:2]
        voxels = voxels.reshape(dim0 * dim1, *voxels.shape[2:])
        scalar = (datapoint.z_start / (65 - self.config.box_width_z)).view(-1, 1).float()
        return self.model(voxels.to(self.device), scalar.to(self.device))

    def forward(self) -> 'JointTrainer':
        self.model.train()
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
        self.model2.train()
        self.output2 = self.model2(self.outputs)
        return self

    def loss(self) -> 'JointTrainer':
        self.loss2 = self.criterion2(self.output2, self.labels)
        self.trackers.update_train(self.loss2.item(), self.labels.shape[0])
        return self

    def backward(self) -> 'JointTrainer':
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        self.loss2.backward()
        return self

    def step(self) -> 'JointTrainer':
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer2.step()
        self.scheduler2.step()
        return self

    def validate(self) -> 'JointTrainer':
        self.model.eval()
        self.model2.eval()
        iterations = 50
        iterations = round(self.config.batch_size * (iterations / self.config.batch_size))
        with torch.no_grad():
            for datapoint_test in islice(self.test_loader_iter, iterations):
                output = self._apply_forward(datapoint_test)
                output = output.reshape(datapoint_test.label.shape[:2])
                output2 = self.model2(output)
                labels = datapoint_test.label.float().mean(dim=1)
                labels = labels.to(self.device)
                val_loss = self.criterion2(output2, labels)
                self.trackers.update_test(val_loss.item(), len(labels))
            self.trackers.update_lr(self.scheduler2.get_last_lr()[0], len(labels))
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
        lr = self.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"Current Loop: {loop}, Learning Rate: {lr}, Epochs: {epochs}, Batch Size: {batch_size}"

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        loop = self.trackers.incrementer.loop
        lr = self.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"{classname}(current_loop={loop}, learning_rate={lr}, epochs={epochs}, batch_size={batch_size})"


# get_train_loader_regular_z = partial(get_train_loader_regular_z, force_cache_reset=False)

if __name__ == '__main__':
    try:
        print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    except RuntimeError:
        print('Failed to get public tensorboard URL')

    EPOCHS = 1
    VALIDATE_INTERVAL = 100
    LOG_INTERVAL = 10

    xl, yl = 2048, 7168  # lower left corner of the test box
    width, height = 2045, 2048
    config_kwargs = dict(training_steps=10_000_000,
                         model=ann.models.HybridModel,
                         volume_dataset_cls=SampleXYZ,
                         crop_box_cls=CropBoxRegular,
                         suffix_cache='regular',
                         test_box=(xl, yl, xl + width, yl + height),  # Hold back rectangle
                         test_box_fragment=2,  # Hold back fragment
                         transformers=ann.transforms.transform1,
                         shuffle=False,
                         group_pixels=True,
                         balance_ink=True,
                         stride_xy=91,
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
            train.forward().forward2().loss().backward().step()

            if i == 0:
                continue
            if i % LOG_INTERVAL == 0:
                train.trackers.log_train()
            if i % VALIDATE_INTERVAL == 0:
                train.validate()
                train.trackers.log_test()
        train.save_model_output()
