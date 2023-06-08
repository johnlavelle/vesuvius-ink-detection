import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Subset, DataLoader

from src import tensorboard_access
from vesuvius import ann
from vesuvius.ann import models
from vesuvius.config import Configuration, ConfigurationModel
from vesuvius.fragment_dataset2 import get_chunked_dataset
from vesuvius.labels import centre_pixel
from vesuvius.sample_processors import SampleXYZ
from vesuvius.sampler import CropBoxRegular
from vesuvius.trackers import Track
from vesuvius.trainer import BaseTrainer
from vesuvius.utils import timer


class AttentionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.outputs = None
        self.loss_value = None

        self.last_model = self.config.model0
        self.model0, self.optimizer0, self.scheduler0, self.criterion0 = self.setup_model(self.last_model)

    def zero_grad(self):
        self.optimizer0.zero_grad()
        return self

    def _apply_forward(self, datapoint) -> torch.Tensor:
        # Rescale z to be between 0 and 1
        # print(torch.cuda.memory_allocated() / 1024 ** 2)
        dp = datapoint.voxels
        # print(torch.flatten(self.model0(dp.to(self.device))))
        # dp = (dp - dp.mean()) / dp.std()
        return self.model0(dp.to(self.device))

    def validate(self) -> 'AttentionTrainer':
        self.model0.eval()
        with torch.no_grad():
            for datapoint_test in self.val_loader_iter:
                outputs = self._apply_forward(datapoint_test)
                val_loss = self.criterion0(outputs, datapoint_test.label.float().to(self.device))
                batch_size = len(datapoint_test.label)
                self.trackers.logger_test_loss.update(val_loss.item(), batch_size)
        self.model0.train()
        return self

    def forward0(self) -> 'AttentionTrainer':
        self.model0.train()
        self.outputs = self._apply_forward(self.datapoint)
        return self

    def loss(self) -> 'AttentionTrainer':
        target = self.datapoint.label.float().to(self.device)
        self.loss_value = self.criterion0(self.outputs, target)
        self.trackers.logger_loss.update(self.loss_value.item(), self.config.batch_size)
        self.trackers.logger_lr.update(self.scheduler0.get_last_lr()[0], self.config.batch_size)
        return self

    def backward(self) -> 'AttentionTrainer':
        self.loss_value.backward()
        return self

    def step(self) -> 'AttentionTrainer':
        self.optimizer0.step()
        self.scheduler0.step()
        return self

    def save_model(self):
        self._save_model(self.model0, suffix='0')


if __name__ == '__main__':
    # dask.config.set(scheduler='synchronous')

    try:
        print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
    except RuntimeError:
        print('Failed to get public tensorboard URL')

    TRAIN = True
    INFERENCE = False
    STORED_CONFIG = False

    EPOCHS = 20
    TOTAL_STEPS = 100_000
    SAVE_INTERVAL_MINUTES = 30
    VALIDATE_INTERVAL = 100
    LOG_INTERVAL = 10
    PRETRAINED_MODEL0 = False
    LEARNING_RATE = 0.3

    save_interval_seconds = SAVE_INTERVAL_MINUTES * 60

    if STORED_CONFIG:
        _config = Configuration.from_dict('configs/Joint/')
        config_model0 = _config.model0
        config_model0.model.requires_grad = False
        config_model1 = _config.model1
        config_model1.model.requires_grad = False
    else:
        config_model0 = ConfigurationModel(
            model=models.CNN1(),
            optimizer_scheduler_cls=ann.optimisers.SGDOneCycleLR,
            learning_rate=LEARNING_RATE,
            l1_lambda=0.0,
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
        stride_xy=91,
        stride_z=65,
        num_workers=6,
        validation_steps=100,
        accumulation_steps=1,
        model0=config_model0,
        fragments=(1, 2, 3)
    )

    # config_val = copy.copy(config)
    # config_val.transformers = ann.transforms.transform_val

    # config_inference = copy.copy(config)
    # config_inference.prefix = "/data/kaggle/input/vesuvius-challenge-ink-detection/test/"
    # config_inference.fragments = ('a', 'b')
    # # config_inference.prefix = "/data/kaggle/input/vesuvius-challenge-ink-detection/train/"
    # # config_inference.fragments = (1, 2)
    # config_inference.balance_ink = False
    # config_inference.validation_steps = 1_000_000

    dataset = get_chunked_dataset('train', (1, 2, 3))
    test_dataset = get_chunked_dataset('test', ('a', 'b'))
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, dataset_size))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # train_dataset = get_dataset_regular_z(config, False, validation=False)
    # val_dataset = get_dataset_regular_z(config_val, False, validation=True)
    # test_dataset = get_dataset_regular_z(config_inference, False, validation=False)
    # print(train_dataset[0])

    start_time = time.time()
    with Track() as track, timer("Training"):
        trainer = AttentionTrainer(config,
                                   track,
                                   train_dataset=train_loader,
                                   val_dataset=val_loader,
                                   test_dataset=test_loader)

        if TRAIN:
            for i, train in enumerate(trainer):
                pass
    #             train.zero_grad().forward0().loss().backward().step()
    #
    #             if i == 0:
    #                 continue
    #             if time.time() - start_time >= save_interval_seconds:
    #                 train.save_model()
    #                 start_time = time.time()
    #             if i % LOG_INTERVAL == 0:
    #                 train.trackers.log_train()
    #             if i % VALIDATE_INTERVAL == 0:
    #                 train.validate()
    #                 train.trackers.log_test()
    #         train.save_model()
    #
    # if INFERENCE:
    #     outputs, labels, coords, fragments = trainer.inference()
    #     # Convert list of tuples into two lists
    #     x_coord, y_coord = zip(*coords)
    #
    #     # Create DataFrame
    #     df = pd.DataFrame({
    #         'X': x_coord,
    #         'Y': y_coord,
    #         'outputs': outputs,
    #         'fragments': fragments
    #     })
    #
    #     # Aggregate duplicates by taking the mean
    #     # df_agg = df.groupby(['X', 'Y']).mean().reset_index()
    #     for frag in (1, 2):
    #         df_agg = df[df['fragments'] == frag]
    #         # Pivot DataFrame to create grid
    #         grid = df_agg.pivot(index='X', columns='Y', values='outputs')
    #
    #         grid.fillna(-1, inplace=True)
    #
    #         # Assuming 'grid' is your 2D array or DataFrame
    #         plt.figure(figsize=(10, 10))  # Adjust size as needed
    #         plt.imshow(grid.transpose(), cmap='hot',
    #                    interpolation='none')  # Change colormap as needed. Other options: 'cool', 'coolwarm', 'Greys', etc.
    #         plt.colorbar(label='labels')  # Shows a color scale
    #         plt.show()
    #
    #         df_agg['outputs'].plot.hist(bins=50)
    #         plt.show()
