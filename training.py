import datetime
import pprint
from timeit import default_timer as timer
from itertools import islice
from dataclasses import dataclass
from typing import Any, Generator

import dask
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from xarray import DataArray
import multiprocessing as mp

from vesuvius.config import Configuration, read_config, save_config
from vesuvius.dataloader import get_train_loader, get_test_loader
from vesuvius.scroll_dataset import BaseDataset
from vesuvius.sampler import CropBoxSobol
from vesuvius.data_utils import TrackerAvg, Datapoint


pp = pprint.PrettyPrinter(indent=4)
dask.config.set(scheduler='synchronous')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using the {DEVICE}\n')

# If READ_EXISTING_CONFIG is False, config is specified in Configuration (below)
# else config is read from CONFIG_PATH.
READ_CONFIG_FILE = False
CONFIG_PATH = 'configs/config.json'

EPOCHS = 1
CACHED_DATA = True
FORCE_CACHE_REST = False  # Deletes cache. Only used if CACHED_DATA is True.
RESET_CACHE_EPOCH_INTERVAL = 10
SAVE_MODEL_INTERVAL = 100_000
TEST_MODEL = True


class SampleXYZ(BaseDataset):

    def get_item_as_data_array(self, index: int) -> Datapoint:
        rnd_slice = self.get_slice(index)
        slice_xy = {key: value for key, value in rnd_slice.items() if key in ('x', 'y')}
        label = self.ds.labels.sel(**slice_xy).transpose('x', 'y')
        voxels = self.ds.images.sel(**rnd_slice).transpose('z', 'x', 'y')
        # voxels = self.normalise_voxels(voxels)
        s = rnd_slice
        dp = Datapoint(voxels, label, self.ds.fragment,
                       s['x'].start, s['x'].stop, s['y'].start, s['y'].stop, s['z'].start, s['z'].stop)
        return dp


# Label processors


def centre_pixel(da: DataArray) -> DataArray:
    return da.isel(x=len(da.x) // 2, y=len(da.y) // 2).astype(np.float32)


# Models


def cnn1_sequential():
    return nn.Sequential(
        nn.Conv3d(1, 16, 3, 1, 1),
        nn.MaxPool3d(2, 2),
        nn.Conv3d(16, 32, 3, 1, 1),
        nn.MaxPool3d(2, 2),
        nn.Conv3d(32, 64, 3, 1, 1),
        nn.MaxPool3d(2, 2),
        nn.Flatten(start_dim=1),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.LazyLinear(1),
        nn.Sigmoid())


class CNN1(nn.Module):

    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.LazyLinear(128)
        self.relu = nn.ReLU()  # Use a single ReLU instance
        self.fc2 = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Configuration


def convert_config(cfg: dataclass) -> dataclass:
    """Convert the string representation of non-basic data types
    in a dataclass object to their corresponding objects."""
    for key, type_ in config.__annotations__.items():
        if type_ not in (str, int, float, bool, list, tuple, dict):
            value = getattr(cfg, key)
            if isinstance(value, str):
                setattr(cfg, key, eval(value))
    return cfg


if READ_CONFIG_FILE:
    print('Reading existing config file...\n')
    config = read_config(CONFIG_PATH)
    config = convert_config(config)
else:
    # Hold back data test box for fragment
    XL, YL = 1100, 3500  # lower left corner of the test box
    WIDTH, HEIGHT = 700, 950

    print('Getting config from Configuration instantiation...\n')
    config = Configuration(info="Sampling every 5th layer",
                           model=cnn1_sequential,
                           volume_dataset_cls=SampleXYZ,
                           crop_box_cls=CropBoxSobol,
                           label_fn=centre_pixel,
                           training_steps=1_000,  # If caching, this should be small enough to fit on disk
                           batch_size=32,
                           fragments=[1, 2, 3],
                           prefix='/data/kaggle/vesuvius/train/',
                           test_box=(XL, YL, XL + WIDTH, YL + HEIGHT),  # Hold back rectangle
                           test_box_fragment=1,  # Hold back fragment
                           box_width_xy=61,
                           box_width_z=10,
                           balance_ink=True,
                           num_workers=min(1, mp.cpu_count() - 1),
                           collate_fn=None)

model = config.model().to(DEVICE)


# Training


def save_model_and_config(cfg: dataclass, torch_model: Any, prefix: str = ''):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(torch_model.state_dict(), f"output/save_model/{prefix}_model_{now}.pt")
    save_config(cfg, f"output/save_model/{prefix}_config_{now}.json")


def train_loaders() -> Generator[Datapoint, None, None]:
    for epoch in range(EPOCHS):
        reset_cache = CACHED_DATA and (epoch != 0 and epoch % RESET_CACHE_EPOCH_INTERVAL == 0)
        reset_cache = reset_cache or (epoch == 0 and FORCE_CACHE_REST)
        yield from get_train_loader(config, cached=CACHED_DATA, reset_cache=reset_cache)


if __name__ == '__main__':

    print(model, '\n')
    pp.pprint(config)
    print('\n')

    if TEST_MODEL:
        rnd_vals = torch.randn(config.batch_size,
                               1,
                               config.box_width_z,
                               config.box_width_xy,
                               config.box_width_xy).to(DEVICE)
        print(f'Shape of sample batches: {rnd_vals.shape}', '\n')
        ma = model.forward(rnd_vals)

    test_loader = get_test_loader(config)

    # Train

    writer = SummaryWriter('output/runs/')

    total = EPOCHS * config.training_steps // config.batch_size

    LEARNING_RATE = 0.03  # Maximum learning rate
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=total)
    criterion = nn.BCELoss()
    config.update_nn_kwargs(optimizer, scheduler, criterion, LEARNING_RATE, EPOCHS)

    train_loss = TrackerAvg('loss/train', writer)
    test_loss = TrackerAvg('loss/test', writer)
    lr_track = TrackerAvg('stats/lr', writer)

    model.train()
    time_start = timer()
    tqdm_kwargs = dict(total=total, disable=False, desc='Training', position=0)
    i = None
    for i, datapoint in tqdm(enumerate(train_loaders()), **tqdm_kwargs):
        if i >= total:
            break
        optimizer.zero_grad()
        outputs = model(datapoint.voxels.to(DEVICE))
        loss = criterion(outputs, datapoint.label.to(DEVICE))
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.update(loss.item(), len(datapoint.voxels))
        lr_track.update(scheduler.get_last_lr()[0], 1)

        if i == 0:
            continue

        if i % SAVE_MODEL_INTERVAL == 0:
            save_model_and_config(config, model, prefix='intermediate')

        if not i % 100 == 0:
            continue

        model.eval()
        with torch.no_grad():
            for iv, datapoint_test in enumerate(islice(test_loader, 5)):
                outputs = model(datapoint_test.voxels.to(DEVICE))
                val_loss = criterion(outputs, datapoint_test.label.to(DEVICE))
                test_loss.update(val_loss.item(), len(datapoint_test.voxels))

        train_loss.log(i)
        lr_track.log(i)
        test_loss.log(i)
        model.train()

    else:
        if i < total - 1:
            print("Training stopped early. Try clearing the cache "
                  "(with FORCE_CACHE_REST = True) and restart the training.")

    save_model_and_config(config, model)
    writer.flush()
    time_end = timer()
    time_spent = time_end - time_start
    print(f"Training took {time_spent:.1f} seconds")
