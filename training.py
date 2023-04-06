import datetime
import pprint
import time
from dataclasses import dataclass
from typing import Tuple, Any

import dask
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from xarray import DataArray

from vesuvius.config import Configuration, read_config, save_config
from vesuvius.data_io import read_dataset_from_zarr
from vesuvius.dataloader import data_loader
from vesuvius.dataset import BaseDataset
from vesuvius.sampler import CropBoxSobol
from vesuvius.train_utils import TrackerAvg


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
SAVE_MODEL_INTERVAL = 1_00_000


# Create samples
# No affect if CACHED_DATA is True

class SubVolumeDatasets(BaseDataset):

    def get_item_as_data_array(self, index: int) -> Tuple[DataArray, DataArray]:
        assert isinstance(index, int)
        assert index >= 0
        if index >= self.max_iterations:
            raise StopIteration
        rnd_slice = self.get_slice(index)
        ink_labs = self.ds.labels.sel(**rnd_slice).transpose('x', 'y')
        voxels = self.ds.images.sel(**rnd_slice).transpose('z', 'x', 'y')
        voxels = self.normalise_voxels(voxels)
        return self.postprocess_data_array(voxels, ink_labs)


class SubVolumeDatasets2(BaseDataset):

    def get_item_as_data_array(self, index: int) -> Tuple[DataArray, DataArray]:
        assert isinstance(index, int)
        assert index >= 0
        if index >= self.max_iterations:
            raise StopIteration
        rnd_slice = self.get_slice(index)
        xy_slice = {'x': slice(rnd_slice['x'].start, rnd_slice['x'].stop, 1),
                    'y': slice(rnd_slice['y'].start, rnd_slice['y'].stop, 1)}
        z_slice = {'z': slice(self.ds['z'][0], self.ds['z'][-1], 5)}
        ink_labs = self.ds.labels.sel(**xy_slice).transpose('x', 'y')
        voxels = self.ds.images.sel(**xy_slice).sel(**z_slice).transpose('z', 'x', 'y')
        voxels = self.normalise_voxels(voxels)
        return self.postprocess_data_array(voxels, ink_labs)


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
                           volume_dataset_cls=SubVolumeDatasets2,
                           crop_box_cls=CropBoxSobol,
                           label_fn=centre_pixel,
                           collate_fn=None,
                           training_steps=10_000,  # If caching, this should be small enough to fit on disk
                           batch_size=32,
                           fragments=[1, 2, 3],
                           box_width_sample=61,
                           prefix='/data/kaggle/vesuvius/train/',
                           test_box=(XL, YL, XL + WIDTH, YL + HEIGHT),  # Hold back rectangle
                           test_box_fragment=1,  # Hold back fragment
                           z_limit=(0, 64 - 13),
                           balance_ink=True,
                           num_workers=5)

model = config.model().to(DEVICE)


# rnd_vals = torch.randn(config.batch_size,
#                        1,
#                        config.z_limit[1] - config.z_limit[0] + 1,
#                        config.box_width_sample,
#                        config.box_width_sample).to(DEVICE)
# ma = model.forward(rnd_vals)

# Training

def save_model_and_config(configuration: dataclass, torch_model: Any, prefix: str = ''):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(torch_model.state_dict(), f"output/save_model/{prefix}_model_{now}.pt")
    save_config(configuration, f"output/save_model/{prefix}_config_{now}.json")


def train_loaders():
    for epoch in range(EPOCHS):
        reset_cache = CACHED_DATA and (epoch != 0 and epoch % RESET_CACHE_EPOCH_INTERVAL == 0)
        reset_cache = reset_cache or (epoch == 0 and FORCE_CACHE_REST)
        yield from data_loader(config, cached=CACHED_DATA, reset_cache=reset_cache)


def get_test_loader():
    """Hold back data test box for fragment 1"""
    ds_test = read_dataset_from_zarr(config.test_box_fragment, config.num_workers, config.prefix)
    ds_test = ds_test.isel(x=slice(config.test_box[0], config.test_box[2]),
                           y=slice(config.test_box[1], config.test_box[3]),
                           z=slice(*config.z_limit))
    ds_test.load()
    ds_test['full_mask'] = ds_test['mask']
    test_dataset = config.volume_dataset_cls(ds_test,
                                             config.box_width_sample,
                                             1_000_000,
                                             crop_box_cls=CropBoxSobol,
                                             label_operation=config.label_fn,
                                             balance_ink=False)
    return DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)


if __name__ == '__main__':
    # print(f'Shape of sample batches: {rnd_vals.shape}', '\n')

    print(model, '\n')
    pp.pprint(config)
    print('\n')

    test_loader = get_test_loader()

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
    time_start = time.time()
    tqdm_kwargs = dict(total=total, disable=False, desc='Training', position=0)
    for i, (sub_volumes, ink_labels) in tqdm(enumerate(train_loaders()), **tqdm_kwargs):
        if i >= total:
            break

        optimizer.zero_grad()
        outputs = model(sub_volumes.to(DEVICE))
        loss = criterion(outputs, ink_labels.to(DEVICE))
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.update(loss.item(), len(sub_volumes))
        lr_track.update(scheduler.get_last_lr()[0], 1)

        if i != 0 and i % 100 == 0:
            with torch.no_grad():
                model.eval()
                cur_learning_rate = scheduler.get_last_lr()[0]
                for iv, (sub_volumes_val, ink_labels_val) in enumerate(test_loader):
                    if iv >= 2:
                        break
                    with torch.no_grad():
                        outputs = model(sub_volumes_val.to(DEVICE))
                        val_loss = criterion(outputs, ink_labels_val.to(DEVICE))

                        test_loss.update(val_loss.item(), len(sub_volumes_val))

            train_loss.log(i)
            lr_track.log(i)
            test_loss.log(i)

        if i != 0 and i % SAVE_MODEL_INTERVAL == 0:
            save_model_and_config(config, model, prefix='intermediate')

        model.train()

    else:
        print("Stopping training early. Try clearing the cache "
              "(with FORCE_CACHE_REST = True) and restarting the training.")

    save_model_and_config(config, model)
    writer.flush()
    time_end = time.time()
    time_spent = time_end - time_start
    print(f"Training took {time_spent:.1f} seconds")
