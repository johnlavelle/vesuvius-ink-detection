import copy
import multiprocessing as mp
import pprint
from dataclasses import dataclass
from itertools import islice
from timeit import default_timer as timer
from typing import Generator

import dask
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from xarray import DataArray

from vesuvius.config import Configuration
from vesuvius.data_io import SaveModel, LoadModel
from vesuvius.dataloader import get_train_loader, get_test_loader, get_train_loader_regular_z
from vesuvius.datapoints import Datapoint, DatapointTuple
from vesuvius.models import CNN1
from vesuvius.sampler import CropBoxSobol, CropBoxRegular
from vesuvius.scroll_dataset import BaseDataset
from vesuvius.trackers import TrackerAvg

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
FORCE_CACHE_RESET = False  # Deletes cache. Only used if CACHED_DATA is True.
RESET_CACHE_EPOCH_INTERVAL = 10
SAVE_MODEL_INTERVAL = 1_000
TEST_MODEL = True


class SampleXYZ(BaseDataset):

    def get_item_as_data_array(self, index: int) -> DatapointTuple:
        rnd_slice = self.get_slice(index)
        slice_xy = {key: value for key, value in rnd_slice.items() if key in ('x', 'y')}
        label = self.ds.labels.sel(**slice_xy).transpose('x', 'y')
        voxels = self.ds.images.sel(**rnd_slice).transpose('z', 'x', 'y')
        # voxels = self.normalise_voxels(voxels)
        voxels = voxels.expand_dims('Cin')
        s = rnd_slice
        dp = Datapoint(voxels, int(self.label_operation(label)), self.ds.fragment,
                       s['x'].start, s['x'].stop, s['y'].start, s['y'].stop, s['z'].start, s['z'].stop)
        return dp.to_namedtuple()


# Label processors


def centre_pixel(da: DataArray) -> DataArray:
    return da.isel(x=len(da.x) // 2, y=len(da.y) // 2).astype(np.float32)


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


def get_config_model(config_path: str, model_path: str):
    lm = LoadModel(config_path, model_path)
    return convert_config(lm.config()), lm.model()


if READ_CONFIG_FILE:
    loader = LoadModel('output/save_model/20230412_235004', 1)
    model = loader.model()
    config = loader.config
else:
    # Hold back data test box for fragment
    XL, YL = 1100, 3500  # lower left corner of the test box
    WIDTH, HEIGHT = 700, 950

    print('Getting config from Configuration instantiation...\n')
    config = Configuration(info="Sampling every 5th layer",
                           model=CNN1,
                           volume_dataset_cls=SampleXYZ,
                           crop_box_cls=CropBoxSobol,
                           label_fn=centre_pixel,
                           training_steps=1_00,  # If caching, this should be small enough to fit on disk
                           batch_size=32,
                           fragments=[1, 2, 3],
                           test_box=(XL, YL, XL + WIDTH, YL + HEIGHT),  # Hold back rectangle
                           test_box_fragment=1,  # Hold back fragment
                           box_width_xy=61,
                           box_width_z=8,
                           balance_ink=True,
                           shuffle=True,
                           group_pixels=False,
                           num_workers=min(1, mp.cpu_count() - 1),
                           prefix='/data/kaggle/vesuvius/train/',
                           suffix_cache='sobol',
                           collate_fn=None)

config_regular_z = copy.copy(config)
config_regular_z.suffix_cache = 'regular'
config_regular_z.crop_box_cls = CropBoxRegular
config_regular_z.shuffle = False
config_regular_z.group_pixels = True
config_regular_z.balance_ink = False
config_regular_z.sampling = 5
config_regular_z.stride_xy = 61
config_regular_z.stride_z = 6


# Training


def train_loaders(cfg) -> Generator[Datapoint, None, None]:
    for epoch in range(EPOCHS):
        reset_cache = CACHED_DATA and (epoch != 0 and epoch % RESET_CACHE_EPOCH_INTERVAL == 0)
        reset_cache = reset_cache or (epoch == 0 and FORCE_CACHE_RESET)
        yield from get_train_loader(cfg, cached=CACHED_DATA, reset_cache=reset_cache)


if __name__ == '__main__':

    model = config.model().to(DEVICE)

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

    train_loader_sobol = train_loaders(config)
    test_loader_sobol = get_test_loader(config)
    # train_loader_regular gets samples across the z dimension
    train_loader_regular_z = get_train_loader_regular_z(config_regular_z, FORCE_CACHE_RESET)

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

    saver = SaveModel(1)

    model.train()
    time_start = timer()
    tqdm_kwargs = dict(total=total, disable=False, desc='Training', position=0)
    i = None
    for i, datapoint in tqdm(enumerate(train_loader_sobol), **tqdm_kwargs):
        if i >= total:
            break
        optimizer.zero_grad()
        outputs = model(datapoint.voxels.to(DEVICE))
        loss = criterion(outputs, datapoint.label.float().to(DEVICE))
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.update(loss.item(), len(datapoint.voxels))
        lr_track.update(scheduler.get_last_lr()[0], 1)

        if i == 0:
            continue

        if i % SAVE_MODEL_INTERVAL == 0:
            saver.model(model)
            saver.config(config)

        if not i % 5 == 0:
            continue

        model.eval()
        with torch.no_grad():
            for iv, datapoint_test in enumerate(islice(train_loader_regular_z, 5)):
                outputs = model(datapoint_test.voxels.to(DEVICE))
                val_loss = criterion(outputs, datapoint_test.label.float().to(DEVICE))
                test_loss.update(val_loss.item(), len(datapoint_test.voxels))

        train_loss.log(i)
        lr_track.log(i)
        test_loss.log(i)
        model.train()

    else:
        if i < total - 1:
            print("Training stopped early. Try clearing the cache "
                  "(with FORCE_CACHE_REST = True) and restart the training.")

    config['performance_dict'] = {'loss/train': train_loss.average_loss,
                                  'loss/test': test_loss.average_loss,
                                  'steps': i}
    saver.model(model)
    saver.config(config)

    writer.flush()
    time_end = timer()
    time_spent = time_end - time_start
    print(f"Training took {time_spent:.1f} seconds")
