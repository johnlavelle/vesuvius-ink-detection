import os
import random
import shutil
import time
from typing import Union

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm

from vesuvius.config import Configuration
from vesuvius.config import save_config
from vesuvius.data_io import dataset_to_zarr
from vesuvius.data_io import read_dataset_from_zarr
from vesuvius.data_utils import Datapoint
from vesuvius.scroll_dataset import CachedDataset
from vesuvius.sampler import CropBoxSobol
from . import scroll_datasets

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

worker_seed = None


def worker_init_fn(worker_id: int):
    global worker_seed
    seed = int(time.time()) ^ (worker_id + os.getpid())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standard_data_loader(configuration: Configuration) -> DataLoader:
    xarray_dataset_iter = scroll_datasets.XarrayDatasetIter(configuration)
    datasets = scroll_datasets.TorchDatasetIter(configuration, xarray_dataset_iter)
    datasets = ConcatDataset(datasets)
    return DataLoader(datasets,
                      batch_size=configuration.batch_size,
                      num_workers=configuration.num_workers,
                      prefetch_factor=None,
                      shuffle=True,
                      worker_init_fn=worker_init_fn,
                      collate_fn=configuration.collate_fn)


def cached_data_loader(cfg: Configuration, reset_cache: bool = False) -> Dataset:
    cache_dir = os.path.join(cfg.prefix, 'data_cache')
    zarr_dir = os.path.join(cache_dir, 'cache.zarr')

    if reset_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    if not os.path.isdir(zarr_dir):
        os.makedirs(cache_dir, exist_ok=True)

        train_loader = get_train_loader(cfg)

        # Save the output of the data loader to a zarr file

        total = cfg.training_steps / cfg.batch_size
        running_sample_len = 0
        datapoint: Datapoint
        for i, datapoint in tqdm(enumerate(train_loader), total=total,
                                 desc='Caching data', position=1, leave=False):
            # Create a dataset with the samples and labels
            sub_volume_len = datapoint.voxels.shape[0]
            sub_volume_coord = np.arange(running_sample_len, running_sample_len + sub_volume_len)
            coords = {'sample': sub_volume_coord}
            voxels_da = xr.DataArray(datapoint.voxels.numpy(), dims=('sample', 'empty', 'z', 'x', 'y'), coords=coords)
            label_da = xr.DataArray(datapoint.label, dims=('sample', 'empty'), coords=coords)
            samples_labels = {'voxels': voxels_da, 'label': label_da}
            dp = datapoint._asdict()
            parameters = {k: xr.DataArray(v, dims=('sample', 'empty'), coords=coords) for k, v in dp.items() if
                          k not in samples_labels.keys()}
            ds = xr.Dataset({**samples_labels, **parameters})

            dataset_to_zarr(ds, zarr_dir, 'sample')
            save_config(cfg, os.path.join(cache_dir, f"config.json"))

            running_sample_len += sub_volume_len
    return CachedDataset(zarr_dir, cfg.batch_size)


def get_train_loader(cfg: Configuration, cached=False, reset_cache=False) -> Union[DataLoader, Dataset]:
    if reset_cache and not cached:
        raise ValueError("reset_cache can only be True if cached is also True")
    if cached:
        return cached_data_loader(cfg, reset_cache)
    else:
        return standard_data_loader(cfg)


def get_test_loader(cfg: Configuration) -> DataLoader:
    """Hold back data test box for fragment 1"""
    ds_test = read_dataset_from_zarr(cfg.test_box_fragment, cfg.num_workers, cfg.prefix)
    ds_test.attrs['fragment'] = cfg.test_box_fragment
    ds_test = ds_test.isel(x=slice(cfg.test_box[0], cfg.test_box[2]),
                           y=slice(cfg.test_box[1], cfg.test_box[3]))
    ds_test.load()
    ds_test['full_mask'] = ds_test['mask']
    test_dataset = cfg.volume_dataset_cls(ds_test,
                                          cfg.box_width_xy,
                                          cfg.box_width_z,
                                          max_iterations=cfg.training_steps,
                                          crop_cls=CropBoxSobol,
                                          label_operation=cfg.label_fn,
                                          balance_ink=False)
    return DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
