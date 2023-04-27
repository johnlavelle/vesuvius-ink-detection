import os
import random
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Union, Generator

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from vesuvius.config import Configuration1
from vesuvius.data_io import dataset_to_zarr, get_dataset
from vesuvius.data_io import read_dataset_from_zarr, SaveModel
from vesuvius.datapoints import Datapoint
from vesuvius.fragment_dataset import CachedDataset
from vesuvius.utils import CustomDataLoaderError
from . import fragment_datasets

worker_seed = None


def worker_init_fn_diff(worker_id: int):
    """Initialize the worker with a different seed for each worker."""
    global worker_seed
    seed = int(time.time()) ^ (worker_id + os.getpid())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


worker_init_fns = {
    'diff': worker_init_fn_diff,
    'same': None
}


def standard_data_loader(cfg: Configuration1, worker_init='diff') -> DataLoader:
    xarray_dataset_iter = fragment_datasets.XarrayDatasetIter(cfg)
    datasets = fragment_datasets.TorchDatasetIter(cfg, xarray_dataset_iter)
    datasets = ConcatDataset(datasets)
    return DataLoader(datasets,
                      batch_size=cfg.batch_size,
                      num_workers=cfg.num_workers,
                      shuffle=cfg.shuffle,
                      worker_init_fn=worker_init_fns[worker_init],
                      collate_fn=cfg.collate_fn)


def cached_data_loader(cfg: Configuration1, reset_cache: bool = False, test_data=False, worker_init='diff') -> Dataset:
    cache_dir = os.path.join(cfg.prefix, f'data_cache_{cfg.suffix_cache}')
    zarr_dir = os.path.join(cache_dir, f'cache.zarr')

    if reset_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    if not os.path.isdir(zarr_dir):
        os.makedirs(cache_dir, exist_ok=True)

        train_loader = get_train_loader(cfg, worker_init=worker_init)
        saver = SaveModel(cache_dir)

        # Save the output of the data loader to a zarr file

        total = cfg.training_steps // cfg.batch_size
        running_sample_len = 0
        datapoint: Datapoint
        for i, datapoint in tqdm(enumerate(train_loader),
                                 total=total, disable=False, desc='Caching data', position=1, leave=False):
            if datapoint is None:
                continue

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
            for k in ['fragment', 'x_start', 'x_stop', 'y_start', 'y_stop', 'z_start', 'z_stop', 'fxy_idx']:
                ds[k] = ds[k].squeeze('empty', drop=True)
            ds = ds.assign_coords(fragment=ds.fragment,
                                  x_start=ds.x_start,
                                  x_stop=ds.x_stop,
                                  y_start=ds.y_start,
                                  y_stop=ds.y_stop,
                                  z_start=ds.z_start,
                                  z_stop=ds.z_stop,
                                  fxy_idx=ds.fxy_idx)
            dataset_to_zarr(ds, zarr_dir, 'sample')
            running_sample_len += sub_volume_len
        saver.config(cfg)

    ds = get_dataset(zarr_dir, fragment=cfg.test_box_fragment, hold_back_box=cfg.test_box, test_data=test_data)
    return CachedDataset(ds, cfg.batch_size, group_pixels=cfg.group_pixels)


def get_train_loader(cfg: Configuration1,
                     cached=False,
                     reset_cache=False,
                     worker_init='diff',
                     test_data=False) -> Union[DataLoader, Dataset]:
    if reset_cache and not cached:
        raise ValueError("reset_cache can only be True if cached is also True")
    if cached:
        return cached_data_loader(cfg, reset_cache, test_data=test_data, worker_init=worker_init)
    else:
        return standard_data_loader(cfg, worker_init=worker_init)


@lru_cache(maxsize=1)
def get_test_dataset(test_box_fragment, num_workers, prefix, x_start, x_stop, y_start, y_stop):
    ds_test = read_dataset_from_zarr(test_box_fragment, num_workers, prefix)
    ds_test.attrs['fragment'] = test_box_fragment
    ds_test = ds_test.isel(x=slice(x_start, x_stop),
                           y=slice(y_start, y_stop))
    ds_test.load()
    ds_test['full_mask'] = ds_test['mask']
    return ds_test


def test_loader(cfg: Configuration1) -> DataLoader:
    """Hold back data test box for fragment 1"""
    ds_test = get_test_dataset(cfg.test_box_fragment, cfg.num_workers, cfg.prefix,
                               cfg.test_box[0], cfg.test_box[2], cfg.test_box[1], cfg.test_box[3])
    test_dataset = cfg.volume_dataset_cls(ds_test,
                                          cfg.box_width_xy,
                                          cfg.box_width_z,
                                          max_iterations=cfg.training_steps,
                                          crop_cls=cfg.crop_box_cls,
                                          label_operation=cfg.label_fn,
                                          balance_ink=cfg.balance_ink,
                                          stride_xy=cfg.stride_xy,
                                          stride_z=cfg.stride_z)
    return DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)


def get_train_loader_regular_z(cfg: dataclass, force_cache_reset, test_data=False) -> DataLoader:
    def collate_catch_errs(batch):
        filtered_batch = []
        for item in batch:
            if isinstance(item, CustomDataLoaderError):
                continue
            else:
                filtered_batch.append(item)
        if filtered_batch:
            return default_collate(filtered_batch)
        else:
            return None

    cfg.collate_fn = collate_catch_errs
    return get_train_loader(cfg, cached=True, reset_cache=force_cache_reset, worker_init='same', test_data=test_data)


def get_train_loaders(cfg, epochs, cached_data, force_cache_reset, reset_cache_epoch_interval) \
        -> Generator[Datapoint, None, None]:
    for epoch in range(epochs):
        reset_cache = cached_data and (epoch != 0 and epoch % reset_cache_epoch_interval == 0)
        reset_cache = reset_cache or (epoch == 0 and force_cache_reset)
        yield from get_train_loader(cfg, cached=cached_data, reset_cache=reset_cache)
