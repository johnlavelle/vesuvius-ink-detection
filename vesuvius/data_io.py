import gc
import glob
import json
import os
import time
import tempfile
from pathlib import Path
import itertools


import shutil
import warnings
from dataclasses import asdict
from functools import lru_cache
from os.path import join
from typing import Any, Tuple
from typing import Dict, Union

import psutil

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import dask
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import zarr
from tqdm import tqdm
from zarr.sync import ProcessSynchronizer

from vesuvius.utils import normalise_images
from vesuvius.config import Configuration


def read_tiffs(fragment: int, prefix: str) -> xr.Dataset:
    fragment = str(fragment)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tiff_fnames = sorted(glob.glob(join(prefix, fragment, 'surface_volume/*.tif')))
        ds = xr.open_mfdataset(tiff_fnames, concat_dim='band', combine='nested', parallel=True).transpose('y', 'x',
                                                                                                          'band')
        ds = ds.rename({'band_data': 'images'})
        labels_ink = xr.open_dataset(join(prefix, fragment, "inklabels.png")).squeeze()
        mask = xr.open_dataset(join(prefix, fragment, "mask.png")).squeeze()
        ds['labels'] = labels_ink['band_data'].astype(bool)
        ds['mask'] = mask['band_data'].astype(bool)
        ds.mask.attrs = ''
        ds = ds.rename({'band': 'z'})
        del ds['spatial_ref']
        for v in ('x', 'y', 'z'):
            ds[v] = np.arange(len(ds[v]))
        return ds


def encodings(variables) -> Dict[int, Dict[str, Any]]:
    compressor = {'compressor': zarr.Blosc(cname='zstd', clevel=5, shuffle=2)}
    return {k: compressor for k in variables}


def dataset_to_zarr(dataset: xr.Dataset, zarr_path: str, append_dim: str) -> None:
    mode: Literal["w", "w-", "a", "r+", None]
    if os.path.exists(zarr_path):
        mode, append_dim, encodings_ = 'a', append_dim, None
    else:
        mode, append_dim, encodings_ = 'w-', None, encodings(['voxels', 'label'])

    dataset['voxels'] = dataset['voxels'].chunk(dataset['voxels'].shape)
    dataset['label'] = dataset['label'].chunk(dataset['label'].shape)
    try:
        dataset.to_zarr(zarr_path, mode=mode, encoding=encodings_, consolidated=True, compute=True,
                        append_dim=append_dim)
    except ValueError:
        print('Skipping', zarr_path)
        raise


def save_zarr(fragment: int, prefix: str, normalize=True) -> str:
    """Write a dataset to a zarr file"""
    dataset = read_tiffs(fragment, prefix).chunk({'z': 1})
    zarr_path = join(prefix, str(fragment), 'surface_volume.zarr')

    # Append one z layer each step, to reduce the memory overhead
    ds_images = dataset['images'].where(dataset.mask).to_dataset(name='images')
    ds_images = normalise_images(ds_images) if normalize else ds_images

    z_chunk_size = 1
    for z in tqdm(range(0, len(ds_images.z.values), z_chunk_size), desc=f'Writing fragment {fragment} to zarr',
                  position=1):
        ds_sub = ds_images.isel(z=slice(z, z + z_chunk_size)).chunk({'x': 128, 'y': 128, 'z': z_chunk_size})

        mode: Literal["w", "w-", "a", "r+", None]
        if z == 0:
            mode, append_dim, encodings_ = 'w-', None, encodings(['images'])
        else:
            mode, append_dim, encodings_ = 'a', 'z', None

        ds_sub.to_zarr(zarr_path, mode=mode, encoding=encodings_, consolidated=True, compute=True,
                       append_dim=append_dim)
        gc.collect()

    # Write the labels and mask
    for v in ['labels', 'mask']:
        ds2d = dataset[[v]].astype(bool)
        encodings_ = encodings([v])
        ds2d.to_zarr(zarr_path, mode='a', encoding=encodings_, consolidated=True, compute=True)

    # dataset_reload = read_dataset_from_zarr(fragment, 0, prefix, normalize=False)
    # dataset_reload = dataset_reload.chunk({'x': 128, 'y': 128, 'z': 65})
    # dataset_reload.to_zarr(zarr_path, encoding=encodings(('labels', 'mask', 'images')))
    return zarr_path


def rechunk(ds: xr.Dataset, zarr_path):
    print('Rechunking dataset ...')
    zarr_path = Path(zarr_path)
    with tempfile.TemporaryDirectory(dir=zarr_path.parent) as temp_dir:
        temp_zarr_path = os.path.join(temp_dir, "temp.zarr")
        ds.to_zarr(temp_zarr_path, mode='w', consolidated=True)

        # Remove the original Zarr file and replace it with the temporary Zarr file
        shutil.rmtree(zarr_path)
        shutil.move(temp_zarr_path, zarr_path)
    print('... finished rechunking', '\n')


def rechunk_org(ds, zarr_path):
    preferred_chunks = tuple(ds['images'].encoding['preferred_chunks'].values())
    chunks = ds['images'].encoding['chunks']
    if preferred_chunks != chunks:
        rechunk(ds, zarr_path)


def read_dataset_from_zarr(fragment: Union[int, str], workers: int, prefix: str, normalize: bool = True) -> xr.Dataset:
    zarr_path = join(prefix, str(fragment), 'surface_volume.zarr')
    if not os.path.exists(zarr_path):
        with dask.config.set(scheduler='processes', num_workers=workers):
            save_zarr(fragment, prefix, normalize)
    sync = ProcessSynchronizer('/tmp/tmpz7x9z5xh')
    with dask.config.set(scheduler='synchronous'):
        dataset = xr.open_zarr(zarr_path,
                               consolidated=True,
                               synchronizer=sync,  # ThreadSynchronizer(),
                               chunks={'z': 65},
                               overwrite_encoded_chunks=True)
        dataset = dataset.unify_chunks()
        dataset.mask.load()
        dataset.labels.load()
        return dataset


# Models
class SaveModel:
    def __init__(self, path):

        self.path = path
        self.name = ''
        self.conf_path = join(self.path, f"config{self.name}.json")
        os.makedirs(self.path, exist_ok=True)

    def model(self, torch_model: nn.Module, name: Union[str, int] = '') -> str:
        self.name = str(name)
        model_path = join(self.path, f"model{self.name}.pt")
        torch.save(torch_model.state_dict(), model_path)
        return model_path

    def config(self, config: Configuration) -> str:
        config_dict = config.as_dict()
        with open(self.conf_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)
            return self.conf_path


class LoadModel:
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path
        self._config = self._load_config()

    def model(self) -> nn.Module:
        _model = self._config.mode0()
        _model.load_state_dict(torch.load(self.model_path))
        return self._config.mode0()

    def _load_config(self) -> Configuration:
        with open(self.config_path, "r") as json_file:
            config_dict = json.load(json_file)
        return Configuration(**config_dict)

    def config(self) -> Configuration:
        return self._config


def get_available_memory() -> int:
    mem = psutil.virtual_memory()
    return mem.available


def get_hold_back_bools(ds: xr.Dataset, fragment, hold_back_box: Tuple[int, int, int, int]) -> xr.DataArray:
    xl, yl, xu, yu = hold_back_box
    hold_back_bool = ((ds.fragment == fragment) &
                      (ds.x_start >= xl) &
                      (ds.x_stop <= xu) &
                      (ds.y_start >= yl) &
                      (ds.y_stop <= yu)).compute()
    return hold_back_bool


def rechunk_cached(ds: xr.Dataset) -> xr.Dataset:
    # Set the desired chunk size for each variable and coordinate along the sample dimension
    variables = list(ds.coords)
    variables.append('label')
    desired_chunk_sizes = {key: {'sample': len(ds.sample)} for key in variables}
    desired_chunk_sizes['voxels'] = {'sample': 16}
    del desired_chunk_sizes['sample']

    # Check if the dataset already has the desired chunk sizes
    needs_rechunking = False
    for var, chunks in desired_chunk_sizes.items():
        if ds[var].chunks[0][0] != chunks['sample']:
            needs_rechunking = True
            break

    zarr_path = ds.encoding["source"]

    # Rechunk and save the dataset, replacing the existing Zarr file if needed
    if needs_rechunking:
        for var, chunks in desired_chunk_sizes.items():
            if var in ds:
                ds[var] = ds[var].chunk(chunks)

        # Remove the 'chunks' encoding from the variables
        for var in itertools.chain(ds.data_vars, ds.coords):
            del ds[var].encoding['chunks']
            del ds[var].encoding['preferred_chunks']

        # Save the rechunked dataset to a temporary Zarr file
        rechunk(ds, zarr_path)

    return xr.open_zarr(zarr_path, consolidated=True)


@lru_cache(maxsize=1)
def open_dataset(zarr_path: str):
    with dask.config.set(scheduler='synchronous'):
        ds = xr.open_zarr(zarr_path, consolidated=True)
        ds = rechunk_cached(ds)
        overhead_gb = (get_available_memory() - ds.nbytes) / (1024 * 1024 * 1024)
        if overhead_gb > 4:
            print('Loading dataset into memory...', '\n')
            ds.load()
        else:
            print('Loading coordinates into memory...', '\n')
            ds[ds.coords.keys()].load()
        return ds


def get_dataset(zarr_path: str, fragment: Union[int, str], hold_back_box: Tuple[int, int, int, int], test_data=False):
    with dask.config.set(scheduler='synchronous'):
        ds = open_dataset(zarr_path)
        if test_data:
            hold_back_bools = get_hold_back_bools(ds, fragment, hold_back_box)
            ds = ds.isel(sample=hold_back_bools)
        return ds
