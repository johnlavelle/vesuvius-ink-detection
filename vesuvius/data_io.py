import ctypes
import gc
import glob
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from functools import lru_cache
from os.path import join
from pathlib import Path
from typing import Dict, Union, Literal, Iterable
from typing import Tuple

import dask
import dask.multiprocessing
import numpy as np
import psutil
import torch.nn as nn
import xarray as xr
import zarr
from dask.distributed import Client, LocalCluster
from rasterio.errors import RasterioIOError
from tqdm import tqdm
from zarr.sync import ProcessSynchronizer

from vesuvius.ann.transforms import *
from vesuvius.config import Configuration


# from utils import voxel_stats


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def memory_usage():
    memory = psutil.virtual_memory()
    print('Total Memory     :', round(memory.total / (1024.0 ** 3), 2), 'GB')
    print('Available Memory :', round(memory.available / (1024.0 ** 3), 2), 'GB')
    print('Used Memory      :', round(memory.used / (1024.0 ** 3), 2), 'GB')
    print('Memory Percentage:', memory.percent, '%')
    return memory.available


def extract_number(path):
    filename = os.path.basename(path)
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return None


def read_tiffs(fragment: int, prefix: str) -> Iterable:
    fragment = str(fragment)

    tiff_fnames = sorted(glob.glob(join(prefix, fragment, 'surface_volume/*.tif')))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = xr.open_dataset(join(prefix, fragment, "mask.png")).squeeze()['band_data']
    mask = mask.astype(bool).load()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels_ink = xr.open_dataset(join(prefix, fragment, "inklabels.png")).squeeze()['band_data']
        labels_ink = labels_ink.astype(bool).load()
    except RasterioIOError:
        labels_ink = None

    for f in tiff_fnames:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = xr.open_dataset(f).transpose('y', 'x', 'band')

        ds['z'] = [extract_number(f)]

        ds = ds.rename({'band_data': 'images'})
        if labels_ink:
            ds['labels'] = labels_ink
        ds['mask'] = mask
        ds.mask.attrs = ''

        del ds['spatial_ref']
        for v in ('x', 'y'):
            ds[v] = np.arange(len(ds[v]))
        del ds['band']
        ds = ds.rename(band='z')
        yield ds


def encodings(variables) -> Dict:
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

    # cd /data/kaggle/input/vesuvius-challenge-ink-detection
    # rm -r */surface_volume.zarr  */data_cache_regular
    zarr_path = join(prefix, str(fragment), 'surface_volume.zarr')

    available_memory = memory_usage()
    available_memory -= 1024 ** 3 / 4
    cluster_memory = str(available_memory / (2 * (1024 ** 3))) + 'GB'
    print(f'Cluster memory: {cluster_memory}')

    tmp_dir = '/data/tmp/'
    if os.path.exists(tmp_dir):
        dask.config.set({'temporary_directory': tmp_dir})
    # cluster = LocalCluster(processes=True, n_workers=5, threads_per_worker=10, memory_limit=cluster_memory)
    with Client() as client:
        print(client.dashboard_link)

        for dataset in tqdm(read_tiffs(fragment, prefix), total=65, postfix=f'Fragment {fragment} to Zarr'):
            z = dataset.z.item()

            dataset = dataset.load()
            dataset['images'] = dataset['images'].where(dataset.mask)

            # Normalise w.r.t z
            mean = dataset['images'].mean().compute()
            std = dataset['images'].std().compute()
            dataset['images'] = (dataset['images'] - mean) / std
            dataset['images'].attrs['ds_z_mean'] = mean.values
            dataset['images'].attrs['ds_z_std'] = std.values

            mode: Literal["w", "w-", "a", "r+", None]
            if z == 0:
                mode = 'w-'
                append_dim = None
                encodings_ = encodings(['images'])
            else:
                mode = 'a'
                append_dim = 'z'
                encodings_ = None

            dataset_images = dataset[['images']]
            dataset_images = dataset_images.chunk({'x': 128, 'y': 128})
            dataset_images = client.scatter(dataset_images)
            dataset_images = dask.persist(dataset_images)[0]
            dataset_images.result().to_zarr(zarr_path,
                                            mode=mode,
                                            encoding=encodings_,
                                            consolidated=True,
                                            compute=True,
                                            append_dim=append_dim)

            client.run(trim_memory)

        if not ('labels' in dataset):
            dataset['labels'] = xr.zeros_like(dataset['mask'])
        for v in tqdm(['labels', 'mask'], total=2, postfix=f'Writing labels and mask'):
            print(v)
            encodings_ = encodings([v])
            ds = dataset[[v]].astype(bool)
            ds = ds.chunk({'x': -1, 'y': -1})
            ds = client.scatter(ds)
            ds = dask.persist(ds)[0]
            ds.result().to_zarr(zarr_path, mode='a', encoding=encodings_, consolidated=True, compute=True)

            client.run(trim_memory)

    # cluster.close()
    client.close()

    return zarr_path


def rechunk_dataset(ds: xr.Dataset, zarr_path):
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
        rechunk_dataset(ds, zarr_path)


def read_dataset_from_zarr(fragment: Union[int, str], workers: int, prefix: str, normalize: bool = True) -> xr.Dataset:
    zarr_path = join(prefix, str(fragment), 'surface_volume.zarr')
    if not os.path.exists(zarr_path):
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
        rechunk_dataset(ds, zarr_path)

    return xr.open_zarr(zarr_path, consolidated=True)


@lru_cache(maxsize=2)
def open_dataset(zarr_path: str):
    with dask.config.set(scheduler='synchronous'):
        ds = xr.open_zarr(zarr_path, consolidated=True)
        ds = rechunk_cached(ds)
        overhead_gb = (get_available_memory() - ds.nbytes) / (1024 * 1024 * 1024)
        if overhead_gb > 10:
            print('Loading dataset into memory...', '\n')
            ds.load()
        else:
            pass
            # print('Loading coordinates into memory...', '\n')
            # ds[ds.coords.keys()].load()
        return ds


@lru_cache(maxsize=2)
def get_dataset(zarr_path: str, fragment: Union[int, str], hold_back_box: Tuple[int, int, int, int], test_data=False):
    with dask.config.set(scheduler='synchronous'):
        ds = open_dataset(zarr_path)
        hold_back_bools = get_hold_back_bools(ds, fragment, hold_back_box)
        if test_data:
            ds = ds.isel(sample=hold_back_bools)
        else:
            ds = ds.isel(sample=~hold_back_bools)
        return ds
