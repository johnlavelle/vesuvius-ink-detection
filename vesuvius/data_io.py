import gc
import glob
import os
import warnings
from os.path import join
from typing import Any
from typing import Dict, Union, Literal

import dask
import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm
from zarr.sync import ProcessSynchronizer

from vesuvius.utils import normalise_images


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
    dataset.to_zarr(zarr_path, mode=mode, encoding=encodings_, consolidated=True, compute=True, append_dim=append_dim)


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
