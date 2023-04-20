import gc
import glob
import json
import os
import warnings
from dataclasses import asdict
from os.path import join

from typing import Any
from typing import Dict, Union

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
from vesuvius.config import Configuration1, serialize


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


# Models
class SaveModel:
    def __init__(self, path, name: Union[str, int] = ''):
        self.name = str(name)
        if self.name:
            self.name = '_' + self.name
        self.path = path
        self.conf_path = join(self.path, f"config{self.name}.json")
        self.model_path = join(self.path, f"model{self.name}.pt")
        os.makedirs(self.path, exist_ok=True)

    def model(self, torch_model: nn.Module) -> str:
        torch.save(torch_model.state_dict(), self.model_path)
        return self.model_path

    def config(self, config: Configuration1) -> str:
        config_dict = asdict(config, dict_factory=lambda obj: {k: serialize(v) for k, v in obj})
        with open(self.conf_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)
            return self.conf_path


class LoadModel:
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path
        self._config = self._load_config()

    def model(self) -> nn.Module:
        _model = self._config.model()
        _model.load_state_dict(torch.load(self.model_path))
        return self._config.model()

    def _load_config(self) -> Configuration1:
        with open(self.config_path, "r") as json_file:
            config_dict = json.load(json_file)
        return Configuration1(**config_dict)

    def config(self) -> Configuration1:
        return self._config
