import random
import time
from abc import ABC, abstractmethod
from typing import Tuple, Union
import os
import shutil

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm
from vesuvius.data_io import dataset_to_zarr
from vesuvius.config import save_config
from vesuvius.dataset import CachedDataset

from vesuvius.config import Configuration
from vesuvius.data_io import read_dataset_from_zarr
from vesuvius.utils import get_mask
from vesuvius.dataset import BaseDataset

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


worker_seed = None


class WeightedSamples:
    """
    Generate balanced samples for each fragment, such that the nuber of pixels samples from each fragment is
    proportional to the number of ink pixels in the fragment.
    """

    def __init__(self, samples: int, prefix: str, fragment_keys: Tuple[int, ...] = (1, 2, 3), num_workers=0):
        """

        :param samples: The number of samples to generate
        :param fragment_keys:
        """
        self.fragment_keys = fragment_keys
        self.num_workers = num_workers
        self.prefix = prefix
        self.ink_sizes = None
        self.ink_sizes_total = None
        self._samples = samples
        self.ink_sizes = {i: self._get_num_of_ink_pixels_per_fragment(i) for i in self.fragment_keys}
        self.ink_sizes_total = sum(self.ink_sizes.values())

    def normalise(self, ink_sizes):
        return round(self._samples * ink_sizes / self.ink_sizes_total)

    def _get_num_of_ink_pixels_per_fragment(self, i):
        ds = read_dataset_from_zarr(i, self.num_workers, self.prefix)['labels']
        return int(ds.sum().values)


class DatasetIter(ABC):

    def __init__(self, config):
        self.config = config

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> xr.Dataset:
        pass


class XarrayDatasetIter(DatasetIter):

    def __init__(self, config: Configuration):
        super().__init__(config)
        self.fragments = iter(config.fragments)
        self.test_box_fragment = config.test_box_fragment

    def __next__(self) -> xr.Dataset:
        try:
            fragment = next(self.fragments)
        except StopIteration:
            raise StopIteration

        data = read_dataset_from_zarr(str(fragment), self.config.num_workers, self.config.prefix)
        data = data.sel(z=slice(*self.config.z_limit))

        if fragment == self.test_box_fragment:
            data['full_mask'] = get_mask(data, self.config.test_box)
        else:
            data['full_mask'] = data['mask']

        return data


class TorchDatasetIter(DatasetIter):
    """
    This class is used to iterate over the slices of a volume dataset.
    This done to reduce the memory footprint of the dataset.
    """

    def __init__(self, config: Configuration, datasets: DatasetIter):
        super().__init__(config)
        self.training_steps = config.training_steps
        self.box_width_sample = config.box_width_sample
        self.volume_dataset_cls = config.volume_dataset_cls
        self.label_operation = config.label_fn
        self.fragments = config.fragments
        self.prefix = config.prefix

        self.dataset_class = config.volume_dataset_cls
        self.z_start, self.z_end = config.z_limit
        self.crop_box_cls = config.crop_box_cls
        self.box_width_sample = config.box_width_sample

        self.label_operation = config.label_fn
        self.samples = config.training_steps

        self.datasets = datasets

        self.samples_handler = WeightedSamples(self.samples,
                                               self.prefix,
                                               self.fragments,
                                               num_workers=config.num_workers)
        self.current_ds = None
        self.balance_ink = config.balance_ink

    def __iter__(self):
        return self

    def __next__(self) -> BaseDataset:
        try:
            self.current_ds = next(self.datasets)
        except StopIteration:
            raise StopIteration

        ink_count = self.current_ds.labels.sum().compute().item()
        samples = self.samples_handler.normalise(ink_count)
        return self.dataset_class(self.current_ds,
                                  self.box_width_sample,
                                  samples,
                                  transformer=None,
                                  crop_box_cls=self.crop_box_cls,
                                  label_operation=self.label_operation,
                                  balance_ink=self.balance_ink)


def worker_init_fn(worker_id: int):
    global worker_seed
    seed = int(time.time()) ^ (worker_id + os.getpid())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standard_data_loader(configuration: Configuration) -> DataLoader:
    xarray_dataset_iter = XarrayDatasetIter(configuration)
    datasets = TorchDatasetIter(configuration, xarray_dataset_iter)
    return DataLoader(ConcatDataset(datasets),
                      batch_size=configuration.batch_size,
                      num_workers=configuration.num_workers,
                      prefetch_factor=4,
                      shuffle=True,
                      worker_init_fn=worker_init_fn,
                      collate_fn=configuration.collate_fn)


def cached_data_loader(configuration: Configuration, reset_cache: bool = False) -> Dataset:
    cache_dir = os.path.join(configuration.prefix, 'data_cache')
    zarr_dir = os.path.join(cache_dir, 'cache.zarr')

    if reset_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    if not os.path.isdir(zarr_dir):
        os.makedirs(cache_dir, exist_ok=True)

        train_loader = data_loader(configuration)

        # Save the output of the data loader to a zarr file

        total = configuration.training_steps/configuration.batch_size
        running_sample_len = 0
        for i, (samples, labels) in tqdm(enumerate(train_loader), total=total,
                                         desc='Caching data', position=1, leave=False):

            # Create a dataset with the samples and labels
            sample_len = samples.shape[0]
            sample_coord = np.arange(running_sample_len, running_sample_len + sample_len)
            coords = {'sample': sample_coord}
            samples = xr.DataArray(samples.numpy(), dims=('sample', 'empty', 'z', 'x', 'y'), coords=coords)
            labels = xr.DataArray(labels, dims=('sample', 'empty'), coords=coords)
            ds = xr.Dataset({'samples': samples, 'labels': labels})

            dataset_to_zarr(ds, zarr_dir, 'sample')
            save_config(configuration, os.path.join(cache_dir, f"config.json"))

            running_sample_len += sample_len
    return CachedDataset(zarr_dir, configuration.batch_size)


def data_loader(configuration: Configuration, cached=False, reset_cache=False) -> Union[DataLoader, Dataset]:
    if reset_cache and not cached:
        raise ValueError("reset_cache can only be True if cached is also True")
    if cached:
        return cached_data_loader(configuration, reset_cache)
    else:
        return standard_data_loader(configuration)
