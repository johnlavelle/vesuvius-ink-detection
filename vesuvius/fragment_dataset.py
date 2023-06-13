from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Type, Union
import random

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from xarray import DataArray

from vesuvius.datapoints import Datapoint, DatapointTuple
from vesuvius.sampler import BaseCropBox, CropBoxSobol, CropBoxRegular, VolumeSamplerRndXYZ, VolumeSamplerRegularZ, \
    BaseVolumeSampler
from vesuvius.utils import CustomDataLoaderError


class BaseDataset(ABC, Dataset):

    def __init__(self,
                 dataset: Any,
                 box_width_xy: int,
                 box_width_z: int,
                 max_iterations: int = None,
                 label_operation: Callable[[DataArray], float] = lambda x: x.mean(),
                 transformer: Callable[[torch.Tensor], DataArray] = None,
                 crop_cls: Type[Union[BaseCropBox, CropBoxSobol, CropBoxRegular]] = CropBoxSobol,
                 balance_ink: bool = False,
                 stride_xy: int = None,
                 stride_z: int = None,
                 seed: int = 42):
        self.ds = dataset
        self.box_width_xy = box_width_xy
        self.box_width_z = box_width_z
        self.max_iterations = max_iterations
        self.label_operation = label_operation
        self.transformer = transformer
        self.crop_box_cls = crop_cls
        self.balance_ink = balance_ink
        self._indexes = set()
        self.stride_xy = stride_xy
        self.stride_z = stride_z

        self.sampler = None
        volume_sampler_dict = {CropBoxSobol: VolumeSamplerRndXYZ,
                               CropBoxRegular: VolumeSamplerRegularZ}
        self.volume_sampler_cls = volume_sampler_dict[self.crop_box_cls]
        self.seed = seed  # this sets self.sampler

    @abstractmethod
    def get_datapoint(self, index: int) -> Datapoint:
        ...

    @lru_cache(maxsize=None)
    def get_sampler(self, seed) -> BaseVolumeSampler:
        return self.volume_sampler_cls(self.ds,
                                       self.box_width_xy,
                                       self.box_width_z,
                                       self.max_iterations,
                                       balance=self.balance_ink,
                                       crop_cls=self.crop_box_cls,
                                       stride_xy=self.stride_xy,
                                       stride_z=self.stride_z,
                                       seed=seed)

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value) -> None:
        self._seed = value
        self.sampler = self.get_sampler(self._seed)

    def nbytes(self) -> int:
        """Total number of bytes in the dataset"""
        return self.ds.nbytes

    # @functools.lru_cache(maxsize=None)
    def get_volume_slice(self, idx: int) -> Any:
        self._indexes.add(idx)
        return self.sampler[idx]

    def __getitem__(self, index: int) -> Union[Datapoint, CustomDataLoaderError]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.seed = worker_info.seed
        try:
            datapoint = self.get_datapoint(index)
            return datapoint
        except IndexError as err:
            return CustomDataLoaderError(str(err))

    def __len__(self) -> int:
        if self.max_iterations is None:
            return len(self.sampler)
        else:
            return self.max_iterations

    # def __len__(self) -> int:
    #     return len(self.sampler)

# class TestVolumeDataset(BaseDataset):
#     """
#     Get the test dataset.
#     """
#
#     def __init__(self,
#                  dataset: Any,
#                  box_width_sample: int,
#                  transformer: Callable[[DataArray], DataArray] = None,
#                  test_box: Tuple[float, float, float, float] = (0, 0, 100, 100),
#                  z_limits: Tuple[int, int] = (0, 9),
#                  label_operation: Callable[[DataArray], DataArray] = lambda x: x.mean(dim='z')):
#         self.transformer = transformer
#         self.label_operation = label_operation
#         self.box_width_sample = box_width_sample
#         slice_dict = dict(x=slice(test_box[0], test_box[2]),
#                           y=slice(test_box[1], test_box[3]), z=slice(*z_limits))
#         self.dataset = dataset.sel(**slice_dict)
#         self.dataset.load()
#         self.dataset = self.dataset
#         self.dataset = self.dataset[['images', 'labels']]
#         self.dataset['labels'] = self.dataset['labels'].astype(int)
#         roll_images = self.get_rolling(self.dataset.images).transpose('sample', 'z', 'x_win', 'y_win')
#         roll_labels = self.get_rolling(self.dataset.labels).transpose('sample', 'x_win', 'y_win')
#         self.ds_test_roll_sample = xr.merge([roll_images, roll_labels]).dropna(dim='sample', how='any')
#
#     def get_datapoint(self, index: int):
#         ds = self.ds_test_roll_sample.isel(sample=index)
#         ds['x'], ds['y'] = ds['x_win'], ds['y_win']
#         ds = ds.swap_dims(x_win='x', y_win='y')
#         return ds.images.transpose('z', 'x', 'y'), self.label_operation(ds.labels.transpose('x', 'y'))
#
#     def get_rolling(self, ds: xr.Dataset):
#         ds_roll = ds.rolling({'x': self.box_width_sample, 'y': self.box_width_sample}, center=True)
#         ds_roll = ds_roll.construct(x='x_win', y='y_win', stride=self.box_width_sample // 4)
#         return ds_roll.stack(sample=('x', 'y'))
#
#     def __len__(self):
#         return len(self.ds_test_roll_sample.sample)
#
#     def get_sampler(self, batch_size):
#         raise NotImplementedError
#
#     def nbytes(self):
#         """Total number of bytes in the dataset"""
#         raise NotImplementedError
#
#     @property
#     def seed(self):
#         raise NotImplementedError
#
#     @seed.setter
#     def seed(self, value):
#         raise NotImplementedError
#
#     def get_volume_slice(self, idx: int) -> Any:
#         raise NotImplementedError


# def unique_values_and_counts(arr):
#     """Returns unique values and their counts from an input list, along with the highest count"""
#     counter = Counter(arr)
#     return list(zip(*counter.items())), max(counter.values())


# class CachedDataset(Dataset):
#     def __init__(self, dataset: xr.Dataset, transformers=None, group_size=32, group_pixels=False):
#         self.transformers = transformers
#         self.hash_mappings = None
#         self.ds = dataset
#         self.group_size = group_size
#         self.length = self.ds.dims['sample'] // group_size  # number of groups
#         np.random.seed(0)
#
#         # Pre-divide the dataset into non-overlapping subsets
#         # Exclude the remainder of the division by group_size to have evenly sized groups
#         indices = np.arange(self.length * group_size)
#         np.random.shuffle(indices)
#         self.groups = np.array_split(indices, self.length)
#         assert all(len(g) == self.group_size for g in self.groups)
#
#     def __getitem__(self, index: int) -> DatapointTuple:
#         # Select the corresponding subset
#         ds = self.ds.isel(sample=self.groups[index])
#         dp = Datapoint(ds['voxels'],
#                        ds['label'],
#                        ds['fragment'],
#                        ds['x_start'],
#                        ds['x_stop'],
#                        ds['y_start'],
#                        ds['y_stop'],
#                        ds['z_start'],
#                        ds['z_stop']).to_namedtuple()
#
#         if self.transformers:
#             dp = dp._replace(voxels=self.transformers(dp.voxels))
#         return dp
#
#     def __len__(self):
#         return self.length


# class CachedDataset(Dataset):
#     def __init__(self, dataset: xr.Dataset, transformers=None, group_size=32, use_cache=False):
#         self.transformers = transformers
#         self.group_size = group_size
#         self.ds = dataset
#         self.use_cache = use_cache
#
#         if self.use_cache:
#             self.grp_all = self.get_group(dataset, group_size)
#             self.sub_grp_size = 2**13
#             self.reuse_counter = 0
#             self.ds_sub = self.get_random_subset(self.ds)
#             self.ds_grp = self.get_group(self.ds_sub, self.group_size)
#         else:
#             sample = self.ds['sample'].values
#             np.random.shuffle(sample)
#             self.ds['sample'] = ('sample', sample)
#             self.ds = self.ds.isel(sample=slice(0, len(self.ds.sample) - len(self.ds.sample) % group_size))
#             groups = xr.DataArray(np.arange(len(self.ds.sample)) // group_size, dims='sample')
#             self.ds = self.ds.assign_coords(group=groups)
#             self.ds_grp = self.ds.groupby('group')
#             self.indexes = list(range(len(self.ds_grp)))
#             random.shuffle(self.indexes)
#
#     def get_random_subset(self, ds) -> xr.Dataset:
#         sample = ds['sample'].values
#         np.random.shuffle(sample)
#         ds['sample'] = ('sample', sample)
#         ds = ds.sortby('sample')
#         ds = ds.isel(sample=slice(0, self.sub_grp_size))
#         return ds.load()
#
#     @staticmethod
#     def get_group(ds, group_size):
#         groups = xr.DataArray(np.arange(len(ds.sample)) // group_size, dims='sample')
#         ds = ds.assign_coords(group=groups)
#         return ds.groupby('group')
#
#     def __getitem__(self, index: int) -> DatapointTuple:
#         if self.use_cache:
#             if index % (self.sub_grp_size // self.group_size) == 0:
#                 self.reuse_counter += 1
#                 if self.reuse_counter >= 10:
#                     self.reuse_counter = 0
#                     del self.ds_sub
#                     del self.ds_grp
#                     self.ds_sub = self.get_random_subset(self.ds)
#                     self.ds_grp = self.get_group(self.ds_sub, self.group_size)
#             ds = self.ds_grp[index % len(self.ds_grp)]  # Use shuffled index
#         else:
#             if index >= self.__len__():
#                 raise IndexError("Dataset exhausted")
#             ds = self.ds_grp[self.indexes[index]]  # Use shuffled index
#
#         dp = Datapoint(ds['voxels'],
#                        ds['label'],
#                        ds['fragment'],
#                        ds['x_start'],
#                        ds['x_stop'],
#                        ds['y_start'],
#                        ds['y_stop'],
#                        ds['z_start'],
#                        ds['z_stop']).to_namedtuple()
#
#         if self.transformers:
#             dp = dp._replace(voxels=self.transformers(dp.voxels))
#         return dp
#
#     def __len__(self):
#         return len(self.grp_all) if self.use_cache else len(self.ds_grp)


def gaussian(x, mu, sigma):
    y = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y / np.sum(y)


class DatasetReader:
    def __init__(self, ds, chunk_size):
        self.ds = ds
        self.chunk_size = chunk_size
        self.num_samples = len(ds.sample)
        self.num_chunks = np.ceil(self.num_samples / self.chunk_size).astype(int)
        self.current_chunk_idx = 0
        self.current_chunk = None
        self.__getitem__(0)

    def _get_chunk(self, chunk_idx):
        start = chunk_idx * self.chunk_size
        end = min((chunk_idx + 1) * self.chunk_size, self.num_samples)
        return self.ds.sel(sample=slice(start, end)).load()

    def __getitem__(self, sample_idx):
        chunk_idx = sample_idx // self.chunk_size
        if self.current_chunk is None or chunk_idx != self.current_chunk_idx:
            del self.current_chunk
            self.current_chunk = self._get_chunk(chunk_idx)
            self.current_chunk_idx = chunk_idx
        return self.current_chunk.sel(sample=sample_idx)

    def get_random_samples(self, number_of_samples):
        if self.current_chunk is None:
            raise ValueError("No chunk is currently loaded.")
        sample_indices = random.sample(range(len(self.current_chunk.sample)), number_of_samples)
        return self.current_chunk.isel(sample=sample_indices)


class CachedDataset(Dataset):
    def __init__(self, dataset: xr.Dataset, transformers=None, group_size=32, in_memory=True):
        self.transformers = transformers
        self.group_size = group_size

        # dataset = dataset.isel(sample=slice(0, None, 2)).load()

        self.ds_all = dataset
        self.ds_all['sample'] = np.arange(len(self.ds_all['sample']))

        self.randomise_ds()
        self.ds_reader = DatasetReader(self.ds_all, 256)

        self.in_memory = in_memory
        self.ds_grp = None

        # if in_memory:
        #     self.sigma = 1
        # else:
        #     # self.ds = dataset.sortby('sample')
        #     self.ds = self.ds_all.isel(sample=slice(0, len(self.ds_all.sample) - len(self.ds_all.sample) % group_size))

    def randomise_ds(self):
        samples = np.arange(len(self.ds_all['sample']))
        np.random.shuffle(samples)
        # Reassign the shuffled coordinates to the 'sample' dimension
        self.ds_all = self.ds_all.assign_coords(sample=samples)
        self.ds_all = self.ds_all.sortby('sample')

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        if self.in_memory:
            try:
                del self.ds
            except AttributeError:
                pass
            self.ds = self.get_random_subset(self.ds_all)

    def get_random_subset(self, ds) -> xr.Dataset:
        choices = np.random.choice(ds.sample.values, size=2 ** 13,
                                   p=gaussian(ds.z_start.values, 32, self._sigma))
        ds = self.ds_all.isel(sample=choices).compute()

        return ds

    def __getitem__(self, index: int) -> DatapointTuple:
        if index == self.__len__() - 1:
            self.randomise_ds()
        if index >= self.__len__():
            raise IndexError("Dataset exhausted")
        # choices = np.random.choice(np.arange(len(self.ds.sample.values)), size=self.group_size, replace=False)
        # ds = self.ds.isel(sample=choices)
        ds = self.ds_reader.get_random_samples(self.group_size)

        dp = Datapoint(ds['voxels'],
                       ds['label'],
                       ds['fragment'],
                       ds['x_start'],
                       ds['x_stop'],
                       ds['y_start'],
                       ds['y_stop'],
                       ds['z_start'],
                       ds['z_stop']).to_namedtuple()

        if self.transformers:
            dp = dp._replace(voxels=self.transformers(dp.voxels))
        return dp

    def __len__(self):
        return len(self.ds_all.sample) // self.group_size
