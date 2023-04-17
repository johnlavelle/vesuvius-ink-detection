from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, Type, Union

import dask
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from xarray import DataArray

from vesuvius.datapoints import Datapoint, DatapointTuple
from vesuvius.sampler import BaseCropBox, CropBoxSobol, CropBoxRegular, VolumeSamplerRndXYZ, VolumeSamplerRegularZ


class BaseDataset(ABC, Dataset):

    def __init__(self,
                 dataset: Any,
                 box_width_xy: int,
                 box_width_z: int,
                 max_iterations: int = 10_000,
                 label_operation: Callable[[DataArray], float] = lambda x: x.mean(),
                 transformer: Callable[[torch.Tensor], DataArray] = None,
                 crop_cls: Type[Union[BaseCropBox, CropBoxSobol, CropBoxRegular]] = CropBoxSobol,
                 balance_ink: bool = False,
                 seed: int = 42,
                 stride_xy: int = None,
                 stride_z: int = None):
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
    def get_sampler(self, seed) -> VolumeSamplerRndXYZ:
        return self.volume_sampler_cls(self.ds,
                                       self.box_width_xy,
                                       self.box_width_z,
                                       self.max_iterations,
                                       balance=self.balance_ink,
                                       crop_cls=self.crop_box_cls,
                                       seed=seed,
                                       stride_xy=self.stride_xy,
                                       stride_z=self.stride_z)

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

    def __getitem__(self, index: int) -> Union[Datapoint, IndexError]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.seed = worker_info.seed

        try:
            datapoint = self.get_datapoint(index)
            return datapoint
        except IndexError as err:
            return err

    def __len__(self) -> int:
        if self.max_iterations is None:
            return len(self.sampler)
        else:
            return self.max_iterations


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


class CachedDataset(Dataset):
    def __init__(self, zarr_path: str, group_size=32, group_pixels=False):
        with dask.config.set(scheduler='synchronous'), xr.open_zarr(zarr_path) as self.ds:
            self.ds = self.ds.chunk({'sample': group_size})
            if group_pixels:
                self.ds = self.ds.sortby('fxy_idx')
                index_set = np.unique(self.ds.fxy_idx)
                np.random.shuffle(index_set)
                self.hash_mappings = {i: v for i, v in enumerate(index_set)}
                self.ds_grp = self.ds.groupby('fxy_idx')
            else:
                self.ds_grp = self.ds.groupby(self.ds.sample // group_size)
                self.hash_mappings = None

    def idx_mapping(self, index: int) -> int:
        if self.hash_mappings:
            return self.hash_mappings[index]
        else:
            return index

    def __getitem__(self, index: int) -> DatapointTuple:
        try:
            ds = self.ds_grp[self.idx_mapping(index)]
        except KeyError:
            raise IndexError

        return Datapoint(ds['voxels'],
                         ds['label'],
                         ds['fragment'],
                         ds['x_start'],
                         ds['x_stop'],
                         ds['y_start'],
                         ds['y_stop'],
                         ds['z_start'],
                         ds['z_stop']).to_namedtuple()

    def __len__(self):
        return len(self.ds_grp)
