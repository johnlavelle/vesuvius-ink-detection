from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple, Any, Callable, Optional

import dask
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from xarray import DataArray

from vesuvius.sampler import CropBoxInter, CropBoxSobol
from vesuvius.sampler import VolumeSampler

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class BaseDataset(ABC, Dataset):
    def __init__(self,
                 dataset: Any,
                 feature_box_width: int,
                 max_iterations: int,
                 label_operation: Callable[[DataArray], float] = lambda x: x.mean(),
                 transformer: Callable[[torch.Tensor], DataArray] = None,
                 crop_box_cls: Callable[[Tuple[int, int, int, int], int, Optional[int]], CropBoxInter] = CropBoxSobol,
                 balance_ink: bool = False,
                 seed: int = 42):
        self.ds = dataset
        self.feature_box_width = feature_box_width
        self.max_iterations = max_iterations
        self.label_operation = label_operation
        self.transformer = transformer
        self.crop_box_cls = crop_box_cls
        self.balance_ink = balance_ink
        self._indexes = set()
        self.seed = seed

    @staticmethod
    def to_tensor(da: DataArray) -> torch.Tensor:
        np_arr = da.values
        np_arr.setflags(write=False)
        return torch.from_numpy(np_arr.copy())

    @abstractmethod
    def get_item_as_data_array(self, index: int) -> Tuple[DataArray, DataArray]:
        ...

    @lru_cache
    def get_sampler(self, seed):
        return VolumeSampler(self.ds['full_mask'], self.ds['labels'], self.feature_box_width,
                             balance=self.balance_ink, crop_box_cls=self.crop_box_cls, seed=seed)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.seed = worker_info.seed
        try:
            voxels, labels = self.get_item_as_data_array(index)
        except ValueError:
            print(f"Error in {index}")
            return self.__getitem__(index)
        voxels = self.to_tensor(voxels.expand_dims('Cin'))
        if self.transformer:
            voxels = self.transformer(voxels)
        return voxels, self.to_tensor(labels).view(1)

    def nbytes(self):
        """Total number of bytes in the dataset"""
        return self.ds.nbytes

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.sampler = self.get_sampler(self._seed)

    # @functools.lru_cache(maxsize=None)
    def get_slice(self, idx: int) -> Any:
        self._indexes.add(idx)
        return next(self.sampler)

    def __len__(self) -> int:
        return self.max_iterations

    @staticmethod
    def normalise_voxels(voxels) -> DataArray:
        voxels_mean_z = voxels.mean(dim=['x', 'y'], skipna=True)
        voxels_std_z = voxels.std(dim=['x', 'y'], skipna=True)
        eps = 1e-6  # set a small non-zero value
        voxels_std_z = voxels_std_z.where(voxels_std_z != 0, eps)
        return (voxels - voxels_mean_z) / voxels_std_z

    def postprocess_data_array(self, voxels, ink_labels):
        labels = self.label_operation(ink_labels)
        try:
            assert np.isfinite(voxels).all().values.item(), 'Non finite number in voxels'
            assert labels in (0, 1), 'Label is not 0 or 1'
        except AssertionError as err:
            raise ValueError(err)
        return voxels, labels


class TestVolumeDataset(BaseDataset):
    """
    Get the test dataset.
    """
    def __init__(self,
                 dataset: Any,
                 box_width_sample: int,
                 transformer: Callable[[DataArray], DataArray] = None,
                 test_box: Tuple[float, float, float, float] = (0, 0, 100, 100),
                 z_limits: Tuple[int, int] = (0, 9),
                 label_operation: Callable[[DataArray], DataArray] = lambda x: x.mean(dim='z')):
        self.transformer = transformer
        self.label_operation = label_operation
        self.box_width_sample = box_width_sample
        slice_dict = dict(x=slice(test_box[0], test_box[2]),
                          y=slice(test_box[1], test_box[3]), z=slice(*z_limits))
        self.dataset = dataset.sel(**slice_dict)
        self.dataset.load()
        self.dataset = self.dataset
        self.dataset = self.dataset[['images', 'labels']]
        self.dataset['labels'] = self.dataset['labels'].astype(int)
        roll_images = self.get_rolling(self.dataset.images).transpose('sample', 'z', 'x_win', 'y_win')
        roll_labels = self.get_rolling(self.dataset.labels).transpose('sample', 'x_win', 'y_win')
        self.ds_test_roll_sample = xr.merge([roll_images, roll_labels]).dropna(dim='sample', how='any')

    def get_item_as_data_array(self, index: int):
        ds = self.ds_test_roll_sample.isel(sample=index)
        ds['x'], ds['y'] = ds['x_win'], ds['y_win']
        ds = ds.swap_dims(x_win='x', y_win='y')
        return ds.images.transpose('z', 'x', 'y'), self.label_operation(ds.labels.transpose('x', 'y'))

    def get_rolling(self, ds: xr.Dataset):
        ds_roll = ds.rolling({'x': self.box_width_sample, 'y': self.box_width_sample}, center=True)
        ds_roll = ds_roll.construct(x='x_win', y='y_win', stride=self.box_width_sample // 4)
        return ds_roll.stack(sample=('x', 'y'))

    def __len__(self):
        return len(self.ds_test_roll_sample.sample)

    def get_sampler(self, batch_size):
        raise NotImplementedError

    def nbytes(self):
        """Total number of bytes in the dataset"""
        raise NotImplementedError

    @property
    def seed(self):
        raise NotImplementedError

    @seed.setter
    def seed(self, value):
        raise NotImplementedError

    def get_slice(self, idx: int) -> Any:
        raise NotImplementedError


class CachedDataset(Dataset):
    def __init__(self, zarr_path: str, group_size=1):
        with dask.config.set(scheduler='synchronous'), xr.open_zarr(zarr_path) as self.ds:
            self.ds = self.ds.chunk({'sample': group_size})
            self.ds_grp = self.ds.groupby(self.ds.sample // group_size)

    def __getitem__(self, index: int):
        try:
            ds_sub = self.ds_grp[index]
            return torch.tensor(ds_sub['samples'].values), torch.tensor(ds_sub['labels'].values)
        except KeyError:
            raise IndexError

    def __len__(self):
        return len(self.ds.sample)
