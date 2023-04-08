from typing import Tuple, Any, Callable, Optional
from abc import ABC, abstractmethod
from functools import lru_cache
import numbers

import dask
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from xarray import DataArray

from vesuvius.sampler import CropBoxIter, CropBoxSobol
from vesuvius.sampler import VolumeSampler
from vesuvius.data_utils import Datapoint

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class BaseDataset(ABC, Dataset):

    def __init__(self,
                 dataset: Any,
                 box_width_xy: int,
                 box_width_z: int,
                 max_iterations: int = 10_000,
                 label_operation: Callable[[DataArray], float] = lambda x: x.mean(),
                 transformer: Callable[[torch.Tensor], DataArray] = None,
                 crop_cls: Callable[
                     [Tuple[int, int, int, int, int, int], int, int, Optional[int]], CropBoxIter] = CropBoxSobol,
                 balance_ink: bool = False,
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
        self.seed = seed

    @abstractmethod
    def get_item_as_data_array(self, index: int) -> Datapoint:
        ...

    @lru_cache
    def get_sampler(self, seed):
        return VolumeSampler(self.ds, self.box_width_xy, self.box_width_z,
                             self.max_iterations,
                             balance=self.balance_ink, crop_cls=self.crop_box_cls, seed=seed)

    def __getitem__(self, index: int) -> Datapoint:
        assert isinstance(index, int)
        assert index >= 0
        if index >= self.max_iterations:
            raise StopIteration

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.seed = worker_info.seed
        try:
            datapoint = self.get_item_as_data_array(index)
            self.validate_datapoint(datapoint)
        except ValueError as err:
            print(f"Error {err} with index {index}")
            return self.__getitem__(index)

        return self.convert_to_tensors(datapoint)

    @staticmethod
    def validate_datapoint(datapoint: Datapoint) -> Datapoint:
        try:
            assert np.isfinite(datapoint.voxels).all().values.item(), 'Non finite number in voxels'
        except AssertionError as err:
            raise ValueError(err)
        return datapoint

    def convert_to_tensors(self, datapoint: Datapoint) -> Datapoint:
        datapoint = datapoint._asdict()
        for k, v in datapoint.items():
            if k == 'label':
                vl = self.label_operation(datapoint[k])
                if isinstance(vl, DataArray):
                    datapoint[k] = vl.values.astype(np.float32)
                else:
                    datapoint[k] = vl
            if k == 'voxels':
                datapoint[k] = v.expand_dims('Cin')
            if isinstance(datapoint[k], DataArray):
                np_arr = datapoint[k].values.astype(np.float32)
                np_arr.setflags(write=False)
                datapoint[k] = np_arr.copy()
            if isinstance(datapoint[k], np.ndarray) and datapoint[k].shape == ():
                datapoint[k] = datapoint[k].item()
            if isinstance(datapoint[k], numbers.Integral):
                datapoint[k] = np.array([datapoint[k]], dtype=np.int64).reshape(-1)
            if isinstance(datapoint[k], numbers.Real):
                datapoint[k] = np.array([datapoint[k]], dtype=np.float32).reshape(-1)
            datapoint[k] = torch.from_numpy(datapoint[k])
        return Datapoint(**datapoint)

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
        except KeyError:
            raise IndexError

        das = (ds_sub['voxels'], ds_sub['label'], ds_sub['fragment'], ds_sub['x_start'], ds_sub['x_stop'],
               ds_sub['y_start'], ds_sub['y_stop'],ds_sub['z_start'], ds_sub['z_stop'])
        return Datapoint(*(torch.tensor(da.values) for da in das))

    def __len__(self):
        return len(self.ds.sample)
