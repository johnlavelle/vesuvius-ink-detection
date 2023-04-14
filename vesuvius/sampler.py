import sys
from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Type

import numpy as np
import xarray as xr
from scipy.stats import qmc

from vesuvius.utils import vectorise_raster, check_points_in_polygon

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

sys.setrecursionlimit(50_000)


class BaseCropBox(ABC):

    def __init__(self,
                 total_bounds: Tuple[int, int, int, int, int, int],
                 width_xy: int,
                 width_z: int,
                 seed: int = 42,
                 stride_xy: int = None,
                 stride_z: int = None):
        """
        :param total_bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
        :param width_xy:
        :param width_z:
        """
        self.bounds = total_bounds
        self.width_xy = width_xy
        self.width_z = width_z
        self.seed = seed
        self.stride_xy = stride_xy
        self.stride_z = stride_z
        self.sampler = None

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int, int]]:
        return self

    @abstractmethod
    def get_sample(self) -> Tuple[int, int, int]:
        ...

    def __next__(self) -> Tuple[int, int, int, int, int, int]:
        x, y, z = self.get_sample()
        return x, x + self.width_xy - 1, y, y + self.width_xy - 1, z, z + self.width_z - 1

    @abstractmethod
    def __getitem__(self, item):
        ...

    @abstractmethod
    def __len__(self):
        ...


class CropBoxSobol(BaseCropBox):

    def __init__(self,
                 total_bounds: Tuple[int, int, int, int, int, int],
                 width_xy: int,
                 width_z: int,
                 seed: int = 42,
                 stride_xy=None,
                 stride_z=None):
        """
        :param total_bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
        :param width_xy:
        :param width_z:
        """
        super().__init__(total_bounds, width_xy, width_z, seed, stride_xy, stride_z)

        if self.stride_xy is not None and self.stride_z is not None:
            raise NotImplementedError("Stride is not implemented for Sobol sampler")

        self.l_bounds = np.array([self.bounds[0],
                                  self.bounds[1],
                                  self.bounds[2]])
        self.u_bounds = np.array([self.bounds[3] - (self.width_xy - 2),
                                  self.bounds[4] - (self.width_xy - 2),
                                  self.bounds[5] - (self.width_z - 2)])
        self.sampler = qmc.Sobol(d=3, scramble=True, optimization=None, seed=seed)

    def get_sample(self) -> Tuple[int, int, int]:
        return tuple(self.sampler.integers(l_bounds=self.l_bounds, u_bounds=self.u_bounds)[0])

    def __getitem__(self, item):
        raise NotImplementedError("Sobol sampler does not support indexing")

    def __len__(self):
        raise NotImplementedError("Sobol sampler does not support len")


class CropBoxRegular(BaseCropBox):

    def __init__(self,
                 total_bounds: Tuple[int, int, int, int, int, int],
                 width_xy: int,
                 width_z: int,
                 seed: int = 42,
                 stride_xy=None,
                 stride_z=None):
        """
        :param total_bounds:
        :param width_xy:
        :param width_z:
        """
        super().__init__(total_bounds, width_xy, width_z, seed, stride_xy, stride_z)

        if self.stride_xy is None and self.stride_z is None:
            self.stride_xy = width_xy
            self.stride_z = width_z

        self.l_bounds = np.array([self.bounds[0],
                                  self.bounds[1],
                                  self.bounds[2]])
        self.u_bounds = np.array([self.bounds[3] - self.width_xy,
                                  self.bounds[4] - self.width_xy,
                                  self.bounds[5] - self.width_z])
        rng = np.random.default_rng(seed) if seed is not None else None
        self.sampler = list(self.xyz_sampler(self.l_bounds, self.u_bounds, self.stride_xy, self.stride_z, rng))

    @staticmethod
    def xyz_sampler(l_bounds_xyz, u_bounds_xyz, stride_xy, stride_z, rng) -> Iterator[Tuple[int, int, int]]:
        xs = np.arange(l_bounds_xyz[0], u_bounds_xyz[0], stride_xy)
        xs_rnd = rng.choice(xs, size=len(xs), replace=False)
        ys = np.arange(l_bounds_xyz[0], u_bounds_xyz[0], stride_xy)
        ys_rnd = rng.choice(ys, size=len(xs), replace=False)
        if rng is None:
            xs_rnd = xs
            ys_rnd = ys
        for x in xs_rnd:
            for y in ys_rnd:
                for z in range(l_bounds_xyz[2], u_bounds_xyz[2], stride_z):
                    yield x, y, z

    def get_sample(self) -> Tuple[int, int, int]:
        return self.sampler.pop(0)

    def __getitem__(self, item) -> Tuple[int, int, int, int, int, int]:
        x, y, z = self.sampler[item]
        return x, x + self.width_xy - 1, y, y + self.width_xy - 1, z, z + self.width_z - 1

    def __len__(self):
        return len(self.sampler)


class BaseVolumeSampler(ABC):

    def __init__(self,
                 dataset: xr.Dataset,
                 box_width: int,
                 z_width: int,
                 samples: int = None,
                 balance=True,
                 crop_cls: Type[BaseCropBox] = CropBoxSobol,
                 seed=42,
                 stride_xy=None,
                 stride_z=None):

        self.ds = dataset
        self.mask_array = self.ds.full_mask
        self.labels_array = self.ds.labels
        self.box_width = box_width
        self.z_width = z_width
        self.balance_ink = balance
        self.total_ink_pixels = 0
        self.total_pixels = 0
        x_lower = self.ds.coords['x'][0].item()
        x_upper = self.ds.coords['x'][-1].item()
        y_lower = self.ds.coords['y'][0].item()
        y_upper = self.ds.coords['y'][-1].item()
        z_lower = self.ds.coords['z'][0].item()
        z_upper = self.ds.coords['z'][-1].item()
        bounds = (x_lower, y_lower, z_lower, x_upper, y_upper, z_upper)
        self.z_upper_max = self.ds.z[-1].item() - self.z_width
        self.crop_box = crop_cls(bounds, self.box_width, self.z_width, seed, stride_xy, stride_z)
        self.samples = samples
        self.running_samples = 0

    def check_slice(self, slice_, balance_ink_override=True):
        # bound extents to slice
        slice_xy = {key: value for key, value in slice_.items() if key in ('x', 'y')}
        mask_array = self.mask_array.sel(**slice_xy)
        assert mask_array.all(), "Cropped box contains masked pixels."
        assert mask_array.shape == (self.box_width, self.box_width)

        # Reduce the number of non-ink pixels in the dataset if unbalanced,
        # by only returning slices that contain ink.
        sub_labels = self.labels_array.sel(**slice_xy)
        count_ink_pixels = int(sub_labels.sum())
        count_pixels = int(sub_labels.size)
        if self.balance_ink and balance_ink_override:
            assert self.helping_to_balance_dataset_labels(count_ink_pixels)
        self.total_ink_pixels += count_ink_pixels
        self.total_pixels += count_pixels

        return slice_

    def helping_to_balance_dataset_labels(self, count_ink_pixels: int):
        """Aiming for half ink and half no-ink pixels in the dataset.
        If the ink pixels are less that 50%, only return the slice if ink pixels are present"""
        if (self.total_pixels > 0) and ((self.total_ink_pixels / self.total_pixels) < 0.5):
            return count_ink_pixels > 0
        else:
            return True

    @abstractmethod
    def __getitem__(self, index: int):
        ...


class VolumeSamplerRndXYZ(BaseVolumeSampler):

    def get_slice(self, crop_box, counter=0, max_recursions=200, balance_ink_override=True):
        try:
            x_lower, x_upper, y_lower, y_upper, z_lower, z_upper = next(crop_box)
            slice_ = dict(x=slice(x_lower, x_upper), y=slice(y_lower, y_upper), z=slice(z_lower, z_upper))
            self.check_slice(slice_, balance_ink_override)
            return slice_
        except AssertionError:
            if counter >= max_recursions:
                balance_ink_override = False  # Give up trying to find ink
            return self.get_slice(crop_box,
                                  counter=counter + 1,
                                  max_recursions=max_recursions,
                                  balance_ink_override=balance_ink_override)

    def __iter__(self):
        return self

    def __next__(self):
        if self.samples and self.running_samples >= self.samples:
            raise StopIteration
        self.running_samples += 1
        return self.get_slice(self.crop_box)

    def __getitem__(self, index: int):
        return next(self)

    def __len__(self):
        return self.samples


class VolumeSamplerRegularZ(BaseVolumeSampler):

    def __init__(self,
                 dataset: xr.Dataset,
                 box_width: int,
                 z_width: int,
                 samples: int = None,
                 balance=True,
                 crop_cls: Type[BaseCropBox] = CropBoxSobol,
                 seed=42,
                 stride_xy=None,
                 stride_z=None):
        # use super to set up the crop box
        super().__init__(dataset, box_width, z_width, samples, balance, crop_cls, seed, stride_xy, stride_z)
        self.crop_box.sampler = self.reduce_samples(self.crop_box.sampler)

    def reduce_samples(self, crop_box_sampler):
        """Reduce the number of samples to the number of pixels in the mask"""
        mask_poly = vectorise_raster(self.ds.mask.values).simplify(200).buffer(self.box_width)
        # self.plot_mask(mask_poly)
        xy_points = list(zip(*list(zip(*crop_box_sampler))[:2]))
        df_check = check_points_in_polygon(xy_points, mask_poly)
        return np.array(crop_box_sampler)[df_check['inside_polygon']].tolist()

    @staticmethod
    def plot_mask(mask_poly):
        import matplotlib.pyplot as plt
        from shapely.wkt import loads

        polygon_from_output = loads(mask_poly.wkt)
        x, y = polygon_from_output.exterior.xy
        plt.figure()
        plt.plot(x, y, marker='o', linestyle='-')
        plt.axis('equal')
        plt.show()

    def __getitem__(self, index: int):
        x_lower, x_upper, y_lower, y_upper, z_lower, z_upper = self.crop_box[index]
        slice_ = dict(x=slice(x_lower, x_upper), y=slice(y_lower, y_upper), z=slice(z_lower, z_upper))
        try:
            self.check_slice(slice_)
            return slice_
        except AssertionError:
            raise IndexError

    def __len__(self):
        return len(self.crop_box)
