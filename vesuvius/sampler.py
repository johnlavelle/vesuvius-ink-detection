from typing import Tuple, Iterator, Callable, Optional

import numpy as np
import xarray as xr
from scipy.stats import qmc

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class CropBoxIter(Protocol):
    total_bounds: Tuple[int, int, int, int]
    sample_box_width: int
    seed: int

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int, int]]:
        ...

    def __next__(self) -> Tuple[int, int, int, int, int, int]:
        ...


class CropBoxSobol(CropBoxIter):

    def __init__(
            self,
            total_bounds: Tuple[int, int, int, int, int, int],
            width_xy: int,
            width_z: int,
            seed: int = 42,
    ):
        """
        :param total_bounds:
        :param width_xy:
        :param width_z:
        """
        self.bounds = total_bounds
        self.width_xy = width_xy
        self.width_z = width_z
        self.sampler = qmc.Sobol(d=3, scramble=True, optimization=None, seed=seed)
        self.l_bounds = np.array([self.bounds[0],
                                  self.bounds[1],
                                  self.bounds[2]])
        self.u_bounds = np.array([self.bounds[3] - (self.width_xy - 2),
                                  self.bounds[4] - (self.width_xy - 2),
                                  self.bounds[5] - (self.width_z - 2)])

    def sobol_sample(self) -> Tuple[int, int, int]:
        return tuple(self.sampler.integers(l_bounds=self.l_bounds, u_bounds=self.u_bounds)[0])

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int, int]]:
        return self

    def __next__(self) -> Tuple[int, int, int, int, int, int]:
        x, y, z = self.sobol_sample()
        return x, x + self.width_xy - 1, y, y + self.width_xy - 1, z, z + self.width_z - 1


# class CropBoxRegular(CropBoxProtocol):
#     def __init__(self, total_bounds: Tuple[int, int, int, int], sample_box_width: int, infinite: bool = False):
#         self.bounds = total_bounds
#         self.box_width = sample_box_width
#         self.l_bounds = np.array([self.bounds[0], self.bounds[1]])
#         self.u_bounds = np.array([self.bounds[2] - sample_box_width, self.bounds[3] - sample_box_width])
#
#     def get_outer_slices(self, dataset: xr.Dataset, window_size: int):
#         x_size, y_size = dataset.dims['x'], dataset.dims['y']
#         for x in range(0, x_size, window_size):
#             for y in range(0, y_size, window_size):
#                 x_upper = min(x + window_size + 1, x_size)
#                 y_upper = min(y + window_size + 1, y_size)
#                 if (x_upper - x >= self.box_width_sample) and (y_upper - y >= self.box_width_sample):
#                     x_slice = slice(x, x_upper)
#                     y_slice = slice(y, y_upper)
#                     yield dict(x=x_slice, y=y_slice)
#
#     def get_dataset_chunks(self):
#         for vds in self.datasets:
#             for slice_ in self.get_outer_slices(vds, self.box_width_large):
#                 yield vds, slice_


class VolumeSampler:

    def __init__(self,
                 dataset: xr.Dataset,
                 box_width: int,
                 z_width: int,
                 samples: int = None,
                 balance=True,
                 crop_cls: Callable[
                     [Tuple[int, int, int, int, int, int], int, int, Optional[int]], CropBoxIter] = CropBoxSobol,
                 seed=42):

        self.ds = dataset
        self.mask_array = self.ds.mask
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
        self.crop_box = crop_cls(bounds, self.box_width, self.z_width, seed)
        self.samples = samples
        self.running_samples = 0

    def get_slice(self, counter=0, max_recursions=50, balance_ink_override=True):
        # Get a random crop box
        x_lower, x_upper, y_lower, y_upper, z_lower, z_upper = next(self.crop_box)
        # bound extents to slice
        slice_ = dict(x=slice(x_lower, x_upper), y=slice(y_lower, y_upper), z=slice(z_lower, z_upper))
        try:
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
        except AssertionError:
            if counter >= max_recursions:
                balance_ink_override = False  # Give up trying to find ink
            return self.get_slice(counter=counter + 1,
                                  max_recursions=max_recursions,
                                  balance_ink_override=balance_ink_override)

    def helping_to_balance_dataset_labels(self, count_ink_pixels: int):
        """Aiming for half ink and half no-ink pixels in the dataset.
        If the ink pixels are less that 50%, only return the slice if ink pixels are present"""
        if (self.total_pixels > 0) and ((self.total_ink_pixels / self.total_pixels) < 0.5):
            return count_ink_pixels > 0
        else:
            return True

    def __iter__(self):
        return self

    def __next__(self):
        if self.samples and self.running_samples >= self.samples:
            raise StopIteration
        self.running_samples += 1
        return self.get_slice()

    def __repr__(self):
        return F"RandomVolumeSampler(mask_array: {self.mask_array.shape},, box_width: {self.box_width})"

    def __str__(self):
        return F"RandomVolumeSampler(mask_array: {self.mask_array.shape},, box_width: {self.box_width})"
