from typing import Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset
from shapely.geometry import Polygon
from skimage.measure import find_contours


def normalise_images(dataset: Dataset) -> Dataset:
    # # normalize across the z layers
    ds_z_mean = dataset['images'].mean(dim=['x', 'y'], skipna=True).compute()
    ds_z_std = dataset['images'].std(dim=['x', 'y'], skipna=True).compute()
    dataset['images'] = (dataset['images'] - ds_z_mean) / ds_z_std
    dataset['images'].attrs['ds_z_mean'] = ds_z_mean.values
    dataset['images'].attrs['ds_z_std'] = ds_z_std.values

    # normalize entire dataset
    # ds_mean = dataset['images'].mean(skipna=True).compute()
    # ds_std = dataset['images'].std(skipna=True).compute()
    # dataset['images'] = (dataset['images'] - ds_mean) / ds_std
    # dataset['images'].attrs['mean'] = ds_mean.item()
    # dataset['images'].attrs['std'] = ds_std.item()

    return dataset


def coords_to_poly(coords: Tuple[int, int, int, int]) -> Polygon:
    # Create the points representing the four corners of the rectangle
    x_l, y_l, x_r, y_t = coords
    lower_left = (x_l, y_l)
    upper_left = (x_l, y_t)
    upper_right = (x_r, y_t)
    lower_right = (x_r, y_l)
    return Polygon([lower_left, upper_left, upper_right, lower_right, lower_left])


def vectorise_raster(raster: np.ndarray) -> Polygon:
    raster.astype(int)
    contours = find_contours(raster, 0.5)
    # select first contour and swap x and y
    contour = contours[0][:, [1, 0]]
    return Polygon(contour)


def get_hold_back_mask(dataset: xr.Dataset, test_box_coords: Tuple[int, int, int, int]) -> DataArray:
    test_filter = ~((dataset.x >= test_box_coords[0]) &
                    (dataset.x <= test_box_coords[0] + 700) &
                    (dataset.y >= test_box_coords[1]) &
                    (dataset.y <= test_box_coords[1] + 950))
    return dataset['mask'].where(test_filter, drop=False, other=False)


@dataclass
class BaseTracker(ABC):
    tag: str
    summary_writer: SummaryWriter = field(default_factory=SummaryWriter)
    value: float = 0.0
    i: int = 0

    @abstractmethod
    def update(self, loss: float, batch_size: int) -> None:
        ...

    @abstractmethod
    def log(self, iteration: int) -> None:
        ...
