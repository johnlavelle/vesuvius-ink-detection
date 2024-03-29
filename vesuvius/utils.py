import dataclasses
from contextlib import contextmanager
from timeit import default_timer
from typing import Tuple, Any

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon, Point
from skimage.measure import find_contours
from xarray import DataArray, Dataset


def voxel_stats(data_array: DataArray) -> Tuple[DataArray, DataArray]:
    # # normalize across the z layers
    ds_z_mean = data_array.mean(dim=['x', 'y'], skipna=True).compute()
    ds_z_std = data_array.std(dim=['x', 'y'], skipna=True).compute()
    # data_array.attrs['ds_z_mean'] = ds_z_mean
    # data_array.attrs['ds_z_std'] = ds_z_std

    # normalize entire dataset
    # ds_mean = dataset['images'].mean(skipna=True).compute()
    # ds_std = dataset['images'].std(skipna=True).compute()
    # dataset['images'] = (dataset['images'] - ds_mean) / ds_std
    # dataset['images'].attrs['mean'] = ds_mean.item()
    # dataset['images'].attrs['std'] = ds_std.item()

    return ds_z_mean, ds_z_std


def normalise_voxels(voxels) -> DataArray:
    voxels_mean_z = voxels.mean(dim=['x', 'y'], skipna=True)
    voxels_std_z = voxels.std(dim=['x', 'y'], skipna=True)
    eps = 1e-6  # set a small non-zero value
    voxels_std_z = voxels_std_z.where(voxels_std_z != 0, eps)
    return (voxels - voxels_mean_z) / voxels_std_z


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
                    (dataset.x <= test_box_coords[2]) &
                    (dataset.y >= test_box_coords[1]) &
                    (dataset.y <= test_box_coords[3]))
    return dataset['mask'].where(test_filter, drop=False, other=False)


def check_points_in_polygon(points, polygon_coords):
    polygon = Polygon(polygon_coords)
    points_df = pd.DataFrame(points, columns=['x', 'y'])
    points_df['geometry'] = points_df.apply(lambda row: Point(row['x'], row['y']), axis=1)
    points_df['inside_polygon'] = points_df['geometry'].apply(lambda point: point.within(polygon))
    return points_df


@contextmanager
def timer(name):
    start_time = default_timer()
    yield
    end_time = default_timer()
    elapsed_time = end_time - start_time
    print(f"{name} took {elapsed_time:.1f} seconds")


class CustomDataLoaderError(Exception):
    pass


def pretty_print_dataclass(obj: Any, indent: int = 4) -> str:
    if not dataclasses.is_dataclass(obj):
        return str(obj)

    indent_str = ' ' * indent
    result = obj.__class__.__name__ + "(\n"

    for field in dataclasses.fields(obj):
        field_value = getattr(obj, field.name)
        field_value_str = pretty_print_dataclass(field_value, indent + 4)
        result += f"{indent_str}{field.name}={field_value_str},\n"

    result += ")"
    print(result)
