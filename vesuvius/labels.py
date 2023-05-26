import numpy as np
from xarray import DataArray


def centre_pixel(da: DataArray) -> DataArray:
    return da.isel(x=len(da.x) // 2, y=len(da.y) // 2).astype(np.float32)


def centre_pixel_region(da: DataArray, buffer=5) -> DataArray:
    c_x = len(da.x) // 2
    c_y = len(da.y) // 2
    return da.isel(x=slice(c_x - buffer, c_x + buffer),
                   y=slice(c_y - buffer, c_y + buffer)).astype(np.float32)
