from itertools import compress
from os import path
from typing import Tuple, List

import dask.array as da
import dask
import dask.config
import numpy as np
import torch
import xarray as xr
from dask import array
from dask.distributed import Client
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import Dataset

from vesuvius.datapoints import Datapoint


def open_dataset(directory, fragment):
    fname = f'/data/kaggle/input/vesuvius-challenge-ink-detection/{directory}/{fragment}/surface_volume.zarr'
    ds = xr.open_zarr(fname,
                      consolidated=True,
                      chunks={'z': 65},
                      overwrite_encoded_chunks=True)
    ds = ds.unify_chunks()
    ds = ds.transpose('z', 'y', 'x')
    return ds


def get_arrays(ds) -> List:
    # Start a Dask Client for parallel execution

    # Remove the last chunk if it is not the same size as the other chunks
    chunks = dict(ds.chunks)
    ds = ds.isel(x=slice(0, -(len(ds.x) % chunks['x'][0])), y=slice(0, -(len(ds.y) % chunks['y'][0])))
    chunks = dict(ds.chunks)
    chunks_x, chunks_y = set(chunks['x']), set(chunks['y'])
    assert len(chunks_x) == 1
    assert len(chunks_y) == 1
    chunks_x, chunks_y = chunks_x.pop(), chunks_y.pop()

    xs, ys = array.meshgrid(ds['x'], ds['y'])
    ds['xs'] = xr.DataArray(xs, dims=('y', 'x'))
    ds['ys'] = xr.DataArray(ys, dims=('y', 'x'))
    ds = ds.chunk(chunks)

    def delay_and_flatten(v):
        return ds[v].data.to_delayed().flatten()

    voxels = delay_and_flatten('images')
    labels = delay_and_flatten('labels')
    mask = delay_and_flatten('mask')
    xs = delay_and_flatten('xs')
    ys = delay_and_flatten('ys')

    mask_all = dask.compute(m.all() for m in mask)
    mask_all = np.array(mask_all[0])

    delayed_arrays = compress(zip(voxels, labels, xs, ys), mask_all)
    # total_pixels = chunks_x * chunks_y
    # ratio_ink = dask.compute((labels.sum() / total_pixels) for voxels, labels, xs, ys in delayed_arrays)
    return delayed_arrays


def fragment_to_int(fragment):
    if isinstance(fragment, int):
        return fragment
    else:
        fragment_mapping = {'a': 1, 'b': 2}
        return fragment_mapping[fragment]


def cache_arrays(delayed_objects):
    # Convert the list of Dask Delayed objects to a Dask array
    dask_array = da.from_delayed(delayed_objects, shape=(len(delayed_objects),), dtype=float)

    # Specify chunk size
    chunk_size = 1000
    dask_array = dask_array.rechunk(chunk_size)

    # Store the Dask array to an HDF5 file
    dask_array.to_hdf5('myfile.h5', '/data/tmp')


class ChunkedDataset(Dataset):

    def __init__(self, directory, fragment):
        self.fragment = fragment
        self.ds = open_dataset(directory, fragment)
        self.delayed_arrays = get_arrays(self.ds)

    def __getitem__(self, item):
        voxel_chunk, labels_chunk, x_chunk, y_chunk = self.delayed_arrays[item]
        voxel_chunk = np.expand_dims(voxel_chunk[27:37, 32:96, 32:96].compute(), axis=0)
        dp = Datapoint(voxels=voxel_chunk,
                       label=int(labels_chunk[64, 64].compute()),
                       fragment=fragment_to_int(self.fragment),
                       x_start=x_chunk[0, 0].compute(),
                       x_stop=x_chunk[0, -1].compute(),
                       y_start=y_chunk[0, 0].compute(),
                       y_stop=y_chunk[-1, 0].compute(),
                       z_start=self.ds['z'][0].item(),
                       z_stop=self.ds['z'][-1].item())
        return dp.to_namedtuple()

    def __len__(self):
        return len(self.delayed_arrays)


def get_chunked_dataset(directory, fragments):
    base_dir = '/data/kaggle/input/vesuvius-challenge-ink-detection/'
    cache_filename = path.join(base_dir, directory, 'dataset_cache.pt')
    if not path.exists(cache_filename):
        with Client(n_workers=4) as client:
            print(client.dashboard_link)
            combined_dataset = ConcatDataset([ChunkedDataset(directory, i) for i in fragments])
            torch.save(combined_dataset, cache_filename)

    dask.config.set(scheduler='synchronous')
    combined_dataset = torch.load(cache_filename)
    return combined_dataset

# cache_filename = '/data/kaggle/input/vesuvius-challenge-ink-detection/train/dataset_cache.pt'
# if not path.exists(cache_filename):
#     with Client(n_workers=8) as client:
#         print(client.dashboard_link)
#         ds1 = ChunkedDataset(1)
#         ds2 = ChunkedDataset(2)
#         ds3 = ChunkedDataset(3)
#         combined_dataset = ConcatDataset([ds1, ds2, ds3])
#         torch.save(combined_dataset, cache_filename)
#
# dask.config.set(scheduler='synchronous')
# combined_dataset = torch.load(cache_filename)
# train_loader = DataLoader(combined_dataset, batch_size=16, num_workers=1, shuffle=True)
# train_loader = iter(train_loader)
# for i in range(100):
#     next(train_loader)
#     print(i)


if __name__ == '__main__':
    cds = get_chunked_dataset('train', [1, 2, 3])
    print(cds[0])

    cds = get_chunked_dataset('test', ['a', 'b'])
    print(cds[0])

    pass
    # dask.config.set(scheduler='synchronous')

    # ds = open_dataset('train', 1)
    # delayed_objects = get_arrays(ds)
    #
    # import dask.array as da
    #
    # # Convert the list of Dask Delayed objects to a list of Dask arrays
    # dask_arrays = [da.from_delayed(obj, shape=(...), dtype=...) for obj in delayed_objects]
    #
    # # Stack the list of Dask arrays into a single Dask array
    # dask_array = da.stack(dask_arrays)
    #
    # # Specify chunk size
    # chunk_size = 32
    # dask_array = dask_array.rechunk(chunk_size)
    #
    # # Store the Dask array to an HDF5 file
    # dask_array.to_hdf5('myfile.h5', '/data/tmp')

