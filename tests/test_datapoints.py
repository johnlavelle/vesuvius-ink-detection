import numpy as np
import pytest
import torch
from xarray import DataArray

from vesuvius.datapoints import cantor_pairing, Datapoint


def create_random_voxels(shape=(3, 3, 3), dims=("x", "y", "z")):
    return DataArray(np.random.rand(*shape), dims=dims)


def test_cantor_pairing():
    assert cantor_pairing(1, 2) == 8
    assert cantor_pairing(3, 4) == 32


def test_datapoint_init():
    voxels = create_random_voxels()
    label = 1
    fragment = 0
    x_start, x_stop = 0, 3
    y_start, y_stop = 0, 3
    z_start, z_stop = 0, 3

    datapoint = Datapoint(voxels, label, fragment, x_start, x_stop, y_start, y_stop, z_start, z_stop)

    assert isinstance(datapoint, Datapoint)
    assert datapoint.label == label
    assert datapoint.fragment == fragment


def test_datapoint_validation():
    voxels = create_random_voxels()
    voxels.values[1, 1, 1] = np.inf
    label = 1
    fragment = 0
    x_start, x_stop = 0, 3
    y_start, y_stop = 0, 3
    z_start, z_stop = 0, 3

    with pytest.raises(ValueError):
        Datapoint(voxels, label, fragment, x_start, x_stop, y_start, y_stop, z_start, z_stop)


def test_datapoint_to_namedtuple():
    voxels = create_random_voxels()
    label = 1
    fragment = 0
    x_start, x_stop = 0, 3
    y_start, y_stop = 0, 3
    z_start, z_stop = 0, 3

    datapoint = Datapoint(voxels, label, fragment, x_start, x_stop, y_start, y_stop, z_start, z_stop)
    datapoint_tuple = datapoint.to_namedtuple()

    voxels_tensor = torch.from_numpy(voxels.expand_dims('Cin').values.astype(np.float32))
    assert datapoint_tuple.voxels.equal(voxels_tensor)
    assert torch.is_tensor(datapoint_tuple.label)
    assert torch.is_tensor(datapoint_tuple.fragment)
    assert torch.is_tensor(datapoint_tuple.x_start)
    assert torch.is_tensor(datapoint_tuple.x_stop)
    assert torch.is_tensor(datapoint_tuple.y_start)
    assert torch.is_tensor(datapoint_tuple.y_stop)
    assert torch.is_tensor(datapoint_tuple.z_start)
    assert torch.is_tensor(datapoint_tuple.z_stop)
    assert torch.is_tensor(datapoint_tuple.fxy_idx)

    assert datapoint_tuple.label.item() == label
    assert datapoint_tuple.fragment.item() == fragment
    assert datapoint_tuple.x_start.item() == x_start
    assert datapoint_tuple.x_stop.item() == x_stop
    assert datapoint_tuple.y_start.item() == y_start
    assert datapoint_tuple.y_stop.item() == y_stop
    assert datapoint_tuple.z_start.item() == z_start
    assert datapoint_tuple.z_stop.item() == z_stop
