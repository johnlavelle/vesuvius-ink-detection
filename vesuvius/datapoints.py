import numbers
from dataclasses import dataclass, field
from typing import NamedTuple, Union
import numpy as np
import torch
from xarray import DataArray


def cantor_pairing(a: int, b: int) -> int:
    return (a + b) * (a + b + 1) // 2 + b


class DatapointTuple(NamedTuple):
    voxels: torch.Tensor
    label: torch.Tensor
    fragment: torch.Tensor
    x_start: torch.Tensor
    x_stop: torch.Tensor
    y_start: torch.Tensor
    y_stop: torch.Tensor
    z_start: torch.Tensor
    z_stop: torch.Tensor
    fxy_idx: torch.Tensor


@dataclass
class Datapoint:
    voxels: DataArray
    label: Union[int, DataArray]
    fragment: Union[int, DataArray]
    x_start: Union[int, DataArray]
    x_stop: Union[int, DataArray]
    y_start: Union[int, DataArray]
    y_stop: Union[int, DataArray]
    z_start: Union[int, DataArray]
    z_stop: Union[int, DataArray]
    fxy_idx: Union[int, DataArray] = field(init=False)

    def __post_init__(self):
        self.validate_data()
        self.fxy_idx = self.to_multi_index()

    def to_multi_index(self) -> int:
        unique_index = cantor_pairing(self.fragment, self.x_start)
        unique_index = cantor_pairing(unique_index, self.y_start)
        return unique_index

    def to_namedtuple(self) -> DatapointTuple:
        fields = [
            "voxels",
            "label",
            "fragment",
            "x_start",
            "x_stop",
            "y_start",
            "y_stop",
            "z_start",
            "z_stop",
            "fxy_idx",
        ]
        return DatapointTuple(*(self._to_tensor(getattr(self, k)) for k in fields))

    @staticmethod
    def _to_tensor(field_value):
        if isinstance(field_value, DataArray):
            field_value = field_value.values
            field_value.setflags(write=False)
            field_value = field_value.copy()
        elif isinstance(field_value, np.ndarray) and np.issubdtype(field_value.dtype, np.floating):
            field_value = field_value.astype(np.float32)
        elif isinstance(field_value, numbers.Integral):
            field_value = np.array([field_value], dtype=np.int64).reshape(-1)
        elif isinstance(field_value, numbers.Real):
            field_value = np.array([field_value], dtype=np.float32).reshape(-1)

        return torch.from_numpy(field_value)

    def validate_data(self):
        try:
            assert np.isfinite(self.voxels).all(), 'Non finite number in voxels'
        except AssertionError as err:
            raise ValueError(err)
