import multiprocessing as mp
import warnings

from torch.nn import Module
from dataclasses import dataclass, field
from typing import Union, Optional, Callable, List, Tuple, Any, Type

from vesuvius.sampler import BaseCropBox
from vesuvius.fragment_dataset import BaseDataset


@dataclass
class Configuration:
    info: str
    model: Type[Module]
    volume_dataset_cls: Type[BaseDataset]
    crop_box_cls: Type[BaseCropBox]
    label_fn: Callable
    training_steps: int
    batch_size: int
    fragments: Union[List[int], Tuple[int]]
    test_box: Tuple[int, int, int, int]
    test_box_fragment: int = 1
    box_width_xy: int = 61
    box_width_z: int = 65
    stride_xy: int = None
    stride_z: int = None
    balance_ink: bool = True
    shuffle: bool = True
    group_pixels: bool = False
    num_workers: int = min(1, mp.cpu_count() - 1)
    prefix: str = field(default_factory=lambda: "")
    suffix_cache: str = field(default_factory=lambda: "")
    collate_fn: Optional[Union[Callable, None]] = None
    nn_dict: Optional[Union[dict, None]] = None
    performance_dict: Optional[Union[dict, None]] = None
    extra_dict: Optional[Union[dict, None]] = None

    def update_nn_kwargs(self, optimizer_: Any, scheduler_: Any, criterion_: Any, learning_rate: float, epochs: int):
        reprs = {'optimizer': str(optimizer_.__repr__),
                 'scheduler': str(scheduler_.__class__),
                 'criterion': str(criterion_.__class__),
                 'learning_rate': learning_rate,
                 'epochs': epochs}
        for k, v in reprs.items():
            if k in self.nn_dict and self.nn_dict[k] != v:
                print(f"Warning: {k} will be updated from {self.nn_dict[k]} in the config to {v}")
            self.nn_dict[k] = v

    def __post_init__(self):
        if self.nn_dict is None:
            self.nn_dict = {}
        if self.extra_dict is None:
            self.extra_dict = {}
        assert self.test_box_fragment in self.fragments, "Test box fragment must be in fragments"

        if (self.stride_xy is not None) or (self.stride_z is not None):
            self._crop_box_has_getitem("Strides are not supported for this crop_box_cls")
        if self.group_pixels:
            raise self._crop_box_has_getitem("group_pixels == True is not supported for this crop_box_cls")
        if self.shuffle:
            try:
                self._crop_box_has_getitem("shuffle == True is not supported for this crop_box_cls")
                warnings.warn("Set shuffle == False, to all windows across z, for each x, y.")
            except NotImplementedError:
                pass

    def _crop_box_has_getitem(self, error_message):
        try:
            cb = self.crop_box_cls(total_bounds=(0, 0, 0, 0, 0, 0), width_xy=0, width_z=0)
            cb.__getitem__(None)
        except NotImplementedError:
            raise NotImplementedError(error_message)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)


def serialize(obj: Any) -> Any:
    if isinstance(obj, type):
        return obj.__name__
    elif callable(obj):
        return obj.__name__
    elif isinstance(obj, (list, tuple)):
        return [serialize(item) for item in obj]
    elif hasattr(obj, "as_dict"):
        return obj.as_dict()
    else:
        return obj
