import multiprocessing as mp
from torch.nn import Module
from dataclasses import dataclass, field
from typing import Union, Optional, Callable, List, Tuple, Any, Type

from vesuvius.sampler import BaseCropBox
from vesuvius.scroll_dataset import BaseDataset


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

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)


def get_fully_qualified_name(obj: Any) -> str:
    module = obj.__module__
    name = obj.__name__
    return f"{module}.{name}"


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
