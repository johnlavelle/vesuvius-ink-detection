import json
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Union, Optional, Callable, List, Tuple, Any

from .sampler import CropBoxIter


@dataclass
class Configuration:
    info: str
    model: Callable
    volume_dataset_cls: Callable
    crop_box_cls: CropBoxIter
    label_fn: Callable
    training_steps: int
    batch_size: int
    fragments: Union[List[int], Tuple[int]]
    prefix: str
    test_box: Tuple[int, int, int, int]
    test_box_fragment: int = 1
    box_width_xy: int = 61
    box_width_z: int = 65
    balance_ink: bool = True
    num_workers: int = min(1, mp.cpu_count() - 1)
    collate_fn: Optional[Union[Callable, None]] = None
    nn_kwargs: Optional[Union[dict, None]] = None
    extra_kwargs: Optional[Union[dict, None]] = None

    def update_nn_kwargs(self, optimizer_: Any, scheduler_: Any, criterion_: Any, learning_rate: float, epochs: int):
        reprs = {'optimizer': str(optimizer_.__repr__),
                 'scheduler': str(scheduler_.__class__),
                 'criterion': str(criterion_.__class__),
                 'learning_rate': learning_rate,
                 'epochs': epochs}
        for k, v in reprs.items():
            if k in self.nn_kwargs and self.nn_kwargs[k] != v:
                print(f"Warning: {k} will be updated from {self.nn_kwargs[k]} in the config to {v}")
            self.nn_kwargs[k] = v

    def __post_init__(self):
        if self.nn_kwargs is None:
            self.nn_kwargs = {}
        if self.extra_kwargs is None:
            self.extra_kwargs = {}
        assert self.test_box_fragment in self.fragments, "Test box fragment must be in fragments"


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


def save_config(config: Configuration, file_path: str) -> None:
    config_dict = asdict(config, dict_factory=lambda obj: {k: serialize(v) for k, v in obj})
    with open(file_path, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)


def read_config(file_path: str) -> Configuration:
    with open(file_path, "r") as json_file:
        config_dict = json.load(json_file)
    return Configuration(**config_dict)
