import multiprocessing as mp
import warnings

from torch.nn import Module
import torchvision.transforms as transforms
from dataclasses import dataclass, field, asdict, InitVar
from typing import Union, Optional, Callable, List, Tuple, Any, Type, Dict, Protocol

from vesuvius.sampler import BaseCropBox, CropBoxRegular
from vesuvius.fragment_dataset import BaseDataset
from vesuvius.ann import optimisers
from vesuvius.ann.models import HybridModel


@dataclass
class ConfigurationModel:
    model: Module = HybridModel()
    learning_rate: float = 0.03
    l1_lambda: float = 0
    criterion: Module = None
    optimizer_scheduler_cls: Type[optimisers.OptimiserScheduler] = optimisers.SGDOneCycleLR
    optimizer_scheduler: optimisers.OptimiserScheduler = field(init=False)
    _total_loops: int = field(init=False, default=0)

    def __post_init__(self):
        self._total_loops = self.total_loops
        self.update_optimizer_scheduler()

    @property
    def total_loops(self):
        return self._total_loops

    @total_loops.setter
    def total_loops(self, value):
        self._total_loops = value
        self.update_optimizer_scheduler()

    def update_optimizer_scheduler(self):
        self.optimizer_scheduler = self.optimizer_scheduler_cls(self.model, self.learning_rate, self.total_loops)


xl, yl = 2048, 7168  # lower left corner of the test box
width, height = 2045, 2048


@dataclass
class Configuration:
    info: str = ""
    total_steps_max: int = 10_000
    epochs: int = 1
    volume_dataset_cls: Optional[Type[BaseDataset]] = None
    crop_box_cls: Optional[Type[BaseCropBox]] = None
    label_fn: Callable[..., Any] = None
    transformers: Optional[Callable[..., Any]] = None
    batch_size: int = 32
    fragments: Union[List[int], Tuple[int, ...]] = (1, 2, 3)
    test_box: Tuple[int, int, int, int] = (xl, yl, xl + width, yl + height)  # Hold back rectangle
    test_box_fragment: int = 2
    box_width_xy: int = 91
    box_width_z: int = 6
    stride_xy: Optional[int] = None
    stride_z: Optional[int] = None
    balance_ink: bool = True
    shuffle: bool = True
    group_pixels: bool = False
    validation_steps: int = 100
    num_workers: int = max(1, mp.cpu_count() - 1)
    prefix: str = "/data/kaggle/input/vesuvius-challenge-ink-detection/train/"
    suffix_cache: str = "sobol"
    collate_fn: Optional[Callable[..., Any]] = None
    nn_dict: Optional[Dict[str, Any]] = None
    model0: Optional[ConfigurationModel] = None
    model1: Optional[ConfigurationModel] = None
    accumulation_steps: int = None
    performance_dict: Optional[Dict[str, Any]] = None
    extra_dict: Optional[Dict[str, Any]] = None
    _loops: int = field(init=False, default=10_000)
    _epochs: int = field(init=False, default=1)

    @property
    def total_steps(self):
        return self.loops * self.epochs

    @total_steps.setter
    def total_steps(self, value):
        raise AttributeError("Cannot set total_steps directly. Update steps and/or epochs instead.")

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        self._epochs = value
        self.update_configuration_model()

    @property
    def loops(self):
        return self._loops

    @loops.setter
    def loops(self, value):
        self._loops = value
        self.update_configuration_model()

    def update_configuration_model(self):
        total_steps = self.get_total_steps()
        if self.model0 is not None:
            self.model0.total_loops = total_steps
            self.model0.update_optimizer_scheduler()
        if self.model1 is not None:
            self.model1.total_loops = total_steps
            self.model1.update_optimizer_scheduler()

    def get_total_steps(self):
        return self._loops * self._epochs

    def __post_init__(self, *args, **kwargs):
        self._loops = kwargs.get("steps", self.loops)
        self._epochs = kwargs.get("epochs", self.epochs)
        self.update_configuration_model()
        assert self.test_box_fragment in self.fragments, "Test box fragment must be in fragments"

        if (self.stride_xy is not None) or (self.stride_z is not None):
            self._crop_box_has_getitem("Strides are not supported for this crop_box_cls")
        if self.group_pixels:
            self._crop_box_has_getitem("group_pixels == True is not supported for this crop_box_cls")
        if self.shuffle:
            try:
                self._crop_box_has_getitem("shuffle == True is not supported for this crop_box_cls")
                warnings.warn("Set shuffle == False, to all windows across z, for each x, y.")
            except NotImplementedError:
                pass

    def _crop_box_has_getitem(self, error_message):
        try:
            assert hasattr(self.crop_box_cls, "__getitem__")
        except AssertionError:
            raise NotImplementedError(error_message) from None

    def serialize(self, obj: Any) -> Any:
        if isinstance(obj, type):
            return obj.__name__
        elif hasattr(obj, "as_dict"):
            return {
                "class": obj.__class__.__name__,
                "params": obj.as_dict()
            }
        elif isinstance(obj, transforms.Compose):
            return "Compose([{}])".format(", ".join([self.serialize(transform) for transform in obj.transforms]))
        elif callable(obj):
            if hasattr(obj, '__name__'):
                return obj.__name__
            else:
                return obj.__class__.__name__
        elif isinstance(obj, (list, tuple)):
            return [self.serialize(item) for item in obj]
        elif hasattr(obj, "as_dict"):
            return obj.as_dict()
        else:
            return obj

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self, dict_factory=lambda obj: {k: self.serialize(v) for k, v in obj})

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)
