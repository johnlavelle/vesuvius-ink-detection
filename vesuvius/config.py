from importlib import import_module
import json
import multiprocessing as mp
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Union, Optional, Callable, Iterable, Tuple, Any, Type, Dict

from torch.nn import BCEWithLogitsLoss

from vesuvius.ann import optimisers
from vesuvius.ann.transforms import *
from vesuvius.fragment_dataset import BaseDataset
from vesuvius.sampler import BaseCropBox


@dataclass
class ConfigurationModel:
    model: Module = None
    learning_rate: float = 0.03
    l1_lambda: float = 0
    criterion: Union[Module, Callable] = None
    optimizer_scheduler_cls: Type[optimisers.OptimiserScheduler] = optimisers.SGDOneCycleLR

    optimizer_scheduler: Union[optimisers.OptimiserScheduler, None] = field(init=False)
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
        if self.total_loops == 0:
            self.optimizer_scheduler = None
        else:
            self.optimizer_scheduler = self.optimizer_scheduler_cls(self.model, self.learning_rate, self.total_loops)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], model_path: Optional[str] = None) -> "ConfigurationModel":
        unwanted_keys = ['optimizer_scheduler', '_total_loops']
        for key in unwanted_keys:
            config_dict.pop(key, None)

        # Import the classes and create an instances

        model_config_dict = config_dict["model"]
        model_class = import_module("vesuvius.ann.models").__getattribute__(model_config_dict["class"])
        model_inst = model_class(**model_config_dict["params"])
        config_dict["model"] = model_inst

        if config_dict['criterion']:
            try:
                config_dict['criterion'] = import_module("vesuvius.ann.criterions").__getattribute__(config_dict["criterion"])()
            except TypeError:
                print('Failed to import criterion')

        opt_inst = import_module('vesuvius.ann.optimisers').__getattribute__(config_dict['optimizer_scheduler_cls'])
        config_dict['optimizer_scheduler_cls'] = opt_inst

        if os.path.exists(model_path):
            model_inst.load_state_dict(torch.load(model_path))

        return cls(**config_dict)


xl, yl = 2048, 7168  # lower left corner of the test box
width, height = 2045, 2048


@dataclass
class Configuration:
    info: str = ""
    samples_max: int = 10_000_000
    volume_dataset_cls: Optional[Type[BaseDataset]] = None
    crop_box_cls: Optional[Type[BaseCropBox]] = None
    label_fn: Callable[..., Any] = None
    transformers: Optional[Callable[..., Any]] = None
    batch_size: int = 32
    fragments: Union[Iterable[int], Iterable[str]] = (1, 2, 3)
    test_box: Tuple[int, int, int, int] = (xl, yl, xl + width, yl + height)  # Hold back rectangle
    test_box_fragment: int = 2
    box_width_xy: int = 91
    box_width_z: int = 6
    box_sub_width_z: int = 6
    stride_xy: Optional[int] = None
    stride_z: Optional[int] = None
    balance_ink: bool = True
    shuffle: bool = True
    validation_steps: int = 100
    num_workers: int = max(1, mp.cpu_count() - 1)
    prefix: str = "/data/kaggle/input/vesuvius-challenge-ink-detection/train/"
    suffix_cache: str = "sobol"
    collate_fn: Optional[Callable[..., Any]] = None
    nn_dict: Optional[Dict[str, Any]] = None
    model0: Optional[ConfigurationModel] = field(default_factory=ConfigurationModel)
    model1: Optional[ConfigurationModel] = field(default_factory=ConfigurationModel)
    epochs: int = 1
    accumulation_steps: int = None
    seed: int = 648
    performance_dict: Optional[Dict[str, Any]] = None
    extra_dict: Optional[Dict[str, Any]] = None
    _loops_per_epoch: int = field(init=False, default=10_000_000)
    _epochs: int = field(init=False, default=1)

    def __post_init__(self, *args, **kwargs):
        self._loops_per_epoch = kwargs.get("steps", self.loops_per_epoch)
        self._epochs = kwargs.get("epochs", self.epochs)
        self.update_configuration_model()
        assert self.test_box_fragment in self.fragments, "Test box fragment must be in fragments"

        if (self.stride_xy is not None) or (self.stride_z is not None):
            self._crop_box_has_getitem("Strides are not supported for this crop_box_cls")
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

    @property
    def total_steps(self):
        return self.loops_per_epoch * self.epochs

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
    def loops_per_epoch(self):
        return self._loops_per_epoch

    @loops_per_epoch.setter
    def loops_per_epoch(self, value):
        self._loops_per_epoch = value
        self.update_configuration_model()

    def update_configuration_model(self):
        total_steps = self.get_total_steps()
        for m in (self.model0, self.model1):
            try:
                setattr(m, 'total_loops', total_steps)
                getattr(m, 'update_optimizer_scheduler')()
            except AttributeError:
                pass

    def get_total_steps(self):
        return self._loops_per_epoch * self._epochs

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
        elif isinstance(obj, dict):
            return {k: self.serialize(v) for k, v in obj.items()}
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

    @classmethod
    def from_dict(cls, config_dir: str) -> "Configuration":
        # Load config.json from the config_dir
        config_path = os.path.join(config_dir, "config.json")
        with open(config_path) as json_file:
            config_dict = json.load(json_file)

        # Remove unwanted items
        config_dict.pop('_loops_per_epoch', None)
        config_dict.pop('_epochs', None)

        # Deserialize objects from the dictionary
        for key, value in config_dict.items():
            if isinstance(value, dict) and "class" in value and "params" in value:
                class_name = value["class"]
                params = value["params"]
                config_dict[key] = cls.deserialize(class_name, params)
            elif isinstance(value, str) and value.startswith("Compose(["):
                transform_strs = value[9:-2].split(", ")
                config_dict[key] = transforms.Compose(
                    [cls.deserialize(transform_str) for transform_str in transform_strs])

        # Deserialize ConfigurationModel objects for model0 and model1
        if config_dict['model0']['model']:
            config_dict["model0"] = ConfigurationModel.from_dict(config_dict["model0"],
                                                                 os.path.join(config_dir, "model0.pt"))
        if config_dict['model1']['model']:
            config_dict["model1"] = ConfigurationModel.from_dict(config_dict["model1"],
                                                                 os.path.join(config_dir, "model1.pt"))

        return cls(**config_dict)

    @staticmethod
    def deserialize(class_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if params is None:
            params = {}
        try:
            cls = globals()[class_name]
        except KeyError:
            raise ValueError(f"Class {class_name} not found in the current module")

        if issubclass(cls, ConfigurationModel):
            return cls.from_dict(params)
        else:
            return cls(**params)
