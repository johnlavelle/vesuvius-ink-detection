import json
import multiprocessing as mp
import pprint
from abc import ABC, abstractmethod
from typing import Generator, Callable, Type

import numpy as np
import torch
from tqdm import tqdm
from xarray import DataArray

from vesuvius.ann.models import HybridModel
from vesuvius.ann.optimisers import OptimiserScheduler
from vesuvius.config import Configuration1, ConfigProtocol
from vesuvius.trackers import Track


def centre_pixel(da: DataArray) -> DataArray:
    return da.isel(x=len(da.x) // 2, y=len(da.y) // 2).astype(np.float32)


class BaseTrainer(ABC):

    def __init__(self,
                 train_loader: Callable,
                 test_loader: Callable,
                 trackers: Track,
                 optimizer_scheduler: Type[OptimiserScheduler],
                 criterion: Callable,
                 config_kwargs: ConfigProtocol,
                 criterion_validate: Callable = None,
                 learning_rate: float = 0.03,
                 l1_lambda: float = 0,
                 epochs: int = 1,
                 validation_steps: int = 20,
                 model_kwargs: dict = None) -> None:

        self.epochs = epochs
        self.validation_steps = validation_steps
        self.train_loader = train_loader
        self.test_loader = test_loader

        if model_kwargs is None:
            model_kwargs = {}
        if config_kwargs is None:
            config_kwargs = {}

        if criterion_validate is None:
            criterion_validate = criterion
        self.criterion_validate = criterion_validate
        self.learning_rate = learning_rate
        self.trackers = trackers
        self.datapoint = self.outputs = None
        self.l1_lambda = l1_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}', '\n')

        self.config = self.configuration(**config_kwargs)

        self.model = self.config.model(**model_kwargs).to(self.device)
        self.batch_size = self.config.batch_size
        self.total = self.epochs * self.config.training_steps
        self.loops = self.total // self.batch_size
        self._check_model()
        pprint.pprint(self.config)
        print()

        self.optimizer_scheduler = optimizer_scheduler(self.model, self.learning_rate, self.total)
        self.optimizer = self.optimizer_scheduler.optimizer()
        self.scheduler = self.optimizer_scheduler.scheduler(self.optimizer)
        self.criterion = criterion
        print(self.optimizer_scheduler, '\n')
        print(self.criterion_validate, '\n')

        self.train_loader_iter = None
        self.test_loader_iter = None

    @abstractmethod
    def forward(self) -> torch.Tensor:
        ...

    @abstractmethod
    def loss(self) -> float:
        ...

    @abstractmethod
    def dummy_input(self) -> torch.Tensor:
        ...

    def _check_model(self) -> None:
        try:
            self.model.eval()
            with torch.no_grad():
                output = self.model.forward(*self.dummy_input())
                assert output.shape == (self.config.batch_size, 1)
        except RuntimeError as e:
            raise RuntimeError(f'{e}\n'
                               'Model is not compatible with the input data size')

    @staticmethod
    def configuration(**kwargs) -> Configuration1:
        xl, yl = 2048, 7168  # lower left corner of the test box
        width, height = 2045, 2048
        default_config = dict(info="",
                              model=HybridModel,
                              volume_dataset_cls=None,
                              crop_box_cls=None,
                              label_fn=centre_pixel,
                              training_steps=32 * (40_000 // 32) - 1,  # This should be small enough to fit on disk
                              batch_size=32,
                              fragments=[1, 2, 3],
                              test_box=(xl, yl, xl + width, yl + height),  # Hold back rectangle
                              test_box_fragment=2,  # Hold back fragment
                              box_width_xy=91,
                              box_width_z=6,
                              balance_ink=True,
                              shuffle=True,
                              group_pixels=False,
                              num_workers=max(1, mp.cpu_count() - 1),
                              prefix='/data/kaggle/input/vesuvius-challenge-ink-detection/train/',
                              suffix_cache='sobol',
                              collate_fn=None)
        return Configuration1(**{**default_config, **kwargs})

    @abstractmethod
    def validate(self) -> None:
        ...

    def save_model_output(self):
        self.config.update_nn_kwargs(self.optimizer, self.scheduler, self.criterion, self.learning_rate, self.epochs)
        self.config['performance_dict'] = {'loss/train': self.trackers.logger_loss.average,
                                           'loss/test': self.trackers.logger_test_loss.average,
                                           'steps': self.trackers.incrementer.count}
        self.trackers.saver.model(self.model)
        json_file_path = self.trackers.saver.config(self.config)
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        config_json = json.dumps(data, indent=4)
        self.trackers.writer.add_text('config', config_json)

    def trainer_generator(self) -> Generator[Type['BaseTrainer'], None, None]:
        while self.trackers.incrementer.loop < self.loops:
            next(self)
            yield self

    def __next__(self) -> 'BaseTrainer':
        if self.trackers.incrementer.loop >= self.loops:
            raise StopIteration
        self.datapoint = next(self.train_loader_iter)
        self.trackers.increment(len(self.datapoint.label))
        return self

    def __iter__(self):
        tqdm_kwargs = dict(total=self.loops, disable=False, desc='Training', position=0)
        try:
            yield from tqdm(self.trainer_generator(), **tqdm_kwargs)
        except Exception as e:
            print("An exception occurred during training: ", e)
            raise

    def __str__(self) -> str:
        loop = self.trackers.incrementer.loop
        lr = self.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"Current Loop: {loop}, Learning Rate: {lr}, Epochs: {epochs}, Batch Size: {batch_size}"

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        loop = self.trackers.incrementer.loop
        lr = self.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"{classname}(current_loop={loop}, learning_rate={lr}, epochs={epochs}, batch_size={batch_size})"
