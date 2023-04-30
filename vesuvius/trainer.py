import json
from abc import ABC, abstractmethod
from typing import Generator, Callable, Type, Tuple

import numpy as np
import torch
from tqdm import tqdm
from xarray import DataArray

from vesuvius.config import Configuration
from vesuvius.trackers import Track


def centre_pixel(da: DataArray) -> DataArray:
    return da.isel(x=len(da.x) // 2, y=len(da.y) // 2).astype(np.float32)


class BaseTrainer(ABC):

    def __init__(self,
                 train_loader: Callable,
                 test_loader: Callable,
                 trackers: Track,
                 config: Configuration) -> None:

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.config = config
        self.trackers = trackers
        self.datapoint = self.outputs = None

        self.train_loader_iter = None
        self.test_loader_iter = None
        self.model0 = None
        self.batch_size = None
        self.total = None
        self.loops = None
        self.os = None

        self.training_steps = self.config.model0.total_steps
        self.epochs = self.config.model0.epochs
        self.batch_size = self.config.batch_size
        self.total = self.epochs * self.training_steps
        self.loops = self.total // self.batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}', '\n')
        self.model0, self.optimizer0, self.scheduler0, self.criterion0 = self.setup_model(self.config.model0)
        self._check_model()
        self.get_train_test_loaders()

    def setup_model(self, model_object):
        model_n = model_object.model.to(self.device)
        os = model_object.optimizer_scheduler
        return model_n, os.optimizer(), os.scheduler(), model_object.criterion

    @abstractmethod
    def get_train_test_loaders(self) -> None:
        self.train_loader_iter = None
        self.test_loader_iter = None

    @abstractmethod
    def forward(self) -> torch.Tensor:
        ...

    @abstractmethod
    def loss(self) -> float:
        ...

    def dummy_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_input = torch.randn(self.config.batch_size,
                                   1,
                                   self.config.box_width_z,
                                   self.config.box_width_xy,
                                   self.config.box_width_xy).to(self.device)
        scalar_input = torch.tensor([2.0] * self.config.batch_size).to(self.device).view(-1, 1).to(self.device)
        return tensor_input, scalar_input

    def _check_model(self) -> None:
        try:
            self.model0.eval()
            with torch.no_grad():
                output = self.model0.forward(*self.dummy_input())
                assert output.shape == (self.config.batch_size, 1)
        except RuntimeError as e:
            raise RuntimeError(f'{e}\n'
                               'Model is not compatible with the input data size')

    @abstractmethod
    def validate(self) -> None:
        ...

    def save_model_output(self):
        self.config.update_nn_kwargs(self.optimizer0, self.scheduler0, self.criterion0, self.learning_rate, self.epochs)
        self.config['performance_dict'] = {'loss/train': self.trackers.logger_loss.average,
                                           'loss/test': self.trackers.logger_test_loss.average,
                                           'steps': self.trackers.incrementer.count}
        self.trackers.saver.model0(self.model0)
        json_file_path = self.trackers.saver.config(self.config)
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        config_json = json.dumps(data, indent=4)
        self.trackers.writer.add_text('config', config_json)

    def trainer_generator(self) -> Generator[Type['BaseTrainer'], None, None]:
        while self.trackers.incrementer.loop < self.loops:
            try:
                next(self)
            except StopIteration:
                return
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
            for item in tqdm(self.trainer_generator(), **tqdm_kwargs):
                yield item
        except Exception as e:
            print("An exception occurred during training: ", e)
            raise

    def __str__(self) -> str:
        loop = self.trackers.incrementer.loop
        lr = self.config.model0.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"Current Loop: {loop}, Learning Rate: {lr}, Epochs: {epochs}, Batch Size: {batch_size}"

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        loop = self.trackers.incrementer.loop
        lr = self.config.model0.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"{classname}(current_loop={loop}, learning_rate={lr}, epochs={epochs}, batch_size={batch_size})"
