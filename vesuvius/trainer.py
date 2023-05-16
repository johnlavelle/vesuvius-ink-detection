import json
from abc import ABC, abstractmethod
from typing import Generator, Any, Type, Tuple, Iterable
from itertools import repeat, chain, islice
from torch.utils.data import DataLoader

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from xarray import DataArray

from vesuvius.config import Configuration
from vesuvius.trackers import Track


def centre_pixel(da: DataArray) -> DataArray:
    return da.isel(x=len(da.x) // 2, y=len(da.y) // 2).astype(np.float32)


class BaseTrainer(ABC):

    def __init__(self,
                 train_dataset: Any,
                 test_dataset: Any,
                 trackers: Track,
                 config: Configuration) -> None:

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.config = config
        self.trackers = trackers
        self.datapoint = self.output0 = None

        self.model0 = None
        self.batch_size = None
        self.total_loops = None
        self.os = None
        self.optimizer0, self.scheduler0, self.criterion0 = None, None, None

        self.train_loader_iter = self.get_train_loader_iter()
        self.test_loader_iter = self.get_test_loader_iter()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}', '\n')

    def setup_model(self, model_object):
        model_ = model_object.model.to(self.device)
        os = model_object.optimizer_scheduler
        optimizer, scheduler, criterion = os.optimizer(), os.scheduler(), model_object.criterion

        if torch.cuda.device_count() >= 2:
            model_ = nn.DataParallel(model_)
            print('Using DataParallel for training.')

        return model_, optimizer, scheduler, criterion

    def get_train_loader_iter(self):
        self.config.loops_per_epoch = min(len(self.train_dataset), self.config.samples_max)
        self.total_loops = self.config.epochs * self.config.loops_per_epoch
        self.train_loader_iter = chain.from_iterable(repeat(self.train_dataset, self.config.epochs))
        return islice(self.train_loader_iter, self.total_loops)

    def get_test_loader_iter(self) -> Iterable:
        return list(islice(self.test_dataset, self.config.validation_steps))

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

    def _save_model(self, model, suffix: str = '') -> None:
        self.config['performance_dict'] = {'loss/train': self.trackers.logger_loss.average,
                                           'loss/test': self.trackers.logger_test_loss.average,
                                           'steps': self.trackers.incrementer.count}
        self.trackers.saver.model(model, suffix)
        json_file_path = self.trackers.saver.config(self.config)
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        config_json = json.dumps(data, indent=4)
        self.trackers.writer.add_text('config', config_json)

    @abstractmethod
    def save_model(self):
        ...

    def trainer_generator(self) -> Generator[Type['BaseTrainer'], None, None]:
        while self.trackers.incrementer.loop < self.total_loops:
            try:
                next(self)
            except StopIteration:
                return
            yield self

    def __next__(self) -> 'BaseTrainer':
        if self.trackers.incrementer.loop >= self.total_loops:
            raise StopIteration
        self.datapoint = next(self.train_loader_iter)
        self.trackers.increment(len(self.datapoint.label))
        return self

    def __iter__(self):
        tqdm_kwargs = dict(total=self.total_loops, disable=False, desc='Training', position=0)
        try:
            for item in tqdm(self.trainer_generator(), **tqdm_kwargs):
                yield item
        except Exception as e:
            print("An exception occurred during training: ", e)
            raise

    def __str__(self) -> str:
        loop = self.trackers.incrementer.loop
        lr = self.config.model0.learning_rate
        epochs = self.config.epochs
        batch_size = self.batch_size
        return f"Current Loop: {loop}, Learning Rate: {lr}, Epochs: {epochs}, Batch Size: {batch_size}"

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        loop = self.trackers.incrementer.loop
        lr = self.config.model0.learning_rate
        epochs = self.config.epochs
        batch_size = self.config.batch_size
        return f"{classname}(current_loop={loop}, learning_rate={lr}, epochs={epochs}, batch_size={batch_size})"
