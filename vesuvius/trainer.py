import gc
import json
import os
import pprint
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from typing import Any, Generator, Callable, Type, Protocol

import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vesuvius.data_io import SaveModel
from vesuvius.trackers import Incrementer, TrackerAvg


class TrainingResources:
    def __init__(self, output_dir="output/runs"):
        self.output_dir = output_dir

    def __enter__(self):
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_subdir = os.path.join(self.output_dir, current_time)
        os.makedirs(self.log_subdir, exist_ok=True)

        log_subdir = os.path.join("output/runs", current_time)
        self.saver = SaveModel(log_subdir, 1)
        self.writer = SummaryWriter(self.log_subdir, flush_secs=60)
        self.incrementer = Incrementer()
        self.logger_loss = TrackerAvg('loss/train', self.writer)
        self.logger_test_loss = TrackerAvg('loss/test', self.writer)
        self.logger_lr = TrackerAvg('stats/lr', self.writer)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.flush()
        self.writer.close()
        torch.cuda.empty_cache()
        gc.collect()


class OptimiserScheduler(Protocol):
    def __init__(self, model: nn.Module, learning_rate: float, total_steps: int):
        ...

    def optimizer(self) -> Optimizer:
        ...

    def scheduler(self, optimizer: Optimizer) -> Any:
        ...


class BaseTrainer(ABC):

    def __init__(self,
                 train_loader: Callable,
                 test_loader: Callable,
                 resources: TrainingResources,
                 optimizer_scheduler: Type[OptimiserScheduler],
                 criterion: Callable,
                 criterion_validate: Callable = None,
                 learning_rate: float = 0.03,
                 l1_lambda: float = 0,
                 epochs: int = 1,
                 model_kwargs: dict = None,
                 config_kwargs: dict = None) -> None:

        self.epochs = epochs

        if model_kwargs is None:
            model_kwargs = {}
        if config_kwargs is None:
            config_kwargs = {}

        if criterion_validate is None:
            criterion_validate = criterion
        self.criterion_validate = criterion_validate
        self.learning_rate = learning_rate
        self.resource = resources
        self.datapoint = self.outputs = None
        self.l1_lambda = l1_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}', '\n')

        self.config = self._default_config(**config_kwargs)

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

        self.train_loader_iter = train_loader(self.config, self.epochs)
        self.test_loader_iter = test_loader(self.config)
        self.train_loader_iter = islice(self.train_loader_iter, self.config.training_steps)

    @abstractmethod
    def forward(self) -> torch.Tensor:
        ...

    @abstractmethod
    def loss(self) -> float:
        ...

    @abstractmethod
    def _check_model(self) -> None:
        ...

    @staticmethod
    @abstractmethod
    def _default_config(**kwargs) -> dataclass:
        ...

    @abstractmethod
    def validate(self) -> None:
        ...

    def save_model_output(self):
        self.config.update_nn_kwargs(self.optimizer, self.scheduler, self.criterion, self.learning_rate, self.epochs)
        self.config['performance_dict'] = {'loss/train': self.resource.logger_loss.average,
                                           'loss/test': self.resource.logger_test_loss.average,
                                           'steps': self.resource.incrementer.count}
        self.resource.saver.model(self.model)
        json_file_path = self.resource.saver.config(self.config)
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        config_json = json.dumps(data, indent=4)
        self.resource.writer.add_text('config', config_json)

    def trainer_generator(self) -> Generator[Type['BaseTrainer'], None, None]:
        while self.resource.incrementer.loop < self.loops:
            next(self)
            yield self

    def __next__(self) -> 'BaseTrainer':
        if self.resource.incrementer.loop >= self.loops:
            raise StopIteration
        self.datapoint = next(self.train_loader_iter)
        self.resource.incrementer.increment(len(self.datapoint.label))
        return self

    def __iter__(self):
        tqdm_kwargs = dict(total=self.loops, disable=False, desc='Training', position=0)
        try:
            yield from tqdm(self.trainer_generator(), **tqdm_kwargs)
        except Exception as e:
            print("An exception occurred during training: ", e)
            raise

    def __str__(self) -> str:
        loop = self.resource.incrementer.loop
        lr = self.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"Current Loop: {loop}, Learning Rate: {lr}, Epochs: {epochs}, Batch Size: {batch_size}"

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        loop = self.resource.incrementer.loop
        lr = self.learning_rate
        epochs = self.epochs
        batch_size = self.batch_size
        return f"{classname}(current_loop={loop}, learning_rate={lr}, epochs={epochs}, batch_size={batch_size})"
