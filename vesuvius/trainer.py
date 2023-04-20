import gc
import json
import os
import pprint
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator, Iterator, Callable
from itertools import islice

import torch
from torch import nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from memory_profiler import profile

from vesuvius.data_io import SaveModel
from vesuvius.datapoints import Datapoint
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


class BaseDataLoader(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train_loaders(self) -> Generator[Datapoint, None, None]:
        ...

    @abstractmethod
    def test_loaders(self) -> Generator[Datapoint, None, None]:
        ...


class BaseTrainer(ABC):

    def __init__(self,
                 train_loader: Callable,
                 test_loader: Callable,
                 resources: TrainingResources,
                 epochs: int = 1,
                 learning_rate: float = 0.03,
                 l1_lambda: float = 0.001,
                 compute_loss: bool = True,
                 criterion_kwargs: dict = None,
                 model_kwargs: dict = None,
                 config_kwargs: dict = None) -> None:

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.compute_loss = compute_loss
        if criterion_kwargs is None:
            criterion_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if config_kwargs is None:
            config_kwargs = {}

        self.resource = resources

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = self.get_config(**config_kwargs)

        self.model = self.config.model(**model_kwargs).to(self.device)
        self.batch_size = self.config.batch_size
        self.total = self.epochs * self.config.training_steps
        self.loops = self.total // self.batch_size
        self.check_model()
        pprint.pprint(self.config)
        print()

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer, self.loops)
        self.criterion = self.get_criterion(**criterion_kwargs)
        self.criterion_val = nn.BCEWithLogitsLoss()

        self.train_loader_iter = train_loader(self.config)
        self.test_loader_iter = test_loader(self.config)
        self.train_loader_iter = islice(self.train_loader_iter,  self.config.training_steps)
        self.test_loader_iter = islice(self.test_loader_iter, 20)

    @abstractmethod
    def get_criterion(self, **kwargs) -> nn.Module:
        ...

    @abstractmethod
    def get_scheduler(self, optimizer, total) -> Any:
        ...

    @abstractmethod
    def get_optimizer(self) -> Optimizer:
        ...

    @abstractmethod
    def forward(self, datapoint) -> torch.Tensor:
        ...

    @abstractmethod
    def check_model(self) -> None:
        ...

    @staticmethod
    @abstractmethod
    def get_config(**kwargs) -> dataclass:
        ...

    @abstractmethod
    def validate(self, i) -> None:
        ...

    @abstractmethod
    def get_loss(self, compute_loss=True) -> float:
        ...

    def loss_generator(self) -> Iterator[float]:
        while self.resource.incrementer.loop < self.loops:
            yield self.get_loss()

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

    def __next__(self) -> float:
        if self.resource.incrementer.loop >= self.loops:
            raise StopIteration
        return self.get_loss(compute_loss=self.compute_loss)

    def __iter__(self):
        tqdm_kwargs = dict(total=self.loops, disable=False, desc='Training', position=0)
        try:
            yield from tqdm(self.loss_generator(), **tqdm_kwargs)
        except Exception as e:
            print("An exception occurred during training: ", e)
            raise
        finally:
            self.save_model_output()
