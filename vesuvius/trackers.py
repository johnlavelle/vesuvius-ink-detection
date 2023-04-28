import gc
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.utils.tensorboard import SummaryWriter

from vesuvius.data_io import SaveModel


class Incrementer:
    def __init__(self, start=0):
        self._counter = start
        self._loop = start

    def increment(self, batch_size=1):
        self._counter += batch_size
        self._loop += 1

    @property
    def count(self):
        return self._counter

    @property
    def loop(self):
        return self._loop

    def __str__(self):
        return f"Datapoints: {self._counter}. Loops: {self._loop}"

    def __repr__(self):
        return f"Datapoints: {self._counter}. Loops: {self._loop}"


@dataclass
class BaseTracker(ABC):
    tag: str
    summary_writer: SummaryWriter
    value: float = 0.0
    count: int = 0
    ignore: bool = True

    @abstractmethod
    def update(self, loss: float, batch_size: int) -> None:
        ...

    @abstractmethod
    def log(self, iteration: int) -> None:
        ...


@dataclass
class TrackerAvg(BaseTracker):

    def update(self, loss: float, batch_size: int) -> None:
        self.value += loss * batch_size
        self.count += batch_size

    @property
    def average(self) -> float:
        try:
            return self.value / self.count
        except ZeroDivisionError:
            return 0.0

    def log(self, iteration: int) -> None:
        self.summary_writer.add_scalar(self.tag, self.average, iteration)
        self.value = 0.0
        self.count = 0


class Track:
    def __init__(self, output_dir="output/runs"):
        self.output_dir = output_dir

    def __enter__(self):
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_subdir = os.path.join(self.output_dir, current_time)
        os.makedirs(self.log_subdir, exist_ok=True)

        self.incrementer = Incrementer()

        log_subdir = os.path.join("output/runs", current_time)
        self.saver = SaveModel(log_subdir, 1)
        self.writer = SummaryWriter(self.log_subdir, flush_secs=60)
        self.logger_loss = TrackerAvg('loss/train', self.writer)
        self.logger_test_loss = TrackerAvg('loss/test', self.writer)
        self.logger_lr = TrackerAvg('stats/lr', self.writer)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.flush()
        self.writer.close()
        torch.cuda.empty_cache()
        gc.collect()

    def update_train(self, loss, batch_size=1):
        self.logger_loss.update(loss, batch_size)

    def update_test(self, loss, batch_size=1):
        self.logger_test_loss.update(loss, batch_size)

    def update_lr(self, lr, batch_size=1):
        self.logger_lr.update(lr, batch_size)

    def log_test(self):
        self.logger_test_loss.log(self.incrementer.count)
        self.logger_lr.log(self.incrementer.count)

    def log_train(self):
        self.logger_loss.log(self.incrementer.count)

    def increment(self, batch_size=1):
        self.incrementer.increment(batch_size)
