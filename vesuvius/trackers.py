from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter


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
        # if self.ignore:
        #     self.ignore = False
        # else:
        self.summary_writer.add_scalar(self.tag, self.average, iteration)
        self.value = 0.0
        self.count = 0
