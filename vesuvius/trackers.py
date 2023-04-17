from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter


@dataclass
class BaseTracker(ABC):
    tag: str
    summary_writer: SummaryWriter
    value: float = 0.0
    i: int = 0

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
        self.i += batch_size

    @property
    def average(self) -> float:
        try:
            return self.value / self.i
        except ZeroDivisionError:
            return 0.0

    def log(self, iteration: int) -> None:
        self.summary_writer.add_scalar(self.tag, self.average, iteration)
        self.value = 0.0
        self.i = 0
