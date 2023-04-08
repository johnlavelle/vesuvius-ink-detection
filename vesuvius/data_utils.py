from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter


@dataclass
class BaseTracker(ABC):
    tag: str
    summary_writer: SummaryWriter = field(default_factory=SummaryWriter)
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

    def log(self, iteration: int) -> None:
        avg = self.value / self.i
        self.summary_writer.add_scalar(self.tag, avg, iteration)
        self.value = 0.0
        self.i = 0


Datapoint = namedtuple("Datapoint", "voxels label fragment x_start x_stop y_start y_stop z_start z_stop")
