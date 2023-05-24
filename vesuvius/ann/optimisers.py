from torch import nn as nn
from torch import optim
from abc import ABC
from typing import Dict, Any


class OptimiserScheduler(ABC):
    def __init__(self, model: nn.Module, learning_rate: float, total_steps: int):
        self.model = model
        self.learning_rate = learning_rate
        self.total_steps = total_steps

    def optimizer(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def scheduler(self) -> optim.lr_scheduler:
        ...

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.__class__.__name__,
            "learning_rate": self.learning_rate,
            "total_steps": self.total_steps,
        }


class SGDOneCycleLR(OptimiserScheduler):

    def scheduler(self) -> optim.lr_scheduler:
        return optim.lr_scheduler.OneCycleLR(self.optimizer(),
                                             max_lr=self.learning_rate,
                                             div_factor=3,
                                             total_steps=self.total_steps)

    def __str__(self):
        return f'SGDOneCycleLR(model={self.model.__class__.__name__}, learning_rate={self.learning_rate}, ' \
               f'total_steps={self.total_steps})'


class AdamOneCycleLR(OptimiserScheduler):

    def scheduler(self) -> optim.lr_scheduler:
        return optim.lr_scheduler.OneCycleLR(self.optimizer(),
                                             max_lr=self.learning_rate,
                                             div_factor=3,
                                             total_steps=self.total_steps)

    def __str__(self):
        return f'AdamOneCycleLR(model={self.model.__class__.__name__}, learning_rate={self.learning_rate}, ' \
               f'total_steps={self.total_steps})'


