from typing import Protocol, Any

import torch
from torch import nn as nn, optim as optim
from torch.optim import Optimizer


class OptimiserScheduler(Protocol):
    def __init__(self, model: nn.Module, learning_rate: float, total_steps: int):
        ...

    def optimizer(self) -> Optimizer:
        ...

    def scheduler(self, optimizer: Optimizer) -> Any:
        ...


class SGDOneCycleLR:
    def __init__(self, model: nn.Module, learning_rate: float, total_steps: int):
        self.model = model
        self.learning_rate = learning_rate
        self.total_steps = total_steps

    def optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def scheduler(self, optimizer: Optimizer):
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=self.total_steps)

    def __str__(self):
        return f'SGDOneCycleLR(model={self.model.__class__.__name__}, learning_rate={self.learning_rate}, ' \
               f'total_steps={self.total_steps})'
