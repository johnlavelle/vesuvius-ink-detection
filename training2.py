from typing import Generator, Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from vesuvius.config import Configuration2
from vesuvius.trainer import BaseTrainer


# Define the model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


# Set the input dimension based on your data
input_dim = 10
model = BinaryClassifier(input_dim)

# Example usage
input_tensor = torch.randn(1, input_dim)  # Replace this with your input tensor
output = model(input_tensor)
print(output)


class Trainer2(BaseTrainer):

    def __int__(self):
        super().__init__()

    @staticmethod
    def get_config(**kwargs) -> Configuration2:
        return Configuration2(
            model=BinaryClassifier
        )

    def get_criterion(self, **kwargs) -> nn.Module:
        ...

    def get_scheduler(self, optimizer, total) -> Any:
        ...

    def get_optimizer(self) -> Optimizer:
        ...

    def apply_forward(self, datapoint) -> torch.Tensor:
        ...

    def check_model(self) -> None:
        ...

    def train_loaders(self) -> Generator:
        ...

    def test_loaders(self) -> Generator:
        ...

    def validate(self, i) -> None:
        ...

    def forward(self, compute_loss=True) -> float:
        ...


if __name__ == '__main__':
    with Trainer2() as trainer2:
        list(trainer2)
    print(trainer2)
