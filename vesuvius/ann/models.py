import torch
from torch import nn as nn
from torch.nn import functional as F


class CNN1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.LazyLinear(128)
        self.relu = nn.ReLU()  # Use a single ReLU instance
        self.fc2 = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def cnn1_sequential():
    return nn.Sequential(
        nn.Conv3d(1, 16, 3, 1, 1),
        nn.MaxPool3d(2, 2),
        nn.Conv3d(16, 32, 3, 1, 1),
        nn.MaxPool3d(2, 2),
        nn.Conv3d(32, 64, 3, 1, 1),
        nn.MaxPool3d(2, 2),
        nn.Flatten(start_dim=1),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.LazyLinear(1),
        nn.Sigmoid())


class HybridBinaryClassifier(nn.Module):
    def __init__(self, dropout_rate: float = 0.3, width: int = 4):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.width = width

        self.conv1 = nn.Conv3d(1, self.width, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(self.width)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.pool1 = nn.MaxPool3d(2, 2)

        self.conv2 = nn.Conv3d(self.width, 2 * self.width, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(2 * self.width)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.pool2 = nn.MaxPool3d(2, 2)

        self.conv3 = nn.Conv3d(2 * self.width, 4 * self.width, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(4 * self.width)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        self.pool3 = nn.AdaptiveMaxPool3d((8, 8, 8))

        self.flatten = nn.Flatten(start_dim=1)

        # FCN part for scalar input
        self.fc_scalar = nn.Linear(1, 16)
        self.bn_scalar = nn.BatchNorm1d(16)
        self.dropout_scalar = nn.Dropout(self.dropout_rate)

        # Combined layers (initialized later)
        self.fc_combined1 = nn.Linear(4 * 8 * 8 * 8 * self.width + 16, 128)
        self.bn_combined = nn.BatchNorm1d(128)
        self.dropout_combined = nn.Dropout(self.dropout_rate)

        self.fc_combined2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, scalar_input: torch.Tensor):
        # CNN part
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        # FCN part
        x_scalar = self.fc_scalar(scalar_input)
        x_scalar = self.bn_scalar(x_scalar)
        scalar_out = F.relu(x_scalar)

        # Combine CNN and FCN outputs
        combined = torch.cat((x, scalar_out), dim=1)

        # Combined layers
        x = self.fc_combined1(combined)
        x = self.bn_combined(x)
        x = F.relu(x)
        x = self.dropout_combined(x)
        x = self.fc_combined2(x)

        return self.sigmoid(x)

    @property
    def requires_grad(self):
        return all(param.requires_grad for param in self.parameters())

    @requires_grad.setter
    def requires_grad(self, value: bool):
        for param in self.parameters():
            param.requires_grad = value

    def as_dict(self):
        return {
            'dropout_rate': self.dropout_rate,
            'width_multiplier': self.width
        }


class HybridBinaryClassifierShallow(nn.Module):
    def __init__(self, dropout_rate: float = 0.3, width: int = 4):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.width = width

        self.conv1 = nn.Conv3d(1, self.width, 5, 1, 2)
        self.bn1 = nn.BatchNorm3d(self.width)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.pool1 = nn.AvgPool3d(4, 4)

        self.flatten = nn.Flatten(start_dim=1)

        # FCN part for scalar input
        self.fc_scalar = nn.Linear(1, 16)
        self.bn_scalar = nn.BatchNorm1d(16)
        self.dropout_scalar = nn.Dropout(self.dropout_rate)

        # Combined layers (initialized later)
        self.fc_combined1 = nn.Linear(500, 128)
        self.bn_combined = nn.BatchNorm1d(128)
        self.dropout_combined = nn.Dropout(self.dropout_rate)

        self.fc_combined2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, scalar_input: torch.Tensor):
        # CNN part
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.flatten(x)

        # FCN part
        x_scalar = self.fc_scalar(scalar_input)
        x_scalar = self.bn_scalar(x_scalar)
        scalar_out = F.relu(x_scalar)

        # Combine CNN and FCN outputs
        combined = torch.cat((x, scalar_out), dim=1)

        # Combined layers
        x = self.fc_combined1(combined)
        x = self.bn_combined(x)
        x = F.relu(x)
        x = self.dropout_combined(x)
        x = self.fc_combined2(x)

        return self.sigmoid(x)

    @property
    def requires_grad(self):
        return all(param.requires_grad for param in self.parameters())

    @requires_grad.setter
    def requires_grad(self, value: bool):
        for param in self.parameters():
            param.requires_grad = value

    def as_dict(self):
        return {
            'dropout_rate': self.dropout_rate,
            'width_multiplier': self.width
        }


class HybridBinaryClassifierShallow2(HybridBinaryClassifierShallow):
    def __init__(self, dropout_rate: float = 0.3, width: int = 4):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.width = width

        self.conv1 = nn.Conv3d(1, self.width, 5, 1, 2)
        self.bn1 = nn.BatchNorm3d(self.width)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.pool1 = nn.AvgPool3d(4, 4)

        self.flatten = nn.Flatten(start_dim=1)

        self.fc_scalar = nn.Linear(1, 32)
        self.bn_scalar = nn.BatchNorm1d(32)
        self.dropout_scalar = nn.Dropout(self.dropout_rate)

        # Adjusted input size
        self.fc_combined1 = nn.Linear(516, 256)
        self.bn_combined = nn.BatchNorm1d(256)
        self.dropout_combined = nn.Dropout(self.dropout_rate)

        self.fc_combined2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()


class BinaryClassifierShallowNoZ(nn.Module):
    def __init__(self, dropout_rate: float = 0.3, width: int = 4):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.width = width

        self.conv1 = nn.Conv3d(1, self.width, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(self.width)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.pool1 = nn.MaxPool3d(2, 2)

        self.flatten = nn.Flatten(start_dim=1)

        self.fc_combined2 = nn.Linear(4050, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, scalar_input: torch.Tensor):
        # CNN part
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.flatten(x)

        # Final FC layer
        x = self.fc_combined2(x)

        return self.sigmoid(x)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.input_layer = None

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        if self.input_layer is None:
            self.input_layer = nn.Linear(x.shape[1], 64)
            self.input_layer.to(x.device)
        x = self.input_layer(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size, dropout_rate=0.0):
        super().__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def as_dict(self):
        return {
            'input_size': self.input_size,
            'dropout_rate': self.dropout_rate
        }


class StackingClassifier(nn.Module):
    def __init__(self, input_size, width_multiplier=1):
        super(StackingClassifier, self).__init__()
        self.width_multiplier = width_multiplier
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

    def as_dict(self):
        return {
            'width_multiplier': self.width_multiplier
        }


class StackingClassifierShallow(nn.Module):
    def __init__(self, input_size: int, width_multiplier: int = 1):
        super().__init__()
        self.width_multiplier = width_multiplier
        self.fc1 = nn.Linear(input_size, 16 * self.width_multiplier)
        self.fc2 = nn.Linear(16 * self.width_multiplier, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def as_dict(self):
        return {
            'width_multiplier': self.width_multiplier
        }
