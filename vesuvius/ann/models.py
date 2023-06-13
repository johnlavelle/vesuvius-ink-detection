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

    def forward(self, x: torch.Tensor, dummy):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class EncoderDecoderSequential(torch.nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()

        self.dropout_rate = dropout_rate

        filters = [16, 32, 64]
        paddings = [1, 1, 1]
        kernel_sizes = [3, 3, 3]
        strides = [2, 2, 2]

        layers = []
        in_channels = 1
        for num_filters, padding, kernel_size, stride in zip(filters, paddings, kernel_sizes, strides):
            layers.extend([
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
                torch.nn.BatchNorm3d(num_features=num_filters),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = num_filters
        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 1)
        )

        # self.fc_combined2 = nn.Linear(3, 1)

    def forward(self, x, dummy):
        features = self.encoder(x)
        return self.decoder(features)

    def as_dict(self):
        return {}


class EncoderDecoder(torch.nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()

        self.dropout_rate = dropout_rate

        filters = [16, 32, 64]
        paddings = [1, 1, 1]
        kernel_sizes = [3, 3, 3]
        strides = [2, 2, 2]

        self.convs = []
        self.relus = []
        self.bns = []
        self.dropouts = []
        in_channels = 1
        for i, (num_filters, padding, kernel_size, stride) in enumerate(zip(filters, paddings, kernel_sizes, strides)):
            conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            relu = nn.ReLU(inplace=True)
            bn = torch.nn.BatchNorm3d(num_features=num_filters)
            dropout = nn.Dropout(self.dropout_rate)
            setattr(self, f"conv_{i}", conv)
            setattr(self, f"relu_{i}", relu)
            setattr(self, f"bn_{i}", bn)
            setattr(self, f"dropout_{i}", dropout)
            self.convs.append(conv)
            self.relus.append(relu)
            self.bns.append(bn)
            self.dropouts.append(dropout)
            in_channels = num_filters
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_channels, 32)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout_fc1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(32, 32)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout_fc2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, dummy):
        for conv, relu, bn, dropout in zip(self.convs, self.relus, self.bns, self.dropouts):
            x = conv(x)
            x = relu(x)
            x = bn(x)
            x = self.pool(x)
            x = dropout(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout_fc2(x)
        x = self.fc3(x)

        return x

    def as_dict(self):
        return {}


import torch
from torch import nn


def gaussian(x, mu, sigma):
    y = torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return y / torch.sum(y)


class EncoderDecoderZ(torch.nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()

        cnn_width = 128

        # filters = [8, 16, 32]
        filters = [16, 32, 64]
        paddings = [1, 1, 1]
        kernel_sizes = [3, 3, 3]
        strides = [2, 2, 2]
        dropout_ratea = [dropout_rate, dropout_rate, dropout_rate]
        self.dropout_rate = 0.2

        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_channels = 1
        for i, (num_filters, padding, kernel_size, stride, dropout_rate) in \
                enumerate(zip(filters, paddings, kernel_sizes, strides, dropout_ratea)):
            conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            relu = nn.ReLU(inplace=True)
            bn = torch.nn.BatchNorm3d(num_features=num_filters)
            dropout = nn.Dropout(dropout_rate)

            self.convs.append(conv)
            self.relus.append(relu)
            self.bns.append(bn)
            self.dropouts.append(dropout)

            in_channels = num_filters

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_channels, cnn_width)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout_fc1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(cnn_width, cnn_width)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout_fc2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(cnn_width, 1)

    def forward(self, x, scalar):
        scalar += 0.001
        # scalar -= 0.5
        # scalar *= 2
        # scalar = gaussian(scalar, 0, 0.4)

        for conv, relu, bn, dropout in zip(self.convs, self.relus, self.bns, self.dropouts):
            x = conv(x)
            x = relu(x)
            x = bn(x)
            x = self.pool(x)
            x = dropout(x)

            # Condition the output of each layer by scaling it with the scalar
            # Here, we need to reshape the scalar to match the shape of x
            scalar = scalar.view(-1, 1, 1, 1, 1)
            x = x * scalar

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        # x = self.dropout_fc2(x)
        # x = self.fc3(x)

        return x

    def as_dict(self):
        return {}


class HybridBinaryClassifier(nn.Module):
    def __init__(self, dropout_rate: float = 0.3, width: int = 4):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.width = width

        self.conv1 = nn.Conv3d(1, self.width, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(self.width)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.conv2 = nn.Conv3d(self.width, 2 * self.width, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(2 * self.width)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.conv3 = nn.Conv3d(2 * self.width, 4 * self.width, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(4 * self.width)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.flatten = nn.Flatten(start_dim=1)

        self.linear1 = nn.Linear(784, 2)

        # FCN part for scalar input
        # self.fc_scalar = nn.Linear(1, 2)
        # self.bn_scalar = nn.BatchNorm1d(4)
        # self.dropout_scalar = nn.Dropout(self.dropout_rate)

        # Combined layers (initialized later)
        self.fc_combined1 = nn.Linear(3, 3)
        # self.bn_combined1 = nn.BatchNorm1d(4)

        self.fc_combined2 = nn.Linear(3, 1)

        # self.dropout_combined = nn.Dropout(self.dropout_rate)

        # self.fc_combined3 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, scalar_input: torch.Tensor):
        # Scale to between zero and one
        scalar_input -= 0.5
        scalar_input *= 2

        # CNN part
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = F.relu(x)

        x = self.linear1(x)
        x = F.relu(x)

        # # FCN part
        # scalar_input = self.fc_scalar(scalar_input)
        # # scalar_input = self.bn_scalar(scalar_input)
        # scalar_input = F.relu(scalar_input)

        # Combine CNN and FCN outputs
        combined = torch.cat((x, scalar_input), dim=1)

        # Combined layers
        x = self.fc_combined1(combined)
        # x = self.bn_combined1(x)
        x = F.relu(x)

        x = self.fc_combined2(x)
        x = F.relu(x)

        # x = self.dropout_combined(x)
        # x = self.fc_combined3(x)

        return x

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
            'width': self.width
        }


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


# class HybridBinaryClassifierShallow(nn.Module):
#     def __init__(self, dropout_rate: float = 0.5, width_multiplier: int = 1):
#         super().__init__()
#         self.dropout_rate = dropout_rate
#         self.width_multiplier = width_multiplier
#
#         self.conv1 = nn.Conv3d(1, self.width_multiplier, 5, 1, 0)
#         self.bn1 = nn.BatchNorm3d(self.width_multiplier)
#         self.dropout1 = nn.Dropout(self.dropout_rate)
#
#         self.pool1 = nn.AvgPool3d((1, 11, 11))
#         self.flatten = nn.Flatten(start_dim=1)
#
#         # FCN part for scalar input
#         self.fc_scalar = nn.Linear(1, 13)
#         self.bn_scalar = nn.BatchNorm1d(13)
#         self.dropout_scalar = nn.Dropout(self.dropout_rate)
#
#         # Combined layers (initialized later)
#         self.fc_combined1 = nn.Linear(62, 8)
#         self.bn_combined = nn.BatchNorm1d(8)
#         self.dropout_combined = nn.Dropout(self.dropout_rate)
#
#         self.fc_combined2 = nn.Linear(8, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x: torch.Tensor, scalar_input: torch.Tensor):
#         # CNN part
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.dropout1(x)
#
#         x = self.flatten(x)
#
#         # FCN part
#         x_scalar = self.fc_scalar(scalar_input)
#         x_scalar = self.bn_scalar(x_scalar)
#         scalar_out = F.relu(x_scalar)
#
#         # Combine CNN and FCN outputs
#         combined = torch.cat((x, scalar_out), dim=1)
#
#         # Combined layers
#         x = self.fc_combined1(combined)
#         x = self.bn_combined(x)
#         x = F.relu(x)
#         # x = self.dropout_combined(x)
#         x = self.fc_combined2(x)
#
#         return F.relu(x)
#
#     @property
#     def requires_grad(self):
#         return all(param.requires_grad for param in self.parameters())
#
#     @requires_grad.setter
#     def requires_grad(self, value: bool):
#         for param in self.parameters():
#             param.requires_grad = value
#
#     def as_dict(self):
#         return {
#             'dropout_rate': self.dropout_rate,
#             'width_multiplier': self.width_multiplier
#         }


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
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
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


class SelfAttention(nn.Module):
    def __init__(self, in_dim, window_size=13):
        super().__init__()
        self.query_conv = nn.Conv3d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv3d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv3d(in_dim, in_dim, 1)
        self.window_size = window_size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, depth, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, depth*height*width)
        key = self.key_conv(x).view(batch_size, -1, depth*height*width)
        value = self.value_conv(x).view(batch_size, -1, depth*height*width)
        out = torch.zeros_like(value)
        for i in range(0, depth*height*width, self.window_size):
            j = min(i + self.window_size, depth*height*width)
            q = query[:, :, i:j]
            k = key[:, :, i:j]
            v = value[:, :, i:j]
            attention = self.softmax(torch.bmm(q.permute(0, 2, 1), k))
            out[:, :, i:j] = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, depth, height, width)
        return out


class CNNWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.attention = SelfAttention(64)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.LazyLinear(128)
        self.relu = nn.ReLU()  # Use a single ReLU instance
        self.fc2 = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.attention(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class StackingClassifierShallow(nn.Module):
    def __init__(self, input_size: int, width: int = 4, dropout_rate=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.input_size = input_size
        self.width = width
        self.fc1 = nn.Linear(input_size, self.width)
        self.fc2 = nn.Linear(self.width, 1)
        self.dropout2 = nn.Dropout(self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def as_dict(self):
        return {
            'input_size': self.input_size,
            'width_multiplier': self.width
        }


class StackingMax(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.max(x)

    @staticmethod
    def as_dict():
        return {}


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.0):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 5)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(5, 1)
        self.max_pool1 = nn.MaxPool1d(3)

    def forward(self, x):
        # x = self.max_pool1(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.linear2(x)
        # x = torch.relu(x)
        return x
