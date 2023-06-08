import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import *


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def get_focal_weights(self, prob: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss weights."""
        return (1 - prob) ** self.gamma

    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        # Apply the sigmoid function to get probabilities
        prob = torch.sigmoid(input)

        # Compute the binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # Get the focal loss weights
        focal_weights = self.get_focal_weights(prob)

        # Apply the focal weights
        focal_loss = self.alpha * focal_weights * bce_loss

        # Reduce the loss (mean, sum or none)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def __str__(self):
        return f'FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction})'
