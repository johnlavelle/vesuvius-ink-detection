import random

import torch
import torchvision.transforms as transforms


class RandomFlipTransform:
    def __init__(self, p_lr: float = 0.5, p_ud: float = 0.5):
        self.p_lr = p_lr
        self.p_ud = p_ud

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_lr:
            tensor = torch.flip(tensor, dims=(-1,))  # Flip left-right
        if random.random() < self.p_ud:
            tensor = torch.flip(tensor, dims=(-2,))  # Flip up-down
        return tensor


transform1 = transforms.Compose([
    RandomFlipTransform(p_lr=0.5, p_ud=0.5)
])

