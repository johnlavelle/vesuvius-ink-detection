import torch
import torchvision.transforms as transforms
from torch.nn import Module


class RandomFlipTransform(Module):
    def __init__(self, p_lr: float = 0.5, p_ud: float = 0.5):
        super().__init__()
        self.p_lr = p_lr
        self.p_ud = p_ud

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_lr:
            tensor = torch.flip(tensor, dims=(-1,))  # Flip left-right
        if torch.rand(1).item() < self.p_ud:
            tensor = torch.flip(tensor, dims=(-2,))  # Flip up-down
        return tensor


class IdentityTransform:
    def __call__(self, x):
        return x


class NormalizeVolume(Module):

    @staticmethod
    def forward(volume):
        return (volume - volume.mean()) / volume.std()


# transform_train = transforms.Compose([
#     RandomFlipTransform(p_lr=0.5, p_ud=0.5)
#     # NormalizeVolume()
# ])


transform_train = transforms.Compose([
    RandomFlipTransform(p_lr=0.5, p_ud=0.5),
    # NormalizeVolume()
])


transform_val = transforms.Compose([
    IdentityTransform(),
    # NormalizeVolume()
])
