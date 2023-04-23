# import pprint
# from functools import partial
#
# import dask
# import torch
# from torch.nn import BCEWithLogitsLoss
#
# import tensorboard_access
# from vesuvius.ann.models import BinaryClassifier
# from vesuvius.ann.optimisers import SGDOneCycleLR
# from vesuvius.config import Configuration2
# from vesuvius.dataloader import get_train_loader_regular_z
# from vesuvius.trackers import Track
# from vesuvius.trainer import BaseTrainer
# from vesuvius.utils import timer
#
#
# # Define the model
#
#
# # Set the input dimension based on your data
# input_dim = 10
# model = BinaryClassifier()
#
# # Example usage
# input_tensor = torch.randn(1, input_dim)  # Replace this with your input tensor
# output = model(input_tensor)
# print(output)
#
#
# class Trainer2(BaseTrainer):
#
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#
#     @staticmethod
#     def configuration(**kwargs) -> Configuration2:
#         default_config = dict(model=BinaryClassifier,
#                               batch_size=32)
#         return Configuration2(**{**default_config, **kwargs})
#
#     def validate(self) -> 'BaseTrainer':
#         ...
#
#     def forward(self) -> 'BaseTrainer':
#         self.model.train1()
#         self.optimizer.zero_grad()
#         self.outputs = self._apply_forward(self.datapoint)
#         return self
#
#     def loss(self) -> 'BaseTrainer':
#         target = self.datapoint.label.float().to(self.device)
#         loss = self.criterion(self.outputs, target)
#         loss.backward()
#         self.optimizer.step()
#         self.scheduler.step()
#
#         batch_size = len(self.datapoint.voxels)
#         self.trackers.logger_loss.update(loss.item(), batch_size)
#         self.trackers.logger_lr.update(self.scheduler.get_last_lr()[0], batch_size)
#         return self
#
#     def dummy_input(self) -> torch.Tensor:
#         return torch.randn(self.config.batch_size, 10).to(self.device)
#
#
# if __name__ == '__main__':
#
#     pp = pprint.PrettyPrinter(indent=4)
#     dask.config.set(scheduler='synchronous')
#     print('Tensorboard URL: ', tensorboard_access.get_public_url(), '\n')
#
#     CACHED_DATA = True
#     FORCE_CACHE_RESET = False  # Deletes cache. Only used if CACHED_DATA is True.
#     EPOCHS = 2
#     RESET_CACHE_EPOCH_INTERVAL = EPOCHS
#     VALIDATE_INTERVAL = 100
#     LOG_INTERVAL = 10
#
#     train_loaders = partial(
#         get_train_loaders,
#         cached_data=CACHED_DATA,
#         force_cache_reset=FORCE_CACHE_RESET,
#         reset_cache_epoch_interval=RESET_CACHE_EPOCH_INTERVAL)
#
#     with Track() as trackers, timer("Training"):
#         trainer1 = Trainer2(get_train_loader_regular_z,
#                             None,
#                             trackers,
#                             SGDOneCycleLR,
#                             BCEWithLogitsLoss,
#                             BCEWithLogitsLoss,
#                             learning_rate=0.03,
#                             l1_lambda=0,
#                             epochs=EPOCHS,
#                             config_kwargs=dict(training_steps=32 * (40000 // 32) - 1))
