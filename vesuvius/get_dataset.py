# import dask.config
# from torch.utils.data import ConcatDataset, DataLoader
# from vesuvius.fragment_dataset2 import ChunkedDataset
# import torch
# from os import path
# from dask.distributed import Client
#
#
# if __name__ == '__main__':
#     cache_filename = '/data/kaggle/input/vesuvius-challenge-ink-detection/train/dataset_cache.pt'
#     if not path.exists(cache_filename):
#         with Client(n_workers=8) as client:
#             print(client.dashboard_link)
#             ds1 = ChunkedDataset(1)
#             ds2 = ChunkedDataset(2)
#             ds3 = ChunkedDataset(3)
#             combined_dataset = ConcatDataset([ds1, ds2, ds3])
#             torch.save(combined_dataset, cache_filename)
#
#     dask.config.set(scheduler='synchronous')
#     combined_dataset = torch.load(cache_filename)
#     train_loader = DataLoader(combined_dataset, batch_size=16, num_workers=1, shuffle=True)
#     train_loader = iter(train_loader)
#     for i in range(100):
#         next(train_loader)
#         print(i)
