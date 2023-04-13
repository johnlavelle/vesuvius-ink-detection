from abc import ABC, abstractmethod

import xarray as xr

from vesuvius.config import Configuration
from vesuvius.data_io import read_dataset_from_zarr
from vesuvius.scroll_dataset import BaseDataset
from vesuvius.utils import get_hold_back_mask
from .weightings import WeightedSamples

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class DatasetIter(ABC):

    def __init__(self, config):
        self.config = config

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> xr.Dataset:
        pass


class XarrayDatasetIter(DatasetIter):

    def __init__(self, config: Configuration):
        super().__init__(config)
        self.fragments = iter(config.fragments)
        self.test_box_fragment = config.test_box_fragment

    def __next__(self) -> xr.Dataset:
        try:
            fragment = next(self.fragments)
        except StopIteration:
            raise StopIteration

        data = read_dataset_from_zarr(str(fragment), self.config.num_workers, self.config.prefix)
        data.attrs['fragment'] = fragment

        if fragment == self.test_box_fragment:
            data['full_mask'] = get_hold_back_mask(data, self.config.test_box)
        else:
            data['full_mask'] = data['mask']

        return data


class TorchDatasetIter(DatasetIter):
    """
    This class is used to iterate over the slices of a volume dataset.
    This done to reduce the memory footprint of the dataset.
    """

    def __init__(self, cfg: Configuration, datasets: DatasetIter):
        super().__init__(cfg)
        self.training_steps = cfg.training_steps
        self.box_width_sample = cfg.box_width_xy
        self.dataset_class = cfg.volume_dataset_cls
        self.label_operation = cfg.label_fn
        self.fragments = cfg.fragments
        self.prefix = cfg.prefix

        self.z_box_width = cfg.box_width_z
        self.crop_box_cls = cfg.crop_box_cls
        self.box_width_sample = cfg.box_width_xy

        self.label_operation = cfg.label_fn
        self.samples = cfg.training_steps
        self.stride_xy = cfg.stride_xy
        self.stride_z = cfg.stride_z

        self.datasets = datasets

        self.samples_handler = WeightedSamples(self.samples,
                                               self.prefix,
                                               self.fragments,
                                               num_workers=cfg.num_workers)
        self.current_ds = None
        self.balance_ink = cfg.balance_ink

    def __iter__(self):
        return self

    def __next__(self) -> BaseDataset:
        try:
            self.current_ds = next(self.datasets)
        except StopIteration:
            raise StopIteration

        ink_count = self.current_ds.labels.sum().compute().item()
        samples = self.samples_handler.normalise(ink_count)
        return self.dataset_class(self.current_ds,
                                  self.box_width_sample,
                                  self.z_box_width,
                                  samples,
                                  transformer=None,
                                  crop_cls=self.crop_box_cls,
                                  label_operation=self.label_operation,
                                  balance_ink=self.balance_ink,
                                  stride_xy=self.stride_xy,
                                  stride_z=self.stride_z)
