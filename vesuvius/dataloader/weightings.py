from typing import Iterable

from vesuvius.data_io import read_dataset_from_zarr


worker_seed = None


class WeightedSamples:
    """
    Generate balanced samples for each fragment, such that the nuber of pixels samples from each fragment is
    proportional to the number of ink pixels in the fragment.
    """

    def __init__(self, samples: int, prefix: str, fragment_keys: Iterable = (1, 2, 3), num_workers=0):
        """

        :param samples: The number of samples to generate
        :param fragment_keys:
        """
        self.fragment_keys = fragment_keys
        self.num_workers = num_workers
        self.prefix = prefix
        self.ink_sizes = None
        self.ink_sizes_total = None
        self._samples = samples
        self.ink_sizes = {i: self._get_num_of_ink_pixels_per_fragment(i) for i in self.fragment_keys}
        self.ink_sizes_total = sum(self.ink_sizes.values())

    def normalise(self, ink_sizes):
        return round(self._samples * ink_sizes / self.ink_sizes_total)

    def _get_num_of_ink_pixels_per_fragment(self, i):
        ds = read_dataset_from_zarr(i, self.num_workers, self.prefix)['labels']
        return int(ds.sum().values)
