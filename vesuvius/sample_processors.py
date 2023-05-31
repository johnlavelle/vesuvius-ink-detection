from vesuvius.datapoints import DatapointTuple, Datapoint
from vesuvius.fragment_dataset import BaseDataset


fragment_to_int = {'a': 1, 'b': 2}


class SampleXYZ(BaseDataset):

    def get_datapoint(self, index: int) -> DatapointTuple:
        current_slice = self.get_volume_slice(index)
        slice_xy = {key: value for key, value in current_slice.items() if key in ('x', 'y')}
        label = self.ds.labels.sel(**slice_xy).transpose('x', 'y')
        voxels = self.ds.images.sel(**current_slice).transpose('z', 'x', 'y')
        voxels = voxels.expand_dims('Cin')
        # voxels = self.normalise_voxels(voxels)
        # voxels = voxels.expand_dims('Cin')
        s = current_slice
        dp = Datapoint(voxels, int(self.label_operation(label)), fragment_to_int[self.ds.fragment],
                       s['x'].start, s['x'].stop, s['y'].start, s['y'].stop, s['z'].start, s['z'].stop)
        return dp.to_namedtuple()
