from itertools import islice
from vesuvius.sampler import CropBoxRegular


def test_crop_box_regular_get_sample():
    total_bounds = (0, 0, 0, 10, 10, 10)
    width_xy = 4
    width_z = 4
    seed = 42
    crop_box = CropBoxRegular(total_bounds, width_xy, width_z, seed=seed)
    expected = [
        (0, 3, 0, 3, 0, 3),
        (0, 3, 0, 3, 4, 7),
        (0, 3, 4, 7, 0, 3),
        (0, 3, 4, 7, 4, 7),
    ]
    for i, v in enumerate(islice(crop_box, 4)):
        assert v == expected[i]
