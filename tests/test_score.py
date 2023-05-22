import numpy as np

from vesuvius.metric import f0_5_score


def test_f0_5_score():
    # Test case 1: identical images
    output_image = np.array([1, 1, 0, 0])
    reference_image = np.array([1, 1, 0, 0])
    score = f0_5_score(output_image, reference_image)
    assert score == 1.0, "Test case 1 failed"

    # Test case 2: completely different images
    output_image = np.array([1, 1, 0, 0])
    reference_image = np.array([0, 0, 1, 1])
    score = f0_5_score(output_image, reference_image)
    assert score == 0.0, "Test case 2 failed"

    # Test case 3: some overlap
    output_image = np.array([1, 1, 0, 0])
    reference_image = np.array([1, 0, 0, 1])
    score = f0_5_score(output_image, reference_image)
    assert 0 < score < 1, "Test case 3 failed"
