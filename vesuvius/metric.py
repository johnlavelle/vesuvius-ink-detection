"""
 The Sørensen–Dice coefficient in Python, where instead of using the F1 score, we are using the F0.5 score.
 The F0.5 score is given by:

\frac{(1 + \beta^2) pr}{\beta^2 p+r}\ \ \mathrm{where}\ \ p = \frac{tp}{tp+fp},\ \ r = \frac{tp}{tp+fn},\ \beta = 0.5

The Sørensen-Dice coefficient is a similarity measure between two sets, given by:

\frac{2|A \cap B|}{|A| + |B|}

For this case, we will implement a function that calculates the Sørensen-Dice coefficient based on the F0.5 score,
which is a variant of the F1 score that puts more emphasis on precision. Here's the implementation in Python:
"""

import numpy as np
from numpy import ndarray


def f0_5_score(output_image: ndarray, reference_image: ndarray) -> float:
    # Flatten the images
    output_image = output_image.flatten()
    reference_image = reference_image.flatten()

    # Calculate true positives (tp), false positives (fp), and false negatives (fn)
    tp = np.sum(np.logical_and(output_image == 1, reference_image == 1))
    fp = np.sum(np.logical_and(output_image == 1, reference_image == 0))
    fn = np.sum(np.logical_and(output_image == 0, reference_image == 1))

    # If there are no true positives, return 0
    if tp == 0:
        return 0.0

    # Calculate precision (p) and recall (r)
    p = tp / (tp + fp)
    r = tp / (tp + fn)

    # Define beta
    beta = 0.5

    # Calculate F0.5 score
    f0_5 = (1 + beta ** 2) * (p * r) / ((beta ** 2 * p) + r)

    return f0_5
