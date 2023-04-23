"""
 The Sørensen–Dice coefficient in Python, where instead of using the F1 score, we are using the F0.5 score.
 The F0.5 score is given by:

\frac{(1 + \beta^2) pr}{\beta^2 p+r}\ \ \mathrm{where}\ \ p = \frac{tp}{tp+fp},\ \ r = \frac{tp}{tp+fn},\ \beta = 0.5

The Sørensen-Dice coefficient is a similarity measure between two sets, given by:

\frac{2|A \cap B|}{|A| + |B|}

For this case, we will implement a function that calculates the Sørensen-Dice coefficient based on the F0.5 score,
which is a variant of the F1 score that puts more emphasis on precision. Here's the implementation in Python:
"""


def f0_5_score(tp, fp, fn):
    beta = 0.5
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f0_5 = (1 + beta ** 2) * (p * r) / (beta ** 2 * p + r)
    return f0_5


def sorensen_dice_coefficient(set1, set2):
    intersection = set(set1).intersection(set(set2))
    tp = len(intersection)
    fp = len(set1) - tp
    fn = len(set2) - tp

    f0_5 = f0_5_score(tp, fp, fn)
    return f0_5


# Example usage:
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}
print(sorensen_dice_coefficient(set1, set2))
