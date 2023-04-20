import torch
import torch.nn as nn
import torch.nn.functional as F

from training import FocalLoss


def test_get_focal_weights():
    focal_loss = FocalLoss()
    prob = torch.tensor([0.2, 0.6, 0.9])
    expected_weights = torch.tensor([(1 - 0.2) ** 2, (1 - 0.6) ** 2, (1 - 0.9) ** 2])
    weights = focal_loss.get_focal_weights(prob)

    assert torch.allclose(weights, expected_weights)


def test_focal_loss_mean_reduction():
    focal_loss = FocalLoss(reduction='mean')
    input = torch.tensor([0.5, -1.0, 1.5])
    target = torch.tensor([1.0, 0.0, 1.0])

    prob = torch.sigmoid(input)
    bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    focal_weights = focal_loss.get_focal_weights(prob)
    expected_loss = (focal_loss.alpha * focal_weights * bce_loss).mean()

    loss = focal_loss(input, target)

    assert torch.allclose(loss, expected_loss)


def test_focal_loss_sum_reduction():
    focal_loss = FocalLoss(reduction='sum')
    input = torch.tensor([0.5, -1.0, 1.5])
    target = torch.tensor([1.0, 0.0, 1.0])

    prob = torch.sigmoid(input)
    bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    focal_weights = focal_loss.get_focal_weights(prob)
    expected_loss = (focal_loss.alpha * focal_weights * bce_loss).sum()

    loss = focal_loss(input, target)

    assert torch.allclose(loss, expected_loss)


def test_focal_loss_none_reduction():
    focal_loss = FocalLoss(reduction='none')
    input = torch.tensor([0.5, -1.0, 1.5])
    target = torch.tensor([1.0, 0.0, 1.0])

    prob = torch.sigmoid(input)
    bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    focal_weights = focal_loss.get_focal_weights(prob)
    expected_loss = focal_loss.alpha * focal_weights * bce_loss

    loss = focal_loss(input, target)

    assert torch.allclose(loss, expected_loss)


def test_focal_loss_equivalence_to_bce_with_logits_loss():
    focal_loss = FocalLoss(alpha=1, gamma=0, reduction='none')
    bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')

    input = torch.tensor([0.5, -1.0, 1.5])
    target = torch.tensor([1.0, 0.0, 1.0])

    focal_loss_result = focal_loss(input, target)
    bce_with_logits_loss_result = bce_with_logits_loss(input, target)

    assert torch.allclose(focal_loss_result, bce_with_logits_loss_result)
