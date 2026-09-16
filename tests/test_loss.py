import torch
from torch.nn import functional as F

from label_smoothing.losses import label_smoothed_cross_entropy, smoothed_targets


def test_smoothed_targets_sum_to_one() -> None:
    targets = smoothed_targets(torch.tensor([0, 1]), epsilon=0.1)
    assert torch.allclose(targets.sum(dim=1), torch.ones(2))
    assert torch.allclose(targets, torch.tensor([[0.95, 0.05], [0.05, 0.95]]))


def test_zero_smoothing_matches_cross_entropy() -> None:
    logits = torch.tensor([[2.0, -0.5], [0.1, 0.9]])
    labels = torch.tensor([0, 1])
    actual = label_smoothed_cross_entropy(logits, labels, epsilon=0.0)
    assert torch.allclose(actual, F.cross_entropy(logits, labels))

