"""Metrics evaluated against original hard labels."""

import math

import torch
from torch.nn import functional as F


def expected_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10
) -> float:
    """Compute equal-width-bin expected calibration error."""
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels).to(torch.float32)
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    ece = torch.zeros((), device=probabilities.device)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            weight = in_bin.to(torch.float32).mean()
            ece += weight * (correct[in_bin].mean() - confidence[in_bin].mean()).abs()
    return ece.item()


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """Return all pre-specified hard-label metrics and diagnostics."""
    probabilities = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)
    one_hot = F.one_hot(labels, num_classes=2).to(torch.float32)
    margins = (logits[:, 0] - logits[:, 1]).abs()
    values = {
        "accuracy": predictions.eq(labels).to(torch.float32).mean().item(),
        "nll": F.cross_entropy(logits, labels).item(),
        "brier": ((probabilities - one_hot) ** 2).sum(dim=1).mean().item(),
        "ece": expected_calibration_error(probabilities, labels),
        "mean_confidence": probabilities.max(dim=1).values.mean().item(),
        "logit_margin": margins.mean().item(),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("all metrics must be finite")
    return values
