"""Model definition."""

from torch import nn


def make_model() -> nn.Linear:
    """Return logistic regression with the protocol's two output logits."""
    return nn.Linear(784, 2)

