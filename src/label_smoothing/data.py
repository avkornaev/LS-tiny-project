"""Data loading and the single fixed train/validation split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST


@dataclass(frozen=True)
class DatasetSplits:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset


def filter_and_map(images: torch.Tensor, labels: torch.Tensor) -> TensorDataset:
    """Keep digits 3 and 7, scale pixels, flatten images, and map to 0 and 1."""
    mask = (labels == 3) | (labels == 7)
    selected_images = images[mask].to(torch.float32).reshape(-1, 784) / 255.0
    selected_labels = (labels[mask] == 7).to(torch.long)
    return TensorDataset(selected_images, selected_labels)


def stratified_indices(
    labels: torch.Tensor, validation_fraction: float = 0.2, split_seed: int = 2026
) -> tuple[np.ndarray, np.ndarray]:
    """Return one deterministic, stratified set of train and validation indices."""
    indices = np.arange(len(labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=split_seed,
        stratify=labels.numpy(),
    )
    return np.sort(train_indices), np.sort(validation_indices)


def load_mnist(data_dir: Path) -> DatasetSplits:
    """Download MNIST and construct the fixed split without inspecting test data."""
    raw_train = MNIST(root=data_dir, train=True, download=True)
    filtered_train = filter_and_map(raw_train.data, raw_train.targets)
    images, labels = filtered_train.tensors
    train_indices, validation_indices = stratified_indices(labels)

    # The official test set is transformed only for final evaluation, never split.
    raw_test = MNIST(root=data_dir, train=False, download=True)
    test = filter_and_map(raw_test.data, raw_test.targets)
    return DatasetSplits(
        train=TensorDataset(images[train_indices], labels[train_indices]),
        validation=TensorDataset(
            images[validation_indices], labels[validation_indices]
        ),
        test=test,
    )


def make_smoke_data() -> DatasetSplits:
    """Create deterministic MNIST-shaped data for a fast, network-free smoke run."""
    generator = torch.Generator().manual_seed(2026)

    def make_split(size: int) -> TensorDataset:
        labels = torch.arange(size) % 2
        images = torch.rand((size, 784), generator=generator) * 0.15
        images[labels == 0, :40] += 0.7
        images[labels == 1, 40:80] += 0.7
        return TensorDataset(images.clamp(0, 1), labels.long())

    return DatasetSplits(
        train=make_split(128), validation=make_split(48), test=make_split(48)
    )
