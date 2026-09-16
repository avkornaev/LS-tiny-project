"""Command-line runner for the controlled label-smoothing study."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from label_smoothing.data import DatasetSplits, load_mnist, make_smoke_data
from label_smoothing.losses import label_smoothed_cross_entropy
from label_smoothing.metrics import classification_metrics
from label_smoothing.model import make_model
from label_smoothing.plots import make_all_figures


@dataclass(frozen=True)
class ExperimentConfig:
    epsilons: tuple[float, ...] = (0.0, 0.1)
    seeds: tuple[int, ...] = (42, 123, 456)
    split_seed: int = 2026
    learning_rate: float = 0.1
    batch_size: int = 128
    epochs: int = 10
    ece_bins: int = 10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset: TensorDataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


@torch.no_grad()
def collect_logits(
    model: nn.Module, dataset: TensorDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for images, labels in DataLoader(dataset, batch_size=512, shuffle=False):
        logits_parts.append(model(images.to(device)).cpu())
        label_parts.append(labels)
    return torch.cat(logits_parts), torch.cat(label_parts)


def run_training(
    datasets: DatasetSplits,
    seed: int,
    epsilon: float,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, float | int]], pd.DataFrame]:
    """Train one run and return its metrics, history, and test predictions."""
    seed_everything(seed)
    model = make_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(datasets.train, config.batch_size, seed)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = label_smoothed_cross_entropy(model(images), labels, epsilon)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            example_count += len(labels)

        validation_logits, validation_labels = collect_logits(
            model, datasets.validation, device
        )
        history.append(
            {
                "seed": seed,
                "epsilon": epsilon,
                "epoch": epoch,
                "train_loss": loss_sum / example_count,
                "validation_nll": nn.functional.cross_entropy(
                    validation_logits, validation_labels
                ).item(),
            }
        )

    # Test data is touched only after all training epochs have finished.
    test_logits, test_labels = collect_logits(model, datasets.test, device)
    metrics = classification_metrics(test_logits, test_labels)
    run = {
        "seed": seed,
        "epsilon": epsilon,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "device": device.type,
        **metrics,
    }
    probabilities = test_logits.softmax(dim=1)
    confidence, predicted = probabilities.max(dim=1)
    predictions = pd.DataFrame(
        {
            "sample_id": np.arange(len(test_labels)),
            "seed": seed,
            "epsilon": epsilon,
            "label": test_labels.numpy(),
            "prediction": predicted.numpy(),
            "confidence": confidence.numpy(),
            "correct": predicted.eq(test_labels).to(torch.int64).numpy(),
            "logit_0": test_logits[:, 0].numpy(),
            "logit_1": test_logits[:, 1].numpy(),
        }
    )
    return run, history, predictions


def save_completed_runs(
    runs: list[dict[str, object]],
    histories: list[dict[str, float | int]],
    predictions: list[pd.DataFrame],
    results_dir: Path,
) -> None:
    """Persist all raw outputs accumulated so far after an individual run."""
    pd.DataFrame(runs).to_csv(results_dir / "runs.csv", index=False)
    pd.DataFrame(histories).to_csv(results_dir / "history.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        results_dir / "predictions.csv", index=False
    )


def aggregate_results(runs_path: Path, summary_path: Path) -> None:
    runs = pd.read_csv(runs_path)
    metric_columns = [
        "accuracy",
        "nll",
        "brier",
        "ece",
        "mean_confidence",
        "logit_margin",
    ]
    summary = runs.groupby("epsilon")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(summary_path, index=False)


def make_prism_tables(results_dir: Path) -> None:
    """Create tidy aggregate tables that can be imported directly into Prism."""
    history = pd.read_csv(results_dir / "history.csv")
    curves = (
        history.groupby(["epsilon", "epoch"])["validation_nll"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "validation_nll_mean",
                "std": "validation_nll_std",
                "count": "n_runs",
            }
        )
    )
    curves.to_csv(results_dir / "validation_curves.csv", index=False)

    predictions = pd.read_csv(results_dir / "predictions.csv")
    edges = np.linspace(0.0, 1.0, 11)
    predictions["confidence_bin"] = np.digitize(
        predictions["confidence"], edges[1:-1]
    )
    reliability = (
        predictions.groupby(["epsilon", "confidence_bin"])
        .agg(
            mean_confidence=("confidence", "mean"),
            observed_accuracy=("correct", "mean"),
            n_examples=("correct", "size"),
        )
        .reset_index()
    )
    reliability.to_csv(results_dir / "reliability_bins.csv", index=False)


def package_versions() -> dict[str, str]:
    packages = ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "matplotlib"]
    return {name: importlib.metadata.version(name) for name in packages}


def resolve_device(requested: str) -> torch.device:
    """Select CUDA when requested and available, otherwise use CPU for auto."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def run_experiment(output_dir: Path, smoke: bool, device_name: str = "auto") -> None:
    config = ExperimentConfig(seeds=(42,), epochs=1) if smoke else ExperimentConfig()
    device = resolve_device(device_name)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = make_smoke_data() if smoke else load_mnist(output_dir / "data")

    metadata = {
        "mode": "smoke" if smoke else "all",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "protocol": asdict(config),
        "versions": package_versions(),
    }
    (results_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    runs: list[dict[str, object]] = []
    histories: list[dict[str, float | int]] = []
    predictions: list[pd.DataFrame] = []
    for seed in config.seeds:
        for epsilon in config.epsilons:
            run, run_history, run_predictions = run_training(
                datasets, seed, epsilon, config, device
            )
            runs.append(run)
            histories.extend(run_history)
            predictions.append(run_predictions)
            save_completed_runs(runs, histories, predictions, results_dir)
            print(f"saved seed={seed}, epsilon={epsilon:g}, device={device}")

    aggregate_results(results_dir / "runs.csv", results_dir / "summary.csv")
    make_prism_tables(results_dir)
    make_all_figures(results_dir, figures_dir)
    print(f"results written under {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="run a tiny CPU check")
    mode.add_argument(
        "--all", action="store_true", help="run all six MNIST experiments"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="root for data and outputs"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute device; auto selects CUDA when available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.output_dir, smoke=args.smoke, device_name=args.device)


if __name__ == "__main__":
    main()
