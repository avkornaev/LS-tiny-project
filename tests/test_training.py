from pathlib import Path

import pandas as pd
import torch

from label_smoothing.data import make_smoke_data
from label_smoothing.experiment import ExperimentConfig, run_experiment
from label_smoothing.losses import label_smoothed_cross_entropy
from label_smoothing.metrics import classification_metrics
from label_smoothing.model import make_model


def test_optimizer_step_changes_parameters() -> None:
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    images, labels = make_smoke_data().train.tensors
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = label_smoothed_cross_entropy(model(images[:8]), labels[:8], 0.1)
    loss.backward()
    optimizer.step()
    pairs = zip(before, model.parameters())  # noqa: B905 (Python 3.9 checks)
    assert any(not torch.equal(old, new) for old, new in pairs)


def test_metrics_are_finite() -> None:
    values = classification_metrics(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1])
    )
    assert values["accuracy"] == 1.0
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())


def test_tiny_cpu_experiment_creates_outputs(tmp_path: Path) -> None:
    run_experiment(tmp_path, smoke=True)
    expected = [
        "results/runs.csv",
        "results/summary.csv",
        "results/history.csv",
        "results/predictions.csv",
        "results/config.json",
        "results/validation_curves.csv",
        "results/reliability_bins.csv",
        "figures/metrics.png",
        "figures/reliability.png",
        "figures/confidence.png",
        "figures/validation_loss.png",
        "figures/tsne.png",
    ]
    assert all((tmp_path / path).is_file() for path in expected)
    runs = pd.read_csv(tmp_path / "results/runs.csv")
    assert len(runs) == 2
    assert set(runs["epsilon"]) == {0.0, 0.1}
    assert ExperimentConfig().seeds == (42, 123, 456)
