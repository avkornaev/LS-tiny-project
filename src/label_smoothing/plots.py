"""Figures generated from saved CSV data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def plot_reliability(predictions_path: Path, figure_path: Path) -> None:
    predictions = pd.read_csv(predictions_path)
    fig, axis = plt.subplots(figsize=(6, 5))
    edges = np.linspace(0.0, 1.0, 11)
    for epsilon, group in predictions.groupby("epsilon"):
        bin_ids = np.digitize(group["confidence"], edges[1:-1])
        points = group.assign(bin=bin_ids).groupby("bin").agg(
            confidence=("confidence", "mean"), accuracy=("correct", "mean")
        )
        axis.plot(
            points["confidence"],
            points["accuracy"],
            marker="o",
            label=f"ε={epsilon:g}",
        )
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect")
    axis.set(
        xlabel="Mean confidence", ylabel="Accuracy", xlim=(0.45, 1), ylim=(0.45, 1)
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def plot_confidence(predictions_path: Path, figure_path: Path) -> None:
    predictions = pd.read_csv(predictions_path)
    fig, axis = plt.subplots(figsize=(6, 4))
    for epsilon, group in predictions.groupby("epsilon"):
        axis.hist(
            group["confidence"],
            bins=np.linspace(0.5, 1, 16),
            alpha=0.55,
            label=f"ε={epsilon:g}",
        )
    axis.set(xlabel="Predicted confidence", ylabel="Count")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def plot_validation_loss(history_path: Path, figure_path: Path) -> None:
    """Plot paired validation NLL trajectories and the condition means."""
    history = pd.read_csv(history_path)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    colors = {0.0: "#4C78A8", 0.1: "#E45756"}
    for epsilon, condition in history.groupby("epsilon"):
        color = colors.get(float(epsilon), None)
        for _, run in condition.groupby("seed"):
            axis.plot(
                run["epoch"],
                run["validation_nll"],
                color=color,
                alpha=0.25,
                linewidth=1,
            )
        mean_curve = condition.groupby("epoch")["validation_nll"].mean()
        axis.plot(
            mean_curve.index,
            mean_curve.values,
            color=color,
            marker="o",
            linewidth=2,
            label=f"ε={epsilon:g} mean",
        )
    axis.set(
        xlabel="Epoch",
        ylabel="Validation hard-label NLL",
        title="Validation loss by label-smoothing condition",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def plot_tsne(predictions_path: Path, figure_path: Path) -> None:
    """Show exploratory t-SNE projections of each run's two test logits."""
    predictions = pd.read_csv(predictions_path)
    seeds = sorted(predictions["seed"].unique())
    epsilons = sorted(predictions["epsilon"].unique())
    fig, axes = plt.subplots(
        len(seeds),
        len(epsilons),
        figsize=(5 * len(epsilons), 4 * len(seeds)),
        squeeze=False,
    )
    for row, seed in enumerate(seeds):
        for column, epsilon in enumerate(epsilons):
            axis = axes[row, column]
            run = predictions[
                (predictions["seed"] == seed)
                & (predictions["epsilon"] == epsilon)
            ]
            perplexity = min(30.0, max(2.0, (len(run) - 1) / 3))
            embedding = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="random",
                learning_rate="auto",
                random_state=2026,
            ).fit_transform(run[["logit_0", "logit_1"]].to_numpy())
            for label, digit, color in [(0, "3", "#4C78A8"), (1, "7", "#E45756")]:
                selected = run["label"].to_numpy() == label
                axis.scatter(
                    embedding[selected, 0],
                    embedding[selected, 1],
                    s=10,
                    alpha=0.65,
                    color=color,
                    label=f"Digit {digit}",
                )
            axis.set(
                title=f"seed={seed}, ε={epsilon:g}",
                xlabel="t-SNE 1",
                ylabel="t-SNE 2",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0 and column == 0:
                axis.legend(markerscale=1.5)
    fig.suptitle("Exploratory t-SNE of final two-logit test representations")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def make_all_figures(results_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability(results_dir / "predictions.csv", figures_dir / "reliability.png")
    plot_confidence(results_dir / "predictions.csv", figures_dir / "confidence.png")
    plot_validation_loss(
        results_dir / "history.csv", figures_dir / "validation_loss.png"
    )
    plot_tsne(results_dir / "predictions.csv", figures_dir / "tsne.png")
