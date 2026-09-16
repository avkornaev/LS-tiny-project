# Implementation plan

## Steps

1. Scaffold a small `src/label_smoothing` package and declare only the requested
   runtime and development dependencies in `pyproject.toml`.
2. Implement separate modules for MNIST loading/splitting, the linear model,
   explicit label-smoothed loss, hard-label metrics, training/orchestration, and
   plots.
3. Add a command-line experiment runner with `--smoke`, `--all`, and
   `--output-dir`. Save each run before aggregation, then create CSV summaries,
   figures, and machine-readable configuration/version metadata.
4. Add focused unit tests and a tiny synthetic-data CPU smoke test. Add a Colab
   notebook that installs the repository and invokes the tested package code.
5. Run `pytest -q`, `ruff check .`, and
   `python -m label_smoothing.experiment --smoke` without running the full study.

## Experimental invariants

- Keep only MNIST digits 3 and 7, mapped consistently to labels 0 and 1.
- Scale pixels to `[0, 1]`, flatten to 784 features, and use no augmentation.
- Make one stratified 80/20 train/validation split with split seed 2026 and use
  it unchanged for every run. Keep the official MNIST test set untouched until
  final evaluation.
- Use exactly one linear layer from 784 inputs to 2 logits.
- Compare only smoothing values 0 and 0.1 with paired seeds 42, 123, and 456.
  Within a pair, reset model initialization and batch order from the same seed;
  only the loss smoothing value may differ.
- Train with SGD, learning rate 0.1, batch size 128, and 20 epochs. This is an
  explicit user-requested revision of the original 10-epoch protocol.
- Implement smoothed targets explicitly as
  `(1 - epsilon) * one_hot + epsilon / num_classes`.
- Evaluate accuracy, hard-label NLL, Brier score, ECE, mean confidence, and
  absolute logit margin against hard labels only.

## Expected outputs

- `results/runs.csv`: one row per completed run, updated after every run.
- `results/summary.csv`: mean and sample standard deviation by smoothing value.
- `results/history.csv`: per-epoch training and validation loss for plotting.
- `results/validation_curves.csv`: Prism-ready mean and standard deviation of
  validation NLL by epoch and condition.
- `results/reliability_bins.csv`: Prism-ready calibration-bin values.
- `results/config.json`: protocol, command mode, and package versions.
- `figures/reliability.png` and `figures/confidence.png`, generated from saved
  raw result CSV files.
- `figures/validation_loss.png`: hard-label validation NLL across epochs, with
  individual paired runs and condition means.
- `figures/tsne.png`: per-run t-SNE views of the final two-logit test
  representation, clearly treated as an exploratory diagnostic.
- `REPORT.md`: protocol, data dictionary, Prism workflow, interpretation rules,
  and reproducibility limitations without invented full-study results.
- `<date>_<time>_<mode>/report.md`: generated per-experiment report containing
  individual metrics and aggregate `mean ± std` values.

An alternate `--output-dir` contains the same `results/` and `figures/`
subdirectories. Smoke mode uses deterministic synthetic MNIST-shaped data, one
seed, both smoothing conditions, and one epoch so it is fast and network-free.

## Tests

- Filtering retains only digits 3 and 7 and maps them correctly.
- The fixed split is stratified, deterministic, and disjoint.
- Smoothed targets sum to one; epsilon zero matches ordinary cross-entropy.
- One optimizer step changes model parameters.
- Metrics are finite and have expected values on a simple example.
- A tiny CPU experiment creates individual, aggregate, history, configuration,
  Prism-ready, and figure files.

## Completion criteria

- The intended package, CLI, tests, notebook, ignore rules, and output folders
  exist and are readable by beginners.
- All three required checks pass locally, or any environment limitation is
  reported precisely.
- The full six-run experiment is left for Colab and no unsupported scientific
  claim is made.
