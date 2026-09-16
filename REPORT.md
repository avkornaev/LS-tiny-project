# Label Smoothing on MNIST 3 vs 7: Analysis Report

## Study status

This repository implements a small controlled study of label smoothing in binary
linear classification. The local smoke run checks that the complete pipeline
works, but it uses synthetic data, one paired seed, and one epoch. It is not
scientific evidence. Numerical conclusions must be written only after the full
six-run MNIST experiment has completed in Colab and its saved CSV files have
been inspected.

## Research question

How does label smoothing with epsilon 0.1, compared with no label smoothing,
change predictive accuracy, hard-label negative log-likelihood, Brier score,
calibration, confidence, and logit margin for logistic regression trained to
distinguish MNIST digits 3 and 7?

## Prespecified protocol

- Population: MNIST digits 3 and 7 only.
- Labels: digit 3 maps to class 0; digit 7 maps to class 1.
- Input: 28 by 28 pixels flattened to 784 values and scaled to `[0, 1]`.
- Model: one linear layer mapping 784 inputs to two logits.
- Conditions: epsilon 0 and epsilon 0.1.
- Paired seeds: 42, 123, and 456.
- Split: one stratified 80/20 split of the official training set, using seed
  2026, shared by all runs.
- Optimization: SGD, learning rate 0.1, batch size 128, and 20 epochs. The
  original design used 10 epochs; this was explicitly extended to make the
  final simulation approximately twice as long.
- Pairing: model initialization and minibatch order match within each seed pair;
  only the smoothing value changes.
- Evaluation: all metrics use hard labels. The official test set is not used for
  fitting, model selection, early stopping, or preprocessing decisions.

For two classes, the training target is calculated explicitly as

`smoothed_target = (1 - epsilon) * one_hot_target + epsilon / 2`.

Thus epsilon 0.1 assigns 0.95 probability to the labeled class and 0.05 to the
other class. Epsilon 0 is exactly ordinary cross-entropy.

## Outcomes

The primary run-level results are in `results/runs.csv`:

| Column | Meaning | Direction usually preferred |
|---|---|---|
| `accuracy` | Fraction of correct test predictions | Higher |
| `nll` | Mean hard-label test negative log-likelihood | Lower |
| `brier` | Mean two-class squared probability error | Lower |
| `ece` | 10-bin expected calibration error | Lower |
| `mean_confidence` | Mean maximum predicted probability | Descriptive |
| `logit_margin` | Mean absolute difference between two logits | Descriptive |

Accuracy measures discrimination, while NLL and Brier score also evaluate the
quality of predicted probabilities. ECE summarizes agreement between confidence
and observed accuracy, but its value depends on the binning scheme. Confidence
and logit margin are diagnostics rather than standalone measures of quality.

## Figures

### Validation loss versus epoch

`figures/validation_loss.png` plots hard-label validation NLL for every seed as a
faint line and the condition mean as a strong line. This figure can show whether
the conditions follow different optimization trajectories. Validation values
are descriptive only; training always lasts 20 epochs and there is no early
stopping or model selection.

### Metrics, reliability, and confidence

- Each timestamped run's `report.md` shows every run-level value and condition
  summaries in mean ± standard deviation notation.
- `figures/reliability.png` compares observed accuracy with mean confidence in
  occupied confidence bins.
- `figures/confidence.png` shows the distribution of maximum predicted
  probabilities.

### t-SNE diagnostic

`figures/tsne.png` applies t-SNE separately to each run's final two-logit test
representation and colors points by true digit. The t-SNE random seed is fixed
at 2026. This plot is exploratory: each panel has its own arbitrary coordinate
system, distances cannot be compared quantitatively between panels, and t-SNE
can distort global geometry. Because the source representation already has only
two dimensions, direct logit plots would be more quantitative; t-SNE is included
only as the requested qualitative visualization. It must not be used to claim a
performance improvement.

## Prism-ready files and workflow

Use the CSV files directly; do not transcribe values from figures.

1. Import `results/runs.csv` as the source for paired run-level comparisons.
   Pair epsilon 0 and 0.1 observations by `seed`. Plot all three paired points,
   not only bars or means.
2. Import `results/validation_curves.csv` for a grouped epoch plot. Use `epoch`
   as X, `epsilon` as the condition, `validation_nll_mean` as Y, and
   `validation_nll_std` as the error value. The table also records `n_runs`.
3. Import `results/reliability_bins.csv` as an XY table. Use `mean_confidence`
   as X and `observed_accuracy` as Y, grouped by `epsilon`. The `n_examples`
   column exposes sparsely populated bins.
4. Use `results/summary.csv` for display values only. Statistical comparisons,
   if performed, should use paired rows from `runs.csv`, not summary means.

With only three paired seeds, emphasize effect sizes, consistency across pairs,
and uncertainty. Avoid a claim of statistical significance or generality based
on this experiment alone.

## Result-writing template

After the full run, replace the bracketed fields using only saved CSV values:

> Across three paired seeds, the no-smoothing condition achieved mean accuracy
> [value] (SD [value]), compared with [value] (SD [value]) for epsilon 0.1.
> Hard-label NLL was [value] versus [value], Brier score was [value] versus
> [value], and 10-bin ECE was [value] versus [value]. Label smoothing
> [reduced/increased/did not consistently change] mean confidence and
> [reduced/increased/did not consistently change] absolute logit margin. These
> results describe a small controlled study with three paired seeds and do not
> establish statistical significance or broad generality.

Report paired seed-level values alongside this paragraph. State explicitly when
accuracy and calibration metrics point in different directions; do not reduce
the result to a single claim that smoothing “helped” or “hurt.”

## Reproducibility and limitations

- Three seeds give a small estimate of run-to-run variation.
- The task uses only two MNIST classes and a linear model, so conclusions need
  not transfer to multiclass or nonlinear systems.
- ECE changes with the number and placement of bins; this study fixes 10
  equal-width bins.
- The same official test set is evaluated for every completed model. It is never
  used for training decisions, but repeated inspection should not motivate
  changes to the fixed protocol.
- t-SNE is stochastic, nonlinear, and qualitative. A fixed seed improves
  reproducibility but does not make panel coordinates directly comparable.
- Package versions and the complete protocol are recorded in
  `results/config.json`; raw run outputs are saved before aggregation.

## Reproduction

Run the full experiment in Colab from a clean checkout:

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
python -m label_smoothing.experiment --all
```

Archive `results/`, `figures/`, and `results/config.json` together. Keep the raw
CSVs unchanged so every table and figure remains traceable to saved evidence.
