# Label Smoothing on MNIST 3 vs 7

A tiny research project for learning how to plan, build, test, run, and report a machine-learning experiment with Codex.

## Research question

How does label smoothing affect the accuracy, confidence, and calibration of logistic regression on MNIST digits 3 and 7?

This is a controlled teaching experiment inspired by [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629), not a reproduction of the full paper.

## Experimental protocol

| Component | Fixed choice |
|---|---|
| Data | MNIST: 3 → 0, 7 → 1 |
| Input | Flattened pixels in $[0,1]$; no augmentation |
| Model | One linear layer: 784 inputs → 2 logits |
| Comparison | $\varepsilon=0$ vs. $\varepsilon=0.1$ |
| Seeds | 42, 123, 456 |
| Split | Fixed stratified 80/20 train/validation split; seed 2026 |
| Training | SGD, learning rate 0.1, batch size 128, 20 epochs |
| Evaluation | Official MNIST test set, used only after training |

For $K=2$ classes, implement label smoothing explicitly:

$$
\tilde{\mathbf y}=(1-\varepsilon)\mathbf y+\frac{\varepsilon}{K}\mathbf 1.
$$

Within each seed, both conditions must use the same split, initialization, batch order, optimizer, and training schedule. Only $\varepsilon$ may differ.

## Required evidence

Report every run and `mean ± std` for:

- accuracy;
- hard-label negative log-likelihood;
- Brier score;
- expected calibration error;
- mean confidence;
- absolute logit margin.

Produce clear figures for:

- validation dynamics;
- calibration;
- confidence distributions.

The project must save machine-readable per-run results, aggregate results, the experiment configuration, and a concise generated report. Tables and figures must be derived from saved results, never entered manually.

Three seeds demonstrate reproducible comparison; they do not justify strong significance claims.

## Technology constraints

Use a compact beginner-friendly stack:

- Python 3.11+;
- PyTorch and torchvision;
- NumPy, pandas, and scikit-learn;
- Matplotlib;
- pytest and Ruff;
- Jupyter / Google Colab.

Do not add training frameworks, experiment trackers, dashboards, or configuration systems.

## Functional requirements

The finished project must:

- install as a small Python package;
- run on CPU and, when available, CUDA;
- provide a fast smoke experiment and the complete six-run experiment;
- run locally and from one restart-and-run-all Colab notebook;
- keep the notebook as an interface to tested package code;
- preserve final outputs on Google Drive;
- record the software environment, device, and Git revision;
- exclude datasets, environments, caches, and generated runs from Git.

The exact project structure, command-line interface, output layout, tests, and Colab procedure are design decisions to be proposed by Codex and approved before implementation.

## Development workflow

1. Codex proposes a plan without editing files.
2. The human reviews the scientific and engineering decisions.
3. Codex saves the approved plan, implements the project, and verifies it.
4. The full experiment runs once in Colab and saves its evidence to Google Drive.
5. The verified report, tables, and figures become inputs for Prism.

After implementation, update this README with the actual project structure and exact usage instructions.

