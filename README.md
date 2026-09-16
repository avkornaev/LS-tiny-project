# Label Smoothing on MNIST 3 vs 7

A small, reproducible project for learning how to plan, implement, test, run, and report a machine-learning experiment with ChatGPT, Codex, Google Colab, and Prism.

## Research question

How does label smoothing change the accuracy, confidence, and calibration of logistic regression on MNIST digits 3 and 7?

This is a focused teaching experiment, not a full reproduction of [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629).

## Experimental design

| Component | Value |
|---|---|
| Data | MNIST digits 3 and 7 |
| Inputs | Flattened pixels scaled to $[0,1]$; no augmentation |
| Model | Linear layer: 784 inputs, 2 logits |
| Conditions | Label smoothing $\varepsilon=0$ and $\varepsilon=0.1$ |
| Seeds | 42, 123, 456 |
| Split | Fixed stratified 80/20 split of the training set; split seed 2026 |
| Training | SGD, learning rate 0.1, batch size 128, 10 epochs |
| Runs | 2 conditions × 3 paired seeds = 6 |
| Metrics | Accuracy, hard-label NLL, Brier score, ECE |
| Diagnostics | Mean confidence and absolute logit margin |

Only label smoothing may differ within each paired comparison. The validation split is fixed across all runs. Within each seed pair, initialization and batch order must also match.

For $K=2$ classes, use

$$
\tilde{\mathbf y}=(1-\varepsilon)\mathbf y+\frac{\varepsilon}{K}\mathbf 1.
$$

All evaluation metrics use the original hard labels.

## Beginner-friendly stack

- Python 3.11+
- PyTorch and torchvision
- NumPy and pandas
- scikit-learn
- Matplotlib
- pytest
- Ruff
- Jupyter / Google Colab

Avoid experiment frameworks, configuration frameworks, and tracking services. This project should remain understandable from its source code.

## Intended structure

```text
.
├── AGENTS.md
├── README.md
├── PLAN.md
├── pyproject.toml
├── src/label_smoothing/
│   ├── data.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── experiment.py
│   └── plots.py
├── tests/
├── notebooks/
│   └── label_smoothing_colab.ipynb
├── results/
└── figures/
```

Do not commit downloaded MNIST files, virtual environments, caches, or model checkpoints.

## Local workflow

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pytest -q
ruff check .
python -m label_smoothing.experiment --smoke
python -m label_smoothing.experiment --all
```

The smoke run uses a tiny subset and one epoch. The full run executes all six experiments and creates:

- `results/runs.csv` — one row per run;
- `results/summary.csv` — mean and standard deviation by condition;
- `results/history.csv` — per-epoch training loss and hard-label validation NLL;
- `results/validation_curves.csv` and `results/reliability_bins.csv` — tidy
  aggregate tables for Prism;
- `figures/metrics.png`;
- `figures/reliability.png`;
- `figures/confidence.png`;
- `figures/validation_loss.png`;
- `figures/tsne.png`.

## Google Colab

Keep the code in GitHub. Colab should clone the repository and install it:

```python
!git clone <YOUR_GITHUB_REPOSITORY_URL>
%cd <REPOSITORY_DIRECTORY>
!pip install -e ".[dev]"
!pytest -q
!python -m label_smoothing.experiment --all
```

Google Drive is optional. Mount it only when results must survive the Colab session:

```python
from google.colab import drive
drive.mount("/content/drive")

!python -m label_smoothing.experiment \
    --all \
    --output-dir "/content/drive/MyDrive/label-smoothing-results"
```

For submission, provide the GitHub commit URL and, if required, upload a ZIP archive containing the source, notebook, saved results, and figures. Do not include the downloaded dataset.

## Reproducibility checklist

- `pytest -q` passes.
- The notebook passes **Runtime → Restart session and run all**.
- The test set is not used for tuning or early stopping.
- Every result records the seed, smoothing value, hyperparameters, and package versions.
- Tables and figures are generated from saved result files.
- The report shows individual runs as well as mean ± standard deviation.
- No statistical significance claim is made from only three seeds.
