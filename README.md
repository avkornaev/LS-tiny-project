# Label Smoothing on MNIST 3 vs 7

A tiny research project for learning how to plan, build, test, run, and report a machine-learning experiment with Codex.

## Question

How does label smoothing affect the accuracy, confidence, and calibration of logistic regression on MNIST digits 3 and 7?

This is a controlled teaching experiment inspired by [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629), not a reproduction of the full paper.

## Experiment

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

Within each seed, the two conditions must use the same split, initialization, batch order, optimizer, and training schedule. Only $\varepsilon$ may differ.

Report every run and `mean ± std` for:

- accuracy;
- hard-label negative log-likelihood;
- Brier score;
- expected calibration error;
- mean confidence;
- absolute logit margin.

Three seeds demonstrate reproducible comparison; they do not justify strong significance claims.

## Stack

- Python 3.11+
- PyTorch and torchvision
- NumPy, pandas, and scikit-learn
- Matplotlib
- pytest and Ruff
- Jupyter / Google Colab

Keep the stack plain. Do not add training frameworks, experiment trackers, or configuration systems.

## Project shape

```text
.
├── README.md
├── AGENTS.md
├── CODEX_PROMPT.md
├── PLAN.md                       # created after plan approval
├── pyproject.toml
├── src/label_smoothing/
│   ├── data.py
│   ├── model.py                  # model and smoothed loss
│   ├── metrics.py
│   ├── experiment.py
│   └── plots.py
├── tests/
│   ├── test_loss.py
│   └── test_smoke.py
└── notebooks/
    └── run_in_colab.ipynb
```

The notebook is a readable interface to the package. It must not duplicate the training implementation.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pytest -q
ruff check .
python -m label_smoothing.experiment --smoke
python -m label_smoothing.experiment --all --device auto
```

`--smoke` uses a tiny synthetic dataset and one epoch. It checks the pipeline, not the hypothesis.

Each full invocation creates a new output directory containing:

```text
report.md
results/config.json
results/runs.csv
results/summary.csv
results/history.csv
figures/validation_loss.png
figures/reliability.png
figures/confidence.png
```

Tables and figures must be generated from saved raw results. Do not type experimental values manually.

## Colab and Google Drive

The same notebook should run in Google Colab or in VS Code connected to a Colab kernel.

It should, in order:

1. verify the runtime and print the selected device;
2. clone or update the GitHub repository;
3. install the project with `pip install -e ".[dev]"`;
4. run `pytest -q` and the smoke test;
5. mount Google Drive;
6. run the six experiments once;
7. save the complete output folder under `MyDrive/LS-tiny-project/reports`;
8. display the final table and figures.

A GPU is optional for this linear model. Colab is useful mainly as a reproducible remote environment and for persistent Drive storage. If CUDA is selected, record the GPU name; if CPU is used, record that instead.

Do not commit downloaded data, virtual environments, caches, or generated output folders. Preserve the Git commit hash with every final experiment.

## Classroom workflow

1. Read the question and freeze the protocol.
2. Open the repository in Codex using **Sol Medium** and plan before editing.
3. Discuss the plan; the human approves scientific and structural decisions.
4. Ask Codex to save `PLAN.md`, implement, and verify the project.
5. Run the notebook in Colab and save the evidence to Drive.
6. Use `report.md`, CSV tables, and figures as verified inputs for Prism.

The student remains responsible for understanding the code, checking the experiment, and defending every claim.

