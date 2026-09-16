# Label Smoothing on MNIST 3 vs 7

A small, reproducible project for learning how to plan, implement, test, run, and report a machine-learning experiment with ChatGPT, Codex, Google Colab, and Prism.

## Starting again with Codex

For a clean implementation attempt, open the repository in Codex and use
[`CODEX_PROMPT.md`](CODEX_PROMPT.md). It preserves the original request and lists
the later protocol and reporting amendments. Codex should read `AGENTS.md` and
create or update `PLAN.md` before implementation.

If rebuilding in a new checkout, keep experimental outputs outside the source
tree or in ignored timestamped folders. Do not copy numerical values from an old
run into a new report.

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
| Training | SGD, learning rate 0.1, batch size 128, 20 epochs |
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
├── CODEX_PROMPT.md
├── README.md
├── PLAN.md
├── REPORT.md
├── pyproject.toml
├── src/label_smoothing/
│   ├── data.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── experiment.py
│   └── plots.py
├── tests/
└── notebooks/
    └── label_smoothing_colab.ipynb
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

For example, `--output-dir experiment-reports` produces paths such as
`experiment-reports/2026-09-17_14-30-00-123456+0300_all/`. The shared MNIST
download is stored in `experiment-reports/data/`.

The runner uses `--device auto` by default: CUDA is selected when PyTorch can
access a GPU, otherwise it uses the CPU. Use `--device cuda` when you specifically
want the command to fail instead of silently falling back to CPU.

The smoke run uses synthetic data and one epoch. The full run uses every eligible
MNIST 3/7 training example and executes all six experiments. Every invocation
creates a separate `<date>_<time>_<mode>/` folder beneath `--output-dir` with:

- `report.md` — individual metrics and aggregate `mean ± std` results;
- `results/runs.csv` — one row per run;
- `results/summary.csv` — mean and standard deviation by condition;
- `results/history.csv` — per-epoch training loss and hard-label validation NLL;
- `results/validation_curves.csv` and `results/reliability_bins.csv` — tidy
  aggregate tables for Prism;
- `figures/reliability.png`;
- `figures/confidence.png`;
- `figures/validation_loss.png`;
- `figures/tsne.png`.

## Train on a Colab GPU from VS Code

The local VS Code window is only the notebook interface. Code cells run on a
separate Colab server, whose filesystem normally starts empty and is temporary.
The workflow below therefore clones the GitHub repository into `/content`, then
writes final results to Google Drive.

Before starting, commit and push the version you want to run to
`https://github.com/avkornaev/LS-tiny-project`. Uncommitted files on your local
computer are not automatically visible to the Colab server.

### 1. Open the project notebook

In VS Code, open
`notebooks/label_smoothing_colab.ipynb`. Do not start the full run from a local
Python kernel.

At the upper right of the notebook:

1. Click **Select Kernel**.
2. Select **Colab**.
3. Sign in to the intended Google account and approve the connection if asked.
4. Select **New Colab Server** and choose an available GPU machine type. Use
   **Auto Connect** only if it assigns a GPU when checked in the next step.

The official extension describes this connection path as **Select Kernel →
Colab**, with either **Auto Connect** or **New Colab Server**. GPU types and free
tier availability can change, so the project does not require a particular GPU.

### 2. Verify that the remote kernel really has a GPU

Run this as the first notebook cell:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
assert torch.cuda.is_available(), "This Colab server does not have a CUDA GPU"
```

Then run:

```python
!nvidia-smi
```

The assertion must pass and `nvidia-smi` must list a GPU. Merely connecting to a
Colab server does not guarantee that PyTorch is using a GPU. If the check fails,
click the **Colab** button in the notebook toolbar, remove the current server,
then select **Select Kernel → Colab → New Colab Server** and choose a GPU.

### 3. Copy and install this exact project on the remote server

#### Recommended for a private repository: upload from VS Code

If GitHub asks for a username while cloning, the repository is private and the
anonymous clone command cannot authenticate. Do not enter a password or paste a
GitHub token into a notebook cell. Instead:

1. In the ordinary VS Code Explorer, right-click the local
   `LS-tiny-project` folder.
2. Select **Upload to Colab**.
3. Select the active Colab server if VS Code asks which server to use.
4. In the Colab extension's **Contents** view, confirm that
   `/content/LS-tiny-project/pyproject.toml` exists.

You can verify the upload from a cell:

```python
from pathlib import Path

project_dir = Path("/content/LS-tiny-project")
assert (project_dir / "pyproject.toml").is_file(), (
    "Upload the local LS-tiny-project folder to /content first"
)
print("Project uploaded to", project_dir)
```

Then install it:

```python
%cd /content/LS-tiny-project
!python -m pip install -e ".[dev]"
```

This route includes your current local files, including changes that have not
yet been pushed. The upload must be repeated after a new Colab server is created
because `/content` is temporary.

#### Alternative for a public repository: clone from GitHub

If the repository is public and the desired revision has already been pushed,
run these cells. The first command also works when the repository was already
cloned earlier in the same runtime:

```python
!if [ -d /content/LS-tiny-project/.git ]; then git -C /content/LS-tiny-project pull --ff-only; else git clone https://github.com/avkornaev/LS-tiny-project.git /content/LS-tiny-project; fi
%cd /content/LS-tiny-project
!python -m pip install -e ".[dev]"
```

Confirm that the installed code is the expected revision:

```python
!git rev-parse HEAD
!git status --short
```

For a Git clone, the status should be clean and the printed commit hash should be
recorded with the experiment results. For a VS Code upload, `git` metadata may be
absent; instead record the local commit hash before uploading and archive the
uploaded source with the results.

### 4. Run tests and a GPU smoke test

```python
!pytest -q
!ruff check .
!python -m label_smoothing.experiment --smoke --device cuda --output-dir /content/smoke-output
```

The smoke command must print `device=cuda` twice. It uses synthetic data and one
epoch, so it verifies the remote GPU path without starting the scientific run.
It is not part of the study results.

You can also verify the recorded device:

```python
import json
from pathlib import Path

smoke_runs = list(Path("/content/smoke-output").glob("*_smoke"))
smoke_dir = max(smoke_runs, key=lambda path: path.stat().st_mtime)
with (smoke_dir / "results/config.json").open() as file:
    smoke_config = json.load(file)
smoke_config["device"], smoke_config["gpu_name"]
```

### 5. Mount Google Drive for persistent results

Colab servers are temporary. In VS Code, open the Command Palette with
`Cmd+Shift+P` on macOS or `Ctrl+Shift+P` on Windows/Linux, run **Colab: Mount
Google Drive to Server...**, and execute the cell it inserts. Alternatively run:

```python
from google.colab import drive

drive.mount("/content/drive")
```

After authorization, create one persistent base folder. The experiment creates
a new timestamped child folder on every invocation, so reports never overwrite
one another.

```python
from pathlib import Path

output_dir = Path("/content/drive/MyDrive/LS-tiny-project/reports")
output_dir.mkdir(parents=True, exist_ok=True)
print(output_dir)
```

### 6. Start the full six-run training job

Run exactly one full experiment command. In a notebook, use `subprocess.run` with
`check=True`; unlike an IPython `!` command, this raises an error and stops the
cell when training fails instead of allowing later display cells to run:

```python
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "label_smoothing.experiment",
        "--all",
        "--device",
        "cuda",
        "--output-dir",
        "/content/drive/MyDrive/LS-tiny-project/reports",
    ],
    check=True,
)
```

`--device cuda` is intentional: if the GPU connection is lost, the experiment
stops with a clear error instead of continuing unnoticed on CPU. The command
downloads MNIST, uses the one fixed split, trains both smoothing conditions for
seeds 42, 123, and 456, and saves each completed run before aggregation. Do not
start another copy of the command while it is running.

This model is only one linear layer, so GPU utilization may look low and a GPU
may not be faster than a modern CPU. The GPU changes the compute device, not the
scientific protocol.

### 7. Confirm and retrieve the results

When the command finishes, locate the newest timestamped full-run folder and
inspect its report, configuration, and tables:

```python
from pathlib import Path
import json
import pandas as pd

output_dir = Path("/content/drive/MyDrive/LS-tiny-project/reports")
run_dir = max(output_dir.glob("*_all"), key=lambda path: path.stat().st_mtime)
with (run_dir / "results/config.json").open() as file:
    config = json.load(file)

print("Device:", config["device"])
print("GPU:", config["gpu_name"])
print("Report folder:", run_dir)
display(pd.read_csv(run_dir / "results/runs.csv"))
display(pd.read_csv(run_dir / "results/summary.csv"))
```

The completed folder should contain:

```text
reports/
├── data/                                  # shared MNIST download; do not commit
└── 2026-09-17_14-30-00-123456+0300_all/
    ├── report.md                          # metrics as mean ± std
    ├── results/
    │   ├── config.json
    │   ├── history.csv
    │   ├── predictions.csv
    │   ├── reliability_bins.csv
    │   ├── runs.csv                       # 3 seeds × 2 conditions
    │   ├── summary.csv
    │   └── validation_curves.csv
    └── figures/
        ├── confidence.png
        ├── reliability.png
        ├── tsne.png
        └── validation_loss.png
```

Check that `runs.csv` has six rows and no missing metrics:

```python
runs = pd.read_csv(run_dir / "results/runs.csv")
assert len(runs) == 6
assert set(runs["seed"]) == {42, 123, 456}
assert set(runs["epsilon"]) == {0.0, 0.1}
assert set(runs["device"]) == {"cuda"}
assert not runs.isna().any().any()
```

The files are already persistent in Google Drive. The Colab extension's
**Contents** view can also browse `/content` and download files, but anything
stored only under `/content` disappears when the server is removed or expires.

### Troubleshooting

- **`CUDA was requested, but PyTorch cannot access a GPU`:** reconnect using a
  new Colab GPU server, rerun the GPU checks, then rerun the command. Do not
  change the command to CPU midway through a paired study.
- **No GPU machine type is offered:** GPU access is subject to Colab availability
  and account limits. Wait and try later or use a paid Colab tier; available GPU
  models and limits are not guaranteed.
- **`ModuleNotFoundError: label_smoothing`:** rerun `%cd
  /content/LS-tiny-project` and `!python -m pip install -e ".[dev]"` in the active
  Colab kernel.
- **`fatal: could not read Username for 'https://github.com'`:** GitHub requires
  authentication, normally because the repository is private. Use **Upload to
  Colab** as described above, or make the repository public before cloning. Do
  not place a personal access token in a saved notebook.
- **The runtime disconnected:** inspect the persistent `runs.csv`. Because raw
  results are saved after each run, completed rows remain, but restart the full
  command in a new empty output folder to preserve a single clean, paired study.
- **`FileNotFoundError` for `results/runs.csv`:** the results cell was run before
  any training run completed. Confirm that Drive is mounted, inspect the output
  from the full training cell, and rerun that cell with `check=True`. Do not run
  the display cell until training finishes without an exception.
- **Local edits are missing:** commit and push them to GitHub, then run the clone
  or pull cell again. The remote server cannot see uncommitted local files.

For current extension behavior, see the official [Google Colab VS Code extension
guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide). For current
resource behavior and availability, see the official [Colab
FAQ](https://research.google.com/colaboratory/faq.html).

For submission, provide the GitHub commit URL and, if required, upload a ZIP archive containing the source, notebook, saved results, and figures. Do not include the downloaded dataset.

## Reproducibility checklist

- `pytest -q` passes.
- The notebook passes **Runtime → Restart session and run all**.
- The test set is not used for tuning or early stopping.
- Every result records the seed, smoothing value, hyperparameters, and package versions.
- Tables and figures are generated from saved result files.
- The report shows individual runs as well as mean ± standard deviation.
- No statistical significance claim is made from only three seeds.
