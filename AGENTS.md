# Codex instructions

## Goal

Build a minimal, readable, and reproducible Python project for the experiment specified in `README.md`. The code is teaching material for students new to project-based machine learning.

## Workflow

1. Read `README.md`, `CODEX_PROMPT.md`, and this file completely, then inspect
   the repository.
2. Create or update `PLAN.md` before substantial implementation.
3. Work in small, reviewable steps.
4. Test each completed component.
5. Finish with a smoke test and a concise report of changes and verification.

Ask a question only when an ambiguity would materially change the scientific protocol. Never change the protocol silently.

## Scientific invariants

- Use only MNIST digits 3 and 7.
- Map the labels consistently to 0 and 1.
- Use one linear layer from 784 inputs to 2 logits.
- Compare only $\varepsilon=0$ and $\varepsilon=0.1$.
- Use paired seeds 42, 123, and 456.
- Use one stratified 80/20 training/validation split with split seed 2026 for all runs.
- Scale pixels to $[0,1]$ and do not use augmentation.
- Use all eligible MNIST 3/7 examples from the official training set; do not
  enlarge training by borrowing validation or test examples.
- Use SGD with learning rate 0.1, batch size 128, and 20 epochs. This is the
  user-authorized revision of the original 10-epoch prompt.
- Keep each paired comparison identical except for label smoothing.
- Fit preprocessing and make decisions using training data only.
- Keep the official test set untouched until final evaluation.
- Evaluate against hard labels.
- Save every individual run before aggregation.

Implement label smoothing explicitly:

$$
\tilde{\mathbf y}=(1-\varepsilon)\mathbf y+\frac{\varepsilon}{K}\mathbf 1.
$$

Do not replace it with a library option unless requested.

## Code principles

- Prefer plain Python and small functions.
- Use type hints where they clarify interfaces.
- Add comments for reasoning, not for obvious syntax.
- Keep model, data, loss, metrics, experiment, and plotting logic separate.
- Use `pathlib` for paths and `argparse` for the command line.
- Support CPU execution and explicit CUDA selection. `--device cuda` must fail
  clearly when CUDA is unavailable instead of silently falling back to CPU.
- Do not add Hydra, Lightning, MLflow, notebooks-as-source, or unnecessary abstractions.
- Do not duplicate core logic in the Colab notebook; import it from `src`.

## Required checks

Run:

```bash
pytest -q
ruff check .
python -m label_smoothing.experiment --smoke
```

Tests must verify at least:

- only digits 3 and 7 remain;
- label mapping is correct;
- smoothed targets sum to one;
- $\varepsilon=0$ matches ordinary cross-entropy;
- model parameters change after one optimizer step;
- metrics are finite and result files are created.
- each invocation creates a unique date-time folder and a generated report.

If a command cannot run, state exactly why and what remains unverified.

## Results and claims

- Save raw results as CSV before making tables or figures.
- Record configuration and package versions.
- Put each invocation in its own date-time folder. Generate `report.md` with
  individual values and condition summaries in `mean ± std` notation.
- Do not generate a generic metrics figure. Keep validation-loss, reliability,
  confidence, and exploratory t-SNE figures; numerical metrics belong in the
  generated report and CSV files.
- Never invent, repair, or manually type experimental values.
- Do not claim that label smoothing helps unless the saved evidence supports it.
- Describe this as a small controlled study, not a reproduction of the full 2019 paper.

Preserve unrelated user changes. Do not commit or push unless explicitly requested.
