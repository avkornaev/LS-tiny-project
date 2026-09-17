# Codex instructions

## Goal

Build the small, readable, and reproducible teaching project specified in `README.md`. Prefer clarity over generality.

## Working method

- Inspect the repository before proposing changes.
- Plan before implementation; do not edit during the planning stage.
- After approval, save the agreed plan as `PLAN.md` and implement it in small steps.
- Test each completed layer and finish with the full required verification.
- Ask only when an ambiguity would change the scientific protocol or project structure.
- Never change a scientific invariant silently.

## Scientific contract

- Use only MNIST digits 3 and 7; map them to 0 and 1 consistently.
- Use all eligible examples from the official training set.
- Use one fixed stratified 80/20 train/validation split with seed 2026.
- Keep the official test set untouched until final evaluation.
- Scale pixels to `[0,1]`; use no augmentation.
- Use one linear layer from 784 inputs to 2 logits.
- Compare only $\varepsilon=0$ and $\varepsilon=0.1$.
- Use paired seeds 42, 123, and 456.
- Use SGD, learning rate 0.1, batch size 128, and 20 epochs.
- Within each seed pair, match initialization and batch order.
- Evaluate against hard labels and save every run before aggregation.

Implement label smoothing directly:

$$
\tilde{\mathbf y}=(1-\varepsilon)\mathbf y+\frac{\varepsilon}{K}\mathbf 1.
$$

Do not replace it with a library option.

## Engineering rules

- Use a standard `src` package and `pyproject.toml`.
- Prefer small functions, plain Python, `pathlib`, `argparse`, and helpful type hints.
- Comment reasoning and non-obvious choices, not ordinary syntax.
- Keep data, model/loss, metrics, experiment, and plotting responsibilities separate.
- Keep the Colab notebook short; import project code instead of copying it.
- Support CPU and CUDA through `--device auto|cpu|cuda`.
- A requested CUDA run must fail clearly when CUDA is unavailable.
- Do not add Hydra, Lightning, MLflow, notebooks-as-source, dashboards, or premature abstractions.

## Evidence

Each full invocation must create a new output folder with:

- configuration and environment information;
- individual and aggregate CSV results;
- per-epoch history;
- a generated `report.md`;
- validation-loss, reliability, and confidence figures.

Never invent, repair, or manually enter results. Do not claim that label smoothing helps unless the saved evidence supports it.

## Required verification

Run:

```bash
pytest -q
ruff check .
python -m label_smoothing.experiment --smoke
```

Tests must verify at least:

- digit filtering and label mapping;
- deterministic splitting;
- valid smoothed targets;
- equivalence to ordinary cross-entropy when $\varepsilon=0$;
- a parameter update after one optimizer step;
- finite metrics and creation of expected output files.

If a check cannot run, state the reason and what remains unverified. Preserve unrelated user changes. Do not commit or push unless explicitly asked.

