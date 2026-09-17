# Codex instructions

## Goal

Build the compact, readable, and reproducible teaching project specified in `README.md`. Prefer clarity over generality.

## Working method

- Inspect the repository before proposing changes.
- Plan before implementation; do not edit during the planning stage.
- Propose the simplest architecture that satisfies the specification.
- Wait for approval, then save the agreed plan as `PLAN.md`.
- Implement in small, reviewable steps and test each completed layer.
- Ask only when an ambiguity would change the scientific protocol or project design.
- Never change a scientific invariant silently.

## Scientific contract

- Use only MNIST digits 3 and 7; map them consistently to 0 and 1.
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

## Engineering principles

- Use the technology constraints in `README.md` and keep dependencies minimal.
- Prefer small functions, plain Python, and descriptive names.
- Use type hints where they clarify interfaces.
- Comment reasoning and non-obvious choices, not ordinary syntax.
- Separate responsibilities without creating unnecessary files or abstractions.
- Keep the Colab notebook short and call tested package code from it.
- Support CPU and optional CUDA; a requested CUDA run must fail clearly when unavailable.
- Do not add Hydra, Lightning, MLflow, notebooks-as-source, or premature abstractions.

## Evidence and verification

- Save individual results before aggregation.
- Record the configuration, package versions, device, and Git revision.
- Generate the report, tables, and figures from saved raw results.
- Include focused unit tests and a fast end-to-end smoke test.
- Verify formatting/linting, tests, and the smoke experiment before completion.
- Never invent, repair, or manually enter experimental results.
- Do not claim that label smoothing helps unless the evidence supports it.

If a check cannot run, state the reason and what remains unverified. Preserve unrelated user changes. Do not commit or push unless explicitly asked.

