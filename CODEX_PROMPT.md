# Codex project prompt

This file preserves the first implementation prompt and the later authorized
changes. For a fresh attempt, give Codex the original prompt together with the
amendments below. `AGENTS.md` is the authoritative repository instruction file
when an original value and a later amendment differ.

## Original first prompt — verbatim

> LS tiny project
>
> Initialize this repository as a small, beginner-friendly Python research
> project.
>
> First read `README.md` and `AGENTS.md`, then inspect the current repository.
> Create `PLAN.md` with a short implementation plan, experimental invariants,
> expected outputs, tests, and completion criteria. After that, implement the
> project described in `README.md`.
>
> Use a standard `src` package with `pyproject.toml`. Keep the dependency set
> small: PyTorch, torchvision, NumPy, pandas, scikit-learn, Matplotlib, pytest,
> and Ruff. Create a Colab notebook that installs the package and calls the
> tested project code rather than duplicating it.
>
> Required capabilities:
>
> - download and filter MNIST to digits 3 and 7;
> - create one fixed, stratified 80/20 training/validation split with split seed
>   2026;
> - train linear logistic regression with two logits;
> - implement label-smoothed cross-entropy explicitly;
> - run paired experiments for smoothing values 0 and 0.1 with seeds 42, 123,
>   and 456;
> - report accuracy, hard-label NLL, Brier score, ECE, mean confidence, and logit
>   margin;
> - save individual and aggregated CSV results;
> - generate clear metrics, reliability, and confidence figures, e.g. val loss,
>   t-sne plot;
> - support `--smoke`, `--all`, and `--output-dir`;
> - record hyperparameters and package versions;
> - include unit tests and a tiny CPU smoke test.
>
> Use pixels scaled to `[0,1]`, no augmentation, SGD with learning rate 0.1,
> batch size 128, and 10 epochs. Within each seed pair, use matching
> initialization and batch order.
>
> Keep the scientific protocol fixed. Do not add model variants, hyperparameter
> searches, dashboards, experiment frameworks, or extra features.
>
> Run all feasible checks. Do not run the full six-run experiment merely to
> demonstrate that the code works; the full experiment will be run in Colab. At
> the end, report:
>
> 1. what you created or changed;
> 2. which commands passed;
> 3. what remains to be run in Colab;
> 4. any scientific or reproducibility limitation you found.

## Later authorized amendments

Apply these amendments without changing any other scientific invariant:

1. Train for 20 epochs instead of 10. This makes the full simulation roughly
   twice as long. Continue using all eligible MNIST 3/7 training examples; never
   add validation or official test examples to training.
2. Support `--device auto`, `--device cpu`, and `--device cuda`. A requested CUDA
   run must fail clearly if CUDA is unavailable. Record the device and GPU name.
3. Every smoke or full invocation must create a unique date-time folder beneath
   `--output-dir`. Keep its report, raw CSVs, configuration, and figures together.
4. Generate `report.md` inside every date-time folder. Include all individual
   paired runs and aggregate metrics using `mean ± std` notation.
5. Do not create `metrics.png`. Put numerical metrics in `report.md` and the CSV
   tables. Keep reliability, confidence, validation-loss-versus-epoch, and
   exploratory t-SNE figures.
6. Save Prism-ready validation-curve and reliability-bin CSV tables.
7. Make the public-repository Colab notebook safe for restart-and-run-all from
   the Google Colab VS Code extension. Clone or update the repository, verify
   CUDA, mount Google Drive, stop on command failures, and save outputs under a
   persistent Drive base directory.
8. Treat t-SNE of the two-logit representation as qualitative only. Do not use it
   as evidence that label smoothing improves performance.
