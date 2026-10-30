# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS 228 Project Checkpoint 1: Mixture of Experts face anti-spoofing on CelebA-Spoof (not in the repo; see README). Done: cleaning/splits (`checkpoint1.preprocess`), EDA (`checkpoint1.eda`) and a ResNet-18 baseline (`checkpoint1.baseline`). The MoE models are not implemented, and `src/checkpoint1/__init__.py` holds a placeholder `main()`.

## Environment

- Python 3.14, managed with `uv` (`pyproject.toml`/`uv.lock` are the source of truth). Regenerate `requirements.txt` with `uv export --no-hashes --no-dev --no-emit-project > requirements.txt`.
- The dataset root comes from `--root`/`--data-root` or `$CELEBA_SPOOF_ROOT`.
- `python -m checkpoint1.preprocess [--n-sample N|all]` writes `manifest_{train,val,test}.csv` and `dropped_samples.csv` to `data/processed/`. The EDA and baseline need these files.
- `python -m checkpoint1.eda [--n-sample N|all]` writes figures to `outputs/figures/`, plus `image_stats.csv` (cache) and `summary.csv` to `outputs/eda/`.
- `python -m checkpoint1.baseline [--n-train|--n-val|--n-test N|all] [--epochs 3] [--recompute]` writes to `outputs/baseline/`. The model and scores are reused while `config.json` matches. In Docker, use `--workers 0`, because the 64 MB `/dev/shm` is too small for DataLoader workers.
- Notebooks `preprocessing`, `eda` and `baseline` call the code above and hold all report numbers. Execute them with `uv run jupyter nbconvert --execute`.
- No tests or linter.

## Conventions

- Keep the code simple and readable. Use descriptive variable names, few comments, and only what the checkpoint spec asks for.
- Package code goes in `src/checkpoint1/`, datasets in `data/`, notebooks in `notebooks/`, results in `outputs/`.
