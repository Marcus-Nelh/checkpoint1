# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS 228 Project Checkpoint 1: Mixture of Experts face anti-spoofing. The EDA (Section III of the checkpoint) is implemented; the baseline (Section IV) and the MoE models are not. `src/checkpoint1/__init__.py` still holds a placeholder `main()`. Dataset: CelebA-Spoof, which is not in the repo (see README for the expected `data/CelebA_Spoof/` layout).

## Environment

- Python 3.14 (`.python-version`), managed with `uv` (`uv.lock`, `uv_build` backend, src layout). Dependencies in `pyproject.toml`: numpy, pandas, scikit-learn, torch, matplotlib, seaborn, pillow, scipy, tqdm, plus ipykernel/nbformat/nbconvert for the notebook.
- `requirements.txt` is generated with `uv export --no-hashes --no-dev --no-emit-project > requirements.txt`; `pyproject.toml`/`uv.lock` are the source of truth for dependencies.
- Setup: `uv sync` (or `python -m venv .venv && source .venv/bin/activate`, per the README).
- Run the entry point: `uv run checkpoint1` (script `checkpoint1 = "checkpoint1:main"`).
- Synthetic dataset (no real data needed): `uv run python -m checkpoint1.fixture` writes `data/fixture/CelebA_Spoof`.
- EDA: `uv run python -m checkpoint1.eda [--data-root DIR] [--n-sample N] [--seed 228] [--recompute]` writes figures + `captions.md` to `outputs/figures/` and stats/summary CSVs to `outputs/eda/`. Exits with a readable message if the dataset is missing. For fixture runs pass `--out-dir outputs/fixture/eda --fig-dir outputs/fixture/figures` and `--n-sample 100` (a smaller sample may miss the planted anomalies).
- No tests, linter, or build config exist yet.

## Layout conventions

- Package code goes in `src/checkpoint1/`; datasets in `data/`; exploratory work in `notebooks/`; results/artifacts in `outputs/`.
