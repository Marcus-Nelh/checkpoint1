# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS 228 Project Checkpoint 1: Mixture of Experts face anti-spoofing. The repo is currently a freshly scaffolded skeleton — `src/checkpoint1/__init__.py` only holds a placeholder `main()`, and `data/`, `notebooks/`, and `outputs/` are empty. The README is a stub (environment setup only).

## Environment

- Python 3.14 (`.python-version`), managed with `uv` (`uv.lock`, `uv_build` backend, src layout). Dependencies in `pyproject.toml`: numpy, pandas, scikit-learn, torch.
- `requirements.txt` is empty; `pyproject.toml`/`uv.lock` are the source of truth for dependencies.
- Setup: `uv sync` (or `python -m venv .venv && source .venv/bin/activate`, per the README).
- Run the entry point: `uv run checkpoint1` (script `checkpoint1 = "checkpoint1:main"`).
- No tests, linter, or build config exist yet.

## Layout conventions

- Package code goes in `src/checkpoint1/`; datasets in `data/`; exploratory work in `notebooks/`; results/artifacts in `outputs/`.
