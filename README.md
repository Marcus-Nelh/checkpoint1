# CS 228 Project Checkpoint 1: Mixture of Experts Face Anti-Spoofing

## Setup
```bash
uv sync                            # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export CELEBA_SPOOF_ROOT=/path/to/CelebA_Spoof
```

## Dataset (CelebA-Spoof)
[CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) is for non-commercial research only
and is not included in this repository.

**Browser:** download every part (`CelebA_Spoof.zip.001`, `.002`, …) from the Google Drive folder linked in the
[CelebA-Spoof README](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-downloads) into one folder, then
run `7z x CelebA_Spoof.zip.001`. This needs about 2x the ~80 GB archive size in free space.

**Script:** install [rclone](https://rclone.org/install/), run `rclone config` and create a remote named `gdrive`
(storage `drive`, scope `drive.readonly`), then:
```bash
bash scripts/fetch_data.sh data/raw/celeba_spoof
```

The folder that `CELEBA_SPOOF_ROOT` points to must contain `Data/` and `metas/intra_test/{train,test}_label.json`.

## Running
Use seed 228 for every step.

| Step | Command | Notebook |
|---|---|---|
| Cleaning, splits | `uv run python -m checkpoint1.preprocess --out data/processed` | `notebooks/preprocessing.ipynb` |
| EDA | `uv run python -m checkpoint1.eda --n-sample all` | `notebooks/eda.ipynb` |
| Baseline | `uv run python -m checkpoint1.baseline --n-train all --n-val all --n-test all` | `notebooks/baseline.ipynb` |

Run `checkpoint1.preprocess` before the notebooks and the baseline. Each script accepts `--help`. For a quick test, pass a
small `--n-sample` (or `--n-train/--n-val/--n-test`) and a separate output folder, e.g. under `outputs/smoke/`.
