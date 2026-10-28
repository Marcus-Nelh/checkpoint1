# CS 228 Project Checkpoint 1: Mixture of Experts Face Anti-Spoofing

## Environment Setup
1. Clone the repository.
2. Create and activate a Python virtual environment (e.g., using `uv` or `venv`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies: `uv sync` (or `pip install -r requirements.txt`).

## Dataset (CelebA-Spoof)

This project uses [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof)
(non-commercial research use only; not redistributed in this repo).

1. Install rclone (see https://rclone.org/install/).
2. Configure a Google Drive remote named `gdrive`:
   `rclone config` → new remote → name `gdrive` → storage `drive` → scope `drive.readonly`.
3. Download (~80 GB, needs ~2x free disk space):
   ```bash
   bash scripts/download_celeba_spoof.sh data/raw/celeba_spoof
   ```
4. Verify: `sha256sum -c data/raw/celeba_spoof/SHA256SUMS.txt`

### Synthetic test data
Until the real data is downloaded, a small fake dataset with the same layout and label schema
(plus a few planted anomalies) can be generated to exercise the whole pipeline:

```bash
uv run python -m checkpoint1.fixture            # writes data/fixture/CelebA_Spoof
```

The notebook uses this fixture automatically when `data/CelebA_Spoof` is absent (results go to `outputs/fixture/`).

## Exploratory data analysis
```bash
uv run python -m checkpoint1.eda                # real data, 10k-image pixel-stat sample, seed 228
uv run python -m checkpoint1.eda --data-root data/fixture/CelebA_Spoof --n-sample 100 \
    --out-dir outputs/fixture/eda --fig-dir outputs/fixture/figures   # synthetic data
```
Outputs: 13 figures and `captions.md` in `outputs/figures/`; `image_stats.csv` and `summary.csv`
(counts and data-quality anomalies) in `outputs/eda/`. `notebooks/01_eda.ipynb` runs the same
code and shows the figures inline. Pixel statistics use a fixed-seed sample (`--seed`, default 228);
label statistics use every image.
