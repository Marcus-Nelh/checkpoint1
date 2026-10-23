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
The dataset is not stored in this repository. Download it from
https://github.com/ZhangYuanhan-AI/CelebA-Spoof (follow their access terms and license) and place it as:

```
data/CelebA_Spoof/
  Data/{train,test}/<subject_id>/{live,spoof}/<img>.{jpg,png}  (+ <img>_BB.txt face boxes)
  metas/intra_test/{train,test}_label.json
```

To use another location, set `CELEBA_SPOOF_ROOT` or pass `--data-root`.

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
