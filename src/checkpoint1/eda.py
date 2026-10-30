"""Exploratory data analysis for CelebA-Spoof.

    uv run python -m checkpoint1.eda [--data-root DIR] [--n-sample N|all] [--seed 228] [--recompute]

Label figures use every image. Pixel statistics (brightness, contrast, sharpness, saturation,
high-frequency energy) are computed on the full image, the face box and the background, on a
seeded sample with the same number of images per spoof type (or on every image with --n-sample all).
They are cached in <out-dir>/image_stats.csv.
"""

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from checkpoint1 import data

LIVE_COLOR = "#2a78d6"
SPOOF_COLOR = "#eb6834"
STAT_SIDE = 256  # images are shrunk to at most 256 px before computing statistics
HF_CUTOFF = 0.25  # cycles/pixel
FEATURES = {
    "bright": "Mean luminance (0-255)",
    "contrast": "Luminance std (contrast)",
    "sharp": "log10 variance of Laplacian (sharpness)",
    "sat": "Mean HSV saturation (0-255)",
    "hf": "High-frequency energy ratio",
}

plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 7,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------- image statistics

def high_freq_ratio(gray):
    """Share of the spectral energy above HF_CUTOFF cycles/pixel."""
    if min(gray.shape) < 8:
        return np.nan
    power = np.abs(np.fft.fft2(gray - gray.mean())) ** 2
    freq_x = np.fft.fftfreq(gray.shape[1])[None, :]
    freq_y = np.fft.fftfreq(gray.shape[0])[:, None]
    radius = np.hypot(freq_x, freq_y)
    total = power.sum()
    if total == 0:
        return np.nan
    return float(power[radius > HF_CUTOFF].sum() / total)


def region_stats(name, gray, saturation, laplacian, mask, hf_image):
    if mask.sum() < 50:
        return {f"{name}_{feature}": np.nan for feature in FEATURES}
    return {
        f"{name}_bright": gray[mask].mean(),
        f"{name}_contrast": gray[mask].std(),
        f"{name}_sharp": laplacian[mask].var(),
        f"{name}_sat": saturation[mask].mean(),
        f"{name}_hf": high_freq_ratio(hf_image),
    }


def image_stats(path):
    """Statistics of one image: size, hashes for duplicate checks, and pixel statistics per region."""
    stats = {"corrupt": False, "file_missing": False, "bbox_missing": False}
    try:
        raw_bytes = Path(path).read_bytes()
    except OSError:
        stats["file_missing"] = True
        return stats
    stats["md5"] = hashlib.md5(raw_bytes).hexdigest()
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
    except Exception:
        stats["corrupt"] = True
        return stats

    width, height = image.size
    stats["w"] = width
    stats["h"] = height
    small = image.copy()
    small.thumbnail((STAT_SIDE, STAT_SIDE))
    scale = small.width / width
    gray = np.asarray(small.convert("L"), dtype=np.float64)
    saturation = np.asarray(small.convert("HSV"), dtype=np.float64)[:, :, 1]
    laplacian = ndimage.laplace(gray)

    # average hash: 16x16 thumbnail thresholded at its mean, for near-duplicate checks
    tiny = np.asarray(small.convert("L").resize((16, 16), Image.BOX), dtype=np.float64)
    stats["ahash"] = np.packbits(tiny > tiny.mean()).tobytes().hex()

    whole = np.ones(gray.shape, dtype=bool)
    stats.update(region_stats("full", gray, saturation, laplacian, whole, gray))

    bbox = data.read_bbox(path, (width, height))
    face = np.zeros(gray.shape, dtype=bool)
    if bbox is None:
        stats["bbox_missing"] = True
        stats.update(region_stats("face", gray, saturation, laplacian, face, gray))
        stats.update(region_stats("bg", gray, saturation, laplacian, face, gray))
        return stats
    box_x, box_y, box_w, box_h, _ = bbox
    left = max(round(box_x * scale), 0)
    top = max(round(box_y * scale), 0)
    right = min(round((box_x + box_w) * scale), gray.shape[1])
    bottom = min(round((box_y + box_h) * scale), gray.shape[0])
    face[top:bottom, left:right] = True
    background = ~face
    # for the background spectrum, fill the face with the mean background value
    background_image = gray.copy()
    if background.any():
        background_image[face] = gray[background].mean()
    stats.update(region_stats("face", gray, saturation, laplacian, face, gray[top:bottom, left:right]))
    stats.update(region_stats("bg", gray, saturation, laplacian, background, background_image))
    return stats


def compute_image_stats(sample, root, out_dir, recompute=False):
    """Pixel statistics for every row of sample, cached in out_dir/image_stats.csv."""
    cache = Path(out_dir) / "image_stats.csv"
    if cache.exists() and not recompute:
        cached = pd.read_csv(cache)
        if list(cached["path"]) == list(sample["path"]):
            print(f"using cached image stats ({cache})")
            return cached
    paths = [str(Path(root) / path) for path in sample["path"]]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        rows = list(tqdm(pool.map(image_stats, paths, chunksize=32), total=len(paths), desc="image stats"))
    stats = pd.DataFrame(rows)
    stats.insert(0, "path", list(sample["path"]))
    stats.to_csv(cache, index=False)
    return stats


def summarize(labels, stats):
    """Table of counts and data-quality problems (schema sanity)."""
    rows = []
    for split, group in labels.groupby("split"):
        rows.append(("counts", f"{split}_images", len(group)))
        rows.append(("counts", f"{split}_subjects", group["subject_id"].nunique()))
        rows.append(("counts", f"{split}_live", (group["live_spoof"] == 0).sum()))
        rows.append(("counts", f"{split}_spoof", (group["live_spoof"] == 1).sum()))
    is_live = labels["live_spoof"] == 0
    train_subjects = set(labels.loc[labels["split"] == "train", "subject_id"])
    test_subjects = set(labels.loc[labels["split"] == "test", "subject_id"])
    rows.append(("anomalies", "duplicate_label_paths", labels["path"].duplicated().sum()))
    rows.append(("anomalies", "label_vs_folder_mismatch", (labels["live_spoof"] != labels["folder_label"]).sum()))
    rows.append(("anomalies", "spoof_type_vs_label_mismatch", ((labels["spoof_type"] == 0) != is_live).sum()))
    rows.append(("anomalies", "subjects_in_both_splits", len(train_subjects & test_subjects)))
    rows.append(("sample", "sampled_images", len(stats)))
    for column in ["corrupt", "file_missing", "bbox_missing"]:
        rows.append(("sample", column, stats[column].sum()))
    readable = stats[~stats["corrupt"] & ~stats["file_missing"]]
    rows.append(("sample", "exact_duplicates_md5", readable["md5"].duplicated().sum()))
    rows.append(("sample", "near_duplicates_ahash", readable["ahash"].duplicated().sum()))
    return pd.DataFrame(rows, columns=["group", "metric", "value"])


# ----------------------------------------------------------------------------- figures

def save(fig, fig_dir, name, title):
    fig.suptitle(title)
    fig.savefig(Path(fig_dir) / f"{name}.png")
    plt.close(fig)


def class_legend(axis, **kwargs):
    handles = [Patch(color=LIVE_COLOR, label="Live"), Patch(color=SPOOF_COLOR, label="Spoof")]
    axis.legend(handles=handles, frameon=False, **kwargs)


def feature_values(stats, column, label):
    """Finite values of one statistic for live (label=0) or spoof (label=1) images; sharpness on a log10 scale."""
    values = stats.loc[stats["live_spoof"] == label, column]
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if column.endswith("_sharp"):
        values = np.log10(values[values > 0])
    return values


def fig_live_spoof(labels, fig_dir):
    counts = labels.groupby(["split", "live_spoof"]).size().unstack(fill_value=0)
    fig, axis = plt.subplots(figsize=(3.5, 2.4))
    positions = np.arange(len(counts))
    axis.bar(positions - 0.2, counts[0], 0.4, color=LIVE_COLOR)
    axis.bar(positions + 0.2, counts[1], 0.4, color=SPOOF_COLOR)
    axis.set_xticks(positions, counts.index)
    axis.set_xlabel("Split")
    axis.set_ylabel("Number of images")
    class_legend(axis)
    save(fig, fig_dir, "eda_live_vs_spoof", "Live vs spoof images per split")


def fig_spoof_types(labels, fig_dir):
    counts = labels["spoof_type"].value_counts().sort_values()
    names = [data.SPOOF_TYPES[spoof_type] for spoof_type in counts.index]
    colors = [LIVE_COLOR if spoof_type == 0 else SPOOF_COLOR for spoof_type in counts.index]
    fig, axis = plt.subplots(figsize=(3.5, 2.8))
    bars = axis.barh(names, counts.values, color=colors)
    axis.bar_label(bars, labels=[f"{count:,}" for count in counts.values], padding=2, fontsize=6)
    axis.margins(x=0.15)
    axis.set_xlabel("Number of images")
    axis.set_ylabel("Image type")
    save(fig, fig_dir, "eda_spoof_types", "Images per presentation type")


def fig_subjects(labels, fig_dir):
    per_subject = labels.groupby("subject_id")["live_spoof"].agg(["size", "mean"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.2), layout="constrained")
    left.hist(per_subject["size"], bins=30, color="gray")
    left.set_xlabel("Images per subject")
    left.set_ylabel("Number of subjects")
    right.hist(per_subject["mean"], bins=20, color=SPOOF_COLOR)
    right.set_xlabel("Fraction of the subject's images that are spoof")
    right.set_ylabel("Number of subjects")
    save(fig, fig_dir, "eda_subjects", "Images and spoof fraction per subject")


def fig_feature_hists(stats, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), layout="constrained")
    for axis, (feature, axis_label) in zip(axes.flat, FEATURES.items()):
        live = feature_values(stats, f"full_{feature}", 0)
        spoof = feature_values(stats, f"full_{feature}", 1)
        bins = np.linspace(min(live.min(), spoof.min()), max(live.max(), spoof.max()), 26)
        axis.hist(live, bins=bins, density=True, color=LIVE_COLOR, alpha=0.55)
        axis.hist(spoof, bins=bins, density=True, color=SPOOF_COLOR, alpha=0.55)
        axis.set_xlabel(axis_label)
        axis.set_ylabel("Density")
    axes.flat[-1].axis("off")
    class_legend(axes.flat[-1], loc="center")
    save(fig, fig_dir, "eda_feature_hists", "Image statistics by class (full image)")


def fig_feature_boxes(stats, fig_dir):
    spoof_types = sorted(stats["spoof_type"].unique())
    names = [data.SPOOF_TYPES[spoof_type] for spoof_type in spoof_types]
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(7.2, 3.2), sharey=True)
    for axis, (feature, axis_label) in zip(axes, FEATURES.items()):
        column = f"full_{feature}"
        values = []
        for spoof_type in spoof_types:
            rows = stats[stats["spoof_type"] == spoof_type]
            values.append(feature_values(rows, column, rows["live_spoof"].iloc[0]))
        boxes = axis.boxplot(values, orientation="horizontal", patch_artist=True, widths=0.6,
                             flierprops={"marker": ".", "markersize": 2}, medianprops={"color": "black"})
        for spoof_type, box in zip(spoof_types, boxes["boxes"]):
            box.set_facecolor(LIVE_COLOR if spoof_type == 0 else SPOOF_COLOR)
        axis.set_yticks(range(1, len(names) + 1), names)
        axis.set_xlabel(axis_label.split(" (")[0], fontsize=6)
    axes[0].set_ylabel("Image type")
    axes[0].invert_yaxis()
    save(fig, fig_dir, "eda_feature_boxes", "Image statistics by presentation type")


def fig_correlation(stats, fig_dir):
    columns = [f"{region}_{feature}" for region in ("full", "face", "bg") for feature in FEATURES]
    columns.append("live_spoof")
    corr = stats[columns].replace([np.inf, -np.inf], np.nan).corr(method="spearman")
    fig, axis = plt.subplots(figsize=(6.4, 5.0))
    sns.heatmap(corr, ax=axis, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".1f",
                annot_kws={"fontsize": 4.5}, square=True, cbar_kws={"label": "Spearman correlation"})
    axis.set_xlabel("Feature")
    axis.set_ylabel("Feature")
    axis.tick_params(labelsize=5)
    save(fig, fig_dir, "eda_correlation", "Spearman correlation of image statistics and label (1 = spoof)")


# ----------------------------------------------------------------------------- driver

def run_eda(root=None, n_sample=10_000, seed=data.SEED, out_dir="outputs/eda", fig_dir="outputs/figures",
            recompute=False):
    """Make the EDA figures and summary.csv. Returns (labels, stats): every label, and the sampled image stats."""
    root = data.check_data_root(root)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    labels = data.load_all_labels(root)
    if n_sample is None or str(n_sample) == "all":
        sample = labels
    else:
        sample = data.sample_per_type(labels, int(n_sample), seed)
    print(f"{len(labels):,} labelled images; pixel statistics on {len(sample):,}")

    stats = compute_image_stats(sample, root, out_dir, recompute)
    keep = ["path", "split", "subject_id", "spoof_type", "illumination", "environment", "live_spoof"]
    stats = stats.merge(sample[keep], on="path")

    summarize(labels, stats).to_csv(Path(out_dir) / "summary.csv", index=False)

    fig_live_spoof(labels, fig_dir)
    fig_spoof_types(labels, fig_dir)
    fig_subjects(labels, fig_dir)
    fig_feature_hists(stats, fig_dir)
    fig_feature_boxes(stats, fig_dir)
    fig_correlation(stats, fig_dir)
    print(f"figures saved to {fig_dir}")
    return labels, stats


def parse_n_sample(text):
    return None if text == "all" else int(text)


def main():
    parser = argparse.ArgumentParser(description="Exploratory data analysis for CelebA-Spoof.")
    parser.add_argument("--data-root", default=None, help="CelebA_Spoof folder (default: $CELEBA_SPOOF_ROOT)")
    parser.add_argument("--n-sample", type=parse_n_sample, default=10_000,
                        help="images used for pixel statistics, or 'all'")
    parser.add_argument("--seed", type=int, default=data.SEED)
    parser.add_argument("--out-dir", default="outputs/eda")
    parser.add_argument("--fig-dir", default="outputs/figures")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached image statistics")
    args = parser.parse_args()
    run_eda(args.data_root, args.n_sample, args.seed, args.out_dir, args.fig_dir, args.recompute)


if __name__ == "__main__":
    main()
