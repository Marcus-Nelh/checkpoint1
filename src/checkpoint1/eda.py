"""Exploratory data analysis for CelebA-Spoof (Checkpoint 1, Section III).

    uv run python -m checkpoint1.eda [--data-root DIR] [--n-sample 10000] [--seed 228] [--recompute]

Label-level figures use every image in the label files. Pixel-level statistics (brightness,
sharpness, frequency content, ...) are computed on a fixed-seed sample stratified by spoof type
and cached in ``<out-dir>/image_stats.csv``. Each statistic is computed on the full image, the
face box and the background (everything outside the box), because the project's two experts see
exactly those two views. Images are downscaled to at most 256 px per side before the pixel
statistics so values are comparable across the very different native resolutions.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from scipy import ndimage, stats
from tqdm import tqdm

from checkpoint1 import data

log = logging.getLogger("checkpoint1.eda")

# Palette: slots 1-2 of the validated categorical order (blue, orange); one hue per class.
LIVE, SPOOF = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SEQ = sns.light_palette(SPOOF, as_cmap=True)
DIV = sns.diverging_palette(250, 25, s=80, l=55, center="light", as_cmap=True)
W1, W2 = 3.5, 7.2  # IEEE single / double column width in inches
STAT_SIDE = 256
HF_CUTOFF = 0.25  # cycles/pixel (half of Nyquist)

plt.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "figure.titlesize": 9,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
})

FEATURES = [  # (suffix, axis label, log10?)
    ("bright", "Mean luminance (0-255)", False),
    ("contrast", "Luminance std (contrast)", False),
    ("sharp", "log10 variance of Laplacian (sharpness)", True),
    ("sat", "Mean HSV saturation (0-255)", False),
    ("hf", "High-frequency energy ratio", False),
]
BOX_TITLES = {"bright": "Brightness", "contrast": "Contrast", "sharp": "Sharpness", "sat": "Saturation", "hf": "High frequency"}
BOX_LABELS = {"bright": "Mean luminance", "contrast": "Luminance std", "sharp": "log10 var. Laplacian",
              "sat": "Mean saturation", "hf": "HF energy ratio"}
REGIONS = ("full", "face", "bg")
ANOMALY_COLS = ["corrupt", "file_missing", "bbox_missing"]


# ----------------------------------------------------------------------------- per-image stats

def _hf_ratio(g: np.ndarray) -> float:
    """Share of spectral energy above ``HF_CUTOFF`` cycles/pixel (proxy for moire/recapture texture)."""
    if min(g.shape) < 8:
        return np.nan
    p = np.abs(np.fft.fft2(g - g.mean())) ** 2
    r = np.hypot(np.fft.fftfreq(g.shape[1])[None, :], np.fft.fftfreq(g.shape[0])[:, None])
    tot = p.sum()
    return float(p[r > HF_CUTOFF].sum() / tot) if tot > 0 else np.nan


def _region(prefix, gray, sat, lap, mask, hf_img):
    n = int(mask.sum())
    if n < 50:
        return {f"{prefix}_{k}": np.nan for k, _, _ in FEATURES}
    return {
        f"{prefix}_bright": float(gray[mask].mean()),
        f"{prefix}_contrast": float(gray[mask].std()),
        f"{prefix}_sharp": float(lap[mask].var()),
        f"{prefix}_sat": float(sat[mask].mean()),
        f"{prefix}_hf": _hf_ratio(hf_img),
    }


def _stats_one(path: str) -> dict:
    out = {"path_abs": path, "corrupt": False, "file_missing": False, "bbox_missing": False}
    try:
        raw = Path(path).read_bytes()
    except OSError:
        out["file_missing"] = True
        return out
    out["md5"] = hashlib.md5(raw).hexdigest()
    try:
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
    except Exception:  # truncated / unreadable image
        out["corrupt"] = True
        return out
    W, H = rgb.size
    out.update(w=W, h=H, aspect=W / H)
    bb = data.read_bbox(path, (W, H))
    th = rgb.copy()
    th.thumbnail((STAT_SIDE, STAT_SIDE))
    s = th.width / W
    arr = np.asarray(th, dtype=np.float64)
    gray = arr @ np.array([0.299, 0.587, 0.114])
    sat = np.asarray(th.convert("HSV"), dtype=np.float64)[..., 1]
    lap = ndimage.laplace(gray)
    small = np.asarray(th.convert("L").resize((16, 16), Image.BOX), dtype=np.float64)
    out["ahash"] = np.packbits(small > small.mean()).tobytes().hex()
    full = np.ones(gray.shape, dtype=bool)
    out.update(_region("full", gray, sat, lap, full, gray))
    if bb is None:
        out["bbox_missing"] = True
        out.update(_region("face", gray, sat, lap, np.zeros_like(full), gray))
        out.update(_region("bg", gray, sat, lap, np.zeros_like(full), gray))
        return out
    x, y, w, h, score = bb
    x0, y0 = max(int(round(x * s)), 0), max(int(round(y * s)), 0)
    x1, y1 = min(int(round((x + w) * s)), gray.shape[1]), min(int(round((y + h) * s)), gray.shape[0])
    out.update(bbox_area_frac=max(x1 - x0, 0) * max(y1 - y0, 0) / (gray.shape[0] * gray.shape[1]), bbox_score=score)
    face = np.zeros_like(full)
    face[y0:y1, x0:x1] = True
    bg = ~face
    bg_img = gray.copy()
    if bg.any():
        bg_img[face] = gray[bg].mean()
    out.update(_region("face", gray, sat, lap, face, gray[y0:y1, x0:x1]))
    out.update(_region("bg", gray, sat, lap, bg, bg_img))
    out["face_minus_bg_bright"] = out["face_bright"] - out["bg_bright"]
    return out


def compute_image_stats(sample: pd.DataFrame, root: Path, out_dir: Path, recompute: bool = False) -> pd.DataFrame:
    """Per-image pixel statistics for the sampled rows, cached in ``out_dir/image_stats.csv``."""
    cache, key_file = out_dir / "image_stats.csv", out_dir / "image_stats.key"
    key = hashlib.md5((str(Path(root).resolve()) + "|".join(sample["path"])).encode()).hexdigest()
    if not recompute and cache.exists() and key_file.exists() and key_file.read_text() == key:
        log.info("using cached image stats (%s)", cache)
        return pd.read_csv(cache)
    paths = [str(Path(root) / p) for p in sample["path"]]
    if len(paths) < 200:
        rows = [_stats_one(p) for p in tqdm(paths, desc="image stats")]
    else:
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            rows = list(tqdm(pool.map(_stats_one, paths, chunksize=32), total=len(paths), desc="image stats"))
    st = pd.DataFrame(rows)
    st.insert(0, "path", list(sample["path"]))
    st = st.drop(columns="path_abs")
    out_dir.mkdir(parents=True, exist_ok=True)
    st.to_csv(cache, index=False)
    key_file.write_text(key)
    return st


# ----------------------------------------------------------------------------- sanity / summary

def summarize(labels: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Long-form (group, metric, value) table: counts and data-quality anomalies."""
    rows = []

    def add(group, metric, value):
        rows.append((group, metric, value))

    for sp, g in labels.groupby("split"):
        add("counts", f"{sp}_images", len(g))
        add("counts", f"{sp}_subjects", g["subject_id"].nunique())
        add("counts", f"{sp}_live", int((g["live_spoof"] == 0).sum()))
        add("counts", f"{sp}_spoof", int((g["live_spoof"] == 1).sum()))
    for t, c in labels["spoof_type"].value_counts().sort_index().items():
        add("spoof_type", data.SPOOF_TYPES.get(t, str(t)), int(c))
    both = set(labels.loc[labels.split == "train", "subject_id"]) & set(labels.loc[labels.split == "test", "subject_id"])
    live = labels["live_spoof"] == 0
    add("anomalies", "duplicate_label_paths", int(labels["path"].duplicated().sum()))
    add("anomalies", "label_vs_folder_mismatch", int((labels["live_spoof"] != labels["folder_label"]).sum()))
    add("anomalies", "spoof_type_vs_label_inconsistent", int(((labels["spoof_type"] == 0) != live).sum()))
    add("anomalies", "live_with_illumination_or_env", int((live & ((labels["illumination"] != 0) | (labels["environment"] != 0))).sum()))
    add("anomalies", "spoof_without_illumination_or_env", int((~live & ((labels["illumination"] == 0) | (labels["environment"] == 0))).sum()))
    add("anomalies", "subjects_in_both_splits", len(both))
    add("sample", "sampled_images", len(df))
    for c in ANOMALY_COLS:
        add("sample", c, int(df[c].sum()))
    ok = df[~df["corrupt"] & ~df["file_missing"]]
    add("sample", "exact_duplicates_md5", int(ok["md5"].duplicated().sum()))
    add("sample", "near_duplicates_ahash", int(ok["ahash"].duplicated().sum()))
    return pd.DataFrame(rows, columns=["group", "metric", "value"])


# ----------------------------------------------------------------------------- figure plumbing

class Report:
    """Saves figures and collects (file, title, caption) for captions.md."""

    def __init__(self, fig_dir: Path):
        self.dir, self.items = Path(fig_dir), []
        self.dir.mkdir(parents=True, exist_ok=True)

    def add(self, fig, name: str, title: str, caption: str):
        fig.suptitle(title, **({} if fig.get_layout_engine() is not None else {"y": 1.02}))
        fig.savefig(self.dir / f"{name}.png")
        plt.close(fig)
        self.items.append((f"{name}.png", title, caption))
        log.info("saved %s", name)

    def write_captions(self):
        lines = ["# EDA figure captions (review and edit before using in the report)\n"]
        for i, (f, t, c) in enumerate(self.items, 1):
            lines.append(f"## Figure {i}: {t}\n`{f}`\n\n{c}\n")
        (self.dir / "captions.md").write_text("\n".join(lines))


def _na(ax):
    ax.text(0.5, 0.5, "n/a (no data)", transform=ax.transAxes, ha="center", va="center", color=INK2)


def _cls(df, col, transform=None):
    """(live values, spoof values) with NaN/inf dropped; optional log10 transform."""
    out = []
    for v in (0, 1):
        s = df.loc[df["live_spoof"] == v, col].replace([np.inf, -np.inf], np.nan).dropna()
        if transform:
            s = np.log10(s[s > 0])
        out.append(s)
    return out


def _class_legend(ax, **kw):
    ax.legend(handles=[Patch(color=LIVE, label="Live"), Patch(color=SPOOF, label="Spoof")], frameon=False, **kw)


def _spearman(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 3 or a[m].nunique() < 2 or b[m].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(a[m], b[m])[0])


def _count_labels(ax, bars, fmt="{:,}"):
    ax.bar_label(bars, labels=[fmt.format(int(b.get_width() if b.get_width() else 0)) for b in bars], padding=2, fontsize=6, color=INK2)


# ----------------------------------------------------------------------------- figures: class balance

def fig_live_spoof(labels, rep):
    t = labels.groupby(["split", "live_spoof"]).size().unstack(fill_value=0)
    t = t.reindex(index=list(data.SPLITS), columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    x, w = np.arange(len(t)), 0.38
    for j, (col, c) in enumerate([(0, LIVE), (1, SPOOF)]):
        pos = x + (j - 0.5) * w
        ax.bar(pos, t[col], w - 0.03, color=c, label=("Live", "Spoof")[j])
        for xi, v, tot in zip(pos, t[col], t.sum(axis=1)):
            ax.annotate(f"{v:,}\n({v / tot:.0%})" if tot else "0", (xi, v), ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x, [s.capitalize() for s in t.index])
    ax.margins(y=0.22)
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of images")
    ax.set_title("Live vs spoof images per split")
    _class_legend(ax, loc="upper right")
    live, spoof = int(t[0].sum()), int(t[1].sum())
    r = spoof / live if live else float("nan")
    rep.add(fig, "eda_01_live_vs_spoof", "Class balance: live vs spoof",
            f"Spoof:live ratio is {r:.1f}:1 ({spoof:,} spoof vs {live:,} live), so we report per-class error rates "
            "(APCER/BPCER) and use class-weighted loss or balanced sampling instead of plain accuracy.")


def fig_spoof_types(labels, rep):
    c = labels["spoof_type"].value_counts().reindex(sorted(data.SPOOF_TYPES), fill_value=0)
    order = c.sort_values()
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    colors = [LIVE if k == 0 else SPOOF for k in order.index]
    bars = ax.barh([data.SPOOF_TYPES[k] for k in order.index], order.values, color=colors, height=0.7)
    ax.bar_label(bars, labels=[f"{v:,}" for v in order.values], padding=2, fontsize=6, color=INK2)
    nz = order[order > 0]
    if len(nz) and nz.max() / nz.min() > 50:
        ax.set_xscale("log")
    ax.margins(x=0.15)
    ax.set_xlabel("Number of images" + (" (log scale)" if ax.get_xscale() == "log" else ""))
    ax.set_ylabel("Image type")
    ax.set_title("Images per presentation type")
    sp = c.drop(0)
    lo, hi = sp.min(), sp.max()
    if lo > 0 and hi / lo < 1.5:
        cap = (f"Attack types are roughly balanced ({lo:,} to {hi:,} images each), so per-attack metrics can be compared "
               "without reweighting the attack mix.")
    else:
        cap = (f"Attack types are uneven, from {hi:,} ({data.SPOOF_TYPES[sp.idxmax()]}) down to {lo:,} "
               f"({data.SPOOF_TYPES[sp.idxmin()]}), so per-attack metrics and balanced attack sampling are needed "
               "to avoid hiding weak attack types.")
    rep.add(fig, "eda_02_spoof_types", "Class balance: attack types", cap)


def fig_conditions(labels, rep):
    sp = labels[labels["live_spoof"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.2), layout="constrained")
    shares = {}
    for ax, col, names, lab in [(axes[0], "illumination", data.ILLUMINATION, "Illumination"),
                                (axes[1], "environment", data.ENVIRONMENT, "Environment")]:
        keys = [k for k in names if k != 0]
        c = sp[col].value_counts().reindex(keys, fill_value=0)
        ax.bar([names[k] for k in keys], c.values, color=SPOOF, width=0.7)
        ax.bar_label(ax.containers[0], labels=[f"{v:,}" for v in c.values], padding=2, fontsize=5.5, color=INK2)
        ax.margins(y=0.15)
        ax.set_xlabel(lab)
        ax.set_ylabel("Number of spoof images")
        ax.set_title(lab, fontsize=7)
        ax.tick_params(axis="x", labelrotation=30)
        shares[col] = (c / c.sum()).max() if c.sum() else np.nan
    rep.add(fig, "eda_03_conditions", "Class balance: capture conditions (spoof only)",
            "Illumination and environment are labeled only for spoof images (live images are 0), so the environment "
            f"expert's auxiliary target covers spoofs only; the most common illumination class holds {shares['illumination']:.0%} of them.")


def fig_subjects(labels, rep):
    g = labels.groupby("subject_id")["live_spoof"].agg(["size", "mean"])
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), layout="constrained")
    axes[0].hist(g["size"], bins=30, color=INK2, rwidth=0.9)
    axes[0].set_xlabel("Images per subject")
    axes[0].set_ylabel("Number of subjects")
    axes[0].set_title("Images per subject")
    axes[1].hist(g["mean"], bins=np.linspace(0, 1, 21), color=SPOOF, rwidth=0.9)
    axes[1].set_xlabel("Fraction of the subject's images that are spoof")
    axes[1].set_ylabel("Number of subjects")
    axes[1].set_title("Spoof fraction per subject")
    all_spoof = int((g["mean"] == 1).sum())
    rep.add(fig, "eda_04_subjects", "Class balance: subjects",
            f"Median subject has {g['size'].median():.0f} images (max {g['size'].max()}) and {all_spoof} subjects have no live "
            "image, so splits must keep each subject inside one partition and sampling should be per subject.")


# ----------------------------------------------------------------------------- figures: univariate

def fig_resolution(df, rep):
    d = df[~df["corrupt"] & df["w"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), layout="constrained")
    if len(d):
        hb = axes[0].hexbin(d["w"], d["h"], gridsize=18, cmap=SEQ, mincnt=1)
        fig.colorbar(hb, ax=axes[0], label="Number of images")
        axes[1].hist(d["aspect"], bins=30, color=INK2, rwidth=0.9)
    else:
        _na(axes[0]); _na(axes[1])
    axes[0].set_xlabel("Width (px)"); axes[0].set_ylabel("Height (px)"); axes[0].set_title("Image size")
    axes[1].set_xlabel("Aspect ratio (width / height)"); axes[1].set_ylabel("Number of images"); axes[1].set_title("Aspect ratio")
    rho = _spearman(d["w"] * d["h"], d["live_spoof"]) if len(d) else np.nan
    cap = (f"Image sizes span {d['w'].min():.0f}-{d['w'].max():.0f} px wide (median {d['w'].median():.0f}x{d['h'].median():.0f}), "
           f"so inputs must be resized to one shape; the pixel-count vs label Spearman rho is {rho:+.2f}, "
           "which shows how far raw resolution alone could leak the label." if len(d) else "No readable images in the sample.")
    rep.add(fig, "eda_05_resolution", "Image resolution", cap)


def fig_feature_hists(df, rep):
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.0), layout="constrained")
    best = ("", 0.0)
    for ax, (k, lab, lg) in zip(axes.flat, FEATURES):
        live, spoof = _cls(df, f"full_{k}", lg)
        if len(live) and len(spoof):
            lo, hi = min(live.min(), spoof.min()), max(live.max(), spoof.max())
            bins = np.linspace(lo, hi, 26) if hi > lo else 10
            for s, c in ((live, LIVE), (spoof, SPOOF)):
                ax.hist(s, bins=bins, density=True, color=c, alpha=0.55, histtype="stepfilled", linewidth=0)
            ks = stats.ks_2samp(live, spoof).statistic
            if ks > best[1]:
                best = (BOX_TITLES[k].lower(), ks)
        else:
            _na(ax)
        ax.set_xlabel(lab); ax.set_ylabel("Density"); ax.set_title(lab.split(" (")[0], fontsize=7)
    last = axes.flat[-1]
    last.axis("off")
    _class_legend(last, loc="center")
    verdict = ("no single image-level statistic separates the classes" if best[1] < 0.5
               else "that statistic separates the classes partly, but the distributions still overlap")
    cap = (f"The largest live/spoof gap is in {best[0]} (KS={best[1]:.2f}, where 0 = identical and 1 = fully separable), "
           f"so {verdict}." if best[0] else "Too few images per class for distributions.")
    rep.add(fig, "eda_06_feature_hists", "Univariate: image statistics by class (full image)", cap)


def fig_feature_boxes(df, rep):
    types = [t for t in sorted(data.SPOOF_TYPES) if (df["spoof_type"] == t).any()]
    fig, axes = plt.subplots(1, 5, figsize=(W2, 3.2), sharey=True)
    top = None
    for ax, (k, lab, lg) in zip(axes, FEATURES):
        col = f"full_{k}"
        series = []
        for t in types:
            s = df.loc[df["spoof_type"] == t, col].replace([np.inf, -np.inf], np.nan).dropna()
            series.append(np.log10(s[s > 0]) if lg else s)
        if not any(len(s) for s in series):
            _na(ax)
        else:
            pos = np.arange(len(types))
            keep = [i for i, s in enumerate(series) if len(s)]
            bp = ax.boxplot([series[i] for i in keep], positions=pos[keep], orientation="horizontal", patch_artist=True, widths=0.6,
                            flierprops=dict(marker=".", markersize=2, markeredgecolor=INK2),
                            medianprops=dict(color=INK), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
            for i, patch in zip(keep, bp["boxes"]):
                patch.set(facecolor=LIVE if types[i] == 0 else SPOOF, alpha=0.7, edgecolor=INK2)
            if k == "hf":
                med = {types[i]: series[i].median() for i in keep}
                top = max(med, key=med.get)
        ax.set_yticks(np.arange(len(types)), [data.SPOOF_TYPES[t] for t in types])
        ax.set_xlabel(BOX_LABELS[k], fontsize=6)
        ax.set_title(BOX_TITLES[k], fontsize=7)
    axes[0].set_ylabel("Image type")
    axes[0].invert_yaxis()
    fig.legend(handles=[Patch(color=LIVE, label="Live"), Patch(color=SPOOF, label="Spoof")], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    cap = ("Spread differs by presentation type" + (f"; {data.SPOOF_TYPES[top]} has the highest median high-frequency ratio" if top is not None else "")
           + ", so attack types leave different low-level footprints and results should be reported per attack.")
    rep.add(fig, "eda_07_feature_boxes", "Univariate: image statistics by presentation type", cap)


def fig_bbox(df, rep):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), layout="constrained")
    meds = {}
    for ax, col, lab in [(axes[0], "bbox_area_frac", "Face box area / image area"), (axes[1], "bbox_score", "Face box detector score")]:
        live, spoof = _cls(df, col)
        if len(live) + len(spoof):
            allv = pd.concat([live, spoof])
            bins = np.linspace(allv.min(), allv.max(), 26) if allv.max() > allv.min() else 10
            for s, c in ((live, LIVE), (spoof, SPOOF)):
                if len(s):
                    ax.hist(s, bins=bins, density=True, color=c, alpha=0.55, histtype="stepfilled", linewidth=0)
        else:
            _na(ax)
        meds[col] = (live.median(), spoof.median())
        ax.set_xlabel(lab); ax.set_ylabel("Density"); ax.set_title(lab)
    fig.legend(handles=[Patch(color=LIVE, label="Live"), Patch(color=SPOOF, label="Spoof")], loc="outside lower center", ncol=2, frameon=False)
    a, l, s = df["bbox_area_frac"].median(), *meds["bbox_area_frac"]
    rep.add(fig, "eda_08_bbox", "Univariate: face box size and score",
            f"The face box covers a median {a:.0%} of the frame (live {l:.0%} vs spoof {s:.0%}), leaving about {1 - a:.0%} of "
            "pixels as background, which is what the environment expert sees.")


# ----------------------------------------------------------------------------- figures: bivariate

def fig_corr(df, rep):
    cols = [f"{r}_{k}" for r in REGIONS for k, _, _ in FEATURES] + ["bbox_area_frac", "face_minus_bg_bright", "live_spoof"]
    cols = [c for c in cols if c in df and df[c].notna().sum() > 2 and df[c].nunique() > 1]
    c = df[cols].replace([np.inf, -np.inf], np.nan).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sns.heatmap(c, ax=ax, cmap=DIV, vmin=-1, vmax=1, center=0, annot=True, fmt=".1f", annot_kws={"fontsize": 4.5},
                linewidths=0.4, linecolor="white", cbar_kws={"label": "Spearman correlation"}, square=True)
    ax.grid(False)
    ax.set_xlabel("Feature"); ax.set_ylabel("Feature")
    ax.set_title("Spearman correlation of image statistics and label (1 = spoof)")
    ax.tick_params(labelsize=5)
    if "live_spoof" in c:
        lab = c["live_spoof"].drop("live_spoof").abs()
        feat = lab.idxmax()
        npairs = int(((c.abs() > 0.8).sum().sum() - len(c)) // 2)
        cap = (f"Strongest rank correlation with the label is {feat} (rho={c.loc[feat, 'live_spoof']:+.2f}) and {npairs} feature "
               "pairs have |rho|>0.8, so those features are largely redundant.")
    else:
        cap = "Label has a single class in the sample; no label correlations."
    rep.add(fig, "eda_09_correlation", "Bivariate: feature correlation matrix", cap)


def fig_face_vs_bg(df, rep):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9))
    for ax, k, lab in [(axes[0], "sharp", "sharpness (variance of Laplacian)"), (axes[1], "hf", "high-frequency energy ratio")]:
        d = df[[f"face_{k}", f"bg_{k}", "live_spoof"]].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[(d[f"face_{k}"] > 0) & (d[f"bg_{k}"] > 0)]
        if len(d):
            for v, c in ((0, LIVE), (1, SPOOF)):
                s = d[d["live_spoof"] == v]
                ax.scatter(s[f"bg_{k}"], s[f"face_{k}"], s=8, color=c, alpha=0.6, edgecolors="white", linewidths=0.3)
            lo, hi = d[[f"face_{k}", f"bg_{k}"]].min().min(), d[[f"face_{k}", f"bg_{k}"]].max().max()
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
            ax.set_xscale("log"); ax.set_yscale("log")
        else:
            _na(ax)
        ax.set_xlabel(f"Background {lab}"); ax.set_ylabel(f"Face {lab}"); ax.set_title(f"Face vs background {lab.split(' (')[0]}")
    _class_legend(axes[0], loc="upper left")
    d = df[["face_sharp", "bg_sharp", "live_spoof"]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d["bg_sharp"] > 0]
    ratio = (d["face_sharp"] / d["bg_sharp"]).groupby(d["live_spoof"]).median()
    rl, rs = ratio.get(0, np.nan), ratio.get(1, np.nan)
    rep.add(fig, "eda_10_face_vs_background", "Bivariate: face vs background statistics",
            f"Median face/background sharpness ratio is {rl:.2f} for live vs {rs:.2f} for spoof (dashed line = equal); "
            "the gap, or lack of one, shows how much complementary signal separate face and environment experts can use.")


def fig_crosstabs(labels, rep):
    sp = labels[labels["live_spoof"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2), sharey=True)
    best = ("", "", 0.0)
    for ax, col, names, lab in [(axes[0], "illumination", data.ILLUMINATION, "Illumination"),
                                (axes[1], "environment", data.ENVIRONMENT, "Environment")]:
        ct = pd.crosstab(sp["spoof_type"], sp[col])
        ct = ct.reindex(index=[t for t in sorted(data.SPOOF_TYPES) if t != 0], columns=[k for k in names if k != 0], fill_value=0)
        share = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
        sns.heatmap(share, ax=ax, cmap=SEQ, vmin=0, vmax=1, annot=True, fmt=".0%", annot_kws={"fontsize": 5.5},
                    linewidths=0.4, linecolor="white", cbar=ax is axes[1], cbar_kws={"label": "Share of attack type"},
                    xticklabels=[names[k] for k in share.columns], yticklabels=[data.SPOOF_TYPES[t] for t in share.index])
        ax.grid(False)
        ax.set_xlabel(lab); ax.set_ylabel("Attack type" if ax is axes[0] else ""); ax.set_title(f"Attack type x {lab.lower()} (row-normalized)")
        if share.notna().any().any():
            t, k = share.stack().idxmax()
            if share.loc[t, k] > best[2]:
                best = (data.SPOOF_TYPES[t], names[k], float(share.loc[t, k]))
    if not best[0]:
        cap = "No spoof images; nothing to cross-tabulate."
    elif best[2] >= 0.6:
        cap = (f"Attack types are unevenly spread over capture conditions (e.g. {best[0]} is {best[2]:.0%} {best[1].lower()}), "
               "so lighting and environment are partly confounded with attack type and results should be reported per condition.")
    else:
        cap = (f"No attack type is concentrated in one condition (largest share: {best[0]} at {best[2]:.0%} {best[1].lower()}), "
               "so lighting and environment are only weakly confounded with attack type.")
    rep.add(fig, "eda_11_attack_vs_conditions", "Bivariate: attack type vs capture conditions", cap)


def fig_attr_corr(labels, rep):
    attrs = [c for c in labels if c.startswith("attr_")]
    r = pd.Series({a[5:]: labels[a].corr(labels["live_spoof"]) for a in attrs if labels[a].nunique() > 1 and labels["live_spoof"].nunique() > 1})
    r = r.dropna().sort_values()
    fig, ax = plt.subplots(figsize=(W1, 3.6))
    if len(r):
        sel = pd.concat([r.head(10), r.tail(10)]).drop_duplicates()
        ax.barh(sel.index, sel.values, color=[SPOOF if v > 0 else LIVE for v in sel.values], height=0.7)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.legend(handles=[Patch(color=SPOOF, label="More spoof"), Patch(color=LIVE, label="More live")], frameon=False, loc="lower right")
        top = r.abs().idxmax()
        verdict = ("all attributes are only weakly tied to the label, so they are unlikely shortcut features" if r.abs().max() < 0.1
                   else "a large |r| flags a possible shortcut or demographic confound to audit")
        cap = f"Strongest attribute association is {top} (r={r[top]:+.2f}); {verdict}."
    else:
        _na(ax)
        cap = "Not enough label variation to correlate attributes."
    ax.set_xlabel("Point-biserial correlation with label (1 = spoof)")
    ax.set_ylabel("CelebA attribute")
    ax.set_title("Attribute vs spoof label (10 lowest, 10 highest)")
    rep.add(fig, "eda_12_attribute_correlation", "Bivariate: face attributes vs label", cap)


# ----------------------------------------------------------------------------- figure: qualitative

def fig_samples(df, root, rep, seed, per_type=3):
    types = [t for t in sorted(data.SPOOF_TYPES) if (df["spoof_type"] == t).any()]
    good = df[~df["corrupt"] & ~df["file_missing"]]
    fig, axes = plt.subplots(len(types), per_type, figsize=(W1, 0.75 * len(types)), squeeze=False)
    rng = np.random.default_rng(seed)
    for i, t in enumerate(types):
        pool = good[good["spoof_type"] == t]
        pick = pool.iloc[np.sort(rng.choice(len(pool), size=min(per_type, len(pool)), replace=False))] if len(pool) else pool
        for j in range(per_type):
            ax = axes[i, j]
            ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j < len(pick):
                p = Path(root) / pick.iloc[j]["path"]
                im = Image.open(p).convert("RGB")
                ax.imshow(im)
                bb = data.read_bbox(p, im.size)
                if bb:
                    ax.add_patch(Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, ec=LIVE if t == 0 else SPOOF, lw=0.8))
            if j == 0:
                ax.set_ylabel(data.SPOOF_TYPES[t], rotation=0, ha="right", va="center", fontsize=6)
            if i == len(types) - 1:
                ax.set_xlabel(f"sample {j + 1}", fontsize=6)
    fig.subplots_adjust(top=0.97, bottom=0.03, hspace=0.08, wspace=0.05)
    rep.add(fig, "eda_13_samples", "Qualitative: random samples per type (box = face region)",
            "Random samples per presentation type with the annotated face box; compare the cues inside the box (texture) "
            "with those outside it (glare, borders, lighting) that the face and environment experts would use.")


# ----------------------------------------------------------------------------- driver

def run_eda(root=None, n_sample=10_000, seed=data.SEED, out_dir="outputs/eda", fig_dir="outputs/figures", recompute=False):
    root = data.check_data_root(root)  # exits with a readable message if the dataset is missing
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = data.load_all_labels(root)
    n_sub = labels["subject_id"].nunique()
    log.info("labels: %s images, %s subjects", f"{len(labels):,}", f"{n_sub:,}")
    if len(labels) != data.EXPECTED_IMAGES or n_sub != data.EXPECTED_SUBJECTS:
        log.warning("expected ~%s images and %s subjects (published); got %s and %s",
                    f"{data.EXPECTED_IMAGES:,}", f"{data.EXPECTED_SUBJECTS:,}", f"{len(labels):,}", f"{n_sub:,}")
    sample = data.sample_paths(labels, n_sample, seed)
    st = compute_image_stats(sample, root, out_dir, recompute)
    keep = ["path", "split", "subject_id", "spoof_type", "illumination", "environment", "live_spoof"]
    df = st.merge(sample[keep], on="path", how="left")
    for c in ANOMALY_COLS:
        df[c] = df[c].fillna(False).astype(bool)

    summary = summarize(labels, df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    log.info("summary:\n%s", summary[summary.group.isin(["anomalies", "sample"])].to_string(index=False))

    rep = Report(fig_dir)
    fig_live_spoof(labels, rep)
    fig_spoof_types(labels, rep)
    fig_conditions(labels, rep)
    fig_subjects(labels, rep)
    fig_resolution(df, rep)
    fig_feature_hists(df, rep)
    fig_feature_boxes(df, rep)
    fig_bbox(df, rep)
    fig_corr(df, rep)
    fig_face_vs_bg(df, rep)
    fig_crosstabs(labels, rep)
    fig_attr_corr(labels, rep)
    fig_samples(df, root, rep, seed)
    rep.write_captions()
    log.info("wrote %d figures + captions.md to %s", len(rep.items), fig_dir)
    return labels, df, rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=None, help="CelebA_Spoof root (default: $CELEBA_SPOOF_ROOT or data/CelebA_Spoof)")
    ap.add_argument("--n-sample", type=int, default=10_000, help="images used for pixel-level statistics")
    ap.add_argument("--seed", type=int, default=data.SEED)
    ap.add_argument("--out-dir", default="outputs/eda")
    ap.add_argument("--fig-dir", default="outputs/figures")
    ap.add_argument("--recompute", action="store_true", help="ignore the cached image statistics")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    run_eda(a.data_root, a.n_sample, a.seed, a.out_dir, a.fig_dir, a.recompute)


if __name__ == "__main__":
    main()
