#!/usr/bin/env python3
"""
preprocess.py - Cleaning and preprocessing for CelebA-Spoof (CS 228 project).

One script, five steps. Each step's reason is written into the report
(out/preprocessing_report.md) so the numbers and the justification travel
together:

  1. Missing values / schema: load the label JSONs, check every record
     (label length and ranges, label consistency, image file, bounding-box
     file, image decodes) and drop broken records, counting each reason.
  2. Splits: keep the official subject-disjoint train/test split, carve a
     subject-disjoint validation set out of train (optionally subsample
     train subjects), all with a fixed seed.
  3. Alignment / cropping: convert the provided boxes (stored relative to a
     224x224 resize) to pixel coordinates and derive two square crops:
     a tight FACE crop (face expert) and a wide CONTEXT crop (environment
     expert). No landmark warping.
  4. Normalization: measure per-channel mean/std of face crops on train and
     compare with ImageNet statistics used by the pretrained backbone.
  5. Feature reduction analysis: attribute-label correlation (leakage),
     attribute-attribute correlation (redundancy), PCA on raw pixels and PCA
     on pretrained ResNet-18 embeddings.

Outputs (in --out):
  manifest_train.csv, manifest_val.csv, manifest_test.csv  clean samples + crop boxes
  dropped_samples.csv                                      every dropped record + reason
  normalization.json, attribute_label_correlation.csv,
  redundant_attribute_pairs.csv, pca_summary.json
  figures/*.png
  preprocessing_report.md                                  all counts + reasons
  crops/{face,context}/...png                              only with --save-crops

Usage:
  python preprocess.py --root data/raw/celeba_spoof/CelebA_Spoof --out data/processed --seed 42
  # quick dry run on a seeded 20k-record sample:
  python preprocess.py --root ... --out data/processed_small --max-samples 20000

Requires: numpy, pandas, pillow, matplotlib, scikit-learn
Optional: torch + torchvision (embedding PCA), tqdm (progress bars)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    def tqdm(it, **kwargs):
        return it

# --------------------------------------------------------------------------
# Dataset schema (from the CelebA-Spoof README)
# Label list per image: [0:40] face attributes, [40] spoof type,
# [41] illumination, [42] environment, [43] live/spoof
# --------------------------------------------------------------------------
FACE_ATTRS = [  # standard CelebA attribute order
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
    "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
    "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]
SPOOF_TYPES = ["Live", "Photo", "Poster", "A4", "Face Mask", "Upper Body Mask",
               "Region Mask", "PC", "Pad", "Phone", "3D Mask"]
ILLUMINATION = ["Live", "Normal", "Strong", "Back", "Dark"]
ENVIRONMENT = ["Live", "Indoor", "Outdoor"]
BOX_REF = 224.0  # boxes in *_BB.txt are relative to a 224x224 resize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REASONS = {
    "bad_label_length": "Label list does not have 44 entries, so attributes cannot be read reliably.",
    "label_out_of_range": "An attribute value lies outside its documented range (schema violation).",
    "bad_path": "Path does not follow Data/<split>/<subject>/<live|spoof>/<file>, so subject or class is unknown.",
    "label_conflict": "Folder name, live/spoof label and spoof type disagree; the true class is unknown.",
    "missing_image": "Image listed in the labels is not on disk.",
    "missing_bbox": "No *_BB.txt file, so the face cannot be cropped consistently.",
    "bad_bbox": "Bounding-box file is unreadable or has non-positive width/height.",
    "bbox_outside_image": "Bounding box does not overlap the image at all.",
    "corrupt_image": "Image cannot be decoded.",
}


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Step 1a: labels and schema checks (no file access yet)
# --------------------------------------------------------------------------
def find_label_file(root: str, split: str) -> str:
    candidates = [
        os.path.join(root, "metas", "intra_test", f"{split}_label.json"),
        os.path.join(root, f"{split}_label.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    sys.exit(f"Could not find {split}_label.json under {root} (tried {candidates}). "
             f"Pass --{split}-labels explicitly.")


def load_labels(path: str, split: str) -> tuple[list[dict], list[dict]]:
    with open(path) as f:
        raw = json.load(f)
    kept, dropped = [], []
    for rel, lab in raw.items():
        rel = rel.replace("\\", "/")
        rec = {"path": rel, "split": split}
        if not isinstance(lab, list) or len(lab) != 44:
            dropped.append({**rec, "reason": "bad_label_length"})
            continue
        lab = [int(v) for v in lab]
        attrs, st, il, env, ls = lab[:40], lab[40], lab[41], lab[42], lab[43]
        if (any(a not in (0, 1) for a in attrs) or not 0 <= st < len(SPOOF_TYPES)
                or not 0 <= il < len(ILLUMINATION) or not 0 <= env < len(ENVIRONMENT)
                or ls not in (0, 1)):
            dropped.append({**rec, "reason": "label_out_of_range"})
            continue
        parts = rel.split("/")
        if len(parts) < 4 or parts[-2] not in ("live", "spoof"):
            dropped.append({**rec, "reason": "bad_path"})
            continue
        rec.update(subject=parts[-3], folder=parts[-2], spoof_type=st,
                   illumination=il, environment=env, live_spoof_raw=ls)
        rec.update({FACE_ATTRS[i]: attrs[i] for i in range(40)})
        kept.append(rec)
    return kept, dropped


def resolve_label_encoding(df: pd.DataFrame) -> int:
    """Return the raw value of label[43] that means LIVE, inferred from the
    records whose spoof type is 0 (= Live). Avoids hard-coding an encoding."""
    live_vals = df.loc[df.spoof_type == 0, "live_spoof_raw"]
    if live_vals.empty:
        return 0
    return int(live_vals.mode().iloc[0])


# --------------------------------------------------------------------------
# Step 1b + 3: file checks and crop boxes (runs in worker processes)
# --------------------------------------------------------------------------
def square_box(cx: float, cy: float, side: float) -> tuple[int, int, int, int]:
    half = side / 2.0
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def inspect_sample(task: tuple) -> dict:
    root, rel, verify, margin, ctx_scale, min_face = task
    img_path = os.path.join(root, rel)
    bb_path = os.path.splitext(img_path)[0] + "_BB.txt"
    out = {"path": rel}

    if not os.path.isfile(img_path):
        return {**out, "drop": "missing_image"}
    if not os.path.isfile(bb_path):
        return {**out, "drop": "missing_bbox"}
    try:
        with open(bb_path) as f:
            vals = [float(v) for v in f.read().split()]
        bx, by, bw, bh = vals[:4]
        score = vals[4] if len(vals) > 4 else float("nan")
    except Exception:
        return {**out, "drop": "bad_bbox"}
    if bw <= 0 or bh <= 0:
        return {**out, "drop": "bad_bbox"}

    try:
        with Image.open(img_path) as im:
            W, H = im.size
            mode = im.mode
            if verify == "full":
                im.load()  # full decode catches truncated/corrupt files
    except Exception:
        return {**out, "drop": "corrupt_image"}

    # Boxes are stored relative to a 224x224 resize; scale to pixel coordinates.
    sx, sy = W / BOX_REF, H / BOX_REF
    x, y, w, h = bx * sx, by * sy, bw * sx, bh * sy
    ix = max(0.0, min(x + w, W) - max(x, 0.0))
    iy = max(0.0, min(y + h, H) - max(y, 0.0))
    inside = (ix * iy) / (w * h)
    if inside <= 0:
        return {**out, "drop": "bbox_outside_image"}

    cx, cy, s = x + w / 2, y + h / 2, max(w, h)
    face = square_box(cx, cy, s * (1 + 2 * margin))
    ctx = square_box(cx, cy, s * ctx_scale)
    return {
        **out, "drop": "", "width": W, "height": H, "mode": mode,
        "bbox_x": round(x, 1), "bbox_y": round(y, 1), "bbox_w": round(w, 1),
        "bbox_h": round(h, 1), "bbox_score": score, "bbox_inside_frac": round(inside, 4),
        "face_l": face[0], "face_t": face[1], "face_r": face[2], "face_b": face[3],
        "ctx_l": ctx[0], "ctx_t": ctx[1], "ctx_r": ctx[2], "ctx_b": ctx[3],
        "flag_partial_bbox": inside < 1.0,
        "flag_small_face": max(w, h) < min_face,
        "flag_non_rgb": mode != "RGB",
    }


def load_crop(root: str, row, kind: str = "face", size: int = 224) -> Image.Image:
    """Crop used everywhere (this script and the training dataloader).
    PIL fills any part of the box outside the image with black, so the face
    stays centred. One fixed interpolation (bicubic) for every image."""
    p = "face" if kind == "face" else "ctx"
    box = (int(row[f"{p}_l"]), int(row[f"{p}_t"]), int(row[f"{p}_r"]), int(row[f"{p}_b"]))
    with Image.open(os.path.join(root, row["path"])) as im:
        im = im.convert("RGB")
        return im.crop(box).resize((size, size), Image.BICUBIC)


def save_crop_task(task: tuple) -> str:
    root, out_dir, row, size = task
    try:
        for kind in ("face", "context"):
            dst = os.path.join(out_dir, "crops", kind, os.path.splitext(row["path"])[0] + ".png")
            if os.path.isfile(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            load_crop(root, row, kind, size).save(dst)  # PNG: lossless, keeps spoof artifacts
        return ""
    except Exception as e:  # pragma: no cover
        return f"{row['path']}: {e}"


# --------------------------------------------------------------------------
# Step 2: subject-disjoint splits
# --------------------------------------------------------------------------
def make_splits(df: pd.DataFrame, val_frac: float, train_subject_frac: float,
                seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train_subj = np.array(sorted(df.loc[df.split == "train", "subject"].unique()))
    rng.shuffle(train_subj)
    n_keep = max(1, int(round(len(train_subj) * train_subject_frac)))
    train_subj = train_subj[:n_keep]
    n_val = max(1, int(round(len(train_subj) * val_frac))) if val_frac > 0 else 0
    val_set = set(train_subj[:n_val])
    keep_set = set(train_subj)

    df = df.copy()
    is_train = df.split == "train"
    df = df[~is_train | df.subject.isin(keep_set)].copy()
    df.loc[(df.split == "train") & df.subject.isin(val_set), "split"] = "val"
    return df


# --------------------------------------------------------------------------
# Step 4: normalization statistics
# --------------------------------------------------------------------------
def balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    per = max(1, n // 2)
    parts = []
    for _, g in df.groupby("is_spoof"):
        parts.append(g.sample(n=min(per, len(g)), random_state=seed))
    return pd.concat(parts).sort_values("path").reset_index(drop=True)


def channel_stats(root: str, sample: pd.DataFrame, size: int) -> tuple[list, list]:
    s1 = np.zeros(3)
    s2 = np.zeros(3)
    n = 0
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="norm stats"):
        a = np.asarray(load_crop(root, row, "face", size), dtype=np.float64) / 255.0
        a = a.reshape(-1, 3)
        s1 += a.sum(0)
        s2 += (a ** 2).sum(0)
        n += a.shape[0]
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    return mean.round(4).tolist(), std.round(4).tolist()


# --------------------------------------------------------------------------
# Step 5: feature reduction analysis
# --------------------------------------------------------------------------
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y).to_numpy().astype(float)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    n = ct.sum()
    expected = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    chi2 = ((ct - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def attribute_correlations(train: pd.DataFrame, figdir: str, leak_thr: float,
                           redund_thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    y = train.is_spoof.astype(float)
    for a in FACE_ATTRS:
        x = train[a].astype(float)
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        rows.append({"attribute": a, "kind": "face (binary)", "measure": "pearson_r",
                     "value": r, "abs_value": abs(r), "positive_rate": x.mean()})
    for a in ("spoof_type", "illumination", "environment"):
        v = cramers_v(train[a], train.is_spoof)
        rows.append({"attribute": a, "kind": "anti-spoof (categorical)",
                     "measure": "cramers_v", "value": v, "abs_value": v,
                     "positive_rate": float("nan")})
    corr = pd.DataFrame(rows)
    corr["leakage"] = corr.abs_value >= leak_thr
    corr = corr.sort_values("abs_value", ascending=False).reset_index(drop=True)

    face = train[FACE_ATTRS].astype(float)
    face = face.loc[:, face.std() > 0]
    cm = face.corr()
    pairs = []
    cols = list(cm.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = cm.iat[i, j]
            if abs(r) >= redund_thr:
                pairs.append({"attr_a": cols[i], "attr_b": cols[j], "pearson_r": round(r, 4)})
    pairs = pd.DataFrame(pairs, columns=["attr_a", "attr_b", "pearson_r"])

    if len(cols) > 1:
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(cm.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(cols)), cols, fontsize=7)
        ax.set_title("Face attribute correlation (train)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "face_attribute_correlation.png"), dpi=150)
        plt.close(fig)

    top = corr.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.attribute, top.abs_value, color=np.where(top.leakage, "#c0392b", "#4a6fa5"))
    ax.axvline(leak_thr, color="#555", ls="--", lw=1)
    ax.set_xlabel("|association with live/spoof label|")
    ax.set_title("Attribute-label association (red = leakage)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "attribute_label_association.png"), dpi=150)
    plt.close(fig)
    return corr, pairs


def dims_for(cum: np.ndarray, frac: float) -> int:
    return int(np.searchsorted(cum, frac) + 1)


def pca_analysis(root: str, sample: pd.DataFrame, figdir: str, seed: int,
                 size: int, raw_size: int, skip_embeddings: bool, batch: int) -> dict:
    imgs = [load_crop(root, row, "face", size)
            for _, row in tqdm(sample.iterrows(), total=len(sample), desc="PCA crops")]
    summary: dict = {"n_samples": len(imgs)}

    # (a) PCA on raw pixels: shows how much fine detail is lost if inputs were compressed
    X = np.stack([np.asarray(im.resize((raw_size, raw_size), Image.BICUBIC),
                             dtype=np.float32).ravel() / 255.0 for im in imgs])
    k = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=k, svd_solver="full", random_state=seed).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    summary["raw_pixels"] = {
        "input_dim": int(X.shape[1]), "resolution": raw_size,
        "dims_for_90pct": dims_for(cum, 0.90), "dims_for_95pct": dims_for(cum, 0.95),
        "dims_for_99pct": dims_for(cum, 0.99),
    }
    curves = {"raw pixels": cum}

    # (b) PCA on pretrained ResNet-18 embeddings
    emb = None
    if skip_embeddings:
        summary["embeddings"] = "skipped (--skip-embeddings)"
    else:
        try:
            emb = resnet18_embeddings(imgs, batch)
        except Exception as e:  # missing torch, no internet for weights, ...
            summary["embeddings"] = f"skipped: {type(e).__name__}: {e}"
    if emb is not None:
        k = min(emb.shape)
        pe = PCA(n_components=k, svd_solver="full", random_state=seed).fit(emb)
        cume = np.cumsum(pe.explained_variance_ratio_)
        summary["embeddings"] = {
            "model": "torchvision resnet18 IMAGENET1K_V1, penultimate layer",
            "input_dim": int(emb.shape[1]), "dims_for_90pct": dims_for(cume, 0.90),
            "dims_for_95pct": dims_for(cume, 0.95), "dims_for_99pct": dims_for(cume, 0.99),
        }
        curves["ResNet-18 embeddings"] = cume
        z = pe.transform(emb)[:, :2]
        fig, ax = plt.subplots(figsize=(7, 6))
        types = sample.spoof_type.to_numpy()
        cmap = plt.get_cmap("tab20")
        for t in sorted(set(types)):
            m = types == t
            ax.scatter(z[m, 0], z[m, 1], s=6, alpha=0.6, color=cmap(t % 20),
                       label=f"{SPOOF_TYPES[t]} (n={m.sum()})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Pretrained ResNet-18 embeddings, PCA (face crops)")
        ax.legend(fontsize=7, markerscale=2, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "pca_embeddings_by_spoof_type.png"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in curves.items():
        ax.plot(np.arange(1, len(c) + 1), c, label=name)
    for f in (0.90, 0.95, 0.99):
        ax.axhline(f, color="#999", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA explained variance (train sample)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "pca_explained_variance.png"), dpi=150)
    plt.close(fig)
    return summary


def resnet18_embeddings(imgs: list, batch: int) -> np.ndarray:
    import torch
    import torchvision
    from torchvision import transforms

    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(imgs), batch), desc="embeddings"):
            x = torch.stack([tf(im) for im in imgs[i:i + batch]]).to(device)
            feats.append(model(x).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(path: str, ctx: dict) -> None:
    a = ctx["args"]
    L = []
    w = L.append
    w("# Cleaning and preprocessing report: CelebA-Spoof\n")
    w(f"Generated by `preprocess.py` with seed **{a.seed}**. Every number below is "
      "reproducible by re-running the script with the same arguments:\n")
    w(f"```\n{ctx['command']}\n```\n")

    w("## 1. Missing values and schema checks\n")
    w("For images, \"missing values\" are broken records rather than empty cells. "
      "Each record was checked, and records failing a check were **dropped rather than "
      "imputed**: an image cannot be meaningfully imputed, and the losses are small "
      "relative to the dataset.\n")
    w(f"- Records in label files: **{ctx['n_raw']:,}**")
    if a.max_samples:
        w(f"- Seeded sample processed (`--max-samples {a.max_samples}`): "
          f"**{ctx['n_considered']:,}**")
    w(f"- Dropped: **{ctx['n_dropped']:,}** ({ctx['n_dropped'] / max(ctx['n_considered'], 1):.3%})")
    w(f"- Clean records: **{ctx['n_clean']:,}**\n")
    w("| Drop reason | Count | Why it is dropped |")
    w("|---|---:|---|")
    for r, c in sorted(ctx["drop_counts"].items(), key=lambda kv: -kv[1]):
        w(f"| `{r}` | {c:,} | {REASONS.get(r, '')} |")
    if not ctx["drop_counts"]:
        w("| (none) | 0 | |")
    w("")
    w(f"Label encoding: raw value **{ctx['live_code']}** of label[43] means *live* "
      "(inferred from records whose spoof type is 0 = Live, then checked against "
      "the `live/` and `spoof/` folder names).\n")
    w("Kept but flagged (reported, not dropped):\n")
    for k, v in ctx["flag_counts"].items():
        w(f"- `{k}`: {v:,}")
    w("")

    w("## 2. Splits (subject-disjoint)\n")
    w("The official train/test split is kept because it is subject-disjoint. A "
      f"validation set of **{a.val_frac:.0%} of training subjects** is held out **by "
      "subject**, so no person appears in two splits; splitting by image would leak "
      "identity and inflate validation scores.")
    if a.train_subject_frac < 1:
        w(f" Training uses a seeded **{a.train_subject_frac:.0%} subset of subjects** "
          f"to keep training time manageable ({ctx['n_subsampled_out']:,} clean training "
          "images left out); the test set is untouched.")
    w("\n")
    w("| Split | Subjects | Images | Live | Spoof | Spoof share |")
    w("|---|---:|---:|---:|---:|---:|")
    for s, r in ctx["split_table"].iterrows():
        w(f"| {s} | {r.subjects:,} | {r.images:,} | {r.live:,} | {r.spoof:,} | {r.spoof / r.images:.1%} |")
    w("")
    w("Images per spoof type:\n")
    w("| Spoof type | " + " | ".join(ctx["type_table"].columns) + " |")
    w("|---|" + "---:|" * len(ctx["type_table"].columns))
    for t, r in ctx["type_table"].iterrows():
        w(f"| {t} | " + " | ".join(f"{int(v):,}" for v in r) + " |")
    w("")

    w("## 3. Alignment and cropping\n")
    w("The `*_BB.txt` boxes are stored relative to a 224x224 resize, so each box is "
      "scaled by (width/224, height/224) to pixel coordinates. Two square crops are "
      "derived from every box:\n")
    w(f"- **Face crop** (face expert, baseline): box side x {1 + 2 * a.face_margin:.2f} "
      f"({a.face_margin:.0%} margin per side), resized to {a.size}x{a.size}. *Reason:* the "
      "face expert should see skin texture and facial structure, not background.")
    w(f"- **Context crop** (environment expert): box side x {a.context_scale:.1f}, resized "
      f"to {a.size}x{a.size}. *Reason:* screen bezels, paper edges, glare, hands and "
      "lighting live outside the face box.")
    w("- Areas of a crop outside the image are filled with black, so the face stays "
      "centred and crops keep a fixed scale relative to the face.")
    w("- **No landmark alignment.** *Reason:* rotation and warping resample pixels and "
      "blur the high-frequency cues (moire, print dots, screen grids) that separate "
      "spoofs from live faces, and liveness does not depend on a frontal pose the way "
      "recognition does.")
    w("- One interpolation (bicubic) for every image. Crops saved with `--save-crops` are "
      "PNG (lossless), because JPEG re-encoding would add compression artifacts.\n")
    w(f"Image size (median): {ctx['img_w_med']}x{ctx['img_h_med']} px; "
      f"face box side (median): {ctx['face_med']:.0f} px.\n")
    w("![Image and face sizes](figures/image_and_face_sizes.png)\n")

    w("## 4. Normalization\n")
    n = ctx["norm"]
    w(f"Per-channel statistics of face crops on {n['n_samples']:,} balanced train images:\n")
    w("| | R | G | B |")
    w("|---|---:|---:|---:|")
    w("| Dataset mean | " + " | ".join(f"{v:.3f}" for v in n["dataset_mean"]) + " |")
    w("| ImageNet mean | " + " | ".join(f"{v:.3f}" for v in IMAGENET_MEAN) + " |")
    w("| Dataset std | " + " | ".join(f"{v:.3f}" for v in n["dataset_std"]) + " |")
    w("| ImageNet std | " + " | ".join(f"{v:.3f}" for v in IMAGENET_STD) + " |")
    w("")
    gap = max(abs(d - i) for d, i in zip(n["dataset_mean"], IMAGENET_MEAN))
    closeness = ("The measured means are within "
                 f"{gap:.3f} of ImageNet's, so re-estimating them would change little."
                 if gap <= 0.1 else
                 f"The measured means differ from ImageNet's by up to {gap:.3f}; ImageNet "
                 "statistics are still used so inputs match what the pretrained filters "
                 "expect, and fine-tuning absorbs the remaining shift.")
    w("Pixels are scaled to [0, 1] and then standardized with the **ImageNet** mean/std. "
      "*Reason:* the pretrained ResNet-18 was trained on that input distribution. "
      f"{closeness} Augmentation is kept mild (horizontal flip, small random crop); blur "
      "and JPEG augmentation are avoided because they erase spoof artifacts.\n")

    w("## 5. Feature reduction\n")
    w("### Attribute correlation (leakage and redundancy)\n")
    corr = ctx["corr"]
    leaks = corr[corr.leakage]
    w(f"Association of each annotation with the live/spoof label (|r| or Cramer's V; "
      f"leakage threshold {a.leak_threshold}):\n")
    w("| Attribute | Measure | Value | Leakage |")
    w("|---|---|---:|:---:|")
    for _, r in corr.head(12).iterrows():
        w(f"| {r.attribute} | {r.measure} | {r.value:.3f} | {'yes' if r.leakage else ''} |")
    w("")
    if len(leaks):
        w(f"**Leakage:** {', '.join(leaks.attribute)} encode the label almost exactly "
          "(value 0 means *Live*). They are **never used as model inputs**, only as "
          "auxiliary training targets and for error analysis.\n")
    w(f"Redundant face-attribute pairs (|r| >= {a.redundancy_threshold}): "
      f"**{len(ctx['pairs'])}**. These attributes are not model inputs, so no selection "
      "is applied; the pairs are reported so they are not double-counted in error "
      "analysis.\n")
    w("![Attribute-label association](figures/attribute_label_association.png)\n")

    w("### PCA\n")
    p = ctx["pca"]
    rp = p["raw_pixels"]
    w(f"PCA on {p['n_samples']:,} balanced train face crops.\n")
    w(f"- **Raw pixels** ({rp['resolution']}x{rp['resolution']}x3 = {rp['input_dim']:,} dims): "
      f"{rp['dims_for_90pct']} / {rp['dims_for_95pct']} / {rp['dims_for_99pct']} components "
      "for 90 / 95 / 99% of variance.")
    if isinstance(p["embeddings"], dict):
        e = p["embeddings"]
        w(f"- **ResNet-18 embeddings** ({e['input_dim']} dims): {e['dims_for_90pct']} / "
          f"{e['dims_for_95pct']} / {e['dims_for_99pct']} components for 90 / 95 / 99%.")
    else:
        w(f"- ResNet-18 embeddings: {p['embeddings']}")
    w("")
    w("**Decision: no PCA on the model inputs.** *Reason:* the CNN learns its own features "
      "end to end, and the leading pixel components mostly describe coarse brightness, "
      "colour and pose, while the low-variance tail that PCA would discard holds the "
      "fine texture that distinguishes spoofs. PCA "
      "is used only as an **analysis tool** on embeddings, to see which spoof types "
      "overlap with live faces.\n")
    w("![PCA explained variance](figures/pca_explained_variance.png)\n")
    if isinstance(p["embeddings"], dict):
        w("![PCA of embeddings](figures/pca_embeddings_by_spoof_type.png)\n")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def size_figure(df: pd.DataFrame, figdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in df.groupby("is_spoof"):
        lab = "spoof" if name else "live"
        axes[0].hist(np.sqrt(g.width * g.height), bins=50, alpha=0.6, label=lab)
        axes[1].hist(np.maximum(g.bbox_w, g.bbox_h), bins=50, alpha=0.6, label=lab)
    axes[0].set_xlabel("image size, sqrt(width x height) px")
    axes[1].set_xlabel("face box side, px")
    for ax in axes:
        ax.set_ylabel("images")
        ax.legend()
    fig.suptitle("Image and face sizes by class (clean records)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "image_and_face_sizes.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="CelebA_Spoof folder (contains Data/ and metas/)")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--train-labels", help="override path to train_label.json")
    ap.add_argument("--test-labels", help="override path to test_label.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="process a seeded random subset of records (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.10, help="fraction of train subjects for validation")
    ap.add_argument("--train-subject-frac", type=float, default=1.0,
                    help="fraction of train subjects to keep (seeded)")
    ap.add_argument("--size", type=int, default=224, help="crop output size")
    ap.add_argument("--face-margin", type=float, default=0.15, help="margin per side for face crop")
    ap.add_argument("--context-scale", type=float, default=2.5, help="context crop side / face box side")
    ap.add_argument("--min-face", type=float, default=40, help="flag faces smaller than this (px)")
    ap.add_argument("--verify", choices=["full", "header"], default="full",
                    help="'full' decodes every image (slower, catches corrupt files)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--norm-samples", type=int, default=2000)
    ap.add_argument("--pca-samples", type=int, default=3000)
    ap.add_argument("--pca-raw-size", type=int, default=32, help="resolution for raw-pixel PCA")
    ap.add_argument("--skip-embeddings", action="store_true", help="skip ResNet-18 embedding PCA")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--leak-threshold", type=float, default=0.9)
    ap.add_argument("--redundancy-threshold", type=float, default=0.7)
    ap.add_argument("--save-crops", action="store_true",
                    help="also write face/context PNG crops for every clean sample")
    a = ap.parse_args()

    t0 = time.time()
    set_seeds(a.seed)
    os.makedirs(a.out, exist_ok=True)
    figdir = os.path.join(a.out, "figures")
    os.makedirs(figdir, exist_ok=True)

    # ---- Step 1a: labels -------------------------------------------------
    print("[1/5] Loading labels and checking schema ...")
    records, dropped = [], []
    for split, override in (("train", a.train_labels), ("test", a.test_labels)):
        path = override or find_label_file(a.root, split)
        k, d = load_labels(path, split)
        records += k
        dropped += d
        print(f"  {split}: {len(k) + len(d):,} records from {path}")
    n_raw = len(records) + len(dropped)

    df = pd.DataFrame(records).sort_values("path").reset_index(drop=True)
    if a.max_samples and a.max_samples < len(df):
        # schema drops are reported in full; only valid records are sampled
        df = df.sample(n=a.max_samples, random_state=a.seed).sort_values("path").reset_index(drop=True)
    n_considered = len(df) + len(dropped)

    live_code = resolve_label_encoding(df)
    df["is_spoof"] = (df.live_spoof_raw != live_code).astype(int)
    conflict = ((df.folder == "spoof").astype(int) != df.is_spoof) | \
               ((df.spoof_type != 0).astype(int) != df.is_spoof)
    for _, r in df[conflict].iterrows():
        dropped.append({"path": r.path, "split": r.split, "reason": "label_conflict"})
    df = df[~conflict].reset_index(drop=True)

    # ---- Step 1b + 3: files, boxes, crops -------------------------------
    print(f"[2/5] Checking files and computing crop boxes for {len(df):,} images "
          f"({a.workers} workers, verify={a.verify}) ...")
    tasks = [(a.root, p, a.verify, a.face_margin, a.context_scale, a.min_face) for p in df.path]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            results = list(tqdm(pool.imap(inspect_sample, tasks, chunksize=256),
                                total=len(tasks), desc="inspect"))
    else:
        results = [inspect_sample(t) for t in tqdm(tasks, desc="inspect")]
    info = pd.DataFrame(results)
    df = df.merge(info, on="path", how="left")
    bad = df["drop"].fillna("") != ""
    for _, r in df[bad].iterrows():
        dropped.append({"path": r.path, "split": r.split, "reason": r["drop"]})
    df = df[~bad].drop(columns=["drop"]).reset_index(drop=True)
    for c in ("width", "height", "face_l", "face_t", "face_r", "face_b",
              "ctx_l", "ctx_t", "ctx_r", "ctx_b"):
        df[c] = df[c].astype(int)
    for c in ("flag_partial_bbox", "flag_small_face", "flag_non_rgb"):
        df[c] = df[c].astype(bool)

    drop_df = pd.DataFrame(dropped, columns=["path", "split", "reason"])
    drop_df.sort_values(["reason", "path"]).to_csv(os.path.join(a.out, "dropped_samples.csv"), index=False)
    if df.empty:
        sys.exit("No clean records left; check --root and the label files.")

    # ---- Step 2: splits -------------------------------------------------
    print("[3/5] Building subject-disjoint splits ...")
    n_clean = len(df)
    df = make_splits(df, a.val_frac, a.train_subject_frac, a.seed)
    n_subsampled_out = n_clean - len(df)
    leak = set(df[df.split == "train"].subject) & set(df[df.split == "val"].subject)
    leak |= set(df[df.split != "test"].subject) & set(df[df.split == "test"].subject)
    if leak:
        print(f"  WARNING: {len(leak)} subjects appear in more than one split")

    cols = (["path", "split", "subject", "is_spoof", "spoof_type", "illumination",
             "environment", "width", "height", "mode", "bbox_x", "bbox_y", "bbox_w",
             "bbox_h", "bbox_score", "bbox_inside_frac", "face_l", "face_t", "face_r",
             "face_b", "ctx_l", "ctx_t", "ctx_r", "ctx_b", "flag_partial_bbox",
             "flag_small_face", "flag_non_rgb"] + FACE_ATTRS)
    for s in ("train", "val", "test"):
        df.loc[df.split == s, cols].sort_values("path").to_csv(
            os.path.join(a.out, f"manifest_{s}.csv"), index=False)

    split_table = df.groupby("split").agg(
        subjects=("subject", "nunique"), images=("path", "size"),
        spoof=("is_spoof", "sum")).reindex(["train", "val", "test"]).dropna()
    split_table = split_table.astype(int)
    split_table["live"] = split_table.images - split_table.spoof
    type_table = pd.crosstab(df.spoof_type, df.split)
    type_table = type_table.reindex(columns=[c for c in ("train", "val", "test") if c in type_table.columns])
    type_table.index = [SPOOF_TYPES[i] for i in type_table.index]
    size_figure(df, figdir)

    if a.save_crops:
        print("  Saving PNG crops ...")
        ctasks = [(a.root, a.out, row, a.size) for row in df[cols].to_dict("records")]
        with Pool(a.workers) as pool:
            errs = [e for e in tqdm(pool.imap(save_crop_task, ctasks, chunksize=64),
                                    total=len(ctasks), desc="crops") if e]
        if errs:
            print(f"  {len(errs)} crops failed, e.g. {errs[0]}")

    train = df[df.split == "train"]

    # ---- Step 4: normalization -----------------------------------------
    print("[4/5] Normalization statistics ...")
    ns = balanced_sample(train, a.norm_samples, a.seed)
    mean, std = channel_stats(a.root, ns, a.size)
    norm = {"n_samples": len(ns), "crop": "face", "dataset_mean": mean, "dataset_std": std,
            "used_mean": IMAGENET_MEAN, "used_std": IMAGENET_STD, "pixel_scale": "x / 255"}
    with open(os.path.join(a.out, "normalization.json"), "w") as f:
        json.dump(norm, f, indent=2)

    # ---- Step 5: feature reduction --------------------------------------
    print("[5/5] Feature reduction analysis ...")
    corr, pairs = attribute_correlations(train, figdir, a.leak_threshold, a.redundancy_threshold)
    corr.to_csv(os.path.join(a.out, "attribute_label_correlation.csv"), index=False)
    pairs.to_csv(os.path.join(a.out, "redundant_attribute_pairs.csv"), index=False)
    ps = balanced_sample(train, a.pca_samples, a.seed + 1)
    pca = pca_analysis(a.root, ps, figdir, a.seed, a.size, a.pca_raw_size,
                       a.skip_embeddings, a.batch_size)
    with open(os.path.join(a.out, "pca_summary.json"), "w") as f:
        json.dump(pca, f, indent=2)

    flag_counts = {
        "flag_partial_bbox (box partly outside image)": int(df.flag_partial_bbox.sum()),
        f"flag_small_face (face box < {a.min_face:.0f} px)": int(df.flag_small_face.sum()),
        "flag_non_rgb (converted to RGB)": int(df.flag_non_rgb.sum()),
    }
    ctx = {
        "args": a, "command": "python " + " ".join(sys.argv),
        "n_raw": n_raw, "n_considered": n_considered, "n_dropped": len(drop_df),
        "n_clean": n_clean, "n_subsampled_out": n_subsampled_out, "drop_counts": dict(Counter(drop_df.reason)),
        "live_code": live_code, "flag_counts": flag_counts,
        "split_table": split_table, "type_table": type_table,
        "img_w_med": int(df.width.median()), "img_h_med": int(df.height.median()),
        "face_med": float(np.maximum(df.bbox_w, df.bbox_h).median()),
        "norm": norm, "corr": corr, "pairs": pairs, "pca": pca,
    }
    write_report(os.path.join(a.out, "preprocessing_report.md"), ctx)
    print(f"Done in {time.time() - t0:.0f}s. Report: {os.path.join(a.out, 'preprocessing_report.md')}")


if __name__ == "__main__":
    main()
