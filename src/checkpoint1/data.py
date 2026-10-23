"""Loading utilities for CelebA-Spoof.

Expected layout under ``DATA_ROOT`` (see README for download steps)::

    CelebA_Spoof/
      Data/{train,test}/<subject_id>/{live,spoof}/<img>.{jpg,png}
      Data/{train,test}/<subject_id>/{live,spoof}/<img>_BB.txt
      metas/intra_test/{train,test}_label.json

Each label is a list of 44 ints: [0:40] CelebA attributes, [40] spoof type,
[41] illumination, [42] environment, [43] live(0)/spoof(1).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_ROOT = Path("data/CelebA_Spoof")
FIXTURE_ROOT = Path("data/fixture/CelebA_Spoof")  # written by `python -m checkpoint1.fixture`
SEED = 228
SPLITS = ("train", "test")

SPOOF_TYPES = {
    0: "Live",
    1: "Photo",
    2: "Poster",
    3: "A4",
    4: "Face Mask",
    5: "Upper Body Mask",
    6: "Region Mask",
    7: "PC",
    8: "Pad",
    9: "Phone",
    10: "3D Mask",
}
ILLUMINATION = {0: "None (live)", 1: "Normal", 2: "Strong", 3: "Back", 4: "Dark"}
ENVIRONMENT = {0: "None (live)", 1: "Indoor", 2: "Outdoor"}
CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
    "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
    "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]
EXPECTED_IMAGES = 625_537
EXPECTED_SUBJECTS = 10_177
LABEL_LEN = 44
BBOX_REF = 224  # _BB.txt coordinates are relative to a 224x224 rescale


def resolve_root(root=None) -> Path:
    """DATA_ROOT from the argument, then $CELEBA_SPOOF_ROOT, then the default."""
    return Path(root or os.environ.get("CELEBA_SPOOF_ROOT") or DEFAULT_ROOT)


def label_json(root: Path, split: str) -> Path:
    return root / "metas" / "intra_test" / f"{split}_label.json"


def missing_parts(root=None) -> list[str]:
    """Required dataset paths that do not exist under ``root`` (empty list = usable dataset)."""
    root = resolve_root(root)
    required = [root / "Data"] + [label_json(root, s) for s in SPLITS]
    return [str(p.relative_to(root)) if root.exists() else str(p) for p in required if not p.exists()]


def check_data_root(root=None) -> Path:
    """Return the dataset root, or exit(1) with a readable message if it is incomplete."""
    root = resolve_root(root)
    missing = missing_parts(root)
    if missing:
        msg = (f"dataset not found at {root} (missing: {', '.join(missing)}); "
               "see README for download steps, or run `python -m checkpoint1.fixture` "
               "for a synthetic test set")
        if not missing_parts(FIXTURE_ROOT):
            msg += f"; a synthetic fixture exists at {FIXTURE_ROOT}, pass --data-root {FIXTURE_ROOT} to use it"
        sys.exit(msg)
    return root


def load_labels(split: str, root=None) -> pd.DataFrame:
    """One row per image: path, split, subject_id, label columns and attr_* columns."""
    root = resolve_root(root)
    with open(label_json(root, split)) as f:
        raw = json.load(f)
    paths = list(raw)
    arr = np.asarray([raw[p] for p in paths], dtype=np.int16)
    if arr.ndim != 2 or arr.shape[1] != LABEL_LEN:
        raise ValueError(f"{split}_label.json: expected {LABEL_LEN}-int vectors, got shape {arr.shape}")
    df = pd.DataFrame(arr[:, :40], columns=[f"attr_{a}" for a in CELEBA_ATTRS])
    df.insert(0, "path", paths)
    df.insert(1, "split", split)
    df.insert(2, "subject_id", [Path(p).parts[2] for p in paths])
    df["spoof_type"] = arr[:, 40]
    df["illumination"] = arr[:, 41]
    df["environment"] = arr[:, 42]
    df["live_spoof"] = arr[:, 43]
    df["folder_label"] = [1 if "/spoof/" in p.replace("\\", "/") else 0 for p in paths]
    return df


def load_all_labels(root=None) -> pd.DataFrame:
    return pd.concat([load_labels(s, root) for s in SPLITS], ignore_index=True)


def read_bbox(img_path, img_size=None):
    """Face box (x, y, w, h, score) in real pixels, or None if the _BB.txt is missing/invalid.

    ``img_size`` is (width, height) of the real image; the stored box is relative to 224x224.
    """
    p = Path(img_path)
    bb = p.with_name(p.stem + "_BB.txt")
    try:
        x, y, w, h, score = (float(v) for v in bb.read_text().split()[:5])
    except (OSError, ValueError):
        return None
    if img_size is not None:
        sx, sy = img_size[0] / BBOX_REF, img_size[1] / BBOX_REF
        x, w, y, h = x * sx, w * sx, y * sy, h * sy
    return x, y, w, h, score


def load_image(path) -> np.ndarray:
    """RGB uint8 array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def sample_paths(df: pd.DataFrame, n: int, seed: int = SEED, stratify: str = "spoof_type") -> pd.DataFrame:
    """Fixed-seed sample of ~n rows, allocated equally across strata (capped at stratum size).

    Equal allocation keeps rare attack types visible in the pixel-level stats.
    """
    if n >= len(df):
        return df.copy()
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(stratify)}
    remaining, budget = dict(groups), n
    take = {}
    while remaining and budget > 0:  # water-filling: small strata give back their share
        share = max(budget // len(remaining), 1)
        small = [k for k, g in remaining.items() if len(g) <= share]
        if not small:
            for k in remaining:
                take[k] = share
            break
        for k in small:
            take[k] = len(remaining.pop(k))
            budget -= take[k]
    parts = [g.iloc[np.sort(rng.choice(len(g), size=min(take[k], len(g)), replace=False))]
             for k, g in groups.items() if k in take]
    return pd.concat(parts).sort_index()
