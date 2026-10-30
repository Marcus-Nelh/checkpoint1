"""Loading helpers for CelebA-Spoof.

Expected layout (see README):

    CelebA_Spoof/
      Data/{train,test}/<subject_id>/{live,spoof}/<img>.jpg and <img>_BB.txt
      metas/intra_test/{train,test}_label.json

Each label is a list of 44 ints: [0:40] CelebA attributes, [40] spoof type,
[41] illumination, [42] environment, [43] live (0) / spoof (1).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_ROOT = Path("data/CelebA_Spoof")
SEED = 228
SPLITS = ("train", "test")
BBOX_REF = 224  # _BB.txt coordinates are relative to a 224x224 resize

SPOOF_TYPES = {0: "Live", 1: "Photo", 2: "Poster", 3: "A4", 4: "Face Mask", 5: "Upper Body Mask",
               6: "Region Mask", 7: "PC", 8: "Pad", 9: "Phone", 10: "3D Mask"}
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


def label_json(root, split):
    return Path(root) / "metas" / "intra_test" / f"{split}_label.json"


def check_data_root(root=None):
    """Return the dataset root, or exit with a message if it is missing."""
    root = Path(root or os.environ.get("CELEBA_SPOOF_ROOT") or DEFAULT_ROOT)
    required = [root / "Data", label_json(root, "train"), label_json(root, "test")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        sys.exit(f"dataset not found (missing: {', '.join(missing)}); see README for download steps")
    return root


def load_labels(root, split):
    """One row per image: path, split, subject_id, attr_* columns and the four anti-spoof labels."""
    with open(label_json(root, split)) as file:
        raw = json.load(file)
    paths = [path.replace("\\", "/") for path in raw]
    values = np.array(list(raw.values()), dtype=np.int16)
    labels = pd.DataFrame(values[:, :40], columns=[f"attr_{name}" for name in CELEBA_ATTRS])
    labels.insert(0, "path", paths)
    labels.insert(1, "split", split)
    labels.insert(2, "subject_id", [path.split("/")[-3] for path in paths])
    labels["spoof_type"] = values[:, 40]
    labels["illumination"] = values[:, 41]
    labels["environment"] = values[:, 42]
    labels["live_spoof"] = values[:, 43]
    labels["folder_label"] = [1 if "/spoof/" in path else 0 for path in paths]
    return labels


def load_all_labels(root):
    return pd.concat([load_labels(root, split) for split in SPLITS], ignore_index=True)


def read_bbox(image_path, image_size):
    """Face box (x, y, w, h, score) in pixels of the real image, or None if the _BB.txt is missing or bad."""
    image_path = Path(image_path)
    bbox_path = image_path.with_name(image_path.stem + "_BB.txt")
    try:
        box_x, box_y, box_w, box_h, score = [float(value) for value in bbox_path.read_text().split()[:5]]
    except (OSError, ValueError):
        return None
    scale_x = image_size[0] / BBOX_REF
    scale_y = image_size[1] / BBOX_REF
    return box_x * scale_x, box_y * scale_y, box_w * scale_x, box_h * scale_y, score


def load_crop(root, row, kind="face", side=224):
    """Square face or context crop from a manifest row, resized with bicubic interpolation.
    Parts of the box outside the image are filled with black."""
    prefix = "face" if kind == "face" else "ctx"
    box = (int(row[f"{prefix}_l"]), int(row[f"{prefix}_t"]), int(row[f"{prefix}_r"]), int(row[f"{prefix}_b"]))
    with Image.open(Path(root) / row["path"]) as image:
        return image.convert("RGB").crop(box).resize((side, side), Image.BICUBIC)


def sample_per_type(labels, n_images, seed=SEED):
    """About n_images rows with the same number from each spoof type (all rows of a type if it is smaller)."""
    per_type = n_images // labels["spoof_type"].nunique()
    parts = []
    for _, group in labels.groupby("spoof_type"):
        parts.append(group.sample(n=min(per_type, len(group)), random_state=seed))
    return pd.concat(parts).sort_index()
