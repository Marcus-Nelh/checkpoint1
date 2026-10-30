"""Clean CelebA-Spoof and write subject-disjoint train/val/test manifests.

    uv run python -m checkpoint1.preprocess --root $CELEBA_SPOOF_ROOT --out data/processed --seed 228

Writes manifest_{train,val,test}.csv (clean images with face and context crop boxes)
and dropped_samples.csv (every dropped image and the reason).
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from checkpoint1 import data

FACE_MARGIN = 0.15  # face crop = box side * (1 + 2 * 0.15)
CONTEXT_SCALE = 2.5  # context crop = box side * 2.5
MIN_FACE = 40  # faces smaller than this (px) are flagged


def square_box(center_x, center_y, side):
    half = side / 2
    return (round(center_x - half), round(center_y - half), round(center_x + half), round(center_y + half))


def check_image(task):
    """Check one image and its box file. Returns the crop boxes, or the reason to drop it."""
    root, path = task
    image_path = os.path.join(root, path)
    if not os.path.isfile(image_path):
        return {"path": path, "drop": "missing_image"}
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
    except Exception:
        return {"path": path, "drop": "corrupt_image"}

    bbox = data.read_bbox(image_path, (width, height))
    if bbox is None:
        bbox_path = os.path.splitext(image_path)[0] + "_BB.txt"
        reason = "bad_bbox" if os.path.isfile(bbox_path) else "missing_bbox"
        return {"path": path, "drop": reason}
    box_x, box_y, box_w, box_h, _ = bbox
    if box_w <= 0 or box_h <= 0:
        return {"path": path, "drop": "bad_bbox"}
    overlap_w = max(0, min(box_x + box_w, width) - max(box_x, 0))
    overlap_h = max(0, min(box_y + box_h, height) - max(box_y, 0))
    inside_fraction = overlap_w * overlap_h / (box_w * box_h)
    if inside_fraction == 0:
        return {"path": path, "drop": "bbox_outside_image"}

    center_x = box_x + box_w / 2
    center_y = box_y + box_h / 2
    side = max(box_w, box_h)
    face = square_box(center_x, center_y, side * (1 + 2 * FACE_MARGIN))
    context = square_box(center_x, center_y, side * CONTEXT_SCALE)
    return {
        "path": path, "drop": "", "bbox_w": round(box_w, 1), "bbox_h": round(box_h, 1),
        "face_l": face[0], "face_t": face[1], "face_r": face[2], "face_b": face[3],
        "ctx_l": context[0], "ctx_t": context[1], "ctx_r": context[2], "ctx_b": context[3],
        "flag_partial_bbox": inside_fraction < 1, "flag_small_face": side < MIN_FACE,
    }


def add_val_split(samples, val_fraction, seed):
    """Move a random val_fraction of the training subjects into a 'val' split."""
    rng = np.random.default_rng(seed)
    train_subjects = sorted(samples.loc[samples["split"] == "train", "subject"].unique())
    rng.shuffle(train_subjects)
    n_val = round(len(train_subjects) * val_fraction)
    val_subjects = set(train_subjects[:n_val])
    is_val = (samples["split"] == "train") & samples["subject"].isin(val_subjects)
    samples.loc[is_val, "split"] = "val"
    return samples


def parse_n_sample(text):
    return None if text == "all" else int(text)


def main():
    parser = argparse.ArgumentParser(description="Clean CelebA-Spoof and write train/val/test manifests.")
    parser.add_argument("--root", default=None, help="CelebA_Spoof folder (default: $CELEBA_SPOOF_ROOT)")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=data.SEED)
    parser.add_argument("--n-sample", type=parse_n_sample, default=None,
                        help="number of images to process (seeded random sample), or 'all' (default)")
    parser.add_argument("--val-fraction", type=float, default=0.10, help="fraction of train subjects used for val")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    root = data.check_data_root(args.root)
    os.makedirs(args.out, exist_ok=True)

    labels = data.load_all_labels(root)
    if args.n_sample is not None and args.n_sample < len(labels):
        labels = labels.sample(n=args.n_sample, random_state=args.seed)
    labels = labels.sort_values("path").reset_index(drop=True)
    print(f"{len(labels):,} labelled images")

    samples = pd.DataFrame({
        "path": labels["path"], "split": labels["split"], "subject": labels["subject_id"],
        "is_spoof": labels["live_spoof"], "spoof_type": labels["spoof_type"],
        "illumination": labels["illumination"], "environment": labels["environment"],
    })
    for name in data.CELEBA_ATTRS:
        samples[name] = labels[f"attr_{name}"]

    # the folder name, the live/spoof label and the spoof type must agree
    conflict = (labels["folder_label"] != labels["live_spoof"]) | ((labels["spoof_type"] != 0) != (labels["live_spoof"] == 1))
    dropped = [samples.loc[conflict, ["path", "split"]].assign(reason="label_conflict")]
    samples = samples[~conflict]

    tasks = [(str(root), path) for path in samples["path"]]
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(check_image, tasks, chunksize=256), total=len(tasks), desc="check images"))
    checks = pd.DataFrame(results)
    samples = samples.merge(checks, on="path")
    bad = samples["drop"] != ""
    dropped.append(samples.loc[bad, ["path", "split", "drop"]].rename(columns={"drop": "reason"}))
    samples = samples[~bad].drop(columns="drop")
    for column in ["face_l", "face_t", "face_r", "face_b", "ctx_l", "ctx_t", "ctx_r", "ctx_b"]:
        samples[column] = samples[column].astype(int)
    samples["flag_partial_bbox"] = samples["flag_partial_bbox"].astype(bool)
    samples["flag_small_face"] = samples["flag_small_face"].astype(bool)

    dropped = pd.concat(dropped).sort_values(["reason", "path"])
    dropped.to_csv(os.path.join(args.out, "dropped_samples.csv"), index=False)
    print(f"dropped {len(dropped):,} images:")
    print(dropped["reason"].value_counts().to_string())

    samples = add_val_split(samples, args.val_fraction, args.seed)
    for split in ("train", "val", "test"):
        manifest = samples[samples["split"] == split]
        manifest.to_csv(os.path.join(args.out, f"manifest_{split}.csv"), index=False)
        print(f"{split}: {len(manifest):,} images, {manifest['subject'].nunique():,} subjects")


if __name__ == "__main__":
    main()
