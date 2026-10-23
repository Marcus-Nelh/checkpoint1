"""Generate a tiny synthetic CelebA-Spoof tree for running the pipeline before the real data exists.

    uv run python -m checkpoint1.fixture --out data/fixture/CelebA_Spoof --n-subjects 8 --seed 228

The tree has the same layout and 44-int label schema as the real dataset. Spoof images carry
a moire overlay, blur or glare so live/spoof plots show a visible difference. Three anomalies
are planted in the train split for the sanity checks: an image with no _BB.txt, a truncated
JPEG, and an exact duplicate.
"""

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from checkpoint1.data import BBOX_REF, CELEBA_ATTRS, SEED

PNG_SPOOFS = {2, 8, 10}  # mix of image formats, as in the real data


def _render(rng, spoof_type: int):
    """Random-size noise image with a bright 'face' ellipse; returns (PIL image, 224-relative bbox)."""
    w, h = (int(v) for v in rng.integers(160, 641, size=2))
    yy, xx = np.mgrid[0:h, 0:w]
    base = rng.uniform(60, 170)
    img = np.full((h, w, 3), base) + rng.normal(0, 18, size=(h, w, 3))
    cx, cy = w * rng.uniform(0.45, 0.55), h * rng.uniform(0.45, 0.55)
    rx, ry = w * rng.uniform(0.17, 0.23), h * rng.uniform(0.22, 0.28)
    face = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    img[face] = np.array([200, 160, 140]) + rng.normal(0, 14, size=(int(face.sum()), 3))
    if spoof_type in (7, 8, 9):  # screen replay: moire stripes + glare
        img += 14 * np.sin(2 * np.pi * (xx * 0.35 + yy * 0.2))[..., None]
        img += 60 * np.exp(-(((xx - w * 0.8) / (w * 0.15)) ** 2 + ((yy - h * 0.2) / (h * 0.15)) ** 2))[..., None]
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if spoof_type in (1, 2, 3):  # print: softer, flatter
        pil = pil.filter(ImageFilter.GaussianBlur(1.5))
    elif spoof_type >= 4 and spoof_type != 7 and spoof_type not in (8, 9):  # masks: darker
        pil = Image.eval(pil, lambda v: int(v * 0.8))
    x, y, bw, bh = cx - rx, cy - ry, 2 * rx, 2 * ry
    s = BBOX_REF
    bbox = (x / w * s, y / h * s, bw / w * s, bh / h * s)
    return pil, bbox


def _save(pil, path: Path, bbox, rng, write_bbox=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, quality=92) if path.suffix == ".jpg" else pil.save(path)
    if write_bbox:
        x, y, w, h = bbox
        path.with_name(path.stem + "_BB.txt").write_text(
            f"{x:.0f} {y:.0f} {w:.0f} {h:.0f} {rng.uniform(0.85, 1.0):.4f}\n"
        )


def make_fixture(out, n_subjects: int = 8, per_subject: int = 8, seed: int = SEED) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(1000, 10000), size=n_subjects, replace=False)
    n_train = max(1, round(0.75 * n_subjects))
    split_of = {int(s): ("train" if i < n_train else "test") for i, s in enumerate(ids)}
    labels = {"train": {}, "test": {}}
    counters = {"spoof": 0, "img": 100000}
    first_train = None

    for sid, split in split_of.items():
        n_live = max(1, per_subject * 3 // 8)
        for k in range(per_subject):
            live = k < n_live
            if live:
                stype, illum, env = 0, 0, 0
            else:
                c = counters["spoof"]
                counters["spoof"] += 1
                stype, illum, env = 1 + c % 10, 1 + c % 4, 1 + c % 2
            counters["img"] += int(rng.integers(1, 50))
            ext = "png" if stype in PNG_SPOOFS else "jpg"
            rel = f"Data/{split}/{sid}/{'live' if live else 'spoof'}/{counters['img']}.{ext}"
            pil, bbox = _render(rng, stype)
            _save(pil, out / rel, bbox, rng)
            attrs = rng.integers(0, 2, size=len(CELEBA_ATTRS)).tolist()
            labels[split][rel] = attrs + [stype, illum, env, 0 if live else 1]
            if split == "train" and first_train is None and not live:
                first_train = rel

    # planted anomalies (train split)
    train = labels["train"]
    spoofs = [r for r in train if "/spoof/" in r and r != first_train and r.endswith(".jpg")]
    lives = [r for r in train if "/live/" in r and r.endswith(".jpg")]
    # 1. missing bbox
    (out / spoofs[0]).with_name(Path(spoofs[0]).stem + "_BB.txt").unlink()
    # 2. truncated JPEG
    buf = io.BytesIO()
    Image.open(out / lives[0]).save(buf, format="JPEG")
    (out / lives[0]).write_bytes(buf.getvalue()[: len(buf.getvalue()) * 2 // 5])
    # 3. exact duplicate of another live image (same subject folder), new name + label row
    src = lives[1]
    dup = str(Path(src).with_name(f"{counters['img'] + 7}.jpg"))
    shutil.copy(out / src, out / dup)
    shutil.copy((out / src).with_name(Path(src).stem + "_BB.txt"), (out / dup).with_name(Path(dup).stem + "_BB.txt"))
    train[dup] = list(train[src])

    meta = out / "metas" / "intra_test"
    meta.mkdir(parents=True, exist_ok=True)
    for split, lab in labels.items():
        (meta / f"{split}_label.json").write_text(json.dumps(lab))
    n = sum(len(v) for v in labels.values())
    print(f"fixture: {n} images, {n_subjects} subjects -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/fixture/CelebA_Spoof")
    ap.add_argument("--n-subjects", type=int, default=8)
    ap.add_argument("--per-subject", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    make_fixture(a.out, a.n_subjects, a.per_subject, a.seed)


if __name__ == "__main__":
    main()
