"""Baseline for CelebA-Spoof: an ImageNet ResNet-18 fine-tuned on the face crop.

    uv run python -m checkpoint1.baseline [--manifest-dir data/processed] [--n-train N|all]
        [--n-val N|all] [--n-test N|all] [--epochs 3] [--seed 228] [--recompute]

Needs the manifests from checkpoint1.preprocess. By default each split is a small seeded subset.
A face is classified as spoof when P(spoof) >= 0.5. The trained model and scores are saved in
--out-dir and reused while the settings in config.json do not change.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from checkpoint1 import data
from checkpoint1.eda import LIVE_COLOR, SPOOF_COLOR

SPLITS = ("train", "val", "test")
DEFAULT_SIZES = {"train": 5_000, "val": 2_000, "test": 5_000}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
THRESHOLD = 0.5


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_workers():
    """DataLoader workers: none when /dev/shm is too small to share batches (e.g. Docker's 64 MB)."""
    if Path("/dev/shm").exists() and shutil.disk_usage("/dev/shm").total < 2**30:
        return 0
    return min(8, (os.cpu_count() or 2) - 1)


# ----------------------------------------------------------------------------- data

def subset(manifest, n_images, seed):
    """The whole split if n_images is None, else about n_images rows with the same spoof-type mix as the split."""
    if n_images is None or n_images >= len(manifest):
        return manifest.reset_index(drop=True)
    fraction = n_images / len(manifest)
    parts = []
    for _, group in manifest.groupby("spoof_type"):
        parts.append(group.sample(frac=fraction, random_state=seed))
    return pd.concat(parts).sort_values("path").reset_index(drop=True)


class FaceCrops(Dataset):
    """(normalised 224x224 face crop, is_spoof). Training uses a horizontal flip and a random crop."""

    def __init__(self, manifest, root, train):
        self.root = root
        self.rows = manifest.to_dict("records")
        if train:
            augment = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(224)]
        else:
            augment = [transforms.CenterCrop(224)]
        self.transform = transforms.Compose(augment + [transforms.ToTensor(),
                                                       transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = data.load_crop(self.root, row, "face", side=240)
        return self.transform(image), torch.tensor(float(row["is_spoof"]))


def make_loader(manifest, root, train, batch_size, workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(FaceCrops(manifest, root, train), batch_size=batch_size, shuffle=train,
                      num_workers=workers, generator=generator)


# ----------------------------------------------------------------------------- model

def build_model():
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 1)  # one logit: spoof vs live
    return model


@torch.no_grad()
def predict(model, manifest, root, device, batch_size, workers, seed):
    """P(spoof) for every row of the manifest, in order."""
    model.eval()
    scores = []
    for images, _ in tqdm(make_loader(manifest, root, False, batch_size, workers, seed), desc="predict", leave=False):
        logits = model(images.to(device)).squeeze(1)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def train(model, train_set, val_set, root, out_dir, config, device):
    """Fine-tune with class-weighted BCE. The epoch with the best validation AUC is saved to model.pt."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    n_spoof = train_set["is_spoof"].sum()
    n_live = len(train_set) - n_spoof
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_live / n_spoof, device=device))
    history = []
    best_auc = -1
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        loader = make_loader(train_set, root, True, config["batch_size"], config["workers"], config["seed"] + epoch)
        for images, targets in tqdm(loader, desc=f"epoch {epoch + 1}/{config['epochs']}"):
            images = images.to(device)
            targets = targets.to(device)
            loss = loss_fn(model(images).squeeze(1), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        val_scores = predict(model, val_set, root, device, config["batch_size"], config["workers"], config["seed"])
        val_auc = roc_auc_score(val_set["is_spoof"], val_scores)
        val_eer, _ = eer(val_set["is_spoof"], val_scores)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_set),
                        "val_auc": val_auc, "val_eer": val_eer})
        print(history[-1])
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), out_dir / "model.pt")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))


# ----------------------------------------------------------------------------- metrics

def eer(labels, scores):
    """(EER, threshold): the point where FAR and FRR are closest."""
    frr, tpr, thresholds = roc_curve(labels, scores)  # spoof is positive, so FPR = live rejected = FRR
    far = 1 - tpr
    i = np.argmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2, min(thresholds[i], 1.0)


def metrics(scores, threshold=THRESHOLD):
    """FAR, FRR, F1 (spoof positive) and accuracy at one threshold, each rate with its counts."""
    is_spoof = scores["is_spoof"] == 1
    predicted_spoof = scores["score"] >= threshold
    n_spoof = is_spoof.sum()
    n_live = (~is_spoof).sum()
    true_spoof = (is_spoof & predicted_spoof).sum()
    false_accepts = (is_spoof & ~predicted_spoof).sum()  # attacks accepted as live
    false_rejects = (~is_spoof & predicted_spoof).sum()  # live faces rejected
    correct = (is_spoof == predicted_spoof).sum()
    f1 = 2 * true_spoof / (2 * true_spoof + false_accepts + false_rejects)
    return {
        "threshold": round(threshold, 4),
        "FAR": f"{false_accepts / n_spoof:.2%} ({false_accepts:,}/{n_spoof:,})",
        "FRR": f"{false_rejects / n_live:.2%} ({false_rejects:,}/{n_live:,})",
        "F1": f"{f1:.2%}",
        "Accuracy": f"{correct / len(scores):.2%} ({correct:,}/{len(scores):,})",
    }


def per_group(scores, column, names, threshold=THRESHOLD):
    """Error rate inside each value of column: FAR for spoof images, FRR for live images."""
    rows = []
    for value, group in scores.groupby(column):
        is_spoof = group["is_spoof"] == 1
        predicted_spoof = group["score"] >= threshold
        n_spoof = is_spoof.sum()
        n_live = (~is_spoof).sum()
        false_accepts = (is_spoof & ~predicted_spoof).sum()
        false_rejects = (~is_spoof & predicted_spoof).sum()
        if n_spoof > 0:
            rows.append({column: names[value], "rate": "FAR", "errors": false_accepts, "total": n_spoof,
                         "error_rate": false_accepts / n_spoof})
        if n_live > 0:
            rows.append({column: names[value], "rate": "FRR", "errors": false_rejects, "total": n_live,
                         "error_rate": false_rejects / n_live})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def fig_far_frr(test_scores, fig_dir):
    frr, tpr, thresholds = roc_curve(test_scores["is_spoof"], test_scores["score"])
    thresholds = np.minimum(thresholds, 1.0)
    test_eer, eer_threshold = eer(test_scores["is_spoof"], test_scores["score"])
    fig, axis = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
    axis.plot(thresholds, 1 - tpr, color=SPOOF_COLOR, label="FAR (spoof accepted)")
    axis.plot(thresholds, frr, color=LIVE_COLOR, label="FRR (live rejected)")
    axis.plot([eer_threshold], [test_eer], "o", color="black", markersize=3,
              label=f"EER {test_eer:.2%} at {eer_threshold:.3f}")
    axis.set(xlabel="Threshold on P(spoof)", ylabel="Error rate", xlim=(0, 1), ylim=(0, 1))
    axis.legend(frameon=False, fontsize=5)
    fig.suptitle("FAR and FRR against threshold (test)")
    fig.savefig(fig_dir / "base_far_frr.png")
    plt.close(fig)


def fig_per_attack(test_scores, fig_dir):
    table = per_group(test_scores, "spoof_type", data.SPOOF_TYPES)
    colors = [SPOOF_COLOR if rate == "FAR" else LIVE_COLOR for rate in table["rate"]]
    fig, axis = plt.subplots(figsize=(3.5, 2.6), layout="constrained")
    bars = axis.barh(table["spoof_type"], table["error_rate"], color=colors)
    bar_labels = [f"{errors:,}/{total:,}" for errors, total in zip(table["errors"], table["total"])]
    axis.bar_label(bars, labels=bar_labels, padding=2, fontsize=5)
    axis.invert_yaxis()
    axis.margins(x=0.3)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set(xlabel="Error rate (FAR for attacks, FRR for live)", ylabel="Presentation type")
    fig.suptitle(f"Test error rate per presentation type (threshold {THRESHOLD})")
    fig.savefig(fig_dir / "base_per_attack.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- driver

def run_baseline(root=None, manifest_dir="data/processed", out_dir="outputs/baseline", sizes=None,
                 seed=data.SEED, epochs=3, batch_size=64, lr=1e-4, weight_decay=1e-4, workers=None,
                 recompute=False):
    """Train (or load the cached) model and score val and test. Returns (history, scores)."""
    root = data.check_data_root(root)
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(seed)
    device = pick_device()
    if workers is None:
        workers = default_workers()

    sizes = sizes or {}
    splits = {}
    for split in SPLITS:
        manifest = pd.read_csv(Path(manifest_dir) / f"manifest_{split}.csv")
        splits[split] = subset(manifest, sizes.get(split, DEFAULT_SIZES[split]), seed)

    config = {"seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
              "workers": workers, "manifest_dir": str(manifest_dir),
              "rows": {split: len(splits[split]) for split in SPLITS}}
    print(f"device {device}; rows {config['rows']}")

    config_path = out_dir / "config.json"
    score_paths = {split: out_dir / f"scores_{split}.csv" for split in ("val", "test")}
    cached = config_path.exists() and json.loads(config_path.read_text()) == config
    if cached and not recompute and all(path.exists() for path in score_paths.values()):
        print(f"reusing cached model and scores in {out_dir}")
        scores = {split: pd.read_csv(path) for split, path in score_paths.items()}
    else:
        config_path.write_text(json.dumps(config, indent=2))
        model = build_model().to(device)
        train(model, splits["train"], splits["val"], root, out_dir, config, device)
        scores = {}
        for split, path in score_paths.items():
            keep = ["path", "is_spoof", "spoof_type", "illumination", "environment"]
            scores[split] = splits[split][keep].copy()
            scores[split]["score"] = predict(model, splits[split], root, device, batch_size, workers, seed)
            scores[split].to_csv(path, index=False)

    history = pd.read_csv(out_dir / "history.csv")
    fig_far_frr(scores["test"], fig_dir)
    fig_per_attack(scores["test"], fig_dir)
    return history, scores


def parse_size(text):
    if text == "all":
        return None
    return int(text)


def main():
    parser = argparse.ArgumentParser(description="ResNet-18 face-crop baseline for CelebA-Spoof.")
    parser.add_argument("--data-root", default=None, help="CelebA_Spoof folder (default: $CELEBA_SPOOF_ROOT)")
    parser.add_argument("--manifest-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs/baseline")
    for split in SPLITS:
        parser.add_argument(f"--n-{split}", type=parse_size, default=DEFAULT_SIZES[split],
                            help=f"{split} images: N or 'all' (default {DEFAULT_SIZES[split]:,})")
    parser.add_argument("--seed", type=int, default=data.SEED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: 0 if /dev/shm < 1 GB, else min(8, CPUs - 1))")
    parser.add_argument("--recompute", action="store_true", help="retrain even if the cached model matches")
    args = parser.parse_args()

    sizes = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    history, scores = run_baseline(args.data_root, args.manifest_dir, args.out_dir, sizes, args.seed, args.epochs,
                                   args.batch_size, args.lr, workers=args.workers, recompute=args.recompute)
    print(history.to_string(index=False))
    print(pd.DataFrame([metrics(scores["test"])]).to_string(index=False))


if __name__ == "__main__":
    main()
