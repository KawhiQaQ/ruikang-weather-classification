import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_DIR = ROOT / "datasets/competition/train"
LABELS = ["cloudy", "rainy", "snowy", "sunny"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--output-dir", default=str(ROOT / "splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def list_samples(train_dir):
    train_dir = Path(train_dir)
    classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if classes != LABELS:
        raise ValueError(f"Unexpected class order: {classes}")

    samples = []
    targets = []
    for class_idx, label in enumerate(classes):
        class_dir = train_dir / label
        paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        for path in paths:
            samples.append(path)
            targets.append(class_idx)
    return classes, samples, np.asarray(targets, dtype=np.int64)


def class_counts(targets, indices, labels):
    counts = np.bincount(targets[indices], minlength=len(labels))
    return {labels[i]: int(counts[i]) for i in range(len(labels))}


def save_split(args, labels, samples, targets):
    out = Path(args.output_dir)
    existing = [
        out / "train_idx.npy",
        out / "val_idx.npy",
        out / "local_test_idx.npy",
        out / "split_manifest.csv",
        out / "split_meta.json",
    ]
    if out.exists() and not args.force and any(path.exists() for path in existing):
        raise FileExistsError(
            f"{out} already contains split files. Use --force to overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    indices = np.arange(len(targets))
    test_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_ratio, random_state=args.seed)
    remain_idx, test_idx = next(test_splitter.split(indices, targets))

    remain_targets = targets[remain_idx]
    val_from_remain = args.val_ratio / (1.0 - args.test_ratio)
    val_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_from_remain, random_state=args.seed)
    train_pos, val_pos = next(
        val_splitter.split(remain_idx, remain_targets))

    train_idx = np.sort(remain_idx[train_pos])
    val_idx = np.sort(remain_idx[val_pos])
    test_idx = np.sort(test_idx)

    np.save(out / "train_idx.npy", train_idx)
    np.save(out / "val_idx.npy", val_idx)
    np.save(out / "local_test_idx.npy", test_idx)

    split_map = {}
    for split_name, split_indices in [
            ("train", train_idx),
            ("val", val_idx),
            ("local_test", test_idx)]:
        for idx in split_indices.tolist():
            split_map[idx] = split_name

    with open(out / "split_manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "split", "label", "path"])
        for idx, path in enumerate(samples):
            label = labels[int(targets[idx])]
            rel_path = path.relative_to(ROOT)
            writer.writerow([idx, split_map[idx], label, str(rel_path)])

    meta = {
        "seed": args.seed,
        "dataset_size": len(samples),
        "classes": labels,
        "ratios": {
            "train": len(train_idx) / len(samples),
            "val": len(val_idx) / len(samples),
            "local_test": len(test_idx) / len(samples),
        },
        "sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "local_test": len(test_idx),
        },
        "class_counts": {
            "train": class_counts(targets, train_idx, labels),
            "val": class_counts(targets, val_idx, labels),
            "local_test": class_counts(targets, test_idx, labels),
        },
        "files": {
            "train_idx": "train_idx.npy",
            "val_idx": "val_idx.npy",
            "local_test_idx": "local_test_idx.npy",
            "manifest": "split_manifest.csv",
        },
    }
    with open(out / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"saved: {out}")


def main():
    args = parse_args()
    labels, samples, targets = list_samples(args.train_dir)
    save_split(args, labels, samples, targets)


if __name__ == "__main__":
    main()
