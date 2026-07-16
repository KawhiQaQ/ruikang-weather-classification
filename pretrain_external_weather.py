import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:
    import timm
except ImportError:
    timm = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "datasets/external/weather_status_4class"
DEFAULT_SPLIT_FILE = ROOT / "splits/external_weather_4class_seed42_val010.npz"
LABELS = ["cloudy", "rainy", "snowy", "sunny"]
DEFAULT_BACKBONE = "efficientnet_b1"
BACKBONES = {
    "efficientnet_b1": {
        "factory": models.efficientnet_b1,
        "weights": models.EfficientNet_B1_Weights,
        "im_size": 240,
        "checkpoint": "effb1_external_weather_pretrain.pth",
        "batch_size": 64,
    },
    "efficientnet_b2": {
        "factory": models.efficientnet_b2,
        "weights": models.EfficientNet_B2_Weights,
        "im_size": 288,
        "checkpoint": "effb2_external_weather_pretrain.pth",
        "batch_size": 48,
    },
    "efficientnet_b3": {
        "factory": models.efficientnet_b3,
        "weights": models.EfficientNet_B3_Weights,
        "im_size": 300,
        "checkpoint": "effb3_external_weather_pretrain.pth",
        "batch_size": 32,
    },
    "resnet50": {
        "factory": models.resnet50,
        "weights": models.ResNet50_Weights,
        "im_size": 224,
        "checkpoint": "resnet50_external_weather_pretrain.pth",
        "batch_size": 64,
    },
    "swin_t": {
        "factory": models.swin_t,
        "weights": models.Swin_T_Weights,
        "im_size": 224,
        "checkpoint": "swin_t_external_weather_pretrain.pth",
        "batch_size": 24,
    },
    "deit_small_patch16_224": {
        "source": "timm",
        "model_name": "deit_small_patch16_224",
        "im_size": 224,
        "checkpoint": "deit_small_patch16_224_external_weather_pretrain.pth",
        "batch_size": 32,
    },
    "convnextv2_pico_fcmae": {
        "source": "timm",
        "model_name": "convnextv2_pico.fcmae_ft_in1k",
        "im_size": 224,
        "checkpoint": "convnextv2_pico_fcmae_external_weather_pretrain.pth",
        "batch_size": 64,
    },
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--output", default=None)
    parser.add_argument("--latest-output", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = BACKBONES[args.backbone]
    args.im_size = config["im_size"]
    if args.batch_size is None:
        args.batch_size = config["batch_size"]
    if args.output is None:
        args.output = str(ROOT / "results/pretrain" / config["checkpoint"])
    if args.latest_output is None:
        out = Path(args.output)
        args.latest_output = str(out.with_name(f"{out.stem}_latest{out.suffix}"))

    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
        args.num_workers = 0
        args.no_pretrained = True
        args.limit_per_class = 8
    return args


def progress_bar(iterable, enabled=True, **kwargs):
    if tqdm is None or not enabled:
        return iterable
    return tqdm(iterable, **kwargs)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(backbone, pretrained=True, num_classes=4):
    config = BACKBONES[backbone]
    if config.get("source", "torchvision") == "torchvision":
        weights = config["weights"].DEFAULT if pretrained else None
        model = config["factory"](weights=weights)
        if hasattr(model, "classifier"):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "head"):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model head for backbone: {backbone}")
    elif config["source"] == "timm":
        if timm is None:
            raise ImportError("timm is required for this backbone: pip install timm")
        model = timm.create_model(
            config["model_name"],
            pretrained=pretrained,
            pretrained_cfg_overlay=config.get("pretrained_cfg_overlay"),
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown backbone source: {config['source']}")
    return model


def build_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    for name, ema_value in ema_state.items():
        model_value = model_state[name].detach()
        if ema_value.dtype.is_floating_point:
            ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


def make_transforms(im_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            im_size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def make_or_load_split(split_file, targets, val_ratio, seed):
    split_file = Path(split_file)
    if split_file.exists():
        split = np.load(split_file)
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        return train_idx, val_idx

    indices = np.arange(len(targets))
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, targets))
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    split_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        split_file,
        train_idx=train_idx,
        val_idx=val_idx,
        val_ratio=val_ratio,
        seed=seed,
    )
    return train_idx, val_idx


def limit_indices_per_class(indices, targets, limit, seed):
    if limit is None:
        return indices
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(len(LABELS)):
        class_idx = np.asarray(indices)[targets[indices] == class_id]
        if len(class_idx) > limit:
            class_idx = rng.choice(class_idx, size=limit, replace=False)
        selected.extend(class_idx.tolist())
    return np.asarray(sorted(selected))


def build_loaders(args, device):
    train_tf, eval_tf = make_transforms(args.im_size)
    base_set = datasets.ImageFolder(args.data_dir)
    train_set = datasets.ImageFolder(args.data_dir, transform=train_tf)
    eval_set = datasets.ImageFolder(args.data_dir, transform=eval_tf)
    if base_set.classes != LABELS:
        raise ValueError(f"Unexpected class order: {base_set.classes}")

    targets = np.asarray(base_set.targets)
    train_idx, val_idx = make_or_load_split(
        args.split_file, targets, args.val_ratio, args.seed)
    train_idx = limit_indices_per_class(
        train_idx, targets, args.limit_per_class, args.seed)
    val_idx = limit_indices_per_class(
        val_idx, targets, args.limit_per_class, args.seed + 1)
    train_counts = np.bincount(targets[train_idx], minlength=len(LABELS))
    val_counts = np.bincount(targets[val_idx], minlength=len(LABELS))
    class_weights = train_counts.sum() / (
        len(LABELS) * np.maximum(train_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        Subset(train_set, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(eval_set, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print("class_to_idx:", base_set.class_to_idx)
    print("split sizes:", {"train": len(train_idx), "val": len(val_idx)})
    print("train class counts:",
          {LABELS[i]: int(train_counts[i]) for i in range(len(LABELS))})
    print("val class counts:",
          {LABELS[i]: int(val_counts[i]) for i in range(len(LABELS))})
    print("loss class weights:",
          {LABELS[i]: round(float(class_weights[i].cpu()), 4)
           for i in range(len(LABELS))})
    return train_loader, val_loader, class_weights


def evaluate(model, loader, criterion, device, name, progress=True):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true, y_pred = [], []
    iterator = progress_bar(loader, enabled=progress, desc=name, leave=False)
    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            if tqdm is not None and progress:
                iterator.set_postfix(
                    loss=f"{loss_sum / total:.4f}",
                    acc=f"{correct / total:.4f}",
                )

    loss = loss_sum / total
    acc = correct / total
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, target_names=LABELS, digits=4, zero_division=0)
    return loss, acc, f1, report


def save_checkpoint(path, model, args, epoch, metrics, model_type):
    torch.save({
        "model_state": model.state_dict(),
        "classes": LABELS,
        "im_size": args.im_size,
        "backbone": args.backbone,
        "model_name": BACKBONES[args.backbone].get("model_name", args.backbone),
        "model_type": model_type,
        "ema_decay": args.ema_decay,
        "epoch": epoch,
        "val_loss": metrics[0],
        "val_acc": metrics[1],
        "val_f1": metrics[2],
        "best_metric": "external_val_macro_f1",
        "pretrain_data": "kaggle_5class_weather_status_4class",
        "data_dir": str(Path(args.data_dir).resolve()),
        "split_file": str(Path(args.split_file).resolve()),
    }, path)


def print_config(args, device, output, latest_output):
    print("========== External Weather Pretrain Config ==========")
    print(f"backbone: {args.backbone}")
    print(f"input size: {args.im_size}x{args.im_size}")
    print(f"data dir: {args.data_dir}")
    print(f"split file: {args.split_file}")
    print(f"best checkpoint: {output}")
    print(f"latest checkpoint: {latest_output}")
    print(f"device: {device}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"label_smoothing: {args.label_smoothing}")
    print(f"ema_decay: {args.ema_decay}")
    print(f"pretrained: {not args.no_pretrained}")
    print("loss: CrossEntropyLoss(class weights + label_smoothing)")
    if args.ema_decay > 0:
        print("selection: best epoch by EMA external val macro F1")
    else:
        print("selection: best epoch by external val macro F1")
    print("======================================================")


def train():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    output = Path(args.output)
    latest_output = Path(args.latest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_output.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_weights = build_loaders(args, device)
    model = build_model(
        args.backbone,
        pretrained=not args.no_pretrained,
        num_classes=len(LABELS),
    ).to(device)
    ema_model = build_ema_model(model) if args.ema_decay > 0 else None
    print_config(args, device, output, latest_output)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))

    best_f1 = -1.0
    best_epoch = 0
    best_report = ""
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        iterator = progress_bar(
            train_loader,
            enabled=not args.no_progress,
            desc=f"train {epoch}/{args.epochs}",
            leave=False,
        )
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)

            total += x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            loss_sum += loss.item() * x.size(0)
            if tqdm is not None and not args.no_progress:
                iterator.set_postfix(
                    loss=f"{loss_sum / total:.4f}",
                    acc=f"{correct / total:.4f}",
                )
        scheduler.step()

        train_loss = loss_sum / total
        train_acc = correct / total
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            name=f"val {epoch}/{args.epochs}",
            progress=not args.no_progress,
        )
        val_loss, val_acc, val_f1, val_report = val_metrics
        selected_metrics = val_metrics
        selected_report = val_report
        selected_model = model
        selected_model_type = "raw"
        ema_line = ""
        if ema_model is not None:
            ema_metrics = evaluate(
                ema_model, val_loader, criterion, device,
                name=f"ema val {epoch}/{args.epochs}",
                progress=not args.no_progress,
            )
            ema_loss, ema_acc, ema_f1, ema_report = ema_metrics
            selected_metrics = ema_metrics
            selected_report = ema_report
            selected_model = ema_model
            selected_model_type = "ema"
            ema_line = (
                f" ema_val_loss={ema_loss:.4f} "
                f"ema_val_acc={ema_acc:.4f} ema_val_f1={ema_f1:.4f}"
            )

        print(f"Epoch {epoch}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_f1={val_f1:.4f}{ema_line}")
        print("External validation report:")
        print(selected_report)

        save_checkpoint(
            latest_output, selected_model, args, epoch,
            selected_metrics, selected_model_type)
        print(f"Saved latest checkpoint to {latest_output}")
        selected_f1 = selected_metrics[2]
        if selected_f1 > best_f1:
            best_f1 = selected_f1
            best_epoch = epoch
            best_report = selected_report
            save_checkpoint(
                output, selected_model, args, epoch,
                selected_metrics, selected_model_type)
            print(f"Saved best checkpoint to {output}")

    metric_name = "EMA_EXTERNAL_VAL_MACRO_F1" if args.ema_decay > 0 else "EXTERNAL_VAL_MACRO_F1"
    print("best external val_f1:", round(best_f1, 6))
    print(f"BEST_{metric_name}={best_f1:.6f} at epoch {best_epoch}/{args.epochs}")
    print(best_report)


if __name__ == "__main__":
    train()
