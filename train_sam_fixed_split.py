import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from train_fixed_split import (
    BACKBONES,
    LABELS,
    ROOT,
    build_ema_model,
    build_loaders,
    build_model,
    evaluate,
    get_device,
    load_checkpoint,
    load_init_checkpoint,
    progress_bar,
    save_checkpoint,
    seed_everything,
    tqdm,
    update_ema_model,
)


DEFAULT_STUDENT_INIT = (
    ROOT
    / "results/pretrain/"
    / "effb1_external_weather_pretrain_e16.pth"
)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        raise NotImplementedError("SAM requires first_step and second_step.")

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                norms.append(grad.norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


def disable_running_stats(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0


def enable_running_stats(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
            del module.backup_momentum


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b1", choices=sorted(BACKBONES))
    parser.add_argument("--train-dir", default=str(ROOT / "datasets/competition/train"))
    parser.add_argument("--splits-dir", default=str(ROOT / "splits"))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_STUDENT_INIT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--latest-output", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--run-name", default="sam_b1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--rrc-scale-min", type=float, default=0.85)
    parser.add_argument("--random-grayscale-p", type=float, default=0.0)
    parser.add_argument("--color-jitter-scale", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--sam-adaptive", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = BACKBONES[args.backbone]
    args.im_size = config["im_size"]
    if args.batch_size is None:
        args.batch_size = config["batch_size"]
    if args.output is None:
        suffix = "asam" if args.sam_adaptive else "sam"
        args.output = str(ROOT / "results" / suffix / f"{args.run_name}.pth")
    if args.latest_output is None:
        out = Path(args.output)
        args.latest_output = str(out.with_name(f"{out.stem}_latest{out.suffix}"))
    if args.result_json is None:
        out = Path(args.output)
        args.result_json = str(out.with_suffix(".json"))

    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
        args.num_workers = 0
    return args


def print_config(args, device):
    print("========== SAM Fine-tune Config ==========")
    print(f"backbone: {args.backbone}")
    print(f"input size: {args.im_size}x{args.im_size}")
    print(f"init checkpoint: {args.init_checkpoint}")
    print(f"best checkpoint: {args.output}")
    print(f"latest checkpoint: {args.latest_output}")
    print(f"device: {device}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"label_smoothing: {args.label_smoothing}")
    print(f"ema_decay: {args.ema_decay}")
    print(f"sam_rho: {args.sam_rho}")
    print(f"sam_adaptive: {args.sam_adaptive}")
    print("loss: CrossEntropyLoss(class weights + label_smoothing)")
    print("selection: best epoch by EMA val macro F1")
    print("final test: local_test split only after training")
    print("==========================================")


def train():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    output = Path(args.output)
    latest_output = Path(args.latest_output)
    result_json = Path(args.result_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_output.parent.mkdir(parents=True, exist_ok=True)
    result_json.parent.mkdir(parents=True, exist_ok=True)

    loader_pack = build_loaders(args, device)
    train_loader, val_loader, test_loader, class_weights = loader_pack[:4]
    model = build_model(args.backbone, num_classes=len(LABELS), pretrained=False).to(device)
    load_init_checkpoint(args.init_checkpoint, model, device)
    ema_model = build_ema_model(model) if args.ema_decay > 0 else None
    print_config(args, device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = SAM(
        model.parameters(),
        optim.AdamW,
        rho=args.sam_rho,
        adaptive=args.sam_adaptive,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer, T_max=max(args.epochs, 1))

    best_f1 = -1.0
    best_epoch = 0
    best_report = ""
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        iterator = progress_bar(
            train_loader,
            enabled=not args.no_progress,
            desc=f"sam train {epoch}/{args.epochs}",
            leave=False,
        )
        for x, y in iterator:
            x, y = x.to(device), y.to(device)

            enable_running_stats(model)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            criterion(model(x), y).backward()
            optimizer.second_step(zero_grad=True)
            enable_running_stats(model)

            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)

            bs = x.size(0)
            total += bs
            correct += (out.argmax(dim=1) == y).sum().item()
            loss_sum += loss.item() * bs
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
            progress=not args.no_progress)
        val_loss, val_acc, val_f1, val_report = val_metrics
        selected_metrics = val_metrics
        selected_report = val_report
        selected_model = model
        selected_model_type = "raw_sam"
        ema_line = ""
        if ema_model is not None:
            ema_metrics = evaluate(
                ema_model, val_loader, criterion, device,
                name=f"ema val {epoch}/{args.epochs}",
                progress=not args.no_progress)
            ema_loss, ema_acc, ema_f1, ema_report = ema_metrics
            selected_metrics = ema_metrics
            selected_report = ema_report
            selected_model = ema_model
            selected_model_type = "ema_sam"
            ema_line = (
                f" ema_val_loss={ema_loss:.4f} "
                f"ema_val_acc={ema_acc:.4f} ema_val_f1={ema_f1:.4f}"
            )

        print(f"Epoch {epoch}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_f1={val_f1:.4f}{ema_line}")
        print("Validation report:")
        print(selected_report)

        save_checkpoint(
            latest_output, selected_model, args, epoch,
            selected_metrics, selected_model_type)
        selected_f1 = selected_metrics[2]
        if selected_f1 > best_f1:
            best_f1 = selected_f1
            best_epoch = epoch
            best_report = selected_report
            best_metrics = selected_metrics
            save_checkpoint(
                output, selected_model, args, epoch,
                selected_metrics, selected_model_type)
            print(f"Saved best checkpoint to {output}")
        print(f"Saved latest checkpoint to {latest_output}")

    print("best sam val_f1:", round(best_f1, 6))
    print(f"BEST_SAM_EMA_VAL_MACRO_F1={best_f1:.6f} at epoch {best_epoch}/{args.epochs}")
    print(best_report)

    print("Loading best checkpoint for local_test evaluation...")
    load_checkpoint(output, model, device)
    test_loss, test_acc, test_f1, test_report = evaluate(
        model, test_loader, criterion, device,
        name="local_test",
        progress=not args.no_progress)
    print("========== SAM Local Test Result ==========")
    print(f"LOCAL_TEST_LOSS={test_loss:.6f}")
    print(f"LOCAL_TEST_ACC={test_acc:.6f}")
    print(f"LOCAL_TEST_MACRO_F1={test_f1:.6f}")
    print(test_report)

    result = {
        "run_name": args.run_name,
        "seed": args.seed,
        "backbone": args.backbone,
        "init_checkpoint": args.init_checkpoint,
        "ema_decay": args.ema_decay,
        "sam_rho": args.sam_rho,
        "sam_adaptive": args.sam_adaptive,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_metrics[0] if best_metrics else None,
        "best_val_acc": best_metrics[1] if best_metrics else None,
        "best_val_f1": best_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "output": str(output),
    }
    result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT_JSON:", result_json)
    print(
        f"SAM_RESULT run={args.run_name} seed={args.seed} "
        f"best_val_f1={best_f1:.6f} best_epoch={best_epoch} "
        f"test_f1={test_f1:.6f}")


if __name__ == "__main__":
    train()
