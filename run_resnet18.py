"""
run_resnet18.py — Train & evaluate ResNet18TokenModel on melting curve data.

Pipeline
--------
  CSV  →  plot (temperature vs melting curve)  →  224×224 image
       →  ResNet18 layer4 tokens (49 × 512)
       →  [CLS] + 1-layer MH self-attention
       →  classification head

Checkpoints: saved per seed to <output_dir>/resnet18_attn_seed{i}.pt
Results    : <output_dir>/results_resnet18.csv  +  summary_resnet18.csv

Usage
-----
  python run_resnet18.py                         # all seeds (0-9)
  python run_resnet18.py --seeds 0 1 2           # specific seeds
  python run_resnet18.py --device cuda:0
  python run_resnet18.py --unfreeze_backbone     # fine-tune backbone too
"""

import argparse
import ast
import csv
import io
import json
import math
import os
import sys
import time
import traceback
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive, safe for multi-processing
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model_resnet18 import ResNet18TokenModel


# ---------------------------------------------------------------------------
# ImageNet transform (matches plot_melt_data  +  standard ResNet preprocessing)
# ---------------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MeltingCurveImageDataset(Dataset):
    """
    Reads a melting-curve CSV, renders each sample as a 224×224 PNG image
    (exactly matching plot_melt_data style), and returns (image_tensor, label).

    Images are rendered once at init time and cached as PIL Images.
    """

    def __init__(
        self,
        csv_path:      str,
        label_encoder: dict,
        transform=None,
        augment:       bool = False,
    ):
        self.df        = pd.read_csv(csv_path).reset_index(drop=True)
        self.le        = label_encoder
        self.transform = transform or TRANSFORM
        self.augment   = augment

        self.labels = [int(self.le[name]) for name in self.df["Species"].values]

        # Pre-parse temperature & melting curve (list literal strings)
        self.temperatures   = []
        self.melting_curves = []
        for _, row in self.df.iterrows():
            mc = np.array(ast.literal_eval(row["Melting Curve Data"]), dtype=np.float32)
            self.melting_curves.append(mc)
            try:
                temp = np.array(ast.literal_eval(str(row["Temperature (°C)"])), dtype=np.float32)
            except Exception:
                temp = np.linspace(60.0, 95.0, len(mc), dtype=np.float32)
            self.temperatures.append(temp)

        # Pre-render all images (done once; ~0.1 s per sample)
        n = len(self.df)
        print(f"    Rendering {n} images...", end=" ", flush=True)
        self._images = [self._render(i) for i in range(n)]
        print("done")

    def _render(self, idx: int) -> Image.Image:
        """Render melting curve as PIL Image (matching plot_melt_data style)."""
        temperature   = self.temperatures[idx]
        melting_curve = self.melting_curves[idx]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(temperature, melting_curve)
        ax.tick_params(labelsize=20)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=50, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).convert("RGB").copy()   # .copy() releases the buffer ref
        plt.close(fig)
        buf.close()
        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_tensor = self.transform(self._images[idx])

        # Light augmentation: small Gaussian noise on pixel values
        if self.augment:
            img_tensor = img_tensor + torch.randn_like(img_tensor) * 0.02
            img_tensor = img_tensor.clamp(-3.0, 3.0)   # stay within normalized range

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss       = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += len(y)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return {
        "loss": total_loss / total,
        "acc":  correct / total,
        "f1":   f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Training loop (single seed)
# ---------------------------------------------------------------------------

def train_one_seed(seed: int, args, device: torch.device) -> dict:
    data_dir = Path(args.data_root) / f"prompt3_splits_prompts_seed{seed}"
    with open(args.label_encoder) as f:
        label_encoder = json.load(f)
    num_classes = len(label_encoder)

    print(f"\n[seed={seed}] Loading datasets from {data_dir}")
    train_ds = MeltingCurveImageDataset(str(data_dir / "train.csv"), label_encoder, augment=True)
    val_ds   = MeltingCurveImageDataset(str(data_dir / "val.csv"),   label_encoder, augment=False)
    test_ds  = MeltingCurveImageDataset(str(data_dir / "test.csv"),  label_encoder, augment=False)

    # num_workers=0: matplotlib is not safe inside DataLoader worker processes
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = ResNet18TokenModel(
        num_classes     = num_classes,
        d_model         = args.d_model,
        n_heads         = args.n_heads,
        d_ff            = args.d_ff,
        dropout         = args.dropout,
        freeze_backbone = not args.unfreeze_backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seed={seed}] Trainable params: {n_params:,}")

    # Optimizer: only trainable params (backbone is frozen by default)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return max(epoch / args.warmup_epochs, 1e-6)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    best_epoch     = 0
    t0             = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_m = evaluate(model, val_loader, device)

        if val_m["loss"] < best_val_loss:
            best_val_loss  = val_m["loss"]
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            best_epoch     = epoch
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:4d}/{args.epochs}  "
                f"val_loss={val_m['loss']:.4f}  val_acc={val_m['acc']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if patience_count >= args.patience:
            print(f"  Early stop at epoch {epoch} (patience={args.patience})")
            break

    train_time = time.time() - t0

    # Load best checkpoint → evaluate val + test
    model.load_state_dict(best_state)
    best_val_m  = evaluate(model, val_loader,  device)
    test_m      = evaluate(model, test_loader, device)

    print(
        f"[seed={seed}] best_epoch={best_epoch}  "
        f"val_loss={best_val_m['loss']:.4f}  val_acc={best_val_m['acc']:.4f}  "
        f"test_acc={test_m['acc']:.4f}  test_f1={test_m['f1']:.4f}"
    )

    # Save checkpoint
    ckpt_dir  = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"resnet18_attn_seed{seed}.pt"
    torch.save(best_state, ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    return {
        "seed":          seed,
        "best_val_loss": best_val_m["loss"],
        "best_val_acc":  best_val_m["acc"],
        "best_val_f1":   best_val_m["f1"],
        "test_acc":      test_m["acc"],
        "test_f1":       test_m["f1"],
        "best_epoch":    best_epoch,
        "train_time_s":  round(train_time, 1),
        "n_params":      n_params,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ROOT = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="ResNet18 token model for melting curve classification"
    )
    p.add_argument("--seeds",      nargs="+", type=int, default=list(range(10)))
    p.add_argument("--data_root",  type=str,  default=str(ROOT))
    p.add_argument("--label_encoder", type=str, default=str(ROOT / "label_encoder.json"))
    p.add_argument("--output_dir", type=str,  default=str(ROOT / "results_resnet18"))
    p.add_argument("--device",     type=str,  default="auto")
    # Model
    p.add_argument("--d_model",    type=int,  default=128)
    p.add_argument("--n_heads",    type=int,  default=8)
    p.add_argument("--d_ff",       type=int,  default=256)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--unfreeze_backbone", action="store_true",
                   help="Fine-tune ResNet18 backbone (default: frozen)")
    # Training
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--patience",       type=int,   default=40)
    p.add_argument("--warmup_epochs",  type=int,   default=10)
    return p.parse_args()


def main():
    args   = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device : {device}")
    print(f"Seeds  : {args.seeds}")
    print(f"Epochs : {args.epochs}  Patience : {args.patience}")
    print(f"Backbone frozen : {not args.unfreeze_backbone}")

    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_resnet18.csv"
    summary_path = out_dir / "summary_resnet18.csv"

    fieldnames = [
        "seed",
        "best_val_loss", "best_val_acc", "best_val_f1",
        "test_acc", "test_f1",
        "best_epoch", "train_time_s", "n_params",
    ]

    csv_file = open(results_path, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if results_path.stat().st_size == 0:
        writer.writeheader()

    all_results = []

    for seed in args.seeds:
        try:
            row = train_one_seed(seed, args, device)
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_file.flush()
            all_results.append(row)
        except Exception as e:
            print(f"ERROR [seed={seed}]: {e}")
            traceback.print_exc()

    csv_file.close()

    if not all_results:
        print("No results.")
        return

    # Summary
    test_accs = [r["test_acc"] for r in all_results]
    test_f1s  = [r["test_f1"]  for r in all_results]

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "n_seeds",
            "test_acc_mean", "test_acc_std",
            "test_f1_mean",  "test_f1_std",
        ])
        w.writeheader()
        w.writerow({
            "model":         "resnet18_attn",
            "n_seeds":       len(all_results),
            "test_acc_mean": f"{np.mean(test_accs):.4f}",
            "test_acc_std":  f"{np.std(test_accs):.4f}",
            "test_f1_mean":  f"{np.mean(test_f1s):.4f}",
            "test_f1_std":   f"{np.std(test_f1s):.4f}",
        })

    print(f"\nResults → {results_path}")
    print(f"Summary → {summary_path}")
    print("\n" + "=" * 60)
    print(f"ResNet18-Attn   "
          f"test_acc = {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}   "
          f"test_f1 = {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
