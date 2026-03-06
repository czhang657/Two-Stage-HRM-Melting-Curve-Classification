"""
run_vit.py — Train & evaluate ViTClassifier on melting curve images.

Pipeline
--------
  CSV  →  plot (temperature vs melting curve)  →  224×224 image
       →  patch embed (16×16 patches → 196 tokens)
       →  [CLS] + n-layer MH self-attention
       →  classification head

Variants
--------
  small : patch_size=16, d_model=128, n_heads=8, n_layers=1  (~260 K params)
  large : patch_size=16, d_model=128, n_heads=8, n_layers=3  (~510 K params)

Checkpoints : <output_dir>/vit_{variant}_seed{i}.pt
Results     : <output_dir>/results_vit.csv  +  summary_vit.csv

Usage
-----
  python run_vit.py                             # small, all seeds 0-9
  python run_vit.py --variant large
  python run_vit.py --seeds 0 1 2 --device cuda:0
  python run_vit.py --variant small large       # run both
"""

import argparse
import csv
import json
import math
import sys
import time
import traceback
import collections
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from run_resnet18 import MeltingCurveImageDataset, evaluate   # reuse dataset + eval
from model_vit import ViTClassifier


# ---------------------------------------------------------------------------
# Variant configs
# ---------------------------------------------------------------------------

VARIANT_CONFIGS = {
    # patch_size=16 → 196 patches — close to the 217-token patching in model.py
    "small": dict(patch_size=16, d_model=128, n_heads=8, n_layers=1, d_ff=256, dropout=0.2),
    "large": dict(patch_size=16, d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.2),
}


# ---------------------------------------------------------------------------
# Training loop (single seed × single variant)
# ---------------------------------------------------------------------------

def train_one_seed(seed: int, variant: str, args, device: torch.device) -> dict:
    data_dir = Path(args.data_root) / f"prompt3_splits_prompts_seed{seed}"
    with open(args.label_encoder) as f:
        label_encoder = json.load(f)
    num_classes = len(label_encoder)

    print(f"\n[seed={seed} | variant={variant}] Loading datasets from {data_dir}")
    train_ds = MeltingCurveImageDataset(str(data_dir / "train.csv"), label_encoder, augment=True)
    val_ds   = MeltingCurveImageDataset(str(data_dir / "val.csv"),   label_encoder, augment=False)
    test_ds  = MeltingCurveImageDataset(str(data_dir / "test.csv"),  label_encoder, augment=False)

    # num_workers=0: matplotlib is not multiprocess-safe
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    cfg   = VARIANT_CONFIGS[variant]
    model = ViTClassifier(num_classes=num_classes, **cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seed={seed} | vit_{variant}] params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

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

    # Load best checkpoint → evaluate
    model.load_state_dict(best_state)
    best_val_m = evaluate(model, val_loader,  device)
    test_m     = evaluate(model, test_loader, device)

    print(
        f"[seed={seed} | vit_{variant}] best_epoch={best_epoch}  "
        f"val_loss={best_val_m['loss']:.4f}  val_acc={best_val_m['acc']:.4f}  "
        f"test_acc={test_m['acc']:.4f}  test_f1={test_m['f1']:.4f}"
    )

    # Save checkpoint
    ckpt_dir  = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"vit_{variant}_seed{seed}.pt"
    torch.save(best_state, ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    return {
        "model":         f"vit_{variant}",
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
    p = argparse.ArgumentParser(description="ViT baseline for melting curve classification")
    p.add_argument("--variant",    nargs="+", default=["small"],
                   choices=list(VARIANT_CONFIGS.keys()),
                   help="Model variant(s) to run (default: small)")
    p.add_argument("--seeds",      nargs="+", type=int, default=list(range(10)))
    p.add_argument("--data_root",  type=str,  default=str(ROOT))
    p.add_argument("--label_encoder", type=str, default=str(ROOT / "label_encoder.json"))
    p.add_argument("--output_dir", type=str,  default=str(ROOT / "results_vit"))
    p.add_argument("--device",     type=str,  default="auto")
    # Training
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--patience",      type=int,   default=40)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    return p.parse_args()


def main():
    args   = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device   : {device}")
    print(f"Variants : {args.variant}")
    print(f"Seeds    : {args.seeds}")
    print(f"Epochs   : {args.epochs}  Patience : {args.patience}")

    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_vit.csv"
    summary_path = out_dir / "summary_vit.csv"

    fieldnames = [
        "model", "seed",
        "best_val_loss", "best_val_acc", "best_val_f1",
        "test_acc", "test_f1",
        "best_epoch", "train_time_s", "n_params",
    ]

    csv_file = open(results_path, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if results_path.stat().st_size == 0:
        writer.writeheader()

    all_results = []

    for variant in args.variant:
        for seed in args.seeds:
            try:
                row = train_one_seed(seed, variant, args, device)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                csv_file.flush()
                all_results.append(row)
            except Exception as e:
                print(f"ERROR [variant={variant}, seed={seed}]: {e}")
                traceback.print_exc()

    csv_file.close()

    if not all_results:
        print("No results.")
        return

    # Summary: group by model variant
    grouped = collections.defaultdict(list)
    for r in all_results:
        grouped[r["model"]].append(r)

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "n_seeds",
            "test_acc_mean", "test_acc_std",
            "test_f1_mean",  "test_f1_std",
        ])
        w.writeheader()
        for model_name, rows in grouped.items():
            w.writerow({
                "model":         model_name,
                "n_seeds":       len(rows),
                "test_acc_mean": f"{np.mean([r['test_acc'] for r in rows]):.4f}",
                "test_acc_std":  f"{np.std([r['test_acc']  for r in rows]):.4f}",
                "test_f1_mean":  f"{np.mean([r['test_f1']  for r in rows]):.4f}",
                "test_f1_std":   f"{np.std([r['test_f1']   for r in rows]):.4f}",
            })

    print(f"\nResults → {results_path}")
    print(f"Summary → {summary_path}")
    print("\n" + "=" * 65)
    print(f"{'Model':<16} {'Test Acc':>18} {'Test F1':>18}")
    print("-" * 65)
    for model_name, rows in grouped.items():
        accs = [r["test_acc"] for r in rows]
        f1s  = [r["test_f1"]  for r in rows]
        print(
            f"{model_name:<16} "
            f"{np.mean(accs):.4f}±{np.std(accs):.4f}  "
            f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
