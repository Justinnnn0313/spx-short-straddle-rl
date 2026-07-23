from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ExitMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, threshold: float) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).long()
    y_long = y.long()
    tp = int(((pred == 1) & (y_long == 1)).sum().item())
    tn = int(((pred == 0) & (y_long == 0)).sum().item())
    fp = int(((pred == 1) & (y_long == 0)).sum().item())
    fn = int(((pred == 0) & (y_long == 1)).sum().item())
    total = max(1, len(y_long))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(pred.float().mean().item()),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train oracle-guided exit imitation MLP.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data = np.load(args.dataset)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    min_hold_bars = int(data["min_hold_bars"][0]) if "min_hold_bars" in data else None

    order = rng.permutation(len(y))
    val_n = int(len(y) * args.val_fraction)
    val_idx = order[:val_n]
    train_idx = order[val_n:]
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_scaled = (x - mean) / std

    train_ds = TensorDataset(
        torch.from_numpy(x_scaled[train_idx]),
        torch.from_numpy(y[train_idx]),
    )
    val_x = torch.from_numpy(x_scaled[val_idx])
    val_y = torch.from_numpy(y[val_idx])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = ExitMLP(x.shape[1], args.hidden_dim)
    pos = float(y[train_idx].sum())
    neg = float(len(train_idx) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())
            val_metrics = metrics_from_logits(val_logits, val_y, args.threshold)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_loss={val_loss:.6f} val_f1={val_metrics['f1']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f}",
            flush=True,
        )

    model_path = output_dir / "exit_imitation_model.pt"
    summary_path = output_dir / "training_summary.json"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": x.shape[1],
            "hidden_dim": args.hidden_dim,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "threshold": args.threshold,
            "min_hold_bars": min_hold_bars,
            "config": vars(args),
            "history": history,
            "positive_rate": float(y.mean()),
        },
        model_path,
    )
    summary = {
        "dataset": args.dataset,
        "rows": int(len(y)),
        "features": int(x.shape[1]),
        "positive_rate": float(y.mean()),
        "min_hold_bars": min_hold_bars,
        "model_path": str(model_path),
        "final": history[-1],
        "config": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"model: {model_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
