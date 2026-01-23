# train_lstm.py (UPDATED: best checkpoint + early stopping + LR scheduler)
# Fix: ReduceLROnPlateau(verbose=...) incompatible avec certaines versions torch
# => on supprime verbose et on log nous-mêmes les changements de LR.

import os
import csv
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH_DEFAULT = os.path.join("data", "workload.csv")
MODEL_DIR_DEFAULT = "models"
MODEL_PATH_DEFAULT = os.path.join(MODEL_DIR_DEFAULT, "lstm_util.pth")

FOG_CPU_DEFAULT = 100
SEQ_LEN_DEFAULT = 20
EPOCHS_DEFAULT = 50              # ↑ pour laisser early stopping choisir
LR_DEFAULT = 1e-3
BATCH_SIZE_DEFAULT = 64
SEED_DEFAULT = 42

PATIENCE_DEFAULT = 8             # early stopping patience
MIN_DELTA_DEFAULT = 1e-4         # amélioration min sur val_loss


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_workload(path: str) -> List[dict]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "task_id": int(row["task_id"]),
                "timestamp": int(row["timestamp"]),
                "service_type": row.get("service_type", ""),
                "cpu_demand": int(row["cpu_demand"]),
                "ram_demand": int(row["ram_demand"]),
                "duration": int(row["duration"]),
            })
    return rows


def heuristic_place_on(cpu_demand: int, threshold: int = 300) -> str:
    """Heuristique OFFLINE pour approximer Fog/Cloud."""
    return "Cloud" if cpu_demand > threshold else "Fog"


def build_pressure_series(workload: List[dict], fog_cpu: int) -> List[float]:
    """
    pressure(t) = active_cpu_fog(t) / fog_cpu
    duration-aware (approx offline proche de l’online)
    """
    if not workload:
        return []

    T = max(x["timestamp"] + max(1, x["duration"]) for x in workload) + 1
    active_cpu_by_t = [0.0 for _ in range(T)]

    for task in workload:
        t0 = int(task["timestamp"])
        dur = max(1, int(task["duration"]))
        cpu = float(task["cpu_demand"])

        placed = heuristic_place_on(int(task["cpu_demand"]))
        if placed != "Fog":
            continue

        t1 = min(T, t0 + dur)
        for t in range(t0, t1):
            active_cpu_by_t[t] += cpu

    fog_cpu = max(int(fog_cpu), 1)
    return [active_cpu_by_t[t] / float(fog_cpu) for t in range(T)]


def make_windows(series: List[float], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X[i] = series[i : i+seq_len]
    y[i] = series[i+seq_len]
    """
    if len(series) <= seq_len:
        return torch.empty(0, seq_len, 1), torch.empty(0, 1)

    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    return X_t, y_t


class LSTMUtil(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out_last = out[:, -1, :]
        return self.fc(out_last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH_DEFAULT)
    ap.add_argument("--model_out", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--fog_cpu", type=int, default=FOG_CPU_DEFAULT)
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN_DEFAULT)
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--lr", type=float, default=LR_DEFAULT)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--patience", type=int, default=PATIENCE_DEFAULT)
    ap.add_argument("--min_delta", type=float, default=MIN_DELTA_DEFAULT)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"{args.data} introuvable. Lance d'abord: python generate_workload.py")

    workload = load_workload(args.data)
    pressure = build_pressure_series(workload, fog_cpu=args.fog_cpu)

    if len(pressure) < args.seq_len + 5:
        raise RuntimeError(
            f"Série trop courte (len={len(pressure)}). "
            "Augmente ticks/duration dans generate_workload.py"
        )

    # Normalisation stable (max_pressure)
    max_pressure = max(pressure) if max(pressure) > 0 else 1.0
    pressure_norm = [p / max_pressure for p in pressure]

    X, y = make_windows(pressure_norm, seq_len=args.seq_len)
    if X.numel() == 0:
        raise RuntimeError("Impossible de créer des fenêtres (X vide). Vérifie seq_len et taille série.")

    # Split train/val (80/20)
    n = X.shape[0]
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMUtil(hidden_dim=32, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Scheduler: réduit LR si val_loss stagne
    # FIX: pas de verbose=... (certaines versions torch n'acceptent pas ce param)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, threshold=1e-4
    )

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = args.patience

    for epoch in range(args.epochs):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        # ---- scheduler + log LR change ----
        lr_before = opt.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after = opt.param_groups[0]["lr"]
        lr_changed = lr_after < lr_before

        # logs
        if (epoch + 1) % 5 == 0 or epoch == 0 or lr_changed:
            msg = (f"[train] epoch {epoch+1}/{args.epochs} "
                   f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                   f"lr={lr_after:.2e} best_val={best_val:.6f}")
            if lr_changed:
                msg += "  (lr reduced)"
            print(msg)

        # ---- BEST CHECKPOINT + EARLY STOPPING ----
        improved = (best_val - val_loss) > args.min_delta
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early_stop] stop at epoch {epoch+1} (best_epoch={best_epoch}, best_val={best_val:.6f})")
                break

    # restaurer le best
    if best_state is None:
        best_state = model.state_dict()
        best_epoch = epoch + 1

    # Save BEST checkpoint compatible test.py
    ckpt = {
        "state_dict": best_state,
        "seq_len": int(args.seq_len),
        "max_util": float(max_pressure),  # ici max_util = max_pressure
        "fog_cpu": int(args.fog_cpu),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "note": "Trained on pressure(t)=active_cpu_fog/fog_cpu (duration-aware), normalized by max_pressure."
    }
    torch.save(ckpt, args.model_out)
    print(f"[OK] Best modèle sauvegardé: {args.model_out}")
    print(f"     best_epoch={best_epoch}, best_val_loss={best_val:.6f}")
    print(f"     seq_len={args.seq_len}, max_util(max_pressure)={max_pressure:.4f}, fog_cpu={args.fog_cpu}")


if __name__ == "__main__":
    main()
