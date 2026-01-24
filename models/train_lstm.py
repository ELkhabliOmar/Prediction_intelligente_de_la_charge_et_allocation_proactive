# train_lstm.py - ENTRAINEMENT LSTM (COMPAT TEST.PY + CHECKPOINT ENRICHI)
# ✅ Corrigé: chemins config.py, import propre, pas de os.path.join("data", DEFAULT_TESTSET)
# ✅ Corrigé: default data = DEFAULT_TRAINSET (pas le testset)
# ✅ Corrigé: model_out = DEFAULT_LSTM
# ✅ Corrigé: création dossier models même si chemin simple
# ✅ Corrigé: scipy optionnel (fallback si non installé)

import os
import csv
import argparse
import random
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# scipy optionnel
try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ajout du dossier parent au path pour trouver config.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_TRAINSET, DEFAULT_LSTM

# =========================
# Defaults
# =========================
DATA_PATH_DEFAULT = DEFAULT_TRAINSET
MODEL_DIR_DEFAULT = "models"
MODEL_PATH_DEFAULT = DEFAULT_LSTM  # ex: "models/lstm_util.pth"

FOG_CPU_DEFAULT = 100
SEQ_LEN_DEFAULT = 30
EPOCHS_DEFAULT = 200
LR_DEFAULT = 1e-3
BATCH_SIZE_DEFAULT = 64
SEED_DEFAULT = 42

PATIENCE_DEFAULT = 25
MIN_DELTA_DEFAULT = 1e-5


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_workload(path: str) -> List[dict]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(_normalize_task_row(row))
    return rows


def _normalize_task_row(row: dict) -> dict:
    # format "déjà normalisé"
    if "task_id" in row and "timestamp" in row:
        return {
            "task_id": int(row["task_id"]),
            "timestamp": int(float(row["timestamp"])),
            "service_type": row.get("service_type", ""),
            "cpu_demand": int(float(row["cpu_demand"])),
            "ram_demand": int(float(row["ram_demand"])),
            "duration": int(float(row["duration"])),
        }

    # Tuple30K schema: TaskName,GenerationTime,TaskID,TaskSize,CyclesPerBit,TransBitRate,DDL,DataType,DeviceType
    task_size = float(row.get("TaskSize", 0.0))
    cycles_per_bit = float(row.get("CyclesPerBit", 0.0))
    trans_rate = max(1.0, float(row.get("TransBitRate", 1.0)))

    cpu_scale = 3000.0
    ram_scale = 1.0

    cpu_demand = int(max(1.0, (task_size * cycles_per_bit) / cpu_scale))
    ram_demand = int(max(64.0, task_size * ram_scale))
    duration = int(max(1, math.ceil(task_size / trans_rate)))

    return {
        "task_id": int(float(row.get("TaskID", 0))),
        "timestamp": int(float(row.get("GenerationTime", 0.0))),
        "service_type": row.get("DataType", row.get("DeviceType", "")),
        "cpu_demand": cpu_demand,
        "ram_demand": ram_demand,
        "duration": duration,
    }


def heuristic_place_on(cpu_demand: int, threshold: int = 300) -> str:
    # Heuristique simple pour construire une pression Fog "plausible" pour l'entraînement LSTM
    return "Cloud" if cpu_demand > threshold else "Fog"


def build_pressure_series(workload: List[dict], fog_cpu: int) -> List[float]:
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
    pressure = np.array([x / float(fog_cpu) for x in active_cpu_by_t], dtype=np.float32)

    # --- outliers robust (IQR) ---
    q25, q75 = np.percentile(pressure, [25, 75])
    iqr = max(1e-9, q75 - q25)
    lower = max(0.0, q25 - 1.5 * iqr)
    upper = q75 + 1.5 * iqr
    pressure = np.clip(pressure, lower, min(upper, 5.0))

    # --- smoothing (si scipy dispo) ---
    if SCIPY_OK and len(pressure) > 10:
        pressure = gaussian_filter1d(pressure, sigma=1.5)

    # --- bruit réaliste ---
    variability_mask = (pressure > 0.2) & (pressure < 0.8)
    variability = np.where(variability_mask, 0.05, 0.02)
    noise = np.random.normal(0.0, variability, size=len(pressure)).astype(np.float32)
    pressure = np.maximum(0.0, pressure + noise)

    # clip final
    pressure = np.minimum(pressure, 3.0)
    return pressure.tolist()


def augment_time_series(series: List[float], augmentation_factor: float = 0.1) -> List[float]:
    if len(series) < 10:
        return series

    augmented = np.array(series, dtype=np.float32)

    noise_level = float(np.std(augmented) * 0.1)
    noise = np.random.normal(0, noise_level, len(augmented)).astype(np.float32)
    augmented = np.maximum(0.0, augmented + noise)

    scale = 1.0 + float(np.random.uniform(-augmentation_factor, augmentation_factor))
    augmented = augmented * scale

    if len(augmented) > 50:
        shift = int(np.random.randint(-3, 4))
        if shift != 0:
            augmented = np.roll(augmented, shift)

    return augmented.tolist()


def make_windows(
    series: List[float],
    seq_len: int,
    stride: int = 1,
    augment: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(series) <= seq_len:
        return torch.empty(0, seq_len, 1), torch.empty(0, 1)

    X, y = [], []
    for i in range(0, len(series) - seq_len, stride):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])

    if augment and len(X) > 0:
        original = len(X)
        for _ in range(2):
            aug = augment_time_series(series)
            for i in range(0, len(aug) - seq_len, stride * 2):
                X.append(aug[i:i + seq_len])
                y.append(aug[i + seq_len])
        print(f"  Augmentation: {original} → {len(X)} échantillons")

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return X_t, y_t


# =========================
# Model
# =========================
class EnhancedLSTM(nn.Module):
    """
    ARCH utilisée en train ET en test (compat).
    """
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        att = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(att * lstm_out, dim=1)  # (B, H)
        context = self.dropout(context)
        out = self.fc_layers(context)  # (B, 1)
        return out


# =========================
# Train helpers
# =========================
def create_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    n = X.shape[0]
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    test_ds = TensorDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, optimizer, criterion, device, gradient_clip: float = 1.0):
    model.train()
    total_loss, total_samples = 0.0, 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)

        # ✅ pas de L2 manuel ici: AdamW gère weight_decay
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / max(1, total_samples)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
    return total_loss / max(1, total_samples)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Entraînement LSTM (compatible test.py)")
    parser.add_argument("--data", default=DATA_PATH_DEFAULT)
    parser.add_argument("--model_out", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--fog_cpu", type=int, default=FOG_CPU_DEFAULT)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--patience", type=int, default=PATIENCE_DEFAULT)
    parser.add_argument("--min_delta", type=float, default=MIN_DELTA_DEFAULT)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    # assure dossier models
    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(MODEL_DIR_DEFAULT, exist_ok=True)
        args.model_out = os.path.join(MODEL_DIR_DEFAULT, args.model_out)

    print("=" * 70)
    print("ENTRAÎNEMENT LSTM (COMPATIBLE TEST.PY)")
    print("=" * 70)
    print(f"Données: {args.data}")
    print(f"Fog CPU: {args.fog_cpu}")
    print(f"Seq len: {args.seq_len}")
    print(f"Architecture: EnhancedLSTM hidden={args.hidden_dim}, layers={args.num_layers}, dropout={args.dropout}")
    print(f"Augmentation: {'NON' if args.no_augment else 'OUI'}")
    print(f"Smoothing scipy: {'OUI' if SCIPY_OK else 'NON (fallback)'}")
    print("=" * 70)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"{args.data} introuvable. Vérifie DEFAULT_TRAINSET dans config.py "
            f"ou passe --data <chemin.csv>."
        )

    workload = load_workload(args.data)
    pressure_series = build_pressure_series(workload, fog_cpu=args.fog_cpu)

    print(f"✓ Données chargées: {len(workload)} tâches")
    print(f"✓ Série temporelle générée: {len(pressure_series)} points")

    if len(pressure_series) < args.seq_len + 30:
        raise RuntimeError(
            f"Série trop courte ({len(pressure_series)} points). "
            f"Augmente la taille/variété du dataset ou baisse --seq_len."
        )

    pressure_array = np.array(pressure_series, dtype=np.float32)
    max_pressure = float(np.percentile(pressure_array, 95))
    if max_pressure < 0.1:
        max_pressure = 1.0

    print(f"\n🔧 NORMALISATION: max_util (p95) = {max_pressure:.3f}")
    # identique inference: clip norm à 2.0
    pressure_norm = [min(p / max_pressure, 2.0) for p in pressure_series]

    print("\n🔄 CRÉATION DES FENÊTRES...")
    X, y = make_windows(pressure_norm, seq_len=args.seq_len, augment=(not args.no_augment))
    if X.numel() == 0:
        raise RuntimeError("Impossible de créer des fenêtres. Vérifie --seq_len.")

    train_loader, val_loader, test_loader = create_dataloaders(X, y, batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n⚙️  Device: {device}")

    model = EnhancedLSTM(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 3,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    criterion = nn.HuberLoss(delta=0.1)
    print(f"Params totaux: {sum(p.numel() for p in model.parameters()):,}")

    print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT...")
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    best_state = None
    last_epoch_trained = 0

    for epoch in range(args.epochs):
        last_epoch_trained = epoch + 1
        tr = train_epoch(model, train_loader, optimizer, criterion, device, gradient_clip=1.0)
        va = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        improvement = best_val - va
        if improvement > args.min_delta:
            best_val = va
            best_epoch = epoch + 1
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or patience == 0:
            star = "🌟" if improvement > args.min_delta else ""
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch+1:3d}/{args.epochs}] "
                f"Train={tr:.6f} | Val={va:.6f} {star} | LR={lr_now:.2e} "
                f"| Best={best_val:.6f} (ep {best_epoch})"
            )

        if patience >= args.patience:
            print(f"\n⏹️ Early stopping à epoch {epoch+1} (pas d'amélioration depuis {args.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_loss = validate_epoch(model, test_loader, criterion, device)
    print(f"\n🧪 Test loss: {test_loss:.6f}")

    try:
        import pandas as pd
        ts = pd.Timestamp.now().isoformat()
    except Exception:
        ts = ""

    ckpt = {
        "arch": "EnhancedLSTM",
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "input_dim": 1,
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "seq_len": int(args.seq_len),
        "max_util": float(max_pressure),
        "fog_cpu": int(args.fog_cpu),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_loss": float(test_loss),
        "training_params": {
            "epochs_trained": int(last_epoch_trained),
            "lr": float(args.lr),
            "batch": int(args.batch),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "augment": bool(not args.no_augment),
        },
        "series_stats": {
            "min": float(pressure_array.min()),
            "max": float(pressure_array.max()),
            "mean": float(pressure_array.mean()),
            "std": float(pressure_array.std()),
            "p95": float(max_pressure),
        },
        "timestamp": ts,
        "note": "Checkpoint compatible test.py (EnhancedLSTM + meta).",
    }

    torch.save(ckpt, args.model_out)
    print(f"\n✅ MODÈLE SAUVEGARDÉ: {args.model_out}")
    print(f"   arch: {ckpt['arch']}")
    print(f"   seq_len: {ckpt['seq_len']}")
    print(f"   max_util(p95): {ckpt['max_util']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
