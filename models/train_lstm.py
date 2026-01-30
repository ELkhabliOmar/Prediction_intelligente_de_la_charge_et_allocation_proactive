# train_lstm.py - LSTM avec prédictions probabilistes et meilleure gestion d'incertitude
# VERSION CORRIGÉE (scheduler compatible)
import os
import csv
import argparse
import random
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Deque
from collections import deque
import json

import numpy as np
try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("⚠️ scipy non disponible, lissage désactivé")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_TRAINSET, DEFAULT_LSTM

# =========================
# Defaults
# =========================
DATA_PATH_DEFAULT = DEFAULT_TRAINSET
MODEL_PATH_DEFAULT = DEFAULT_LSTM.replace(".pth", "_improved.pth")

# Hyperparamètres optimisés
SEQ_LEN_DEFAULT = 50  # Augmenté pour capturer plus de patterns
EPOCHS_DEFAULT = 300
LR_DEFAULT = 5e-4
BATCH_SIZE_DEFAULT = 128
HIDDEN_DIM_DEFAULT = 256
NUM_LAYERS_DEFAULT = 3
DROPOUT_DEFAULT = 0.4
MC_DROPOUT_SAMPLES = 30  # Nombre d'échantillons pour Monte Carlo Dropout

# =========================
# Fonctions de chargement et prétraitement
# =========================
def load_workload(path: str) -> List[dict]:
    """Charge les données de workload"""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(_normalize_task_row(row))
    return rows

def _normalize_task_row(row: dict) -> dict:
    """Normalise une ligne de task"""
    # Format déjà normalisé
    if "task_id" in row and "timestamp" in row:
        return {
            "task_id": int(row["task_id"]),
            "timestamp": int(float(row["timestamp"])),
            "service_type": row.get("service_type", ""),
            "cpu_demand": int(float(row["cpu_demand"])),
            "ram_demand": int(float(row["ram_demand"])),
            "duration": int(float(row["duration"])),
        }

    # Tuple30K schema
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
    """Heuristique simple de placement"""
    return "Cloud" if cpu_demand > threshold else "Fog"

def extract_temporal_features(timestamps: List[int]) -> Dict[str, List[float]]:
    """Extrait des features temporelles pour améliorer les prédictions"""
    features = {
        'hour_of_day': [],
        'day_of_week': [],
        'is_peak_hour': [],
        'trend': []
    }
    
    if not timestamps:
        return features
    
    # Normaliser les timestamps
    ts_min = min(timestamps)
    ts_max = max(timestamps)
    ts_range = ts_max - ts_min if ts_max > ts_min else 1
    
    # Calculer des features temporelles
    for ts in timestamps:
        # Feature cyclique: heure du jour (si les timestamps sont en secondes)
        hour = (ts % 86400) / 86400  # Normalisé entre 0 et 1
        features['hour_of_day'].append(hour)
        
        # Jour de la semaine (si applicable)
        day = (ts // 86400) % 7 / 7  # Normalisé
        features['day_of_week'].append(day)
        
        # Pics horaires (8h-20h)
        is_peak = 1.0 if 0.33 <= hour <= 0.83 else 0.0
        features['is_peak_hour'].append(is_peak)
        
        # Tendance temporelle
        trend = (ts - ts_min) / ts_range
        features['trend'].append(trend)
    
    return features

def build_enhanced_pressure_series(workload: List[dict], fog_cpu: int) -> Tuple[List[float], Dict[str, List[float]]]:
    """Construit une série de pression avec features temporelles"""
    if not workload:
        return [], {}
    
    # Extraire tous les timestamps
    timestamps = [task["timestamp"] for task in workload]
    temporal_features = extract_temporal_features(timestamps)
    
    # Trouver la durée totale
    max_time = 0
    for task in workload:
        end_time = task["timestamp"] + max(1, task["duration"])
        if end_time > max_time:
            max_time = end_time
    
    T = max_time + 1
    active_cpu_by_t = [0.0 for _ in range(T)]
    task_count_by_t = [0 for _ in range(T)]
    
    for task in workload:
        t0 = int(task["timestamp"])
        dur = max(1, int(task["duration"]))
        cpu = float(task["cpu_demand"])
        
        # Placement plus réaliste
        placement_score = (
            cpu / fog_cpu * 0.6 +
            (task["ram_demand"] / 1024) * 0.2
        )
        
        if placement_score < 0.7:  # Place sur Fog
            t1 = min(T, t0 + dur)
            for t in range(t0, t1):
                active_cpu_by_t[t] += cpu
                task_count_by_t[t] += 1
    
    fog_cpu = max(int(fog_cpu), 1)
    pressure = np.array([x / float(fog_cpu) for x in active_cpu_by_t], dtype=np.float32)
    task_density = np.array([x for x in task_count_by_t], dtype=np.float32)
    
    # Normaliser task_density
    if task_density.max() > 0:
        task_density = task_density / task_density.max()
    
    # Traitement des outliers avec méthode robuste
    q10, q90 = np.percentile(pressure, [10, 90])
    iqr = q90 - q10
    lower = max(0.0, q10 - 2.5 * iqr)
    upper = q90 + 2.5 * iqr
    pressure = np.clip(pressure, lower, min(upper, 5.0))
    
    # Lissage adaptatif
    if SCIPY_OK and len(pressure) > 20:
        # Plus de lissage pour les séries bruyantes
        noise_level = np.std(pressure) / (np.mean(pressure) + 1e-9)
        sigma = 2.0 if noise_level > 0.5 else 1.0
        pressure = gaussian_filter1d(pressure, sigma=sigma)
    
    # Feature engineering additionnel
    temporal_features_series = {
        'pressure': pressure.tolist(),
        'task_density': task_density.tolist() if len(task_density) > 0 else [],
        'hour_of_day': [temporal_features['hour_of_day'][t % len(temporal_features['hour_of_day'])] 
                       if t < len(temporal_features['hour_of_day']) else 0.0 
                       for t in range(len(pressure))],
        'trend': np.linspace(0, 1, len(pressure)).tolist()
    }
    
    return pressure.tolist(), temporal_features_series

# =========================
# Modèle Bayesian LSTM amélioré
# =========================
class BayesianLSTM(nn.Module):
    """LSTM avec prédictions probabilistes et estimation d'incertitude"""
    def __init__(self, input_dim=1, hidden_dim=256, num_layers=3, dropout=0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        
        # LSTM avec dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Couches pour prédiction moyenne
        self.mean_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Couche pour prédiction de la variance (log variance)
        self.logvar_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Pour garantir variance positive
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, return_uncertainty=False, mc_samples=1):
        """
        Forward pass avec option pour Monte Carlo Dropout
        """
        if mc_samples > 1:
            # Mode Monte Carlo
            means, logvars = [], []
            for _ in range(mc_samples):
                # Activer dropout
                F.dropout(x, p=self.dropout_p, training=True)
                
                lstm_out, _ = self.lstm(x)
                
                # Attention
                att_weights = self.attention(lstm_out)
                att_noise = torch.randn_like(att_weights) * 0.1
                att_weights = F.softmax(att_weights + att_noise, dim=1)
                
                context = torch.sum(att_weights * lstm_out, dim=1)
                context = F.dropout(context, p=self.dropout_p, training=True)
                
                mean = self.mean_layer(context)
                logvar = self.logvar_layer(context)
                means.append(mean)
                logvars.append(logvar)
            
            mean = torch.stack(means).mean(dim=0)
            logvar = torch.stack(logvars).mean(dim=0)
            var = torch.exp(logvar)
            
            if return_uncertainty:
                uncertainty = 1.96 * torch.sqrt(var)  # Intervalle de confiance à 95%
                return mean, uncertainty
            
            return mean
        else:
            # Forward normal
            lstm_out, _ = self.lstm(x)
            
            # Attention
            att_weights = self.attention(lstm_out)
            att_weights = F.softmax(att_weights, dim=1)
            
            context = torch.sum(att_weights * lstm_out, dim=1)
            context = F.dropout(context, p=self.dropout_p, training=self.training)
            
            mean = self.mean_layer(context)
            logvar = self.logvar_layer(context)
            var = torch.exp(logvar)
            
            if return_uncertainty:
                uncertainty = 1.96 * torch.sqrt(var)
                return mean, uncertainty
            
            return mean
    
    def predict_with_uncertainty(self, x, n_samples=MC_DROPOUT_SAMPLES):
        """Prédiction avec estimation d'incertitude via Monte Carlo Dropout"""
        self.train()  # Important: garder dropout activé pour MC
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, return_uncertainty=False, mc_samples=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # (n_samples, batch, 1)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Intervalle de confiance à 95%
        uncertainty = 1.96 * std_pred
        
        return mean_pred, uncertainty

# =========================
# Loss functions
# =========================
class NegativeLogLikelihoodLoss(nn.Module):
    """Loss NLL pour l'apprentissage probabiliste"""
    def __init__(self):
        super().__init__()
    
    def forward(self, mean_pred, logvar_pred, target):
        variance = torch.exp(logvar_pred)
        loss = 0.5 * (logvar_pred + (target - mean_pred)**2 / variance)
        return loss.mean()

class AleatoricEpistemicLoss(nn.Module):
    """Loss combinant incertitude aléatoire et épistémique"""
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, mean_pred, logvar_pred, target, epistemic_uncertainty=None):
        # Terme aléatoire
        aleatoric_loss = 0.5 * torch.mean(logvar_pred + (target - mean_pred)**2 / torch.exp(logvar_pred))
        
        # Terme épistémique (si disponible)
        if epistemic_uncertainty is not None:
            epistemic_loss = torch.mean(epistemic_uncertainty)
            loss = self.beta * aleatoric_loss + (1 - self.beta) * epistemic_loss
        else:
            loss = aleatoric_loss
            
        return loss

# =========================
# Fonctions d'entraînement
# =========================
def create_enhanced_dataset(pressure_series: List[float], 
                           features_dict: Dict[str, List[float]], 
                           seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crée un dataset avec multiples features"""
    series_len = len(pressure_series)
    
    if series_len <= seq_len:
        return torch.empty(0, seq_len, 4), torch.empty(0, 1)
    
    # Vérifier que toutes les séries ont la même longueur
    for key, feat in features_dict.items():
        if len(feat) != series_len:
            # Remplir avec des zéros
            features_dict[key] = feat[:series_len] + [0.0] * (series_len - len(feat))
    
    X, y = [], []
    for i in range(series_len - seq_len):
        # Combiner toutes les features
        seq_features = []
        for j in range(seq_len):
            features = [
                pressure_series[i + j],
                features_dict['task_density'][i + j],
                features_dict['hour_of_day'][i + j],
                features_dict['trend'][i + j]
            ]
            seq_features.append(features)
        X.append(seq_features)
        y.append(pressure_series[i + seq_len])
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

def augment_time_series(series: List[float], augmentation_factor: float = 0.1) -> List[float]:
    """Augmente les séries temporelles avec du bruit"""
    if len(series) < 10:
        return series
    
    augmented = np.array(series, dtype=np.float32)
    
    # Bruit gaussien
    noise_level = float(np.std(augmented) * 0.1)
    noise = np.random.normal(0, noise_level, len(augmented)).astype(np.float32)
    augmented = np.maximum(0.0, augmented + noise)
    
    # Changement d'échelle
    scale = 1.0 + float(np.random.uniform(-augmentation_factor, augmentation_factor))
    augmented = augmented * scale
    
    # Décalage temporel
    if len(augmented) > 50:
        shift = int(np.random.randint(-3, 4))
        if shift != 0:
            augmented = np.roll(augmented, shift)
    
    return augmented.tolist()

def make_windows_with_features(pressure_series: List[float], 
                              features_dict: Dict[str, List[float]],
                              seq_len: int, stride: int = 1,
                              augment: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crée des fenêtres avec features"""
    if len(pressure_series) <= seq_len:
        return torch.empty(0, seq_len, 4), torch.empty(0, 1)
    
    # Créer les fenêtres originales
    X, y = [], []
    for i in range(0, len(pressure_series) - seq_len, stride):
        seq_features = []
        for j in range(seq_len):
            features = [
                pressure_series[i + j],
                features_dict['task_density'][i + j],
                features_dict['hour_of_day'][i + j],
                features_dict['trend'][i + j]
            ]
            seq_features.append(features)
        X.append(seq_features)
        y.append(pressure_series[i + seq_len])
    
    # Augmentation
    if augment and len(X) > 0:
        original = len(X)
        for _ in range(2):
            aug_pressure = augment_time_series(pressure_series)
            # Pour la simplicité, on utilise les mêmes features
            for i in range(0, len(aug_pressure) - seq_len, stride * 2):
                seq_features = []
                for j in range(seq_len):
                    features = [
                        aug_pressure[i + j],
                        features_dict['task_density'][i + j] if i + j < len(features_dict['task_density']) else 0.0,
                        features_dict['hour_of_day'][i + j] if i + j < len(features_dict['hour_of_day']) else 0.0,
                        features_dict['trend'][i + j] if i + j < len(features_dict['trend']) else 0.0
                    ]
                    seq_features.append(features)
                X.append(seq_features)
                y.append(aug_pressure[i + seq_len])
        print(f"  Augmentation: {original} → {len(X)} échantillons")
    
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return X_t, y_t

def create_dataloaders(X: torch.Tensor, y: torch.Tensor, 
                      batch_size: int = 128,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15):
    """Crée les DataLoaders pour train/val/test"""
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
    """Entraîne le modèle pour une epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass avec prédiction de moyenne et variance
        mean_pred = model(x, return_uncertainty=False)
        
        # Pour la simplicité, on utilise MSE pour commencer
        # Dans une version plus avancée, on utiliserait la variance prédite
        loss = criterion(mean_pred, y)
        
        # Regularisation L2
        l2_reg = sum(p.norm(2) for p in model.parameters())
        loss = loss + 1e-5 * l2_reg
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    
    return total_loss / max(1, total_samples)

def validate_epoch(model, dataloader, criterion, device, mc_samples: int = 10):
    """Valide le modèle avec estimation d'incertitude"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_uncertainty = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Prédiction avec incertitude
            mean_pred, uncertainty = model.predict_with_uncertainty(x, n_samples=mc_samples)
            
            # Calcul des métriques
            loss = criterion(mean_pred, y)
            mae = F.l1_loss(mean_pred, y)
            
            total_loss += loss.item() * x.size(0)
            total_mae += mae.item() * x.size(0)
            total_uncertainty += uncertainty.mean().item() * x.size(0)
            total_samples += x.size(0)
    
    avg_loss = total_loss / max(1, total_samples)
    avg_mae = total_mae / max(1, total_samples)
    avg_uncertainty = total_uncertainty / max(1, total_samples)
    
    return avg_loss, avg_mae, avg_uncertainty

# =========================
# Fonction principale - CORRIGÉE
# =========================
def main():
    parser = argparse.ArgumentParser(description="Entraînement LSTM bayésien amélioré")
    parser.add_argument("--data", default=DATA_PATH_DEFAULT)
    parser.add_argument("--model_out", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--fog_cpu", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM_DEFAULT)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS_DEFAULT)
    parser.add_argument("--dropout", type=float, default=DROPOUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()
    
    # Initialisation des seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 70)
    print("LSTM BAYÉSIEN AMÉLIORÉ - Entraînement")
    print("=" * 70)
    print(f"Données: {args.data}")
    print(f"Fog CPU: {args.fog_cpu}")
    print(f"Seq len: {args.seq_len}")
    print(f"Architecture: BayesianLSTM hidden={args.hidden_dim}, layers={args.num_layers}, dropout={args.dropout}")
    print(f"Augmentation: {'NON' if args.no_augment else 'OUI'}")
    print(f"Smoothing scipy: {'OUI' if SCIPY_OK else 'NON'}")
    print("=" * 70)
    
    # Vérifier le fichier de données
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Fichier introuvable: {args.data}")
    
    # Chargement des données
    workload = load_workload(args.data)
    print(f"✓ Données chargées: {len(workload)} tâches")
    
    # Construction des séries
    pressure_series, features_dict = build_enhanced_pressure_series(workload, args.fog_cpu)
    print(f"✓ Série temporelle générée: {len(pressure_series)} points")
    
    if len(pressure_series) < args.seq_len + 30:
        raise RuntimeError(f"Série trop courte ({len(pressure_series)} points). "
                          f"Augmentez le dataset ou réduisez --seq_len.")
    
    # Normalisation
    pressure_array = np.array(pressure_series, dtype=np.float32)
    max_pressure = float(np.percentile(pressure_array, 95))
    if max_pressure < 0.1:
        max_pressure = 1.0
    
    print(f"\n🔧 NORMALISATION: max_util (p95) = {max_pressure:.3f}")
    pressure_norm = [min(p / max_pressure, 2.0) for p in pressure_series]
    
    # Normaliser aussi dans le features_dict
    features_dict['pressure'] = pressure_norm
    
    # Création du dataset
    print("\n🔄 CRÉATION DES FENÊTRES...")
    X, y = make_windows_with_features(
        pressure_norm, 
        features_dict, 
        seq_len=args.seq_len, 
        augment=not args.no_augment
    )
    
    if X.numel() == 0:
        raise RuntimeError("Impossible de créer des fenêtres. Vérifiez --seq_len.")
    
    print(f"✓ Dataset créé: {X.shape[0]} échantillons")
    print(f"✓ Features par échantillon: {X.shape[2]}")
    
    # Split des données
    train_loader, val_loader, test_loader = create_dataloaders(
        X, y, 
        batch_size=args.batch,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Initialisation du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙️ Device: {device}")
    
    model = BayesianLSTM(
        input_dim=4,  # 4 features: pressure, task_density, hour_of_day, trend
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    
    print(f"✓ Modèle créé: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Optimiseur et scheduler - VERSION CORRIGÉE
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # CORRECTION ICI: Utiliser OneCycleLR au lieu de ReduceLROnPlateau
    # OneCycleLR est plus stable et compatible avec toutes les versions
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    
    # Création du scheduler avec OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 3,  # Pic de learning rate
        total_steps=total_steps,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% pour la montée
        anneal_strategy='cos',  # Décroissance cosinus
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,  # max_lr / initial_lr
        final_div_factor=1e4  # initial_lr / final_lr
    )
    
    criterion = nn.MSELoss()  # Commencer avec MSE simple
    
    # Entraînement
    print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT...")
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    best_state = None
    last_epoch_trained = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_uncertainty': [],
        'learning_rate': []
    }
    
    for epoch in range(args.epochs):
        last_epoch_trained = epoch + 1
        
        # Entraînement
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation avec incertitude
        val_loss, val_mae, val_uncertainty = validate_epoch(
            model, val_loader, criterion, device, mc_samples=10
        )
        
        # Mise à jour du scheduler (OneCycleLR se met à jour chaque batch)
        # Mais on peut aussi appeler step() après chaque epoch
        scheduler.step()
        
        # Sauvegarde de l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_uncertainty'].append(val_uncertainty)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Vérification early stopping
        improvement = best_val_loss - val_loss
        if improvement > 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_improvement = improvement
        else:
            patience_counter += 1
        
        # Affichage
        if (epoch + 1) % 5 == 0 or epoch == 0 or improvement > 1e-4:
            star = "🌟" if improvement > 1e-4 else ""
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch+1:3d}/{args.epochs}] "
                f"Train={train_loss:.6f} | Val={val_loss:.6f} (MAE={val_mae:.6f}) "
                f"Unc={val_uncertainty:.6f} {star} | LR={lr_now:.2e}"
            )
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️ Early stopping à epoch {epoch+1} (pas d'amélioration depuis {args.patience})")
            break
    
    # Charger le meilleur modèle
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Évaluation finale sur le test set
    model.eval()
    test_loss, test_mae, test_uncertainty = validate_epoch(
        model, test_loader, criterion, device, mc_samples=20
    )
    
    print(f"\n📊 RÉSULTATS FINAUX:")
    print(f"   Test Loss: {test_loss:.6f}")
    print(f"   Test MAE: {test_mae:.6f}")
    print(f"   Test Incertitude: {test_uncertainty:.6f}")
    print(f"   Ratio Incertitude/MAE: {test_uncertainty/(test_mae+1e-6):.2f}")
    
    # Sauvegarde du modèle
    try:
        import pandas as pd
        ts = pd.Timestamp.now().isoformat()
    except Exception:
        ts = ""
    
    checkpoint = {
        'arch': 'BayesianLSTM',
        'state_dict': best_state if best_state is not None else model.state_dict(),
        'model_config': {
            'input_dim': 4,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'seq_len': args.seq_len,
        },
        'normalization': {
            'pressure_mean': float(np.mean(pressure_array)),
            'pressure_std': float(np.std(pressure_array)),
            'pressure_max': float(max_pressure),
            'pressure_p95': float(np.percentile(pressure_array, 95)),
        },
        'fog_cpu': int(args.fog_cpu),
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'test_metrics': {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'test_uncertainty': float(test_uncertainty),
        },
        'training_params': {
            'epochs_trained': int(last_epoch_trained),
            'lr': float(args.lr),
            'batch_size': int(args.batch),
            'patience': int(args.patience),
            'augment': bool(not args.no_augment),
        },
        'features_info': {
            'used_features': list(features_dict.keys()),
            'feature_descriptions': {
                'pressure': 'Normalized CPU pressure',
                'task_density': 'Normalized task arrival density',
                'hour_of_day': 'Cyclical hour feature',
                'trend': 'Linear trend feature'
            }
        },
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'timestamp': ts,
        'note': 'Bayesian LSTM with uncertainty estimation - Compatible with test.py',
    }
    
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(checkpoint, args.model_out)
    
    print(f"\n✅ MODÈLE SAUVEGARDÉ: {args.model_out}")
    print(f"   arch: {checkpoint['arch']}")
    print(f"   seq_len: {checkpoint['model_config']['seq_len']}")
    print(f"   max_util(p95): {checkpoint['normalization']['pressure_max']:.4f}")
    
    # Sauvegarder l'historique séparément
    history_file = args.model_out.replace('.pth', '_history.json')
    with open(history_file, 'w') as f:
        json.dump(checkpoint['history'], f, indent=2)
    print(f"✓ Historique sauvegardé: {history_file}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()