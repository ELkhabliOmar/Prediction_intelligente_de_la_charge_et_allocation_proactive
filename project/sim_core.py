# project/sim_core.py (corrigé: logs clairs + downscale stable + prédiction/plan AVANT metrics)
import csv
import math
import random
import os
from collections import deque, defaultdict
from typing import Dict, List, Any, DefaultDict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from project.ui_utils import print_tick, print_mape_block

# EdgeSimPy import + fallback stubs
try:
    from edge_sim_py import EdgeServer, Application, Service, ContainerImage
    EDGE_SIM_AVAILABLE = True
except ImportError:
    EDGE_SIM_AVAILABLE = False

    class EdgeServer:
        @staticmethod
        def all(): return []
        def __init__(self, cpu=100, memory=4096, disk=10000):
            self.cpu = cpu
            self.memory = memory
            self.disk = disk
            self.name = ""
            self.coordinates = [0, 0]
            self.base_cpu = cpu

    class Application:
        def __init__(self):
            self.name = ""
            self.image = None
            self.model = None

    class Service:
        def __init__(self, cpu_demand=0, memory_demand=0):
            self.cpu_demand = cpu_demand
            self.memory_demand = memory_demand
            self.name = ""
            self.application = None
            self.image = None
            self.model = None
            self.placed_on = None
            self.placed_on_server = None
            self.duration = 0
        def provision(self, server): pass

    class ContainerImage:
        def __init__(self):
            self.name = ""
            self.size = 0
            self.layers_digests = []

# ---------- Image global ----------
def build_global_image_no_layers() -> ContainerImage:
    img = ContainerImage()
    img.name = "task-image"
    img.size = 0
    img.layers_digests = []
    return img

GLOBAL_IMAGE = build_global_image_no_layers()

# =========================================================
# Models (LSTM + DQN)
# =========================================================
class EnhancedLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
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

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        att_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(att_w * lstm_out, dim=1)
        context = self.dropout(context)
        return self.fc_layers(context)

class DQN(nn.Module):
    def __init__(self, input_dim=5, output_dim=2, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    def forward(self, x): return self.net(x)

# =========================================================
# Module1 / Module2 / Module3
# =========================================================
class Module1_LSTMPredictor:
    def __init__(self, model_path: str, horizons=None, device="cpu"):
        self.horizons = horizons or [5, 15, 30, 60]
        self.seq_len = 30
        self.max_util = 1.0
        self.hidden_dim = 128
        self.num_layers = 2
        self.dropout = 0.3
        self.model_loaded = False
        self.device = device
        self.model: Optional[nn.Module] = None

        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location="cpu")
                self.seq_len = int(ckpt.get("seq_len", self.seq_len))
                self.max_util = float(ckpt.get("max_util", self.max_util))
                self.hidden_dim = int(ckpt.get("hidden_dim", self.hidden_dim))
                self.num_layers = int(ckpt.get("num_layers", self.num_layers))
                self.dropout = float(ckpt.get("dropout", self.dropout))

                self.model = EnhancedLSTM(
                    input_dim=1,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                ).to(self.device)
                self.model.load_state_dict(ckpt["state_dict"], strict=True)
                self.model.eval()
                self.model_loaded = True
                print(f"[Module1] ✅ LSTM chargé: {model_path}")
            except Exception as e:
                print(f"[Module1] ❌ Erreur chargement LSTM: {e} -> fallback")
                self.model_loaded = False
        else:
            print(f"[Module1] ⚠️ modèle LSTM introuvable: {model_path} -> fallback")

    @staticmethod
    def _rolling_std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.05
        arr = np.array(values, dtype=float)
        return float(max(0.01, arr.std()))

    def predict(self, pressure_history: deque) -> Dict[int, Dict[str, float]]:
        hist_all = list(pressure_history)
        if len(hist_all) < 3:
            return {h: {"prediction": 0.10, "uncertainty": 0.05, "used_fallback": True} for h in self.horizons}

        last_p = float(hist_all[-1])
        last5 = hist_all[-5:] if len(hist_all) >= 5 else hist_all
        mean_last_5 = float(sum(last5) / len(last5))

        if (not self.model_loaded) or (self.model is None):
            pred = max(0.0, last_p * 0.7) if (last_p < 0.2 and mean_last_5 < 0.3) else last_p
            unc = max(0.05, 0.08 + 0.08 * min(pred, 2.0))
            return {
                h: {
                    "prediction": float(pred),
                    "uncertainty": float(min(unc + 0.01 * h, 0.5)),
                    "used_fallback": True,
                }
                for h in self.horizons
            }

        seq_len = self.seq_len
        hist = hist_all[-seq_len:]
        if len(hist) < seq_len:
            hist = [hist[-1]] * (seq_len - len(hist)) + hist

        hist_clip = [max(0.0, min(x, 3.0)) for x in hist]
        hist_norm = [max(0.0, min(x / self.max_util, 3.0)) for x in hist_clip]

        x = torch.tensor(hist_norm, dtype=torch.float32).view(1, seq_len, 1).to(self.device)
        with torch.no_grad():
            y_norm = float(self.model(x).item())
        pred_p = max(0.0, y_norm * self.max_util)

        if mean_last_5 < 0.3:
            pred_p = min(pred_p, mean_last_5 * 2.0 + 0.3)
        if last_p < 0.1 and pred_p > 0.5:
            pred_p *= 0.3
        pred_p = min(pred_p, 3.0)
        if abs(pred_p - last_p) > 1.0:
            pred_p = 0.7 * last_p + 0.3 * pred_p

        std10 = self._rolling_std(hist_all[-10:])
        variance_factor = max(0.05, std10)

        preds = {}
        for h in self.horizons:
            base_unc = 0.05 + 0.12 * min(pred_p, 2.0)
            trend_unc = 0.01 * (h / 5.0)
            unc = base_unc + 0.5 * variance_factor + trend_unc
            unc = max(0.05, min(unc, 0.5))
            preds[h] = {"prediction": float(pred_p), "uncertainty": float(unc), "used_fallback": False}
        return preds


class Module2_HVWPO_Planner:
    """
    Module 2: H-VWPO (Horizontal-Vertical Workload Prediction & Offloading).
    TYPE: Algorithme Heuristique (Règles fixes) - Pas d'entraînement nécessaire.

    Implémente la stratégie proactive :
    1. Vertical : Scaling des ressources Fog (Scale UP/DOWN) basé sur la prédiction LSTM.
    2. Horizontal : Calcul du ratio de délestage (Offloading) vers le Cloud.
    """
    def __init__(self, target_util=0.70, min_fog_cpu=30, ema_alpha=0.25, cooldown_windows=2, max_scale_mult=4.0):
        print(f"[Module2] ✅ H-VWPO Planner initialisé (Target={target_util}, ScaleMult={max_scale_mult})")
        self.target_util = float(target_util)
        self.min_fog_cpu = int(min_fog_cpu)
        self.ema_alpha = float(ema_alpha)
        self.cooldown_windows = int(cooldown_windows)
        self.max_scale_mult = float(max_scale_mult)
        self.last_scale_t = -10**9
        self.ema_predicted_active_cpu: Optional[float] = None

        # ✅ stabilité downscale
        self.low_pressure_windows = 0

    def plan(self, predictions, total_fog_capacity, total_incoming_demand, current_pressure, current_t, W_window, worst_pressure=0.0):
        pred_h5 = predictions.get(5, {"prediction": current_pressure, "uncertainty": 0.10})
        pred_p = float(pred_h5.get("prediction", current_pressure))
        unc = float(pred_h5.get("uncertainty", 0.10))

        robust_p = min(max(pred_p + (unc * 0.5), 0.0), 3.0)

        current_cap = float(max(1.0, total_fog_capacity))
        # Correction: robust_p inclut déjà l'historique (donc la demande récente). On n'ajoute pas total_incoming_demand en double.
        predicted_active_cpu = (robust_p * current_cap)

        alpha = self.ema_alpha
        if self.ema_predicted_active_cpu is not None and predicted_active_cpu < self.ema_predicted_active_cpu:
            alpha = 0.80  # decay TRES rapide après un pic pour favoriser le downscale

        if self.ema_predicted_active_cpu is None:
            self.ema_predicted_active_cpu = predicted_active_cpu
        else:
            self.ema_predicted_active_cpu = alpha * predicted_active_cpu + (1.0 - alpha) * self.ema_predicted_active_cpu
        ema_cpu = float(self.ema_predicted_active_cpu)

        required_cpu = ema_cpu / max(self.target_util, 0.3)

        cooldown_ticks = max(1, W_window) * self.cooldown_windows
        in_cooldown = (current_t - self.last_scale_t) < cooldown_ticks

        # ✅ stabilité downscale (fenêtres)
        LOW_P_TH = 0.75  # Augmenté à 0.75 : si on est sous la cible + marge, on compte comme "basse pression"
        LOW_P_N = 1      # Réduit au minimum (était 2) : réaction immédiate
        if current_pressure < LOW_P_TH:
            self.low_pressure_windows += 1
        else:
            self.low_pressure_windows = 0

        decision = "none"
        reason = "within band"
        
        up_th, down_th = 1.15, 0.95  # Seuil down très agressif (0.95) : si on a 5% de trop, on réduit

        if not in_cooldown:
            if required_cpu > current_cap * up_th:
                decision = "up"
                reason = f"required_cpu({required_cpu:.1f}) > {up_th}*cap({current_cap:.1f})"
            elif required_cpu < current_cap * down_th:
                # Downscale si pression stablement basse OU demande très faible (<30%) immédiate
                if self.low_pressure_windows >= LOW_P_N or required_cpu < current_cap * 0.30:
                    decision = "down"
                    reason = f"low demand (win={self.low_pressure_windows}) | required_cpu({required_cpu:.1f}) < {down_th}*cap"
                else:
                    # Debug: dire pourquoi on ne downscale pas encore
                    reason = f"waiting stability (win={self.low_pressure_windows}/{LOW_P_N}) | required < cap"

        # --- Garde-fous ---
        if decision == "up" and current_pressure < 0.30:
            decision = "none"
            reason = f"cancelled (low global pressure {current_pressure:.2f})"

        if worst_pressure > 0.95 and decision != "up" and not in_cooldown:
            decision = "up"
            reason = f"emergency scale (worst_fog={worst_pressure:.2f})"

        # --- Offload ---
        offload_ratio = 0.0
        offload_reason = "no offload"
        demand_vs_capacity = ema_cpu / max(current_cap, 1.0)

        if demand_vs_capacity > 0.95:
            # ✅ CORRECTION: Délestage proactif dès 95% de la demande estimée
            excess_ratio = min(1.0, (demand_vs_capacity - 0.95) * 2.0)
            offload_ratio = min(0.90, excess_ratio)
            offload_reason = f"demand/capacity={demand_vs_capacity:.2f}"

        if current_pressure > 0.90:
            # ✅ CORRECTION: Sécurité agressive. Si p=1.0 -> 50% offload. Si p=1.1 -> 100% offload.
            safety_offload = (current_pressure - 0.90) * 5.0
            safety_offload = min(1.0, max(0.0, safety_offload))
            
            offload_ratio = max(offload_ratio, safety_offload)
            offload_reason += f" | SAFETY pressure={current_pressure:.2f}"

        if current_pressure < 0.40:
            offload_ratio = 0.0
            offload_reason = "low pressure"

        if ema_cpu < current_cap * 0.15:
            offload_ratio = 0.0
            offload_reason = "very low demand"

        if decision in ("up", "down"):
            self.last_scale_t = current_t

        return {
            "robust_pred": float(robust_p),
            "pred_active_cpu": float(predicted_active_cpu),
            "ema_active_cpu": float(ema_cpu),
            "scale_decision": decision,
            "scale_reason": reason + (f" | cooldown({cooldown_ticks})" if in_cooldown else ""),
            "offload_ratio": float(offload_ratio),
            "offload_reason": offload_reason,
        }


class Module3_Scheduler:
    def __init__(self, dqn_path: str = None, cpu_threshold_cloud=300, warmup_ticks=15):
        self.cpu_threshold_cloud = int(cpu_threshold_cloud)
        self.warmup_ticks = int(warmup_ticks)
        self.use_dqn = False
        self.dqn_fallback_count = 0
        self.max_fallback = 25
        self.hidden_dim = 128
        self.dropout = 0.1
        self.dqn = None

        if dqn_path and os.path.exists(dqn_path):
            try:
                ckpt = torch.load(dqn_path, map_location="cpu")
                self.hidden_dim = int(ckpt.get("hidden_dim", 128))
                self.dropout = float(ckpt.get("dropout", 0.1))
                self.dqn = DQN(hidden_dim=self.hidden_dim, dropout=self.dropout)
                self.dqn.load_state_dict(ckpt["state_dict"], strict=True)
                self.dqn.eval()
                self.use_dqn = True
                print(f"[Module3] ✅ DQN chargé: {dqn_path}")
            except Exception as e:
                print(f"[Module3] ❌ Erreur chargement DQN ({e}) -> baseline")
                self.use_dqn = False
        else:
            print("[Module3] ℹ️ DQN absent -> baseline")

    def baseline(self, task_cpu: int, offload_ratio: float, pressure: float) -> str:
        if pressure < 0.40:
            offload_ratio = 0.0
        if random.random() < offload_ratio:
            return "Cloud"
        if pressure > 0.85 and task_cpu > int(self.cpu_threshold_cloud * 0.7):
            return "Cloud"
        if task_cpu > self.cpu_threshold_cloud:
            return "Cloud"
        return "Fog"

    def decide(self, task_cpu: int, task_ram: int, pressure: float, fog_cpu: int, offload_ratio: float, t: int):
        # ✅ SAFETY OVERRIDE: Si le Fog est saturé (>95%), on force le Cloud immédiatement
        # Cela protège le système même si le DQN ou le Planner sont en retard.
        if pressure >= 0.95:
            return "Cloud", False

        if pressure < 0.40:
            offload_ratio = 0.0
        if t <= self.warmup_ticks or (not self.use_dqn) or (self.dqn is None) or (self.dqn_fallback_count >= self.max_fallback):
            return self.baseline(task_cpu, offload_ratio, pressure), True

        cpu_norm = min(float(task_cpu) / 500.0, 2.0)
        ram_norm = min(float(task_ram) / 4096.0, 2.0)
        pressure_clip = min(max(float(pressure), 0.0), 3.0)
        fog_cpu_norm = float(fog_cpu) / 200.0  # identique training

        state = torch.tensor([cpu_norm, ram_norm, pressure_clip, fog_cpu_norm, float(offload_ratio)],
                             dtype=torch.float32).unsqueeze(0)

        try:
            with torch.no_grad():
                q_vals = self.dqn(state)
                if torch.any(torch.isnan(q_vals)) or torch.any(torch.isinf(q_vals)):
                    self.dqn_fallback_count += 1
                    return self.baseline(task_cpu, offload_ratio, pressure), True
                a = int(torch.argmax(q_vals, dim=1).item())
                return ("Fog" if a == 0 else "Cloud"), False
        except Exception:
            self.dqn_fallback_count += 1
            return self.baseline(task_cpu, offload_ratio, pressure), True

# =========================================================
# Workload helpers
# =========================================================
def load_workload_indexed(path: str) -> DefaultDict[int, List[dict]]:
    idx = defaultdict(list)
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            task = _normalize_task_row(row)
            idx[task["timestamp"]].append(task)
    return idx

def _normalize_task_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if "task_id" in row and "timestamp" in row:
        return {
            "task_id": int(row["task_id"]),
            "timestamp": int(float(row["timestamp"])),
            "service_type": row.get("service_type", "NA"),
            "cpu_demand": int(float(row["cpu_demand"])),
            "ram_demand": int(float(row["ram_demand"])),
            "duration": int(float(row["duration"])),
        }

    # --- Conversion du format Tuple30K (Raw) vers Simulation (Normalized) ---
    # Colonnes CSV: TaskName, GenerationTime, TaskID, TaskSize, CyclesPerBit, TransBitRate, ...
    # Formules de conversion :
    task_size = float(row.get("TaskSize", 0.0))
    cycles_per_bit = float(row.get("CyclesPerBit", 0.0))
    trans_rate = max(1.0, float(row.get("TransBitRate", 1.0)))

    cpu_scale = 3000.0
    ram_scale = 1.0

    cpu_demand = int(max(1.0, (task_size * cycles_per_bit) / cpu_scale))
    ram_demand = int(max(64.0, task_size * ram_scale))
    duration = int(max(1.0, math.ceil(task_size / trans_rate)))

    return {
        "task_id": int(float(row.get("TaskID", 0))),
        "timestamp": int(float(row.get("GenerationTime", 0.0))),
        "service_type": row.get("DataType", row.get("DeviceType", "NA")),
        "cpu_demand": cpu_demand,
        "ram_demand": ram_demand,
        "duration": duration,
    }

# =========================================================
# Multi-node helpers
# =========================================================
def active_cpu_on_server(active_services: List[Service], server: EdgeServer) -> int:
    return int(sum(s.cpu_demand for s in active_services if getattr(s, "placed_on_server", None) == server))

def pressure_server(active_services: List[Service], server: EdgeServer) -> float:
    cpu = active_cpu_on_server(active_services, server)
    return float(cpu) / max(1.0, float(server.cpu))

def pick_best_fog(fogs: List[EdgeServer], active_services: List[Service]) -> Tuple[EdgeServer, float]:
    best = fogs[0]
    best_p = pressure_server(active_services, best)
    for f in fogs[1:]:
        p = pressure_server(active_services, f)
        if p < best_p:
            best, best_p = f, p
    return best, best_p

def pick_most_loaded_fog(fogs: List[EdgeServer], active_services: List[Service]) -> Tuple[EdgeServer, float]:
    worst = fogs[0]
    worst_p = pressure_server(active_services, worst)
    for f in fogs[1:]:
        p = pressure_server(active_services, f)
        if p > worst_p:
            worst, worst_p = f, p
    return worst, worst_p

def pick_least_loaded_fog(fogs: List[EdgeServer], active_services: List[Service]) -> Tuple[EdgeServer, float]:
    return pick_best_fog(fogs, active_services)

# =========================================================
# Global simulation state (géré ici)
# =========================================================
WORKLOAD_IDX = defaultdict(list)
ACTIVE_SERVICES: List[Service] = []
PRESSURE_HISTORY = deque(maxlen=200)

CURRENT_T = 0
SIMULATION_METRICS: List[Dict[str, Any]] = []
PREDICTIONS: Dict[int, Dict[str, float]] = {}
PLAN: Dict[str, Any] = {"scale_decision": "none", "offload_ratio": 0.0}

MODULE1: Optional[Module1_LSTMPredictor] = None
MODULE2: Optional[Module2_HVWPO_Planner] = None
MODULE3: Optional[Module3_Scheduler] = None

W_WINDOW = 10
CLOUD_RR = 0
TOTAL_SCALE_UP = 0
TOTAL_SCALE_DOWN = 0
TOTAL_ENERGY_JOULES = 0.0  # ✅ Nouveau compteur énergie

def setup_state(workload_idx, module1, module2, module3, W: int, **kwargs):
    global WORKLOAD_IDX, MODULE1, MODULE2, MODULE3, W_WINDOW, TOTAL_SCALE_UP, TOTAL_SCALE_DOWN, TOTAL_ENERGY_JOULES
    WORKLOAD_IDX = workload_idx
    MODULE1 = module1
    MODULE2 = module2
    MODULE3 = module3
    W_WINDOW = int(W)
    TOTAL_SCALE_UP = 0
    TOTAL_SCALE_DOWN = 0
    TOTAL_ENERGY_JOULES = 0.0

def get_metrics():
    return SIMULATION_METRICS

# =========================================================
# Main algorithm
# =========================================================
def proactive_placement_algorithm(parameters):
    global CURRENT_T, ACTIVE_SERVICES, PRESSURE_HISTORY, PREDICTIONS, PLAN, SIMULATION_METRICS, CLOUD_RR, TOTAL_SCALE_UP, TOTAL_SCALE_DOWN, TOTAL_ENERGY_JOULES

    simulator = parameters["simulator"]

    fogs = [s for s in EdgeServer.all() if "fog" in s.name.lower()]
    clouds = [s for s in EdgeServer.all() if "cloud" in s.name.lower()]
    if not fogs or not clouds:
        print("[ERROR] Pools Fog/Cloud non trouvés (vérifie les noms des serveurs).")
        return

    for f in fogs:
        if not hasattr(f, "base_cpu"):
            f.base_cpu = f.cpu

    # 1) Fin des services
    remaining = []
    for s in ACTIVE_SERVICES:
        s.duration -= 1
        if s.duration > 0:
            remaining.append(s)
    ACTIVE_SERVICES = remaining

    # 2) Injection + Scheduling
    tasks_now = WORKLOAD_IDX.get(CURRENT_T, [])
    total_incoming_demand = sum(t["cpu_demand"] for t in tasks_now)

    tasks_placed_fog_this_tick = 0
    tasks_placed_cloud_this_tick = 0
    dqn_fallback_used_tick = 0

    # ✅ Fix confusion: on compte aussi les nœuds distincts utilisés ce tick
    fog_nodes_used_this_tick = set()
    cloud_nodes_used_this_tick = set()

    for task in tasks_now:
        app = Application()
        app.name = f"App-{task['task_id']}"
        app.image = GLOBAL_IMAGE
        app.model = simulator

        service = Service(cpu_demand=task["cpu_demand"], memory_demand=task["ram_demand"])
        service.name = f"Task-{task['task_id']}"
        service.application = app
        service.image = GLOBAL_IMAGE
        service.model = simulator
        service.duration = int(task["duration"])

        total_active_cpu_fog = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
        total_fog_capacity = sum(int(f.cpu) for f in fogs)
        pressure_global = float(total_active_cpu_fog) / max(1.0, float(total_fog_capacity))

        fog_choice, _ = pick_best_fog(fogs, ACTIVE_SERVICES)

        decision, used_fallback_dqn = MODULE3.decide(
            task_cpu=int(task["cpu_demand"]),
            task_ram=int(task["ram_demand"]),
            pressure=float(pressure_global),
            fog_cpu=int(fog_choice.cpu),
            offload_ratio=float(PLAN.get("offload_ratio", 0.0)),
            t=int(CURRENT_T),
        )
        if used_fallback_dqn:
            dqn_fallback_used_tick += 1

        service.placed_on = decision

        if decision == "Fog":
            service.placed_on_server = fog_choice
            fog_nodes_used_this_tick.add(getattr(fog_choice, "name", "Fog"))
            tasks_placed_fog_this_tick += 1
            service.provision(fog_choice)
        else:
            cloud_node = clouds[CLOUD_RR % len(clouds)]
            CLOUD_RR += 1
            service.placed_on_server = cloud_node
            cloud_nodes_used_this_tick.add(getattr(cloud_node, "name", "Cloud"))
            tasks_placed_cloud_this_tick += 1
            service.provision(cloud_node)

        ACTIVE_SERVICES.append(service)

    # 3) Monitoring
    total_active_cpu_fog = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
    total_fog_capacity = sum(int(f.cpu) for f in fogs)
    pressure_global_real = float(total_active_cpu_fog) / max(1.0, float(total_fog_capacity))
    PRESSURE_HISTORY.append(min(max(pressure_global_real, 0.0), 3.0))

    fog_pressures = [pressure_server(ACTIVE_SERVICES, f) for f in fogs]
    worst_fog_p = max(fog_pressures) if fog_pressures else 0.0

    # ✅ AMÉLIORATION: Calcul Énergie (Modèle Linéaire)
    # Hypothèse: Serveur Fog = 100W idle, 250W max
    P_IDLE = 100.0
    P_MAX = 250.0
    energy_tick = 0.0
    for f in fogs:
        util = pressure_server(ACTIVE_SERVICES, f)
        # Power = Idle + (Max - Idle) * Utilization
        power = P_IDLE + (P_MAX - P_IDLE) * min(1.0, util)
        energy_tick += power  # Watts * 1 sec = Joules
    TOTAL_ENERGY_JOULES += energy_tick

    # ✅ 4) MAPE/Plan toutes W ticks (AVANT metrics pour cohérence CSV)
    if CURRENT_T > 0 and (CURRENT_T % W_WINDOW == 0):
        PREDICTIONS = MODULE1.predict(PRESSURE_HISTORY)
        current_p_clip = PRESSURE_HISTORY[-1] if PRESSURE_HISTORY else 0.0

        worst_p_now = max([pressure_server(ACTIVE_SERVICES, f) for f in fogs], default=0.0)

        PLAN = MODULE2.plan(
            predictions=PREDICTIONS,
            total_fog_capacity=float(total_fog_capacity),
            total_incoming_demand=float(total_incoming_demand),
            current_pressure=float(current_p_clip),
            current_t=int(CURRENT_T),
            W_window=int(W_WINDOW),
            worst_pressure=float(worst_p_now),
        )

        # scaling sur 1 fog
        if PLAN["scale_decision"] in ("up", "down"):
            if PLAN["scale_decision"] == "up":
                target_fog, p = pick_most_loaded_fog(fogs, ACTIVE_SERVICES)
                step = max(20, int(target_fog.cpu * 0.25))
                max_cap = int(getattr(target_fog, "base_cpu", target_fog.cpu) * MODULE2.max_scale_mult)
                target_fog.cpu = int(min(max_cap, target_fog.cpu + step))
                TOTAL_SCALE_UP += 1
                print(f"[scale] UP on {target_fog.name} (p={p:.2f})")
            else:
                target_fog, p = pick_least_loaded_fog(fogs, ACTIVE_SERVICES)
                step = max(20, int(target_fog.cpu * 0.20))
                min_cap = max(int(MODULE2.min_fog_cpu), int(getattr(target_fog, "base_cpu", target_fog.cpu) * 0.4))
                target_fog.cpu = int(max(min_cap, target_fog.cpu - step))
                TOTAL_SCALE_DOWN += 1
                print(f"[scale] DOWN on {target_fog.name} (p={p:.2f})")

        # Affichage tableau MAPE
        print_mape_block(CURRENT_T, PREDICTIONS, PLAN)

    # ✅ Affichage tick (maintenant avec nodes_used)
    print_tick(
        t=CURRENT_T,
        active_cpu=int(total_active_cpu_fog),
        cap=int(total_fog_capacity),
        pressure=float(pressure_global_real),
        worst=float(worst_fog_p),
        fog_n=int(tasks_placed_fog_this_tick),
        cloud_n=int(tasks_placed_cloud_this_tick),
        fog_nodes=int(len(fog_nodes_used_this_tick)),
        cloud_nodes=int(len(cloud_nodes_used_this_tick)),
    )

    # ✅ Metrics (cohérents: PREDICTIONS/PLAN déjà mis à jour)
    pred_h5 = PREDICTIONS.get(5, {})
    SIMULATION_METRICS.append({
        "t": int(CURRENT_T),
        "active_cpu_fog": int(total_active_cpu_fog),
        "fog_capacity": int(total_fog_capacity),
        "pressure": float(pressure_global_real),
        "worst_fog_pressure": float(worst_fog_p),
        "predicted_pressure": float(pred_h5.get("prediction", pressure_global_real)),
        "prediction_uncertainty": float(pred_h5.get("uncertainty", 0.10)),
        "lstm_fallback_used": int(pred_h5.get("used_fallback", True)) if pred_h5 else 1,
        "dqn_fallback_used_tasks": int(dqn_fallback_used_tick),
        "scale_decision": str(PLAN.get("scale_decision", "none")),
        "offload_ratio": float(PLAN.get("offload_ratio", 0.0)),
        "offload_reason": str(PLAN.get("offload_reason", "")),
        "tasks_on_fog": int(sum(1 for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")),
        "tasks_on_cloud": int(sum(1 for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Cloud")),
        "tasks_placed_fog": int(tasks_placed_fog_this_tick),
        "tasks_placed_cloud": int(tasks_placed_cloud_this_tick),
        "fog_nodes_used_tick": int(len(fog_nodes_used_this_tick)),
        "cloud_nodes_used_tick": int(len(cloud_nodes_used_this_tick)),
        "scale_up_total": int(TOTAL_SCALE_UP),
        "scale_down_total": int(TOTAL_SCALE_DOWN),
        "energy_joules_cumul": float(TOTAL_ENERGY_JOULES), # ✅ Métrique sauvegardée
    })

    CURRENT_T += 1
