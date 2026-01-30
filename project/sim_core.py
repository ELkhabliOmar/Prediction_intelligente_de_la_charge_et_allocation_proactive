# project/sim_core.py (VERSION COMPLÈTE CORRIGÉE)
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
            self.status = "active"
            self.device_type = "Fog"

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
# Models (LSTM + DQN) - INCHANGÉS
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
# Module1 - INCHANGÉ
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


# =========================================================
# Module2 - VERSION CORRIGÉE: Horizontal Only + Scale Down Actif
# =========================================================
class Module2_HVWPO_Planner:
    """
    Module 2: H-VWPO (Horizontal-Vertical Workload Prediction & Offloading).
    VERSION: Scale Up Horizontal Only, Scale Down Actif
    """
    def __init__(self, target_util=0.70, min_fog_cpu=30, ema_alpha=0.25, 
                 cooldown_windows=2, max_scale_mult=4.0, down_threshold=0.30):
        print(f"[Module2] ✅ H-VWPO Planner initialisé (Horizontal Only)")
        print(f"  - Target={target_util}, DownTh={down_threshold}")
        print(f"  - Scale Up: Horizontal Only (pas de vertical up)")
        print(f"  - Scale Down: Horizontal + Vertical")
        self.target_util = float(target_util)
        self.down_threshold = float(down_threshold)
        self.min_fog_cpu = int(min_fog_cpu)
        self.ema_alpha = float(ema_alpha)
        self.cooldown_windows = int(cooldown_windows)
        self.max_scale_mult = float(max_scale_mult)

        # Tracking EMA
        self.ema_predicted_active_cpu: Optional[float] = None
        
        # ✅ Cooldowns séparés
        self.last_scale_up_t = -10**9
        self.last_scale_down_t = -10**9
        
        # ✅ Fenêtre plus longue pour stabilité downscale
        self.pressure_window = deque(maxlen=8)

    def plan(self, predictions, total_fog_capacity, total_incoming_demand, current_pressure, 
             current_t, W_window, worst_pressure=0.0, active_nodes_count=1):
        
        pred_h5 = predictions.get(5, {"prediction": current_pressure, "uncertainty": 0.10})
        pred_p = float(pred_h5.get("prediction", current_pressure))
        unc = float(pred_h5.get("uncertainty", 0.10))

        robust_p = min(max(pred_p + (unc * 0.5), 0.0), 3.0)

        current_cap = float(max(1.0, total_fog_capacity))
        predicted_active_cpu = (robust_p * current_cap)

        # EMA adaptatif (plus rapide pour downscale)
        alpha = self.ema_alpha
        if self.ema_predicted_active_cpu is not None and predicted_active_cpu < self.ema_predicted_active_cpu:
            alpha = 0.70  # Decay rapide pour détecter baisse

        if self.ema_predicted_active_cpu is None:
            self.ema_predicted_active_cpu = predicted_active_cpu
        else:
            self.ema_predicted_active_cpu = alpha * predicted_active_cpu + (1.0 - alpha) * self.ema_predicted_active_cpu
        ema_cpu = float(self.ema_predicted_active_cpu)

        required_cpu = ema_cpu / max(self.target_util, 0.3)

        # ✅ Cooldowns différenciés
        cooldown_ticks = max(1, W_window) * self.cooldown_windows
        in_cooldown_up = (current_t - self.last_scale_up_t) < cooldown_ticks
        in_cooldown_down = (current_t - self.last_scale_down_t) < (cooldown_ticks * 0.6)

        # ✅ Fenêtre de pression pour stabilité
        self.pressure_window.append(current_pressure)
        avg_pressure = sum(self.pressure_window) / len(self.pressure_window)
        max_pressure_window = max(self.pressure_window)
        
        decision = "none"
        reason = "within band"
        
        # ✅ SEUILS RÉGLÉS POUR SIMULATION RÉALISTE
        up_th = 0.85    # Seuil haut
        down_th = self.down_threshold  # 0.30

        # ===== DÉCISION DE SCALING =====
        
        # ✅ SCALE UP: Horizontal Only (condition stricte)
        if not in_cooldown_up:
            if current_pressure > up_th or worst_pressure > 0.95:
                decision = "up"
                reason = f"high pressure (cur={current_pressure:.2f}, worst={worst_pressure:.2f})"
        
        # ✅ SCALE DOWN: Conditions ASSOUPLIES
        if decision == "none" and not in_cooldown_down and active_nodes_count > 1:
            
            # Condition 1: Pression moyenne basse sur fenêtre
            if avg_pressure < 0.35 and max_pressure_window < 0.50:
                decision = "down"
                reason = f"low stable pressure (avg={avg_pressure:.2f}, max={max_pressure_window:.2f})"
            
            # Condition 2: Prédiction très basse
            elif ema_cpu < current_cap * 0.40:
                decision = "down"
                reason = f"low predicted demand (ema={ema_cpu:.0f}, cap={current_cap:.0f})"
            
            # Condition 3: Tous les nœuds très peu chargés
            elif worst_pressure < 0.25:
                decision = "down"
                reason = f"all nodes idle (worst={worst_pressure:.2f})"

        # --- Offload (INCHANGÉ) ---
        offload_ratio = 0.0
        offload_reason = "no offload"
        demand_vs_capacity = ema_cpu / max(current_cap, 1.0)

        if demand_vs_capacity > 0.95:
            excess_ratio = min(1.0, (demand_vs_capacity - 0.95) * 2.0)
            offload_ratio = min(0.90, excess_ratio)
            offload_reason = f"demand/capacity={demand_vs_capacity:.2f}"

        if current_pressure > 0.90:
            safety_offload = (current_pressure - 0.90) * 5.0
            safety_offload = min(1.0, max(0.0, safety_offload))
            offload_ratio = max(offload_ratio, safety_offload)
            offload_reason += f" | SAFETY pressure={current_pressure:.2f}"

        if current_pressure < 0.40:
            offload_ratio = 0.0
            offload_reason = "low pressure"

        # ✅ Mise à jour des timestamps
        if decision == "up":
            self.last_scale_up_t = current_t
        elif decision == "down":
            self.last_scale_down_t = current_t

        return {
            "robust_pred": float(robust_p),
            "pred_active_cpu": float(predicted_active_cpu),
            "ema_active_cpu": float(ema_cpu),
            "scale_decision": decision,
            "scale_reason": reason,
            "offload_ratio": float(offload_ratio),
            "offload_reason": offload_reason,
        }


# =========================================================
# Module3 - INCHANGÉ
# =========================================================
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
        if pressure >= 0.95:
            return "Cloud", False

        if pressure < 0.40:
            offload_ratio = 0.0
        if t <= self.warmup_ticks or (not self.use_dqn) or (self.dqn is None) or (self.dqn_fallback_count >= self.max_fallback):
            return self.baseline(task_cpu, offload_ratio, pressure), True

        cpu_norm = min(float(task_cpu) / 500.0, 2.0)
        ram_norm = min(float(task_ram) / 4096.0, 2.0)
        pressure_clip = min(max(float(pressure), 0.0), 3.0)
        fog_cpu_norm = float(fog_cpu) / 200.0

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
# Workload helpers - INCHANGÉS
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
# Multi-node helpers - INCHANGÉS
# =========================================================
def active_cpu_on_server(active_services: List[Service], server: EdgeServer) -> int:
    return int(sum(s.cpu_demand for s in active_services if getattr(s, "placed_on_server", None) == server))

def pressure_server(active_services: List[Service], server: EdgeServer) -> float:
    cpu = active_cpu_on_server(active_services, server)
    return float(cpu) / max(1.0, float(server.cpu))

def pick_best_fog(fogs: List[EdgeServer], active_services: List[Service], task_cpu: int = 0) -> Tuple[EdgeServer, float]:
    active_fogs = [f for f in fogs if getattr(f, "status", "active") == "active"]
    if not active_fogs:
        return fogs[0], 10.0

    best = active_fogs[0]
    
    def get_projected_pressure(srv):
        current_load = active_cpu_on_server(active_services, srv)
        return (current_load + task_cpu) / max(1.0, float(srv.cpu))

    best_p = get_projected_pressure(best)

    for f in active_fogs[1:]:
        p = get_projected_pressure(f)
        if p < best_p:
            best, best_p = f, p
    return best, best_p

def pick_most_loaded_active_fog(fogs: List[EdgeServer], active_services: List[Service]) -> Tuple[EdgeServer, float]:
    active_fogs = [f for f in fogs if getattr(f, "status", "active") == "active"]
    if not active_fogs: return fogs[0], 0.0
    
    worst = active_fogs[0]
    worst_p = pressure_server(active_services, worst)
    for f in active_fogs[1:]:
        p = pressure_server(active_services, f)
        if p > worst_p:
            worst, worst_p = f, p
    return worst, worst_p

def pick_least_loaded_active_fog(fogs: List[EdgeServer], active_services: List[Service]) -> Tuple[EdgeServer, float]:
    return pick_best_fog(fogs, active_services, task_cpu=0)

# =========================================================
# Helper vertical downscale - AMÉLIORÉ
# =========================================================
def _vertical_downscale(fog_node, active_services, min_fog_cpu=30):
    """
    ✅ Vertical downscale uniquement (pas de up)
    """
    base_cpu = getattr(fog_node, "base_cpu", fog_node.cpu)
    min_cap = max(int(min_fog_cpu), int(base_cpu * 0.25))
    
    current_load = active_cpu_on_server(active_services, fog_node)
    
    # Nouvelle capacité = max(charge actuelle * 1.20, min_cap)
    new_cpu = int(max(min_cap, current_load * 1.20))
    
    # Downscale si on peut réduire d'au moins 15%
    if new_cpu < fog_node.cpu * 0.85:
        old_cpu = fog_node.cpu
        fog_node.cpu = new_cpu
        old_pressure = current_load / old_cpu if old_cpu > 0 else 0
        new_pressure = current_load / new_cpu if new_cpu > 0 else 0
        print(f"[scale] ✅ VERTICAL DOWN on {fog_node.name}: "
              f"{old_cpu} → {new_cpu} CPU (load={current_load}, "
              f"pressure {old_pressure:.2f} → {new_pressure:.2f})")
        return True
    return False

# =========================================================
# Global simulation state
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
TOTAL_ENERGY_JOULES = 0.0

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
# Main algorithm - VERSION CORRIGÉE: Horizontal Only
# =========================================================
def proactive_placement_algorithm(parameters):
    global CURRENT_T, ACTIVE_SERVICES, PRESSURE_HISTORY, PREDICTIONS, PLAN, SIMULATION_METRICS, CLOUD_RR, TOTAL_SCALE_UP, TOTAL_SCALE_DOWN, TOTAL_ENERGY_JOULES

    simulator = parameters["simulator"]

    fogs = [s for s in EdgeServer.all() if ("Fog" in getattr(s, "device_type", "Fog") or s.name.startswith("f")) and not s.name.startswith("c")]
    clouds = [s for s in EdgeServer.all() if ("Cloud" in getattr(s, "device_type", "Cloud") or s.name.startswith("c"))]
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

        active_fogs_list = [f for f in fogs if getattr(f, "status", "active") == "active"]
        
        total_active_cpu_fog = sum(s.cpu_demand for s in ACTIVE_SERVICES 
                                   if getattr(s, "placed_on", None) == "Fog" 
                                   and getattr(s, "placed_on_server", None) in active_fogs_list)
                                   
        total_fog_capacity = sum(int(f.cpu) for f in active_fogs_list)
        pressure_global = float(total_active_cpu_fog) / max(1.0, float(total_fog_capacity))

        fog_choice, _ = pick_best_fog(fogs, ACTIVE_SERVICES, task_cpu=int(task["cpu_demand"]))

        fog_projected_load = active_cpu_on_server(ACTIVE_SERVICES, fog_choice) + task["cpu_demand"]
        fog_is_full = fog_projected_load > fog_choice.cpu

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

        if decision == "Fog" and fog_is_full:
            decision = "Cloud"

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
    active_fogs_list = [f for f in fogs if getattr(f, "status", "active") == "active"]
    total_active_cpu_fog = sum(s.cpu_demand for s in ACTIVE_SERVICES 
                               if getattr(s, "placed_on", None) == "Fog" 
                               and getattr(s, "placed_on_server", None) in active_fogs_list)
    total_fog_capacity = sum(int(f.cpu) for f in active_fogs_list)
    pressure_global_real = float(total_active_cpu_fog) / max(1.0, float(total_fog_capacity))
    PRESSURE_HISTORY.append(min(max(pressure_global_real, 0.0), 3.0))

    fog_pressures = [pressure_server(ACTIVE_SERVICES, f) for f in active_fogs_list]
    worst_fog_p = max(fog_pressures) if fog_pressures else 0.0

    # Énergie
    P_IDLE = 100.0
    P_MAX = 250.0
    energy_tick = 0.0
    for f in active_fogs_list:
        util = pressure_server(ACTIVE_SERVICES, f)
        power = P_IDLE + (P_MAX - P_IDLE) * min(1.0, util)
        energy_tick += power
    TOTAL_ENERGY_JOULES += energy_tick

    # ✅ 4) MAPE/Plan toutes W ticks
    if CURRENT_T > 0 and (CURRENT_T % W_WINDOW == 0):
        PREDICTIONS = MODULE1.predict(PRESSURE_HISTORY)
        current_p_clip = PRESSURE_HISTORY[-1] if PRESSURE_HISTORY else 0.0

        worst_p_now = max([pressure_server(ACTIVE_SERVICES, f) for f in active_fogs_list], default=0.0)

        PLAN = MODULE2.plan(
            predictions=PREDICTIONS,
            total_fog_capacity=float(total_fog_capacity),
            total_incoming_demand=float(total_incoming_demand),
            current_pressure=float(current_p_clip),
            current_t=int(CURRENT_T),
            W_window=int(W_WINDOW),
            worst_pressure=float(worst_p_now),
            active_nodes_count=len(active_fogs_list),
        )

        # ✅ SCALING HYBRIDE CORRIGÉ: HORIZONTAL ONLY
        if PLAN["scale_decision"] == "up":
            inactive_fogs = [f for f in fogs if getattr(f, "status", "active") == "inactive"]
            
            if inactive_fogs:
                # ✅ HORIZONTAL UP ONLY: Activer un nœud inactif
                new_node = inactive_fogs[0]
                new_node.status = "active"
                new_node.cpu = getattr(new_node, "base_cpu", new_node.cpu)
                TOTAL_SCALE_UP += 1
                print(f"[scale] ✅ HORIZONTAL UP: Activated {new_node.name} (cpu={new_node.cpu})")
            else:
                # ✅ PAS DE VERTICAL UP: Seulement horizontal
                print(f"[scale] ⚠️ No inactive nodes for horizontal up (pressure={current_p_clip:.2f})")

        elif PLAN["scale_decision"] == "down":
            if len(active_fogs_list) > 1:
                # Trier par charge croissante (les moins chargés d'abord)
                sorted_fogs = sorted(active_fogs_list, 
                                    key=lambda f: pressure_server(ACTIVE_SERVICES, f))
                
                target_fog = sorted_fogs[0]  # Le moins chargé
                target_pressure = pressure_server(ACTIVE_SERVICES, target_fog)
                target_load = active_cpu_on_server(ACTIVE_SERVICES, target_fog)
                
                # ✅ PRIORITÉ: Horizontal Down (désactiver nœud)
                if target_pressure < 0.25:  # Seuil bas pour down horizontal
                    other_fogs = sorted_fogs[1:]
                    other_capacity = sum(f.cpu for f in other_fogs)
                    total_load = sum(s.cpu_demand for s in ACTIVE_SERVICES 
                                   if getattr(s, "placed_on", None) == "Fog")
                    
                    future_pressure = total_load / max(1.0, other_capacity)
                    
                    if future_pressure < 0.70:  # Seuil de sécurité à 70%
                        target_fog.status = "inactive"
                        TOTAL_SCALE_DOWN += 1
                        print(f"[scale] ✅ HORIZONTAL DOWN: Deactivated {target_fog.name} "
                              f"(load={target_pressure:.2f}, future_p={future_pressure:.2f})")
                        # Après désactivation, on sort car on a déjà downscale
                        print_mape_block(CURRENT_T, PREDICTIONS, PLAN)
                        CURRENT_T += 1
                        return
                    else:
                        # Horizontal impossible → Vertical Down
                        if _vertical_downscale(target_fog, ACTIVE_SERVICES, MODULE2.min_fog_cpu):
                            TOTAL_SCALE_DOWN += 1
                
                # ✅ Vertical Down sur les nœuds peu chargés
                else:
                    downscaled_count = 0
                    for fog in sorted_fogs:
                        if pressure_server(ACTIVE_SERVICES, fog) < 0.40:  # Seuil à 40%
                            if _vertical_downscale(fog, ACTIVE_SERVICES, MODULE2.min_fog_cpu):
                                TOTAL_SCALE_DOWN += 1
                                downscaled_count += 1
                                if downscaled_count >= 2:
                                    break
            else:
                # Un seul nœud: Vertical Down uniquement
                if _vertical_downscale(active_fogs_list[0], ACTIVE_SERVICES, MODULE2.min_fog_cpu):
                    TOTAL_SCALE_DOWN += 1

        print_mape_block(CURRENT_T, PREDICTIONS, PLAN)

    # Affichage tick
    print_tick(
        t=CURRENT_T,
        active_cpu=int(total_active_cpu_fog),
        cap=int(total_fog_capacity),
        pressure=float(pressure_global_real),
        worst=float(worst_fog_p),
        fog_n=int(tasks_placed_fog_this_tick),
        cloud_n=int(tasks_placed_cloud_this_tick),
        fog_nodes=fog_nodes_used_this_tick,
        cloud_nodes=int(len(cloud_nodes_used_this_tick)),
    )

    # Metrics
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
        "energy_joules_cumul": float(TOTAL_ENERGY_JOULES),
    })

    CURRENT_T += 1