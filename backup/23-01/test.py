# test.py - SIMULATEUR INTÉGRÉ FOG-CLOUD (MULTI-FOG + MULTI-CLOUD)
# ✅ Compatible train_lstm.py (EnhancedLSTM checkpoint enrichi) et train_dqn.py (DQN checkpoint)
# ✅ Multi-Fog + Multi-Cloud (France/Europe)
# ✅ Pression globale (pool fog) pour LSTM/Planner
# ✅ Placement Fog -> Fog le moins chargé
# ✅ Placement Cloud -> Round-robin sur clouds
# ✅ Planner = pool scaling: scale sur 1 fog (UP: plus chargé, DOWN: moins chargé)
# ✅ IMPORTANT: DQN normalisation identique au training (fog_cpu_norm = fog_cpu / 200.0)

import argparse
import csv
import os
import random
import warnings
import math
from collections import deque, defaultdict
from typing import Dict, List, Any, DefaultDict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from config import (
    DEFAULT_WORKLOAD,
    DEFAULT_LSTM,
    DEFAULT_DQN,
    DEFAULT_RESULTS,
)

warnings.filterwarnings("ignore")

# =========================================================
# EdgeSimPy import + fallback stubs (pour éviter crash si absent)
# =========================================================
try:
    from edge_sim_py import Simulator, EdgeServer, Application, Service, ContainerImage
    EDGE_SIM_AVAILABLE = True
except ImportError:
    print("⚠️  edge-sim-py non installé. Installation: pip install edge-sim-py")
    EDGE_SIM_AVAILABLE = False

    class Simulator:
        def __init__(self, tick_duration=1, tick_unit="seconds"):
            self.tick_duration = tick_duration
            self.tick_unit = tick_unit
            self.stopping_criterion = None
            self.resource_management_algorithm = None
            self.resource_management_algorithm_parameters = None

        def run_model(self):
            pass

    class EdgeServer:
        @staticmethod
        def all():
            return []

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
            self.placed_on = None         # "Fog" / "Cloud"
            self.placed_on_server = None  # EdgeServer instance
            self.duration = 0

        def provision(self, server):
            pass

    class ContainerImage:
        def __init__(self):
            self.name = ""
            self.size = 0
            self.layers_digests = []


# =========================================================
# EdgeSimPy FIX: ContainerImage sans layers
# =========================================================
def build_global_image_no_layers() -> ContainerImage:
    img = ContainerImage()
    img.name = "task-image"
    img.size = 0
    img.layers_digests = []
    return img


GLOBAL_IMAGE = build_global_image_no_layers()


# =========================================================
# ENHANCED LSTM (doit matcher train_lstm.py)
# =========================================================
class EnhancedLSTM(nn.Module):
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

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        att_w = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(att_w * lstm_out, dim=1)  # (B, H)
        context = self.dropout(context)
        return self.fc_layers(context)  # (B,1)


# =========================================================
# DQN (doit matcher train_dqn.py)
# =========================================================
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

    def forward(self, x):
        return self.net(x)


# =========================================================
# Module 1: LSTM Predictor (robuste + compat)
# =========================================================
class Module1_LSTMPredictor:
    def __init__(self, model_path: str, horizons=None, device="cpu"):
        self.horizons = horizons or [5, 15, 30, 60]
        self.model_path = model_path
        self.device = device

        self.seq_len = 30
        self.max_util = 1.0
        self.hidden_dim = 128
        self.num_layers = 2
        self.dropout = 0.3
        self.model_loaded = False
        self.model: Optional[nn.Module] = None

        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location="cpu")
                self.seq_len = int(ckpt.get("seq_len", self.seq_len))
                self.max_util = float(ckpt.get("max_util", self.max_util))
                self.hidden_dim = int(ckpt.get("hidden_dim", self.hidden_dim))
                self.num_layers = int(ckpt.get("num_layers", self.num_layers))
                self.dropout = float(ckpt.get("dropout", self.dropout))
                arch = ckpt.get("arch", "EnhancedLSTM")

                if self.max_util <= 0.1:
                    self.max_util = 1.0
                if self.max_util > 10.0:
                    print(f"[Module1] ⚠️ max_util aberrant ({self.max_util:.3f}) -> clamp à 5.0")
                    self.max_util = 5.0
                if arch != "EnhancedLSTM":
                    print(f"[Module1] ⚠️ arch={arch} non reconnu -> tentative EnhancedLSTM")

                self.model = EnhancedLSTM(
                    input_dim=1,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                ).to(self.device)

                self.model.load_state_dict(ckpt["state_dict"], strict=True)
                self.model.eval()
                self.model_loaded = True
                print(
                    f"[Module1] ✅ LSTM chargé: {model_path} "
                    f"(seq_len={self.seq_len}, max_util={self.max_util:.3f}, "
                    f"H={self.hidden_dim}, L={self.num_layers}, drop={self.dropout})"
                )
            except Exception as e:
                print(f"[Module1] ❌ Erreur chargement LSTM: {e} -> fallback persistence")
                self.model_loaded = False
        else:
            print(f"[Module1] ⚠️ modèle LSTM introuvable: {model_path} -> fallback persistence")

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

        if not self.model_loaded or self.model is None:
            pred = max(0.0, last_p * 0.7) if (last_p < 0.2 and mean_last_5 < 0.3) else last_p
            unc = max(0.05, 0.08 + 0.08 * min(pred, 2.0))
            return {
                h: {"prediction": float(pred), "uncertainty": float(min(unc + 0.01 * h, 0.5)), "used_fallback": True}
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
# Module 2: Proactive Planner (pool scaling)
# =========================================================
class Module2_ProactivePlanner:
    def __init__(self, target_util=0.70, min_fog_cpu=30, ema_alpha=0.25, cooldown_windows=2, max_scale_mult=4.0):
        self.target_util = float(target_util)
        self.min_fog_cpu = int(min_fog_cpu)
        self.ema_alpha = float(ema_alpha)
        self.cooldown_windows = int(cooldown_windows)
        self.max_scale_mult = float(max_scale_mult)

        self.last_scale_t = -10**9
        self.ema_predicted_active_cpu: Optional[float] = None

        print(
            f"[Module2] Planner pool: target_util={self.target_util}, min_fog_cpu={self.min_fog_cpu}, "
            f"EMAα={self.ema_alpha}, cooldown_windows={self.cooldown_windows}, max_scale_mult={self.max_scale_mult}"
        )

    def plan(self, predictions, total_fog_capacity, total_incoming_demand, current_pressure, current_t, W_window):
        pred_h5 = predictions.get(5, {"prediction": current_pressure, "uncertainty": 0.10})
        pred_p = float(pred_h5.get("prediction", current_pressure))
        unc = float(pred_h5.get("uncertainty", 0.10))

        robust_p = min(max(pred_p + unc, 0.0), 3.0)

        current_cap = float(max(1.0, total_fog_capacity))
        predicted_active_cpu = (robust_p * current_cap) + float(total_incoming_demand)

        if self.ema_predicted_active_cpu is None:
            self.ema_predicted_active_cpu = predicted_active_cpu
        else:
            self.ema_predicted_active_cpu = (
                self.ema_alpha * predicted_active_cpu + (1.0 - self.ema_alpha) * self.ema_predicted_active_cpu
            )
        ema_cpu = float(self.ema_predicted_active_cpu)

        required_cpu = ema_cpu / max(self.target_util, 0.3)

        cooldown_ticks = max(1, W_window) * self.cooldown_windows
        in_cooldown = (current_t - self.last_scale_t) < cooldown_ticks

        decision = "none"
        reason = "within band"

        up_th = 1.20
        down_th = 0.70

        if not in_cooldown:
            if required_cpu > current_cap * up_th:
                decision = "up"
                reason = f"required_cpu({required_cpu:.1f}) > {up_th}*cap({current_cap:.1f})"
            elif required_cpu < current_cap * down_th:
                decision = "down"
                reason = f"required_cpu({required_cpu:.1f}) < {down_th}*cap({current_cap:.1f})"
            elif current_pressure < 0.15:
                decision = "down"
                reason = f"very low pressure({current_pressure:.2f})"

        offload_ratio = 0.0
        offload_reason = "no offload"

        demand_vs_capacity = ema_cpu / max(current_cap, 1.0)
        if demand_vs_capacity > 1.0:
            excess_ratio = min(1.0, (demand_vs_capacity - 1.0) * 0.4)
            offload_ratio = min(0.60, excess_ratio)
            offload_reason = f"demand/capacity={demand_vs_capacity:.2f}"

        if current_pressure > 1.0:
            safety_offload = 0.20 + 0.10 * (current_pressure - 1.0)
            safety_offload = min(0.60, safety_offload)
            offload_ratio = max(offload_ratio, safety_offload)
            offload_reason += f" | SAFETY pressure={current_pressure:.2f}"

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


# =========================================================
# Module 3: Scheduler (DQN + baseline) — NORMALISATION IDENTIQUE AU TRAINING
# =========================================================
class Module3_Scheduler:
    def __init__(self, dqn_path: str = None, cpu_threshold_cloud=300, warmup_ticks=15):
        self.cpu_threshold_cloud = int(cpu_threshold_cloud)
        self.dqn_path = dqn_path
        self.use_dqn = False
        self.warmup_ticks = int(warmup_ticks)

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
                self.dqn = DQN(input_dim=5, output_dim=2, hidden_dim=self.hidden_dim, dropout=self.dropout)
                self.dqn.load_state_dict(ckpt["state_dict"], strict=True)
                self.dqn.eval()
                self.use_dqn = True
                print(f"[Module3] ✅ DQN chargé: {dqn_path} (hidden_dim={self.hidden_dim}, dropout={self.dropout})")
            except Exception as e:
                print(f"[Module3] ❌ Erreur chargement DQN ({e}) -> baseline")
                self.use_dqn = False
        else:
            print("[Module3] ℹ️ DQN absent -> baseline")

    def baseline(self, task_cpu: int, offload_ratio: float, pressure: float) -> str:
        if random.random() < offload_ratio:
            return "Cloud"
        if pressure > 0.85 and task_cpu > int(self.cpu_threshold_cloud * 0.7):
            return "Cloud"
        if task_cpu > self.cpu_threshold_cloud:
            return "Cloud"
        return "Fog"

    def decide(self, task_cpu: int, task_ram: int, pressure: float, fog_cpu: int, offload_ratio: float, t: int) -> Tuple[str, bool]:
        if t <= self.warmup_ticks or (not self.use_dqn) or (self.dqn is None) or (self.dqn_fallback_count >= self.max_fallback):
            return self.baseline(task_cpu, offload_ratio, pressure), True

        # ✅ EXACTEMENT comme train_dqn.py
        cpu_norm = min(float(task_cpu) / 500.0, 2.0)
        ram_norm = min(float(task_ram) / 4096.0, 2.0)
        pressure_clip = min(max(float(pressure), 0.0), 3.0)
        fog_cpu_norm = float(fog_cpu) / 200.0  # identique training

        state = torch.tensor([cpu_norm, ram_norm, pressure_clip, fog_cpu_norm, float(offload_ratio)], dtype=torch.float32).unsqueeze(0)

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
# Dataset helpers
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
# Multi-node helper functions
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
# Global simulation state
# =========================================================
WORKLOAD_IDX = defaultdict(list)
ACTIVE_SERVICES: List[Service] = []
PRESSURE_HISTORY = deque(maxlen=200)

CURRENT_T = 0
SIMULATION_METRICS = []
PREDICTIONS = {}
PLAN = {"scale_decision": "none", "offload_ratio": 0.0}

MODULE1: Optional[Module1_LSTMPredictor] = None
MODULE2: Optional[Module2_ProactivePlanner] = None
MODULE3: Optional[Module3_Scheduler] = None

W_WINDOW = 10
CLOUD_RR = 0


def proactive_placement_algorithm(parameters):
    global CURRENT_T, ACTIVE_SERVICES, PRESSURE_HISTORY, PREDICTIONS, PLAN, SIMULATION_METRICS, CLOUD_RR

    simulator = parameters["simulator"]

    fogs = [s for s in EdgeServer.all() if "fog" in s.name.lower()]
    clouds = [s for s in EdgeServer.all() if "cloud" in s.name.lower()]

    if not fogs or not clouds:
        print("[ERROR] Pools Fog/Cloud non trouvés (vérifie les noms des serveurs).")
        return

    for f in fogs:
        if not hasattr(f, "base_cpu"):
            f.base_cpu = f.cpu

    # ---- 1) Fin des services ----
    remaining = []
    for s in ACTIVE_SERVICES:
        s.duration -= 1
        if s.duration > 0:
            remaining.append(s)
    ACTIVE_SERVICES = remaining

    # ---- 2) Injection + Scheduling ----
    tasks_now = WORKLOAD_IDX.get(CURRENT_T, [])
    total_incoming_demand = sum(t["cpu_demand"] for t in tasks_now)

    total_active_cpu_fog_before = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
    total_fog_capacity_before = sum(int(f.cpu) for f in fogs)
    pressure_global_before = float(total_active_cpu_fog_before) / max(1.0, float(total_fog_capacity_before))

    tasks_placed_fog_this_tick = 0
    tasks_placed_cloud_this_tick = 0
    dqn_fallback_used_tick = 0

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

        # pression pool (pour décision globale)
        total_active_cpu_fog = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
        total_fog_capacity = sum(int(f.cpu) for f in fogs)
        pressure_global = float(total_active_cpu_fog) / max(1.0, float(total_fog_capacity))

        # ✅ Fog choisi AVANT le DQN, pour passer fog_cpu identique training
        fog_choice, fog_choice_p = pick_best_fog(fogs, ACTIVE_SERVICES)

        decision, used_fallback_dqn = MODULE3.decide(
            task_cpu=int(task["cpu_demand"]),
            task_ram=int(task["ram_demand"]),
            pressure=float(pressure_global),
            fog_cpu=int(fog_choice.cpu),  # ✅ identique training: fog_cpu d'un fog (pas pool)
            offload_ratio=float(PLAN.get("offload_ratio", 0.0)),
            t=int(CURRENT_T),
        )
        if used_fallback_dqn:
            dqn_fallback_used_tick += 1

        service.placed_on = decision

        if decision == "Fog":
            service.placed_on_server = fog_choice
            tasks_placed_fog_this_tick += 1
            service.provision(fog_choice)
        else:
            cloud_node = clouds[CLOUD_RR % len(clouds)]
            CLOUD_RR += 1
            service.placed_on_server = cloud_node
            tasks_placed_cloud_this_tick += 1
            service.provision(cloud_node)

        ACTIVE_SERVICES.append(service)

    # ---- 3) Monitoring global ----
    total_active_cpu_fog = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
    total_fog_capacity = sum(int(f.cpu) for f in fogs)
    pressure_global_real = float(total_active_cpu_fog) / max(1.0, float(total_fog_capacity))

    PRESSURE_HISTORY.append(min(max(pressure_global_real, 0.0), 3.0))

    fog_pressures = [pressure_server(ACTIVE_SERVICES, f) for f in fogs]
    worst_fog_p = max(fog_pressures) if fog_pressures else 0.0

    print(
        f"[t={CURRENT_T:03d}] pool_fog: active_cpu={total_active_cpu_fog:5d} "
        f"cap={total_fog_capacity:5d} pressure={pressure_global_real:.2f} worst_fog={worst_fog_p:.2f} "
        f"| placed: fog={tasks_placed_fog_this_tick:3d} cloud={tasks_placed_cloud_this_tick:3d}"
    )

    pred_h5 = PREDICTIONS.get(5, {})
    SIMULATION_METRICS.append({
        "t": int(CURRENT_T),
        "pool_active_cpu_fog": int(total_active_cpu_fog),
        "pool_fog_capacity": int(total_fog_capacity),
        "pool_pressure": float(pressure_global_real),
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
    })

    # ---- 4) MAPE toutes W ticks ----
    if CURRENT_T > 0 and (CURRENT_T % W_WINDOW == 0):
        PREDICTIONS = MODULE1.predict(PRESSURE_HISTORY)
        current_p_clip = PRESSURE_HISTORY[-1] if PRESSURE_HISTORY else 0.0

        PLAN = MODULE2.plan(
            predictions=PREDICTIONS,
            total_fog_capacity=float(total_fog_capacity),
            total_incoming_demand=float(total_incoming_demand),
            current_pressure=float(current_p_clip),
            current_t=int(CURRENT_T),
            W_window=int(W_WINDOW),
        )

        if PLAN["scale_decision"] in ("up", "down"):
            if PLAN["scale_decision"] == "up":
                target_fog, p = pick_most_loaded_fog(fogs, ACTIVE_SERVICES)
                step = max(20, int(target_fog.cpu * 0.25))
                max_cap = int(getattr(target_fog, "base_cpu", target_fog.cpu) * MODULE2.max_scale_mult)
                old = int(target_fog.cpu)
                target_fog.cpu = int(min(max_cap, target_fog.cpu + step))
                print(f"[t={CURRENT_T:03d}] Scaling UP on {target_fog.name}: {old} -> {target_fog.cpu} (p={p:.2f})")
            else:
                target_fog, p = pick_least_loaded_fog(fogs, ACTIVE_SERVICES)
                step = max(20, int(target_fog.cpu * 0.20))
                min_cap = max(int(MODULE2.min_fog_cpu), int(getattr(target_fog, "base_cpu", target_fog.cpu) * 0.4))
                old = int(target_fog.cpu)
                target_fog.cpu = int(max(min_cap, target_fog.cpu - step))
                print(f"[t={CURRENT_T:03d}] Scaling DOWN on {target_fog.name}: {old} -> {target_fog.cpu} (p={p:.2f})")

        print(f"\n[t={CURRENT_T:03d}] === Cycle MAPE (pool) ===")
        print("  Predictions (pressure + incertitude):")
        for h in sorted(PREDICTIONS.keys()):
            p = PREDICTIONS[h]["prediction"]
            u = PREDICTIONS[h]["uncertainty"]
            fb = PREDICTIONS[h].get("used_fallback", False)
            print(f"    H={h:>2}: pred={p:.2f}  unc={u:.2f}  robust={p+u:.2f}  fallback={fb}")
        print("  Plan:")
        print(f"    robust_pred={PLAN['robust_pred']:.2f}")
        print(f"    pred_active_cpu={PLAN['pred_active_cpu']:.1f} | ema_active_cpu={PLAN['ema_active_cpu']:.1f}")
        print(f"    scale_decision={PLAN['scale_decision']}  reason={PLAN['scale_reason']}")
        print(f"    offload_ratio={PLAN['offload_ratio']:.2f} reason={PLAN['offload_reason']}\n")

    CURRENT_T += 1


def main():
    global WORKLOAD_IDX, MODULE1, MODULE2, MODULE3, W_WINDOW, CURRENT_T, ACTIVE_SERVICES, PRESSURE_HISTORY, PLAN, SIMULATION_METRICS, CLOUD_RR

    if not EDGE_SIM_AVAILABLE:
        print("❌ edge-sim-py n'est pas installé. Simulation impossible.")
        print("Installation: pip install edge-sim-py")
        return

    CURRENT_T = 0
    CLOUD_RR = 0
    ACTIVE_SERVICES = []
    PRESSURE_HISTORY = deque(maxlen=200)
    SIMULATION_METRICS = []
    PLAN = {"scale_decision": "none", "offload_ratio": 0.0}

    ap = argparse.ArgumentParser(description="Simulateur multi-Fog / multi-Cloud (proactif)")
    ap.add_argument("--workload", default=DEFAULT_WORKLOAD)
    ap.add_argument("--lstm_model", default=DEFAULT_LSTM)
    ap.add_argument("--dqn_model", default=DEFAULT_DQN)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--target_util", type=float, default=0.70)
    ap.add_argument("--min_fog_cpu", type=int, default=30)
    ap.add_argument("--output_csv", default=DEFAULT_RESULTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fog_scale", type=float, default=1.0, help="multiplie la CPU de chaque fog (ex: 1.0, 1.5, 2.0)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    W_WINDOW = int(args.W)

    if not os.path.exists(args.workload):
        raise FileNotFoundError(f"Workload introuvable: {args.workload}")

    WORKLOAD_IDX = load_workload_indexed(args.workload)
    nb_tasks = sum(len(v) for v in WORKLOAD_IDX.values())
    print(f"[OK] Workload chargé: {args.workload} ({nb_tasks} tâches)")

    MODULE1 = Module1_LSTMPredictor(model_path=args.lstm_model, device="cpu")
    MODULE2 = Module2_ProactivePlanner(
        target_util=args.target_util,
        min_fog_cpu=args.min_fog_cpu,
        ema_alpha=0.25,
        cooldown_windows=2,
        max_scale_mult=4.0,
    )
    MODULE3 = Module3_Scheduler(dqn_path=args.dqn_model, cpu_threshold_cloud=300, warmup_ticks=15)

    simulator = Simulator(tick_duration=1, tick_unit="seconds")

    # -----------------------------
    # Création MULTI-FOG + MULTI-CLOUD (France/Europe)
    # -----------------------------
    # 5 Fog nodes (f0..f4) = régionaux, ressources moyennes
    fog_specs = [
        ("Fog-Paris",     [48.8566,  2.3522], 120),
        ("Fog-Lille",     [50.6292,  3.0573],  90),
        ("Fog-Lyon",      [45.7640,  4.8357], 110),
        ("Fog-Toulouse",  [43.6047,  1.4442], 100),
        ("Fog-Bordeaux",  [44.8378, -0.5792],  95),
    ]
    for name, coord, cpu in fog_specs:
        cpu_scaled = int(max(10, cpu * float(args.fog_scale)))
        fog = EdgeServer(cpu=cpu_scaled, memory=8192, disk=20000)
        fog.name = name
        fog.coordinates = coord
        fog.base_cpu = cpu_scaled

    # 2 Cloud nodes (c0,c1) = très puissants, distants
    cloud_specs = [
        ("Cloud-FR", [48.8566, 2.3522], 1500),
        ("Cloud-BE", [50.4738, 3.8038], 1500),
    ]
    for name, coord, cpu in cloud_specs:
        cloud = EdgeServer(cpu=cpu, memory=200000, disk=200000)
        cloud.name = name
        cloud.coordinates = coord

    simulator.stopping_criterion = lambda sim: (globals()["CURRENT_T"] >= args.ticks)

    print(f"\n{'='*80}")
    print("DÉMARRAGE SIMULATION MULTI-FOG / MULTI-CLOUD (PROACTIF + STABLE)")
    print(f"{'='*80}")
    fogs = [s for s in EdgeServer.all() if "fog" in s.name.lower()]
    clouds = [s for s in EdgeServer.all() if "cloud" in s.name.lower()]
    print(f"Durée: {args.ticks} ticks | Fenêtre MAPE: {W_WINDOW}")
    print(f"Fog nodes: {len(fogs)} | Cloud nodes: {len(clouds)}")
    print("Fog capacities:", ", ".join([f"{f.name}={int(f.cpu)}" for f in fogs]))
    print("Cloud capacities:", ", ".join([f"{c.name}={int(c.cpu)}" for c in clouds]))
    print(f"Target utilization: {args.target_util}")
    print(f"{'='*80}\n")

    simulator.resource_management_algorithm = proactive_placement_algorithm
    simulator.resource_management_algorithm_parameters = {"simulator": simulator}

    try:
        simulator.run_model()
        print(f"{'='*80}")
        print("SIMULATION TERMINÉE AVEC SUCCÈS")
        print(f"{'='*80}")
    except Exception as e:
        print(f"\n❌ ERREUR pendant la simulation: {e}")
        import traceback
        traceback.print_exc()

    if SIMULATION_METRICS:
        avg_pressure = float(np.mean([m["pool_pressure"] for m in SIMULATION_METRICS]))
        max_pressure = float(np.max([m["pool_pressure"] for m in SIMULATION_METRICS]))
        avg_offload = float(np.mean([m["offload_ratio"] for m in SIMULATION_METRICS]))

        fog_total = int(sum(m["tasks_placed_fog"] for m in SIMULATION_METRICS))
        cloud_total = int(sum(m["tasks_placed_cloud"] for m in SIMULATION_METRICS))

        lstm_fallback_pct = float(np.mean([m.get("lstm_fallback_used", 1) for m in SIMULATION_METRICS]))
        dqn_fallback_tasks = int(sum(m.get("dqn_fallback_used_tasks", 0) for m in SIMULATION_METRICS))

        print(f"\n📊 STATISTIQUES FINALES:")
        print(f"  Pool pressure moyenne: {avg_pressure:.2f}")
        print(f"  Pool pressure max:     {max_pressure:.2f}")
        print(f"  Offload moyen:         {avg_offload:.2%}")
        print(f"  Tâches Fog totales:    {fog_total}")
        print(f"  Tâches Cloud totales:  {cloud_total}")
        print(f"  LSTM fallback (ticks): {lstm_fallback_pct:.1%}")
        print(f"  DQN fallback (tâches): {dqn_fallback_tasks}")

    # Sauvegarde CSV
    if args.output_csv and SIMULATION_METRICS:
        output_path = args.output_csv
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SIMULATION_METRICS[0].keys())
            writer.writeheader()
            writer.writerows(SIMULATION_METRICS)
        print(f"\n💾 Métriques de simulation sauvegardées dans: {output_path}")


if __name__ == "__main__":
    main()
