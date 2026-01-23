# test.py - SIMULATEUR INTÉGRÉ FOG-CLOUD (CORRIGÉ + STABLE + COMPAT LSTM/DQN)
# - Charge correctement EnhancedLSTM (train_lstm.py) et DQN (train_dqn.py)
# - Planner plus stable: EMA + cooldown + hysteresis renforcée
# - Logs + métriques: indique si fallback LSTM/DQN a été utilisé

import argparse
import csv
import os
import random
import warnings
from collections import deque, defaultdict
from typing import Dict, List, Any, DefaultDict, Optional

import torch
import torch.nn as nn
import numpy as np

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
            # Stub: ne simule pas réellement
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
            self.placed_on = None
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
    def __init__(self, input_dim=5, output_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# Module 1: LSTM Predictor (robuste + compat)
# - checkpoint attendu (train_lstm corrigé): state_dict, seq_len, max_util, hidden_dim, num_layers, dropout, arch="EnhancedLSTM"
# - fallback: persistence améliorée si chargement échoue
# =========================================================
class Module1_LSTMPredictor:
    def __init__(self, model_path: str, horizons=None, device="cpu"):
        self.horizons = horizons or [5, 15, 30, 60]
        self.model_path = model_path
        self.device = device

        # defaults (seront remplacés si checkpoint présent)
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

                # garde-fous
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
        # On travaille sur une pression "capée" pour éviter explosion numérique
        hist_all = list(pressure_history)
        if len(hist_all) < 3:
            return {h: {"prediction": 0.10, "uncertainty": 0.05, "used_fallback": True} for h in self.horizons}

        last_p = float(hist_all[-1])
        last5 = hist_all[-5:] if len(hist_all) >= 5 else hist_all
        mean_last_5 = float(sum(last5) / len(last5))

        # fallback persistence améliorée
        if not self.model_loaded or self.model is None:
            # décroissance si faible charge
            if last_p < 0.2 and mean_last_5 < 0.3:
                pred = max(0.0, last_p * 0.7)
            else:
                pred = last_p

            unc = max(0.05, 0.08 + 0.08 * min(pred, 2.0))
            return {h: {"prediction": float(pred), "uncertainty": float(min(unc + 0.01 * h, 0.5)), "used_fallback": True}
                    for h in self.horizons}

        # préparer séquence
        seq_len = self.seq_len
        hist = hist_all[-seq_len:]
        if len(hist) < seq_len:
            hist = [hist[-1]] * (seq_len - len(hist)) + hist

        # clip réaliste pour l'entrée
        hist_clip = [max(0.0, min(x, 3.0)) for x in hist]
        hist_norm = [max(0.0, min(x / self.max_util, 3.0)) for x in hist_clip]

        x = torch.tensor(hist_norm, dtype=torch.float32).view(1, seq_len, 1).to(self.device)

        with torch.no_grad():
            y_norm = float(self.model(x).item())

        pred_p = max(0.0, y_norm * self.max_util)

        # garde-fous anti “explosion”
        if mean_last_5 < 0.3:
            pred_p = min(pred_p, mean_last_5 * 2.0 + 0.3)
        if last_p < 0.1 and pred_p > 0.5:
            pred_p *= 0.3
        pred_p = min(pred_p, 3.0)

        # blend si saut trop brutal
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
# Module 2: Proactive Planner (STABLE)
# - EMA sur predicted_active_cpu
# - cooldown pour éviter scaling trop fréquent
# - hysteresis + scale-down plus agressif si pression basse durable
# =========================================================
class Module2_ProactivePlanner:
    def __init__(
        self,
        target_util=0.70,
        min_fog_cpu=30,
        ema_alpha=0.25,
        cooldown_windows=2,
        max_scale_mult=4.0,
    ):
        self.target_util = float(target_util)
        self.min_fog_cpu = int(min_fog_cpu)
        self.ema_alpha = float(ema_alpha)
        self.cooldown_windows = int(cooldown_windows)
        self.max_scale_mult = float(max_scale_mult)

        self.last_scale_t = -10**9
        self.ema_predicted_active_cpu: Optional[float] = None

        print(
            f"[Module2] Planner stable: target_util={self.target_util}, min_fog_cpu={self.min_fog_cpu}, "
            f"EMAα={self.ema_alpha}, cooldown_windows={self.cooldown_windows}, max_scale_mult={self.max_scale_mult}"
        )

    def plan(
        self,
        predictions: Dict[int, Dict[str, float]],
        fog_node: EdgeServer,
        total_incoming_demand: float,
        current_pressure: float,
        current_t: int,
        W_window: int,
    ) -> Dict[str, Any]:
        pred_h5 = predictions.get(5, {"prediction": current_pressure, "uncertainty": 0.10})
        pred_p = float(pred_h5.get("prediction", current_pressure))
        unc = float(pred_h5.get("uncertainty", 0.10))

        # robust_p capé
        robust_p = min(max(pred_p + unc, 0.0), 3.0)

        base_cpu = getattr(fog_node, "base_cpu", fog_node.cpu)
        current_cpu = float(fog_node.cpu)

        max_cap = max(float(base_cpu) * self.max_scale_mult, current_cpu)
        min_cap = max(self.min_fog_cpu, int(float(base_cpu) * 0.4))

        predicted_active_cpu = (robust_p * current_cpu) + float(total_incoming_demand)

        # EMA pour éviter les oscillations
        if self.ema_predicted_active_cpu is None:
            self.ema_predicted_active_cpu = predicted_active_cpu
        else:
            self.ema_predicted_active_cpu = (
                self.ema_alpha * predicted_active_cpu + (1.0 - self.ema_alpha) * self.ema_predicted_active_cpu
            )

        ema_cpu = float(self.ema_predicted_active_cpu)

        # CPU requis pour target_util
        required_cpu = ema_cpu / max(self.target_util, 0.3)
        required_cpu = min(required_cpu, max_cap)

        # cooldown en "fenêtres MAPE"
        cooldown_ticks = max(1, W_window) * self.cooldown_windows
        in_cooldown = (current_t - self.last_scale_t) < cooldown_ticks

        decision = "none"
        reason = "within band"
        next_cpu = int(current_cpu)

        # hysteresis plus forte
        up_th = 1.35
        down_th = 0.55

        if not in_cooldown:
            # Scale up si vraiment nécessaire
            if required_cpu > current_cpu * up_th and current_cpu < max_cap:
                decision = "up"
                reason = f"required_cpu({required_cpu:.1f}) > {up_th}*current_cpu({current_cpu:.1f})"
            # Scale down si clairement sur-provisionné
            elif required_cpu < current_cpu * down_th and current_cpu > min_cap:
                decision = "down"
                reason = f"required_cpu({required_cpu:.1f}) < {down_th}*current_cpu({current_cpu:.1f})"
            # si pression très faible: down plus probable
            elif current_pressure < 0.15 and current_cpu > min_cap:
                decision = "down"
                reason = f"very low pressure({current_pressure:.2f})"

        # pas de petits steps: steps adaptatifs
        step_up = max(20, int(current_cpu * 0.25))
        step_down = max(20, int(current_cpu * 0.20))

        if decision == "up":
            next_cpu = int(min(max_cap, current_cpu + step_up))
            self.last_scale_t = current_t
        elif decision == "down":
            # down plus agressif si très faible charge
            if current_pressure < 0.10:
                step_down = max(step_down, int(current_cpu * 0.30))
            next_cpu = int(max(min_cap, current_cpu - step_down))
            self.last_scale_t = current_t

        # Offload ratio (raisonnable + safety)
        offload_ratio = 0.0
        offload_reason = "no offload"

        demand_vs_capacity = ema_cpu / max(next_cpu, 1)
        if demand_vs_capacity > 1.0:
            excess_ratio = min(1.0, (demand_vs_capacity - 1.0) * 0.6)
            offload_ratio = max(offload_ratio, min(0.85, excess_ratio))
            offload_reason = f"demand/capacity={demand_vs_capacity:.2f}"

        # Safety net si surcharge
        if current_pressure > 1.0:
            safety_offload = 0.35 + 0.18 * (current_pressure - 1.0)
            safety_offload = min(0.85, safety_offload)
            offload_ratio = max(offload_ratio, safety_offload)
            offload_reason += f" | SAFETY pressure={current_pressure:.2f}"

        # si charge vraiment basse
        if ema_cpu < float(base_cpu) * 0.10:
            offload_ratio = 0.0
            offload_reason = "very low demand"

        return {
            "robust_pred": float(robust_p),
            "pred_active_cpu": float(predicted_active_cpu),
            "ema_active_cpu": float(ema_cpu),
            "scale_decision": decision,
            "scale_reason": reason + (f" | cooldown({cooldown_ticks})" if in_cooldown else ""),
            "next_cpu": int(next_cpu),
            "offload_ratio": float(offload_ratio),
            "offload_reason": offload_reason,
        }


# =========================================================
# Module 3: Scheduler (DQN + baseline)
# - Charge DQN matching train_dqn.py
# - Normalisation identique au training
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
        self.dqn = None

        if dqn_path and os.path.exists(dqn_path):
            try:
                ckpt = torch.load(dqn_path, map_location="cpu")
                self.hidden_dim = int(ckpt.get("hidden_dim", 128))
                self.dqn = DQN(input_dim=5, output_dim=2, hidden_dim=self.hidden_dim)
                self.dqn.load_state_dict(ckpt["state_dict"], strict=True)
                self.dqn.eval()
                self.use_dqn = True
                print(f"[Module3] ✅ DQN chargé: {dqn_path} (hidden_dim={self.hidden_dim})")
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

    def decide(self, task_cpu: int, task_ram: int, pressure: float, fog_cpu: int, offload_ratio: float, t: int) -> (str, bool):
        # retourne (decision, used_fallback_dqn)
        if t <= self.warmup_ticks or not self.use_dqn or self.dqn is None or self.dqn_fallback_count >= self.max_fallback:
            return self.baseline(task_cpu, offload_ratio, pressure), True

        # Normalisation IDENTIQUE AU TRAINING DQN
        cpu_norm = min(float(task_cpu) / 500.0, 2.0)
        ram_norm = min(float(task_ram) / 4096.0, 2.0)
        pressure_clip = min(max(float(pressure), 0.0), 3.0)
        fog_cpu_norm = float(fog_cpu) / 200.0  # 100->0.5

        state = torch.tensor(
            [cpu_norm, ram_norm, pressure_clip, fog_cpu_norm, float(offload_ratio)],
            dtype=torch.float32
        ).unsqueeze(0)

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
            task = {
                "task_id": int(row["task_id"]),
                "timestamp": int(row["timestamp"]),
                "service_type": row.get("service_type", "NA"),
                "cpu_demand": int(row["cpu_demand"]),
                "ram_demand": int(row["ram_demand"]),
                "duration": int(row["duration"]),
            }
            idx[task["timestamp"]].append(task)
    return idx


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


def proactive_placement_algorithm(parameters):
    global CURRENT_T, ACTIVE_SERVICES, PRESSURE_HISTORY, PREDICTIONS, PLAN, SIMULATION_METRICS
    simulator = parameters["simulator"]

    # Récupération des nœuds
    fog_node = None
    cloud_node = None
    for s in EdgeServer.all():
        if "fog" in s.name.lower():
            fog_node = s
        if "cloud" in s.name.lower():
            cloud_node = s

    if fog_node is None or cloud_node is None:
        print("[ERROR] Nœuds Fog ou Cloud non trouvés!")
        return

    if not hasattr(fog_node, "base_cpu"):
        fog_node.base_cpu = fog_node.cpu

    # ---- 0) Emergency scaling si pression critique ----
    active_cpu_fog_check = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
    pressure_check = active_cpu_fog_check / fog_node.cpu if fog_node.cpu > 0 else 0.0

    if pressure_check > 1.25:
        # autoriser plus haut que 3x pour éviter explosion > 5
        max_cap_emergency = int(getattr(fog_node, "base_cpu", 100) * 5.0)
        if fog_node.cpu < max_cap_emergency:
            old_cpu = fog_node.cpu
            fog_node.cpu = min(max_cap_emergency, fog_node.cpu + max(50, int(fog_node.cpu * 0.25)))
            print(f"[t={CURRENT_T:02d}] 🚨 EMERGENCY UP: {old_cpu} -> {fog_node.cpu} (Pressure={pressure_check:.2f})")
            PLAN["scale_decision"] = "emergency_up"

    # ---- 1) Fin des services (durées) ----
    remaining = []
    for s in ACTIVE_SERVICES:
        s.duration -= 1
        if s.duration > 0:
            remaining.append(s)
    ACTIVE_SERVICES = remaining

    # ---- 2) Injection + Scheduling ----
    tasks_now = WORKLOAD_IDX.get(CURRENT_T, [])
    total_incoming_demand = sum(t["cpu_demand"] for t in tasks_now)

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

        active_cpu_fog_before = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
        pressure_before = active_cpu_fog_before / fog_node.cpu if fog_node.cpu > 0 else 0.0

        decision, used_fallback_dqn = MODULE3.decide(
            task_cpu=task["cpu_demand"],
            task_ram=task["ram_demand"],
            pressure=pressure_before,
            fog_cpu=fog_node.cpu,
            offload_ratio=PLAN.get("offload_ratio", 0.0),
            t=CURRENT_T,
        )
        if used_fallback_dqn:
            dqn_fallback_used_tick += 1

        service.placed_on = decision
        if decision == "Fog":
            tasks_placed_fog_this_tick += 1
        else:
            tasks_placed_cloud_this_tick += 1

        service.duration = task["duration"]
        ACTIVE_SERVICES.append(service)
        service.provision(fog_node if decision == "Fog" else cloud_node)

    # ---- 3) Monitoring ----
    active_cpu_fog = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
    pressure_real = active_cpu_fog / fog_node.cpu if fog_node.cpu > 0 else 0.0

    # pour l’historique LSTM, on clip à 3.0 (robuste)
    PRESSURE_HISTORY.append(min(max(pressure_real, 0.0), 3.0))

    print(f"[t={CURRENT_T:02d}] active_cpu_fog={active_cpu_fog:4d} fog_cpu={fog_node.cpu:4d} pressure={pressure_real:.2f}")

    pred_h5 = PREDICTIONS.get(5, {})
    SIMULATION_METRICS.append({
        "t": CURRENT_T,
        "active_cpu_fog": int(active_cpu_fog),
        "fog_capacity": int(fog_node.cpu),
        "pressure": float(pressure_real),  # vraie pression (peut dépasser 3)
        "predicted_pressure": float(pred_h5.get("prediction", pressure_real)),
        "prediction_uncertainty": float(pred_h5.get("uncertainty", 0.10)),
        "lstm_fallback_used": int(pred_h5.get("used_fallback", True)) if pred_h5 else 1,
        "dqn_fallback_used_tasks": int(dqn_fallback_used_tick),
        "scale_decision": PLAN.get("scale_decision", "none"),
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
            fog_node=fog_node,
            total_incoming_demand=total_incoming_demand,
            current_pressure=float(current_p_clip),
            current_t=CURRENT_T,
            W_window=W_WINDOW,
        )

        if PLAN["scale_decision"] in ("up", "down"):
            old_cpu = fog_node.cpu
            fog_node.cpu = int(PLAN["next_cpu"])
            print(f"[t={CURRENT_T}] Scaling: {old_cpu} -> {fog_node.cpu} ({PLAN['scale_decision']})")

        print(f"\n[t={CURRENT_T}] === Cycle MAPE ===")
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
    global WORKLOAD_IDX, MODULE1, MODULE2, MODULE3, W_WINDOW, CURRENT_T, ACTIVE_SERVICES, PRESSURE_HISTORY, PLAN, SIMULATION_METRICS

    if not EDGE_SIM_AVAILABLE:
        print("❌ edge-sim-py n'est pas installé. Simulation impossible.")
        print("Installation: pip install edge-sim-py")
        return

    CURRENT_T = 0
    ACTIVE_SERVICES = []
    PRESSURE_HISTORY = deque(maxlen=200)
    SIMULATION_METRICS = []
    PLAN = {"scale_decision": "none", "offload_ratio": 0.0}

    ap = argparse.ArgumentParser(description="Simulateur Fog-Cloud avec approche proactive (corrigé)")
    ap.add_argument("--workload", default=os.path.join("data", "workload.csv"))
    ap.add_argument("--lstm_model", default=os.path.join("models", "lstm_util.pth"))
    ap.add_argument("--dqn_model", default=os.path.join("models", "dqn_fog_cloud.pth"))
    ap.add_argument("--fog_cpu", type=int, default=100)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--target_util", type=float, default=0.70)
    ap.add_argument("--min_fog_cpu", type=int, default=30)
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--seed", type=int, default=42)
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

    cloud = EdgeServer(cpu=1000, memory=100000, disk=100000)
    cloud.name = "Cloud-AWS"
    cloud.coordinates = [10, 10]

    fog = EdgeServer(cpu=args.fog_cpu, memory=4096, disk=10000)
    fog.name = "Fog-1"
    fog.coordinates = [5, 5]
    fog.base_cpu = args.fog_cpu

    simulator.stopping_criterion = lambda sim: (globals()["CURRENT_T"] >= args.ticks)

    print(f"\n{'='*70}")
    print("DÉMARRAGE SIMULATION FOG-CLOUD PROACTIF (CORRIGÉ + STABLE)")
    print(f"{'='*70}")
    print(f"Durée: {args.ticks} ticks | Fenêtre MAPE: {W_WINDOW}")
    print(f"Fog initial: {args.fog_cpu} CPU | Cloud: 1000 CPU")
    print(f"Target utilization: {args.target_util}")
    print(f"{'='*70}\n")

    simulator.resource_management_algorithm = proactive_placement_algorithm
    simulator.resource_management_algorithm_parameters = {"simulator": simulator}

    try:
        simulator.run_model()
        print(f"{'='*70}")
        print("SIMULATION TERMINÉE AVEC SUCCÈS")
        print(f"{'='*70}")
    except Exception as e:
        print(f"\n❌ ERREUR pendant la simulation: {e}")
        import traceback
        traceback.print_exc()

    # Stats finales (console)
    if SIMULATION_METRICS:
        avg_pressure = sum(m["pressure"] for m in SIMULATION_METRICS) / len(SIMULATION_METRICS)
        max_pressure = max(m["pressure"] for m in SIMULATION_METRICS)
        avg_offload = sum(m["offload_ratio"] for m in SIMULATION_METRICS) / len(SIMULATION_METRICS)

        fog_total = sum(m["tasks_placed_fog"] for m in SIMULATION_METRICS)
        cloud_total = sum(m["tasks_placed_cloud"] for m in SIMULATION_METRICS)

        scale_decisions = [m["scale_decision"] for m in SIMULATION_METRICS]
        ups = scale_decisions.count("up") + scale_decisions.count("emergency_up")
        downs = scale_decisions.count("down")

        lstm_fallback_pct = sum(m.get("lstm_fallback_used", 1) for m in SIMULATION_METRICS) / len(SIMULATION_METRICS)
        dqn_fallback_tasks = sum(m.get("dqn_fallback_used_tasks", 0) for m in SIMULATION_METRICS)

        print(f"\n📊 STATISTIQUES FINALES:")
        print(f"  Pression moyenne: {avg_pressure:.2f}")
        print(f"  Pression max: {max_pressure:.2f}")
        print(f"  Offload moyen: {avg_offload:.2%}")
        print(f"  Tâches Fog totales: {fog_total}")
        print(f"  Tâches Cloud totales: {cloud_total}")
        print(f"  Scaling UP: {ups} fois | Scaling DOWN: {downs} fois")
        print(f"  LSTM fallback (ticks): {lstm_fallback_pct:.1%}")
        print(f"  DQN fallback (tâches): {dqn_fallback_tasks}")

    # Sauvegarde CSV
    if args.output_csv:
        output_path = args.output_csv
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if SIMULATION_METRICS:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SIMULATION_METRICS[0].keys())
                writer.writeheader()
                writer.writerows(SIMULATION_METRICS)
            print(f"\n💾 Métriques de simulation sauvegardées dans: {output_path}")


if __name__ == "__main__":
    main()
