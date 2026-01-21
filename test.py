# test.py
import argparse
import csv
import os
import random
from collections import deque, defaultdict
from typing import Dict, List, Any, DefaultDict

import torch
import torch.nn as nn

try:
    from edge_sim_py import Simulator, EdgeServer, Application, Service, ContainerImage
except ImportError:
    raise ImportError("edge-sim-py non installé. Fais: pip install edge-sim-py")


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
# Module 1: LSTM (Inference) -> sur PRESSURE (peut dépasser 1)
# checkpoint doit contenir: state_dict, seq_len, max_util
# =========================================================
class LSTMUtil(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class Module1_LSTMPredictor:
    def __init__(self, model_path: str, horizons=None, device="cpu"):
        self.horizons = horizons or [5, 15, 30, 60]
        self.model_path = model_path
        self.device = device

        self.model = LSTMUtil().to(self.device)
        self.seq_len = 20
        self.max_util = 1.0
        self.model_loaded = False

        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(ckpt["state_dict"])
            self.seq_len = int(ckpt.get("seq_len", self.seq_len))
            self.max_util = float(ckpt.get("max_util", 1.0))
            if self.max_util <= 0:
                self.max_util = 1.0
            self.model_loaded = True
            print(f"[Module1] LSTM chargé: {model_path} (seq_len={self.seq_len}, max_util={self.max_util:.3f})")
        else:
            print(f"[Module1] ⚠️ modèle LSTM introuvable: {model_path} -> fallback persistence")

        self.model.eval()

    def _rolling_std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.05
        m = sum(values) / len(values)
        v = sum((x - m) ** 2 for x in values) / len(values)
        return max(0.0, v ** 0.5)
    
    def predict(self, pressure_history: deque) -> Dict[int, Dict[str, float]]:
        if len(pressure_history) < 3:
            return {h: {"prediction": 0.10, "uncertainty": 0.05} for h in self.horizons}

        last_p = float(pressure_history[-1])

        # fallback: persistence
        if not self.model_loaded:
            unc = max(0.05, 0.10 * abs(last_p))
            return {h: {"prediction": last_p, "uncertainty": unc} for h in self.horizons}

        seq_len = self.seq_len
        hist = list(pressure_history)[-seq_len:]
        if len(hist) < seq_len:
            hist = [hist[-1]] * (seq_len - len(hist)) + hist

        # normalisation comme au training
        hist_norm = [max(0.0, min(x / self.max_util, 5.0)) for x in hist]
        x = torch.tensor(hist_norm, dtype=torch.float32).view(1, seq_len, 1).to(self.device)

        with torch.no_grad():
            y_norm = float(self.model(x).item())

        pred_p = y_norm * self.max_util

        # clamp cohérent (pas 20 fixe)
        pred_p = max(0.0, min(pred_p, 2.0 * self.max_util))

        # ======== GARDE-FOUS ANTI HALLUCINATION ========
        # Si la charge actuelle est faible, on empêche une explosion irréaliste
        if last_p < 0.5:
            pred_p = min(pred_p, last_p * 2.0 + 0.2)

        # Si le modèle part trop loin vs dernier point -> on blend avec persistence
        if pred_p > 3.0 * max(last_p, 0.1):
            pred_p = 0.8 * last_p + 0.2 * pred_p

        std10 = self._rolling_std(list(pressure_history)[-10:])

        preds = {}
        for h in self.horizons:
            unc = max(0.05, 0.10 * pred_p + 0.20 * std10 + 0.002 * h)
            preds[h] = {"prediction": float(pred_p), "uncertainty": float(unc)}
        return preds

    
    


# =========================================================
# Module 2: Planner (scaling + offload) -> sur PRESSURE
# =========================================================
class Module2_ProactivePlanner:
    def __init__(self, target_util=0.70):
        self.target_util = target_util
        print("[Module2] Planner initialisé (scaling + offload).")

    def plan(self, predictions: Dict[int, Dict[str, float]], fog_node: EdgeServer, total_incoming_demand: float, current_pressure: float = 0.0) -> Dict[str, Any]:
        # On utilise la prédiction pour anticiper la charge FOG, mais on ajoute la demande entrante
        # pour ne pas être trompé par un offload massif.
        pred_h5 = predictions.get(5, {"prediction": 0.0, "uncertainty": 0.0})
        robust_p = pred_h5["prediction"] + pred_h5["uncertainty"]

        # Demande future estimée = charge résiduelle prédite + charge entrante totale
        predicted_active_cpu = (robust_p * fog_node.cpu) + total_incoming_demand

        base_cpu = getattr(fog_node, "base_cpu", fog_node.cpu)
        current_cpu = fog_node.cpu

        max_cap = base_cpu * 2
        min_cap = int(base_cpu * 0.5)

        # CPU requis pour viser target_util
        required_cpu = predicted_active_cpu / max(self.target_util, 1e-6)

        decision = "none" 
        reason = "within hysteresis band"
        if required_cpu > current_cpu * 1.1 and current_cpu < max_cap:
            decision = "up"
            reason = f"required_cpu({required_cpu:.1f}) > 1.1*current_cpu({current_cpu})"
        elif required_cpu < current_cpu * 0.9 and current_cpu > min_cap:
            decision = "down"
            reason = f"required_cpu({required_cpu:.1f}) < 0.9*current_cpu({current_cpu})"

        step = 50
        next_cpu = current_cpu
        if decision == "up":
            next_cpu = min(max_cap, current_cpu + step)
        elif decision == "down":
            next_cpu = max(min_cap, current_cpu - step)

        # Offload si la demande prédite dépasse la capacité prochaine
        offload_ratio = 0.0
        offload_reason = "no offload needed"
        if predicted_active_cpu > next_cpu:
            excess = predicted_active_cpu - next_cpu
            offload_ratio = max(0.0, min(1.0, excess / max(predicted_active_cpu, 1e-6)))
            offload_reason = f"demand({predicted_active_cpu:.1f}) > next_cpu({next_cpu})"

        # Force offload if current pressure is critical (Safety Net)
        if current_pressure > 1.0:
            min_offload = 0.3 + 0.2 * (current_pressure - 1.0) # ex: press=1.5 -> 0.3 + 0.1 = 0.4
            offload_ratio = max(offload_ratio, min(0.9, min_offload))
            offload_reason += f" | FORCED (pressure={current_pressure:.2f})"

        return {
            "robust_pred": robust_p,
            "scale_decision": decision,
            "scale_reason": reason,
            "next_cpu": next_cpu,
            "offload_ratio": offload_ratio,
            "offload_reason": offload_reason,
        }


# =========================================================
# Module 3: Scheduler Baseline + DQN (state compatible train_dqn corrigé)
# state = [cpu_norm, ram_norm, pressure_clip, fog_cpu_norm, offload_ratio]
# =========================================================
class DQN(nn.Module):
    def __init__(self, input_dim=5, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Module3_Scheduler:
    def __init__(self, dqn_path: str = None, cpu_threshold_cloud=300, warmup_ticks=15):
        self.cpu_threshold_cloud = cpu_threshold_cloud
        self.dqn_path = dqn_path
        self.use_dqn = False
        self.warmup_ticks = warmup_ticks

        self.dqn = DQN(input_dim=5, output_dim=2)
        if dqn_path and os.path.exists(dqn_path):
            ckpt = torch.load(dqn_path, map_location="cpu")
            self.dqn.load_state_dict(ckpt["state_dict"])
            self.dqn.eval()
            self.use_dqn = True
            print(f"[Module3] DQN Scheduler chargé: {dqn_path}")
        else:
            print("[Module3] Baseline Scheduler (DQN absent).")

    def baseline(self, task_cpu: int, offload_ratio: float) -> str:
        if random.random() < offload_ratio:
            return "Cloud"
        if task_cpu > self.cpu_threshold_cloud:
            return "Cloud"
        return "Fog"

    def decide(self, task_cpu: int, task_ram: int, pressure: float, fog_cpu: int, offload_ratio: float, t: int) -> str:
        if t <= self.warmup_ticks:
            return self.baseline(task_cpu, offload_ratio)

        if self.use_dqn:
            cpu_norm = min(task_cpu / 500.0, 5.0)
            ram_norm = min(task_ram / 4096.0, 5.0)
            pressure_clip = min(max(pressure, 0.0), 3.0)
            fog_cpu_norm = min(fog_cpu / 200.0, 2.0)

            state = torch.tensor(
                [cpu_norm, ram_norm, pressure_clip, fog_cpu_norm, float(offload_ratio)],
                dtype=torch.float32
            ).unsqueeze(0)  # (1,5)

            with torch.no_grad():
                a = int(torch.argmax(self.dqn(state), dim=1).item())
            return "Fog" if a == 0 else "Cloud"

        return self.baseline(task_cpu, offload_ratio)


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

MODULE1 = None
MODULE2 = None
MODULE3 = None

W_WINDOW = 10


def proactive_placement_algorithm(parameters):
    global CURRENT_T, ACTIVE_SERVICES, PRESSURE_HISTORY, PREDICTIONS, PLAN, SIMULATION_METRICS

    simulator = parameters["simulator"]
    fog_node = [s for s in EdgeServer.all() if "Fog" in s.name][0]
    cloud_node = [s for s in EdgeServer.all() if "Cloud" in s.name][0]

    if not hasattr(fog_node, "base_cpu"):
        fog_node.base_cpu = fog_node.cpu

    # ---- 0) Reactive Check (Emergency Scaling) ----
    # Si la pression explose entre deux cycles MAPE, on agit tout de suite
    active_cpu_fog_check = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
    pressure_check = active_cpu_fog_check / fog_node.cpu if fog_node.cpu > 0 else 0.0
    
    if pressure_check > 1.2: # Seuil critique d'urgence
        max_cap = getattr(fog_node, "base_cpu", 100) * 3 # Autoriser x3 en urgence
        if fog_node.cpu < max_cap:
            old_cpu = fog_node.cpu
            fog_node.cpu = min(max_cap, fog_node.cpu + 50)
            print(f"[t={CURRENT_T:02d}] 🚨 EMERGENCY SCALE UP: {old_cpu} -> {fog_node.cpu} (Pressure={pressure_check:.2f})")
            PLAN["scale_decision"] = "emergency_up"

    # ---- 1) Fin des services (durées)
    remaining = []
    for s in ACTIVE_SERVICES:
        s.duration -= 1
        if s.duration > 0:
            remaining.append(s)
    ACTIVE_SERVICES = remaining

    # ---- 2) Injection + Scheduling (IMPORTANT: avant le log)
    tasks_now = WORKLOAD_IDX.get(CURRENT_T, [])
    # Calcul de la demande totale entrante pour le planner
    total_incoming_demand = sum(t["cpu_demand"] for t in tasks_now)

    tasks_placed_fog_this_tick = 0
    tasks_placed_cloud_this_tick = 0
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

        # pressure courant (avant ajout) pour la décision
        active_cpu_fog_before = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
        pressure_before = active_cpu_fog_before / fog_node.cpu if fog_node.cpu > 0 else 0.0

        decision = MODULE3.decide(
            task_cpu=task["cpu_demand"],
            task_ram=task["ram_demand"],
            pressure=pressure_before,
            fog_cpu=fog_node.cpu,
            offload_ratio=PLAN.get("offload_ratio", 0.0),
            t=CURRENT_T
        )
        

        service.placed_on = decision
        if decision == "Fog":
            tasks_placed_fog_this_tick += 1
        else:
            tasks_placed_cloud_this_tick += 1
        service.duration = task["duration"]
        ACTIVE_SERVICES.append(service)

        service.provision(fog_node if decision == "Fog" else cloud_node)

    # ---- 3) Monitoring après injection (log réaliste)
    active_cpu_fog = sum(s.cpu_demand for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog")
    pressure = active_cpu_fog / fog_node.cpu if fog_node.cpu > 0 else 0.0
    PRESSURE_HISTORY.append(pressure)

    print(f"[t={CURRENT_T:02d}] active_cpu_fog={active_cpu_fog:4d} fog_cpu={fog_node.cpu:3d} pressure={pressure:.2f}")

    # ---- Store metrics for plotting ----
    pred_h5 = PREDICTIONS.get(5, {})
    SIMULATION_METRICS.append({
        "t": CURRENT_T,
        "active_cpu_fog": active_cpu_fog,
        "fog_capacity": fog_node.cpu,
        "pressure": pressure,
        "predicted_pressure": pred_h5.get("prediction", 0.0),
        "prediction_uncertainty": pred_h5.get("uncertainty", 0.0),
        "scale_decision": PLAN.get("scale_decision", "none"),
        "offload_ratio": PLAN.get("offload_ratio", 0.0),
        "tasks_on_fog": sum(1 for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Fog"),
        "tasks_on_cloud": sum(1 for s in ACTIVE_SERVICES if getattr(s, "placed_on", None) == "Cloud"),
        "tasks_placed_fog": tasks_placed_fog_this_tick,
        "tasks_placed_cloud": tasks_placed_cloud_this_tick,
    })

    # ---- 4) MAPE toutes W (agit sur les ticks suivants)
    if CURRENT_T > 0 and (CURRENT_T % W_WINDOW == 0):
        PREDICTIONS = MODULE1.predict(PRESSURE_HISTORY)
        current_p = PRESSURE_HISTORY[-1] if PRESSURE_HISTORY else 0.0
        PLAN = MODULE2.plan(PREDICTIONS, fog_node, total_incoming_demand=total_incoming_demand, current_pressure=current_p)

        if PLAN["scale_decision"] in ("up", "down"):
            fog_node.cpu = PLAN["next_cpu"]

        print(f"\n[t={CURRENT_T}] === Cycle MAPE ===")
        print("  Predictions (pressure + incertitude):")
        for h in sorted(PREDICTIONS.keys()):
            p = PREDICTIONS[h]["prediction"]
            u = PREDICTIONS[h]["uncertainty"]
            print(f"    H={h:>2}: pred={p:.2f}  unc={u:.2f}  robust={p+u:.2f}")
        print("  Plan:")
        print(f"    robust_pred={PLAN['robust_pred']:.2f}")
        print(f"    scale_decision={PLAN['scale_decision']}  reason={PLAN['scale_reason']}")
        print(f"    offload_ratio={PLAN['offload_ratio']:.2f} reason={PLAN['offload_reason']}\n")

    CURRENT_T += 1


def main():
    global WORKLOAD_IDX, MODULE1, MODULE2, MODULE3, W_WINDOW, CURRENT_T, ACTIVE_SERVICES, PRESSURE_HISTORY, PLAN, SIMULATION_METRICS

    # reset (utile si tu relances plusieurs fois dans le même process)
    CURRENT_T = 0
    ACTIVE_SERVICES = []
    PRESSURE_HISTORY = deque(maxlen=200)
    SIMULATION_METRICS = []
    PLAN = {"scale_decision": "none", "offload_ratio": 0.0}

    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default=os.path.join("data", "workload.csv"))
    ap.add_argument("--lstm_model", default=os.path.join("models", "lstm_util.pth"))
    ap.add_argument("--dqn_model", default=os.path.join("models", "dqn_fog_cloud.pth"))
    ap.add_argument("--fog_cpu", type=int, default=100)
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--output_csv", default=None, help="Chemin pour sauvegarder les métriques CSV.")
    args = ap.parse_args()

    W_WINDOW = args.W

    if not os.path.exists(args.workload):
        raise FileNotFoundError(f"Workload introuvable: {args.workload}")

    WORKLOAD_IDX = load_workload_indexed(args.workload)
    nb_tasks = sum(len(v) for v in WORKLOAD_IDX.values())
    print(f"[OK] Workload chargé: {args.workload} ({nb_tasks} tâches)")

    MODULE1 = Module1_LSTMPredictor(model_path=args.lstm_model)
    MODULE2 = Module2_ProactivePlanner(target_util=0.70)
    MODULE3 = Module3_Scheduler(dqn_path=args.dqn_model, cpu_threshold_cloud=300)

    simulator = Simulator(tick_duration=1, tick_unit="seconds")

    cloud = EdgeServer(cpu=1000, memory=100000, disk=100000)
    cloud.name = "Cloud-AWS"
    cloud.coordinates = [10, 10]

    fog = EdgeServer(cpu=args.fog_cpu, memory=4096, disk=10000)
    fog.name = "Fog-1"
    fog.coordinates = [5, 5]

    simulator.stopping_criterion = lambda sim: (globals()["CURRENT_T"] >= args.ticks)

    print("\n--- Démarrage simulation EdgeSimPy ---")
    simulator.resource_management_algorithm = proactive_placement_algorithm
    simulator.resource_management_algorithm_parameters = {"simulator": simulator}
    simulator.run_model()
    print("--- Fin simulation ---")

    if args.output_csv:
        output_path = args.output_csv
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if SIMULATION_METRICS:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SIMULATION_METRICS[0].keys())
                writer.writeheader()
                writer.writerows(SIMULATION_METRICS)
            print(f"[OK] Métriques de simulation sauvegardées dans: {output_path}")



if __name__ == "__main__":
    main()
