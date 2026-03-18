"""

python simple_baseline_arima_threshold.py --workload "dataset/Pakistan/data/Tuple30K/testset.csv" --topology "topology/fog_cloud_topology.json" --ticks 200 --output_csv "data/results_baseline_test.csv" --output_plot "data/plot_baseline_test.png"

python simple_baseline_arima_threshold.py --workload "dataset/Pakistan/data/Tuple30K/trainset.csv" --topology "topology/fog_cloud_topology.json" --ticks 200 --output_csv "data/results_baseline_train.csv" --output_plot "data/plot_baseline_train.png"



Baseline scientifique corrigée : ARIMA + TOPSIS + Seuils robustes

Améliorations principales :
- Correction de l'incohérence d'unités dans TOPSIS (utilise node['capacity']).
- Warm-up avant scaling pour éviter le scale-down prématuré.
- Hystérésis / patience pour scale-up et scale-down.
- Cooldown entre décisions de scaling.
- Sélection cloud en round-robin pondéré par charge.
- Collecte de métriques plus complète.
- Figure de baseline enrichie pour comparaison scientifique avec l'approche proactive.

Exemple :
python simple_baseline_arima_threshold_corrected.py \
  --workload "dataset/Pakistan/data/Tuple30K/testset.csv" \
  --topology "topology/fog_cloud_topology.json" \
  --ticks 200 \
  --output_csv "data/results_baseline_corrected.csv" \
  --output_plot "data/plot_baseline_corrected.png"
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels non trouvé. Fallback automatique sur EWMA.")


# ============================================================
# 1) Chargement & Normalisation
# ============================================================

def normalize_task_row(row):
    if "cpu_demand" in row and "ram_demand" in row:
        return {
            "timestamp": int(float(row.get("timestamp", 0))),
            "cpu_demand": float(row["cpu_demand"]),
            "ram_demand": float(row["ram_demand"]),
            "duration": max(1, int(float(row.get("duration", 1))))
        }

    task_size = float(row.get("TaskSize", 0.0))
    cycles_per_bit = float(row.get("CyclesPerBit", 0.0))
    trans_rate = max(1.0, float(row.get("TransBitRate", 1.0)))
    cpu_scale = 3000.0
    ram_scale = 1.0

    cpu_demand = max(1.0, (task_size * cycles_per_bit) / cpu_scale)
    ram_demand = max(64.0, task_size * ram_scale)
    duration = max(1, int(math.ceil(task_size / trans_rate)))

    return {
        "timestamp": int(float(row.get("GenerationTime", 0.0))),
        "cpu_demand": cpu_demand,
        "ram_demand": ram_demand,
        "duration": duration
    }


def load_workload_csv(path):
    tasks = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tasks.append(normalize_task_row(row))
    return tasks


def load_topology_json(path):
    with open(path, "r", encoding="utf-8") as f:
        topo = json.load(f)

    fog_nodes = []
    cloud_nodes = []

    for n in topo["Nodes"]:
        dev = n.get("DeviceType", "")
        node_struct = {
            "id": n["NodeId"],
            "name": n.get("NodeName", f"node_{n['NodeId']}"),
            "status": n.get("Status", "inactive"),
            "cpu_raw": float(n.get("MaxCpuFreq", 1000.0)),
            "ram_raw": float(n.get("MaxBufferSize", 4096.0)),
            "idle_coef": float(n.get("IdleEnergyCoef", 100.0)),
            "exe_coef": float(n.get("ExeEnergyCoef", 150.0)),
            "loc_x": float(n.get("LocX", 0.0)),
            "loc_y": float(n.get("LocY", 0.0)),
        }
        if "Fog" in dev:
            fog_nodes.append(node_struct)
        elif "Cloud" in dev:
            cloud_nodes.append(node_struct)

    return {"fog_nodes": fog_nodes, "cloud_nodes": cloud_nodes}


# ============================================================
# 2) Module de Prédiction : ARIMA / EWMA
# ============================================================

class ARIMAPredictor:
    def __init__(self, order=(1, 0, 0), history_size=50, alpha=0.35):
        self.order = order
        self.history = deque(maxlen=history_size)
        self.alpha = alpha
        self.ewma = 0.0

    def update(self, value):
        value = float(max(0.0, value))
        self.history.append(value)
        self.ewma = self.alpha * value + (1.0 - self.alpha) * self.ewma

    def predict(self, steps=1):
        # Warm-up : pas assez d'historique
        if len(self.history) < 12:
            return float(self.ewma), 0.08

        data = list(self.history)
        if np.std(data) < 1e-9:
            return float(data[-1]), 0.0

        if not STATSMODELS_AVAILABLE:
            return float(self.ewma), max(0.05, float(np.std(data[-min(len(data), 10):])))

        try:
            model = ARIMA(data, order=self.order)
            fit = model.fit()
            forecast = fit.forecast(steps=steps)
            pred = float(max(0.0, forecast[-1]))
            resid = np.asarray(fit.resid, dtype=float)
            unc = float(np.std(resid)) if resid.size > 1 else 0.08
            unc = max(0.03, min(0.35, unc))
            return pred, unc
        except Exception:
            return float(self.ewma), max(0.05, float(np.std(data[-min(len(data), 10):])))


# ============================================================
# 3) TOPSIS corrigé
# ============================================================

class TOPSISSelector:
    def __init__(self, weights=None):
        self.weights = weights or {"cpu": 0.55, "ram": 0.20, "dist": 0.25}

    def _task_origin(self, task_profile):
        return float(task_profile.get("src_x", 0.0)), float(task_profile.get("src_y", 0.0))

    def select_best_node(self, candidates, task_profile, current_loads):
        if not candidates:
            return None
        if len(candidates) == 1:
            node = candidates[0]
            load = current_loads.get(node["id"], {"cpu": 0.0, "ram": 0.0})
            cpu_ratio = (load["cpu"] + task_profile["cpu_demand"]) / max(1.0, node["capacity"])
            ram_ratio = (load["ram"] + task_profile["ram_demand"]) / max(1.0, node["ram_raw"])
            if cpu_ratio <= 1.0 and ram_ratio <= 1.0:
                return node
            return None

        origin_x, origin_y = self._task_origin(task_profile)
        matrix = []
        valid_candidates = []

        for node in candidates:
            node_id = node["id"]
            load = current_loads.get(node_id, {"cpu": 0.0, "ram": 0.0})

            # ✅ CORRECTION MAJEURE : utiliser la capacité simulée, pas cpu_raw
            cpu_cap = max(1.0, float(node["capacity"]))
            ram_cap = max(1.0, float(node["ram_raw"]))

            proj_cpu_load = (float(load["cpu"]) + float(task_profile["cpu_demand"])) / cpu_cap
            proj_ram_load = (float(load["ram"]) + float(task_profile["ram_demand"])) / ram_cap
            dist = math.sqrt((float(node["loc_x"]) - origin_x) ** 2 + (float(node["loc_y"]) - origin_y) ** 2)

            if proj_cpu_load > 1.0 or proj_ram_load > 1.0:
                continue

            matrix.append([proj_cpu_load, proj_ram_load, dist])
            valid_candidates.append(node)

        if not valid_candidates:
            return None

        np_matrix = np.array(matrix, dtype=float)
        norm = np.sqrt((np_matrix ** 2).sum(axis=0)) + 1e-9
        norm_matrix = np_matrix / norm

        w = np.array([self.weights["cpu"], self.weights["ram"], self.weights["dist"]], dtype=float)
        weighted = norm_matrix * w

        ideal_best = weighted.min(axis=0)
        ideal_worst = weighted.max(axis=0)

        dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
        scores = dist_worst / (dist_best + dist_worst + 1e-9)

        return valid_candidates[int(np.argmax(scores))]


# ============================================================
# 4) Simulation baseline corrigée
# ============================================================

class BaselineARIMA_TOPSIS_Corrected:
    def __init__(
        self,
        topo,
        ticks=300,
        cpu_scale=1000.0,
        predictor_order=(1, 0, 0),
        predictor_horizon=5,
        warmup_ticks=20,
        scale_up_threshold=0.80,
        scale_down_threshold=0.30,
        offload_threshold=0.92,
        target_util=0.70,
        up_patience=2,
        down_patience=3,
        min_active_fogs=2,
        scaling_cooldown=8,
        low_node_util_threshold=0.10,
    ):
        self.topo = topo
        self.ticks = int(ticks)
        self.cpu_scale = float(cpu_scale)
        self.predictor_horizon = int(max(1, predictor_horizon))
        self.warmup_ticks = int(max(0, warmup_ticks))
        self.scale_up_threshold = float(scale_up_threshold)
        self.scale_down_threshold = float(scale_down_threshold)
        self.offload_threshold = float(offload_threshold)
        self.target_util = float(target_util)
        self.up_patience = int(max(1, up_patience))
        self.down_patience = int(max(1, down_patience))
        self.min_active_fogs = int(max(1, min_active_fogs))
        self.scaling_cooldown = int(max(0, scaling_cooldown))
        self.low_node_util_threshold = float(low_node_util_threshold)

        self.fogs = []
        for fn in topo["fog_nodes"]:
            self.fogs.append({
                **fn,
                "capacity": max(1.0, float(fn["cpu_raw"]) / self.cpu_scale),
                "active": (fn["status"] == "active"),
                "used_cpu": 0.0,
                "used_ram": 0.0,
                "tasks": [],
            })

        self.clouds = []
        for cn in topo["cloud_nodes"]:
            self.clouds.append({
                **cn,
                "capacity": 999999.0,
                "used_cpu": 0.0,
                "used_ram": 0.0,
                "tasks": [],
            })

        # Garantir un minimum de fogs actifs au démarrage
        active_count = sum(1 for f in self.fogs if f["active"])
        if active_count < self.min_active_fogs:
            for f in sorted(self.fogs, key=lambda x: -x["capacity"]):
                if not f["active"]:
                    f["active"] = True
                    active_count += 1
                    if active_count >= self.min_active_fogs:
                        break

        self.predictor = ARIMAPredictor(order=predictor_order, history_size=40, alpha=0.35)
        self.selector = TOPSISSelector(weights={"cpu": 0.55, "ram": 0.20, "dist": 0.25})

        self.metrics = []
        self.total_scale_up = 0
        self.total_scale_down = 0
        self.total_energy_joules = 0.0
        self.total_fog_tasks = 0
        self.total_cloud_tasks = 0
        self.total_overload_ticks = 0
        self.cloud_rr = 0

        self.up_counter = 0
        self.down_counter = 0
        self.last_scaling_tick = -10**9

    def _active_fogs(self):
        return [f for f in self.fogs if f["active"]]

    def _pool_stats(self):
        active = self._active_fogs()
        total_cap = sum(f["capacity"] for f in active)
        total_used = sum(f["used_cpu"] for f in active)
        pressure = (total_used / total_cap) if total_cap > 0 else 0.0
        return total_cap, total_used, pressure

    def _select_cloud(self):
        if not self.clouds:
            return None
        # Cloud choisi parmi les deux moins chargés, avec round robin local pour éviter tout envoyer vers c0
        ranked = sorted(self.clouds, key=lambda c: (c["used_cpu"], c["name"]))
        topk = ranked[:max(1, min(2, len(ranked)))]
        chosen = topk[self.cloud_rr % len(topk)]
        self.cloud_rr += 1
        return chosen

    def _release_finished_tasks(self):
        for node in self.fogs + self.clouds:
            remaining_tasks = []
            freed_cpu = 0.0
            freed_ram = 0.0
            for dur, cpu, ram in node["tasks"]:
                if dur > 1:
                    remaining_tasks.append((dur - 1, cpu, ram))
                else:
                    freed_cpu += cpu
                    freed_ram += ram
            node["tasks"] = remaining_tasks
            node["used_cpu"] = max(0.0, node["used_cpu"] - freed_cpu)
            node["used_ram"] = max(0.0, node.get("used_ram", 0.0) - freed_ram)

    def _apply_scaling(self, t, pred_pressure):
        scale_decision = "none"
        active = self._active_fogs()
        cooldown_ok = (t - self.last_scaling_tick) >= self.scaling_cooldown

        if t < self.warmup_ticks or not cooldown_ok:
            return scale_decision

        if pred_pressure >= self.scale_up_threshold:
            self.up_counter += 1
            self.down_counter = 0
        elif pred_pressure <= self.scale_down_threshold:
            self.down_counter += 1
            self.up_counter = 0
        else:
            self.up_counter = 0
            self.down_counter = 0

        if self.up_counter >= self.up_patience:
            inactive = [f for f in self.fogs if not f["active"]]
            if inactive:
                cand = sorted(inactive, key=lambda x: (-x["capacity"], x["name"]))[0]
                cand["active"] = True
                self.total_scale_up += 1
                self.last_scaling_tick = t
                self.up_counter = 0
                return "up"

        if self.down_counter >= self.down_patience and len(active) > self.min_active_fogs:
            candidates = []
            for f in active:
                util = f["used_cpu"] / max(1.0, f["capacity"])
                if util <= self.low_node_util_threshold and len(f["tasks"]) == 0:
                    candidates.append(f)

            if candidates:
                cand = sorted(candidates, key=lambda x: (x["capacity"], x["name"]))[0]
                cand["active"] = False
                self.total_scale_down += 1
                self.last_scaling_tick = t
                self.down_counter = 0
                return "down"

        return scale_decision

    def run(self, tasks):
        tasks_by_time = defaultdict(list)
        for task in tasks:
            if 0 <= int(task["timestamp"]) < self.ticks:
                tasks_by_time[int(task["timestamp"])].append(task)

        print(f"🚀 Démarrage Simulation Baseline corrigée (ARIMA + TOPSIS) sur {self.ticks} ticks...")

        pred_pressure = 0.0
        pred_unc = 0.0

        for t in range(self.ticks):
            # 1) Libération des tâches terminées
            self._release_finished_tasks()

            # 2) Observer l'état courant, puis mettre à jour le prédicteur
            cap_before, used_before, pressure_before = self._pool_stats()
            self.predictor.update(pressure_before)

            # Refait un forecast périodique, mais garde une valeur à chaque tick
            if (t % 5 == 0) or (t < self.warmup_ticks):
                pred_pressure, pred_unc = self.predictor.predict(steps=self.predictor_horizon)

            # 3) Scaling robuste
            scale_decision = self._apply_scaling(t, pred_pressure)

            # 4) Allocation des tâches
            new_tasks = tasks_by_time.get(t, [])
            tasks_placed_fog = 0
            tasks_placed_cloud = 0

            current_loads = {
                f["id"]: {"cpu": float(f["used_cpu"]), "ram": float(f.get("used_ram", 0.0))}
                for f in self._active_fogs()
            }

            for task in new_tasks:
                cap_now, used_now, pressure_now = self._pool_stats()
                should_offload = (pressure_now >= self.offload_threshold) or (pred_pressure >= 0.98)

                target_node = None
                decision = "Cloud"

                if not should_offload:
                    active_fogs = self._active_fogs()
                    best_fog = self.selector.select_best_node(active_fogs, task, current_loads)
                    if best_fog is not None:
                        cpu_after = best_fog["used_cpu"] + task["cpu_demand"]
                        ram_after = best_fog.get("used_ram", 0.0) + task["ram_demand"]
                        if cpu_after <= best_fog["capacity"] and ram_after <= best_fog["ram_raw"]:
                            target_node = best_fog
                            decision = "Fog"

                if decision == "Fog" and target_node is not None:
                    target_node["used_cpu"] += float(task["cpu_demand"])
                    target_node["used_ram"] += float(task["ram_demand"])
                    target_node["tasks"].append((int(task["duration"]), float(task["cpu_demand"]), float(task["ram_demand"])))
                    current_loads[target_node["id"]]["cpu"] += float(task["cpu_demand"])
                    current_loads[target_node["id"]]["ram"] += float(task["ram_demand"])
                    tasks_placed_fog += 1
                    self.total_fog_tasks += 1
                else:
                    target_node = self._select_cloud()
                    if target_node is not None:
                        target_node["used_cpu"] += float(task["cpu_demand"])
                        target_node["used_ram"] += float(task["ram_demand"])
                        target_node["tasks"].append((int(task["duration"]), float(task["cpu_demand"]), float(task["ram_demand"])))
                    tasks_placed_cloud += 1
                    self.total_cloud_tasks += 1

            # 5) Mesures réelles après allocation
            cap_after, used_after, pressure_after = self._pool_stats()
            if pressure_after > 1.0:
                self.total_overload_ticks += 1

            # Énergie
            energy_tick = 0.0
            for f in self.fogs:
                if f["active"]:
                    util = min(1.0, f["used_cpu"] / max(1.0, f["capacity"]))
                    power = float(f["idle_coef"]) + float(f["exe_coef"]) * util
                    energy_tick += power
            self.total_energy_joules += energy_tick

            # Stats détaillées par nœud
            per_node_stats = {}
            for f in self.fogs:
                util = min(1.2, f["used_cpu"] / max(1.0, f["capacity"]))
                power = 0.0
                if f["active"]:
                    power = float(f["idle_coef"]) + float(f["exe_coef"]) * min(1.0, util)
                safe_name = f["name"].replace("-", "_").replace(" ", "")
                per_node_stats[f"fog_{safe_name}_p"] = util if f["active"] else 0.0
                per_node_stats[f"fog_{safe_name}_load"] = float(f["used_cpu"])
                per_node_stats[f"fog_{safe_name}_power"] = power
                per_node_stats[f"fog_{safe_name}_active"] = 1 if f["active"] else 0

            for c in self.clouds:
                safe_name = c["name"].replace("-", "_").replace(" ", "")
                per_node_stats[f"cloud_{safe_name}_load"] = float(c["used_cpu"])

            metric_row = {
                "t": t,
                "active_cpu_fog": used_after,
                "fog_capacity": cap_after,
                "pressure": pressure_after,
                "predicted_pressure": pred_pressure,
                "prediction_uncertainty": pred_unc,
                "tasks_placed_fog": tasks_placed_fog,
                "tasks_placed_cloud": tasks_placed_cloud,
                "arrivals": len(new_tasks),
                "scale_decision": scale_decision,
                "scale_up_total": self.total_scale_up,
                "scale_down_total": self.total_scale_down,
                "energy_joules_cumul": self.total_energy_joules,
                "energy_tick": energy_tick,
                "offload_ratio": (tasks_placed_cloud / max(1, tasks_placed_fog + tasks_placed_cloud)),
                "active_fog_nodes": len(self._active_fogs()),
                "overloaded": 1 if pressure_after > 1.0 else 0,
            }
            metric_row.update(per_node_stats)
            self.metrics.append(metric_row)

            if t % 50 == 0:
                print(
                    f"   Tick {t}: Pression={pressure_after:.2f} (Pred={pred_pressure:.2f}) | "
                    f"Fog={tasks_placed_fog} Cloud={tasks_placed_cloud} | Fogs actifs={len(self._active_fogs())}"
                )

        return pd.DataFrame(self.metrics)


# ============================================================
# 5) Métriques analytiques
# ============================================================

def calculate_performance_metrics(df):
    df = df.copy()
    if "pressure" not in df.columns or "predicted_pressure" not in df.columns:
        raise ValueError("Colonnes pressure/predicted_pressure manquantes")

    err = df["pressure"] - df["predicted_pressure"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    bias = float(np.mean(err))

    avg_pressure = float(df["pressure"].mean())
    max_pressure = float(df["pressure"].max())
    overload_percentage = float((df["pressure"] > 1.0).mean())
    pressure_std = float(df["pressure"].std(ddof=0))

    avg_offload = float(df["offload_ratio"].mean()) if "offload_ratio" in df else 0.0
    max_offload = float(df["offload_ratio"].max()) if "offload_ratio" in df else 0.0
    total_fog_tasks = int(df["tasks_placed_fog"].sum()) if "tasks_placed_fog" in df else 0
    total_offloaded_tasks = int(df["tasks_placed_cloud"].sum()) if "tasks_placed_cloud" in df else 0

    num_scaling = int((df["scale_decision"] != "none").sum()) if "scale_decision" in df else 0
    scaling_frequency = float(num_scaling / max(1, len(df)))

    total_arrivals = int(df["arrivals"].sum()) if "arrivals" in df else (total_fog_tasks + total_offloaded_tasks)
    fog_efficiency = float(total_fog_tasks / max(1, total_arrivals))
    estimated_cost = float(total_offloaded_tasks * 0.03)

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "avg_pressure": avg_pressure,
        "max_pressure": max_pressure,
        "overload_percentage": overload_percentage,
        "pressure_std": pressure_std,
        "avg_offload": avg_offload,
        "max_offload": max_offload,
        "total_fog_tasks": total_fog_tasks,
        "total_offloaded_tasks": total_offloaded_tasks,
        "num_scaling": num_scaling,
        "scaling_frequency": scaling_frequency,
        "fog_efficiency": fog_efficiency,
        "estimated_cost": estimated_cost,
        "total_arrivals": total_arrivals,
    }


# ============================================================
# 6) Fonctions de visualisation enrichies
# ============================================================

def plot_load_capacity(ax, df):
    ax.plot(df["t"], df["active_cpu_fog"], label="Charge Fog (CPU)", linewidth=2)
    ax.plot(df["t"], df["fog_capacity"], label="Capacité Fog (CPU)", linestyle="--", linewidth=2)

    ax.fill_between(
        df["t"],
        df["active_cpu_fog"],
        df["fog_capacity"],
        where=(df["active_cpu_fog"] > df["fog_capacity"]),
        alpha=0.2,
        label="Dépassement",
    )

    for _, row in df.iterrows():
        if row["scale_decision"] == "up":
            ax.annotate("↑", xy=(row["t"], row["fog_capacity"]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
        elif row["scale_decision"] == "down":
            ax.annotate("↓", xy=(row["t"], row["fog_capacity"]), xytext=(0, -14),
                        textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Unités CPU")
    ax.set_title("① CHARGE vs CAPACITÉ FOG", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)


def plot_pressure_predictions(ax, df, metrics):
    ax.plot(df["t"], df["pressure"], label="Pression Réelle", linewidth=2)
    ax.plot(df["t"], df["predicted_pressure"], label="Prédiction ARIMA", linestyle="--", linewidth=1.5)

    lower = np.maximum(0.0, df["predicted_pressure"] - df["prediction_uncertainty"])
    upper = df["predicted_pressure"] + df["prediction_uncertainty"]
    ax.fill_between(df["t"], lower, upper, alpha=0.2, label="Incertitude (±)")

    ax.axhline(y=1.0, linestyle="-", linewidth=1.2, alpha=0.7, label="Seuil surcharge (100%)")
    ax.axhline(y=0.70, linestyle=":", linewidth=1.0, alpha=0.7, label="Cible (70%)")
    ax.axhline(y=0.30, linestyle=":", linewidth=1.0, alpha=0.7, label="Sous-utilisation (30%)")

    ax.fill_between(df["t"], 1.0, np.maximum(df["pressure"], 1.0), where=(df["pressure"] > 1.0), alpha=0.10)

    txt = f"MAE={metrics['mae']:.3f} | RMSE={metrics['rmse']:.3f} | Biais={metrics['bias']:+.3f}"
    ax.text(0.01, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.set_ylabel("Pression (Utilisation)")
    ax.set_title("② PRESSION RÉELLE vs PRÉDICTIONS", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(-0.05, max(1.2, min(2.5, float(df["pressure"].max()) * 1.15)))


def plot_performance_metrics_left(ax, metrics):
    ax.axis("off")
    text = (
        "📈 MÉTRIQUES DE PERFORMANCE\n\n"
        f"• MAE Prédiction: {metrics['mae']:.4f}\n"
        f"• RMSE Prédiction: {metrics['rmse']:.4f}\n"
        f"• Biais: {metrics['bias']:+.4f}\n"
        f"• Pression Moyenne: {metrics['avg_pressure']:.1%}\n"
        f"• Pression Max: {metrics['max_pressure']:.1%}\n"
        f"• Surcharge (>100%): {metrics['overload_percentage']:.1%}\n"
        f"• Stabilité (σ): {metrics['pressure_std']:.3f}"
    )
    ax.text(0.05, 0.92, text, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))
    ax.set_title("③ MÉTRIQUES PRÉDICTION", fontsize=12, fontweight="bold", pad=8)


def plot_performance_metrics_right(ax, metrics):
    ax.axis("off")
    text = (
        "⚙️ MÉTRIQUES OPÉRATIONNELLES\n\n"
        f"• Offload Moyen: {metrics['avg_offload']:.1%}\n"
        f"• Offload Max: {metrics['max_offload']:.1%}\n"
        f"• Tâches Fog: {metrics['total_fog_tasks']}\n"
        f"• Tâches Cloud: {metrics['total_offloaded_tasks']}\n"
        f"• Changements Scaling: {metrics['num_scaling']}\n"
        f"• Fréquence Scaling: {metrics['scaling_frequency']:.2f}/tick\n"
        f"• Efficacité Fog: {metrics['fog_efficiency']:.1%}\n"
        f"• Coût Estimé: ${metrics['estimated_cost']:.2f}"
    )
    ax.text(0.05, 0.92, text, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3))
    ax.set_title("④ MÉTRIQUES OPÉRATIONNELLES", fontsize=12, fontweight="bold", pad=8)


def plot_prediction_errors(ax, df):
    work = df.copy()
    work["pred_error"] = work["pressure"] - work["predicted_pressure"]
    work["pressure_bin"] = pd.cut(work["pressure"], bins=[0, 0.3, 0.7, 1.0, 2.0, 5.0], include_lowest=True)

    if work["pressure_bin"].notna().any():
        agg = work.groupby("pressure_bin", observed=False)["pred_error"].agg(["mean", "std", "count"])
        x = np.arange(len(agg))
        yerr = agg["std"].fillna(0.0).to_numpy()
        ax.bar(x, agg["mean"].to_numpy(), yerr=yerr, alpha=0.5, capsize=4)
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in agg.index], rotation=45, ha="right", fontsize=8)
        for i, c in enumerate(agg["count"].to_numpy()):
            ax.text(i, agg["mean"].iloc[i] + 0.02, f"n={int(c)}", ha="center", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Pas assez de données pour l'analyse d'erreur", ha="center", va="center")

    ax.set_ylabel("Erreur (Réel - Prédit)")
    ax.set_title("⑤ ANALYSE DES ERREURS DE PRÉDICTION", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)


def plot_fog_nodes_heatmap(ax, df):
    fog_cols = [c for c in df.columns if c.startswith("fog_") and c.endswith("_p")]
    if not fog_cols:
        ax.text(0.5, 0.5, "Pas de données détaillées par nœud", ha="center", va="center")
        return

    labels = [c.replace("fog_", "").replace("_p", "") for c in fog_cols]
    data = df[fog_cols].T.values
    im = ax.imshow(data, aspect="auto", cmap="plasma", vmin=0, vmax=1.2, interpolation="nearest")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Temps (ticks)")
    ax.set_title("⑥ UTILISATION DÉTAILLÉE PAR NŒUD FOG (Heatmap)", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
    cbar.set_label("Pression (0-1+)", fontsize=9)


def plot_energy_and_cloud_stats(ax, df):
    power_cols = [c for c in df.columns if c.startswith("fog_") and c.endswith("_power")]
    cloud_cols = [c for c in df.columns if c.startswith("cloud_") and c.endswith("_load")]

    ax.axis("off")

    if power_cols:
        energy_sums = df[power_cols].sum().values
        fog_labels = [c.replace("fog_", "").replace("_power", "") for c in power_cols]
        ax1 = ax.inset_axes([0, 0, 0.48, 1])
        x = np.arange(len(fog_labels))
        ax1.bar(x, energy_sums, alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(fog_labels, rotation=45, ha="right", fontsize=8)
        ax1.set_title("Consommation Énergétique Totale par Nœud Fog (J)", fontsize=10, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

    if cloud_cols:
        cloud_means = df[cloud_cols].mean().values
        cloud_labels = [c.replace("cloud_", "").replace("_load", "") for c in cloud_cols]
        ax2 = ax.inset_axes([0.52, 0, 0.48, 1])
        x2 = np.arange(len(cloud_labels))
        ax2.bar(x2, cloud_means, alpha=0.7)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(cloud_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_title("Charge Moyenne par Nœud Cloud (CPU)", fontsize=10, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

    ax.set_title("⑦ ÉNERGIE & CHARGE CLOUD", fontsize=12, fontweight="bold", pad=16)


def plot_simulation_results(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Erreur: fichier introuvable: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    metrics = calculate_performance_metrics(df)

    fig = plt.figure(figsize=(18, 24))
    gs = fig.add_gridspec(6, 2, height_ratios=[3, 3, 1.6, 2.5, 3, 2.5], hspace=0.55)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[3, :])
    ax6 = fig.add_subplot(gs[4, :])
    ax7 = fig.add_subplot(gs[5, :])

    workload_name = os.path.basename(csv_path).replace(".csv", "").replace("results_", "")
    fig.suptitle(f"ANALYSE BASELINE CORRIGÉE (ARIMA+TOPSIS) - {workload_name.upper()}",
                 fontsize=20, y=0.98, fontweight="bold")

    plot_load_capacity(ax1, df)
    plot_pressure_predictions(ax2, df, metrics)
    plot_performance_metrics_left(ax3, metrics)
    plot_performance_metrics_right(ax4, metrics)
    plot_prediction_errors(ax5, df)
    plot_fog_nodes_heatmap(ax6, df)
    plot_energy_and_cloud_stats(ax7, df)

    plt.tight_layout(rect=[0, 0.02, 1, 0.965])
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Graphique sauvegardé dans: {output_path}")


# ============================================================
# 7) Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline corrigée : ARIMA + TOPSIS")
    parser.add_argument("--workload", required=True, help="Chemin CSV workload")
    parser.add_argument("--topology", required=True, help="Chemin JSON topologie")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--output_csv", default="baseline_results_corrected.csv")
    parser.add_argument("--output_plot", default="baseline_plot_corrected.png")

    parser.add_argument("--cpu_scale", type=float, default=1000.0)
    parser.add_argument("--warmup_ticks", type=int, default=20)
    parser.add_argument("--predictor_horizon", type=int, default=5)
    parser.add_argument("--scale_up_threshold", type=float, default=0.80)
    parser.add_argument("--scale_down_threshold", type=float, default=0.30)
    parser.add_argument("--offload_threshold", type=float, default=0.92)
    parser.add_argument("--up_patience", type=int, default=2)
    parser.add_argument("--down_patience", type=int, default=3)
    parser.add_argument("--min_active_fogs", type=int, default=2)
    parser.add_argument("--scaling_cooldown", type=int, default=8)
    parser.add_argument("--low_node_util_threshold", type=float, default=0.10)
    args = parser.parse_args()

    tasks = load_workload_csv(args.workload)
    topo = load_topology_json(args.topology)

    sim = BaselineARIMA_TOPSIS_Corrected(
        topo=topo,
        ticks=args.ticks,
        cpu_scale=args.cpu_scale,
        predictor_horizon=args.predictor_horizon,
        warmup_ticks=args.warmup_ticks,
        scale_up_threshold=args.scale_up_threshold,
        scale_down_threshold=args.scale_down_threshold,
        offload_threshold=args.offload_threshold,
        up_patience=args.up_patience,
        down_patience=args.down_patience,
        min_active_fogs=args.min_active_fogs,
        scaling_cooldown=args.scaling_cooldown,
        low_node_util_threshold=args.low_node_util_threshold,
    )

    df = sim.run(tasks)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"💾 Résultats sauvegardés : {args.output_csv}")

    metrics = calculate_performance_metrics(df)
    last_row = df.iloc[-1]
    total_e_kj = float(last_row["energy_joules_cumul"]) / 1000.0

    print("\n" + "=" * 60)
    print("RÉSULTATS BASELINE CORRIGÉE (ARIMA + TOPSIS)")
    print("=" * 60)
    print(f"Pression Moyenne       : {metrics['avg_pressure']:.2%}")
    print(f"Pression Max           : {metrics['max_pressure']:.2%}")
    print(f"MAE / RMSE             : {metrics['mae']:.4f} / {metrics['rmse']:.4f}")
    print(f"Énergie Totale         : {total_e_kj:.2f} kJ")
    print(f"Taux Offloading Moyen  : {metrics['avg_offload']:.2%}")
    print(f"Tâches Fog / Cloud     : {metrics['total_fog_tasks']} / {metrics['total_offloaded_tasks']}")
    print(f"Scalings (up/down)     : {int(last_row['scale_up_total'])} / {int(last_row['scale_down_total'])}")
    print(f"Surcharge (>100%)      : {metrics['overload_percentage']:.2%}")
    print("=" * 60)

    plot_simulation_results(args.output_csv, args.output_plot)


if __name__ == "__main__":
    main()
