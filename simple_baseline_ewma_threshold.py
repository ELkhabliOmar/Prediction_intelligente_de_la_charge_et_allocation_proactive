# simple_baseline_ewma_threshold.py
import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ============================================================
# 1) Chargement workload (compatible formats multiples)
# ============================================================

def normalize_task_row(row):
    # Format déjà normalisé
    if "cpu_demand" in row and "ram_demand" in row:
        return {
            "timestamp": int(float(row.get("timestamp", 0))),
            "cpu_demand": float(row["cpu_demand"]),
            "ram_demand": float(row["ram_demand"]),
            "duration": max(1, int(float(row.get("duration", 1))))
        }

    # Format Tuple30K / Pakistan dataset (approximation cohérente)
    # Ajuste ces scales si besoin selon ton dataset réel
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


# ============================================================
# 2) Topologie (respecte ta topologie JSON)
# ============================================================

def load_topology_json(path):
    with open(path, "r", encoding="utf-8") as f:
        topo = json.load(f)

    nodes = topo["Nodes"]
    edges = topo["Edges"]
    defaults = topo.get("NetworkDefaults", {})
    constraints = topo.get("ScalingConstraints", {})

    fog_nodes = []
    cloud_nodes = []

    for n in nodes:
        dev = n.get("DeviceType", "")
        if dev == "Fog":
            fog_nodes.append({
                "id": n["NodeId"],
                "name": n["NodeName"],
                "status": n.get("Status", "inactive"),
                "cpu_raw": float(n["MaxCpuFreq"]),
                "buf": float(n.get("MaxBufferSize", 4096)),
                "idle_coef": float(n.get("IdleEnergyCoef", 10)),
                "exe_coef": float(n.get("ExeEnergyCoef", 50)),
                "activation_time": float(n.get("ActivationTimeSec", 0)),
                "activation_energy_cost": float(n.get("ActivationEnergyCost", 0)),
            })
        elif dev == "Cloud":
            cloud_nodes.append({
                "id": n["NodeId"],
                "name": n["NodeName"],
                "status": n.get("Status", "active"),
                "cpu_raw": float(n["MaxCpuFreq"]),
                "idle_coef": float(n.get("IdleEnergyCoef", 200)),
                "exe_coef": float(n.get("ExeEnergyCoef", 1000)),
            })

    # Liens edge->fog / edge->cloud (latence)
    edge_id = None
    for n in nodes:
        if n.get("DeviceType") == "Edge":
            edge_id = n["NodeId"]
            break

    lat_to_fog = {}
    lat_to_cloud = {}
    bw_to_fog = {}
    bw_to_cloud = {}

    for e in edges:
        if e.get("SrcNodeID") != edge_id:
            continue
        dst = e.get("DstNodeID")
        lat = float(e.get("LatencyMs", 10))
        bw = float(e.get("Bandwidth", 1000))
        # router vers fog/cloud
        if any(fn["id"] == dst for fn in fog_nodes):
            lat_to_fog[dst] = lat
            bw_to_fog[dst] = bw
        if any(cn["id"] == dst for cn in cloud_nodes):
            lat_to_cloud[dst] = lat
            bw_to_cloud[dst] = bw

    topo_parsed = {
        "fog_nodes": fog_nodes,
        "cloud_nodes": cloud_nodes,
        "lat_to_fog": lat_to_fog,
        "lat_to_cloud": lat_to_cloud,
        "bw_to_fog": bw_to_fog,
        "bw_to_cloud": bw_to_cloud,
        "defaults": defaults,
        "constraints": constraints,
    }
    return topo_parsed


# ============================================================
# 3) Baseline simple : EWMA + Threshold scaling/offload
# ============================================================

class SimpleBaselineEWMA:
    """
    Baseline simple:
    - prédiction: EWMA sur pression
    - scaling: seuils sur pression prédite
    - allocation: priorité Fog (nœud le moins chargé), sinon Cloud
    """

    def __init__(
        self,
        topo,
        ticks=300,
        ewma_alpha=0.35,
        target_pressure=0.70,
        up_th=0.80,
        down_th=0.30,
        emergency_th=1.00,
        min_active_fog=3,
        max_active_fog=8,
        cooldown_up=5,
        cooldown_down=10,
        cpu_unit_scale=1000.0,
    ):
        self.topo = topo
        self.ticks = ticks

        self.alpha = ewma_alpha
        self.target = target_pressure
        self.up_th = up_th
        self.down_th = down_th
        self.emergency_th = emergency_th

        self.min_active_fog = int(min_active_fog)
        self.max_active_fog = int(max_active_fog)

        self.cooldown_up = cooldown_up
        self.cooldown_down = cooldown_down

        # Conversion MaxCpuFreq -> unités CPU de simulation
        # ex: 110000 -> 110
        self.cpu_unit_scale = cpu_unit_scale

        # état runtime
        self.fogs = []
        for fn in topo["fog_nodes"]:
            self.fogs.append({
                **fn,
                "capacity": max(1.0, fn["cpu_raw"] / self.cpu_unit_scale),
                "active": (fn["status"] == "active"),
                "used_cpu": 0.0,
                "tasks_running": [],  # list of (remaining_dur, cpu)
                "cooldown": 0,
            })

        self.clouds = []
        for cn in topo["cloud_nodes"]:
            self.clouds.append({
                **cn,
                "capacity": max(1.0, cn["cpu_raw"] / self.cpu_unit_scale),
                "used_cpu": 0.0,
                "tasks_running": [],  # list of (remaining_dur, cpu)
            })

        # appliquer contraintes topo si dispo
        c = topo.get("constraints", {})
        self.min_active_fog = int(c.get("MinActiveFogNodes", self.min_active_fog))
        self.max_active_fog = int(c.get("MaxActiveFogNodes", self.max_active_fog))

        # cooldowns topo en secondes -> approx ticks
        # (si 1 tick ~ 1s)
        if c.get("ActivationCooldownSec") is not None:
            self.cooldown_up = int(max(1, c["ActivationCooldownSec"] // 60))  # adouci
        if c.get("DeactivationCooldownSec") is not None:
            self.cooldown_down = int(max(1, c["DeactivationCooldownSec"] // 90))

        self.pressure_ewma = 0.0
        self.press_hist = deque(maxlen=10)

        self.rows = []
        self.total_scaling = 0

    def _active_fogs(self):
        return [f for f in self.fogs if f["active"]]

    def _inactive_fogs(self):
        return [f for f in self.fogs if not f["active"]]

    def _fog_pool_capacity(self):
        return sum(f["capacity"] for f in self._active_fogs())

    def _fog_pool_used(self):
        return sum(f["used_cpu"] for f in self._active_fogs())

    def _step_release_tasks(self):
        # Fog
        for f in self.fogs:
            new_tasks = []
            released = 0.0
            for rem, cpu in f["tasks_running"]:
                rem -= 1
                if rem <= 0:
                    released += cpu
                else:
                    new_tasks.append((rem, cpu))
            f["tasks_running"] = new_tasks
            f["used_cpu"] = max(0.0, f["used_cpu"] - released)
            if f["cooldown"] > 0:
                f["cooldown"] -= 1

        # Cloud
        for c in self.clouds:
            new_tasks = []
            released = 0.0
            for rem, cpu in c["tasks_running"]:
                rem -= 1
                if rem <= 0:
                    released += cpu
                else:
                    new_tasks.append((rem, cpu))
            c["tasks_running"] = new_tasks
            c["used_cpu"] = max(0.0, c["used_cpu"] - released)

    def _predict_pressure(self, current_pressure):
        # EWMA + petite marge d'incertitude
        self.pressure_ewma = self.alpha * current_pressure + (1 - self.alpha) * self.pressure_ewma
        self.press_hist.append(current_pressure)

        if len(self.press_hist) >= 2:
            unc = float(np.std(self.press_hist))
        else:
            unc = 0.05

        pred = self.pressure_ewma
        return pred, max(0.03, unc)

    def _scale_decision(self, pred_pressure, current_pressure):
        decision = "none"

        active = self._active_fogs()
        inactive = self._inactive_fogs()

        # emergency_up si surcharge réelle
        if current_pressure > self.emergency_th and len(active) < self.max_active_fog and inactive:
            cand = sorted(inactive, key=lambda x: (-x["capacity"], x["id"]))[0]
            cand["active"] = True
            cand["cooldown"] = self.cooldown_up
            decision = "emergency_up"
            self.total_scaling += 1
            return decision

        # up si pression prédite élevée
        if pred_pressure > self.up_th and len(active) < self.max_active_fog and inactive:
            cand = sorted(inactive, key=lambda x: (-x["capacity"], x["id"]))[0]
            cand["active"] = True
            cand["cooldown"] = self.cooldown_up
            decision = "up"
            self.total_scaling += 1
            return decision

        # down si prédiction basse et assez de fog actifs
        if pred_pressure < self.down_th and len(active) > self.min_active_fog:
            # désactiver le fog le moins chargé, sans cooldown, et sans tâches
            candidates = [f for f in active if f["cooldown"] == 0 and len(f["tasks_running"]) == 0]
            if candidates:
                cand = sorted(candidates, key=lambda x: (x["used_cpu"], x["capacity"]))[0]
                cand["active"] = False
                cand["cooldown"] = self.cooldown_down
                decision = "down"
                self.total_scaling += 1
                return decision

        return decision

    def _place_task_on_fog(self, task_cpu, dur):
        active = self._active_fogs()
        if not active:
            return None

        # choisir nœud fog avec pression minimale (best-fit light)
        best = None
        best_score = None
        for f in active:
            projected = (f["used_cpu"] + task_cpu) / max(1e-9, f["capacity"])
            # on préfère <=1.0, sinon quand même possible (surcharge)
            score = projected
            if (best is None) or (score < best_score):
                best = f
                best_score = score

        best["used_cpu"] += task_cpu
        best["tasks_running"].append((dur, task_cpu))
        return best

    def _place_task_on_cloud(self, task_cpu, dur):
        # choisir cloud le moins chargé
        c = sorted(self.clouds, key=lambda x: x["used_cpu"] / max(1e-9, x["capacity"]))[0]
        c["used_cpu"] += task_cpu
        c["tasks_running"].append((dur, task_cpu))
        return c

    def run(self, tasks, verbose=True):
        # Grouper les tâches par tick
        tasks_by_t = defaultdict(list)
        for task in tasks:
            t = int(task["timestamp"])
            if 0 <= t < self.ticks:
                tasks_by_t[t].append(task)

        if verbose:
            total_in_window = sum(len(v) for v in tasks_by_t.values())
            print(f"Topologie chargée: fog={len(self.fogs)}, cloud={len(self.clouds)}")
            print(f"Tâches (arrivent dans [0..{self.ticks-1}]): {total_in_window}")

        for t in range(self.ticks):
            self._step_release_tasks()

            # pression actuelle avant nouvelles tâches
            pool_cap = self._fog_pool_capacity()
            pool_used = self._fog_pool_used()
            current_pressure = (pool_used / pool_cap) if pool_cap > 0 else 0.0

            pred_pressure, pred_unc = self._predict_pressure(current_pressure)
            scale_decision = self._scale_decision(pred_pressure, current_pressure)

            # recalcul après scaling
            pool_cap = self._fog_pool_capacity()
            pool_used = self._fog_pool_used()
            current_pressure = (pool_used / pool_cap) if pool_cap > 0 else 0.0

            # Politique d'offload simple
            # si pression élevée -> offload proportionnel
            if current_pressure <= 0.70:
                offload_ratio = 0.0
            elif current_pressure >= 1.10:
                offload_ratio = 0.80
            else:
                # interpolation linéaire entre 0.70 et 1.10
                offload_ratio = (current_pressure - 0.70) / (1.10 - 0.70) * 0.80
            offload_ratio = float(np.clip(offload_ratio, 0.0, 0.80))

            # Placement des tâches arrivantes
            arrived = tasks_by_t[t]
            placed_fog = 0
            placed_cloud = 0

            for task in arrived:
                cpu = float(task["cpu_demand"])
                dur = int(task["duration"])

                # décision cloud probabiliste basée sur offload_ratio
                # (déterministe si tu veux: threshold sur pression + taille CPU)
                send_cloud = False
                if offload_ratio > 0:
                    # plus la tâche est grosse, plus on la déleste quand pression haute
                    task_weight = min(1.0, cpu / 25.0)
                    cloud_prob = min(1.0, offload_ratio * (0.6 + 0.8 * task_weight))
                    send_cloud = (np.random.rand() < cloud_prob)

                if send_cloud:
                    self._place_task_on_cloud(cpu, dur)
                    placed_cloud += 1
                else:
                    self._place_task_on_fog(cpu, dur)
                    placed_fog += 1

            # mesures finales tick
            pool_cap = self._fog_pool_capacity()
            pool_used = self._fog_pool_used()
            pressure = (pool_used / pool_cap) if pool_cap > 0 else 0.0

            # métriques par nœud fog/cloud
            row = {
                "t": t,
                "pool_active_cpu_fog": round(pool_used, 4),
                "pool_fog_capacity": round(pool_cap, 4),
                "pool_pressure": round(pressure, 6),
                "predicted_pressure": round(float(pred_pressure), 6),
                "prediction_uncertainty": round(float(pred_unc), 6),
                "offload_ratio": round(offload_ratio, 6),
                "tasks_placed_fog": int(placed_fog),
                "tasks_placed_cloud": int(placed_cloud),
                "scale_decision": scale_decision,
            }

            # détails fog
            for f in self.fogs:
                # pression par nœud (même si inactif => 0)
                fp = (f["used_cpu"] / f["capacity"]) if (f["active"] and f["capacity"] > 0) else 0.0
                # puissance simplifiée: idle + exe*utilization si actif
                if f["active"]:
                    util = min(1.5, fp)
                    power = f["idle_coef"] + f["exe_coef"] * min(1.0, util)
                else:
                    power = 0.0
                row[f"fog_{f['name']}_p"] = round(fp, 6)
                row[f"fog_{f['name']}_power"] = round(power, 4)

            # détails cloud
            for c in self.clouds:
                cp = (c["used_cpu"] / c["capacity"]) if c["capacity"] > 0 else 0.0
                row[f"cloud_{c['name']}_load"] = round(cp, 6)

            self.rows.append(row)

            if verbose and (t % 20 == 0 or t == self.ticks - 1):
                active_names = [f["name"] for f in self._active_fogs()]
                print(
                    f"T{t:03d} poolCPU={pool_used:.0f}/{pool_cap:.0f} p={pressure:.2f} | "
                    f"pred={pred_pressure:.2f} off={offload_ratio:.2f} | "
                    f"placed fog={placed_fog} cloud={placed_cloud} | fog actifs={len(active_names)} {active_names}"
                )

        return pd.DataFrame(self.rows)


# ============================================================
# 4) Graph simple (même esprit que ton plot)
# ============================================================

def plot_simple_results(csv_path, output_png, title="Approche simple (EWMA + Threshold)"):
    df = pd.read_csv(csv_path)

    # compat avec ton style
    df = df.rename(columns={
        "pool_pressure": "pressure",
        "pool_active_cpu_fog": "active_cpu_fog",
        "pool_fog_capacity": "fog_capacity",
    }).copy()

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # 1) Pression + prédiction
    ax = axes[0]
    ax.plot(df["t"], df["pressure"], label="Pression réelle", linewidth=2.2)
    if "predicted_pressure" in df.columns:
        ax.plot(df["t"], df["predicted_pressure"], "--", label="Prédiction EWMA", linewidth=1.6)
    if "prediction_uncertainty" in df.columns:
        low = df["predicted_pressure"] - df["prediction_uncertainty"]
        up = df["predicted_pressure"] + df["prediction_uncertainty"]
        ax.fill_between(df["t"], low, up, alpha=0.15, label="Incertitude")
    ax.axhline(0.7, linestyle=":", linewidth=1.2, label="Cible 70%")
    ax.axhline(1.0, linestyle=":", linewidth=1.2, label="Surcharge 100%")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Pression")
    ax.set_title("Pression réelle vs prédiction (EWMA)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    # 2) Charge vs capacité
    ax = axes[1]
    ax.plot(df["t"], df["active_cpu_fog"], label="Charge Fog CPU", linewidth=2)
    ax.plot(df["t"], df["fog_capacity"], "--", label="Capacité Fog CPU", linewidth=2)
    ax.fill_between(
        df["t"], df["active_cpu_fog"], df["fog_capacity"],
        where=df["active_cpu_fog"] > df["fog_capacity"], alpha=0.2, label="Dépassement"
    )
    ax.set_ylabel("CPU")
    ax.set_title("Charge Fog vs Capacité Fog")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    # 3) Offload + scaling events
    ax = axes[2]
    ax.plot(df["t"], df["offload_ratio"], label="Offload ratio", linewidth=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    if "scale_decision" in df.columns:
        for _, r in df[df["scale_decision"].isin(["up", "down", "emergency_up"])].iterrows():
            if r["scale_decision"] in ["up", "emergency_up"]:
                ax.axvline(r["t"], linestyle=":", alpha=0.35)
            else:
                ax.axvline(r["t"], linestyle="--", alpha=0.25)
    ax.set_ylabel("Offload")
    ax.set_xlabel("Temps (ticks)")
    ax.set_title("Délestage Cloud + événements de scaling")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    # stats rapides
    mae = 0.0
    valid = df["predicted_pressure"] > 0
    if valid.any():
        mae = float((df.loc[valid, "pressure"] - df.loc[valid, "predicted_pressure"]).abs().mean())

    overload = float((df["pressure"] > 1.0).mean())
    pavg = float(df["pressure"].mean())
    offavg = float(df["offload_ratio"].mean())
    nscale = int(df["scale_decision"].isin(["up", "down", "emergency_up"]).sum())

    # énergie totale fog
    power_cols = [c for c in df.columns if c.startswith("fog_") and c.endswith("_power")]
    energy_kj = float(df[power_cols].sum().sum() / 1000.0) if power_cols else 0.0

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(
        0.99, 0.01,
        f"P_moy={pavg:.2%} | Surcharge={overload:.1%} | Offload={offavg:.1%} | Scaling={nscale} | MAE={mae:.4f} | Energie={energy_kj:.2f} kJ",
        ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.12)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] Graphe sauvegardé: {output_png}")


# ============================================================
# 5) Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Approche simple EWMA + Threshold (prediction + allocation + plot)")
    ap.add_argument("--workload", required=True, help="CSV workload")
    ap.add_argument("--topology", required=True, help="JSON topologie")
    ap.add_argument("--ticks", type=int, default=300, help="Nombre de ticks")
    ap.add_argument("--output_csv", default="outputs/results_simple_baseline.csv")
    ap.add_argument("--output_plot", default="outputs/results_simple_baseline.png")
    ap.add_argument("--seed", type=int, default=42)

    # paramètres baseline
    ap.add_argument("--alpha", type=float, default=0.35, help="EWMA alpha")
    ap.add_argument("--target", type=float, default=0.70)
    ap.add_argument("--up_th", type=float, default=0.80)
    ap.add_argument("--down_th", type=float, default=0.30)
    ap.add_argument("--emergency_th", type=float, default=1.00)
    ap.add_argument("--cpu_scale", type=float, default=1000.0,
                    help="Conversion MaxCpuFreq -> CPU simulation (1000 => 110000 -> 110)")

    args = ap.parse_args()
    np.random.seed(args.seed)

    tasks = load_workload_csv(args.workload)
    topo = load_topology_json(args.topology)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_plot) or ".", exist_ok=True)

    sim = SimpleBaselineEWMA(
        topo=topo,
        ticks=args.ticks,
        ewma_alpha=args.alpha,
        target_pressure=args.target,
        up_th=args.up_th,
        down_th=args.down_th,
        emergency_th=args.emergency_th,
        cpu_unit_scale=args.cpu_scale,
    )

    df = sim.run(tasks, verbose=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[OK] CSV résultats sauvegardé: {args.output_csv}")

    # Stats console
    pavg = df["pool_pressure"].mean()
    overload = (df["pool_pressure"] > 1.0).mean()
    offavg = df["offload_ratio"].mean()
    nscale = df["scale_decision"].isin(["up", "down", "emergency_up"]).sum()
    mae = 0.0
    valid = df["predicted_pressure"] > 0
    if valid.any():
        mae = (df.loc[valid, "pool_pressure"] - df.loc[valid, "predicted_pressure"]).abs().mean()
    power_cols = [c for c in df.columns if c.startswith("fog_") and c.endswith("_power")]
    energy_kj = df[power_cols].sum().sum() / 1000.0 if power_cols else 0.0

    print("\n" + "=" * 72)
    print("STATISTIQUES - Approche simple (EWMA + Threshold)")
    print("=" * 72)
    print(f"Pression moyenne      : {pavg:.2%}")
    print(f"Surcharge (>100%)     : {overload:.1%}")
    print(f"Offload moyen         : {offavg:.1%}")
    print(f"Nb scaling            : {int(nscale)}")
    print(f"MAE prédiction (EWMA) : {mae:.4f}")
    print(f"Energie Fog totale    : {energy_kj:.2f} kJ")
    print("=" * 72)

    # Plot simple
    plot_simple_results(args.output_csv, args.output_plot, title="Approche simple (EWMA + Threshold)")


if __name__ == "__main__":
    main()