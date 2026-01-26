# project/test.py
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
import networkx as nx

import numpy as np
import torch

# Ajoute la racine du projet au PYTHONPATH pour trouver config.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_WORKLOAD, DEFAULT_LSTM, DEFAULT_DQN, DEFAULT_RESULTS

from project.sim_core import (
    setup_state,
    proactive_placement_algorithm,
    get_metrics,
    load_workload_indexed,
    Module1_LSTMPredictor,
    Module2_HVWPO_Planner,
    Module3_Scheduler,
)

from project.ui_utils import banner, print_table, print_final_stats, save_metrics_csv

try:
    from edge_sim_py import Simulator, EdgeServer
    EDGE_SIM_AVAILABLE = True
except Exception:
    EDGE_SIM_AVAILABLE = False


def _workload_window_stats(workload_idx: dict, ticks: int):
    """Retourne (nb_total, nb_in_window, min_ts, max_ts)."""
    all_ts = sorted(workload_idx.keys())
    if not all_ts:
        return 0, 0, None, None

    nb_total = sum(len(v) for v in workload_idx.values())
    nb_in_window = sum(len(v) for t, v in workload_idx.items() if 0 <= int(t) < int(ticks))
    return nb_total, nb_in_window, int(all_ts[0]), int(all_ts[-1])


def main():
    if not EDGE_SIM_AVAILABLE:
        print("❌ edge-sim-py n'est pas installé. pip install edge-sim-py")
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default=DEFAULT_WORKLOAD)
    ap.add_argument("--lstm_model", default=DEFAULT_LSTM)
    ap.add_argument("--dqn_model", default=DEFAULT_DQN)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--target_util", type=float, default=0.70)
    ap.add_argument("--min_fog_cpu", type=int, default=30)
    ap.add_argument("--output_csv", default=DEFAULT_RESULTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fog_scale", type=float, default=1.0)

    # ✅ debug
    ap.add_argument("--debug_accounting", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    banner("SIMULATION MULTI-FOG / MULTI-CLOUD", "Affichage tableau + MAPE")

    if not os.path.exists(args.workload):
        raise FileNotFoundError(f"Workload introuvable: {args.workload}")

    workload_idx = load_workload_indexed(args.workload)

    nb_total, nb_in_window, min_ts, max_ts = _workload_window_stats(workload_idx, args.ticks)

    # ✅ Infos très importantes : CSV total vs fenêtre de simulation
    info_rows = [
        ["Workload", args.workload],
        ["Tasks (CSV total)", str(nb_total)],
        ["Tasks (arrivent dans [0..ticks-1])", str(nb_in_window)],
        ["Ticks", str(args.ticks)],
        ["W (MAPE)", str(args.W)],
        ["Target util", str(args.target_util)],
        ["Output CSV", args.output_csv],
    ]
    if min_ts is not None:
        info_rows.insert(2, ["Timestamps range", f"[{min_ts} .. {max_ts}]"])

    print_table(["Param", "Value"], info_rows, title="⚙️ Paramètres")

    # ✅ Alerte si quasi toutes les tâches sont hors fenêtre
    if nb_total > 0 and nb_in_window < 0.2 * nb_total:
        print(
            "\n⚠️  INFO: La majorité des tâches du CSV ont un timestamp > ticks.\n"
            f"    Tu simules ticks={args.ticks}, donc seulement {nb_in_window}/{nb_total} tâches arrivent.\n"
            "    👉 Si tu veux traiter plus de tâches, augmente --ticks ou remappe les timestamps dans le dataset.\n"
        )

    if args.debug_accounting:
        # Affiche la répartition simple des timestamps
        keys = sorted(workload_idx.keys())
        sample = keys[:10] + (["..."] if len(keys) > 20 else []) + keys[-10:]
        print("\n🔎 DEBUG timestamps keys (sample):", sample)

        # Histogramme grossier par tranches de 50 ticks
        bucket = {}
        for t, tasks in workload_idx.items():
            b = (int(t) // 50) * 50
            bucket[b] = bucket.get(b, 0) + len(tasks)
        for b in sorted(bucket.keys())[:10]:
            print(f"  ts[{b:4d}..{b+49:4d}] -> {bucket[b]} tasks")
        if len(bucket) > 10:
            print("  ...")

    # ============================
    # Modules
    # ============================
    module1 = Module1_LSTMPredictor(model_path=args.lstm_model, device="cpu")

    # ⚠️ Le downscaling se corrige dans Module2_ProactivePlanner (sim_core.py)
    module2 = Module2_HVWPO_Planner(
        min_fog_cpu=args.min_fog_cpu,
        ema_alpha=0.25,
        cooldown_windows=2,
        max_scale_mult=4.0,
    )

    module3 = Module3_Scheduler(dqn_path=args.dqn_model, cpu_threshold_cloud=300, warmup_ticks=15)

    # ✅ On passe aussi le nb de noeuds pour éviter confusion dans les logs si sim_core l’utilise
    n_fog_nodes = 5
    n_cloud_nodes = 2

    setup_state(
        workload_idx,
        module1,
        module2,
        module3,
        W=args.W,
        # si setup_state accepte des kwargs, ça aide à logger/valider
        n_fog_nodes=n_fog_nodes,
        n_cloud_nodes=n_cloud_nodes,
        ticks=args.ticks,
    )

    # ============================
    # Simulator
    # ============================
    simulator = Simulator(tick_duration=1, tick_unit="seconds")

    # Nodes
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
        fog.base_cpu = cpu_scaled  # utile pour downscale si sim_core l’utilise

    cloud_specs = [
        ("Cloud-FR", [48.8566, 2.3522], 1500),
        ("Cloud-BE", [50.4738, 3.8038], 1500),
    ]
    for name, coord, cpu in cloud_specs:
        cloud = EdgeServer(cpu=cpu, memory=200000, disk=200000)
        cloud.name = name
        cloud.coordinates = coord

    # ============================
    # 🌐 TOPOLOGIE RÉSEAU (NetworkX)
    # ============================
    # On définit un graphe pour gérer les latences (Fog=rapide, Cloud=lent)
    topology = nx.Graph()
    core_switch = "Internet-Core"
    topology.add_node(core_switch, type="Switch")

    for server in EdgeServer.all():
        # Latence: Fog=5ms, Cloud=50ms
        is_fog = "Fog" in server.name
        link_latency = 5 if is_fog else 50
        link_bw = 1000 if is_fog else 10000
        topology.add_edge(server, core_switch, latency=link_latency, bandwidth=link_bw)

    simulator.topology = topology

    # ✅ Stopping criterion + algo
    import project.sim_core as sim_core
    simulator.stopping_criterion = lambda sim: (sim_core.CURRENT_T >= int(args.ticks))
    simulator.resource_management_algorithm = proactive_placement_algorithm
    simulator.resource_management_algorithm_parameters = {"simulator": simulator}

    simulator.run_model()

    metrics = get_metrics()

    # ✅ Ajoute un résumé clair : noeuds vs tâches
    print("\n🧠 CLARIFICATION LOGS")
    print(f"Nodes: fog={n_fog_nodes}, cloud={n_cloud_nodes}")
    print("Note: 'placed fog=X cloud=Y' = nb de tâches placées à un tick, PAS le nb de noeuds.\n")

    print_final_stats(metrics)

    if args.output_csv and metrics:
        save_metrics_csv(metrics, args.output_csv)


if __name__ == "__main__":
    main()
