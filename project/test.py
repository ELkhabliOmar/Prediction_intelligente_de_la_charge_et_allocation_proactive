# project/test.py
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

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
    Module2_ProactivePlanner,
    Module3_Scheduler,
)

from project.ui_utils import banner, print_table, print_final_stats, save_metrics_csv

try:
    from edge_sim_py import Simulator, EdgeServer
    EDGE_SIM_AVAILABLE = True
except Exception:
    EDGE_SIM_AVAILABLE = False


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
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    banner("SIMULATION MULTI-FOG / MULTI-CLOUD", "Affichage tableau + MAPE")

    if not os.path.exists(args.workload):
        raise FileNotFoundError(f"Workload introuvable: {args.workload}")

    workload_idx = load_workload_indexed(args.workload)
    nb_tasks = sum(len(v) for v in workload_idx.values())

    print_table(
        ["Param", "Value"],
        [
            ["Workload", args.workload],
            ["Tasks", str(nb_tasks)],
            ["Ticks", str(args.ticks)],
            ["W (MAPE)", str(args.W)],
            ["Target util", str(args.target_util)],
            ["Output CSV", args.output_csv],
        ],
        title="⚙️ Paramètres",
    )

    module1 = Module1_LSTMPredictor(model_path=args.lstm_model, device="cpu")
    module2 = Module2_ProactivePlanner(
        target_util=args.target_util,
        min_fog_cpu=args.min_fog_cpu,
        ema_alpha=0.25,
        cooldown_windows=2,
        max_scale_mult=4.0,
    )
    module3 = Module3_Scheduler(dqn_path=args.dqn_model, cpu_threshold_cloud=300, warmup_ticks=15)

    setup_state(workload_idx, module1, module2, module3, W=args.W)

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
        fog.base_cpu = cpu_scaled

    cloud_specs = [
        ("Cloud-FR", [48.8566, 2.3522], 1500),
        ("Cloud-BE", [50.4738, 3.8038], 1500),
    ]
    for name, coord, cpu in cloud_specs:
        cloud = EdgeServer(cpu=cpu, memory=200000, disk=200000)
        cloud.name = name
        cloud.coordinates = coord

    import project.sim_core as sim_core
    simulator.stopping_criterion = lambda sim: (sim_core.CURRENT_T >= int(args.ticks))
    simulator.resource_management_algorithm = proactive_placement_algorithm
    simulator.resource_management_algorithm_parameters = {"simulator": simulator}

    simulator.run_model()

    metrics = get_metrics()
    print_final_stats(metrics)

    if args.output_csv and metrics:
        save_metrics_csv(metrics, args.output_csv)


if __name__ == "__main__":
    main()
