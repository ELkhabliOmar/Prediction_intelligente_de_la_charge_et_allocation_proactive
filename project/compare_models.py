"""
COMPARAISON D'APPROCHES - FOG/CLOUD SIMULATION
================================================================================
Ce script compare trois stratégies d'allocation de ressources :
1. PROACTIVE (DQN + LSTM) : Votre modèle.
2. REACTIVE (Threshold)   : Baseline standard (sans IA, règles fixes).
3. RANDOM                 : Baseline naïve (choix aléatoire).

Il génère un graphique comparatif (Bar Chart) des métriques clés :
- Pression Moyenne
- Consommation Énergétique
- Taux de Surcharge (SLA Violations)
- Coût Estimé
================================================================================
"""
import argparse
import os
import sys
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

# Ajout du chemin racine
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_WORKLOAD, DEFAULT_LSTM, DEFAULT_DQN, DEFAULT_RESULTS
import project.sim_core as sim_core
from project.sim_core import (
    setup_state,
    proactive_placement_algorithm,
    get_metrics,
    load_workload_indexed,
    Module1_LSTMPredictor,
    Module2_HVWPO_Planner,
    Module3_Scheduler,
    EdgeServer
)
from project.ui_utils import banner, color

try:
    from edge_sim_py import Simulator
    EDGE_SIM_AVAILABLE = True
except ImportError:
    EDGE_SIM_AVAILABLE = False

# --- 1. Classes Mock pour les Baselines ---

class RandomScheduler:
    """Stratégie Naïve : Choisit Fog ou Cloud au hasard."""
    def __init__(self):
        self.use_dqn = False
    
    def decide(self, task_cpu, task_ram, pressure, fog_cpu, offload_ratio, t):
        # 50/50 chance, ou biaisé légèrement vers Fog
        if random.random() < 0.5:
            return "Fog", False
        return "Cloud", False

class ReactivePredictor:
    """Stratégie Réactive : La 'prédiction' est juste la valeur actuelle (pas d'anticipation)."""
    def predict(self, pressure_history):
        current = pressure_history[-1] if pressure_history else 0.0
        # On retourne la valeur actuelle pour tous les horizons -> Le système réagit au présent
        return {h: {"prediction": current, "uncertainty": 0.0, "used_fallback": True} for h in [5, 15, 30]}

# --- 2. Fonction d'exécution d'un scénario ---

def run_scenario(name, args, workload_idx, topo_data, strategy="proactive"):
    print(f"\n🚀 Lancement du scénario : {color(name, 'cyan', bold=True)}")
    
    # 1. Configuration des Modules selon la stratégie
    if strategy == "proactive":
        # Votre approche complète
        module1 = Module1_LSTMPredictor(model_path=args.lstm_model, device="cpu")
        module3 = Module3_Scheduler(dqn_path=args.dqn_model, cpu_threshold_cloud=300)
        planner_alpha = 0.25 # EMA normal
        
    elif strategy == "reactive":
        # Pas de LSTM (prédiction = valeur actuelle), Pas de DQN (Baseline heuristique)
        module1 = ReactivePredictor()
        module3 = Module3_Scheduler(dqn_path=None) # Force le mode baseline
        planner_alpha = 1.0 # Pas de lissage, réaction immédiate
        
    elif strategy == "random":
        # Pas de LSTM, Scheduler Aléatoire
        module1 = ReactivePredictor()
        module3 = RandomScheduler()
        planner_alpha = 1.0

    # Module 2 (Planner) est configuré différemment selon le mode
    module2 = Module2_HVWPO_Planner(
        target_util=args.target_util,
        down_threshold=args.down_threshold,
        min_fog_cpu=args.min_fog_cpu,
        ema_alpha=planner_alpha, 
        cooldown_windows=(2 if strategy == "proactive" else 0) # Pas de cooldown en réactif
    )

    # 2. Reset de l'état global de sim_core
    setup_state(workload_idx, module1, module2, module3, W=args.W)

    # 3. Initialisation Simulateur
    simulator = Simulator(tick_duration=1, tick_unit="seconds")
    
    # Chargement Topologie (Simplifié pour le script de comparaison)
    # On recrée les objets EdgeServer à chaque run pour éviter les résidus d'état
    EdgeServer.all().clear() # Nettoyage interne edge-sim-py si nécessaire
    
    nodes_map = {}
    for node_data in topo_data.get("Nodes", []):
        name_n = node_data.get("NodeName", f"Node-{node_data.get('NodeId')}")
        cpu_freq = int(node_data.get("MaxCpuFreq", 1000))
        base_cpu = int((cpu_freq / 1000))
        server = EdgeServer(cpu=base_cpu, memory=4096, disk=100000)
        server.name = name_n
        server.base_cpu = base_cpu
        server.status = node_data.get("Status", "active")
        server.device_type = node_data.get("DeviceType", "Edge")
        nodes_map[node_data["NodeId"]] = server

    # Algorithme
    import project.sim_core as sim_core
    simulator.stopping_criterion = lambda sim: (sim_core.CURRENT_T >= int(args.ticks))
    simulator.resource_management_algorithm = proactive_placement_algorithm
    simulator.resource_management_algorithm_parameters = {"simulator": simulator}

    # 4. Exécution
    simulator.run_model()
    
    # 5. Collecte des métriques
    raw_metrics = get_metrics()
    
    # Calcul des agrégats pour la comparaison
    df = pd.DataFrame(raw_metrics)
    summary = {
        "name": name,
        "avg_pressure": df["pressure"].mean(),
        "overload_rate": (df["pressure"] > 1.0).mean() * 100, # %
        "total_energy": df["energy_joules_cumul"].iloc[-1] / 1000.0 if not df.empty else 0, # kJ
        "avg_offload": df["offload_ratio"].mean() * 100, # %
        "fog_tasks": df["tasks_placed_fog"].sum(),
        "cloud_tasks": df["tasks_placed_cloud"].sum(),
        "scaling_ops": df["scale_up_total"].iloc[-1] + df["scale_down_total"].iloc[-1] if not df.empty else 0
    }
    
    print(f"   ✅ Terminé. Pression Moy: {summary['avg_pressure']:.2f}, Énergie: {summary['total_energy']:.2f} kJ")
    return summary

# --- 3. Génération des Graphiques ---

def plot_comparison(results, output_file):
    """Génère un graphique comparatif (Bar Chart)."""
    df = pd.DataFrame(results)
    df.set_index("name", inplace=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparaison des Stratégies Fog/Cloud", fontsize=16, fontweight='bold')
    
    # Couleurs
    colors = ['#2ecc71', '#3498db', '#95a5a6'] # Vert (Pro), Bleu (React), Gris (Rand)
    colors = ['#95a5a6', '#3498db', '#2ecc71'] # Gris (Rand), Bleu (React), Vert (Pro)
    
    # 1. Pression Moyenne
    ax1 = axes[0, 0]
    df["avg_pressure"].plot(kind="bar", ax=ax1, color=colors, alpha=0.8)
    ax1.set_title("Pression Moyenne du Fog (Cible ~0.7)", fontsize=12)
    ax1.set_ylabel("Utilisation (0-1)")
    ax1.axhline(0.7, color='red', linestyle='--', alpha=0.5, label="Cible")
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Taux de Surcharge (SLA Violations)
    ax2 = axes[0, 1]
    df["overload_rate"].plot(kind="bar", ax=ax2, color=colors, alpha=0.8)
    ax2.set_title("Taux de Surcharge (>100% Capacité)", fontsize=12)
    ax2.set_ylabel("Pourcentage du temps (%)")
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Énergie Totale
    ax3 = axes[1, 0]
    df["total_energy"].plot(kind="bar", ax=ax3, color=colors, alpha=0.8)
    ax3.set_title("Consommation Énergétique Totale", fontsize=12)
    ax3.set_ylabel("Énergie (kJ)")
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Répartition des Tâches (Stacked)
    ax4 = axes[1, 1]
    df[["fog_tasks", "cloud_tasks"]].plot(kind="bar", stacked=True, ax=ax4, 
                                         color=['#27ae60', '#e74c3c'], alpha=0.8)
    ax4.set_title("Répartition des Tâches (Fog vs Cloud)", fontsize=12)
    ax4.set_ylabel("Nombre de Tâches")
    ax4.legend(["Fog", "Cloud"])
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    print(f"\n📊 Graphique comparatif sauvegardé : {output_file}")

# --- 4. Main ---

def main():
    parser = argparse.ArgumentParser(description="Comparaison de modèles Fog Computing")
    parser.add_argument("--workload", default=DEFAULT_WORKLOAD)
    parser.add_argument("--lstm_model", default=DEFAULT_LSTM)
    parser.add_argument("--dqn_model", default=DEFAULT_DQN)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--W", type=int, default=10)
    parser.add_argument("--target_util", type=float, default=0.70)
    parser.add_argument("--down_threshold", type=float, default=0.30)
    parser.add_argument("--min_fog_cpu", type=int, default=30)
    parser.add_argument("--output_plot", default="comparison_results.png")
    args = parser.parse_args()

    if not EDGE_SIM_AVAILABLE:
        print("❌ edge-sim-py manquant.")
        return

    banner("COMPARAISON DE MODÈLES", "Proactive vs Reactive vs Random")
    
    # Chargement Workload
    if not os.path.exists(args.workload):
        print(f"❌ Workload introuvable: {args.workload}")
        return
    workload_idx = load_workload_indexed(args.workload)
    
    # Chargement Topologie
    topo_path = ROOT_DIR / "topology" / "fog_cloud_topology.json"
    with open(topo_path, "r") as f:
        topo_data = json.load(f)

    # --- Exécution des 3 Scénarios ---
    results = []
    
    # 1. Random (Baseline faible)
    # Scénario 1: Random (Baseline faible)
    res_rand = run_scenario("Random", args, workload_idx, topo_data, strategy="random")
    results.append(res_rand)
    
    # 2. Reactive (Baseline standard)
    # Scénario 2: Reactive (Baseline standard)
    res_react = run_scenario("Reactive (Threshold)", args, workload_idx, topo_data, strategy="reactive")
    results.append(res_react)
    
    # 3. Proactive (Notre modèle)
    # Scénario 3: Proactive (Notre modèle)
    res_pro = run_scenario("Proactive (DQN+LSTM)", args, workload_idx, topo_data, strategy="proactive")
    results.append(res_pro)
    
    # --- Affichage Tableau Récapitulatif ---
    print("\n" + "="*80)
    print(f"{'STRATÉGIE':<25} | {'PRESSION':<10} | {'SURCHARGE':<10} | {'ÉNERGIE (kJ)':<15} | {'SCALING':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25} | {r['avg_pressure']:<10.2f} | {r['overload_rate']:<10.1f}% | {r['total_energy']:<15.2f} | {r['scaling_ops']:<10}")
    print("="*80)
    
    # --- Génération Graphique ---
    plot_comparison(results, args.output_plot)

if __name__ == "__main__":
    main()
