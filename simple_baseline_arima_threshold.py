"""
python simple_baseline_arima_threshold.py --workload "dataset/Pakistan/data/Tuple30K/testset.csv" --topology "topology/fog_cloud_topology.json" --ticks 200 --output_csv "data/results_baseline.csv" --output_plot "data/plot_baseline.png"
>> 
🚀 Démarrage Simulation Baseline (ARIMA + TOPSIS) sur 200 ticks...


BASELINE SCIENTIFIQUE : ARIMA + TOPSIS
======================================

Ce script implémente une approche de référence (baseline) pour comparer avec
l'approche proactive (LSTM + DQN).

Méthodologie :
1.  **Prédiction de Charge (ARIMA)** :
    Utilise un modèle statistique ARIMA (AutoRegressive Integrated Moving Average)
    pour prédire la charge future du système basée sur l'historique récent.
    Si `statsmodels` n'est pas installé, un fallback EWMA est utilisé.

2.  **Planification (Seuils)** :
    Décide du scaling (activation/désactivation de nœuds) en comparant la
    pression prédite à des seuils fixes (Scale Up > 80%, Scale Down < 30%).

3.  **Placement des Tâches (TOPSIS)** :
    Pour chaque tâche, sélectionne le meilleur nœud Fog actif en utilisant
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).
    Critères : Minimiser la charge CPU, la charge RAM et la distance.
    Si aucun nœud Fog n'est apte, la tâche est délestée vers le Cloud.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

# Tentative d'import de statsmodels pour ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️  statsmodels non trouvé. Installation recommandée: pip install statsmodels")
    print("    -> Fallback sur une méthode simplifiée (EWMA/AR) pour la prédiction.")

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
    
    # Format Tuple30K
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
            "name": n["NodeName"],
            "status": n.get("Status", "inactive"),
            "cpu_raw": float(n["MaxCpuFreq"]),
            "ram_raw": float(n.get("MaxBufferSize", 4096)),
            "idle_coef": float(n.get("IdleEnergyCoef", 10)),
            "exe_coef": float(n.get("ExeEnergyCoef", 50)),
            "loc_x": float(n.get("LocX", 0)),
            "loc_y": float(n.get("LocY", 0)),
        }
        if "Fog" in dev:
            fog_nodes.append(node_struct)
        elif "Cloud" in dev:
            cloud_nodes.append(node_struct)
            
    return {"fog_nodes": fog_nodes, "cloud_nodes": cloud_nodes}

# ============================================================
# 2) Module de Prédiction : ARIMA
# ============================================================

class ARIMAPredictor:
    """
    Prédicteur de charge basé sur ARIMA (AutoRegressive Integrated Moving Average).
    Utilise une fenêtre glissante historique pour ajuster le modèle et prédire t+k.
    """
    def __init__(self, order=(1, 0, 0), history_size=50):
        self.order = order
        self.history_size = history_size
        self.history = deque(maxlen=history_size)
        self.model = None
        self.last_pred = 0.0
        self.fallback_ewma = 0.0
        self.alpha = 0.3

    def update(self, value):
        self.history.append(value)
        # Mise à jour EWMA pour fallback
        self.fallback_ewma = self.alpha * value + (1 - self.alpha) * self.fallback_ewma

    def predict(self, steps=1):
        if len(self.history) < 10:
            return self.fallback_ewma, 0.1 # Pas assez de données

        if not STATSMODELS_AVAILABLE:
            return self.fallback_ewma, 0.1 # Fallback si lib manquante

        try:
            # Entraînement rapide sur l'historique courant
            # En simulation réelle, on ne ferait pas fit() à chaque tick, mais ici c'est une baseline
            # On utilise un modèle simple AR(1) ou ARIMA(1,0,0) qui est rapide
            data = list(self.history)
            # Pour éviter les erreurs de convergence sur données constantes
            if np.std(data) < 1e-6:
                return data[-1], 0.0
                
            model = ARIMA(data, order=self.order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            
            pred_val = float(forecast[-1])
            # Estimation simple de l'incertitude (écart-type des résidus)
            residuals = model_fit.resid
            uncertainty = float(np.std(residuals))
            
            return max(0.0, pred_val), uncertainty
            
        except Exception as e:
            # En cas d'erreur d'ajustement (fréquent sur petites séries), fallback
            return self.fallback_ewma, 0.2

# ============================================================
# 3) Module de Décision : TOPSIS (MCDM)
# ============================================================

class TOPSISSelector:
    """
    Sélection de nœud basée sur TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).
    Critères: Charge CPU (Min), Charge RAM (Min), Latence/Distance (Min).
    """
    def __init__(self, weights={'cpu': 0.5, 'ram': 0.2, 'dist': 0.3}):
        self.weights = weights

    def select_best_node(self, candidates, task_profile, current_loads):
        """
        candidates: liste de dicts nœuds
        task_profile: dict {cpu, ram, ...}
        current_loads: dict {node_id: {'cpu': val, 'ram': val}}
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # 1. Construction de la matrice de décision
        # Lignes = Candidats, Colonnes = Critères [CPU_Load, RAM_Load, Distance]
        matrix = []
        valid_candidates = []

        for node in candidates:
            node_id = node['id']
            load = current_loads.get(node_id, {'cpu': 0, 'ram': 0})
            
            # Critère 1: CPU Load projeté (si on ajoute la tâche)
            cpu_cap = max(1.0, node['cpu_raw'])
            proj_cpu_load = (load['cpu'] + task_profile['cpu_demand']) / cpu_cap
            
            # Critère 2: RAM Load projeté
            ram_cap = max(1.0, node['ram_raw'])
            proj_ram_load = (load['ram'] + task_profile['ram_demand']) / ram_cap
            
            # Critère 3: Distance (Approximation Latence) - ici simplifiée par distance euclidienne inverse ou 0
            # On suppose que task arrive à (0,0) pour simplifier ou on prend la loc du noeud
            dist = math.sqrt(node['loc_x']**2 + node['loc_y']**2)
            
            # On rejette si surcharge immédiate (>100%)
            if proj_cpu_load > 1.0 or proj_ram_load > 1.0:
                continue

            matrix.append([proj_cpu_load, proj_ram_load, dist])
            valid_candidates.append(node)

        if not valid_candidates:
            # Si tous surchargés, on retourne le moins chargé (fallback)
            return min(candidates, key=lambda n: current_loads.get(n['id'], {'cpu':0})['cpu'])

        # 2. Normalisation Vectorielle
        np_matrix = np.array(matrix)
        norm_matrix = np_matrix / (np.sqrt((np_matrix**2).sum(axis=0)) + 1e-9)

        # 3. Pondération
        w = np.array([self.weights['cpu'], self.weights['ram'], self.weights['dist']])
        weighted_matrix = norm_matrix * w

        # 4. Solutions Idéales (On veut MINIMISER tous les critères ici: charge, charge, distance)
        # Ideal Positive (A+): Min de chaque colonne
        # Ideal Negative (A-): Max de chaque colonne
        ideal_best = weighted_matrix.min(axis=0)
        ideal_worst = weighted_matrix.max(axis=0)

        # 5. Distances Euclidiennes
        dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))

        # 6. Score de proximité relative (C*)
        # C = D- / (D+ + D-)
        # Plus C est proche de 1, plus on est proche de l'idéal
        scores = dist_worst / (dist_best + dist_worst + 1e-9)

        # 7. Classement
        best_idx = np.argmax(scores)
        return valid_candidates[best_idx]

# ============================================================
# 4) Moteur de Simulation Baseline
# ============================================================

class BaselineARIMA_TOPSIS:
    def __init__(self, topo, ticks=300, cpu_scale=1000.0):
        self.topo = topo
        self.ticks = ticks
        self.cpu_scale = cpu_scale
        
        # Initialisation Nœuds
        self.fogs = []
        for fn in topo["fog_nodes"]:
            self.fogs.append({
                **fn,
                "capacity": max(1.0, fn["cpu_raw"] / self.cpu_scale),
                "active": (fn["status"] == "active"),
                "used_cpu": 0.0,
                "used_ram": 0.0,
                "tasks": [] # (duration, cpu, ram)
            })
            
        self.clouds = []
        for cn in topo["cloud_nodes"]:
            self.clouds.append({
                **cn,
                "capacity": 999999.0, # Infini
                "used_cpu": 0.0,
                "tasks": []
            })
            
        # Composants Intelligence
        self.predictor = ARIMAPredictor(order=(1,0,0), history_size=30)
        self.selector = TOPSISSelector(weights={'cpu': 0.6, 'ram': 0.2, 'dist': 0.2})
        
        # Paramètres de contrôle
        self.target_util = 0.70
        self.scale_up_threshold = 0.80
        self.scale_down_threshold = 0.40
        self.offload_threshold = 0.90
        
        self.metrics = []
        self.total_scale_up = 0
        self.total_scale_down = 0
        self.total_energy_joules = 0.0

    def _active_fogs(self):
        return [f for f in self.fogs if f["active"]]

    def _get_pool_stats(self):
        active = self._active_fogs()
        total_cap = sum(f["capacity"] for f in active)
        total_used = sum(f["used_cpu"] for f in active)
        pressure = (total_used / total_cap) if total_cap > 0 else 0.0
        return total_cap, total_used, pressure

    def run(self, tasks):
        # Indexer les tâches par temps
        tasks_by_time = defaultdict(list)
        for task_item in tasks:
            if task_item["timestamp"] < self.ticks:
                tasks_by_time[task_item["timestamp"]].append(task_item)
        
        print(f"🚀 Démarrage Simulation Baseline (ARIMA + TOPSIS) sur {self.ticks} ticks...")
        
        pred_pressure, pred_unc = 0.0, 0.0
        
        for t in range(self.ticks):
            # 1. Libération des ressources (fin de tâches)
            for node in self.fogs + self.clouds:
                remaining_tasks = []
                freed_cpu = 0
                freed_ram = 0
                for dur, cpu, ram in node["tasks"]:
                    if dur > 1:
                        remaining_tasks.append((dur - 1, cpu, ram))
                    else:
                        freed_cpu += cpu
                        freed_ram += ram
                node["tasks"] = remaining_tasks
                node["used_cpu"] = max(0.0, node["used_cpu"] - freed_cpu)
                node["used_ram"] = max(0.0, node.get("used_ram", 0) - freed_ram)

            # 2. Monitoring & Prédiction (ARIMA)
            cap, used, pressure = self._get_pool_stats()
            
            # Mise à jour ARIMA tous les 5 ticks pour performance
            if t % 5 == 0:
                self.predictor.update(pressure)
                pred_pressure, pred_unc = self.predictor.predict(steps=5)
            
            # 3. Scaling Proactif (Basé sur prédiction ARIMA)
            scale_decision = "none"
            if pred_pressure > self.scale_up_threshold:
                # Trouver un nœud inactif
                inactive = [f for f in self.fogs if not f["active"]]
                if inactive:
                    # Activer le plus puissant
                    cand = sorted(inactive, key=lambda x: -x["capacity"])[0]
                    cand["active"] = True
                    scale_decision = "up"
                    self.total_scale_up += 1
            
            elif pred_pressure < self.scale_down_threshold:
                # Trouver un nœud actif vide
                active = self._active_fogs()
                # CORRECTION: On autorise la désactivation si le noeud est très peu chargé (< 5%), pas seulement vide
                candidates = [f for f in active if (f["used_cpu"] / f["capacity"]) < 0.05]
                if len(active) > 1 and candidates:
                    cand = sorted(candidates, key=lambda x: x["capacity"])[0]
                    cand["active"] = False
                    scale_decision = "down"
                    self.total_scale_down += 1

            # 4. Allocation des tâches (TOPSIS)
            new_tasks = tasks_by_time.get(t, [])
            tasks_placed_fog = 0
            tasks_placed_cloud = 0
            
            # Préparer l'état de charge pour TOPSIS
            current_loads = {f["id"]: {'cpu': f["used_cpu"], 'ram': f.get("used_ram", 0)} for f in self._active_fogs()}
            
            for task in new_tasks:
                # Stratégie d'Offloading
                # Si pression actuelle OU prédite est critique -> Cloud
                should_offload = (pressure > self.offload_threshold) or (pred_pressure > 0.95)
                
                target_node = None
                decision = "Cloud"
                
                if not should_offload:
                    # Essayer de trouver le meilleur Fog via TOPSIS
                    active_fogs = self._active_fogs()
                    best_fog = self.selector.select_best_node(active_fogs, task, current_loads)
                    
                    if best_fog:
                        # Vérifier capacité stricte
                        if (best_fog["used_cpu"] + task["cpu_demand"] <= best_fog["capacity"]):
                            target_node = best_fog
                            decision = "Fog"
                
                if decision == "Fog" and target_node:
                    target_node["used_cpu"] += task["cpu_demand"]
                    target_node["used_ram"] = target_node.get("used_ram", 0) + task["ram_demand"]
                    target_node["tasks"].append((task["duration"], task["cpu_demand"], task["ram_demand"]))
                    # Mettre à jour load pour la prochaine tâche du même tick
                    current_loads[target_node["id"]]['cpu'] += task["cpu_demand"]
                    tasks_placed_fog += 1
                else:
                    # Cloud (Round Robin simple ou premier dispo)
                    target_node = self.clouds[0] # Simplification: un seul gros cloud
                    target_node["used_cpu"] += task["cpu_demand"]
                    target_node["tasks"].append((task["duration"], task["cpu_demand"], task["ram_demand"]))
                    tasks_placed_cloud += 1

            # 5. Collecte Métriques
            # Calcul énergie (simplifié)
            energy_tick = 0.0
            for f in self.fogs:
                if f["active"]:
                    util = min(1.0, f["used_cpu"] / f["capacity"]) if f["capacity"] > 0 else 0
                    energy_tick += f["idle_coef"] + (f["exe_coef"] * util)
            self.total_energy_joules += energy_tick

            # Collecte des stats par nœud pour compatibilité plot
            per_node_stats = {}
            for f in self.fogs:
                util = min(1.0, f["used_cpu"] / f["capacity"]) if f["capacity"] > 0 else 0
                load = f["used_cpu"]
                is_active = f["active"]
                
                power = 0.0
                if is_active:
                    power = f["idle_coef"] + (f["exe_coef"] * util)
                    
                safe_name = f["name"].replace("-", "_").replace(" ", "")
                per_node_stats[f"fog_{safe_name}_p"] = util
                per_node_stats[f"fog_{safe_name}_load"] = load
                per_node_stats[f"fog_{safe_name}_power"] = power

            for c in self.clouds:
                safe_name = c["name"].replace("-", "_").replace(" ", "")
                load = c["used_cpu"]
                per_node_stats[f"cloud_{safe_name}_load"] = load

            metric_row = {
                "t": t,
                "active_cpu_fog": used,
                "fog_capacity": cap,
                "pressure": pressure,
                "predicted_pressure": pred_pressure,
                "prediction_uncertainty": pred_unc,
                "tasks_placed_fog": tasks_placed_fog,
                "tasks_placed_cloud": tasks_placed_cloud,
                "scale_decision": scale_decision,
                "scale_up_total": self.total_scale_up,
                "scale_down_total": self.total_scale_down,
                "energy_joules_cumul": self.total_energy_joules,
                "offload_ratio": tasks_placed_cloud / (tasks_placed_fog + tasks_placed_cloud) if (tasks_placed_fog + tasks_placed_cloud) > 0 else 0,
            }
            metric_row.update(per_node_stats)
            self.metrics.append(metric_row)
            
            if t % 50 == 0:
                print(f"   Tick {t}: Pression={pressure:.2f} (Pred={pred_pressure:.2f}) | Fog={tasks_placed_fog} Cloud={tasks_placed_cloud}")

        return pd.DataFrame(self.metrics)

# ============================================================
# 5) Plotting & Main
# ============================================================
def plot_load_capacity(ax, df):
    """Graphique charge vs capacité."""
    ax.plot(df['t'], df['active_cpu_fog'], label='Charge Fog (CPU)', 
           color='dodgerblue', linewidth=2.5, alpha=0.8)
    ax.plot(df['t'], df['fog_capacity'], label='Capacité Fog (CPU)', 
           color='crimson', linestyle='--', linewidth=2)
    
    ax.fill_between(df['t'], df['active_cpu_fog'], df['fog_capacity'],
                   where=(df['active_cpu_fog'] > df['fog_capacity']),
                   color='red', alpha=0.2, label='Dépassement')
    
    for idx, row in df.iterrows():
        if row['scale_decision'] == 'up':
            ax.annotate('↑', xy=(row['t'], row['fog_capacity']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', color='green', fontsize=12, fontweight='bold')
        elif row['scale_decision'] == 'down':
            ax.annotate('↓', xy=(row['t'], row['fog_capacity']),
                       xytext=(0, -15), textcoords='offset points',
                       ha='center', color='purple', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Unités CPU", fontsize=12)
    ax.set_title("① CHARGE vs CAPACITÉ FOG", fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

def plot_pressure_predictions(ax, df):
    """Graphique pression et prédictions."""
    ax.plot(df['t'], df['pressure'], label='Pression Réelle', 
           color='darkorange', linewidth=3, alpha=0.8)
    
    ax.plot(df['t'], df['predicted_pressure'], label='Prédiction ARIMA', 
           color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.fill_between(df['t'], 
                   df['predicted_pressure'] - df['prediction_uncertainty'],
                   df['predicted_pressure'] + df['prediction_uncertainty'],
                   color='gray', alpha=0.2, label='Incertitude (±)')
    
    ax.axhline(y=1.0, color='red', linestyle='-', linewidth=1.5, alpha=0.5, 
              label='Seuil Surcharge (100%)')
    
    ax.set_ylabel("Pression (Utilisation)", fontsize=12)
    ax.set_title("② PRESSION RÉELLE vs PRÉDICTIONS", fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(-0.05, min(2.5, df['pressure'].max() * 1.2))

def plot_fog_nodes_heatmap(ax, df):
    """Heatmap de la pression par nœud Fog individuel."""
    fog_cols = [c for c in df.columns if c.startswith('fog_') and c.endswith('_p')]
    
    if not fog_cols:
        ax.text(0.5, 0.5, "Pas de données détaillées par nœud", ha='center', va='center')
        return

    labels = [c.replace('fog_', '').replace('_p', '') for c in fog_cols]
    data = df[fog_cols].T.values
    
    im = ax.imshow(data, aspect='auto', cmap='plasma', vmin=0, vmax=1.2, interpolation='nearest')
    
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Temps (ticks)", fontsize=10)
    ax.set_title("③ UTILISATION DÉTAILLÉE PAR NŒUD FOG (Heatmap)", fontsize=12, fontweight='bold', pad=10)
    
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label("Pression (0-1+)", fontsize=9)

def plot_energy_and_cloud_stats(ax, df):
    """Graphique combiné : Énergie par Fog et Charge par Cloud."""
    power_cols = [c for c in df.columns if c.startswith('fog_') and c.endswith('_power')]
    energy_sums = []
    fog_labels = []
    
    if power_cols:
        energy_sums = df[power_cols].sum().values
        fog_labels = [c.replace('fog_', '').replace('_power', '') for c in power_cols]

    cloud_cols = [c for c in df.columns if c.startswith('cloud_') and c.endswith('_load')]
    cloud_sums = []
    cloud_labels = []
    
    if cloud_cols:
        cloud_sums = df[cloud_cols].mean().values
        cloud_labels = [c.replace('cloud_', '').replace('_load', '') for c in cloud_cols]

    ax.axis('off')
    
    if len(energy_sums) > 0:
        ax1 = ax.inset_axes([0, 0, 0.48, 1])
        x_pos = range(len(fog_labels))
        ax1.bar(x_pos, energy_sums, color='orange', alpha=0.7, edgecolor='darkorange')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(fog_labels, rotation=45, ha='right', fontsize=8)
        ax1.set_title("Consommation Énergétique Totale (Joules)", fontsize=10, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

    if len(cloud_sums) > 0:
        ax2 = ax.inset_axes([0.52, 0, 0.48, 1])
        x_pos2 = range(len(cloud_labels))
        ax2.bar(x_pos2, cloud_sums, color='skyblue', alpha=0.7, edgecolor='blue')
        ax2.set_xticks(x_pos2)
        ax2.set_xticklabels(cloud_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_title("Charge Moyenne par Nœud Cloud (CPU)", fontsize=10, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

def plot_simulation_results(csv_path: str, output_path: str):
    """Génère un graphique de synthèse pour la baseline."""
    if not os.path.exists(csv_path):
        print(f"Erreur: Fichier de résultats introuvable: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    fig = plt.figure(figsize=(20, 22))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 3, 3, 2], hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])

    workload_name = os.path.basename(csv_path).replace('.csv', '').replace('results_', '')
    fig.suptitle(f"ANALYSE BASELINE (ARIMA+TOPSIS) - {workload_name.upper()}", 
                 fontsize=22, y=0.98, fontweight='bold')

    # 1. Charge vs. Capacité
    plot_load_capacity(ax1, df)
    
    # 2. Pression et Prédictions
    plot_pressure_predictions(ax2, df)

    # 3. Heatmap Utilisation par Nœud Fog
    plot_fog_nodes_heatmap(ax3, df)

    # 4. Énergie et Cloud
    plot_energy_and_cloud_stats(ax4, df)
    ax4.set_title("④ ÉNERGIE & CHARGE CLOUD", fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Graphique sauvegardé dans: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Baseline Scientifique: ARIMA + TOPSIS")
    parser.add_argument("--workload", required=True, help="Chemin CSV workload")
    parser.add_argument("--topology", required=True, help="Chemin JSON topologie")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--output_csv", default="baseline_results.csv")
    parser.add_argument("--output_plot", default="baseline_plot.png")
    args = parser.parse_args()

    # Chargement
    tasks = load_workload_csv(args.workload)
    topo = load_topology_json(args.topology)
    
    # Simulation
    sim = BaselineARIMA_TOPSIS(topo, ticks=args.ticks)
    df = sim.run(tasks)
    
    # Sauvegarde
    df.to_csv(args.output_csv, index=False)
    print(f"💾 Résultats sauvegardés : {args.output_csv}")
    
    # Stats finales
    last_row = df.iloc[-1]
    avg_p = df['pressure'].mean()
    total_e_kj = last_row['energy_joules_cumul'] / 1000
    total_fog = df['tasks_placed_fog'].sum()
    total_cloud = df['tasks_placed_cloud'].sum()
    offload_ratio = total_cloud / (total_fog + total_cloud) if (total_fog + total_cloud) > 0 else 0
    scale_up = int(last_row['scale_up_total'])
    scale_down = int(last_row['scale_down_total'])

    print("\n" + "="*50)
    print("RÉSULTATS BASELINE (ARIMA + TOPSIS)")
    print("="*50)
    print(f"Pression Moyenne : {avg_p:.2%}")
    print(f"Énergie Totale   : {total_e_kj:.2f} kJ")
    print(f"Taux Offloading  : {offload_ratio:.2%}")
    print(f"Nb Scalings      : {scale_up} up / {scale_down} down")
    print("="*50)
    
    # Plot
    plot_simulation_results(args.output_csv, args.output_plot)

if __name__ == "__main__":
    main()