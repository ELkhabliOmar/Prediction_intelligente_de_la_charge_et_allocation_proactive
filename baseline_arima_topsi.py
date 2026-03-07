# simple_baseline_ewma_threshold.py
# RENAMED INTERNALLY TO: baseline_arima_topsis.py
# Implémentation Baseline Scientifique: Prédiction ARIMA + Décision TOPSIS

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
        self.scale_down_threshold = 0.30
        self.offload_threshold = 0.90
        
        self.metrics = []
        self.total_scaling = 0

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
        for task in tasks:
            if t["timestamp"] < self.ticks:
                tasks_by_time[t["timestamp"]].append(t)
        
        print(f"🚀 Démarrage Simulation Baseline (ARIMA + TOPSIS) sur {self.ticks} ticks...")
        
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
                    self.total_scaling += 1
            
            elif pred_pressure < self.scale_down_threshold:
                # Trouver un nœud actif vide
                active = self._get_active_fogs()
                candidates = [f for f in active if len(f["tasks"]) == 0]
                if len(active) > 1 and candidates:
                    cand = sorted(candidates, key=lambda x: x["capacity"])[0]
                    cand["active"] = False
                    scale_decision = "down"
                    self.total_scaling += 1

            # 4. Allocation des tâches (TOPSIS)
            new_tasks = tasks_by_time.get(t, [])
            fog_count = 0
            cloud_count = 0
            
            # Préparer l'état de charge pour TOPSIS
            current_loads = {f["id"]: {'cpu': f["used_cpu"], 'ram': f.get("used_ram", 0)} for f in self._get_active_fogs()}
            
            for task in new_tasks:
                # Stratégie d'Offloading
                # Si pression actuelle OU prédite est critique -> Cloud
                should_offload = (pressure > self.offload_threshold) or (pred_pressure > 0.95)
                
                target_node = None
                decision = "Cloud"
                
                if not should_offload:
                    # Essayer de trouver le meilleur Fog via TOPSIS
                    active_fogs = self._get_active_fogs()
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
                    fog_count += 1
                else:
                    # Cloud (Round Robin simple ou premier dispo)
                    target_node = self.clouds[0] # Simplification: un seul gros cloud
                    target_node["used_cpu"] += task["cpu_demand"]
                    target_node["tasks"].append((task["duration"], task["cpu_demand"], task["ram_demand"]))
                    cloud_count += 1

            # 5. Collecte Métriques
            # Calcul énergie (simplifié)
            energy_tick = 0.0
            for f in self.fogs:
                if f["active"]:
                    util = min(1.0, f["used_cpu"] / f["capacity"]) if f["capacity"] > 0 else 0
                    energy_tick += f["idle_coef"] + (f["exe_coef"] * util)
            
            self.metrics.append({
                "t": t,
                "pressure": pressure,
                "predicted_pressure": pred_pressure,
                "active_cpu_fog": used,
                "fog_capacity": cap,
                "tasks_fog": fog_count,
                "tasks_cloud": cloud_count,
                "offload_ratio": cloud_count / (fog_count + cloud_count) if (fog_count + cloud_count) > 0 else 0,
                "scale_decision": scale_decision,
                "energy": energy_tick
            })
            
            if t % 50 == 0:
                print(f"   Tick {t}: Pression={pressure:.2f} (Pred={pred_pressure:.2f}) | Fog={fog_count} Cloud={cloud_count}")

        return pd.DataFrame(self.metrics)

# ============================================================
# 5) Plotting & Main
# ============================================================

def plot_results(df, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # 1. Pression & Prédiction
    ax = axes[0]
    ax.plot(df['t'], df['pressure'], label='Pression Réelle', color='blue', linewidth=2)
    ax.plot(df['t'], df['predicted_pressure'], label='Prédiction ARIMA', color='orange', linestyle='--')
    ax.axhline(0.8, color='red', linestyle=':', label='Seuil Scale Up')
    ax.axhline(0.3, color='green', linestyle=':', label='Seuil Scale Down')
    ax.set_ylabel('Pression Fog')
    ax.set_title('Performance Prédiction ARIMA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Allocation & Scaling
    ax = axes[1]
    ax.bar(df['t'], df['tasks_fog'], label='Tâches Fog (TOPSIS)', color='green', alpha=0.6)
    ax.bar(df['t'], df['tasks_cloud'], bottom=df['tasks_fog'], label='Tâches Cloud', color='gray', alpha=0.6)
    
    # Marquer les scalings
    scale_up = df[df['scale_decision'] == 'up']
    scale_down = df[df['scale_decision'] == 'down']
    ax.scatter(scale_up['t'], [0]*len(scale_up), marker='^', color='red', s=100, label='Scale Up', zorder=5)
    ax.scatter(scale_down['t'], [0]*len(scale_down), marker='v', color='blue', s=100, label='Scale Down', zorder=5)
    
    ax.set_ylabel('Nombre de Tâches')
    ax.set_title('Allocation des Tâches & Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Énergie
    ax = axes[2]
    ax.plot(df['t'], df['energy'].cumsum() / 1000, label='Énergie Cumulée (kJ)', color='purple')
    ax.set_ylabel('Énergie (kJ)')
    ax.set_xlabel('Temps (ticks)')
    ax.set_title('Consommation Énergétique')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📊 Graphique généré : {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Baseline Scientifique: ARIMA + TOPSIS")
    parser.add_argument("--workload", required=True, help="Chemin CSV workload")
    parser.add_argument("--topology", required=True, help="Chemin JSON topologie")
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--output_csv", default="baseline_results.csv")
    parser.add_argument("--output_plot", default="baseline_plot.png")
    args = ap.parse_args()

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
    avg_p = df['pressure'].mean()
    total_e = df['energy'].sum() / 1000
    offload = df['offload_ratio'].mean()
    print("\n" + "="*50)
    print("RÉSULTATS BASELINE (ARIMA + TOPSIS)")
    print("="*50)
    print(f"Pression Moyenne : {avg_p:.2%}")
    print(f"Énergie Totale   : {total_e:.2f} kJ")
    print(f"Taux Offloading  : {offload:.2%}")
    print(f"Nb Scalings      : {sim.total_scaling}")
    print("="*50)
    
    # Plot
    plot_results(df, args.output_plot)

if __name__ == "__main__":
    main()