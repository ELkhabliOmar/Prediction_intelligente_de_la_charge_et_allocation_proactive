# plot_results.py
# Usage:
#   python plot_results.py --input data/results_up.csv --output images/results_up.png

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_simulation_results(csv_path: str, output_path: str):
    """
    Génère un graphique professionnel à partir des métriques de simulation.
    """
    if not os.path.exists(csv_path):
        print(f"Erreur: Fichier de résultats introuvable: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # --- Création de la figure et des sous-graphiques ---
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1,
        figsize=(16, 22),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 3, 2, 2]}
    )
    fig.suptitle(f"Analyse de la Simulation - {os.path.basename(csv_path)}", fontsize=20, y=0.95)

    # --- Graphique 1: Charge vs. Capacité du Fog ---
    ax1.plot(df['t'], df['active_cpu_fog'], label='Charge Fog (CPU)', color='dodgerblue', linewidth=2)
    ax1.plot(df['t'], df['fog_capacity'], label='Capacité Fog (CPU)', color='red', linestyle='--', linewidth=2)
    ax1.set_ylabel("Unités CPU")
    ax1.set_title("Charge et Capacité du Nœud Fog", fontsize=14)
    ax1.legend()
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

    # Annoter les décisions de scaling
    scale_up_times = df[df['scale_decision'] == 'up']['t']
    scale_down_times = df[df['scale_decision'] == 'down']['t']
    for t in scale_up_times:
        ax1.axvline(x=t, color='green', linestyle=':', linewidth=1.5, label=f'Scale Up at t={t}' if t == scale_up_times.iloc[0] else "")
    for t in scale_down_times:
        ax1.axvline(x=t, color='purple', linestyle=':', linewidth=1.5, label=f'Scale Down at t={t}' if t == scale_down_times.iloc[0] else "")
    # Pour éviter les duplications de labels dans la légende
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    # --- Graphique 2: Pression (Utilisation) et Prédictions ---
    ax2.plot(df['t'], df['pressure'], label='Pression Réelle (Utilisation)', color='darkorange', linewidth=2)
    ax2.plot(df['t'], df['predicted_pressure'], label='Pression Prédite (LSTM)', color='black', linestyle=':', linewidth=1.5)
    # Zone d'incertitude
    ax2.fill_between(
        df['t'],
        df['predicted_pressure'] - df['prediction_uncertainty'],
        df['predicted_pressure'] + df['prediction_uncertainty'],
        color='gray', alpha=0.2, label='Incertitude de prédiction'
    )
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Seuil de Surcharge (100%)')
    ax2.set_ylabel("Pression (Ratio)")
    ax2.set_title("Pression du Fog et Prédictions LSTM", fontsize=14)
    ax2.legend()
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # --- Graphique 3: Ratio de Délestage (Offloading) ---
    ax3.plot(df['t'], df['offload_ratio'], label='Ratio de Délestage', color='sienna', marker='.', linestyle='-')
    ax3.set_ylabel("Ratio de Délestage")
    ax3.set_title("Décisions de Délestage (Offloading)", fontsize=14)
    ax3.legend(loc='upper left')
    ax3.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Axe Y secondaire pour le nombre de tâches délestées
    ax3b = ax3.twinx()
    ax3b.bar(df['t'], df['tasks_placed_cloud'], label='Tâches délestées (par tick)', color='lightcoral', alpha=0.6)
    ax3b.set_ylabel("Nb Tâches délestées")
    ax3b.legend(loc='upper right')

    # --- Graphique 4: Répartition des Tâches Actives ---
    ax4.stackplot(
        df['t'], df['tasks_on_fog'], df['tasks_on_cloud'],
        labels=['Tâches sur Fog', 'Tâches sur Cloud'],
        colors=['skyblue', 'salmon'],
        alpha=0.8
    )
    ax4.set_xlabel("Temps (ticks)", fontsize=12)
    ax4.set_ylabel("Nombre de Tâches Actives")
    ax4.set_title("Répartition des Tâches Actives (Fog vs. Cloud)", fontsize=14)
    ax4.legend(loc='upper left')
    ax4.grid(True, which='both', linestyle=':', linewidth=0.5)

    # --- Finalisation ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Créer le dossier de sortie si nécessaire
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Graphique sauvegardé dans: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère des graphiques à partir des résultats de simulation.")
    parser.add_argument(
        "--input",
        required=True,
        help="Chemin vers le fichier CSV des résultats (ex: data/results_up.csv)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Chemin pour sauvegarder l'image PNG (ex: images/results_up.png)"
    )
    args = parser.parse_args()

    plot_simulation_results(csv_path=args.input, output_path=args.output)