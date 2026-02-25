# plot_results.py (AMÉLIORÉ)
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle


def plot_simulation_results(csv_path: str, output_path: str):
    """
    Génère un graphique professionnel avec analyse des performances.
    """
    if not os.path.exists(csv_path):
        print(f"Erreur: Fichier de résultats introuvable: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # ✅ CORRECTION: Renommage des colonnes pour correspondre à sim_core.py
    rename_map = {
        "pool_pressure": "pressure",
        "pool_active_cpu_fog": "active_cpu_fog",
        "pool_fog_capacity": "fog_capacity"
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Calcul des métriques de performance
    metrics = calculate_performance_metrics(df)
    
    # --- Création de la figure et des sous-graphiques ---
    fig = plt.figure(figsize=(20, 28))
    
    # Définition des grilles
    gs = fig.add_gridspec(9, 2, height_ratios=[3, 3, 0, 0, 3, 3, 0, 3, 3], hspace=0.5)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
   # ax3 = fig.add_subplot(gs[2, :])
   # ax4 = fig.add_subplot(gs[3, :])
    ax5 = fig.add_subplot(gs[4, 0])  # Métriques gauche
    ax6 = fig.add_subplot(gs[4, 1])  # Métriques droite
    ax7 = fig.add_subplot(gs[5, :])  # Prédictions vs Réalité
  #  ax8 = fig.add_subplot(gs[6, :])  # Heatmap décisions
    ax9 = fig.add_subplot(gs[7, :])  # ✅ Heatmap Utilisation Fog par Nœud
    ax10 = fig.add_subplot(gs[8, :]) # ✅ Stats Énergie & Cloud

    # Titre principal
    workload_name = os.path.basename(csv_path).replace('.csv', '').replace('results_', '')
    fig.suptitle(f"ANALYSE DÉTAILLÉE - Simulation {workload_name.upper()}", 
                 fontsize=22, y=0.98, fontweight='bold')

    # --- 1. Charge vs. Capacité du Fog ---
    plot_load_capacity(ax1, df)
    
    # --- 2. Pression et Prédictions avec analyse d'erreur ---
    plot_pressure_predictions(ax2, df, metrics)
    
    # --- 3. Offloading et Décisions ---
   # plot_offloading_decisions(ax3, df, metrics)
    
    # --- 4. Distribution de la Pression (Histogramme) ---
   # plot_pressure_histogram(ax4, df)
    
    # --- 5. Métriques de Performance (Gauche) ---
    plot_performance_metrics_left(ax5, metrics)
    
    # --- 6. Métriques de Performance (Droite) ---
    plot_performance_metrics_right(ax6, metrics)
    
    # --- 7. Erreurs de Prédiction par Pression ---
    plot_prediction_errors(ax7, df, metrics)
    
    # --- 8. Heatmap des Décisions ---
    #plot_decision_heatmap(ax8, df)

    # --- 9. Heatmap Utilisation par Nœud Fog ---
    plot_fog_nodes_heatmap(ax9, df)

    # --- 10. Énergie et Cloud par Nœud ---
    plot_energy_and_cloud_stats(ax10, df)
    
    # --- Finalisation ---
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Créer le dossier de sortie si nécessaire
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"[OK] Graphique sauvegardé dans: {output_path}")
    
    # Afficher les métriques dans la console
    print("\n📊 MÉTRIQUES DE PERFORMANCE:")
    print(f"   MAE de prédiction: {metrics['mae']:.4f}")
    print(f"   RMSE de prédiction: {metrics['rmse']:.4f}")
    print(f"   Pression moyenne: {metrics['avg_pressure']:.2%}")
    print(f"   Surcharge (>100%): {metrics['overload_percentage']:.1%}")
    print(f"   Offload moyen: {metrics['avg_offload']:.1%}")
    print(f"   Nombre de scaling: {metrics['num_scaling']}")
    print(f"   Efficacité Fog: {metrics['fog_efficiency']:.1%}")
    print(f"   Stabilité (σ pression): {metrics['pressure_std']:.3f}")
    
    plt.close()


def calculate_performance_metrics(df):
    """Calcule les métriques de performance."""
    metrics = {}
    
    # Erreurs de prédiction
    valid_preds = df[df['predicted_pressure'] > 0]
    if len(valid_preds) > 0:
        errors = valid_preds['pressure'] - valid_preds['predicted_pressure']
        metrics['mae'] = np.mean(np.abs(errors))
        metrics['rmse'] = np.sqrt(np.mean(errors**2))
        metrics['bias'] = np.mean(errors)  # Biais systématique
    else:
        metrics['mae'] = metrics['rmse'] = metrics['bias'] = 0
    
    # Métriques de charge
    metrics['avg_pressure'] = df['pressure'].mean()
    metrics['max_pressure'] = df['pressure'].max()
    metrics['overload_percentage'] = (df['pressure'] > 1.0).mean()
    metrics['pressure_std'] = df['pressure'].std()
    
    # Métriques d'offload
    metrics['avg_offload'] = df['offload_ratio'].mean()
    metrics['max_offload'] = df['offload_ratio'].max()
    metrics['total_offloaded_tasks'] = df['tasks_placed_cloud'].sum()
    metrics['total_fog_tasks'] = df['tasks_placed_fog'].sum()
    
    # Métriques de scaling
    scale_changes = df[df['scale_decision'].isin(['up', 'down', 'emergency_up'])]
    metrics['num_scaling'] = len(scale_changes)
    metrics['scaling_frequency'] = metrics['num_scaling'] / len(df)
    
    # Efficacité
    capacity_used = df['active_cpu_fog'].sum()
    capacity_available = df['fog_capacity'].sum()
    metrics['fog_efficiency'] = capacity_used / capacity_available if capacity_available > 0 else 0
    
    # Coût estimé (simplifié)
    cloud_cost = metrics['total_offloaded_tasks'] * 0.02  # Coût cloud par tâche
    fog_cost = metrics['total_fog_tasks'] * 0.001  # Coût fog par tâche
    scaling_cost = metrics['num_scaling'] * 0.05  # Coût de changement de capacité
    metrics['estimated_cost'] = cloud_cost + fog_cost + scaling_cost
    
    return metrics


def plot_load_capacity(ax, df):
    """Graphique charge vs capacité."""
    ax.plot(df['t'], df['active_cpu_fog'], label='Charge Fog (CPU)', 
           color='dodgerblue', linewidth=2.5, alpha=0.8)
    ax.plot(df['t'], df['fog_capacity'], label='Capacité Fog (CPU)', 
           color='crimson', linestyle='--', linewidth=2)
    
    # Remplissage entre charge et capacité
    ax.fill_between(df['t'], df['active_cpu_fog'], df['fog_capacity'],
                   where=(df['active_cpu_fog'] > df['fog_capacity']),
                   color='red', alpha=0.2, label='Dépassement')
    
    # Annotations des décisions de scaling
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


def plot_pressure_predictions(ax, df, metrics):
    """Graphique pression et prédictions."""
    # Pression réelle
    ax.plot(df['t'], df['pressure'], label='Pression Réelle', 
           color='darkorange', linewidth=3, alpha=0.8)
    
    # Prédictions
    ax.plot(df['t'], df['predicted_pressure'], label='Prédiction LSTM', 
           color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Zone d'incertitude
    ax.fill_between(df['t'], 
                   df['predicted_pressure'] - df['prediction_uncertainty'],
                   df['predicted_pressure'] + df['prediction_uncertainty'],
                   color='gray', alpha=0.2, label='Incertitude (±)')
    
    # Seuils importants
    ax.axhline(y=1.0, color='red', linestyle='-', linewidth=1.5, alpha=0.5, 
              label='Seuil Surcharge (100%)')
    ax.axhline(y=0.7, color='green', linestyle=':', linewidth=1, alpha=0.5,
              label='Cible (70%)')
    ax.axhline(y=0.3, color='blue', linestyle=':', linewidth=1, alpha=0.5,
              label='Sous-utilisation (30%)')
    
    # Remplissage de surcharge
    ax.fill_between(df['t'], 1.0, df['pressure'], 
                   where=(df['pressure'] > 1.0),
                   color='red', alpha=0.1, label='Zone de Surcharge')
    
    ax.set_ylabel("Pression (Utilisation)", fontsize=12)
    ax.set_title("② PRESSION RÉELLE vs PRÉDICTIONS", fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(-0.05, min(2.5, df['pressure'].max() * 1.2))


def plot_offloading_decisions(ax, df, metrics):
    """Graphique offloading et décisions."""
    # Ratio d'offload
    ax.plot(df['t'], df['offload_ratio'], label='Ratio de Délestage', 
           color='sienna', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    
    # Tâches délestées
    ax2 = ax.twinx()
    bars = ax2.bar(df['t'], df['tasks_placed_cloud'], 
                  label='Tâches Cloud/tick', color='lightcoral', alpha=0.6, width=0.8)
    
    # Annoter les pics d'offload
    offload_peaks = df.nlargest(3, 'offload_ratio')
    for _, peak in offload_peaks.iterrows():
        ax.annotate(f"{peak['offload_ratio']:.0%}", 
                   xy=(peak['t'], peak['offload_ratio']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_ylabel("Ratio de Délestage", fontsize=12, color='sienna')
    ax2.set_ylabel("Tâches Cloud", fontsize=12, color='lightcoral')
    ax.set_title("③ DÉLESTAGE (OFFLOADING)", fontsize=14, fontweight='bold', pad=10)
    
    # Légende combinée
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(-0.05, 1.05)


def plot_pressure_histogram(ax, df):
    """
    Affiche l'histogramme de la pression pour analyser la stabilité.
    Remplace l'ancien graphique de répartition des tâches.
    """
    # On filtre les données pour éviter le bruit du démarrage (pression > 0.01)
    data = df[df['pressure'] > 0.01]['pressure']
    
    if len(data) == 0:
        ax.text(0.5, 0.5, "Pas assez de données de pression", ha='center', va='center')
        return

    # Histogramme
    # density=True pour avoir une densité de probabilité comparable
    ax.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7, density=True, label='Fréquence')
    
    # Lignes verticales (Cible et Surcharge)
    ax.axvline(x=0.7, color='green', linestyle='--', linewidth=2, label='Cible (0.7)')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Surcharge (1.0)')

    # Zones de couleur en arrière-plan pour l'interprétation
    # Gris: Sous-utilisation / Vert: Zone Optimale / Rouge: Surcharge
    ylim = ax.get_ylim()
    ax.axvspan(0, 0.4, color='gray', alpha=0.1) 
    ax.axvspan(0.4, 0.9, color='green', alpha=0.05)
    ax.axvspan(1.0, max(2.0, data.max()), color='red', alpha=0.05)

    ax.set_xlabel("Pression (Charge / Capacité)", fontsize=10)
    ax.set_ylabel("Densité de probabilité", fontsize=10)
    ax.set_title("④ DISTRIBUTION DE LA PRESSION (STABILITÉ)", fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(1.2, data.max() + 0.1))


def plot_performance_metrics_left(ax, metrics):
    """Affiche les métriques de performance (gauche)."""
    ax.axis('off')
    
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
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    ax.set_title("⑤ MÉTRIQUES PRÉDICTION", fontsize=12, fontweight='bold', pad=10)


def plot_performance_metrics_right(ax, metrics):
    """Affiche les métriques de performance (droite)."""
    ax.axis('off')
    
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
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.set_title("⑥ MÉTRIQUES OPÉRATIONNELLES", fontsize=12, fontweight='bold', pad=10)


def plot_prediction_errors(ax, df, metrics):
    """Graphique d'analyse des erreurs de prédiction."""
    # Erreur par niveau de pression
    df['pred_error'] = df['pressure'] - df['predicted_pressure']
    df['pressure_bin'] = pd.cut(df['pressure'], bins=[0, 0.3, 0.7, 1.0, 2.0, 5.0])
    
    if df['pressure_bin'].notna().any():
        error_by_bin = df.groupby('pressure_bin', observed=False)['pred_error'].agg(['mean', 'std', 'count'])
        
        x_pos = np.arange(len(error_by_bin))
        ax.bar(x_pos, error_by_bin['mean'], yerr=error_by_bin['std'],
              capsize=5, color='skyblue', edgecolor='navy', alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(bin) for bin in error_by_bin.index], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Annoter le nombre d'échantillons
        for i, (idx, row) in enumerate(error_by_bin.iterrows()):
            ax.text(i, row['mean'] + (row['std'] if not np.isnan(row['std']) else 0),
                   f"n={int(row['count'])}", ha='center', fontsize=8)
    
    ax.set_ylabel("Erreur (Réel - Prédit)", fontsize=10)
    ax.set_xlabel("Bins de Pression", fontsize=10)
    ax.set_title("⑦ ANALYSE DES ERREURS DE PRÉDICTION", fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)


def plot_decision_heatmap(ax, df):
    """Heatmap des décisions dans le temps."""
    # Préparer les données pour le heatmap
    decisions_map = {'none': 0, 'up': 1, 'down': 2, 'emergency_up': 3}
    df['decision_code'] = df['scale_decision'].map(decisions_map)
    
    # Créer une matrice pour le heatmap
    time_bins = 20
    bin_size = len(df) // time_bins
    if bin_size == 0:
        bin_size = 1
    
    heatmap_data = []
    for i in range(0, len(df), bin_size):
        bin_data = df.iloc[i:i+bin_size]
        avg_pressure = bin_data['pressure'].mean()
        avg_offload = bin_data['offload_ratio'].mean()
        dominant_decision = bin_data['decision_code'].mode()[0] if not bin_data.empty else 0
        heatmap_data.append([avg_pressure, avg_offload, dominant_decision])
    
    if heatmap_data:
        heatmap_array = np.array(heatmap_data).T
        
        im = ax.imshow(heatmap_array, aspect='auto', cmap='RdYlBu_r',
                      interpolation='nearest')
        
        # Labels
        ax.set_ylabel("Mesure")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Pression', 'Offload', 'Décision'])
        ax.set_xlabel("Fenêtres Temporelles")
        
        # Barre de couleur
        plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    
    ax.set_title("⑧ HEATMAP DES DÉCISIONS DANS LE TEMPS", fontsize=12, fontweight='bold', pad=10)


def plot_fog_nodes_heatmap(ax, df):
    """Heatmap de la pression par nœud Fog individuel."""
    # Identifier les colonnes de pression des nœuds fog (format: fog_Nom_p)
    fog_cols = [c for c in df.columns if c.startswith('fog_') and c.endswith('_p')]
    
    if not fog_cols:
        ax.text(0.5, 0.5, "Pas de données détaillées par nœud", ha='center', va='center')
        return

    # Extraire les données et transposer pour avoir (Nœuds x Temps)
    # On nettoie les noms pour l'affichage
    labels = [c.replace('fog_', '').replace('_p', '') for c in fog_cols]
    data = df[fog_cols].T.values # (N_nodes, T)
    
    # Création du heatmap
    im = ax.imshow(data, aspect='auto', cmap='plasma', vmin=0, vmax=1.2, interpolation='nearest')
    
    # Configuration des axes
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Temps (ticks)", fontsize=10)
    ax.set_title("⑨ UTILISATION DÉTAILLÉE PAR NŒUD FOG (Heatmap)", fontsize=12, fontweight='bold', pad=10)
    
    # Barre de couleur
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label("Pression (0-1+)", fontsize=9)


def plot_energy_and_cloud_stats(ax, df):
    """Graphique combiné : Énergie par Fog et Charge par Cloud."""
    # Préparation des données Énergie
    power_cols = [c for c in df.columns if c.startswith('fog_') and c.endswith('_power')]
    energy_sums = []
    fog_labels = []
    
    if power_cols:
        # Somme de la puissance sur le temps = Énergie totale (si tick=1s)
        energy_sums = df[power_cols].sum().values
        fog_labels = [c.replace('fog_', '').replace('_power', '') for c in power_cols]

    # Préparation des données Cloud
    cloud_cols = [c for c in df.columns if c.startswith('cloud_') and c.endswith('_load')]
    cloud_sums = []
    cloud_labels = []
    
    if cloud_cols:
        # Moyenne de la charge
        cloud_sums = df[cloud_cols].mean().values
        cloud_labels = [c.replace('cloud_', '').replace('_load', '') for c in cloud_cols]

    # Création de deux sous-graphiques côte à côte
    ax.axis('off')
    
    # Sous-plot 1: Énergie Fog
    if len(energy_sums) > 0:
        ax1 = ax.inset_axes([0, 0, 0.48, 1])
        # Utilisation de positions numériques pour éviter ConversionError
        x_pos = range(len(fog_labels))
        bars = ax1.bar(x_pos, energy_sums, color='orange', alpha=0.7, edgecolor='darkorange')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(fog_labels, rotation=45, ha='right', fontsize=8)
        ax1.set_title("Consommation Énergétique Totale par Nœud Fog (Joules)", fontsize=10, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

    # Sous-plot 2: Charge Cloud
    if len(cloud_sums) > 0:
        ax2 = ax.inset_axes([0.52, 0, 0.48, 1])
        x_pos2 = range(len(cloud_labels))
        bars2 = ax2.bar(x_pos2, cloud_sums, color='skyblue', alpha=0.7, edgecolor='blue')
        ax2.set_xticks(x_pos2)
        ax2.set_xticklabels(cloud_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_title("Charge Moyenne par Nœud Cloud (CPU)", fontsize=10, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Génère des graphiques d'analyse détaillée à partir des résultats de simulation."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Chemin vers le fichier CSV des résultats (ex: data/results_up.csv)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Chemin pour sauvegarder l'image PNG (ex: images/analysis_up.png)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Afficher les métriques détaillées dans la console"
    )
    args = parser.parse_args()

    plot_simulation_results(csv_path=args.input, output_path=args.output)