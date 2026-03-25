# Prédiction Intelligente de la Charge et Allocation Proactive des Ressources dans les Environnements Fog-Cloud

Ce projet met en œuvre un système d'allocation proactive de ressources pour les environnements Fog-Cloud, ciblant spécifiquement les applications IoT agroalimentaires. Il utilise une boucle **MAPE (Monitoring-Analysis-Planning-Execution)** pour anticiper les variations de charge et optimiser l'utilisation des ressources (latence, énergie, coût) avant que la congestion ne se produise.

## Fonctionnalités Clés

*   **Cycle MAPE Proactif** : Le cœur du simulateur est une boucle MAPE qui s'exécute périodiquement pour adapter les ressources de manière préventive.
*   **Prédiction Probabiliste (Bayesian LSTM)** : Utilise un réseau LSTM Bayésien avec **Monte Carlo Dropout** pour prédire la charge future et estimer l'incertitude, permettant des décisions plus robustes.
*   **Planification Stratégique H-VWPO (Horizontal Only)** : Optimise le pool Fog via un scaling horizontal proactif (activation/désactivation de nœuds) et un scaling vertical dynamique pour le repli (downscale).
*   **Ordonnancement Intelligent (DQN)** : Un agent DQN optimise le placement des tâches avec une fonction de récompense pondérée : **80% Saturation, 10% Énergie, 10% Latence**.
*   **Gestion d'Énergie** : La simulation suit une estimation de la consommation d'énergie en Joules des nœuds Fog.
*   **Baseline Scientifique** : Inclut une comparaison avec une approche standard **ARIMA + TOPSIS** pour valider les gains de l'approche proactive.
*   **Lanceur Interactif Amélioré** : Un menu (`menu.py`) complet pour gérer les datasets, les entraînements et les pipelines complets.
*   **Visualisation Détaillée** : Un script (`plot_results.py`) génère des rapports graphiques complets sur les performances de la simulation (pression, prédictions, scaling, délestage, etc.).

## Architecture

Le système est construit sur le simulateur `edge-sim-py` et se compose de trois modules principaux qui implémentent la boucle MAPE.

### 1. Module 1: Prédiction de Charge Multi-Horizon (Analyse)
*   **Objectif :** Prédire l'utilisation future des ressources (pression CPU) sur le pool Fog.
*   **Méthode :** Réseau **Bayesian LSTM** avec Attention. Il traite 4 features (Pression, Densité, Heure, Tendance) pour fournir une moyenne de prédiction et un intervalle de confiance.

### 2. Module 2: Planification Proactive (Planification)
*   **Objectif :** Déterminer les ajustements stratégiques des ressources sur la base des prédictions.
*   **Méthode :** Implémente une logique heuristique inspirée de H-VWPO.
*   **Décisions :**
    *   **Scaling Horizontal :** Active de nouveaux nœuds Fog si la charge projetée dépasse 70%.
    *   **Scaling Down Hybride :** Désactive les nœuds inutilisés ou réduit leur capacité CPU (Vertical Down) en période de faible charge.
    *   **Délestage (Offloading) :** Calcule un ratio global de délestage vers le Cloud si la charge prédite risque de dépasser la capacité du Fog.

### 3. Module 3: Ordonnancement Proactif (Exécution)
*   **Objectif :** Assurer le placement fin de chaque service/tâche individuel.
*   **Méthode :**
    *   **Fog vs. Cloud :** Agent **DQN** entraîné pour minimiser la saturation du Fog (poids 0.8) tout en balançant l'énergie et la latence.
    *   **Sélection du Nœud Fog :** Algorithme de sélection multicritère (Saturation, Énergie, Latence) pour le load balancing intra-fog.

## Structure du Projet

```
.
├── config.py               # Fichier de configuration central (chemins, etc.)
├── menu.py                 # Lanceur interactif pour le projet
├── plot_results.py         # Script pour générer les graphiques d'analyse
├── generate_workload.py    # Script pour générer les datasets de charge
├── simple_baseline_arima_threshold.py # Comparaison avec ARIMA + TOPSIS
├── requirements.txt        # Dépendances Python
├── models/
│   ├── train_lstm.py       # Entraînement du LSTM Bayésien (Incertitude)
│   └── train_dqn.py        # Script d'entraînement du modèle DQN
├── project/
│   ├── test.py             # Script principal de simulation (boucle MAPE)
│   ├── sim_core.py         # Cœur de la logique de simulation et des modules
│   └── ui_utils.py         # Utilitaires pour l'affichage console
├── data/
│   ├── workload.csv        # Dataset généré pour l'entraînement/simulation
│   └── ...
├── results/
│   ├── results.csv         # Fichier de sortie des métriques de simulation
│   └── plot.png            # Graphique d'analyse généré
└── README.md
```

## Prérequis

*   Python 3.8+
*   `torch`
*   `edge-sim-py`
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `networkx`
*   `scipy` (optionnel, pour un meilleur lissage dans `train_lstm.py`)

## Installation

1.  Clonez le dépôt.
2.  Il est recommandé de créer un environnement virtuel :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Linux/macOS
    .\venv\Scripts\activate   # Sur Windows
    ```
3.  Installez les dépendances requises :
    ```bash
    python -m pip install -r requirements.txt
    ```

## Utilisation (Recommandée)

Le moyen le plus simple d'utiliser le projet est de passer par le lanceur interactif.

```bash
python menu.py
```

Le menu vous guidera à travers les différentes étapes :
1.  **Entraîner le LSTM** : Crée le modèle de prédiction de charge.
2.  **Entraîner le DQN** : Crée le modèle de décision de placement.
3.  **Lancer la simulation** : Exécute la simulation avec les modèles entraînés.
4.  **Générer le graphique** : Visualise les résultats de la dernière simulation.
5.  **Pipeline complet** : Exécute les étapes 1 à 4 en séquence.

## Utilisation Avancée (Scripts Manuels)

Vous pouvez également exécuter chaque script manuellement pour plus de contrôle.

### 1. Génération des Données
Le projet nécessite des datasets de charge. Pour les générer :
    ```bash
    python generate_workload.py
    ```
Cela créera `data/workload.csv` (pour l'entraînement) et d'autres scénarios de test.

### 2. Entraînement des Modèles
Entraînez d'abord le LSTM, puis le DQN.
```bash
# Entraîner le prédicteur de charge
python models/train_lstm.py --epochs 200 --lr 0.001

# Entraîner l'agent de décision
python models/train_dqn.py --steps 20000
```
Les modèles seront sauvegardés dans le dossier `models/` (configurable dans `config.py`).

### 3. Lancer la Simulation
Exécutez la simulation en spécifiant le workload et les modèles.
```bash
python project/test.py --ticks 200 --W 10 --workload data/workload.csv
```
Les résultats seront sauvegardés dans `results/results.csv`.

### 4. Visualiser les Résultats
Générez le graphique d'analyse à partir du fichier de résultats.
```bash
python plot_results.py --input results/results.csv --output results/plot.png
```

---
Basé sur le projet de recherche de Omar EL-KHABLI.