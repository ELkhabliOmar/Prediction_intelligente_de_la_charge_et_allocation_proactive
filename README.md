# Prédiction Intelligente de la Charge et Allocation Proactive des Ressources dans les Environnements Fog-Cloud

Ce projet met en œuvre un système d'allocation proactive de ressources pour les environnements Fog-Cloud, ciblant spécifiquement les applications IoT agroalimentaires. Il utilise une boucle **MAPE (Monitoring-Analysis-Planning-Execution)** pour anticiper les variations de charge et optimiser l'utilisation des ressources (latence, énergie, coût) avant que la congestion ne se produise.

## Fonctionnalités Clés

*   **Cycle MAPE Proactif** : Le cœur du simulateur est une boucle MAPE qui s'exécute périodiquement pour adapter les ressources de manière préventive.
*   **Prédiction de Charge (LSTM)** : Un modèle `EnhancedLSTM` avec mécanisme d'attention prédit la charge future du pool de nœuds Fog.
*   **Planification Stratégique (H-VWPO)** : Un planificateur heuristique prend des décisions de *scaling vertical* (augmenter/diminuer la capacité CPU des nœuds Fog) et de *scaling horizontal* (définir un ratio de délestage vers le Cloud).
*   **Ordonnancement (DQN)** : Un agent d'Apprentissage par Renforcement Profond (DQN) décide pour chaque tâche si elle doit être placée sur le Fog ou le Cloud, en tenant compte du plan proactif.
*   **Gestion d'Énergie** : La simulation suit une estimation de la consommation d'énergie en Joules des nœuds Fog.
*   **Lanceur Interactif** : Un menu (`menu.py`) guide l'utilisateur pour entraîner les modèles, lancer des simulations et générer des graphiques.
*   **Visualisation Détaillée** : Un script (`plot_results.py`) génère des rapports graphiques complets sur les performances de la simulation (pression, prédictions, scaling, délestage, etc.).

## Architecture

Le système est construit sur le simulateur `edge-sim-py` et se compose de trois modules principaux qui implémentent la boucle MAPE.

### 1. Module 1: Prédiction de Charge Multi-Horizon (Analyse)
*   **Objectif :** Prédire l'utilisation future des ressources (pression CPU) sur le pool Fog.
*   **Méthode :** Utilise un réseau de neurones **EnhancedLSTM** (Long Short-Term Memory avec attention) chargé à partir d'un fichier pré-entraîné (`.pth`). Le modèle fournit des prédictions de charge et une estimation de l'incertitude.

### 2. Module 2: Planification Proactive (Planification)
*   **Objectif :** Déterminer les ajustements stratégiques des ressources sur la base des prédictions.
*   **Méthode :** Implémente une logique heuristique inspirée de H-VWPO.
*   **Décisions :**
    *   **Scaling Vertical :** Décide de `scale_up` (augmenter) ou `scale_down` (diminuer) la capacité CPU d'un nœud Fog pour s'aligner sur la charge prédite tout en visant une utilisation cible (ex: 70%).
    *   **Délestage (Offloading) :** Calcule un ratio global de délestage vers le Cloud si la charge prédite risque de dépasser la capacité du Fog.

### 3. Module 3: Ordonnancement Proactif (Exécution)
*   **Objectif :** Assurer le placement fin de chaque service/tâche individuel.
*   **Méthode :**
    *   **Fog vs. Cloud :** Utilise un modèle **DQN** (Deep Q-Network) pour décider du placement (Fog ou Cloud) en fonction de l'état actuel du système et du ratio de délestage fourni par le planificateur.
    *   **Sélection du Nœud Fog :** Utilise une heuristique simple pour sélectionner le nœud Fog le moins chargé (`pick_best_fog`).

## Structure du Projet

```
.
├── config.py               # Fichier de configuration central (chemins, etc.)
├── menu.py                 # Lanceur interactif pour le projet
├── plot_results.py         # Script pour générer les graphiques d'analyse
├── generate_workload.py    # Script pour générer les datasets de charge
├── requirements.txt        # Dépendances Python
├── models/
│   ├── train_lstm.py       # Script d'entraînement du modèle LSTM
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