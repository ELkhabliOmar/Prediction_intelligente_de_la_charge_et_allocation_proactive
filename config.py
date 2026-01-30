# config.py - VÉRIFIE CES VALEURS
import os
from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).parent

# Chemins des datasets
DATASET_DIR = PROJECT_ROOT / "dataset" / "Pakistan" / "data" / "Tuple30K"
DATASET_DIR_50k = PROJECT_ROOT / "dataset" / "Pakistan" / "data" / "Tuple50K"
DATASET_DIR_100k = PROJECT_ROOT / "dataset" / "Pakistan" / "data" / "Tuple100K"


# ✅ CORRECTION: Assure-toi que DEFAULT_WORKLOAD pointe sur le bon fichier
DEFAULT_TRAINSET = str(DATASET_DIR / "trainset.csv")
DEFAULT_TESTSET = str(DATASET_DIR / "testset.csv")
DEFAULT_WORKLOAD = str(DATASET_DIR / "testset.csv")  # ⚠️ CECI EST IMPORTANT!

#
DEFAULT_TRAINSET_50k = str(DATASET_DIR_50k / "trainset.csv")
DEFAULT_TESTSET_50k = str(DATASET_DIR_50k / "testset.csv")
DEFAULT_TRAINSET_100k = str(DATASET_DIR_100k / "trainset.csv")
DEFAULT_TESTSET_100k = str(DATASET_DIR_100k / "testset.csv")
#

# Chemins des modèles
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_LSTM = str(MODELS_DIR / "lstm_util.pth")
DEFAULT_DQN = str(MODELS_DIR / "dqn_fog_cloud.pth")

# Chemins des résultats
RESULTS_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS = str(RESULTS_DIR / "results_proactive.csv")
DEFAULT_PLOT_OUT = str(RESULTS_DIR / "plot_proactive.png")