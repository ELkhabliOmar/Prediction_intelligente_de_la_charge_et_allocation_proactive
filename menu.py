# menu.py - VERSION AMÉLIORÉE avec vérification des datasets
import os
import time
import subprocess
import sys
from pathlib import Path
import json

from config import (
    DEFAULT_TRAINSET,
    DEFAULT_TESTSET,
    DEFAULT_TRAINSET_50k,
    DEFAULT_TESTSET_50k,
    DEFAULT_TRAINSET_100k,
    DEFAULT_TESTSET_100k,
    DEFAULT_WORKLOAD,
    DEFAULT_LSTM,
    DEFAULT_DQN,
    DEFAULT_RESULTS,
    DEFAULT_PLOT_OUT,
)

PROJECT_NAME = "Prédiction intelligente de la charge et allocation proactive des ressources dans les environnements fog-cloud"
VERSION = "v1.1"
AUTHOR = "EL-KHABLI"

# chemins des scripts
TRAIN_LSTM_SCRIPT = "models/train_lstm.py"
TRAIN_LSTM_IMPROVED_SCRIPT = "models/train_lstm.py"  # Nouveau script amélioré
TRAIN_DQN_SCRIPT  = "models/train_dqn.py"
SIM_SCRIPT        = "project/test.py"
PLOT_SCRIPT       = "plot_results.py"
BASELINE_SCRIPT   = "simple_baseline_arima_threshold.py"

# --- Helpers UI ---
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def hr(char="═", n=72):
    return char * n

def color(text, c):
    codes = {
        "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
        "blue": "\033[94m", "purple": "\033[95m", "cyan": "\033[96m",
        "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m"
    }
    return f"{codes.get(c,'')}{text}{codes['reset']}"

def exists_badge(path, verbose=True):
    p = Path(path)
    if p.exists():
        if verbose and p.suffix == '.pth':
            # Vérifier le type de modèle LSTM
            try:
                import torch
                checkpoint = torch.load(p, map_location="cpu")
                if 'arch' in checkpoint:
                    model_type = checkpoint['arch']
                    if model_type == 'BayesianLSTM':
                        return color("🧠✅", "green") + f" {path} (BayesianLSTM)"
                    elif model_type == 'EnhancedLSTM':
                        return color("🧠✅", "green") + f" {path} (EnhancedLSTM)"
                # Vérifier si c'est un DQN
                elif 'model_type' in checkpoint and checkpoint['model_type'] == 'dqn':
                    return color("🎯✅", "green") + f" {path} (DQN)"
                return color("✅", "green") + f" {path}"
            except:
                return color("✅", "green") + f" {path}"
        return color("✅", "green") + f" {path}"
    return color("❌", "red") + f" {path}"

def pause(msg="Appuie sur Entrée pour continuer..."):
    input(color(msg, "dim"))

def run_cmd(cmd, title=None):
    clear()
    if title:
        print(color(title, "cyan"))
        print(hr())
    print(color("Commande:", "yellow"), cmd)
    print(hr())
    try:
        result = subprocess.call(cmd, shell=True)
        print(hr())
        if result == 0:
            print(color("✅ Terminé avec succès.", "green"))
        else:
            print(color(f"❌ Erreur (code {result}).", "red"))
    except KeyboardInterrupt:
        print("\n" + color("⛔ Interrompu par l'utilisateur.", "yellow"))
    pause()

def ask_int(prompt, default):
    s = input(f"{prompt} [{default}] : ").strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default

def ask_float(prompt, default):
    s = input(f"{prompt} [{default}] : ").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default

def ask_str(prompt, default):
    s = input(f"{prompt} [{default}] : ").strip()
    return s if s else default

def ask_bool(prompt, default=True):
    default_str = "O" if default else "N"
    s = input(f"{prompt} [{default_str}] : ").strip().upper()
    if not s:
        return default
    return s.startswith("O") or s.startswith("Y")

def get_dataset_info(path):
    """Obtient des informations sur un dataset CSV"""
    p = Path(path)
    if not p.exists():
        return None
    
    try:
        import csv
        with open(p, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            return {"count": 0, "columns": []}
        
        # Compter les tâches
        task_count = len(rows)
        
        # Obtenir la plage temporelle
        timestamps = []
        for row in rows:
            if 'timestamp' in row:
                try:
                    timestamps.append(float(row['timestamp']))
                except:
                    pass
            elif 'GenerationTime' in row:
                try:
                    timestamps.append(float(row['GenerationTime']))
                except:
                    pass
        
        min_ts = min(timestamps) if timestamps else 0
        max_ts = max(timestamps) if timestamps else 0
        
        return {
            "count": task_count,
            "columns": list(rows[0].keys()) if rows else [],
            "time_range": (min_ts, max_ts),
            "exists": True
        }
    except Exception as e:
        return {"error": str(e), "exists": True}

# --- Menus ---
def banner():
    print(color(hr(), "purple"))
    print(color(f"  {PROJECT_NAME}", "bold"))
    print(color(f"  {VERSION}  |  by {AUTHOR}", "dim"))
    print(color(hr(), "purple"))
    print()

def status_panel():
    print(color("📌 ÉTAT DU PROJET", "bold"))
    print(hr("-", 72))
    
    # Informations détaillées sur les datasets
    train_info = get_dataset_info(DEFAULT_TRAINSET)
    test_info = get_dataset_info(DEFAULT_TESTSET)
    workload_info = get_dataset_info(DEFAULT_WORKLOAD)
    
    if train_info and train_info.get('exists'):
        train_status = f"{exists_badge(DEFAULT_TRAINSET, verbose=False)}"
        if 'count' in train_info:
            train_status += f" ({train_info['count']} tâches)"
    else:
        train_status = exists_badge(DEFAULT_TRAINSET)
    
    if test_info and test_info.get('exists'):
        test_status = f"{exists_badge(DEFAULT_TESTSET, verbose=False)}"
        if 'count' in test_info:
            test_status += f" ({test_info['count']} tâches)"
    else:
        test_status = exists_badge(DEFAULT_TESTSET)
    
    if workload_info and workload_info.get('exists'):
        workload_status = f"{exists_badge(DEFAULT_WORKLOAD, verbose=False)}"
        if 'count' in workload_info:
            workload_status += f" ({workload_info['count']} tâches)"
    else:
        workload_status = exists_badge(DEFAULT_WORKLOAD)
    
    print("Trainset :", train_status)
    print("Testset  :", test_status)
    print("Workload :", workload_status)
    print("LSTM     :", exists_badge(DEFAULT_LSTM))
    print("DQN      :", exists_badge(DEFAULT_DQN))
    print("Results  :", exists_badge(DEFAULT_RESULTS))
    
    # Avertissement si datasets différents
    if (Path(DEFAULT_TRAINSET).exists() and Path(DEFAULT_TESTSET).exists() and 
        Path(DEFAULT_WORKLOAD).exists()):
        if (DEFAULT_TRAINSET != DEFAULT_WORKLOAD or 
            DEFAULT_TESTSET != DEFAULT_WORKLOAD):
            print(color("\n⚠️  ATTENTION: Les datasets sont différents!", "yellow"))
            print(f"   Entraînement sur: {Path(DEFAULT_TRAINSET).name}")
            print(f"   Test sur: {Path(DEFAULT_TESTSET).name}")
            print(f"   Simulation sur: {Path(DEFAULT_WORKLOAD).name}")
    
    print(hr("-", 72))
    print()

def show_menu():
    print(color("🎛️  MENU PRINCIPAL (choisis un numéro)", "bold"))
    print(hr("-", 72))
    print(color(" 1", "cyan"), "- Entraîner LSTM (train_lstm_improved.py)")
    print(color(" 2", "cyan"), "- Entraîner DQN (models/train_dqn.py)")
    print(color(" 3", "cyan"), "- Lancer simulation (project/test.py)")
    print(color(" 4", "cyan"), "- Générer graph (plot_results.py)")
    print(color(" 5", "cyan"), "- Pipeline complet (LSTM → DQN → SIM → PLOT)")
    print(color(" 6", "cyan"), "- Vérifier fichiers / chemins")
    print(color(" 7", "cyan"), "- Comparer les datasets")
    print(color(" 8", "cyan"), "- Lancer Baseline (ARIMA)")
    print(color(" 0", "cyan"), "- Quitter")
    print(hr("-", 72))

def verify_files():
    clear()
    banner()
    status_panel()
    
    # Vérifications détaillées
    print(color("🔎 VÉRIFICATION DÉTAILLÉE:", "bold"))
    print("-" * 40)
    
    # Vérifier LSTM
    lstm_path = Path(DEFAULT_LSTM)
    if lstm_path.exists():
        try:
            import torch
            checkpoint = torch.load(lstm_path, map_location="cpu")
            if 'arch' in checkpoint:
                print(f"✅ LSTM: {checkpoint['arch']}")
                if 'model_config' in checkpoint:
                    cfg = checkpoint['model_config']
                    print(f"   - Seq len: {cfg.get('seq_len', 'N/A')}")
                    print(f"   - Hidden dim: {cfg.get('hidden_dim', 'N/A')}")
                    print(f"   - Dropout: {cfg.get('dropout', 'N/A')}")
                if 'normalization' in checkpoint:
                    norm = checkpoint['normalization']
                    print(f"   - Pressure max: {norm.get('pressure_max', 'N/A'):.3f}")
            else:
                print("✅ LSTM (ancien format)")
        except Exception as e:
            print(f"❌ LSTM: Erreur de lecture - {e}")
    else:
        print("❌ LSTM: Fichier absent")
    
    # Vérifier DQN
    dqn_path = Path(DEFAULT_DQN)
    if dqn_path.exists():
        try:
            import torch
            checkpoint = torch.load(dqn_path, map_location="cpu")
            print("✅ DQN chargé")
            if 'steps_trained' in checkpoint:
                print(f"   - Steps entraînés: {checkpoint['steps_trained']}")
        except Exception as e:
            print(f"❌ DQN: Erreur de lecture - {e}")
    else:
        print("❌ DQN: Fichier absent")
    
    print()
    pause()

def compare_datasets():
    clear()
    banner()
    
    print(color("📊 COMPARAISON DES DATASETS", "bold"))
    print(hr("-", 72))
    
    datasets = {
        "Entraînement (Trainset)": DEFAULT_TRAINSET,
        "Test (Testset)": DEFAULT_TESTSET,
        "Simulation (Workload)": DEFAULT_WORKLOAD
    }
    
    infos = {}
    for name, path in datasets.items():
        info = get_dataset_info(path)
        infos[name] = info
        print(f"\n{color(name, 'cyan')}:")
        print(f"  Chemin: {path}")
        
        if info and info.get('exists'):
            if 'error' in info:
                print(f"  ❌ Erreur: {info['error']}")
            else:
                print(f"  ✅ Tâches: {info.get('count', 'N/A')}")
                print(f"  📊 Colonnes: {', '.join(info.get('columns', []))[:50]}...")
                if 'time_range' in info:
                    min_ts, max_ts = info['time_range']
                    print(f"  ⏱️  Plage temporelle: {min_ts:.0f} → {max_ts:.0f}")
                    print(f"  🕒 Durée: {max_ts - min_ts:.0f} ticks")
        else:
            print(f"  ❌ Fichier introuvable")
    
    # Vérifier la cohérence
    print("\n" + color("🔍 VÉRIFICATION DE COHÉRENCE:", "bold"))
    
    if all(info and info.get('exists') and 'error' not in info for info in infos.values()):
        train_count = infos["Entraînement (Trainset)"]['count']
        test_count = infos["Test (Testset)"]['count']
        sim_count = infos["Simulation (Workload)"]['count']
        
        if train_count == test_count == sim_count:
            print("✅ Tous les datasets ont le même nombre de tâches")
        else:
            print("⚠️  Les datasets ont des tailles différentes:")
            print(f"   - Entraînement: {train_count} tâches")
            print(f"   - Test: {test_count} tâches")
            print(f"   - Simulation: {sim_count} tâches")
            
            # Recommandation
            if DEFAULT_TRAINSET != DEFAULT_TESTSET:
                print(color("\n💡 RECOMMANDATION:", "yellow"))
                print("Pour un entraînement et test cohérents:")
                print("1. Utilise le trainset pour l'entraînement")
                print("2. Utilise le testset pour la simulation")
                print("3. Vérifie config.py si DEFAULT_WORKLOAD ≠ DEFAULT_TESTSET")
    
    print(hr("-", 72))
    pause()

def train_lstm():
    clear()
    banner()
    
    print(color("🧠 ENTRAÎNEMENT LSTM BAYÉSIEN AMÉLIORÉ", "bold"))
    print(hr("-", 72))
    
    # Vérifier le dataset d'entraînement
    train_info = get_dataset_info(DEFAULT_TRAINSET)
    if not train_info or not train_info.get('exists'):
        print(color(f"❌ Dataset d'entraînement introuvable: {DEFAULT_TRAINSET}", "red"))
        pause()
        return
    
    print(f"📊 Dataset d'entraînement: {Path(DEFAULT_TRAINSET).name}")
    print(f"   - Tâches: {train_info.get('count', 'N/A')}")
    if 'time_range' in train_info:
        min_ts, max_ts = train_info['time_range']
        print(f"   - Plage: {min_ts:.0f} → {max_ts:.0f}")
    
    print("\n" + color("⚙️  PARAMÈTRES D'ENTRAÎNEMENT:", "bold"))
    
    # Demander le type de modèle
    print("\n🤖 Type de modèle:")
    print("  1. Bayesian LSTM amélioré (recommandé)")
    print("  2. LSTM classique")
    model_type = input("Choisis [1/2] (défaut: 1) : ").strip() or "1"
    
    if model_type == "1":
        script = TRAIN_LSTM_IMPROVED_SCRIPT
        model_out = DEFAULT_LSTM.replace(".pth", "_improved.pth")
        print("✅ Bayesian LSTM sélectionné")
        
        # Paramètres pour Bayesian LSTM
        seq_len = ask_int("Longueur séquence", 50)
        hidden_dim = ask_int("Dimension cachée", 256)
        num_layers = ask_int("Nombre couches", 3)
        dropout = ask_float("Dropout", 0.4)
        epochs = ask_int("Epochs", 300)
        lr = ask_float("Learning rate", 0.0005)
        batch = ask_int("Batch size", 128)
    else:
        script = TRAIN_LSTM_SCRIPT
        model_out = DEFAULT_LSTM
        print("✅ LSTM classique sélectionné")
        
        # Paramètres pour LSTM classique
        seq_len = ask_int("Longueur séquence", 30)
        hidden_dim = ask_int("Dimension cachée", 128)
        num_layers = ask_int("Nombre couches", 2)
        dropout = ask_float("Dropout", 0.3)
        epochs = ask_int("Epochs", 200)
        lr = ask_float("Learning rate", 0.001)
        batch = ask_int("Batch size", 64)
    
    fog_cpu = ask_int("Fog CPU", 100)
    
    # Confirmation
    print("\n" + color("📋 RÉCAPITULATIF:", "bold"))
    print(f"  Script: {script}")
    print(f"  Dataset: {Path(DEFAULT_TRAINSET).name}")
    print(f"  Modèle sortie: {model_out}")
    print(f"  Seq len: {seq_len}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    print(f"  Epochs: {epochs}")
    print(f"  LR: {lr}")
    print(f"  Batch: {batch}")
    print(f"  Fog CPU: {fog_cpu}")
    
    if not ask_bool("\nLancer l'entraînement?", True):
        return
    
    # Construction de la commande
    if model_type == "1":
        # Bayesian LSTM
        cmd = (f'python "{script}" '
               f'--data "{DEFAULT_TRAINSET}" '
               f'--model_out "{model_out}" '
               f'--fog_cpu {fog_cpu} --seq_len {seq_len} --epochs {epochs} '
               f'--lr {lr} --batch {batch} --hidden_dim {hidden_dim} '
               f'--num_layers {num_layers} --dropout {dropout}')
    else:
        # LSTM classique
        cmd = (f'python "{script}" '
               f'--data "{DEFAULT_TRAINSET}" '
               f'--model_out "{model_out}" '
               f'--fog_cpu {fog_cpu} --seq_len {seq_len} --epochs {epochs} '
               f'--lr {lr} --batch {batch} --hidden_dim {hidden_dim} '
               f'--num_layers {num_layers} --dropout {dropout}')
    
    run_cmd(cmd, "🧠 ENTRAÎNEMENT LSTM")

def train_dqn():
    clear()
    banner()
    
    print(color("🎯 ENTRAÎNEMENT DQN", "bold"))
    print(hr("-", 72))
    
    # Vérifier le dataset d'entraînement
    train_info = get_dataset_info(DEFAULT_TRAINSET)
    if not train_info or not train_info.get('exists'):
        print(color(f"❌ Dataset d'entraînement introuvable: {DEFAULT_TRAINSET}", "red"))
        pause()
        return
    
    print(f"📊 Dataset d'entraînement: {Path(DEFAULT_TRAINSET).name}")
    
    # Paramètres - SUPPRIMER lr ici
    fog_cpu = ask_int("Fog CPU", 4000) # ✅ Corrigé: 4000 suffit pour saturer avec des tâches de ~500
    steps = ask_int("Steps", 20000)
    batch = ask_int("Batch size", 128)
    hidden = ask_int("Hidden dim", 128)
    # SUPPRIMER: lr = ask_float("Learning rate", 0.001)
    
    # Demander le dataset à utiliser
    print("\n📊 Choix du dataset:")
    print(f"  1. trainset.csv ({DEFAULT_TRAINSET})")
    print(f"  2. testset.csv ({DEFAULT_TESTSET})")
    dataset_choice = input("Choisis [1/2] (défaut: 1) : ").strip() or "1"
    
    if dataset_choice == "2":
        data_file = DEFAULT_TESTSET
        print(f"✅ Testset sélectionné pour l'entraînement DQN")
    else:
        data_file = DEFAULT_TRAINSET
        print(f"✅ Trainset sélectionné pour l'entraînement DQN")
    
    # Confirmation
    print("\n" + color("📋 RÉCAPITULATIF:", "bold"))
    print(f"  Dataset: {Path(data_file).name}")
    print(f"  Steps: {steps}")
    print(f"  Batch: {batch}")
    print(f"  Hidden dim: {hidden}")
    # SUPPRIMER: print(f"  LR: {lr}")
    print(f"  Fog CPU: {fog_cpu}")
    
    if not ask_bool("\nLancer l'entraînement DQN?", True):
        return
    
    # CORRECTION ICI : Supprimer --lr et utiliser --out au lieu de --output_model
    cmd = (f'python "{TRAIN_DQN_SCRIPT}" '
           f'--fog_cpu {fog_cpu} --steps {steps} --batch {batch} '
           f'--hidden_dim {hidden} --data "{data_file}" '
           f'--out "{DEFAULT_DQN}"')
    
    run_cmd(cmd, "🎯 ENTRAÎNEMENT DQN")

def run_simulation():
    clear()
    banner()
    
    print(color("🌫️☁️  SIMULATION FOG-CLOUD", "bold"))
    print(hr("-", 72))
    
    # Vérifier que le modèle LSTM existe
    if not Path(DEFAULT_LSTM).exists():
        print(color("⚠️  ATTENTION: Modèle LSTM non trouvé!", "yellow"))
        print("Il est recommandé d'entraîner un modèle LSTM d'abord (option 1).")
        if not ask_bool("Continuer sans LSTM?", False):
            return
    
    # Paramètres de simulation
    ticks = ask_int("Ticks", 200)
    W = ask_int("Fenêtre MAPE (W)", 10)
    target = ask_float("Target utilization", 0.70)
    down_threshold = ask_float("Down threshold (0.25-0.40)", 0.30)
    min_fog_cpu = ask_int("Min Fog CPU", 30)
    
    # Choix du dataset pour la simulation
    print("\n📊 Choix du dataset pour la simulation:")
    print(f"  1. trainset.csv ({DEFAULT_TRAINSET}) - POUR L'ENTRAÎNEMENT")
    print(f"  2. testset.csv ({DEFAULT_TESTSET}) - POUR LE TEST (recommandé)")
    print(f"  3. DEFAULT_WORKLOAD ({DEFAULT_WORKLOAD})")
    dataset_choice = input("Choisis [1/2/3] (défaut: 2) : ").strip()
    
    if dataset_choice == "1":
        workload_file = DEFAULT_TRAINSET
        dataset_name = "trainset (ENTRAÎNEMENT)"
        print(color("\n⚠️  ATTENTION: Vous utilisez le dataset d'entraînement!", "yellow"))
        print("Cela peut causer du surapprentissage. Utilisez testset.csv pour le test.")
    elif dataset_choice == "2":
        workload_file = DEFAULT_TESTSET
        dataset_name = "testset (TEST)"
    else:
        workload_file = DEFAULT_WORKLOAD
        dataset_name = "workload"
        print(f"\nℹ️  Utilisation de DEFAULT_WORKLOAD: {workload_file}")
    
    # Vérifier si le fichier existe
    if not Path(workload_file).exists():
        print(color(f"❌ ERREUR: Fichier {workload_file} introuvable!", "red"))
        print("Vérifiez config.py ou utilisez un autre dataset.")
        pause()
        return
    
    # Informations sur le dataset
    dataset_info = get_dataset_info(workload_file)
    if dataset_info and dataset_info.get('exists') and 'error' not in dataset_info:
        print(f"\n📊 Informations du dataset:")
        print(f"   - Tâches: {dataset_info.get('count', 'N/A')}")
        if 'time_range' in dataset_info:
            min_ts, max_ts = dataset_info['time_range']
            print(f"   - Plage temporelle: {min_ts:.0f} → {max_ts:.0f}")
            print(f"   - Durée: {max_ts - min_ts:.0f} ticks")
    
    out_csv = ask_str("Output CSV", DEFAULT_RESULTS)
    
    # Choix du modèle LSTM
    print("\n🤖 Choix du modèle LSTM:")
    lstm_files = []
    if Path(DEFAULT_LSTM).exists():
        lstm_files.append(DEFAULT_LSTM)
    if Path(DEFAULT_LSTM.replace(".pth", "_improved.pth")).exists():
        lstm_files.append(DEFAULT_LSTM.replace(".pth", "_improved.pth"))
    
    if not lstm_files:
        print("❌ Aucun modèle LSTM trouvé!")
        pause()
        return
    
    for i, lstm_file in enumerate(lstm_files, 1):
        try:
            import torch
            checkpoint = torch.load(lstm_file, map_location="cpu")
            if 'arch' in checkpoint:
                model_type = checkpoint['arch']
                print(f"  {i}. {Path(lstm_file).name} ({model_type})")
            else:
                print(f"  {i}. {Path(lstm_file).name} (LSTM classique)")
        except:
            print(f"  {i}. {Path(lstm_file).name}")
    
    lstm_choice = input(f"Choisis [1-{len(lstm_files)}] (défaut: 1) : ").strip()
    if not lstm_choice or not lstm_choice.isdigit() or int(lstm_choice) > len(lstm_files):
        lstm_model = lstm_files[0]
    else:
        lstm_model = lstm_files[int(lstm_choice)-1]
    
    print(f"\n✅ Modèle LSTM sélectionné: {Path(lstm_model).name}")
    
    # Confirmation
    print("\n" + color("📋 RÉCAPITULATIF DE LA SIMULATION:", "bold"))
    print(f"  Ticks: {ticks}")
    print(f"  W (MAPE): {W}")
    print(f"  Target util: {target}")
    print(f"  Down threshold: {down_threshold}")
    print(f"  Min Fog CPU: {min_fog_cpu}")
    print(f"  Dataset: {dataset_name}")
    print(f"  LSTM model: {Path(lstm_model).name}")
    print(f"  Output: {out_csv}")
    
    if not ask_bool("\nLancer la simulation?", True):
        return
    
    # Construction de la commande
    cmd = (f'python "{SIM_SCRIPT}" '
           f'--ticks {ticks} '
           f'--W {W} '
           f'--target_util {target} '
           f'--down_threshold {down_threshold} '
           f'--min_fog_cpu {min_fog_cpu} '
           f'--output_csv "{out_csv}" '
           f'--workload "{workload_file}" '
           f'--lstm_model "{lstm_model}"')
    
    run_cmd(cmd, "🌫️☁️  SIMULATION FOG-CLOUD")

def run_plot():
    input_csv = ask_str("CSV input", DEFAULT_RESULTS)
    output_png = ask_str("PNG output", DEFAULT_PLOT_OUT)
    
    if not Path(input_csv).exists():
        print(color(f"❌ Fichier CSV introuvable: {input_csv}", "red"))
        pause()
        return
    
    cmd = f'python "{PLOT_SCRIPT}" --input "{input_csv}" --output "{output_png}"'
    run_cmd(cmd, "📈 GÉNÉRATION GRAPHIQUE")

def run_baseline():
    clear()
    banner()
    print(color("📉 Lancer la Baseline (ARIMA)", "bold"))
    if not Path(BASELINE_SCRIPT).exists():
        print(color(f"❌ Script introuvable: {BASELINE_SCRIPT}", "red"))
        pause()
        return
    cmd = f'python "{BASELINE_SCRIPT}"'
    run_cmd(cmd, "📉 BASELINE ARIMA")

def full_pipeline():
    clear()
    banner()
    print(color("🚀 PIPELINE COMPLET", "bold"))
    print("Ce pipeline fait: LSTM → DQN → SIM → PLOT\n")
    
    print("📊 VÉRIFICATION DES DATASETS:")
    print("-" * 40)
    
    # Vérifier tous les datasets
    datasets = [
        ("Entraînement LSTM", DEFAULT_TRAINSET),
        ("Test/Simulation", DEFAULT_TESTSET),
        ("Entraînement DQN", DEFAULT_TRAINSET)  # DQN aussi sur trainset
    ]
    
    for name, path in datasets:
        info = get_dataset_info(path)
        if info and info.get('exists'):
            print(f"✅ {name}: {Path(path).name} ({info.get('count', 'N/A')} tâches)")
        else:
            print(f"❌ {name}: {path} - INTROUVABLE")
            print("Le pipeline ne peut pas continuer.")
            pause()
            return
    
    print("\n⚙️  CONFIGURATION DU PIPELINE:")
    print("-" * 40)
    
    # Configuration
    ticks = 200
    W = 10
    target = 0.70
    down_threshold = 0.30
    min_fog_cpu = 30
    
    print(f"Ticks: {ticks}")
    print(f"W (MAPE): {W}")
    print(f"Target util: {target}")
    print(f"Down threshold: {down_threshold}")
    print(f"Min Fog CPU: {min_fog_cpu}")
    print(f"Dataset simulation: {Path(DEFAULT_TESTSET).name} (TESTSET)")
    print("-" * 40)
    
    print(color("\n💡 STRATÉGIE:", "cyan"))
    print("1. LSTM entraîné sur trainset.csv")
    print("2. DQN entraîné sur trainset.csv")
    print("3. Simulation sur testset.csv (pour évaluation)")
    print("4. Graphiques générés")
    
    if not ask_bool("\nLancer le pipeline complet?", True):
        return
    
    # Étape 1: Entraînement LSTM sur trainset
    print("\n" + color("🧠 Étape 1/4 : Entraînement LSTM", "cyan"))
    cmd_lstm = (f'python "{TRAIN_LSTM_IMPROVED_SCRIPT}" '
                f'--data "{DEFAULT_TRAINSET}" '
                f'--model_out "{DEFAULT_LSTM.replace(".pth", "_improved.pth")}" '
                f'--fog_cpu 100 --seq_len 50 --epochs 300 '
                f'--lr 0.0005 --batch 128 --hidden_dim 256 '
                f'--num_layers 3 --dropout 0.4')
    run_cmd(cmd_lstm, "🧠 Étape 1/4 : Entraînement LSTM")
    
    # Étape 2: Entraînement DQN sur trainset
    print("\n" + color("🎯 Étape 2/4 : Entraînement DQN", "cyan"))
    cmd_dqn = (f'python "{TRAIN_DQN_SCRIPT}" '
               f'--fog_cpu 100 --steps 20000 --batch 128 '
               f'--hidden_dim 128 --data "{DEFAULT_TRAINSET}" '
               f'--output_model "{DEFAULT_DQN}"')
    run_cmd(cmd_dqn, "🎯 Étape 2/4 : Entraînement DQN")
    
    # Étape 3: Simulation sur testset
    print("\n" + color("🌫️☁️  Étape 3/4 : Simulation", "cyan"))
    # Utiliser le modèle amélioré s'il existe
    lstm_model = DEFAULT_LSTM.replace(".pth", "_improved.pth") if Path(DEFAULT_LSTM.replace(".pth", "_improved.pth")).exists() else DEFAULT_LSTM
    
    cmd_sim = (f'python "{SIM_SCRIPT}" '
               f'--ticks {ticks} --W {W} --target_util {target} '
               f'--down_threshold {down_threshold} '
               f'--min_fog_cpu {min_fog_cpu} '
               f'--output_csv "{DEFAULT_RESULTS}" '
               f'--workload "{DEFAULT_TESTSET}" '
               f'--lstm_model "{lstm_model}"')
    
    run_cmd(cmd_sim, "🌫️☁️  Étape 3/4 : Simulation")
    
    # Étape 4: Plot
    if Path(DEFAULT_RESULTS).exists():
        print("\n" + color("📈 Étape 4/4 : Graphique", "cyan"))
        cmd_plot = f'python "{PLOT_SCRIPT}" --input "{DEFAULT_RESULTS}" --output "{DEFAULT_PLOT_OUT}"'
        run_cmd(cmd_plot, "📈 Étape 4/4 : Graphique")
    else:
        print(color(f"\n⚠️  Aucun résultat trouvé à {DEFAULT_RESULTS}", "yellow"))
    
    print(color("\n🎉 PIPELINE TERMINÉ AVEC SUCCÈS !", "green"))
    
    # Résumé
    print("\n" + color("📊 RÉSUMÉ DU PIPELINE:", "bold"))
    print(f"  • LSTM entraîné sur: {Path(DEFAULT_TRAINSET).name}")
    print(f"  • DQN entraîné sur: {Path(DEFAULT_TRAINSET).name}")
    print(f"  • Simulation testée sur: {Path(DEFAULT_TESTSET).name}")
    print(f"  • Résultats: {DEFAULT_RESULTS}")
    print(f"  • Graphiques: {DEFAULT_PLOT_OUT}")
    
    pause()

def main():
    while True:
        clear()
        banner()
        status_panel()
        show_menu()

        choice = input(color("👉 Ton choix: ", "yellow")).strip()

        if choice == "1":
            train_lstm()
        elif choice == "2":
            train_dqn()
        elif choice == "3":
            run_simulation()
        elif choice == "4":
            run_plot()
        elif choice == "5":
            full_pipeline()
        elif choice == "6":
            verify_files()
        elif choice == "7":
            compare_datasets()
        elif choice == "8":
            run_baseline()
        elif choice == "0":
            clear()
            print(color("👋 Bye !", "cyan"))
            time.sleep(0.3)
            break
        else:
            print(color("Choix invalide. Essaie 0-7.", "red"))
            time.sleep(1.0)

if __name__ == "__main__":
    main()