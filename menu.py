# menu.py - LAUNCHER INTERACTIF (Fog/Cloud Simulator)
import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_NAME = "Prédiction intelligente de la charge et allocationproactive des ressources dans les environnements fog-cloud"
VERSION = "v1.0"
AUTHOR = "EL-KHABLI"

# --- Chemins par défaut ---
DEFAULT_WORKLOAD = "data/workload.csv"
DEFAULT_LSTM = "models/lstm_util.pth"
DEFAULT_DQN = "models/dqn_fog_cloud.pth"
DEFAULT_RESULTS = "data/results_proactive.csv"
DEFAULT_PLOT_OUT = "images/analysis_proactive.png"

# --- Helpers UI ---
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def hr(char="═", n=72):
    return char * n

def color(text, c):
    # Simple ANSI (marche dans la plupart des terminaux modernes, y compris PowerShell récent)
    codes = {
        "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
        "blue": "\033[94m", "purple": "\033[95m", "cyan": "\033[96m",
        "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m"
    }
    return f"{codes.get(c,'')}{text}{codes['reset']}"

def exists_badge(path):
    p = Path(path)
    if p.exists():
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
        # shell=True pour Windows (et permet d’appeler python facilement)
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
    print("Workload :", exists_badge(DEFAULT_WORKLOAD))
    print("LSTM     :", exists_badge(DEFAULT_LSTM))
    print("DQN      :", exists_badge(DEFAULT_DQN))
    print("Results  :", exists_badge(DEFAULT_RESULTS))
    print(hr("-", 72))
    print()

def show_menu():
    print(color("🎛️  MENU PRINCIPAL (choisis un numéro)", "bold"))
    print(hr("-", 72))
    print(color(" 1", "cyan"), "- Entraîner LSTM (train_lstm.py)")
    print(color(" 2", "cyan"), "- Entraîner DQN (train_dqn.py)")
    print(color(" 3", "cyan"), "- Lancer simulation (test.py)")
    print(color(" 4", "cyan"), "- Générer graph (plot_results.py)")
    print(color(" 5", "cyan"), "- Pipeline complet (LSTM → DQN → SIM → PLOT)")
    print(color(" 6", "cyan"), "- Vérifier fichiers / chemins")
    print(color(" 0", "cyan"), "- Quitter")
    print(hr("-", 72))

def verify_files():
    clear()
    banner()
    status_panel()
    print(color("🔎 Conseils:", "yellow"))
    if not Path(DEFAULT_WORKLOAD).exists():
        print(" - Workload absent → lance generate_workload.py si tu l'as.")
    if not Path(DEFAULT_LSTM).exists():
        print(" - LSTM absent → lance option 1.")
    if not Path(DEFAULT_DQN).exists():
        print(" - DQN absent → lance option 2.")
    if not Path(DEFAULT_RESULTS).exists():
        print(" - Résultats absents → lance option 3.")
    print()
    pause()

def train_lstm():
    # Paramètres optionnels
    fog_cpu = ask_int("Fog CPU", 100)
    seq_len = ask_int("Seq len", 30)
    epochs = ask_int("Epochs", 200)
    lr = ask_float("Learning rate", 0.001)
    batch = ask_int("Batch size", 64)

    cmd = f'python train_lstm.py --fog_cpu {fog_cpu} --seq_len {seq_len} --epochs {epochs} --lr {lr} --batch {batch}'
    run_cmd(cmd, "🧠 ENTRAÎNEMENT LSTM")

def train_dqn():
    fog_cpu = ask_int("Fog CPU", 100)
    steps = ask_int("Steps", 20000)
    batch = ask_int("Batch size", 128)
    hidden = ask_int("Hidden dim", 128)

    cmd = f'python train_dqn.py --fog_cpu {fog_cpu} --steps {steps} --batch {batch} --hidden_dim {hidden}'
    run_cmd(cmd, "🎯 ENTRAÎNEMENT DQN")

def run_simulation():
    ticks = ask_int("Ticks", 200)
    W = ask_int("Fenêtre MAPE (W)", 10)
    target = ask_float("Target utilization", 0.70)
    out_csv = ask_str("Output CSV", DEFAULT_RESULTS)

    # s'assurer que le dossier existe
    out_dir = Path(out_csv).parent
    if str(out_dir) and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    cmd = f'python test.py --ticks {ticks} --W {W} --target_util {target} --output_csv "{out_csv}"'
    run_cmd(cmd, "🌫️☁️  SIMULATION FOG-CLOUD")

def run_plot():
    input_csv = ask_str("CSV input", DEFAULT_RESULTS)
    output_png = ask_str("PNG output", DEFAULT_PLOT_OUT)

    out_dir = Path(output_png).parent
    if str(out_dir) and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    cmd = f'python plot_results.py --input "{input_csv}" --output "{output_png}"'
    run_cmd(cmd, "📈 GÉNÉRATION GRAPHIQUE")

def full_pipeline():
    clear()
    banner()
    print(color("🚀 PIPELINE COMPLET", "bold"))
    print("Ce pipeline fait: LSTM → DQN → SIM → PLOT\n")
    pause("Appuie sur Entrée pour lancer le pipeline...")

    # 1) LSTM
    run_cmd("python train_lstm.py", "🧠 Étape 1/4 : Entraînement LSTM")

    # 2) DQN
    run_cmd("python train_dqn.py", "🎯 Étape 2/4 : Entraînement DQN")

    # 3) Simulation
    run_cmd(f'python test.py --ticks 200 --W 10 --target_util 0.70 --output_csv "{DEFAULT_RESULTS}"',
            "🌫️☁️  Étape 3/4 : Simulation")

    # 4) Plot
    run_cmd(f'python plot_results.py --input "{DEFAULT_RESULTS}" --output "{DEFAULT_PLOT_OUT}"',
            "📈 Étape 4/4 : Graphique")


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
        elif choice == "0":
            clear()
            print(color("👋 Bye !", "cyan"))
            time.sleep(0.3)
            break
        else:
            print(color("Choix invalide. Essaie 0-6.", "red"))
            time.sleep(1.0)


if __name__ == "__main__":
    main()
