# train_dqn.py - ENTRAINEMENT DQN (COMPATIBLE TEST.PY + CHECKPOINT ENRICHI)
import os
import csv
import random
import argparse
import math
import sys
from pathlib import Path
from collections import deque
from typing import List, Dict

# Ajout du dossier parent au path pour trouver config.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import *
import torch
import torch.nn as nn
import numpy as np


# --- Classe DQN ---
# Réseau de neurones simple (Fully Connected) qui estime la Q-value (récompense future) pour chaque action possible.
class DQN(nn.Module):
    """
    input: 5 features
    output: 2 actions (Fog/Cloud)
    """
    def __init__(self, input_dim=5, output_dim=2, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# Fixe les graines aléatoires pour assurer la reproductibilité des entraînements.
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_workload(path: str) -> List[Dict]:
    tasks = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            tasks.append(_normalize_task_row(row))
    return tasks


def _normalize_task_row(row: Dict) -> Dict:
    if "cpu_demand" in row and "ram_demand" in row:
        return {
            "timestamp": int(float(row.get("timestamp", 0))),
            "cpu_demand": int(float(row["cpu_demand"])),
            "ram_demand": int(float(row["ram_demand"])),
            "duration": int(float(row.get("duration", 1))),
        }

    task_size = float(row.get("TaskSize", 0.0))
    cycles_per_bit = float(row.get("CyclesPerBit", 0.0))
    trans_rate = max(1.0, float(row.get("TransBitRate", 1.0)))
    cpu_scale = 3000.0
    ram_scale = 1.0
    cpu_demand = int(max(1.0, (task_size * cycles_per_bit) / cpu_scale))
    ram_demand = int(max(64.0, task_size * ram_scale))
    duration = int(max(1.0, math.ceil(task_size / trans_rate)))

    return {
        "timestamp": int(float(row.get("GenerationTime", 0.0))),
        "cpu_demand": cpu_demand,
        "ram_demand": ram_demand,
        "duration": duration,
    }


# --- Fonction Principale (Main) ---
# Orchestre l'entraînement de l'agent DQN :
# 1. Initialise l'environnement simulé et les réseaux (Policy & Target).
# 2. Boucle sur les étapes (steps) pour générer des expériences (State, Action, Reward, Next State).
# 3. Stocke les expériences dans le Replay Buffer.
# 4. Entraîne le réseau sur des batchs aléatoires du buffer.
def main():
    ap = argparse.ArgumentParser(description="Entraînement DQN Fog/Cloud (compatible test.py)")
    ap.add_argument("--data", default=DEFAULT_TRAINSET, help="CSV dataset (trainset/testset Tuple30K)")
    ap.add_argument("--out", default=DEFAULT_DQN, help="chemin de sauvegarde du modèle DQN")
    ap.add_argument("--fog_cpu", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--replay_size", type=int, default=100000)
    ap.add_argument("--target_update", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"{args.data} introuvable.")

    tasks = load_workload(args.data)
    if not tasks:
        raise RuntimeError("Dataset vide.")

    print("=" * 60)
    print("ENTRAÎNEMENT DQN (COMPATIBLE TEST.PY)")
    print("=" * 60)
    print(f"Données: {args.data} ({len(tasks)} tâches)")
    print(f"Fog CPU: {args.fog_cpu}")
    print(f"Steps: {args.steps}")
    print(f"Architecture: DQN hidden={args.hidden_dim}, dropout={args.dropout}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dqn = DQN(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    target = DQN(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    target.load_state_dict(dqn.state_dict())

    opt = torch.optim.Adam(dqn.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    replay = deque(maxlen=args.replay_size)
    gamma = 0.99

    eps_start, eps_end, eps_decay = 1.0, 0.01, 0.9995
    eps = eps_start

    # état interne "pression" (env simplifié)
    pressure = 0.0
    DECAY = 0.95

    # Coûts (équilibrés)
    LAMBDA_OVER = 5.0
    LAMBDA_CLOUD = 0.03
    LAMBDA_FOG = 0.002
    LAMBDA_LATENCY = 0.01

    stats = {
        "rewards": [],
        "losses": [],
        "pressures": [],
        "fog_decisions": 0,
        "cloud_decisions": 0,
    }

    # Construit le vecteur d'état normalisé (Tensor) à partir des valeurs brutes de l'environnement.
    def make_state(cpu: float, ram: float, pressure_val: float, fog_cpu: float, offload_ratio: float):
        cpu_norm = min(cpu / 500.0, 2.0)
        ram_norm = min(ram / 4096.0, 2.0)
        pressure_clip = min(max(pressure_val, 0.0), 3.0)
        fog_cpu_norm = fog_cpu / 200.0
        # offload_ratio vient du planner en vrai, ici on le simule (mais on ne le rend pas 100% redondant)
        return torch.tensor([cpu_norm, ram_norm, pressure_clip, fog_cpu_norm, offload_ratio],
                            dtype=torch.float32, device=device)

    print("\n🎯 Début de l'entraînement...")

    # ✅ AMÉLIORATION: Parcours séquentiel pour préserver les "bursts" de charge
    task_idx = 0
    num_tasks = len(tasks)

    for step in range(args.steps):
        task = tasks[task_idx]
        task_idx = (task_idx + 1) % num_tasks
        
        cpu = float(task["cpu_demand"])
        ram = float(task["ram_demand"])
        dur = max(1.0, float(task.get("duration", 1)))

        # simule une consigne offload externe (simple)
        offload_ratio = min(0.6, max(0.0, (pressure - 0.7) * 0.6))
        state = make_state(cpu, ram, pressure, args.fog_cpu, offload_ratio)

        if random.random() < eps:
            a = random.randint(0, 1)
        else:
            with torch.no_grad():
                qvals = dqn(state.unsqueeze(0))
                a = int(torch.argmax(qvals, dim=1).item())

        if a == 0:
            stats["fog_decisions"] += 1
        else:
            stats["cloud_decisions"] += 1

        pressure_next = pressure * DECAY

        if a == 0:  # Fog
            pressure_next += (cpu / max(args.fog_cpu, 1.0)) * min(1.0, dur / 10.0)
            latency_cost = LAMBDA_LATENCY * 0.1
            cloud_cost = 0.0
            fog_cost = LAMBDA_FOG * (cpu / 100.0)
        else:  # Cloud
            latency_cost = LAMBDA_LATENCY * 1.0
            cloud_cost = LAMBDA_CLOUD * (cpu / 100.0)
            fog_cost = 0.0

        overload = max(0.0, pressure_next - 1.0)
        reward = - (LAMBDA_OVER * overload) - cloud_cost - fog_cost - latency_cost

        # petits bonus
        if a == 0 and overload < 0.2:
            reward += 0.1
        elif a == 1 and pressure > 0.8:
            reward += 0.05

        stats["rewards"].append(reward)
        stats["pressures"].append(pressure_next)

        next_state = make_state(cpu, ram, pressure_next, args.fog_cpu, offload_ratio)
        replay.append((state, a, reward, next_state))

        pressure = min(pressure_next, 3.0)

        if len(replay) >= args.warmup:
            batch = random.sample(replay, args.batch)
            S = torch.stack([b[0] for b in batch])
            A = torch.tensor([b[1] for b in batch], device=device)
            R = torch.tensor([b[2] for b in batch], device=device)
            NS = torch.stack([b[3] for b in batch])

            q = dqn(S).gather(1, A.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                q_next = target(NS).max(1)[0]
                y = R + gamma * q_next

            loss = loss_fn(q, y)
            stats["losses"].append(loss.item())

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dqn.parameters(), 1.0)
            opt.step()

        if step > 0 and step % args.target_update == 0:
            target.load_state_dict(dqn.state_dict())

            if len(stats["rewards"]) > 100:
                avg_reward = float(np.mean(stats["rewards"][-100:]))
                avg_loss = float(np.mean(stats["losses"][-100:])) if stats["losses"] else 0.0
                avg_pressure = float(np.mean(stats["pressures"][-100:]))
                fog_pct = stats["fog_decisions"] / max(1, stats["fog_decisions"] + stats["cloud_decisions"])

                print(f"[Step {step:6d}] eps={eps:.3f} reward={avg_reward:.3f} "
                      f"loss={avg_loss:.4f} pressure={avg_pressure:.2f} fog%={fog_pct:.1%}")

        eps = max(eps_end, eps * eps_decay)

    print("\n✅ Entraînement terminé!")

    dqn.eval()
    with torch.no_grad():
        test_states = []
        for _ in range(200):
            task = random.choice(tasks)
            cpu = float(task["cpu_demand"])
            ram = float(task["ram_demand"])
            test_pressure = random.uniform(0.0, 2.0)
            st = make_state(cpu, ram, test_pressure, args.fog_cpu, 0.3)
            test_states.append(st)

        test_states = torch.stack(test_states)
        q_vals = dqn(test_states)
        fog_q = q_vals[:, 0].mean().item()
        cloud_q = q_vals[:, 1].mean().item()
        print(f"\n📊 Valeurs Q finales: Fog={fog_q:.3f} | Cloud={cloud_q:.3f} | Préf={'Fog' if fog_q > cloud_q else 'Cloud'}")

    ckpt = {
        "arch": "DQN",
        "state_dict": dqn.state_dict(),
        "input_dim": 5,
        "output_dim": 2,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "fog_cpu": int(args.fog_cpu),
        "training_steps": int(args.steps),
        "epsilon_final": float(eps),
        "notes": "Compatible test.py",
        "stats": {
            "total_steps": int(args.steps),
            "fog_decisions": int(stats["fog_decisions"]),
            "cloud_decisions": int(stats["cloud_decisions"]),
        }
    }

    torch.save(ckpt, args.out)
    print(f"\n💾 Modèle DQN sauvegardé: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
