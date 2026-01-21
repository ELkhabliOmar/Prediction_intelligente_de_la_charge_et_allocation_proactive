# train_dqn.py
import os
import csv
import random
import argparse
from collections import deque
from typing import List, Dict

import torch
import torch.nn as nn


# -------------------------
# DQN: 5 features -> 2 actions (Fog/Cloud)
# state = [cpu_norm, ram_norm, pressure_clip, fog_cpu_norm, offload_ratio]
# action: 0 Fog, 1 Cloud
# -------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=5, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_workload(path: str) -> List[Dict]:
    tasks = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            tasks.append({
                "timestamp": int(row.get("timestamp", 0)),
                "cpu_demand": int(row["cpu_demand"]),
                "ram_demand": int(row["ram_demand"]),
                "duration": int(row.get("duration", 1)),
            })
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "workload.csv"))
    ap.add_argument("--out", default=os.path.join("models", "dqn_fog_cloud.pth"))
    ap.add_argument("--fog_cpu", type=int, default=100)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not os.path.exists(args.data):
        raise FileNotFoundError("Lance d'abord: python generate_workload.py")

    tasks = load_workload(args.data)
    if not tasks:
        raise RuntimeError("Dataset vide.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dqn = DQN(input_dim=5, output_dim=2).to(device)
    target = DQN(input_dim=5, output_dim=2).to(device)
    target.load_state_dict(dqn.state_dict())

    opt = torch.optim.Adam(dqn.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    replay = deque(maxlen=50000)
    gamma = 0.98

    # Exploration
    eps = 0.30
    eps_min = 0.05
    eps_decay = 0.9995

    # Paramètres "environnement simplifié"
    # pressure = active_cpu_fog / fog_cpu (peut dépasser 1)
    pressure = 0.0
    DECAY = 0.97  # simule fin des tâches (pressure diminue progressivement)

    # coûts / pénalités (à ajuster)
    LAMBDA_OVER = 10.0        # pénalise surcharge >1
    LAMBDA_CLOUD = 0.02       # coût cloud
    LAMBDA_FOG   = 0.001      # petite pénalité fog (énergie/usage)
    TARGET_UPDATE = 500
    WARMUP = 1000

    def make_state(cpu: float, ram: float, pressure_val: float, fog_cpu: float, offload_ratio: float):
        cpu_norm = min(cpu / 500.0, 5.0)
        ram_norm = min(ram / 4096.0, 5.0)

        # on clippe la pression pour l'entrée réseau (sinon instable)
        pressure_clip = min(max(pressure_val, 0.0), 3.0)  # 0..3

        fog_cpu_norm = min(fog_cpu / 200.0, 2.0)  # 100 -> 0.5, 200 -> 1.0 etc.

        return torch.tensor(
            [cpu_norm, ram_norm, pressure_clip, fog_cpu_norm, offload_ratio],
            dtype=torch.float32,
            device=device,
        )

    for step in range(args.steps):
        task = random.choice(tasks)
        cpu = float(task["cpu_demand"])
        ram = float(task["ram_demand"])
        dur = max(1.0, float(task.get("duration", 1)))

        # offload_ratio "fake" (si tu veux, tu peux le randomiser un peu)
        offload_ratio = 0.0

        state = make_state(cpu, ram, pressure, args.fog_cpu, offload_ratio)

        # epsilon-greedy
        if random.random() < eps:
            a = random.randint(0, 1)
        else:
            with torch.no_grad():
                qvals = dqn(state.unsqueeze(0))  # (1,2)
                a = int(torch.argmax(qvals, dim=1).item())

        # ----- transition -----
        # pression diminue (tâches finissent)
        pressure_next = pressure * DECAY

        # si Fog, la tâche ajoute de la pression proportionnellement à cpu/fog_cpu
        if a == 0:
            # pondération par duration: plus dur => plus de pression "moyenne" ressentie
            pressure_next += (cpu / max(args.fog_cpu, 1.0)) * min(1.0, dur / 5.0)
            cloud_cost = 0.0
            fog_cost = LAMBDA_FOG * (cpu / 100.0)
        else:
            cloud_cost = LAMBDA_CLOUD * (cpu / 100.0)
            fog_cost = 0.0

        overload = max(0.0, pressure_next - 1.0)
        reward = - (LAMBDA_OVER * overload) - cloud_cost - fog_cost

        # next_state: on simule le prochain état (même task pour simplifier)
        next_state = make_state(cpu, ram, pressure_next, args.fog_cpu, offload_ratio)

        replay.append((state, a, reward, next_state))

        # update pressure pour le prochain step
        pressure = pressure_next

        # ----- train -----
        if len(replay) >= WARMUP:
            batch = random.sample(replay, args.batch)
            S = torch.stack([b[0] for b in batch])                 # (B,5)
            A = torch.tensor([b[1] for b in batch], device=device) # (B,)
            R = torch.tensor([b[2] for b in batch], device=device) # (B,)
            NS = torch.stack([b[3] for b in batch])                # (B,5)

            q = dqn(S).gather(1, A.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                q_next = target(NS).max(1)[0]
                y = R + gamma * q_next

            loss = loss_fn(q, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dqn.parameters(), 1.0)
            opt.step()

        if step > 0 and step % TARGET_UPDATE == 0:
            target.load_state_dict(dqn.state_dict())
            print(f"[train_dqn] step={step} eps={eps:.3f} pressure={pressure:.2f}")

        eps = max(eps_min, eps * eps_decay)

    torch.save(
        {
            "state_dict": dqn.state_dict(),
            "input_dim": 5,
            "fog_cpu": args.fog_cpu,
            "notes": "Trained on pressure=active_cpu_fog/fog_cpu, state=[cpu,ram,pressure_clip,fog_cpu_norm,offload_ratio]."
        },
        args.out
    )
    print(f"[OK] DQN sauvegardé: {args.out}")


if __name__ == "__main__":
    main()
