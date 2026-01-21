# generate_workload.py
# Génère:
#  - data/workload.csv       (train: mix low/medium/high + phases à 0)
#  - data/workload_up.csv    (test: pressure ~0.3->1.2 avant t=10, puis faible)
#  - data/workload_down.csv  (test: faible avant et après t=10)

import os
import csv
import argparse
import random
from typing import List, Dict

FIELDS = ["task_id", "timestamp", "service_type", "cpu_demand", "ram_demand", "duration"]

SERVICE_TYPES = ["Monitoring", "IrrigationControl", "ImageAnalysis"]

def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _pick_service_type(rng: random.Random) -> str:
    # On garde ImageAnalysis mais avec CPU modéré (sinon placement offline -> Cloud)
    p = rng.random()
    if p < 0.55:
        return "Monitoring"
    if p < 0.85:
        return "IrrigationControl"
    return "ImageAnalysis"

def _make_task(task_id: int, t: int, rng: random.Random,
               cpu_range=(8, 30), ram_range=(64, 512), dur_range=(6, 14)) -> Dict:
    return {
        "task_id": task_id,
        "timestamp": t,
        "service_type": _pick_service_type(rng),
        "cpu_demand": rng.randint(*cpu_range),
        "ram_demand": rng.randint(*ram_range),
        "duration": rng.randint(*dur_range),
    }

def generate_workload_train(ticks: int, seed: int) -> List[Dict]:
    """
    Dataset d'entraînement: plusieurs régimes de charge + phases à 0
    pour que le LSTM apprenne aussi le "retour à zéro".
    """
    rng = random.Random(seed)
    rows: List[Dict] = []
    tid = 0

    # Régimes (t windows):
    # - 0..19   : faible
    # - 20..39  : moyen
    # - 40..49  : zéro (silence)
    # - 50..69  : fort (mais réaliste)
    # - 70..79  : faible
    # - 80..89  : zéro (silence)
    # - 90..end : moyen
    for t in range(ticks):
        if 0 <= t <= 19:
            n = rng.randint(0, 2)                 # faible
            cpu_rng = (8, 18)
            dur_rng = (5, 10)
        elif 20 <= t <= 39:
            n = rng.randint(2, 4)                 # moyen
            cpu_rng = (10, 26)
            dur_rng = (6, 12)
        elif 40 <= t <= 49:
            n = 0                                 # silence => pressure=0
            cpu_rng = (8, 18)
            dur_rng = (5, 10)
        elif 50 <= t <= 69:
            n = rng.randint(4, 7)                 # fort (mais pas délirant)
            cpu_rng = (12, 35)
            dur_rng = (8, 16)
        elif 70 <= t <= 79:
            n = rng.randint(1, 2)                 # faible
            cpu_rng = (8, 18)
            dur_rng = (5, 10)
        elif 80 <= t <= 89:
            n = 0                                 # silence
            cpu_rng = (8, 18)
            dur_rng = (5, 10)
        else:
            n = rng.randint(2, 4)                 # moyen
            cpu_rng = (10, 26)
            dur_rng = (6, 12)

        for _ in range(n):
            rows.append(_make_task(tid, t, rng, cpu_range=cpu_rng, dur_range=dur_rng))
            tid += 1

    return rows

def generate_workload_up(seed: int) -> List[Dict]:
    """
    Objectif:
      - avant t=10: pressure ~0.3 -> 1.2 (FogCPU=100 => active_cpu ~30 -> 120)
      - après t=10: charge faible pour que ça redescende et scale down apparaisse
    On obtient ce profil grâce à des durées 10..14 avant t=10 (accumulation),
    puis très peu d'injections après.
    """
    rng = random.Random(seed + 1)
    rows: List[Dict] = []
    tid = 0

    # Phase "ramp-up" 0..9 : injecte 2 tâches/tick, CPU modéré, durée assez longue => accumulation
    for t in range(0, 10):
        for _ in range(2):
            rows.append(_make_task(
                tid, t, rng,
                cpu_range=(12, 22),      # ~ 12-22 => 2 tâches => 24-44 cpu/tick
                ram_range=(128, 512),
                dur_range=(10, 14)       # accumulation => pressure monte vers ~1.0
            ))
            tid += 1

    # Phase "post" 10..24 : faible (1 tâche tous les 2 ticks)
    for t in range(10, 25):
        if t % 2 == 0:
            rows.append(_make_task(
                tid, t, rng,
                cpu_range=(6, 14),
                ram_range=(64, 256),
                dur_range=(4, 8)
            ))
            tid += 1

    # Phase silence 25..40 : rien (retour à 0 clair)
    # => pas de lignes

    return rows

def generate_workload_down(seed: int) -> List[Dict]:
    """
    Charge faible => scale down à t=10 (W=10).
    """
    rng = random.Random(seed + 2)
    rows: List[Dict] = []
    tid = 0

    # 0..9 : très faible (1 tâche / tick, CPU petit, durée courte)
    for t in range(0, 10):
        rows.append(_make_task(
            tid, t, rng,
            cpu_range=(4, 10),
            ram_range=(64, 192),
            dur_range=(4, 7)
        ))
        tid += 1

    # 10..18 : encore plus faible (1 tâche tous les 3 ticks)
    for t in range(10, 19):
        if t % 3 == 0:
            rows.append(_make_task(
                tid, t, rng,
                cpu_range=(4, 8),
                ram_range=(64, 160),
                dur_range=(3, 6)
            ))
            tid += 1

    # 19..40 : silence
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_rows = generate_workload_train(ticks=args.ticks, seed=args.seed)
    up_rows = generate_workload_up(seed=args.seed)
    down_rows = generate_workload_down(seed=args.seed)

    train_path = os.path.join(args.out_dir, "workload.csv")
    up_path = os.path.join(args.out_dir, "workload_up.csv")
    down_path = os.path.join(args.out_dir, "workload_down.csv")

    write_csv(train_path, train_rows)
    write_csv(up_path, up_rows)
    write_csv(down_path, down_rows)

    print(f"[OK] {train_path} généré : {len(train_rows)} tâches sur {args.ticks} ticks.")
    print(f"[OK] {up_path} généré : {len(up_rows)} tâches (ramp-up vers t=10 puis faible).")
    print(f"[OK] {down_path} généré : {len(down_rows)} tâches (faible => down à t=10).")

if __name__ == "__main__":
    main()
