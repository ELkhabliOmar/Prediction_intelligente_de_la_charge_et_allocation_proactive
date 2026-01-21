
import csv
import random
import math
import os

DATA_DIR = "data"

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "timestamp", "service_type", "cpu_demand", "ram_demand", "duration"])
        for r in rows:
            w.writerow(r)

def generate_workload(filename=os.path.join(DATA_DIR, "workload.csv"), duration=100, seed=42):
    """
    Dataset générique (workload.csv) : charge variable (sinus + bruit).
    """
    random.seed(seed)
    ensure_dirs()

    rows = []
    task_id = 0
    for t in range(duration):
        load_intensity = 5 + 5 * math.sin(t * 0.1)
        num_tasks = int(load_intensity + random.randint(-2, 2))
        num_tasks = max(0, num_tasks)

        for _ in range(num_tasks):
            service_type = random.choice(["Monitoring", "IrrigationControl", "ImageAnalysis"])

            if service_type == "Monitoring":
                cpu = random.randint(5, 15)
                ram = random.randint(16, 64)
                dur = random.randint(5, 10)
            elif service_type == "IrrigationControl":
                cpu = random.randint(20, 40)
                ram = random.randint(64, 128)
                dur = random.randint(10, 20)
            else:  # ImageAnalysis
                cpu = random.randint(100, 500)
                ram = random.randint(256, 1024)
                dur = random.randint(20, 60)

            rows.append([task_id, t, service_type, cpu, ram, dur])
            task_id += 1

    write_csv(filename, rows)
    print(f"[OK] {filename} généré : {task_id} tâches sur {duration} ticks.")

def generate_workload_up(filename=os.path.join(DATA_DIR, "workload_up.csv")):
    """
    workload_up.csv :
    - très forte charge sur Fog entre t=0..9 (beaucoup de tâches <= 300 CPU => restent Fog)
    - durées longues => elles restent actives jusqu’à t=10
    => à t=10 : util élevée => Scale UP attendu.
    """
    ensure_dirs()
    rows = []
    task_id = 0

    # Charge forte: 3 tâches "medium" par tick, durée 30, CPU 90 (reste Fog, lourde mais <=300)
    for t in range(0, 10):
        for _ in range(3):
            rows.append([task_id, t, "IrrigationControl", 90, 200, 30])
            task_id += 1

    # Après t=10: charge plus faible
    for t in range(10, 25):
        rows.append([task_id, t, "Monitoring", 8, 32, 5])
        task_id += 1

    write_csv(filename, rows)
    print(f"[OK] {filename} généré : {task_id} tâches (Scale UP vers t=10).")

def generate_workload_down(filename=os.path.join(DATA_DIR, "workload_down.csv")):
    """
    workload_down.csv :
    - charge très faible (petites tâches) => util basse
    => à t=10 : Scale DOWN attendu.
    """
    ensure_dirs()
    rows = []
    task_id = 0

    for t in range(0, 25):
        rows.append([task_id, t, "Monitoring", 5, 16, 3])
        task_id += 1

    write_csv(filename, rows)
    print(f"[OK] {filename} généré : {task_id} tâches (Scale DOWN vers t=10).")

if __name__ == "__main__":
    generate_workload()
    generate_workload_up()
    generate_workload_down()
    
