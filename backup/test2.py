import random
from collections import deque
from typing import List, Dict
import csv
import os
import torch

try:
    from edge_sim_py import (
        Simulator, EdgeServer, Application, Service,
        ContainerImage
    )
except ImportError:
    print("Error: edge-sim-py not installed. Please run: pip install edge-sim-py")
    raise


# ==========================================
# EDGE-SIM-PY: Global container image (NO LAYERS = FIX)
# ==========================================
def build_global_image_no_layers():
    """
    IMPORTANT FIX:
    - If layers_digests is empty, EdgeSimPy will NOT try to download layers from registries.
    - This avoids: IndexError: list index out of range (registries_with_layer empty).
    """
    image = ContainerImage()
    image.name = "task-image"
    image.size = 0
    image.layers_digests = []   # ✅ critical
    return image

GLOBAL_IMAGE = build_global_image_no_layers()


# ==========================================
# REAL LSTM MODEL (PyTorch)
# ==========================================
class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# MODULE 1: PREDICTION (LSTM Mock)
# ==========================================
class Module1_LSTMPredictor:
    def __init__(self, horizons: List[int] = None):
        self.horizons = horizons or [5, 15, 30, 60]
        print("   [Module 1] Initialisé (Prédicteur de charge LSTM - PyTorch).")
        
        # Initialisation du modèle LSTM
        self.model = LSTMModel(input_dim=1, hidden_dim=32, output_dim=1)
        # Note: Le modèle est initialisé avec des poids aléatoires. 
        # En production, on chargerait les poids : self.model.load_state_dict(torch.load("model.pth"))

    def predict(self, history: deque) -> Dict[int, Dict[str, float]]:
        if len(history) < 2:
            return {h: {'prediction': 0.1, 'uncertainty': 0.05} for h in self.horizons}

        # Préparation des données pour PyTorch
        data = list(history)
        max_val = max(data) if max(data) > 0 else 1.0
        norm_data = [x / max_val for x in data] # Normalisation
        
        input_tensor = torch.tensor(norm_data, dtype=torch.float32).view(1, -1, 1)
        
        with torch.no_grad():
            prediction_norm = self.model(input_tensor).item()
            
        prediction = prediction_norm * max_val
        
        predictions = {}
        for i, h in enumerate(self.horizons):
            # Extrapolation simple pour les horizons multiples (car modèle single-output ici)
            pred_h = prediction 
            uncertainty = pred_h * 0.1 
            predictions[h] = {'prediction': abs(pred_h), 'uncertainty': uncertainty}
        return predictions


# ==========================================
# MODULE 2: PROACTIVE PLANNING (H-VWPO Mock)
# ==========================================
class Module2_ProactivePlanner:
    def __init__(self):
        print("   [Module 2] Initialisé (Planificateur proactif H-VWPO Mock).")

    def plan(self, predictions: Dict[int, Dict[str, float]], fog_node: EdgeServer) -> Dict:
        # 1. Aggregation: Take the maximum robust load across all horizons to be conservative
        max_robust_load = 0.0
        for h, data in predictions.items():
            # Robust load = prediction + uncertainty
            r_load = data['prediction'] + data['uncertainty']
            if r_load > max_robust_load:
                max_robust_load = r_load
        
        robust_load_prediction = max_robust_load

        plan = {'scale_decision': 'none', 'offload_ratio': 0.0}
        
        # Context
        current_cpu = fog_node.cpu
        base_cpu = getattr(fog_node, 'base_cpu', fog_node.cpu)
        
        # Constraints
        MAX_CAPACITY = base_cpu * 2
        MIN_CAPACITY = int(base_cpu * 0.5)

        # 2. Target Utilization Strategy (e.g., aim for 70% load to leave headroom)
        TARGET_UTIL = 0.70
        
        # Calculate required CPU to meet target utilization
        # robust_load_prediction is a ratio of CURRENT capacity
        required_cpu = (robust_load_prediction * current_cpu) / TARGET_UTIL

        # 3. Decision Logic with Hysteresis (Buffer)
        # Only scale if deviation is significant (>10%) to avoid oscillation
        if required_cpu > current_cpu * 1.1:
            plan['scale_decision'] = 'up' if current_cpu < MAX_CAPACITY else 'none'
        elif required_cpu < current_cpu * 0.9:
            plan['scale_decision'] = 'down' if current_cpu > MIN_CAPACITY else 'none'

        # 4. Offloading Logic
        # Estimate capacity for next step
        next_capacity = current_cpu
        if plan['scale_decision'] == 'up':
            next_capacity = min(MAX_CAPACITY, current_cpu + 50)
        elif plan['scale_decision'] == 'down':
            next_capacity = max(MIN_CAPACITY, current_cpu - 50)
            
        # Calculate absolute load demand
        absolute_load = robust_load_prediction * current_cpu
        
        # If demand exceeds next capacity, offload the excess
        if absolute_load > next_capacity:
            excess = absolute_load - next_capacity
            plan['offload_ratio'] = excess / absolute_load
        else:
            plan['offload_ratio'] = 0.0

        return plan


# ==========================================
# MODULE 3: SCHEDULING (DRL - PyTorch DQN)
# ==========================================
class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class Module3_DRLScheduler:
    def __init__(self, state_dim=5, action_dim=2):
        print("   [Module 3] Initialisé (Ordonnanceur DRLMOTS - PyTorch DQN).")
        # state_dim = 5 : [cpu_req, ram_req, fog_util, pred_load, uncertainty]
        # action_dim = 2 : [Fog, Cloud]
        self.model = DQN(state_dim, action_dim)
        self.epsilon = 0.1 # Facteur d'exploration (10% aléatoire)

    def decide_fog_vs_cloud(self, service: Service, system_state: Dict, predictions: Dict) -> str:
        # 1. Construction du vecteur d'état (State Vector) selon Algorithme 3
        horizon = min(predictions.keys()) if predictions else 5
        pred_data = predictions.get(horizon, {'prediction': 0, 'uncertainty': 0})
        
        # Normalisation simple
        state = [
            service.cpu_demand / 500.0,      # Normalisé approx
            service.memory_demand / 1024.0,
            system_state.get('fog_utilization', 0),
            pred_data['prediction'],
            pred_data['uncertainty']
        ]
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # 2. Décision (Forward Pass du DQN)
        if random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_values = self.model(state_t)
                action = torch.argmax(q_values).item()
        
        return "Fog" if action == 0 else "Cloud"

    def select_fog_node(self, service: Service, fog_nodes: List[EdgeServer]) -> EdgeServer:
        # Note: L'implémentation complète FAHP/FTOPSIS (MCDM) nécessiterait des matrices de poids.
        # Pour l'instant, on garde la sélection du premier nœud disponible.
        return fog_nodes[0] if fog_nodes else None


# ==========================================
# SIMULATOR ENGINE STATE
# ==========================================
module1 = Module1_LSTMPredictor()
module2 = Module2_ProactivePlanner()
module3 = Module3_DRLScheduler()

proactive_plan = {'scale_decision': 'none', 'offload_ratio': 0.0}
latest_predictions = {}
metrics_history = deque(maxlen=100)
current_simulation_step = 0

workload_buffer = []
active_services = []


def load_dataset(filename="workload.csv"):
    global workload_buffer
    workload_buffer = []
    if not os.path.exists(filename):
        print(f"Dataset {filename} not found.")
        return

    with open(filename, 'r', newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['timestamp'] = int(row['timestamp'])
            row['cpu_demand'] = int(row['cpu_demand'])
            row['ram_demand'] = int(row['ram_demand'])
            row['duration'] = int(row['duration'])
            workload_buffer.append(row)

    print(f"Loaded {len(workload_buffer)} tasks from dataset.")


def proactive_placement_algorithm(parameters):
    global current_simulation_step, active_services, latest_predictions, proactive_plan

    simulator = parameters['simulator']

    fog_node = [s for s in EdgeServer.all() if "Fog" in s.name][0]
    cloud_node = [s for s in EdgeServer.all() if "Cloud" in s.name][0]

    if not hasattr(fog_node, 'base_cpu'):
        fog_node.base_cpu = fog_node.cpu

    # Simulate completion (simple duration)
    remaining = []
    for service in active_services:
        service.duration -= 1
        if service.duration > 0:
            remaining.append(service)
    active_services = remaining

    # Utilization proxy (because cpu_demand may not reflect your custom tasks)
    # Here we approximate: active services CPU demand / fog cpu
    # FIX: Count all services NOT on Cloud as Fog demand (including those waiting/failed)
    active_cpu = sum(s.cpu_demand for s in active_services if getattr(s, "server", None) != cloud_node)
    current_util = active_cpu / fog_node.cpu if fog_node.cpu > 0 else 0
    metrics_history.append(current_util)

    # Proactive cycle every W steps
    W_WINDOW = 10
    if current_simulation_step > 0 and current_simulation_step % W_WINDOW == 0:
        print(f"\n[t={current_simulation_step}] === Cycle Proactif (MAPE) ===")
        latest_predictions = module1.predict(metrics_history)
        proactive_plan = module2.plan(latest_predictions, fog_node)

        decision = proactive_plan['scale_decision']
        if decision == 'up':
            fog_node.cpu = min(fog_node.base_cpu * 2, fog_node.cpu + 50)
            print(f"    Scale UP: New CPU {fog_node.cpu}")
        elif decision == 'down':
            fog_node.cpu = max(int(fog_node.base_cpu * 0.5), fog_node.cpu - 50)
            print(f"    Scale DOWN: New CPU {fog_node.cpu}")

    # Inject tasks at timestamp t
    current_tasks = [t for t in workload_buffer if t['timestamp'] == current_simulation_step]

    for task_data in current_tasks:
        app = Application()
        app.name = f"App-{task_data.get('task_id','X')}"
        app.image = GLOBAL_IMAGE
        app.model = simulator

        service = Service(cpu_demand=task_data['cpu_demand'],
                          memory_demand=task_data['ram_demand'])
        service.name = f"Task-{task_data.get('task_id','X')}"
        service.application = app
        service.image = GLOBAL_IMAGE
        service.model = simulator
        service.duration = task_data['duration']

        active_services.append(service)

        system_state = {
            'fog_utilization': current_util,
            'proactive_offload_ratio': proactive_plan['offload_ratio']
        }

        target_type = module3.decide_fog_vs_cloud(service, system_state, latest_predictions)

        if target_type == "Fog":
            target_server = module3.select_fog_node(service, [fog_node])
            print(f"   [Placement] {service.name} ({task_data.get('service_type','NA')}) -> Fog")
            service.provision(target_server)
        else:
            print(f"   [Placement] {service.name} ({task_data.get('service_type','NA')}) -> Cloud")
            service.provision(cloud_node)

    current_simulation_step += 1


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    simulator = Simulator(tick_duration=1, tick_unit="seconds")
    simulator.stopping_criterion = lambda sim: current_simulation_step >= 100

    # Topology
    cloud = EdgeServer(cpu=1000, memory=100000, disk=100000)
    cloud.name = "Cloud-AWS"
    cloud.coordinates = [10, 10]

    fog = EdgeServer(cpu=100, memory=4096, disk=10000)
    fog.name = "Fog-1"
    fog.coordinates = [5, 5]

    # Load workload
    load_dataset("workload.csv")

    print("--- Démarrage de la simulation avec edge-sim-py ---")

    if hasattr(simulator, 'run_model'):
        simulator.resource_management_algorithm = proactive_placement_algorithm
        simulator.resource_management_algorithm_parameters = {"simulator": simulator}
        simulator.run_model()
    elif hasattr(simulator, 'run'):
        simulator.run(algorithm=proactive_placement_algorithm)
    elif hasattr(simulator, 'start'):
        simulator.algorithm = proactive_placement_algorithm
        simulator.start()
    else:
        print(f"Error: Simulator has no 'run_model', 'run', or 'start'. Available: {dir(simulator)}")

    print("--- Fin de la simulation ---")
