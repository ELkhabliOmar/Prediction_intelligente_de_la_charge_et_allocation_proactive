# Intelligent Load Prediction and Proactive Resource Allocation in Fog-Cloud Environments

This project implements a proactive resource allocation system for Fog-Cloud environments, specifically targeting agro-food IoT applications. It utilizes a **MAPE (Monitoring-Analysis-Planning-Execution)** loop to anticipate workload variations and optimize resource usage (latency, energy, cost) before congestion occurs.

## Architecture

The system is built on top of the `edge-sim-py` simulator and consists of three main modules:

### 1. Module 1: Multi-Horizon Load Prediction (Analysis)
*   **Goal:** Predict future resource usage at different horizons (e.g., 5, 15, 30, 60 minutes).
*   **Method:** Uses Long Short-Term Memory (LSTM) networks (currently mocked) to forecast load and calculate uncertainty margins for robust decision-making.

### 2. Module 2: Proactive Planning (Planning)
*   **Goal:** Determine strategic resource adjustments based on predictions.
*   **Method:** Implements the H-VWPO (Hybrid Vultures and Waterwheel Plant Optimization) logic.
*   **Decisions:**
    *   **Scaling:** Scale Fog node capacity Up or Down.
    *   **Offloading:** Set a global offloading ratio to the Cloud if Fog capacity is predicted to be exceeded.

### 3. Module 3: Proactive Scheduling (Execution)
*   **Goal:** Fine-grained placement of individual services/tasks.
*   **Method:**
    *   **Fog vs. Cloud:** Uses DRLMOTS (Deep Reinforcement Learning) logic to decide placement based on the proactive plan.
    *   **Fog Selection:** Uses FAHP/FTOPSIS (Multi-Criteria Decision Making) to select the optimal Fog node.

## Requirements

*   Python 3.8+
*   `edge-sim-py`
*   `numpy`
*   `networkx`
*   `torch`
*   `simpy`

## Installation

1.  Install the required dependencies:
    ```bash
    python -m pip install -r requirements.txt
    ```

## Project Structure

*   `test.py`: Main simulation script implementing the MAPE loop and the three decision modules.
*   `data_generator.py`: Utility script to generate synthetic IoT workload datasets (`workload.csv`).
*   `requirements.txt`: Python dependencies.

## Data Generation

The simulation requires a workload dataset (`workload.csv`) containing IoT task definitions.

*   **Automatic Generation:** The simulation script checks for `workload.csv` and generates it automatically if missing.
*   **Manual Generation:** To regenerate the dataset with new random parameters, run:
    ```bash
    python generate_workload.py
    ```

## Usage

Run the main simulation script:

```bash
python test.py
```

The simulation will initialize the Fog-Cloud topology and the application workflow. It will then execute the proactive MAPE cycle periodically, printing decisions (Scaling, Offloading, Placement) to the console.

---
Based on the research project by Omar EL-KHABLI.