import csv
import random
import math
import os

def generate_workload(filename="workload.csv", duration=100):
    """
    Generates a synthetic workload dataset aligned with the research paper.
    Includes: Timestamp, Service Type, CPU/RAM demand, Duration.
    """
    print(f"--- Generating {filename} ---")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header matching the Task Model (Section 1.3.1)
        writer.writerow(["task_id", "timestamp", "service_type", "cpu_demand", "ram_demand", "duration"])
        
        task_id = 0
        for t in range(duration):
            # Simulate variable load (e.g., Day/Night cycle or sensor bursts)
            # Using a sine wave + random noise
            load_intensity = 5 + 5 * math.sin(t * 0.1) 
            num_tasks = int(load_intensity + random.randint(-2, 2))
            num_tasks = max(0, num_tasks)
            
            for _ in range(num_tasks):
                # Randomly assign a service type with different characteristics
                service_type = random.choice(["Monitoring", "IrrigationControl", "ImageAnalysis"])
                
                if service_type == "Monitoring":
                    # Light task, latency sensitive
                    cpu = random.randint(5, 15)
                    ram = random.randint(16, 64)
                    dur = random.randint(5, 10)
                elif service_type == "IrrigationControl":
                    # Medium task
                    cpu = random.randint(20, 40)
                    ram = random.randint(64, 128)
                    dur = random.randint(10, 20)
                else: # ImageAnalysis
                    # Heavy task (likely offloaded to Cloud)
                    cpu = random.randint(100, 500)
                    ram = random.randint(256, 1024)
                    dur = random.randint(20, 60)
                
                writer.writerow([task_id, t, service_type, cpu, ram, dur])
                task_id += 1
                
    print(f"Successfully generated {task_id} tasks over {duration} time steps.")

if __name__ == "__main__":
    generate_workload()
