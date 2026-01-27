# testUnitaire.py
import unittest
import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import tempfile
import shutil
import sys
from pathlib import Path

# --- Configuration du chemin pour les imports ---
# Ajoute la racine du projet au PYTHONPATH pour trouver les modules custom
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- Imports depuis le projet ---
# On importe les classes et fonctions à tester
from project.sim_core import (
    EnhancedLSTM,
    DQN,
    Module1_LSTMPredictor,
    Module2_HVWPO_Planner,
    Module3_Scheduler,
    _normalize_task_row,
    load_workload_indexed,
)

# --- Classe de base pour les tests avec fichiers temporaires ---

class BaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Crée un dossier temporaire pour les artefacts de test (modèles, datasets)
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        # Supprime le dossier temporaire et son contenu après les tests
        shutil.rmtree(cls.temp_dir)

# --- Tests des modèles PyTorch (LSTM & DQN) ---

class TestModels(BaseTest):
    def test_enhanced_lstm_forward_pass(self):
        """Teste si le forward pass du modèle EnhancedLSTM fonctionne avec un tenseur de la bonne taille."""
        seq_len = 30
        batch_size = 4
        model = EnhancedLSTM(input_dim=1, hidden_dim=64, num_layers=1, dropout=0.1)
        
        # Crée un tenseur d'entrée factice
        dummy_input = torch.randn(batch_size, seq_len, 1)
        
        # Exécute le forward pass
        output = model(dummy_input)
        
        # Vérifie que la sortie a la bonne forme (batch_size, 1)
        self.assertEqual(output.shape, (batch_size, 1))

    def test_dqn_forward_pass(self):
        """Teste si le forward pass du modèle DQN fonctionne."""
        input_dim = 5
        output_dim = 2
        batch_size = 4
        model = DQN(input_dim=input_dim, output_dim=output_dim, hidden_dim=64, dropout=0.1)
        
        # Crée un tenseur d'entrée factice
        dummy_input = torch.randn(batch_size, input_dim)
        
        # Exécute le forward pass
        output = model(dummy_input)
        
        # Vérifie que la sortie a la bonne forme (batch_size, output_dim)
        self.assertEqual(output.shape, (batch_size, output_dim))

# --- Tests du chargement et de la normalisation des données ---

class TestDataLoading(BaseTest):
    def test_normalize_task_row_native_format(self):
        """Teste la normalisation pour le format de données déjà traité."""
        row = {
            "task_id": "101",
            "timestamp": "10.0",
            "service_type": "typeA",
            "cpu_demand": "150.5",
            "ram_demand": "256.0",
            "duration": "5.0",
        }
        normalized = _normalize_task_row(row)
        expected = {
            "task_id": 101,
            "timestamp": 10,
            "service_type": "typeA",
            "cpu_demand": 150,
            "ram_demand": 256,
            "duration": 5,
        }
        self.assertEqual(normalized, expected)

    def test_normalize_task_row_tuple30k_format(self):
        """Teste la normalisation pour le format de données brut (style Tuple30K)."""
        row = {
            "TaskID": "202",
            "GenerationTime": "20.0",
            "TaskSize": "1000",
            "CyclesPerBit": "1000",
            "TransBitRate": "500",
            "DataType": "sensor",
        }
        # Note: les valeurs attendues dépendent des formules dans _normalize_task_row
        normalized = _normalize_task_row(row)
        self.assertEqual(normalized["task_id"], 202)
        self.assertEqual(normalized["timestamp"], 20)
        self.assertIsInstance(normalized["cpu_demand"], int)
        self.assertIsInstance(normalized["ram_demand"], int)
        self.assertIsInstance(normalized["duration"], int)
        self.assertGreater(normalized["cpu_demand"], 0)
        self.assertGreater(normalized["ram_demand"], 0)
        self.assertGreater(normalized["duration"], 0)

    def test_load_workload_indexed(self):
        """Teste le chargement d'un CSV et son indexation par timestamp."""
        csv_content = (
            "task_id,timestamp,cpu_demand,ram_demand,duration\n"
            "1,5,100,128,10\n"
            "2,5,50,64,5\n"
            "3,8,200,256,8\n"
        )
        # Crée un fichier CSV temporaire
        csv_path = os.path.join(self.temp_dir, "workload.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)
            
        workload_idx = load_workload_indexed(csv_path)
        
        # Vérifie la structure
        self.assertIn(5, workload_idx)
        self.assertIn(8, workload_idx)
        self.assertNotIn(1, workload_idx)
        
        # Vérifie le contenu
        self.assertEqual(len(workload_idx[5]), 2)
        self.assertEqual(len(workload_idx[8]), 1)
        self.assertEqual(workload_idx[5][0]["task_id"], 1)
        self.assertEqual(workload_idx[8][0]["task_id"], 3)

# --- Tests des 3 modules principaux de la simulation ---

class TestModules(BaseTest):

    @classmethod
    def setUpClass(cls):
        """Crée des modèles factices pour les tests des modules."""
        super().setUpClass()
        
        # Crée un modèle LSTM factice
        cls.lstm_path = os.path.join(cls.temp_dir, "dummy_lstm.pth")
        lstm_model = EnhancedLSTM(hidden_dim=32)
        ckpt_lstm = {
            "state_dict": lstm_model.state_dict(),
            "seq_len": 30,
            "max_util": 1.5,
            "hidden_dim": 32,
            "num_layers": 2,
            "dropout": 0.1,
        }
        torch.save(ckpt_lstm, cls.lstm_path)
        
        # Crée un modèle DQN factice
        cls.dqn_path = os.path.join(cls.temp_dir, "dummy_dqn.pth")
        dqn_model = DQN(hidden_dim=32)
        ckpt_dqn = {
            "state_dict": dqn_model.state_dict(),
            "hidden_dim": 32,
            "dropout": 0.1,
        }
        torch.save(ckpt_dqn, cls.dqn_path)

    def test_module1_lstm_predictor_loading(self):
        """Teste le chargement correct et l'échec du Module1."""
        # Test 1: Chargement réussi
        predictor_ok = Module1_LSTMPredictor(model_path=self.lstm_path)
        self.assertTrue(predictor_ok.model_loaded)
        self.assertIsNotNone(predictor_ok.model)
        
        # Test 2: Chemin invalide -> fallback
        predictor_fail = Module1_LSTMPredictor(model_path="path/non/existent.pth")
        self.assertFalse(predictor_fail.model_loaded)
        self.assertIsNone(predictor_fail.model)

    def test_module1_lstm_prediction(self):
        """Teste la sortie de la méthode de prédiction (fallback et modèle)."""
        history = deque(np.random.rand(50).tolist(), maxlen=200)
        
        # Test 1: Prédiction avec fallback
        predictor_fail = Module1_LSTMPredictor(model_path="invalid.pth")
        preds_fallback = predictor_fail.predict(history)
        self.assertIn(5, preds_fallback) # horizon 5
        self.assertIn("prediction", preds_fallback[5])
        self.assertTrue(preds_fallback[5]["used_fallback"])
        
        # Test 2: Prédiction avec modèle chargé
        predictor_ok = Module1_LSTMPredictor(model_path=self.lstm_path)
        preds_model = predictor_ok.predict(history)
        self.assertIn(5, preds_model)
        self.assertIn("prediction", preds_model[5])
        self.assertFalse(preds_model[5]["used_fallback"])

    def test_module1_prediction_logic(self):
        """Teste la logique de prédiction du LSTM dans des scénarios spécifiques."""
        from unittest.mock import MagicMock

        predictor = Module1_LSTMPredictor(model_path=self.lstm_path)
        self.assertTrue(predictor.model_loaded, "Le modèle LSTM factice doit être chargé pour ce test.")

        # --- MOCK pour rendre le test déterministe ---
        # Le modèle LSTM est initialisé aléatoirement, donc ses prédictions sont imprévisibles.
        # On remplace l'appel au modèle par un Mock qui retourne une valeur contrôlée.
        # Dans sim_core.py: y_norm = float(self.model(x).item())
        predictor.model = MagicMock()

        # Scénario 1: Pression basse et stable
        # On simule une sortie modèle basse (ex: 0.1). Avec max_util=1.5 -> pred ~ 0.15
        predictor.model.return_value.item.return_value = 0.1
        
        history_low = deque([0.15] * 50, maxlen=200)
        preds_low = predictor.predict(history_low)
        pred_val_low = preds_low[5]["prediction"]
        self.assertLess(pred_val_low, 0.5, "Avec une histoire basse et stable, la prédiction doit rester basse.")

        # Scénario 2: Pression haute et stable
        # On simule une sortie modèle haute (ex: 0.6). Avec max_util=1.5 -> pred ~ 0.9
        predictor.model.return_value.item.return_value = 0.6

        history_high = deque([0.9] * 50, maxlen=200)
        preds_high = predictor.predict(history_high)
        pred_val_high = preds_high[5]["prediction"]
        self.assertGreater(pred_val_high, 0.7, "Avec une histoire haute et stable, la prédiction doit rester haute.")

        # Scénario 3: Test de la logique de fallback (règle spécifique)
        predictor_fallback = Module1_LSTMPredictor(model_path="invalid.pth")
        history_fallback_low = deque([0.1] * 10, maxlen=200) # last_p=0.1, mean_last_5=0.1
        preds_fallback_low = predictor_fallback.predict(history_fallback_low)
        # La règle est: si last_p < 0.2 et mean_last_5 < 0.3, pred = last_p * 0.7
        expected_pred = 0.1 * 0.7
        self.assertAlmostEqual(preds_fallback_low[5]["prediction"], expected_pred, places=4, msg="Le fallback doit appliquer la règle de réduction pour une pression très basse.")

    def test_module2_planner_decisions(self):
        """Teste les décisions logiques du planificateur H-VWPO."""
        # Scénario 1: Scale UP (charge prédite > capacité * seuil)
        # Chaque scénario utilise une instance de planner distincte pour éviter les effets de bord (ex: cooldown).
        planner_up = Module2_HVWPO_Planner(target_util=0.7, cooldown_windows=2)
        predictions_high = {5: {"prediction": 1.5, "uncertainty": 0.2}} # Forte charge prédite
        plan_up = planner_up.plan(predictions_high, total_fog_capacity=100, total_incoming_demand=50,
                               current_pressure=1.2, current_t=100, W_window=10)
        self.assertEqual(plan_up["scale_decision"], "up")
        
        # Scénario 2: Scale DOWN (pression basse et stable)
        planner_down = Module2_HVWPO_Planner(target_util=0.7, cooldown_windows=2)
        predictions_low = {5: {"prediction": 0.2, "uncertainty": 0.1}}
        planner_down.low_pressure_windows = 3 # Simule une pression basse stable
        plan_down = planner_down.plan(predictions_low, total_fog_capacity=200, total_incoming_demand=10,
                                 current_pressure=0.3, current_t=100, W_window=10)
        self.assertEqual(plan_down["scale_decision"], "down")
        
        # Scénario 3: Offload (pression actuelle très haute)
        planner_offload = Module2_HVWPO_Planner(target_util=0.7, cooldown_windows=2)
        # On réutilise les prédictions hautes, mais avec une pression actuelle qui déclenche la sécurité
        plan_offload = planner_offload.plan(predictions_high, total_fog_capacity=100, total_incoming_demand=50,
                                    current_pressure=1.1, current_t=100, W_window=10)
        self.assertGreater(plan_offload["offload_ratio"], 0)
        self.assertIn("SAFETY", plan_offload["offload_reason"])

    def test_module3_scheduler_decisions(self):
        """Teste les décisions du scheduler (DQN, baseline, override)."""
        # Test 1: Scheduler avec DQN
        scheduler_dqn = Module3_Scheduler(dqn_path=self.dqn_path)
        self.assertTrue(scheduler_dqn.use_dqn)
        
        # Test 2: Scheduler en mode baseline
        scheduler_base = Module3_Scheduler(dqn_path="invalid.pth")
        self.assertFalse(scheduler_base.use_dqn)
        
        # Test 3: Décision baseline - Tâche normale
        decision_base, fallback = scheduler_base.decide(task_cpu=100, task_ram=128, pressure=0.5,
                                                        fog_cpu=200, offload_ratio=0.1, t=20)
        self.assertIn(decision_base, ["Fog", "Cloud"])
        self.assertTrue(fallback)
        
        # Test 4: Décision baseline - Tâche très gourmande
        # Doit toujours aller sur le Cloud, peu importe la pression
        decision_base_heavy, _ = scheduler_base.decide(task_cpu=scheduler_base.cpu_threshold_cloud + 1, task_ram=128,
                                                       pressure=0.1, fog_cpu=200, offload_ratio=0.0, t=20)
        self.assertEqual(decision_base_heavy, "Cloud", "Une tâche dépassant le seuil CPU doit aller sur le Cloud en mode baseline.")
        
        # Test 5: Décision avec DQN
        decision_dqn, fallback_dqn = scheduler_dqn.decide(task_cpu=100, task_ram=128, pressure=0.5,
                                                          fog_cpu=200, offload_ratio=0.1, t=20)
        self.assertIn(decision_dqn, ["Fog", "Cloud"])
        self.assertFalse(fallback_dqn)
        
        # Test 6: Safety Override (pression > 0.95)
        decision_safe, _ = scheduler_dqn.decide(task_cpu=10, task_ram=128, pressure=0.98,
                                                fog_cpu=200, offload_ratio=0.0, t=20)
        self.assertEqual(decision_safe, "Cloud")

        # Test 7: Décision DQN - Pression très basse
        # Pour une tâche raisonnable et une pression très basse, le DQN devrait préférer le Fog.
        # On ne peut pas garantir la décision finale, mais on peut vérifier les Q-values qui la motivent.
        cpu_norm = 50 / 500.0
        ram_norm = 128 / 4096.0
        pressure_clip = 0.1
        fog_cpu_norm = 200 / 200.0
        offload_ratio = 0.0
        state = torch.tensor([cpu_norm, ram_norm, pressure_clip, fog_cpu_norm, offload_ratio], dtype=torch.float32)
        with torch.no_grad():
            q_vals = scheduler_dqn.dqn(state.unsqueeze(0))
            # L'action 0 est "Fog", l'action 1 est "Cloud"
            self.assertGreater(q_vals[0][0], q_vals[0][1],
                               "Pour une pression très basse, le DQN devrait avoir une Q-value plus élevée pour le Fog.")

# --- Exécution des tests ---

if __name__ == '__main__':
    """
    Lance la suite de tests unitaires.
    Exécutez ce fichier directement pour tester les composants du projet.
    """
    print("="*70)
    print("Lancement des tests unitaires pour le simulateur Fog-Cloud")
    print("="*70)
    unittest.main(verbosity=2)