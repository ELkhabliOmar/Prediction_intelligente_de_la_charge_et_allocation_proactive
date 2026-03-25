import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# 1) ENTRE TES VALEURS ICI
# =========================================================

# Noms des nœuds Fog
fog_nodes = ["f0", "f1", "f2", "f3", "f4"]

# Consommation énergétique Fog - approche proactive (LSTM)
fog_lstm = [42000, 28000, 38000, 12000, 38000]

# Consommation énergétique Fog - baseline ARIMA+TOPSIS
fog_arima = [48000, 28000, 38000, 22000, 38000]

# Noms des nœuds Cloud
cloud_nodes = ["c0", "c1"]

# Charge moyenne Cloud - approche proactive (LSTM)
cloud_lstm = [140, 135]

# Charge moyenne Cloud - baseline ARIMA+TOPSIS
cloud_arima = [155, 150]

# =========================================================
# 2) GRAPHIQUE COMPARATIF FOG
# =========================================================

x_fog = np.arange(len(fog_nodes))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x_fog - width/2, fog_lstm, width, label="Approche proactive (LSTM)")
plt.bar(x_fog + width/2, fog_arima, width, label="Baseline ARIMA+TOPSIS")

plt.xticks(x_fog, fog_nodes)
plt.xlabel("Nœuds Fog")
plt.ylabel("Consommation énergétique totale")
plt.title("Comparaison de la consommation énergétique des nœuds Fog")
plt.legend()
plt.tight_layout()
plt.savefig("comparaison_fog.png", dpi=300)
plt.show()

# =========================================================
# 3) GRAPHIQUE COMPARATIF CLOUD
# =========================================================

x_cloud = np.arange(len(cloud_nodes))

plt.figure(figsize=(7, 5))
plt.bar(x_cloud - width/2, cloud_lstm, width, label="Approche proactive (LSTM)")
plt.bar(x_cloud + width/2, cloud_arima, width, label="Baseline ARIMA+TOPSIS")

plt.xticks(x_cloud, cloud_nodes)
plt.xlabel("Nœuds Cloud")
plt.ylabel("Charge moyenne")
plt.title("Comparaison de la charge moyenne des nœuds Cloud")
plt.legend()
plt.tight_layout()
plt.savefig("comparaison_cloud.png", dpi=300)
plt.show()