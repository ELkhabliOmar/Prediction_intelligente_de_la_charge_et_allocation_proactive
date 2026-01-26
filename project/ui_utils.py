# project/ui_utils.py affichage + couleurs + tableaux + stats + save
import os
import csv
from typing import List, Dict, Any, Optional

# ---------- Couleurs ANSI ----------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

FG = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
}

def c(text: str, color: str = "white", bold: bool = False, dim: bool = False) -> str:
    s = ""
    if bold: s += BOLD
    if dim: s += DIM
    s += FG.get(color, FG["white"]) + str(text) + RESET
    return s

def hr(char="═", n=90, color="magenta") -> str:
    return c(char * n, color)

def banner(title: str, subtitle: str = ""):
    print(hr("═", 90, "magenta"))
    print(c(f"  {title}", "cyan", bold=True))
    if subtitle:
        print(c(f"  {subtitle}", "white", dim=True))
    print(hr("═", 90, "magenta"))

# ---------- Table ASCII ----------
def print_table(headers: List[str], rows: List[List[str]], title: Optional[str] = None):
    # convert all to str
    rows = [[str(x) for x in r] for r in rows]
    headers = [str(h) for h in headers]

    cols = len(headers)
    widths = [len(headers[i]) for i in range(cols)]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]))

    def line(left="╔", mid="╦", right="╗", fill="═"):
        return left + mid.join([fill * (w + 2) for w in widths]) + right

    def line_mid(left="╠", mid="╬", right="╣", fill="═"):
        return left + mid.join([fill * (w + 2) for w in widths]) + right

    def line_bottom(left="╚", mid="╩", right="╝", fill="═"):
        return left + mid.join([fill * (w + 2) for w in widths]) + right

    def fmt_row(r):
        cells = []
        for i in range(cols):
            cells.append(" " + r[i].ljust(widths[i]) + " ")
        return "║" + "║".join(cells) + "║"

    if title:
        print(c(f"\n{title}", "yellow", bold=True))

    print(c(line(), "magenta"))
    print(c(fmt_row(headers), "cyan", bold=True))
    print(c(line_mid(), "magenta"))
    for r in rows:
        print(fmt_row(r))
    print(c(line_bottom(), "magenta"))

# ---------- Affichage tick ----------
def print_tick(
    t: int,
    active_cpu: int,
    cap: int,
    pressure: float,
    worst: float,
    fog_n: int,
    cloud_n: int,
    fog_nodes: int = 0,
    cloud_nodes: int = 0
):
    bar_len = 26
    p = max(0.0, min(2.0, float(pressure)))  # bar 0..2
    filled = int((p / 2.0) * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    # couleurs selon charge
    if pressure < 0.6:
        p_col = "green"
    elif pressure < 1.0:
        p_col = "yellow"
    else:
        p_col = "red"

    print(
        f"{c(f'T{t:03d}', 'cyan', bold=True)} "
        f"poolCPU={c(active_cpu,'white',bold=True)}/{cap} "
        f"p={c(f'{pressure:.2f}', p_col, bold=True)} "
        f"{c(bar, p_col)} "
        f"worst={c(f'{worst:.2f}', 'magenta', bold=True)} "
        f"| placed tasks fog={c(fog_n,'green',bold=True)} cloud={c(cloud_n,'blue',bold=True)} "
        f"| nodes fog={c(fog_nodes,'green',bold=True)} cloud={c(cloud_nodes,'blue',bold=True)}"
    )


def print_mape_block(t: int, predictions: Dict[int, Dict[str, float]], plan: Dict[str, Any]):
    rows = []
    for h in sorted(predictions.keys()):
        p = predictions[h]["prediction"]
        u = predictions[h]["uncertainty"]
        rb = p + u
        fb = predictions[h].get("used_fallback", False)
        rows.append([f"H={h}", f"{p:.2f}", f"{u:.2f}", f"{rb:.2f}", "YES" if fb else "NO"])

    print_table(
        ["Horizon", "Pred", "Unc", "Robust", "Fallback"],
        rows,
        title=f"🧠 Cycle MAPE @ t={t}"
    )

    # plan résumé
    plan_rows = [
        ["robust_pred", f"{plan.get('robust_pred',0.0):.2f}"],
        ["pred_active_cpu", f"{plan.get('pred_active_cpu',0.0):.1f}"],
        ["ema_active_cpu", f"{plan.get('ema_active_cpu',0.0):.1f}"],
        ["scale_decision", str(plan.get("scale_decision","none"))],
        ["scale_reason", str(plan.get("scale_reason",""))],
        ["offload_ratio", f"{plan.get('offload_ratio',0.0):.2f}"],
        ["offload_reason", str(plan.get("offload_reason",""))],
    ]
    print_table(["Plan", "Value"], plan_rows, title="🧩 Plan (planner)")

# ---------- Stats finales + CSV ----------
def print_final_stats(metrics: List[Dict[str, Any]]):
    if not metrics:
        print(c("Aucune métrique générée.", "red", bold=True))
        return

    import numpy as np
    avg_pressure = float(np.mean([m["pressure"] for m in metrics]))
    max_pressure = float(np.max([m["pressure"] for m in metrics]))
    avg_offload = float(np.mean([m["offload_ratio"] for m in metrics]))
    
    fog_total = int(sum(m["tasks_placed_fog"] for m in metrics))
    cloud_total = int(sum(m["tasks_placed_cloud"] for m in metrics))
    lstm_fb = float(np.mean([m.get("lstm_fallback_used", 1) for m in metrics]))
    dqn_fb_tasks = int(sum(m.get("dqn_fallback_used_tasks", 0) for m in metrics))

    # ✅ Scale UP/DOWN totals (pris du dernier tick si dispo)
    last = metrics[-1] if metrics else {}
    scale_up = int(last.get("scale_up_total", 0))
    scale_down = int(last.get("scale_down_total", 0))
    total_energy_kj = float(last.get("energy_joules_cumul", 0.0)) / 1000.0

    rows = [
        ["Avg pool pressure", f"{avg_pressure:.2f}"],
        ["Max pool pressure", f"{max_pressure:.2f}"],
        ["Avg offload ratio", f"{avg_offload:.2%}"],
        ["Total tasks on Fog", str(fog_total)],
        ["Total tasks on Cloud", str(cloud_total)],
        ["LSTM fallback (ticks)", f"{lstm_fb:.1%}"],
        ["DQN fallback (tasks)", str(dqn_fb_tasks)],
        # ✅ Ajout demandé
        ["Total scale UP", str(scale_up)],
        ["Total scale DOWN", str(scale_down)],
        ["Total Energy (kJ)", f"{total_energy_kj:.2f}"] # ✅ Affichage Énergie
    ]
    print_table(["Metric", "Value"], rows, title="📊 STATISTIQUES FINALES")

def save_metrics_csv(metrics: List[Dict[str, Any]], output_path: str):
    if not metrics:
        return
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        w.writerows(metrics)
    print(c(f"\n💾 CSV sauvegardé: {output_path}", "green", bold=True))
# project/ui_utils.py  (patch MINIMAL: print_tick accepte fog_nodes/cloud_nodes)
# ⚠️ Garde le reste de ton fichier tel quel, tu ajoutes/modifies seulement print_tick.
