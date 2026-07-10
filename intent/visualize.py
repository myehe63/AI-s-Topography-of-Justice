import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))

AXES = ["state_authority", "corporate_ethics", "civilian_safety"]
AXES_SHORT = ["State\nAuthority", "Corporate\nEthics", "Civilian\nSafety"]

DOD_COLOR = "#E74C3C"
ANT_COLOR = "#3498DB"

TP_LABELS = {
    "T1": "T1\nDoD Strategy",
    "T2": "T2\nNegotiation\nBreakdown",
    "T3": "T3\nSupply Chain\nDesignation",
    "T4": "T4\nCourt Battle",
    "T5": "T5\nInjunction"
}

def load_data():
    with open(os.path.join(BASE, "data", "extracted.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def get_timepoint_avg(vectors, actor, tp_id):
    vecs = [v["vector"] for v in vectors if v["actor"] == actor and v["timepoint"] == tp_id]
    if not vecs:
        return None
    return {ax: sum(v[ax] for v in vecs) / len(vecs) for ax in AXES}


def plot_drift_lines(data):
    """Graph 1: Score change per axis over time"""
    vectors = data["vectors"]
    timepoints = data["timepoints"]
    tp_ids = [tp["id"] for tp in timepoints]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Intent Vector Drift over Time  (DoD vs Anthropic)", fontsize=14, fontweight="bold")

    for i, ax_name in enumerate(AXES):
        ax = axes[i]

        dod_scores, dod_tps = [], []
        ant_scores, ant_tps = [], []

        for j, tp_id in enumerate(tp_ids):
            dod_vec = get_timepoint_avg(vectors, "DoD", tp_id)
            ant_vec = get_timepoint_avg(vectors, "Anthropic", tp_id)
            if dod_vec:
                dod_scores.append(dod_vec[ax_name])
                dod_tps.append(j)
            if ant_vec:
                ant_scores.append(ant_vec[ax_name])
                ant_tps.append(j)

        ax.plot(dod_tps, dod_scores, "o-", color=DOD_COLOR, linewidth=2.5, markersize=8, label="DoD", zorder=3)
        ax.plot(ant_tps, ant_scores, "s-", color=ANT_COLOR, linewidth=2.5, markersize=8, label="Anthropic", zorder=3)

        for x, y in zip(dod_tps, dod_scores):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=DOD_COLOR, fontweight="bold")
        for x, y in zip(ant_tps, ant_scores):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=9, color=ANT_COLOR, fontweight="bold")

        ax.set_ylim(-5, 110)
        ax.set_xticks(range(len(tp_ids)))
        ax.set_xticklabels([TP_LABELS.get(tp, tp) for tp in tp_ids], fontsize=7.5)
        ax.set_ylabel("Score (0-100)", fontsize=9)
        ax.set_title(AXES_SHORT[i], fontsize=11, fontweight="bold")
        ax.axhline(50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        ax.axhspan(50, 110, alpha=0.04, color=DOD_COLOR)
        ax.axhspan(-5, 50, alpha=0.04, color=ANT_COLOR)

    plt.tight_layout()
    out = os.path.join(BASE, "data", "viz_drift_lines.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


def plot_3d_vectors(data):
    """Graph 2: Vector directions in 3D moral space"""
    vectors = data["vectors"]
    timepoints = data["timepoints"]
    tp_ids = [tp["id"] for tp in timepoints]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    scale = 100

    dod_tps_found, ant_tps_found = [], []
    for j, tp_id in enumerate(tp_ids):
        dod_vec = get_timepoint_avg(vectors, "DoD", tp_id)
        ant_vec = get_timepoint_avg(vectors, "Anthropic", tp_id)
        if dod_vec:
            dod_tps_found.append((j, tp_id, dod_vec))
        if ant_vec:
            ant_tps_found.append((j, tp_id, ant_vec))

    for idx, (j, tp_id, vec) in enumerate(dod_tps_found):
        alpha = 0.4 + 0.6 * (idx / max(len(dod_tps_found) - 1, 1))
        norm = np.linalg.norm(list(vec.values()))
        unit = {k: v / norm for k, v in vec.items()}
        ax.quiver(0, 0, 0,
                  unit[AXES[0]] * scale, unit[AXES[1]] * scale, unit[AXES[2]] * scale,
                  color=DOD_COLOR, alpha=alpha, linewidth=2, arrow_length_ratio=0.12)
        ax.text(unit[AXES[0]] * scale * 1.08, unit[AXES[1]] * scale * 1.08, unit[AXES[2]] * scale * 1.08,
                f"DoD {tp_id}", fontsize=7, color=DOD_COLOR)

    for idx, (j, tp_id, vec) in enumerate(ant_tps_found):
        alpha = 0.4 + 0.6 * (idx / max(len(ant_tps_found) - 1, 1))
        norm = np.linalg.norm(list(vec.values()))
        unit = {k: v / norm for k, v in vec.items()}
        ax.quiver(0, 0, 0,
                  unit[AXES[0]] * scale, unit[AXES[1]] * scale, unit[AXES[2]] * scale,
                  color=ANT_COLOR, alpha=alpha, linewidth=2, arrow_length_ratio=0.12)
        ax.text(unit[AXES[0]] * scale * 1.08, unit[AXES[1]] * scale * 1.08, unit[AXES[2]] * scale * 1.08,
                f"Ant {tp_id}", fontsize=7, color=ANT_COLOR)

    ax.set_xlabel("State Authority", fontsize=9, labelpad=8)
    ax.set_ylabel("Corporate Ethics", fontsize=9, labelpad=8)
    ax.set_zlabel("Civilian Safety", fontsize=9, labelpad=8)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 100)

    dod_patch = mpatches.Patch(color=DOD_COLOR, label="DoD (darker = later)")
    ant_patch = mpatches.Patch(color=ANT_COLOR, label="Anthropic (darker = later)")
    ax.legend(handles=[dod_patch, ant_patch], fontsize=9, loc="upper left")
    ax.set_title("Intent Vectors in 3D Moral Space\n(normalized unit vectors)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(BASE, "data", "viz_3d_vectors.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


def plot_angle_over_time(data):
    """Graph 3: Angular divergence between DoD and Anthropic over time"""
    vectors = data["vectors"]
    timepoints = data["timepoints"]
    tp_ids = [tp["id"] for tp in timepoints]

    angles, valid_tps = [], []
    for tp_id in tp_ids:
        dod_vec = get_timepoint_avg(vectors, "DoD", tp_id)
        ant_vec = get_timepoint_avg(vectors, "Anthropic", tp_id)
        if dod_vec and ant_vec:
            v1 = np.array([dod_vec[ax] for ax in AXES])
            v2 = np.array([ant_vec[ax] for ax in AXES])
            cos = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
            angles.append(np.degrees(np.arccos(cos)))
            valid_tps.append(tp_id)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(valid_tps)), angles, "o-", color="#8E44AD", linewidth=2.5, markersize=10)

    for i, (tp, angle) in enumerate(zip(valid_tps, angles)):
        ax.annotate(f"{angle:.1f}°", (i, angle),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=11, fontweight="bold", color="#8E44AD")

    ax.set_xticks(range(len(valid_tps)))
    ax.set_xticklabels([TP_LABELS.get(tp, tp) for tp in valid_tps], fontsize=9)
    ax.set_ylabel("Angular Distance (degrees)", fontsize=10)
    ax.set_title("DoD vs Anthropic — Intent Divergence over Time\n(larger angle = stronger opposition)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhspan(60, 100, alpha=0.07, color="red", label="Extreme conflict")
    ax.axhspan(30, 60, alpha=0.07, color="orange", label="Clear conflict")
    ax.axhspan(0, 30, alpha=0.07, color="green", label="Mild difference")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out = os.path.join(BASE, "data", "viz_angle_over_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


if __name__ == "__main__":
    data = load_data()
    print("=== Graph 1: Drift Line Chart ===")
    plot_drift_lines(data)
    print("\n=== Graph 2: 3D Vectors ===")
    plot_3d_vectors(data)
    print("\n=== Graph 3: Divergence Angle ===")
    plot_angle_over_time(data)
    print("\nDone!")
