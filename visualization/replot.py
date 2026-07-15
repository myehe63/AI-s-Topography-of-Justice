"""
persona_simulation_results.json 에서 그래프만 다시 그리는 스크립트
Usage: python replot.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "/Users/parkjeongseo/Desktop/ideal/"

with open(SAVE_DIR + "persona_simulation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

groups = data["groups"]
personas = data["personas"]

n_clusters = data["n_clusters"]
colors = plt.cm.Set1(np.linspace(0, 0.8, n_clusters))

X = np.array([
    [
        p["scores"].get("state_authority", 50),
        p["scores"].get("corporate_ethics", 50),
        p["scores"].get("civilian_safety", 50),
    ]
    for p in personas
])
labels = np.array([p["cluster"] for p in personas])

# ── Figure 1: 3D scatter + ideal vectors ──────────────────────────────────────
fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121, projection='3d')
for k, (group, color) in enumerate(zip(groups, colors)):
    mask = labels == k
    ax1.scatter(
        X[mask, 0], X[mask, 1], X[mask, 2],
        c=[color], alpha=0.6, s=30,
        label=f"Cluster {k} (n={group['n_members']})"
    )
    w = group['w_g']
    ax1.scatter(*w, c=[color], s=150, marker='*', edgecolors='black', linewidth=1)

ax1.set_xlabel("State Authority")
ax1.set_ylabel("Corporate Ethics")
ax1.set_zlabel("Civilian Safety")
ax1.set_title("Persona Clusters in Moral Space")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(122, projection='3d')
scale = 100
for k, (group, color) in enumerate(zip(groups, colors)):
    ideal = np.array(group['ideal_unit_vector'])
    ax2.quiver(
        0, 0, 0,
        ideal[0]*scale, ideal[1]*scale, ideal[2]*scale,
        color=color, linewidth=3,
        arrow_length_ratio=0.15,
        label=f"Cluster {k} ideal"
    )

ax2.set_xlabel("State Authority")
ax2.set_ylabel("Corporate Ethics")
ax2.set_zlabel("Civilian Safety")
ax2.set_xlim(0, 100); ax2.set_ylim(0, 100); ax2.set_zlim(0, 100)
ax2.set_title("Ideal Vectors per Cluster")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(SAVE_DIR + "persona_clusters.png", dpi=150)
print("Saved: persona_clusters.png")
plt.show()

# ── Figure 2: 분산 비교 ───────────────────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(10, 5))
axes_names = ["state_authority", "corporate_ethics", "civilian_safety"]
x = np.arange(len(axes_names))
width = 0.8 / n_clusters

for k, (group, color) in enumerate(zip(groups, colors)):
    stds = [group['std_scores'][ax] for ax in axes_names]
    ax3.bar(x + k*width, stds, width, label=f"Cluster {k}", color=color, alpha=0.8)

ax3.set_xticks(x + width*(n_clusters-1)/2)
ax3.set_xticklabels(axes_names)
ax3.set_ylabel("Std Deviation")
ax3.set_title("Moral Uncertainty per Cluster (Σ^g diagonal)\nHigher = more internal disagreement")
ax3.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR + "cluster_variance.png", dpi=150)
print("Saved: cluster_variance.png")
plt.show()
