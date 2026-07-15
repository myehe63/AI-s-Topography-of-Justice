"""
Ideal + Intent Vector 통합 비교 그래프
- Ideal: ITIF 서베이, 멀티에이전트 디베이트, Persona Cluster 1, Cluster 3
- Intent: DoD T4, Anthropic T3

Usage: python plot_ideal_intent.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

SAVE_DIR = "/Users/parkjeongseo/Desktop/ideal/"

# ── 데이터 ────────────────────────────────────────────────────────────────────

# Ideal vectors
survey  = np.array([0.321, 0.587, 0.743])
debate  = np.array([0.188, 0.676, 0.713])

with open(SAVE_DIR + "persona_simulation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
groups = {g["cluster_id"]: g for g in data["groups"]}
cluster1 = np.array(groups[1]["ideal_unit_vector"])
cluster3 = np.array(groups[3]["ideal_unit_vector"])

# Intent vectors (정규화)
def normalize(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)

dod_intent       = normalize([86.5, 16.0, 17.5])   # DoD T4
anthropic_intent = normalize([35.0, 60.0, 55.0])   # Anthropic T3

# ── 각도 계산 ─────────────────────────────────────────────────────────────────

def angle(v1, v2):
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

ideals = {
    "ITIF Survey":   survey,
    "Debate":        debate,
    "Persona C1":    cluster1,
    "Persona C3":    cluster3,
}

print("=== DoD Intent vs Ideals ===")
for name, ideal in ideals.items():
    print(f"  DoD ↔ {name}: {angle(dod_intent, ideal):.1f}°")

print("\n=== Anthropic Intent vs Ideals ===")
for name, ideal in ideals.items():
    print(f"  Anthropic ↔ {name}: {angle(anthropic_intent, ideal):.1f}°")

print(f"\n=== DoD ↔ Anthropic Intent: {angle(dod_intent, anthropic_intent):.1f}° ===")

# ── 3D 그래프 ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
scale = 100

# Ideal — 실선
ideal_vectors = [
    (survey,   "#E74C3C", "ITIF Survey"),
    (debate,   "#2ECC71", "Multi-Agent Debate"),
    (cluster1, "#3498DB", "Persona C1 (high corp/safety)"),
    (cluster3, "#F39C12", "Persona C3 (high state auth)"),
]

for vec, color, label in ideal_vectors:
    ax.quiver(0, 0, 0,
              vec[0]*scale, vec[1]*scale, vec[2]*scale,
              color=color, linewidth=2.5,
              arrow_length_ratio=0.12,
              label=label)

# Intent — 점선 (linestyle 적용위해 plot으로)
intent_vectors = [
    (dod_intent,       "#8B0000", "DoD Intent (T4)"),
    (anthropic_intent, "#00008B", "Anthropic Intent (T3)"),
]

for vec, color, label in intent_vectors:
    ax.quiver(0, 0, 0,
              vec[0]*scale, vec[1]*scale, vec[2]*scale,
              color=color, linewidth=2.5,
              arrow_length_ratio=0.12,
              linestyle='dashed',
              label=label)
    # 점선 효과를 위한 보조선
    ax.plot([0, vec[0]*scale], [0, vec[1]*scale], [0, vec[2]*scale],
            color=color, linewidth=1.5, linestyle='--', alpha=0.6)

ax.set_xlabel("State Authority", labelpad=10)
ax.set_ylabel("Corporate Ethics", labelpad=10)
ax.set_zlabel("Civilian Safety", labelpad=10)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
ax.legend(fontsize=9, loc='upper left')
ax.set_title("Ideal vs Intent Vectors\n(solid = ideal, dashed = intent)", fontsize=11)

plt.tight_layout()
plt.savefig(SAVE_DIR + "ideal_intent.png", dpi=150)
print(f"\nSaved: {SAVE_DIR}ideal_intent.png")
plt.show()
