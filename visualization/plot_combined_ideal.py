"""
통합 Ideal Vector 비교 그래프
- ITIF 서베이
- 멀티에이전트 디베이트
- Cluster 1 (페르소나 극단 — corporate_ethics/civilian_safety 높은 쪽)
- Cluster 3 (페르소나 극단 — state_authority 높은 쪽)

Usage: python plot_combined_ideal.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

SAVE_DIR = "/Users/parkjeongseo/Desktop/ideal/"

# ── 데이터 로드 ────────────────────────────────────────────────────────────────

# ITIF 서베이 + 디베이트
survey  = np.array([0.321, 0.587, 0.743])
debate  = np.array([0.188, 0.676, 0.713])

# 페르소나 군집에서 극단치 2개
with open(SAVE_DIR + "persona_simulation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

groups = {g["cluster_id"]: g for g in data["groups"]}
cluster1 = np.array(groups[1]["ideal_unit_vector"])  # corporate_ethics/civilian_safety 극단
cluster3 = np.array(groups[3]["ideal_unit_vector"])  # state_authority 극단

# ── 각도 계산 ─────────────────────────────────────────────────────────────────

def angle(v1, v2):
    cos = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

vectors = {
    "ITIF Survey":       survey,
    "Multi-Agent Debate": debate,
    "Persona Cluster 1\n(high corp/safety)": cluster1,
    "Persona Cluster 3\n(high state auth)":  cluster3,
}

print("=== Pairwise Angles ===")
names = list(vectors.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a = angle(vectors[names[i]], vectors[names[j]])
        print(f"  {names[i].replace(chr(10),' ')} ↔ {names[j].replace(chr(10),' ')}: {a:.1f}°")

# ── 3D 그래프 ─────────────────────────────────────────────────────────────────

colors = {
    "ITIF Survey":       "#E74C3C",
    "Multi-Agent Debate": "#2ECC71",
    "Persona Cluster 1\n(high corp/safety)": "#3498DB",
    "Persona Cluster 3\n(high state auth)":  "#F39C12",
}

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scale = 100

for label, vec in vectors.items():
    color = colors[label]
    ax.quiver(
        0, 0, 0,
        vec[0]*scale, vec[1]*scale, vec[2]*scale,
        color=color, linewidth=3,
        arrow_length_ratio=0.12,
        label=label.replace("\n", " "),
    )
    # 벡터 끝에 레이블
    ax.text(
        vec[0]*scale*1.05,
        vec[1]*scale*1.05,
        vec[2]*scale*1.05,
        label.replace("\n", "\n"),
        fontsize=8, color=color
    )

ax.set_xlabel("State Authority", labelpad=10)
ax.set_ylabel("Corporate Ethics", labelpad=10)
ax.set_zlabel("Civilian Safety", labelpad=10)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
ax.legend(fontsize=9, loc='upper left')
ax.set_title("Ideal Vectors — Combined Comparison\n(Survey / Debate / Persona extremes)", fontsize=11)

plt.tight_layout()
plt.savefig(SAVE_DIR + "combined_ideal.png", dpi=150)
print(f"\nSaved: {SAVE_DIR}combined_ideal.png")
plt.show()
