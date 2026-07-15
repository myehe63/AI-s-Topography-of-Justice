"""
Persona Simulation Pipeline for Justice Vector Simulator
=========================================================
1. LLM으로 경험 기반 랜덤 페르소나 N개 생성
2. 각 페르소나가 딜레마에 대해 세 축 점수 매김
3. GMM으로 사후 군집 발견
4. 각 군집의 group norm (w^g) + 분산 (Σ^g) 계산
5. 군집별 ideal vector 추출 + 시각화

Usage:
  export OPENAI_API_KEY="..."
  python persona_simulation.py
"""

import asyncio
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from dataclasses import dataclass, field
from typing import Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Dilemma & Axes (기존 코드에서 재활용) ─────────────────────────────────────
DILEMMA = """
In 2026, an AI company signed a $200M contract with a national military.
Both parties agreed to two restrictions: the AI could not be used for fully
autonomous weapons systems, and could not be used for mass domestic surveillance.

Later, the military demanded the right to use the AI for "all lawful purposes"
without exception, arguing these restrictions blocked legitimate defense operations.
The company refused, citing the original agreed terms. The military responded by
terminating the contract and designating the company a national security risk —
a label previously reserved for foreign adversaries.

The military's position: national defense requires unrestricted access to the best
available tools. Allowing a private vendor to unilaterally limit lawful military
operations sets a dangerous precedent where corporations override elected governments.

The company's position: unrestricted use would enable autonomous killing machines
and mass surveillance of citizens — uses the company believes cross an ethical line
regardless of legality.
"""

AXES_DESCRIPTION = """
Rate the following on 0-100 based on the dilemma above:
- state_authority: Should the state/military have the right to override a private
  company's ethical restrictions on contracted technology? (100 = state authority supreme)
- corporate_ethics: Should private companies have the right to refuse uses of their
  technology that violate their ethical principles? (100 = companies have this right)
- civilian_safety: Should protecting civilians from autonomous weapons and mass
  surveillance be treated as a non-negotiable priority above both national security
  arguments and corporate interests? (100 = civilian protection paramount)
"""

COUNTRIES = [
    "South Korea", "Nigeria", "Germany", "Brazil", "India",
    "Japan", "Egypt", "Mexico", "Russia", "Australia",
    "Iran", "USA", "China", "Colombia", "Kenya",
    "Turkey", "Indonesia", "France", "Pakistan", "Argentina",
    "Ethiopia", "Vietnam", "Poland", "Saudi Arabia", "Ukraine",
    "Canada", "Morocco", "Thailand", "Israel", "South Africa",
    "Bangladesh", "Romania", "Chile", "Kazakhstan", "Ghana",
    "Peru", "Netherlands", "Syria", "Singapore", "New Zealand",
    "Afghanistan", "Sweden", "Philippines", "Cuba", "Iraq",
    "Finland", "Myanmar", "Venezuela", "Zimbabwe", "Haiti"
 ] 

# ── 페르소나 생성 ──────────────────────────────────────────────────────────────

PERSONA_GENERATION_PROMPT = """
Generate a detailed background profile for a fictional person who will respond to a moral dilemma.

Requirements:
- The person should have a SPECIFIC, CONCRETE life experience that would shape their moral views
- Do NOT define their political views or moral stance directly
- Include: age, nationality, occupation, and 2-3 key life experiences that would shape their worldview
- Make them feel like a real, complex human — not a stereotype
- Vary widely across: cultures, professions, life experiences, ages (18-80), regions of the world
- Some examples of interesting profiles (do NOT copy these, make your own):
  * A former military drone operator who later became a civil liberties lawyer
  * A Uyghur survivor of mass surveillance who now works in AI ethics
  * A small-town sheriff in rural America who relies on federal tech contracts
  * A teenager in South Korea who grew up with pervasive government monitoring

Respond ONLY in this JSON format:
{
  "age": <integer>,
  "nationality": "<country>",
  "occupation": "<job title>",
  "background": "<2-3 sentences describing key life experiences that shape their worldview>"
}
"""

async def generate_persona(idx: int) -> dict:
    """LLM으로 페르소나 한 명 생성"""
    country = COUNTRIES[idx % len(COUNTRIES)]
    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You generate diverse fictional persona backgrounds. Be creative and specific."},
            {"role": "user", "content": PERSONA_GENERATION_PROMPT + f"\n\nREQUIRED: This person MUST be from {country}. No exceptions."}
        ],
        temperature=1.1,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        data["id"] = idx
        return data
    except json.JSONDecodeError:
        print(f"  [Persona {idx}] JSON parse failed, skipping")
        return {"id": idx, "error": True}


async def generate_personas_batch(n: int, batch_size: int = 20) -> list[dict]:
    """페르소나 N개를 batch로 생성"""
    print(f"\n[Step 1] Generating {n} personas (batch_size={batch_size})...")
    personas = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = await asyncio.gather(*[generate_persona(i) for i in range(start, end)])
        valid = [p for p in batch if not p.get("error")]
        personas.extend(valid)
        print(f"  Generated {len(personas)}/{n} personas...")
    return personas


# ── 딜레마 점수 매기기 ──────────────────────────────────────────────────────────

async def score_dilemma(persona: dict) -> Optional[dict]:
    """페르소나가 딜레마에 점수 매김"""
    persona_desc = (
        f"You are {persona['occupation']}, {persona['age']} years old, "
        f"from {persona['nationality']}. "
        f"Your background: {persona['background']}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                f"{persona_desc}\n\n"
                "You are responding to a moral dilemma based on your personal background and experiences. "
                "Your response should reflect your unique perspective — not a generic or neutral view. "
                "Be honest about how your life experiences would shape your moral judgments."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Read this dilemma:\n{DILEMMA}\n\n"
                f"Based on your background and experiences, rate the following:\n{AXES_DESCRIPTION}\n\n"
                "Respond ONLY in this JSON format:\n"
                "{\n"
                '  "reasoning": "<1-2 sentences explaining your perspective based on your background>",\n'
                '  "scores": {\n'
                '    "state_authority": <integer 0-100>,\n'
                '    "corporate_ethics": <integer 0-100>,\n'
                '    "civilian_safety": <integer 0-100>\n'
                "  }\n"
                "}"
            ),
        },
    ]

    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.8,
        )
        raw = resp.choices[0].message.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return {
            "persona_id": persona["id"],
            "persona": persona,
            "reasoning": data.get("reasoning", ""),
            "scores": data.get("scores", {}),
        }
    except Exception as e:
        print(f"  [Persona {persona['id']}] Scoring failed: {e}")
        return None


async def score_all_personas(personas: list[dict], batch_size: int = 30) -> list[dict]:
    """모든 페르소나 점수 매기기"""
    print(f"\n[Step 2] Scoring {len(personas)} personas...")
    results = []
    for start in range(0, len(personas), batch_size):
        batch = personas[start:start + batch_size]
        scored = await asyncio.gather(*[score_dilemma(p) for p in batch])
        valid = [r for r in scored if r is not None and r.get("scores")]
        results.extend(valid)
        print(f"  Scored {len(results)}/{len(personas)} personas...")
    return results


# ── GMM 군집화 ────────────────────────────────────────────────────────────────

def find_clusters_gmm(scored: list[dict], n_components_range=(2, 8)) -> tuple:
    """
    GMM으로 최적 군집 수 찾고 군집화
    BIC score로 최적 k 선택
    """
    print(f"\n[Step 3] GMM clustering...")

    # 점수 행렬 만들기
    X = np.array([
        [
            r["scores"].get("state_authority", 50),
            r["scores"].get("corporate_ethics", 50),
            r["scores"].get("civilian_safety", 50),
        ]
        for r in scored
    ], dtype=float)

    # BIC로 최적 k 선택
    bic_scores = []
    models = []
    k_range = range(n_components_range[0], n_components_range[1] + 1)

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=5)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        models.append(gmm)
        print(f"  k={k}: BIC={gmm.bic(X):.1f}")

    best_k_idx = np.argmin(bic_scores)
    best_k = list(k_range)[best_k_idx]
    best_gmm = models[best_k_idx]

    print(f"\n  Best k={best_k} (lowest BIC={bic_scores[best_k_idx]:.1f})")

    labels = best_gmm.predict(X)
    return X, labels, best_gmm, best_k


# ── Group norm + 분산 계산 ────────────────────────────────────────────────────

def compute_group_stats(X: np.ndarray, labels: np.ndarray, gmm, scored: list[dict]) -> list[dict]:
    """
    각 군집의 w^g (중심) + Σ^g (공분산) 계산
    논문의 hierarchical moral principles 모델 기반
    """
    print(f"\n[Step 4] Computing group norms and covariances...")

    groups = []
    n_clusters = gmm.n_components
    axes = ["state_authority", "corporate_ethics", "civilian_safety"]

    for k in range(n_clusters):
        mask = labels == k
        X_k = X[mask]
        members = [scored[i] for i in range(len(scored)) if labels[i] == k]

        w_g = gmm.means_[k]          # group norm (중심)
        sigma_g = gmm.covariances_[k]  # 공분산 행렬

        # 정규화해서 unit vector로
        norm = np.linalg.norm(w_g)
        ideal_unit = w_g / norm if norm > 1e-10 else w_g

        # 각 축의 표준편차 (분산의 제곱근)
        std_per_axis = np.sqrt(np.diag(sigma_g))

        # 대표 페르소나 샘플 (중심에서 가까운 3명)
        dists = np.linalg.norm(X_k - w_g, axis=1)
        closest_idx = np.argsort(dists)[:3]
        representative_personas = [members[i]["persona"] for i in closest_idx]

        group_info = {
            "cluster_id": k,
            "n_members": int(np.sum(mask)),
            "w_g": w_g.tolist(),           # raw center
            "ideal_unit_vector": ideal_unit.tolist(),  # normalized
            "sigma_g_diag": std_per_axis.tolist(),     # std per axis
            "sigma_g_full": sigma_g.tolist(),          # full covariance
            "mean_scores": {axes[i]: float(w_g[i]) for i in range(3)},
            "std_scores": {axes[i]: float(std_per_axis[i]) for i in range(3)},
            "representative_personas": representative_personas,
        }

        groups.append(group_info)

        print(f"\n  Cluster {k} (n={group_info['n_members']}):")
        print(f"    w^g (center): {np.round(w_g, 1)}")
        print(f"    ideal unit:   {np.round(ideal_unit, 3)}")
        print(f"    std per axis: {np.round(std_per_axis, 1)}")
        print(f"    Sample persona: {representative_personas[0].get('occupation', '?')}, "
              f"{representative_personas[0].get('nationality', '?')}")

    return groups


# ── 시각화 ────────────────────────────────────────────────────────────────────

def visualize_clusters(X: np.ndarray, labels: np.ndarray, groups: list[dict]):
    """군집 3D scatter + ideal vectors 시각화"""
    print(f"\n[Step 5] Visualizing...")

    fig = plt.figure(figsize=(16, 7))

    # ── Plot 1: 3D scatter ──
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(groups)))

    for k, (group, color) in enumerate(zip(groups, colors)):
        mask = labels == k
        ax1.scatter(
            X[mask, 0], X[mask, 1], X[mask, 2],
            c=[color], alpha=0.6, s=30,
            label=f"Cluster {k} (n={group['n_members']})"
        )
        # 중심점
        w = group['w_g']
        ax1.scatter(*w, c=[color], s=150, marker='*', edgecolors='black', linewidth=1)

    ax1.set_xlabel("State Authority")
    ax1.set_ylabel("Corporate Ethics")
    ax1.set_zlabel("Civilian Safety")
    ax1.set_title("Persona Clusters in Moral Space")
    ax1.legend(fontsize=8)

    # ── Plot 2: ideal vectors per cluster ──
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
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_zlim(0, 100)
    ax2.set_title("Ideal Vectors per Cluster")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("persona_clusters.png", dpi=150)
    print("  Saved: persona_clusters.png")

    # ── Plot 3: 분산 비교 (축별 std) ──
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    axes_names = ["state_authority", "corporate_ethics", "civilian_safety"]
    x = np.arange(len(axes_names))
    width = 0.8 / len(groups)

    for k, (group, color) in enumerate(zip(groups, colors)):
        stds = [group['std_scores'][ax] for ax in axes_names]
        ax3.bar(x + k*width, stds, width, label=f"Cluster {k}", color=color, alpha=0.8)

    ax3.set_xticks(x + width*(len(groups)-1)/2)
    ax3.set_xticklabels(axes_names)
    ax3.set_ylabel("Std Deviation")
    ax3.set_title("Moral Uncertainty per Cluster (Σ^g diagonal)\nHigher = more internal disagreement")
    ax3.legend()
    plt.tight_layout()
    plt.savefig("cluster_variance.png", dpi=150)
    print("  Saved: cluster_variance.png")

    plt.show()


# ── 결과 저장 ────────────────────────────────────────────────────────────────

def save_results(scored: list[dict], groups: list[dict], labels: np.ndarray):
    results = {
        "n_personas": len(scored),
        "n_clusters": len(groups),
        "groups": groups,
        "personas": [
            {
                "persona_id": r["persona_id"],
                "persona": r["persona"],
                "reasoning": r["reasoning"],
                "scores": r["scores"],
                "cluster": int(labels[i]),
            }
            for i, r in enumerate(scored)
        ]
    }
    with open("/Users/parkjeongseo/Desktop/ideal/persona_simulation_results.json", "w", encoding="utf-8") as f:
     json.dump(results, f, ensure_ascii=False, indent=2)
    print("  Saved: persona_simulation_results.json")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(n_personas: int = 200):
    print("=" * 60)
    print("PERSONA SIMULATION — Justice Vector Simulator")
    print("=" * 60)

    # 1. 페르소나 생성
    personas = await generate_personas_batch(n_personas)

    # 2. 딜레마 점수 매기기
    scored = await score_all_personas(personas)
    print(f"\n  Valid scored personas: {len(scored)}")

    # 3. GMM 군집화
    X, labels, gmm, best_k = find_clusters_gmm(scored, n_components_range=(2, 6))  # 상한 6으로

    # 4. Group norm + 분산
    groups = compute_group_stats(X, labels, gmm, scored)

    # 5. 시각화
    visualize_clusters(X, labels, groups)
    plt.savefig("/Users/parkjeongseo/Desktop/ideal/persona_clusters.png", dpi=150)
    plt.savefig("/Users/parkjeongseo/Desktop/ideal/cluster_variance.png", dpi=150)
    # 6. 저장
    save_results(scored, groups, labels)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  {len(scored)} personas → {best_k} clusters")
    print(f"  Results: persona_simulation_results.json")
    print(f"  Plots: persona_clusters.png, cluster_variance.png")

    return groups, scored, labels


if __name__ == "__main__":
    asyncio.run(main(n_personas=200))
