"""Stage 3b: 페르소나 생성 + GMM 클러스터링 → 군집별 ideal vector.

입력: config.json (case_context, axes, ideal_methods.clustering)
출력: outputs/03b_ideal_clustering.json

그래프(persona_clusters.png, cluster_variance.png)는 이 stage의 책임이 아니라
stage4b_visualize.py가 이 json을 읽어서 그린다 (계산/그리기 분리, 체크포인트 스킵 시에도
그래프를 다시 그릴 수 있도록).
"""

import asyncio
import warnings
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from pipeline.common import (
    call_llm,
    checkpoint_skip,
    dilemma_axes_description,
    load_json,
    output_path,
    parse_llm_json,
    save_json,
    score_json_fields,
)

warnings.filterwarnings("ignore")

STAGE_NAME = "stage3b"
OUTPUT_FILENAME = "03b_ideal_clustering.json"

PERSONA_MODEL = "gpt-4.1-mini"

PERSONA_GENERATION_PROMPT = """
Generate a detailed background profile for a fictional person who will respond to a moral dilemma.

Requirements:
- The person should have a SPECIFIC, CONCRETE life experience that would shape their moral views
- Do NOT define their political views or moral stance directly
- Include: age, nationality, occupation, and 2-3 key life experiences that would shape their worldview
- Make them feel like a real, complex human — not a stereotype
- Vary widely across: cultures, professions, life experiences, ages (18-80), regions of the world

Respond ONLY in this JSON format:
{
  "age": <integer>,
  "nationality": "<country>",
  "occupation": "<job title>",
  "background": "<2-3 sentences describing key life experiences that shape their worldview>"
}
"""


async def generate_persona(idx: int, countries: list[str]) -> dict:
    country = countries[idx % len(countries)]
    raw = await call_llm(
        provider="openai",
        model=PERSONA_MODEL,
        system="You generate diverse fictional persona backgrounds. Be creative and specific.",
        user=PERSONA_GENERATION_PROMPT + f"\n\nREQUIRED: This person MUST be from {country}. No exceptions.",
        temperature=1.1,
    )
    try:
        data = parse_llm_json(raw)
        data["id"] = idx
        return data
    except Exception:
        print(f"  [Persona {idx}] JSON parse failed, skipping")
        return {"id": idx, "error": True}


async def generate_personas_batch(n: int, countries: list[str], batch_size: int = 20) -> list[dict]:
    print(f"\n[Step 1] Generating {n} personas (batch_size={batch_size})...")
    personas = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = await asyncio.gather(*[generate_persona(i, countries) for i in range(start, end)])
        valid = [p for p in batch if not p.get("error")]
        personas.extend(valid)
        print(f"  Generated {len(personas)}/{n} personas...")
    return personas


async def score_dilemma(persona: dict, case_context: str, axes: list[dict]) -> Optional[dict]:
    persona_desc = (
        f"You are {persona['occupation']}, {persona['age']} years old, "
        f"from {persona['nationality']}. "
        f"Your background: {persona['background']}"
    )
    axes_description = dilemma_axes_description(axes)
    score_fields = score_json_fields(axes)

    system = (
        f"{persona_desc}\n\n"
        "You are responding to a moral dilemma based on your personal background and experiences. "
        "Your response should reflect your unique perspective — not a generic or neutral view. "
        "Be honest about how your life experiences would shape your moral judgments."
    )
    user = (
        f"Read this dilemma:\n{case_context}\n\n"
        f"Based on your background and experiences, rate the following:\n{axes_description}\n\n"
        "Respond ONLY in this JSON format:\n"
        "{\n"
        '  "reasoning": "<1-2 sentences explaining your perspective based on your background>",\n'
        '  "scores": {\n'
        f"{score_fields}\n"
        "  }\n"
        "}"
    )

    try:
        raw = await call_llm(
            provider="openai", model=PERSONA_MODEL, system=system, user=user, temperature=0.8
        )
        data = parse_llm_json(raw)
        return {
            "persona_id": persona["id"],
            "persona": persona,
            "reasoning": data.get("reasoning", ""),
            "scores": data.get("scores", {}),
        }
    except Exception as e:
        print(f"  [Persona {persona['id']}] Scoring failed: {e}")
        return None


async def score_all_personas(
    personas: list[dict], case_context: str, axes: list[dict], batch_size: int = 30
) -> list[dict]:
    print(f"\n[Step 2] Scoring {len(personas)} personas...")
    results = []
    for start in range(0, len(personas), batch_size):
        batch = personas[start : start + batch_size]
        scored = await asyncio.gather(*[score_dilemma(p, case_context, axes) for p in batch])
        valid = [r for r in scored if r is not None and r.get("scores")]
        results.extend(valid)
        print(f"  Scored {len(results)}/{len(personas)} personas...")
    return results


def find_clusters_gmm(scored: list[dict], axis_names: list[str], k_range=(2, 6)) -> tuple:
    print("\n[Step 3] GMM clustering...")
    X = np.array([[r["scores"].get(name, 50) for name in axis_names] for r in scored], dtype=float)

    bic_scores = []
    models = []
    k_values = range(k_range[0], k_range[1] + 1)

    for k in k_values:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        models.append(gmm)
        print(f"  k={k}: BIC={gmm.bic(X):.1f}")

    best_k_idx = int(np.argmin(bic_scores))
    best_k = list(k_values)[best_k_idx]
    best_gmm = models[best_k_idx]
    print(f"\n  Best k={best_k} (lowest BIC={bic_scores[best_k_idx]:.1f})")

    labels = best_gmm.predict(X)
    return X, labels, best_gmm, best_k


def compute_group_stats(
    X: np.ndarray, labels: np.ndarray, gmm, scored: list[dict], axis_names: list[str]
) -> list[dict]:
    print("\n[Step 4] Computing group norms and covariances...")
    groups = []
    n_clusters = gmm.n_components

    for k in range(n_clusters):
        mask = labels == k
        X_k = X[mask]
        members = [scored[i] for i in range(len(scored)) if labels[i] == k]

        w_g = gmm.means_[k]
        sigma_g = gmm.covariances_[k]

        norm = np.linalg.norm(w_g)
        ideal_unit = w_g / norm if norm > 1e-10 else w_g
        std_per_axis = np.sqrt(np.diag(sigma_g))

        dists = np.linalg.norm(X_k - w_g, axis=1)
        closest_idx = np.argsort(dists)[:3]
        representative_personas = [members[i]["persona"] for i in closest_idx]

        group_info = {
            "cluster_id": k,
            "n_members": int(np.sum(mask)),
            "w_g": w_g.tolist(),
            "ideal_unit_vector": ideal_unit.tolist(),
            "sigma_g_diag": std_per_axis.tolist(),
            "sigma_g_full": sigma_g.tolist(),
            "mean_scores": {axis_names[i]: float(w_g[i]) for i in range(len(axis_names))},
            "std_scores": {axis_names[i]: float(std_per_axis[i]) for i in range(len(axis_names))},
            "representative_personas": representative_personas,
        }
        groups.append(group_info)

        print(f"\n  Cluster {k} (n={group_info['n_members']}):")
        print(f"    w^g (center): {np.round(w_g, 1)}")
        print(f"    ideal unit:   {np.round(ideal_unit, 3)}")
        print(f"    std per axis: {np.round(std_per_axis, 1)}")
        print(
            f"    Sample persona: {representative_personas[0].get('occupation', '?')}, "
            f"{representative_personas[0].get('nationality', '?')}"
        )

    return groups


async def _run_async(config: dict) -> dict:
    case_context = config["case_context"]
    axes = config["axes"]
    axis_names = [ax["name"] for ax in axes]
    clustering_cfg = config["ideal_methods"]["clustering"]

    n_personas = clustering_cfg.get("n_personas", 200)
    k_range = clustering_cfg.get("k_range", [2, 6])
    countries = clustering_cfg["countries"]

    print("=" * 60)
    print("PERSONA SIMULATION — GMM Clustering")
    print("=" * 60)

    personas = await generate_personas_batch(n_personas, countries)
    scored = await score_all_personas(personas, case_context, axes)
    print(f"\n  Valid scored personas: {len(scored)}")

    X, labels, gmm, best_k = find_clusters_gmm(scored, axis_names, k_range=tuple(k_range))
    groups = compute_group_stats(X, labels, gmm, scored, axis_names)

    result = {
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
        ],
    }

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  {len(scored)} personas → {best_k} clusters")
    return result


def run(config: dict, force: bool = False) -> dict:
    out_path = output_path(config, OUTPUT_FILENAME)
    if not force and checkpoint_skip(out_path, STAGE_NAME):
        return load_json(out_path)

    result = asyncio.run(_run_async(config))
    save_json(out_path, result)
    print(f"\n[{STAGE_NAME}] 결과 저장 → {out_path}")
    return result
