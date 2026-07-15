"""Stage 4a: intent vs 각 ideal 소스 간 각도 계산.

입력: outputs/01_intent_extracted.json, outputs/03a_ideal_debate.json(있으면),
      outputs/03b_ideal_clustering.json(있으면), config.survey_ideal_vector(있으면)
출력: outputs/04_comparison.json (pairwise 각도)
"""

import os
from itertools import combinations

from pipeline.common import (
    checkpoint_skip,
    load_json,
    normalize_vector,
    output_path,
    save_json,
    vector_angle,
)

STAGE_NAME = "stage4a"
OUTPUT_FILENAME = "04_comparison.json"
INTENT_FILENAME = "01_intent_extracted.json"
DEBATE_FILENAME = "03a_ideal_debate.json"
CLUSTERING_FILENAME = "03b_ideal_clustering.json"


def latest_actor_vectors(intent_data: dict) -> dict:
    """각 actor의 마지막 timepoint 평균 벡터(정규화 포함)를 반환."""
    axis_names = [ax["name"] for ax in intent_data["axes"]]
    tp_order = [tp["id"] for tp in intent_data["timepoints"]]
    vectors = intent_data["vectors"]

    actors = sorted(set(v["actor"] for v in vectors))
    result = {}
    for actor in actors:
        actor_tps = [
            tp for tp in tp_order
            if any(v["actor"] == actor and v["timepoint"] == tp for v in vectors)
        ]
        if not actor_tps:
            continue
        last_tp = actor_tps[-1]
        tp_vectors = [v["vector"] for v in vectors if v["actor"] == actor and v["timepoint"] == last_tp]
        avg = {ax: sum(v[ax] for v in tp_vectors) / len(tp_vectors) for ax in axis_names}
        result[actor] = {
            "timepoint": last_tp,
            "raw_vector": avg,
            "unit_vector": normalize_vector([avg[ax] for ax in axis_names]),
        }
    return result


def collect_ideal_sources(config: dict) -> dict:
    """survey / debate / persona cluster별 ideal unit vector 모음. 없는 소스는 건너뜀."""
    sources = {}

    survey = config.get("survey_ideal_vector")
    if survey:
        sources["survey"] = normalize_vector(survey)

    debate_path = output_path(config, DEBATE_FILENAME)
    if os.path.exists(debate_path):
        sources["debate"] = load_json(debate_path)["ideal_unit_vector"]

    clustering_path = output_path(config, CLUSTERING_FILENAME)
    if os.path.exists(clustering_path):
        for group in load_json(clustering_path)["groups"]:
            sources[f"persona_cluster_{group['cluster_id']}"] = group["ideal_unit_vector"]

    return sources


def _compare(config: dict) -> dict:
    intent_data = load_json(output_path(config, INTENT_FILENAME))
    actor_vectors = latest_actor_vectors(intent_data)
    ideal_sources = collect_ideal_sources(config)

    print("=== Intent vs Ideal 각도 ===")
    intent_vs_ideal = []
    for actor, info in actor_vectors.items():
        for source_name, ideal_vec in ideal_sources.items():
            angle = vector_angle(info["unit_vector"], ideal_vec)
            intent_vs_ideal.append({
                "actor": actor,
                "actor_timepoint": info["timepoint"],
                "ideal_source": source_name,
                "angle": round(angle, 2),
            })
            print(f"  {actor} ({info['timepoint']}) ↔ {source_name}: {angle:.1f}°")

    print("\n=== Intent 간 각도 (actor pairwise) ===")
    intent_vs_intent = []
    for a1, a2 in combinations(actor_vectors.keys(), 2):
        angle = vector_angle(actor_vectors[a1]["unit_vector"], actor_vectors[a2]["unit_vector"])
        intent_vs_intent.append({
            "actor_1": a1,
            "actor_1_timepoint": actor_vectors[a1]["timepoint"],
            "actor_2": a2,
            "actor_2_timepoint": actor_vectors[a2]["timepoint"],
            "angle": round(angle, 2),
        })
        print(f"  {a1} ↔ {a2}: {angle:.1f}°")

    print("\n=== Ideal 소스간 각도 ===")
    ideal_vs_ideal = []
    for s1, s2 in combinations(ideal_sources.keys(), 2):
        angle = vector_angle(ideal_sources[s1], ideal_sources[s2])
        ideal_vs_ideal.append({"source_1": s1, "source_2": s2, "angle": round(angle, 2)})
        print(f"  {s1} ↔ {s2}: {angle:.1f}°")

    return {
        "actor_intent_vectors": {
            actor: {"timepoint": info["timepoint"], "unit_vector": info["unit_vector"]}
            for actor, info in actor_vectors.items()
        },
        "ideal_sources": ideal_sources,
        "intent_vs_ideal": intent_vs_ideal,
        "intent_vs_intent": intent_vs_intent,
        "ideal_vs_ideal": ideal_vs_ideal,
    }


def run(config: dict, force: bool = False) -> dict:
    out_path = output_path(config, OUTPUT_FILENAME)
    if not force and checkpoint_skip(out_path, STAGE_NAME):
        return load_json(out_path)

    result = _compare(config)
    save_json(out_path, result)
    print(f"\n[{STAGE_NAME}] 비교 완료 → {out_path}")
    return result
