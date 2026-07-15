"""Stage 2: intent 벡터 내부 일관성/drift 분석.

입력: outputs/01_intent_extracted.json
출력: outputs/02_intent_drift.json
"""

from collections import defaultdict

from pipeline.common import (
    average_vector,
    checkpoint_skip,
    cosine_angle,
    load_json,
    output_path,
    save_json,
)

STAGE_NAME = "stage2"
INPUT_FILENAME = "01_intent_extracted.json"
OUTPUT_FILENAME = "02_intent_drift.json"


def interpret_consistency(angle: float) -> str:
    if angle < 5:
        return "매우 일관됨 — 소스간 신뢰도 높음"
    elif angle < 15:
        return "대체로 일관됨"
    elif angle < 30:
        return "소스간 불일치 존재 — 해석 주의"
    else:
        return "소스간 심각한 불일치 — 의도 모호성 높음"


def interpret_drift(angle: float) -> str:
    if angle < 5:
        return "거의 변화 없음"
    elif angle < 15:
        return "완만한 drift"
    elif angle < 30:
        return "뚜렷한 방향 전환"
    else:
        return "급격한 입장 변화 — 주목 필요"


def analyze_source_consistency(vectors: list) -> list:
    results = []
    groups = defaultdict(list)
    for v in vectors:
        key = f"{v['actor']}_{v['timepoint']}"
        groups[key].append(v)

    for group_key, group_vectors in groups.items():
        if len(group_vectors) < 2:
            continue
        for i in range(len(group_vectors)):
            for j in range(i + 1, len(group_vectors)):
                v1, v2 = group_vectors[i], group_vectors[j]
                angle = cosine_angle(v1["vector"], v2["vector"])
                results.append({
                    "group": group_key,
                    "source_1": f"{v1['source_id']} ({v1['type']})",
                    "source_2": f"{v2['source_id']} ({v2['type']})",
                    "angle": round(angle, 2),
                    "interpretation": interpret_consistency(angle),
                })
    return results


def analyze_drift(vectors: list, timepoints: list) -> list:
    results = []
    actors = list(set(v["actor"] for v in vectors))
    tp_order = [tp["id"] for tp in timepoints]

    for actor in actors:
        actor_vectors = [v for v in vectors if v["actor"] == actor]
        tp_vectors = {}
        for tp_id in tp_order:
            tp_vecs = [v["vector"] for v in actor_vectors if v["timepoint"] == tp_id]
            if tp_vecs:
                tp_vectors[tp_id] = average_vector(tp_vecs)

        if len(tp_vectors) < 2:
            continue

        tp_ids = [tp for tp in tp_order if tp in tp_vectors]
        for i in range(len(tp_ids) - 1):
            t1, t2 = tp_ids[i], tp_ids[i + 1]
            angle = cosine_angle(tp_vectors[t1], tp_vectors[t2])
            results.append({
                "actor": actor,
                "from_timepoint": t1,
                "to_timepoint": t2,
                "angle": round(angle, 2),
                "interpretation": interpret_drift(angle),
                "vector_from": tp_vectors[t1],
                "vector_to": tp_vectors[t2],
            })
    return results


def _analyze(data: dict) -> dict:
    vectors = data["vectors"]
    timepoints = data["timepoints"]
    axes = data["axes"]

    print("=== 축 정보 ===")
    for ax in axes:
        print(f"  [{ax['name']}] 0={ax['low_label']} → 100={ax['high_label']}")

    print("\n=== 소스간 일관성 분석 (정당성 검증) ===")
    consistency = analyze_source_consistency(vectors)
    if consistency:
        for c in consistency:
            print(f"  {c['source_1']} vs {c['source_2']}")
            print(f"    각도: {c['angle']}° → {c['interpretation']}")
    else:
        print("  (같은 시점에 동일 actor의 소스가 2개 이상 있어야 분석 가능)")

    print("\n=== 시점간 Drift 분석 ===")
    drift = analyze_drift(vectors, timepoints)
    for d in drift:
        print(f"  [{d['actor']}] {d['from_timepoint']} → {d['to_timepoint']}")
        print(f"    각도: {d['angle']}° → {d['interpretation']}")

    return {
        "consistency_analysis": consistency,
        "drift_analysis": drift,
    }


def run(config: dict, force: bool = False) -> dict:
    out_path = output_path(config, OUTPUT_FILENAME)
    if not force and checkpoint_skip(out_path, STAGE_NAME):
        return load_json(out_path)

    data = load_json(output_path(config, INPUT_FILENAME))
    result = _analyze(data)
    save_json(out_path, result)
    print(f"\n[{STAGE_NAME}] 분석 완료 → {out_path}")
    return result
