import json
import math
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))


def cosine_angle(v1: dict, v2: dict) -> float:
    keys = list(v1.keys())
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(v1[k] ** 2 for k in keys))
    mag2 = math.sqrt(sum(v2[k] ** 2 for k in keys))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosine))


def average_vector(vectors: list[dict]) -> dict:
    if not vectors:
        return {}
    keys = list(vectors[0].keys())
    return {k: sum(v[k] for v in vectors) / len(vectors) for k in keys}


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
                    "interpretation": interpret_consistency(angle)
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
                "vector_to": tp_vectors[t2]
            })
    return results


def run_analysis(extracted_path: str):
    with open(extracted_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    result = {
        "consistency_analysis": consistency,
        "drift_analysis": drift
    }

    out_path = os.path.join(BASE, "data", "analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n분석 완료 → {out_path}")
    return result


if __name__ == "__main__":
    run_analysis(os.path.join(BASE, "data", "extracted.json"))
