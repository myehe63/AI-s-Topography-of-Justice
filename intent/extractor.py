import anthropic
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# 기존 축 고정 (자동생성 X)
AXES = [
    {
        "id": "axis_1",
        "name": "state_authority",
        "description": "Should the state/military have the right to override a private company's ethical restrictions on contracted technology?",
        "low_label": "Company ethical restrictions must be respected",
        "high_label": "State authority is supreme"
    },
    {
        "id": "axis_2",
        "name": "corporate_ethics",
        "description": "Should private companies have the right to refuse uses of their technology that violate their ethical principles?",
        "low_label": "Companies have no right to refuse government use",
        "high_label": "Companies have full right to refuse"
    },
    {
        "id": "axis_3",
        "name": "civilian_safety",
        "description": "Should protecting civilians from autonomous weapons and mass surveillance be treated as a non-negotiable priority above both national security arguments and corporate interests?",
        "low_label": "National security / corporate interests can override civilian safety",
        "high_label": "Civilian protection is paramount and non-negotiable"
    }
]


def classify_timepoints(case_context: str, sources: list) -> dict:
    dates_and_types = "\n".join([
        f"{s['id']} | {s['actor']} | {s['date']} | {s['type']} | {s['text'][:100]}"
        for s in sources
    ])

    prompt = f"""다음 사건의 소스들을 시간순으로 보고, 의미있는 시점 구간을 3~5개 설계해줘.
그리고 각 소스가 어느 시점에 속하는지 분류해줘.

사건 맥락:
{case_context}

소스 목록:
{dates_and_types}

JSON 형식으로만 답해줘:
{{
  "timepoints": [
    {{
      "id": "T0",
      "label": "시점 이름",
      "description": "이 시점의 특징",
      "date_range": "YYYY-MM ~ YYYY-MM"
    }}
  ],
  "source_timepoint_map": {{
    "src_d01": "T0",
    "src_d02": "T1"
  }}
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def extract_vector(source: dict, case_context: str) -> dict:
    axes_description = "\n".join([
        f"- {ax['name']}: {ax['description']} (0={ax['low_label']}, 100={ax['high_label']})"
        for ax in AXES
    ])

    prompt = f"""다음 텍스트가 각 축에서 어떤 입장을 취하는지 0-100으로 점수화해줘.

사건 맥락: {case_context}

텍스트:
[{source['actor']} / {source['date']} / {source['type']}]
{source['text']}

축 설명:
{axes_description}

규칙:
- 텍스트에 해당 축 언급이 없으면 50 (중립)
- 점수는 텍스트 내용만 보고 판단

JSON 형식으로만 답해줘:
{{
  "scores": {{
    "state_authority": 점수,
    "corporate_ethics": 점수,
    "civilian_safety": 점수
  }},
  "reasoning": "한 줄 설명"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)

    return {
        "source_id": source["id"],
        "actor": source["actor"],
        "date": source["date"],
        "type": source["type"],
        "vector": result["scores"],
        "reasoning": result["reasoning"]
    }


def run_extraction(data_path: str) -> dict:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    case_context = data["case_context"]
    sources = data["sources"]

    print("=== 축 (고정) ===")
    for ax in AXES:
        print(f"  [{ax['name']}] {ax['description']}")

    print("\n=== 1단계: 시점 분류 ===")
    timepoint_data = classify_timepoints(case_context, sources)
    for tp in timepoint_data["timepoints"]:
        print(f"  {tp['id']} ({tp['date_range']}): {tp['label']}")

    print("\n=== 2단계: 벡터 추출 ===")
    vectors = []
    for source in sources:
        print(f"  추출 중: {source['id']} ({source['actor']} / {source['date']})")
        vec = extract_vector(source, case_context)
        tp_id = timepoint_data["source_timepoint_map"].get(source["id"], "unknown")
        vec["timepoint"] = tp_id
        vectors.append(vec)
        print(f"    → {vec['vector']}")

    result = {
        "case_context": case_context,
        "axes": AXES,
        "timepoints": timepoint_data["timepoints"],
        "vectors": vectors
    }

    out_path = os.path.join(BASE, "data", "extracted.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n추출 완료 → {out_path}")
    return result


if __name__ == "__main__":
    run_extraction(os.path.join(BASE, "data", "sources.json"))
