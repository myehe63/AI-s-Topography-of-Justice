"""Stage 1: 실제 발언(raw_sources.json) → intent 벡터화.

입력: config.json (case_context, axes), raw_sources.json
출력: outputs/01_intent_extracted.json
"""

import asyncio
import os

from pipeline.common import (
    call_llm,
    checkpoint_skip,
    load_json,
    output_path,
    parse_llm_json,
    save_json,
)

STAGE_NAME = "stage1"
OUTPUT_FILENAME = "01_intent_extracted.json"


async def classify_timepoints(case_context: str, sources: list, llm_config: dict) -> dict:
    dates_and_types = "\n".join(
        f"{s['id']} | {s['actor']} | {s['date']} | {s['type']} | {s['text'][:100]}"
        for s in sources
    )

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

    raw = await call_llm(
        provider="anthropic",
        model=llm_config["model"],
        user=prompt,
        max_tokens=llm_config.get("max_tokens", 1000),
    )
    return parse_llm_json(raw)


async def extract_vector(source: dict, case_context: str, axes: list, llm_config: dict) -> dict:
    axes_description = "\n".join(
        f"- {ax['name']}: {ax['description']} (0={ax['low_label']}, 100={ax['high_label']})"
        for ax in axes
    )

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
{("," + chr(10)).join(f'    "{ax["name"]}": 점수' for ax in axes)}
  }},
  "reasoning": "한 줄 설명"
}}"""

    raw = await call_llm(
        provider="anthropic",
        model=llm_config["model"],
        user=prompt,
        max_tokens=500,
    )
    result = parse_llm_json(raw)

    return {
        "source_id": source["id"],
        "actor": source["actor"],
        "date": source["date"],
        "type": source["type"],
        "vector": result["scores"],
        "reasoning": result["reasoning"],
    }


async def _run_async(config: dict) -> dict:
    case_context = config["case_context"]
    axes = config["axes"]
    llm_config = config.get("llm", {"model": "claude-sonnet-4-6"})

    sources_path = os.path.join(config["_case_dir"], config.get("sources_path", "raw_sources.json"))
    sources_data = load_json(sources_path)
    sources = sources_data["sources"]

    print("=== 축 (config) ===")
    for ax in axes:
        print(f"  [{ax['name']}] {ax['description']}")

    print("\n=== 1단계: 시점 분류 ===")
    timepoint_data = await classify_timepoints(case_context, sources, llm_config)
    for tp in timepoint_data["timepoints"]:
        print(f"  {tp['id']} ({tp['date_range']}): {tp['label']}")

    print("\n=== 2단계: 벡터 추출 ===")
    vectors = []
    for source in sources:
        print(f"  추출 중: {source['id']} ({source['actor']} / {source['date']})")
        vec = await extract_vector(source, case_context, axes, llm_config)
        tp_id = timepoint_data["source_timepoint_map"].get(source["id"], "unknown")
        vec["timepoint"] = tp_id
        vectors.append(vec)
        print(f"    → {vec['vector']}")

    return {
        "case_context": case_context,
        "axes": axes,
        "timepoints": timepoint_data["timepoints"],
        "vectors": vectors,
    }


def run(config: dict, force: bool = False) -> dict:
    out_path = output_path(config, OUTPUT_FILENAME)
    if not force and checkpoint_skip(out_path, STAGE_NAME):
        return load_json(out_path)

    result = asyncio.run(_run_async(config))
    save_json(out_path, result)
    print(f"\n[{STAGE_NAME}] 추출 완료 → {out_path}")
    return result
