"""Pipeline 공통 함수: JSON 입출력, 체크포인트, 케이스 config 로드, 벡터/각도 계산, LLM 호출."""

import json
import math
import os
from typing import Any, Optional


# ── JSON 입출력 ──────────────────────────────────────────────────────────────

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_llm_json(raw: str) -> Any:
    """LLM 응답에서 마크다운 코드펜스(```json ... ```)를 제거하고 JSON으로 파싱."""
    clean = raw.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def checkpoint_skip(output_path: str, stage_name: str) -> bool:
    """출력 파일이 이미 있으면 True를 반환하고 스킵 메시지를 출력한다."""
    if os.path.exists(output_path):
        print(f"[{stage_name}] 이미 존재, 건너뜀 → {output_path}")
        return True
    return False


# ── 케이스 config 로드 ────────────────────────────────────────────────────────

def load_case_config(case_id: str, cases_root: str = "cases") -> dict:
    """cases/<case_id>/config.json을 읽고, 케이스/출력 디렉토리 경로를 덧붙여 반환."""
    case_dir = os.path.join(cases_root, case_id)
    config = load_json(os.path.join(case_dir, "config.json"))
    config["_case_dir"] = case_dir
    config["_outputs_dir"] = os.path.join(case_dir, "outputs")
    config["_graphs_dir"] = os.path.join(case_dir, "outputs", "graphs")
    return config


def axis_names(config: dict) -> list[str]:
    return [ax["name"] for ax in config["axes"]]


def output_path(config: dict, filename: str) -> str:
    return os.path.join(config["_outputs_dir"], filename)


def graph_path(config: dict, filename: str) -> str:
    return os.path.join(config["_graphs_dir"], filename)


# ── 벡터/각도 계산 ────────────────────────────────────────────────────────────

def _angle_from_components(dot: float, mag1: float, mag2: float) -> float:
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosine))


def cosine_angle(v1: dict, v2: dict) -> float:
    """{축이름: 점수} 형태 벡터 두 개 사이의 각도(도). 공통 축만 사용."""
    keys = [k for k in v1 if k in v2]
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(v1[k] ** 2 for k in keys))
    mag2 = math.sqrt(sum(v2[k] ** 2 for k in keys))
    return _angle_from_components(dot, mag1, mag2)


def vector_angle(v1, v2) -> float:
    """순서가 있는 벡터(list/tuple/np.ndarray) 두 개 사이의 각도(도)."""
    v1, v2 = list(v1), list(v2)
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return _angle_from_components(dot, mag1, mag2)


def normalize_vector(vec) -> list[float]:
    vec = list(vec)
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 1e-10 else vec


def average_vector(vectors: list[dict]) -> dict:
    """{축이름: 점수} 딕셔너리 리스트의 축별 평균."""
    if not vectors:
        return {}
    keys = list(vectors[0].keys())
    return {k: sum(v[k] for v in vectors) / len(vectors) for k in keys}


def dict_to_vector(scores: dict, axes: list[str]) -> list[float]:
    """{축이름: 점수} 딕셔너리를 axes 순서의 리스트로 변환. 누락된 축은 50(중립)."""
    return [scores.get(ax, 50) for ax in axes]


# ── LLM 프롬프트용 축 설명 / JSON 템플릿 (stage3a, stage3b 공통) ──────────────────

def dilemma_axes_description(axes: list[dict]) -> str:
    """딜레마 프롬프트 뒤에 이어붙이는 축 설명 (에이전트/페르소나 시뮬레이션용)."""
    lines = ["Rate the following on 0-100 based on the dilemma above:"]
    for ax in axes:
        lines.append(f"- {ax['name']}: {ax['description']} (100 = {ax['high_label']})")
    return "\n".join(lines)


def score_json_fields(axes: list[dict], placeholder: str = "<integer 0-100>") -> str:
    """{축이름: placeholder} 형태의 JSON 필드 블록을 축 개수만큼 동적으로 생성."""
    return ",\n".join(f'    "{ax["name"]}": {placeholder}' for ax in axes)


# ── LLM 호출 ─────────────────────────────────────────────────────────────────

async def call_llm(
    provider: str,
    model: str,
    user: str,
    system: str = "",
    temperature: float = 0.8,
    max_tokens: int = 1000,
) -> str:
    """OpenAI / Anthropic / Gemini 통합 호출. provider: "openai" | "anthropic" | "gemini" """

    if provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return resp.content[0].text.strip()

    if provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        resp = await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return resp.choices[0].message.content.strip()

    if provider == "gemini":
        import asyncio
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        cfg = {"temperature": temperature}
        if system:
            cfg["system_instruction"] = system
        resp = await asyncio.to_thread(
            client.models.generate_content, model=model, contents=user, config=cfg
        )
        return resp.text.strip()

    raise ValueError(f"Unknown provider: {provider}")
