"""Stage 3a: 멀티에이전트 토론 → ideal vector.

입력: config.json (case_context, axes, ideal_methods.debate)
출력: outputs/03a_ideal_debate.json

Turn 1 (독립 반영) → [Round 1..n_rounds: 상호 코멘트 → 답변(입장/점수 갱신)] → 요약자(consensus 추출)
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

from pipeline.common import (
    call_llm,
    checkpoint_skip,
    dict_to_vector,
    dilemma_axes_description,
    load_json,
    normalize_vector,
    output_path,
    parse_llm_json,
    save_json,
    score_json_fields,
)

STAGE_NAME = "stage3a"
OUTPUT_FILENAME = "03a_ideal_debate.json"

SUMMARIZER_PROVIDER = "openai"
SUMMARIZER_MODEL = "gpt-4.1-mini"


@dataclass
class Agent:
    id: int
    model: str
    provider: str
    label: str


@dataclass
class AgentState:
    agent: Agent
    position: str = ""
    scores: Optional[dict] = None
    comment_target: Optional[int] = None
    comment_text: Optional[str] = None
    reply_text: Optional[str] = None
    history: list = field(default_factory=list)


def infer_provider(model: str) -> str:
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    raise ValueError(f"모델 '{model}'의 provider를 추론할 수 없습니다.")


def build_agents(models: list[str], agents_per_model: int) -> list[Agent]:
    agents = []
    agent_id = 1
    for model in models:
        provider = infer_provider(model)
        for i in range(agents_per_model):
            agents.append(Agent(id=agent_id, model=model, provider=provider, label=f"{model}-{i + 1}"))
            agent_id += 1
    return agents


# ── Turn 1: 독립 반영 ─────────────────────────────────────────────────────────

async def run_turn1(state: AgentState, case_context: str, axes_desc: str, score_fields: str) -> None:
    system = (
        "You are a thoughtful participant in a moral deliberation exercise. "
        "Respond concisely and honestly based on your own reasoning. "
        "Do not try to be neutral — take a genuine position."
    )
    user = (
        f"Read the following dilemma carefully:\n{case_context}\n\n"
        "1) Give your overall assessment of this situation and where you stand. "
        "State your position in 2-3 sentences.\n\n"
        f"2) {axes_desc}\n\n"
        "Respond ONLY in this JSON format:\n"
        "{\n"
        '  "position": "your 2-3 sentence stance",\n'
        '  "scores": {\n'
        f"{score_fields}\n"
        "  }\n"
        "}"
    )

    raw = await call_llm(
        provider=state.agent.provider, model=state.agent.model, system=system, user=user,
        temperature=0.8, max_tokens=1024,
    )
    try:
        data = parse_llm_json(raw)
        state.position = data.get("position", raw)
        state.scores = data.get("scores", {})
    except Exception:
        state.position = raw
        state.scores = {}

    state.history.append({"round": 0, "position": state.position, "scores": state.scores})
    print(f"  [{state.agent.label}] Turn 1 done | scores: {state.scores}")


# ── Round N: 상호 코멘트 ──────────────────────────────────────────────────────

async def run_comment(state: AgentState, all_states: list[AgentState], round_num: int) -> None:
    others = [s for s in all_states if s.agent.id != state.agent.id]
    others_text = "\n\n".join(f"Agent {s.agent.label} (ID {s.agent.id}):\n{s.position}" for s in others)

    system = (
        "You are participating in a structured moral debate. "
        "You have already stated your own position. "
        "Now read other participants' current views and engage critically with one of them."
    )
    user = (
        f"Your own position so far:\n{state.position}\n\n"
        f"Other participants currently say:\n{others_text}\n\n"
        "Pick ONE participant whose view you want to challenge or build upon. "
        "Explain why you chose them and what you think about their position in 2-3 sentences.\n\n"
        "Respond ONLY in this JSON format:\n"
        "{\n"
        '  "target_agent_id": <integer>,\n'
        '  "comment": "your 2-3 sentence comment"\n'
        "}"
    )

    raw = await call_llm(
        provider=state.agent.provider, model=state.agent.model, system=system, user=user,
        temperature=0.8, max_tokens=1024,
    )
    try:
        data = parse_llm_json(raw)
        state.comment_target = int(data.get("target_agent_id", others[0].agent.id))
        state.comment_text = data.get("comment", raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        state.comment_target = others[0].agent.id
        state.comment_text = raw

    print(f"  [{state.agent.label}] Round {round_num} comment → targeting Agent {state.comment_target}")


# ── Round N: 답변 (입장/점수 갱신) ─────────────────────────────────────────────

async def run_reply(state: AgentState, all_states: list[AgentState], round_num: int, score_fields: str) -> None:
    incoming = [s for s in all_states if s.comment_target == state.agent.id and s.comment_text]

    if not incoming:
        state.reply_text = None
        print(f"  [{state.agent.label}] Round {round_num} reply skipped (no comments received)")
        return

    comments_text = "\n\n".join(f"From {s.agent.label}:\n{s.comment_text}" for s in incoming)

    system = (
        "You are in a moral deliberation. "
        "After hearing challenges to your position, reflect carefully. "
        "You may update your position and scores if the arguments genuinely changed your mind."
    )
    user = (
        f"Your position so far:\n{state.position}\n\n"
        f"Your scores so far: {json.dumps(state.scores)}\n\n"
        f"You received these comments:\n{comments_text}\n\n"
        "Respond to the comments. Have any arguments changed your view? "
        "Update your position and scores if your thinking shifted.\n\n"
        "Respond ONLY in this JSON format:\n"
        "{\n"
        '  "updated_position": "your updated 2-3 sentence stance",\n'
        '  "updated_scores": {\n'
        f"{score_fields}\n"
        "  }\n"
        "}"
    )

    raw = await call_llm(
        provider=state.agent.provider, model=state.agent.model, system=system, user=user,
        temperature=0.8, max_tokens=1024,
    )
    try:
        data = parse_llm_json(raw)
        state.reply_text = data.get("updated_position", raw)
        state.position = state.reply_text
        updated = data.get("updated_scores", {})
        if updated:
            state.scores = updated
    except Exception:
        state.reply_text = raw

    print(f"  [{state.agent.label}] Round {round_num} reply done | scores: {state.scores}")


# ── 요약자: 전체 라운드 transcript → consensus ideal vector ───────────────────

def build_transcript_text(states: list[AgentState]) -> str:
    transcript = ""
    for s in states:
        transcript += f"\n=== {s.agent.label} ({s.agent.provider}) ===\n"
        for entry in s.history:
            if entry["round"] == 0:
                transcript += f"Initial position: {entry['position']}\n"
                transcript += f"Initial scores: {entry['scores']}\n"
            else:
                r = entry["round"]
                if entry["comment"]:
                    transcript += f"Round {r} (commented on Agent {entry['comment_target']}): {entry['comment']}\n"
                if entry["reply"]:
                    transcript += f"Round {r} reply: {entry['reply']}\n"
                    transcript += f"Round {r} scores: {entry['scores']}\n"
    return transcript


async def run_summarizer(states: list[AgentState], axes_desc: str, score_fields: str) -> dict:
    transcript = build_transcript_text(states)

    system = (
        "You are a neutral synthesizer reading a moral debate transcript. "
        "Your job is to extract the collective moral consensus — not the average, "
        "but the direction the group converged toward after deliberation."
    )
    user = (
        f"Here is the full debate transcript across all rounds:\n{transcript}\n\n"
        f"The axes are:\n{axes_desc}\n\n"
        "Based on the overall arc of this debate — where the group ended up "
        "after challenging each other across all rounds — extract a consensus ideal vector.\n\n"
        "Respond ONLY in this JSON format:\n"
        "{\n"
        '  "consensus_summary": "2-3 sentence description of where the debate converged",\n'
        '  "ideal_vector": {\n'
        f"{score_fields}\n"
        "  }\n"
        "}"
    )

    raw = await call_llm(
        provider=SUMMARIZER_PROVIDER, model=SUMMARIZER_MODEL, system=system, user=user,
        temperature=0.3, max_tokens=1024,
    )
    try:
        data = parse_llm_json(raw)
        print(f"\n  [Summarizer] Consensus: {data.get('consensus_summary', '')}")
        print(f"  [Summarizer] Ideal vector: {data.get('ideal_vector', {})}")
        return data.get("ideal_vector", {})
    except Exception:
        print(f"  [Summarizer] JSON parse failed, raw: {raw[:200]}")
        return {}


# ── 전체 디베이트 파이프라인 ───────────────────────────────────────────────────

async def run_debate(agents: list[Agent], case_context: str, axes: list[dict], n_rounds: int) -> tuple:
    axes_desc = dilemma_axes_description(axes)
    fields = score_json_fields(axes)
    axis_names = [ax["name"] for ax in axes]

    print("\n" + "=" * 60)
    print("MULTI-AGENT DEBATE — Justice Vector Simulator")
    print("=" * 60)

    states = [AgentState(agent=a) for a in agents]

    print("\n[Turn 1] All agents reflecting independently...")
    await asyncio.gather(*(run_turn1(s, case_context, axes_desc, fields) for s in states))

    for r in range(1, n_rounds + 1):
        for s in states:
            s.comment_target = None
            s.comment_text = None
            s.reply_text = None

        print(f"\n[Round {r}] Cross-agent commenting...")
        await asyncio.gather(*(run_comment(s, states, r) for s in states))

        print(f"\n[Round {r}] Replies to comments...")
        await asyncio.gather(*(run_reply(s, states, r, fields) for s in states))

        for s in states:
            s.history.append({
                "round": r,
                "comment_target": s.comment_target,
                "comment": s.comment_text,
                "reply": s.reply_text,
                "position": s.position,
                "scores": s.scores,
            })

    print("\n[Summarizer] Extracting consensus vector...")
    consensus = await run_summarizer(states, axes_desc, fields)

    if consensus:
        raw = dict_to_vector(consensus, axis_names)
    else:
        scored = [s.scores for s in states if s.scores]
        if scored:
            raw = [sum(sc.get(name, 50) for sc in scored) / len(scored) for name in axis_names]
        else:
            raw = [50.0] * len(axis_names)

    debate_ideal_unit = normalize_vector(raw)

    print(f"\n  Debate ideal vector (raw):       {[round(v, 1) for v in raw]}")
    print(f"  Debate ideal vector (normalized): {[round(v, 3) for v in debate_ideal_unit]}")

    print("\n[Per-agent final scores]")
    for s in states:
        print(f"  {s.agent.label:12s} ({s.agent.provider:9s}): {s.scores}")

    return debate_ideal_unit, states


async def _run_async(config: dict) -> dict:
    case_context = config["case_context"]
    axes = config["axes"]
    axis_names = [ax["name"] for ax in axes]
    debate_cfg = config["ideal_methods"]["debate"]

    models = debate_cfg["models"]
    agents_per_model = debate_cfg.get("agents_per_model", 2)
    n_rounds = debate_cfg.get("n_rounds", 1)

    agents = build_agents(models, agents_per_model)
    debate_ideal_unit, states = await run_debate(agents, case_context, axes, n_rounds)

    return {
        "ideal_unit_vector": debate_ideal_unit,
        "ideal_vector": {name: v for name, v in zip(axis_names, debate_ideal_unit)},
        "n_rounds": n_rounds,
        "per_agent_scores": [
            {"agent": s.agent.label, "model": s.agent.model, "provider": s.agent.provider, "final_scores": s.scores}
            for s in states
        ],
        "debate_transcript": [
            {"agent": s.agent.label, "model": s.agent.model, "provider": s.agent.provider, "history": s.history}
            for s in states
        ],
    }


def run(config: dict, force: bool = False) -> dict:
    out_path = output_path(config, OUTPUT_FILENAME)
    if not force and checkpoint_skip(out_path, STAGE_NAME):
        return load_json(out_path)

    result = asyncio.run(_run_async(config))
    save_json(out_path, result)
    print(f"\n[{STAGE_NAME}] 결과 저장 → {out_path}")
    return result
