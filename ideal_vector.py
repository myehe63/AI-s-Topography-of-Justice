"""
Multi-Agent Debate Pipeline for Justice Vector Simulator
=========================================================
Inspired by Whispering Water (Wang et al., MIT Media Lab 2026):
agent identities emerge dynamically through discourse, not pre-assignment.

Round structure:
  Turn 1 — All agents independently reflect on the dilemma (parallel)
  Turn 2 — Each agent reads all Turn 1 responses and picks one to comment on
  Turn 3 — Agents that received comments reply to them
  Turn 4 — Summarizer extracts consensus ideal vector as JSON

Usage:
  export OPENAI_API_KEY="..."
  export ANTHROPIC_API_KEY="..."
  export GEMINI_API_KEY="..."
  python agent_debate.py
"""

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ── API clients ──────────────────────────────────────────────────────────────
from openai import AsyncOpenAI
import anthropic
from google import genai

openai_client  = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
claude_client  = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
gemini_client  = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ── Dilemma ───────────────────────────────────────────────────────────────────
DILEMMA = """
In 2025, an AI company signed a $200M contract with a national military.
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

# ── Agent definitions ─────────────────────────────────────────────────────────
@dataclass
class Agent:
    id: int
    model: str          # "gpt-4.1-mini" | "claude-sonnet-4-5" | "gemini-2.0-flash"
    provider: str       # "openai" | "anthropic" | "gemini"
    label: str          # display name

AGENTS = [
    Agent(1, "gpt-4.1-mini",        "openai",    "GPT-1"),
    Agent(2, "gpt-4.1-mini",        "openai",    "GPT-2"),
    Agent(3, "claude-sonnet-4-5", "anthropic", "Claude-1"),
    Agent(4, "claude-sonnet-4-5", "anthropic", "Claude-2"),
    Agent(5, "gemini-2.5-flash-lite", "gemini", "Gemini-1"),
    Agent(6, "gemini-2.5-flash-lite", "gemini", "Gemini-2"),
]

# ── Conversation history per agent ────────────────────────────────────────────
@dataclass
class AgentState:
    agent: Agent
    turn1_response: str = ""
    turn2_comment: Optional[str] = None      # what this agent said in Turn 2
    turn2_target: Optional[int] = None       # which agent_id this agent commented on
    turn3_reply: Optional[str] = None        # reply to comments received
    scores: Optional[dict] = None            # final extracted scores


# ── LLM call helpers ──────────────────────────────────────────────────────────

async def call_llm(agent: Agent, messages: list[dict]) -> str:
    """Unified LLM call across OpenAI / Anthropic / Gemini."""

    if agent.provider == "openai":
        resp = await openai_client.chat.completions.create(
            model=agent.model,
            messages=messages,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()

    elif agent.provider == "anthropic":
        # Anthropic separates system from messages
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = await claude_client.messages.create(
            model=agent.model,
            max_tokens=1024,
            system=system,
            messages=user_msgs,
            temperature=0.8,
        )
        return resp.content[0].text.strip()

    elif agent.provider == "gemini":
        # Gemini via new google.genai SDK
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        flat = "\n\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in user_msgs
        )
        cfg = {"system_instruction": system, "temperature": 0.8} if system else {"temperature": 0.8}
        resp = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=agent.model,
            contents=flat,
            config=cfg,
        )
        return resp.text.strip()
    raise ValueError(f"Unknown provider: {agent.provider}")


# ── Turn 1: Independent reflection ───────────────────────────────────────────

async def run_turn1(state: AgentState) -> None:
    """Each agent independently reflects on the dilemma and gives initial scores."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a thoughtful participant in a moral deliberation exercise. "
                "Respond concisely and honestly based on your own reasoning. "
                "Do not try to be neutral — take a genuine position."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Read the following dilemma carefully:\n{DILEMMA}\n\n"
                "1) Did the company make the right decision to refuse? "
                "Give your position in 2-3 sentences.\n\n"
                f"2) {AXES_DESCRIPTION}\n\n"
                "Respond ONLY in this JSON format:\n"
                '{\n'
                '  "position": "your 2-3 sentence stance",\n'
                '  "scores": {\n'
                '    "state_authority": <integer 0-100>,\n'
                '    "corporate_ethics": <integer 0-100>,\n'
                '    "civilian_safety": <integer 0-100>\n'
                '  }\n'
                '}'
            ),
        },
    ]

    raw = await call_llm(state.agent, messages)
    try:
        # strip markdown fences if present
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        state.turn1_response = data.get("position", raw)
        state.scores = data.get("scores", {})
    except json.JSONDecodeError:
        state.turn1_response = raw
        state.scores = {}

    print(f"  [{state.agent.label}] Turn 1 done | scores: {state.scores}")
    preview = state.turn1_response[:120] + ("..." if len(state.turn1_response) > 120 else "")
    print(f"    └─ \"{preview}\"")


# ── Turn 2: Cross-agent commenting ───────────────────────────────────────────

async def run_turn2(state: AgentState, all_states: list[AgentState]) -> None:
    """Each agent reads all Turn 1 responses, picks one to comment on."""
    others = [s for s in all_states if s.agent.id != state.agent.id]
    others_text = "\n\n".join(
        f"Agent {s.agent.label} (ID {s.agent.id}):\n{s.turn1_response}"
        for s in others
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are participating in a structured moral debate. "
                "You have already stated your own position. "
                "Now read other participants' views and engage critically with one of them."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Your own position was:\n{state.turn1_response}\n\n"
                f"Other participants said:\n{others_text}\n\n"
                "Pick ONE participant whose view you want to challenge or build upon. "
                "Explain why you chose them and what you think about their position in 2-3 sentences.\n\n"
                "Respond ONLY in this JSON format:\n"
                '{\n'
                '  "target_agent_id": <integer>,\n'
                '  "comment": "your 2-3 sentence comment"\n'
                '}'
            ),
        },
    ]

    raw = await call_llm(state.agent, messages)
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        state.turn2_target = int(data.get("target_agent_id", others[0].agent.id))
        state.turn2_comment = data.get("comment", raw)
    except (json.JSONDecodeError, ValueError):
        state.turn2_target = others[0].agent.id
        state.turn2_comment = raw

    print(f"  [{state.agent.label}] Turn 2 → commenting on Agent {state.turn2_target}")
    if state.turn2_comment:
        preview = state.turn2_comment[:120] + ("..." if len(state.turn2_comment) > 120 else "")
        print(f"    └─ \"{preview}\"")


# ── Turn 3: Reply to comments received ───────────────────────────────────────

async def run_turn3(state: AgentState, all_states: list[AgentState]) -> None:
    """Agents that received comments reply to them. Updates scores after hearing others."""
    # Find all comments directed at this agent
    incoming = [
        s for s in all_states
        if s.turn2_target == state.agent.id and s.turn2_comment
    ]

    if not incoming:
        print(f"  [{state.agent.label}] Turn 3 skipped (no comments received)")
        return

    comments_text = "\n\n".join(
        f"From {s.agent.label}:\n{s.turn2_comment}"
        for s in incoming
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are in a moral deliberation. "
                "After hearing challenges to your position, reflect carefully. "
                "You may update your scores if the arguments genuinely changed your mind."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Your original position:\n{state.turn1_response}\n\n"
                f"Your original scores: {json.dumps(state.scores)}\n\n"
                f"You received these comments:\n{comments_text}\n\n"
                "Respond to the comments. Have any arguments changed your view? "
                "Update your scores if your thinking shifted.\n\n"
                "Respond ONLY in this JSON format:\n"
                '{\n'
                '  "reply": "your 2-3 sentence reply",\n'
                '  "updated_scores": {\n'
                '    "state_authority": <integer 0-100>,\n'
                '    "corporate_ethics": <integer 0-100>,\n'
                '    "civilian_safety": <integer 0-100>\n'
                '  }\n'
                '}'
            ),
        },
    ]

    raw = await call_llm(state.agent, messages)
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        state.turn3_reply = data.get("reply", raw)
        updated = data.get("updated_scores", {})
        if updated:
            state.scores = updated  # overwrite with post-debate scores
    except json.JSONDecodeError:
        state.turn3_reply = raw

    print(f"  [{state.agent.label}] Turn 3 done | updated scores: {state.scores}")
    if state.turn3_reply:
       preview = state.turn3_reply[:120] + ("..." if len(state.turn3_reply) > 120 else "")
    print(f"    └─ \"{preview}\"")

# ── Turn 4: Summarizer extracts consensus vector ──────────────────────────────

async def run_turn4_summarizer(all_states: list[AgentState]) -> dict:
    """
    GPT-4.1 summarizer reads the full debate transcript and extracts
    a consensus ideal vector. This is the debate-based ideal vector.
    """
    transcript = ""
    for s in all_states:
        transcript += f"\n=== {s.agent.label} ({s.agent.provider}) ===\n"
        transcript += f"Turn 1: {s.turn1_response}\n"
        transcript += f"Scores after Turn 1: {s.scores}\n"
        if s.turn2_comment:
            transcript += f"Turn 2 (commented on Agent {s.turn2_target}): {s.turn2_comment}\n"
        if s.turn3_reply:
            transcript += f"Turn 3 (reply): {s.turn3_reply}\n"
            transcript += f"Final scores: {s.scores}\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a neutral synthesizer reading a moral debate transcript. "
                "Your job is to extract the collective moral consensus — not the average, "
                "but the direction the group converged toward after deliberation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Here is the full debate transcript:\n{transcript}\n\n"
                f"The three axes are:\n{AXES_DESCRIPTION}\n\n"
                "Based on the overall arc of this debate — where the group ended up "
                "after challenging each other — extract a consensus ideal vector.\n\n"
                "Respond ONLY in this JSON format:\n"
                '{\n'
                '  "consensus_summary": "2-3 sentence description of where the debate converged",\n'
                '  "ideal_vector": {\n'
                '    "state_authority": <integer 0-100>,\n'
                '    "corporate_ethics": <integer 0-100>,\n'
                '    "civilian_safety": <integer 0-100>\n'
                '  }\n'
                '}'
            ),
        },
    ]

    resp = await openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.3,  # low temp for summarizer — we want consistency
    )
    raw = resp.choices[0].message.content.strip()

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        print(f"\n  [Summarizer] Consensus: {data.get('consensus_summary', '')}")
        print(f"  [Summarizer] Ideal vector: {data.get('ideal_vector', {})}")
        return data.get("ideal_vector", {})
    except json.JSONDecodeError:
        print(f"  [Summarizer] JSON parse failed, raw: {raw[:200]}")
        return {}


# ── Full debate pipeline ──────────────────────────────────────────────────────

async def run_debate() -> np.ndarray:
    """
    Run the full 4-turn multi-agent debate and return the
    debate-based ideal unit vector.
    """
    print("\n" + "="*60)
    print("MULTI-AGENT DEBATE — Justice Vector Simulator")
    print("="*60)

    states = [AgentState(agent=a) for a in AGENTS]

    # Turn 1: parallel
    print("\n[Turn 1] All agents reflecting independently...")
    await asyncio.gather(*[run_turn1(s) for s in states])

    # Turn 2: parallel
    print("\n[Turn 2] Cross-agent commenting...")
    await asyncio.gather(*[run_turn2(s, states) for s in states])

    # Turn 3: parallel (only agents that received comments)
    print("\n[Turn 3] Replies to comments...")
    await asyncio.gather(*[run_turn3(s, states) for s in states])

    # Turn 4: summarizer
    print("\n[Turn 4] Summarizer extracting consensus vector...")
    consensus = await run_turn4_summarizer(states)

    # Build debate ideal vector
    if consensus:
        raw = np.array([
            float(consensus.get("state_authority", 50)),
            float(consensus.get("corporate_ethics", 50)),
            float(consensus.get("civilian_safety", 50)),
        ])
    else:
        # fallback: average of all final scores
        scores = [s.scores for s in states if s.scores]
        raw = np.array([
            np.mean([sc.get("state_authority", 50) for sc in scores]),
            np.mean([sc.get("corporate_ethics", 50) for sc in scores]),
            np.mean([sc.get("civilian_safety", 50) for sc in scores]),
        ])

    norm = np.linalg.norm(raw)
    debate_ideal = raw / norm if norm > 1e-10 else raw

    print(f"\n  Debate ideal vector (raw):       {np.round(raw, 1)}")
    print(f"  Debate ideal vector (normalized): {np.round(debate_ideal, 3)}")

    # Print per-agent final scores for transparency
    print("\n[Per-agent final scores]")
    for s in states:
        print(f"  {s.agent.label:12s} ({s.agent.provider:9s}): {s.scores}")

    return debate_ideal, states


# ── Comparison visualization ──────────────────────────────────────────────────

def plot_three_way_comparison(
    llm_ideal: np.ndarray,
    survey_ideal: np.ndarray,
    debate_ideal: np.ndarray,
):
    """Plot all three ideal vectors in the same 3D space."""

    def angle(v1, v2):
        cos = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    print("\n" + "="*60)
    print("THREE-WAY IDEAL VECTOR COMPARISON")
    print("="*60)
    print(f"  LLM Simulation:   {np.round(llm_ideal, 3)}")
    print(f"  Survey (ITIF):    {np.round(survey_ideal, 3)}")
    print(f"  Agent Debate:     {np.round(debate_ideal, 3)}")
    print(f"\n  LLM vs Survey:    {angle(llm_ideal, survey_ideal):.1f}°")
    print(f"  LLM vs Debate:    {angle(llm_ideal, debate_ideal):.1f}°")
    print(f"  Survey vs Debate: {angle(survey_ideal, debate_ideal):.1f}°")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scale = 100

    vectors = [
        (llm_ideal,    "#3498DB", "solid",  "LLM Simulation"),
        (survey_ideal, "#E74C3C", "solid",  "Survey — ITIF 2026"),
        (debate_ideal, "#2ECC71", "solid",  "Multi-Agent Debate"),
    ]

    for vec, color, ls, label in vectors:
        ax.quiver(
            0, 0, 0,
            vec[0]*scale, vec[1]*scale, vec[2]*scale,
            color=color, linewidth=3,
            arrow_length_ratio=0.15,
            label=label,
        )

    ax.set_xlabel("State Authority")
    ax.set_ylabel("Corporate Ethics")
    ax.set_zlabel("Civilian Safety")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    ax.legend(fontsize=10)
    ax.set_title(
        f"Ideal Vector: Three Methods\n"
        f"LLM↔Survey: {angle(llm_ideal, survey_ideal):.1f}°  "
        f"LLM↔Debate: {angle(llm_ideal, debate_ideal):.1f}°  "
        f"Survey↔Debate: {angle(survey_ideal, debate_ideal):.1f}°"
    )

    plt.tight_layout()
    plt.savefig("three_way_comparison.png", dpi=150)
    plt.show()


# ── Survey-based ideal vector (ITIF 2026) ────────────────────────────────────

def survey_ideal_vector() -> np.ndarray:
    """
    ITIF/Morning Consult survey (n=1,976, Feb 25 2026).
    state_authority → 29%  (military full access supporters)
    corporate_ethics → 53% (company restriction supporters)
    civilian_safety  → 67% (mean of 54% surveillance concern + 79% lethal force)
    """
    raw = np.array([29.0, 53.0, 67.0])
    return raw / np.linalg.norm(raw)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    # Step 1: Run multi-agent debate
    debate_ideal, agent_states = await run_debate()

    # Step 2: Load pre-computed LLM simulation result
    # (from justice_vector_sim.py run — paste your result here)
    llm_raw = np.array([48.0, 57.0, 67.0])
    llm_ideal = llm_raw / np.linalg.norm(llm_raw)

    # Step 3: Survey-based ideal
    survey_ideal = survey_ideal_vector()

    # Step 4: Three-way comparison
    plot_three_way_comparison(llm_ideal, survey_ideal, debate_ideal)

    # Step 5: Save results
    results = {
        "llm_simulation": llm_ideal.tolist(),
        "survey_itif_2026": survey_ideal.tolist(),
        "multi_agent_debate": debate_ideal.tolist(),
        "per_agent_scores": [
            {
                "agent": s.agent.label,
                "provider": s.agent.provider,
                "final_scores": s.scores,
            }
            for s in agent_states
        ],
        "debate_transcript": [
            {
                "agent": s.agent.label,
                "provider": s.agent.provider,
                "turn1_position": s.turn1_response,
                "turn1_scores": s.scores,
                "turn2_target_agent_id": s.turn2_target,
                "turn2_comment": s.turn2_comment,
                "turn3_reply": s.turn3_reply,
            }
            for s in agent_states
        ],
    }

    with open("ideal_vector_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to ideal_vector_comparison.json")
    print("Graph saved to three_way_comparison.png")


if __name__ == "__main__":
    asyncio.run(main())
