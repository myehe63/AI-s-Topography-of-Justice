# Justice Vector Simulator
> justice as a vector — experiment notes

---

## 1. Core Concept

This project treats justice not as a fixed answer but as a convergence process — representing intentions and outcomes as vectors in a vector space, and using that framework to geometrically diagnose real-world cases.

| Vector | Description |
|---|---|
| **Ideal Vector** | A reference direction derived from collective moral intuition. Not a destination, but a unit vector used as a baseline for comparison. |
| **Intent Vector** | The goals an actor officially claims. Extracted from public statements and policy documents. |
| **Outcome Vector** | What actually happened. The real-world result plotted in the same moral space. |

| Diagnosis | Condition | Interpretation |
|---|---|---|
| Practical Limitation | Intent ≈ Ideal, Outcome is far | The direction was right, but execution fell short. |
| Intent Divergence | Large angle between Intent and Ideal | The stated goal itself diverges from the reference direction. |
| Outcome Deviation | Intent ≈ Ideal, Outcome outside trajectory | Stated goals and actual results point in different directions. |

---

## 2. Case Study: Anthropic vs. U.S. Department of Defense (2026)

To apply this framework to a traceable case, I picked the Anthropic–DoD conflict — a dispute over AI ethical restrictions in a government contract — where both sides' official positions and outcomes are available through public records.

- https://edition.cnn.com/2026/03/26/business/anthropic-pentagon-injunction-supply-chain-risk

| Moral Axis | Definition | High score means |
|---|---|---|
| `state_authority` | Does the state/military have the right to override a company's ethical restrictions on contracted technology? | Supports DoD logic |
| `corporate_ethics` | Does a private company have the right to refuse uses of its technology that violate its ethical principles? | Supports Anthropic logic |
| `civilian_safety` | Is protecting civilians from autonomous weapons and mass surveillance a priority? | Civilian protection first |

---

## 3. Ideal Vector Calculation: Three Methods Compared

| Method | Result Vector | Notes |
|---|---|---|
| ① LLM Persona Simulation | [0.479, 0.569, 0.669] | Tends to overestimate state_authority. Doesn't capture emotional polarization well. |
| ② Real-world Survey (ITIF 2026) | [0.321, 0.587, 0.743] | Reflects actual human opinion. Results depend on question design. |
| ③ Multi-Agent Debate (GPT+Claude+Gemini) | [0.188, 0.676, 0.713] | Closest to the survey result among the three. Model-specific biases surface during discussion. |

| Comparison | Angular Difference | Interpretation |
|---|---|---|
| LLM ↔ Survey | 10.0° | LLM structurally overestimates state_authority |
| LLM ↔ Debate | 13.9° | Debate most strongly reduces the LLM gap |
| Survey ↔ Debate | 6.2° | Debate result is closest to the survey |

![Ideal Vector Comparison](figure1.png)

*Ideal vectors from three methods. The blue vector (LLM) is skewed toward the state_authority axis, while red (Survey) and green (Multi-Agent Debate) sit close together.*

---

## 4. Multi-Agent Debate — Experiment Results

Six agents total — two each of GPT-4.1, Claude Sonnet, and Gemini 2.5 — debated the dilemma over 4 rounds.

### Final Scores by Model

| Agent | State Authority | Corporate Ethics | Civilian Safety | Notes |
|---|---|---|---|---|
| Claude-1 (Anthropic) | 20 | 90 | 95 | Most strongly argues against the DoD position |
| Claude-2 (Anthropic) | 20 | 85 | 90 | Frames DoD retaliation as overreach |
| GPT-2 (OpenAI) | 30 | 90 | 90 | state_authority dropped during debate (40→30) |
| GPT-1 (OpenAI) | 40 | 85 | 90 | Maintains a middle position throughout |
| Gemini-1 (Google) | 40 | 90 | 95 | Maintains a middle position throughout |
| Gemini-2 (Google) | 45 | 90 | 98 | civilian_safety rose during debate (90→98) |

### Notable Statements

**Claude-1 (Turn 3):**
> "The military's retaliatory designation itself validates the need for ethical resistance — it's a self-fulfilling proof of authoritarian overreach."

**Claude-2 (Turn 3):**
> "The circular logic here (company restricts military use → military labels company a security threat) validates the original ethical concerns."

**Gemini-2 (Turn 3):**
> "Ethical considerations concerning AI's catastrophic potential must sometimes supersede even state authority for the greater good."

### Observation: Model-Specific Bias in Debate

The two Claude agents reinforced each other's positions noticeably. This likely reflects RLHF training values surfacing in debate — rather than emergent polarization per se.

GPT and Gemini held relatively stable positions. Gemini-2 was an exception: its civilian_safety score rose independently during discussion, which suggests it wasn't simply following the group.

---

## 5. Observations So Far

**1. Dilemma type affects cluster structure**

- Trolley problem: K=6 clusters, max angle 5.3° — relatively broad agreement across participants
- Anthropic/DoD case: K=2 clusters, angle 10.7–11.9° — a clear split in moral orientation

**2. Axis specificity determines sensitivity**

- General axes (fairness/survival/autonomy): 2.2° — below the measurable threshold
- Domain-specific axes (state_authority/corporate_ethics/civilian_safety): 11.9° — conflict visible

**3. LLM personas compress moral variance**

RLHF fine-tuning pulls all models toward median values, reducing the spread of opinion that actually exists in human populations. This is a methodological characteristic, not a failure.

**4. Multi-agent debate gets closer to survey results**

Debate results were closer to the ITIF survey than standalone LLM simulation (6.2° vs 10.0° gap). Mutual rebuttal seems to partially offset RLHF median compression.

**5. Model-specific biases are visible across companies**

Same dilemma, different baselines: Claude scored state_authority at 20, GPT at 30–40, Gemini at 40–45. This seems to reflect differences in training approach across organizations. Adding Grok would be a useful test of whether the pattern holds in the opposite direction.

---

## 6. Next Steps

**Phase 1: Intent and Outcome Vector Extraction**
- [ ] Extract intent vectors from DoD official statements, court filings, and Hegseth's remarks
- [ ] Extract intent vectors from Anthropic's statements and court filings
- [ ] Extract outcome vectors from contract termination, court rulings, and operational use during Iran operations
- [ ] Calculate intent–ideal angle (divergence vs. limitation) and outcome–ideal distance (justice gap)

**Phase 2: Expand Debate Agents**
- [ ] Add Grok (xAI) — contracted with DoD shortly after Anthropic's contract ended. Expected to skew toward state_authority.
- [ ] Add DeepSeek — introduces a non-Western institutional perspective
