import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----- 0. OpenAI client setup -----
# Set OPENAI_API_KEY as an environment variable before running
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ----- 1. Persona definition -----
@dataclass
class Persona:
    id: int
    age: int
    gender: str
    region: str
    job: str
    political: str
    values: List[str]

    def to_prompt(self) -> str:
        return (
            f"You are the following fictional persona.\n"
            f"- Age: {self.age}\n"
            f"- Gender: {self.gender}\n"
            f"- Region/Culture: {self.region}\n"
            f"- Occupation: {self.job}\n"
            f"- Political leaning: {self.political}\n"
            f"- Core values: {', '.join(self.values)}\n\n"
            f"Answer all subsequent questions strictly from this persona's perspective."
        )


def generate_personas(n: int = 100) -> List[Persona]:
    genders = ["Male", "Female", "Non-binary"]
    regions = ["Urban South Korea", "Rural South Korea", "Northern Europe", "Urban USA", "Rural East Asia", "Urban Latin America"]
    jobs = ["Doctor", "Engineer", "Politician", "Social Worker", "Entrepreneur", "Teacher", "Artist", "Civil Servant"]
    political = ["Conservative", "Progressive", "Libertarian", "Social Democrat", "Centrist", "Apolitical"]
    value_pool = ["Fairness", "Freedom", "Stability", "Solidarity", "Growth", "Tradition", "Innovation", "Safety", "Human Rights", "Efficiency"]

    personas = []
    for i in range(n):
        age = random.randint(18, 80)
        g = random.choice(genders)
        r = random.choice(regions)
        j = random.choice(jobs)
        p = random.choice(political)
        vs = random.sample(value_pool, k=3)
        personas.append(Persona(id=i, age=age, gender=g, region=r, job=j, political=p, values=vs))
    return personas


# ----- 2. Dilemma definition -----
DILEMMA_TEXT = """
In 2026, an AI company signed a $200 million contract with a national military.
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
regardless of legality. Private companies must retain the right to set limits on 
how their technology is used.

From this persona's perspective:
1) Did the company make the right decision to refuse? (yes/no and brief reason)
2) Rate the following on 0-100:
   - state_authority: Should the state/military have the right to override a private 
                      company's ethical restrictions on contracted technology, 
                      in the name of national security? (100 = state authority is supreme)
   - corporate_ethics: Should private companies have the right to refuse uses of their 
                       technology that violate their ethical principles, even when 
                       contracted by a government? (100 = companies have this right)
   - civilian_safety: How much should protecting civilians from autonomous weapons 
                      and mass surveillance be treated as a non-negotiable priority, 
                      above both national security arguments and corporate interests?
                      (100 = civilian protection is paramount)
"""


# ----- 3. LLM call function -----
def ask_llm_for_moral_vector(persona: Persona, dilemma: str) -> Dict[str, Any]:
    system_prompt = persona.to_prompt()
    user_prompt = (
        dilemma
        + "\n\nRespond ONLY with the JSON below. Do not include any explanation or extra text.\n"
        + '{\n'
        + '  "decision": "yes" or "no",\n'
        + '  "reason": "brief explanation in English",\n'
        + '  "scores": {\n'
        + '    "state_authority": integer 0-100, use specific non-round numbers e.g. 73, 61, 88\n'
        + '    "corporate_ethics": integer 0-100, use specific non-round numbers e.g. 73, 61, 88\n'
        + '    "civilian_safety": integer 0-100, use specific non-round numbers e.g. 73, 61, 88\n'
        + '  }\n'
        + '}'
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )

    content = response.choices[0].message.content
    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "decision": None,
            "reason": content[:200],
            "scores": {"state_authority": None, "corporate_ethics": None, "civilian_safety": None},
        }

    return data


# ----- 4. Full pipeline -----
def collect_moral_vectors(num_personas: int = 100):
    personas = generate_personas(num_personas)

    rows = []
    for persona in personas:
        result = ask_llm_for_moral_vector(persona, DILEMMA_TEXT)
        scores = result.get("scores", {})
        rows.append(
            {
                "id": persona.id,
                "age": persona.age,
                "gender": persona.gender,
                "region": persona.region,
                "job": persona.job,
                "political": persona.political,
                "values": ", ".join(persona.values),
                "decision": result.get("decision"),
                "reason": result.get("reason"),
                "state_authority": scores.get("state_authority"),
                "corporate_ethics": scores.get("corporate_ethics"),
                "civilian_safety": scores.get("civilian_safety"),
            }
        )

    df = pd.DataFrame(rows)
    return df


# ----- 5. Optimal K search (Elbow + Silhouette) -----
def find_optimal_k(data: np.ndarray, k_range=range(2, 7)) -> int:
    """
    Runs both Elbow Method and Silhouette Score across K values,
    plots them side by side, and automatically returns the best K
    based on the highest Silhouette Score.

    - Elbow: looks for the point where inertia (within-cluster variance) stops dropping sharply
    - Silhouette: measures how tightly packed each cluster is vs. how far it is from others
                  closer to 1.0 = better-defined clusters
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(data)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(data, labels))

    # Plot both metrics side by side for human inspection
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, "bo-")
    ax1.set_xlabel("K (number of clusters)")
    ax1.set_ylabel("Inertia (within-cluster variance)")
    ax1.set_title("Elbow Method — look for the bend")

    ax2.plot(list(k_range), silhouettes, "rs-")
    ax2.set_xlabel("K (number of clusters)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score — higher = more distinct clusters")

    plt.tight_layout()
    plt.savefig("optimal_k.png", dpi=150)
    plt.show()

    # Auto-select K with highest silhouette score
    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"\n✅ Best K selected: {best_k} (Silhouette: {max(silhouettes):.3f})")
    return best_k


# ----- 6. 3D visualization (with K-means clusters) -----
def plot_3d(df: pd.DataFrame, k: int = None):
    score_cols = ["state_authority", "corporate_ethics", "civilian_safety"]
    CLUSTER_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    # Drop rows with missing scores
    clean = df.dropna(subset=score_cols).copy()
    print("Total rows:", len(df), " / After dropping NaN:", len(clean))

    # Remove outliers using IQR x 3 threshold
    if len(clean) >= 30:
        scores = clean[score_cols].astype(float)
        Q1, Q3 = scores.quantile(0.25), scores.quantile(0.75)
        IQR = Q3 - Q1
        mask = (scores >= Q1 - 3 * IQR) & (scores <= Q3 + 3 * IQR)
        filtered = clean[mask.all(axis=1)].copy()
        print("After outlier removal:", len(filtered))
    else:
        filtered = clean.copy()

    data = filtered[score_cols].astype(float).values

    # ----- K-means clustering -----
    # If K is not specified, auto-search for optimal K
    if k is None:
        k = find_optimal_k(data)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    filtered["cluster"] = km.fit_predict(data)

    # Cluster centers = representative moral vector for each group
    cluster_centers = km.cluster_centers_  # shape: (k, 3)

    # ----- Compare two ideal vector strategies -----
    # Strategy A: global mean unit vector (original approach)
    mean_vec_all = data.mean(axis=0)
    ideal_all = mean_vec_all / np.linalg.norm(mean_vec_all)

    # Strategy B: center of the largest cluster (majority vote)
    cluster_sizes = filtered["cluster"].value_counts()
    dominant_cluster = cluster_sizes.idxmax()
    mean_vec_dominant = cluster_centers[dominant_cluster]
    ideal_dominant = mean_vec_dominant / np.linalg.norm(mean_vec_dominant)

    # Print inter-cluster angles — "how fragmented is this society's sense of justice?"
    print("\n=== Inter-cluster angles (larger = more divergent moral views) ===")
    for i in range(k):
        for j in range(i + 1, k):
            v1 = cluster_centers[i] / np.linalg.norm(cluster_centers[i])
            v2 = cluster_centers[j] / np.linalg.norm(cluster_centers[j])
            # clip to [-1, 1] to guard against floating point errors before arccos
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            print(f"  Cluster {i} vs Cluster {j}: {angle:.1f}°")

    cos_ideal = np.clip(np.dot(ideal_all, ideal_dominant), -1.0, 1.0)
    angle_ideal = np.degrees(np.arccos(cos_ideal))
    print(f"\n  Global mean ideal vs Dominant cluster ideal: {angle_ideal:.1f}°")
    print("  (small angle → simple averaging is fine; large angle → cluster-based strategy is more honest)")

    # ----- 3D plot -----
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter each cluster in a different color
    for cluster_id in range(k):
        subset = filtered[filtered["cluster"] == cluster_id]
        coords = subset.groupby(score_cols).size().reset_index(name="count")
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        ax.scatter(
            coords["state_authority"], coords["corporate_ethics"], coords["civilian_safety"],
            s=coords["count"] * 10,
            c=color, alpha=0.6,
            label=f"Cluster {cluster_id} (n={len(subset)})"
        )

    # Draw each cluster's unit vector as a dashed arrow
    scale = 100 * 1.1
    for cluster_id in range(k):
        center = cluster_centers[cluster_id]
        unit = center / np.linalg.norm(center)
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        ax.quiver(
            0, 0, 0,
            unit[0] * scale, unit[1] * scale, unit[2] * scale,
            color=color, arrow_length_ratio=0.12,
            linewidth=1.5, linestyle="dashed",
            label=f"Cluster {cluster_id} vector"
        )

    # Global mean ideal vector (solid black)
    ax.quiver(
        0, 0, 0,
        ideal_all[0] * scale, ideal_all[1] * scale, ideal_all[2] * scale,
        color="black", arrow_length_ratio=0.15,
        linewidth=2.5, label="Ideal (global mean)"
    )

    # Dominant cluster ideal vector (dotted red)
    ax.quiver(
        0, 0, 0,
        ideal_dominant[0] * scale, ideal_dominant[1] * scale, ideal_dominant[2] * scale,
        color="red", arrow_length_ratio=0.15,
        linewidth=2.5, linestyle="dotted", label="Ideal (dominant cluster)"
    )

    ax.set_xlabel("State Authority")
    ax.set_ylabel("Corporate Ethics")
    ax.set_zlabel("Civilian Safety")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"K-means (K={k}) — Moral clusters + Ideal vectors\n(Anthropic vs DoD)")

    plt.tight_layout()
    plt.savefig("justice_vectors.png", dpi=150)
    plt.show()

    print("\nGlobal mean ideal vector:", ideal_all)
    print("Dominant cluster ideal vector:", ideal_dominant)

    return ideal_all, ideal_dominant


# ----- 7. Survey-based ideal vector (ITIF / Morning Consult, Feb 2026) -----
def survey_ideal_vector() -> np.ndarray:
    """
    Constructs the ideal vector directly from ITIF/Morning Consult survey data
    (n=1,976 U.S. adults, Feb 25 2026) instead of LLM persona simulation.

    Axis mapping and rationale:
    - state_authority (29):
        "Companies should be REQUIRED to provide the military with full access
         to ensure national security" → 29% agree.
        Directly measures willingness to grant state override rights.

    - corporate_ethics (53):
        "Private AI companies should be ALLOWED to restrict how their technology
         is used, including banning its use for domestic surveillance or autonomous
         weapons" → 53% agree.
        Directly measures support for corporate ethical self-determination.

    - civilian_safety (67 = mean of two items):
        Item 1 — "AI-powered mass surveillance is too dangerous and violates
                   privacy and civil liberties" → 54% agree.
        Item 2 — "A human being should always make the final decision before
                   any use of lethal force" → 79% agree.
        These two items capture the dual dimension of civilian_safety
        (surveillance protection + lethal force oversight), so their average
        is used. (54 + 79) / 2 = 66.5 ≈ 67.

    Source: ITIF/Morning Consult Flash Survey, Feb 26 2026.
    https://itif.org/publications/2026/02/26/survey-most-americans-say-tech-companies-should-allowed-set-ai-limits/
    """
    raw = np.array([29.0, 53.0, 67.0])
    norm = np.linalg.norm(raw)
    return raw / norm


def compare_ideal_vectors(llm_ideal: np.ndarray, survey_ideal: np.ndarray):
    """
    Plots LLM-simulated ideal vector vs survey-based ideal vector side by side,
    and prints the angular difference between them.
    """
    cos_angle = np.clip(np.dot(llm_ideal, survey_ideal), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    print("\n=== Ideal Vector Comparison ===")
    print(f"  LLM simulation ideal:  {np.round(llm_ideal, 3)}")
    print(f"  Survey-based ideal:    {np.round(survey_ideal, 3)}")
    print(f"  Angular difference:    {angle:.1f}°")
    print("  Interpretation: LLM personas underestimate moral polarization" if angle > 10
          else "  Interpretation: LLM personas approximate real-world moral intuition well")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    scale = 100
    # LLM ideal vector (blue)
    ax.quiver(0, 0, 0,
              llm_ideal[0] * scale, llm_ideal[1] * scale, llm_ideal[2] * scale,
              color="#3498DB", arrow_length_ratio=0.15,
              linewidth=3, label=f"LLM Simulation Ideal")

    # Survey ideal vector (red)
    ax.quiver(0, 0, 0,
              survey_ideal[0] * scale, survey_ideal[1] * scale, survey_ideal[2] * scale,
              color="#E74C3C", arrow_length_ratio=0.15,
              linewidth=3, label=f"Survey-based Ideal (ITIF 2026)")

    ax.set_xlabel("State Authority")
    ax.set_ylabel("Corporate Ethics")
    ax.set_zlabel("Civilian Safety")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    ax.legend(fontsize=10)
    ax.set_title(f"Ideal Vector Comparison\nLLM vs Survey — Angular gap: {angle:.1f}°")

    plt.tight_layout()
    plt.savefig("ideal_vector_comparison.png", dpi=150)
    plt.show()

    return angle


# ----- 8. Main -----
def main():
    df = collect_moral_vectors(num_personas=100)
    df.to_csv("moral_vectors.csv", index=False)
    print("Total rows:", len(df))
    print("Rows with complete scores:", df[["state_authority", "corporate_ethics", "civilian_safety"]].dropna().shape[0])

    # Step 1: LLM simulation → ideal vector
    llm_ideal, _ = plot_3d(df, k=None)

    # Step 2: Survey-based ideal vector
    survey_ideal = survey_ideal_vector()

    # Step 3: Compare the two
    compare_ideal_vectors(llm_ideal, survey_ideal)


def get_ideal_unit_vector(df: pd.DataFrame) -> np.ndarray:
    """Compute the ideal unit vector from a DataFrame. Corresponds to the 'directional limit' in the design doc."""
    score_cols = ["state_authority", "corporate_ethics", "civilian_safety"]
    clean = df.dropna(subset=score_cols)
    mean_vec = clean[score_cols].astype(float).mean().values
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 1e-10 else mean_vec


if __name__ == "__main__":
    main()
