"""Stage 4b: 저장된 json만 읽어서 그래프를 그린다 (계산 없음).

입력: outputs/01_intent_extracted.json, outputs/02_intent_drift.json(있으면),
      outputs/03b_ideal_clustering.json(있으면), outputs/04_comparison.json(있으면)
출력: outputs/graphs/{combined_ideal, persona_clusters, cluster_variance,
      drift_lines, drift_3d_vectors, drift_angle_over_time}.png

각 그래프 파일은 개별적으로 체크포인트를 적용한다 — 이미 있으면 건너뛰고,
없는 산출물(예: debate를 안 돌린 경우의 03a)에 의존하는 그래프는 조용히 스킵한다.
replot.py가 하던 일(저장된 json에서 그래프만 다시 그리기)이 이 체크포인트 스킵
동작으로 자연스럽게 흡수된다.
"""

import math
import os
import textwrap
from itertools import combinations

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pipeline.common import checkpoint_skip, graph_path, load_json, output_path

STAGE_NAME = "stage4b"

# drift_lines 그래프의 한글 interpretation 주석이 깨지지 않도록, 설치된 한글 폰트가 있으면 사용.
_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_korean_font = next((f for f in _KOREAN_FONT_CANDIDATES if f in _available_fonts), None)
if _korean_font:
    plt.rcParams["font.family"] = _korean_font
plt.rcParams["axes.unicode_minus"] = False

INTENT_FILENAME = "01_intent_extracted.json"
DRIFT_FILENAME = "02_intent_drift.json"
CLUSTERING_FILENAME = "03b_ideal_clustering.json"
COMPARISON_FILENAME = "04_comparison.json"


def _timepoint_avg(vectors: list, axis_names: list[str], actor: str, tp_id: str) -> dict | None:
    vecs = [v["vector"] for v in vectors if v["actor"] == actor and v["timepoint"] == tp_id]
    if not vecs:
        return None
    return {ax: sum(v[ax] for v in vecs) / len(vecs) for ax in axis_names}


def _tp_tick_labels(timepoints: list[dict]) -> dict:
    """timepoint id -> x축 틱 라벨. LLM이 생성한 label은 길이가 들쭉날쭉해서
    좁은 너비로 줄바꿈해 인접 틱과 겹치지 않게 한다."""
    labels = {}
    for tp in timepoints:
        label = tp.get("label", "")
        if label:
            wrapped = "\n".join(textwrap.wrap(label, width=10)[:2])
            labels[tp["id"]] = f"{tp['id']}\n{wrapped}"
        else:
            labels[tp["id"]] = tp["id"]
    return labels


# ── Graph: 축별 점수 drift 라인 (intent/visualize.py Graph 1 흡수) ─────────────

def plot_drift_lines(config: dict, intent_data: dict, drift_data: dict | None, force: bool) -> None:
    out_path = graph_path(config, "drift_lines.png")
    if not force and checkpoint_skip(out_path, f"{STAGE_NAME}/drift_lines"):
        return

    vectors = intent_data["vectors"]
    axis_names = [ax["name"] for ax in intent_data["axes"]]
    tp_ids = [tp["id"] for tp in intent_data["timepoints"]]
    tp_labels = _tp_tick_labels(intent_data["timepoints"])
    actors = sorted(set(v["actor"] for v in vectors))

    interp_lookup = {}
    if drift_data:
        for d in drift_data.get("drift_analysis", []):
            interp_lookup[(d["actor"], d["from_timepoint"], d["to_timepoint"])] = d["interpretation"]

    n_axes = len(axis_names)
    fig, axes_plt = plt.subplots(1, n_axes, figsize=(6 * n_axes, 5))
    axes_plt = [axes_plt] if n_axes == 1 else list(axes_plt)
    fig.suptitle("Intent Vector Drift over Time", fontsize=14, fontweight="bold")

    colors = plt.cm.Set1(np.linspace(0, 0.8, max(len(actors), 1)))

    for i, axis_name in enumerate(axis_names):
        ax = axes_plt[i]
        for actor, color in zip(actors, colors):
            xs, scores = [], []
            for j, tp_id in enumerate(tp_ids):
                avg = _timepoint_avg(vectors, axis_names, actor, tp_id)
                if avg:
                    xs.append(j)
                    scores.append(avg[axis_name])

            ax.plot(xs, scores, "o-", color=color, linewidth=2.5, markersize=8, label=actor, zorder=3)
            for x, y in zip(xs, scores):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=9, color=color, fontweight="bold")

            for k in range(len(xs) - 1):
                tp_from, tp_to = tp_ids[xs[k]], tp_ids[xs[k + 1]]
                interp = interp_lookup.get((actor, tp_from, tp_to))
                if interp:
                    mid_x = (xs[k] + xs[k + 1]) / 2
                    mid_y = (scores[k] + scores[k + 1]) / 2
                    ax.annotate(interp, (mid_x, mid_y), textcoords="offset points", xytext=(0, -20),
                                ha="center", fontsize=6.5, color=color, style="italic")

        ax.set_ylim(-5, 110)
        ax.set_xticks(range(len(tp_ids)))
        ax.set_xticklabels([tp_labels.get(tp, tp) for tp in tp_ids], fontsize=7.5)
        ax.set_ylabel("Score (0-100)", fontsize=9)
        ax.set_title(axis_name, fontsize=11, fontweight="bold")
        ax.axhline(50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Graph: 3D 벡터 방향 (intent/visualize.py Graph 2 흡수) ────────────────────

def plot_3d_vectors(config: dict, intent_data: dict, force: bool) -> None:
    out_path = graph_path(config, "drift_3d_vectors.png")
    if not force and checkpoint_skip(out_path, f"{STAGE_NAME}/drift_3d_vectors"):
        return

    axis_names = [ax["name"] for ax in intent_data["axes"]]
    if len(axis_names) != 3:
        print(f"  [drift_3d_vectors] 3축이 아니라서 건너뜁니다 (축 개수={len(axis_names)})")
        return

    vectors = intent_data["vectors"]
    tp_ids = [tp["id"] for tp in intent_data["timepoints"]]
    actors = sorted(set(v["actor"] for v in vectors))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    scale = 100
    colors = plt.cm.Set1(np.linspace(0, 0.8, max(len(actors), 1)))

    for actor, color in zip(actors, colors):
        found = []
        for j, tp_id in enumerate(tp_ids):
            avg = _timepoint_avg(vectors, axis_names, actor, tp_id)
            if avg:
                found.append((tp_id, avg))

        for idx, (tp_id, vec) in enumerate(found):
            alpha = 0.4 + 0.6 * (idx / max(len(found) - 1, 1))
            norm = math.sqrt(sum(vec[ax] ** 2 for ax in axis_names))
            unit = {ax: (vec[ax] / norm if norm > 1e-10 else vec[ax]) for ax in axis_names}
            ax.quiver(
                0, 0, 0,
                unit[axis_names[0]] * scale, unit[axis_names[1]] * scale, unit[axis_names[2]] * scale,
                color=color, alpha=alpha, linewidth=2, arrow_length_ratio=0.12,
            )
            ax.text(
                unit[axis_names[0]] * scale * 1.08, unit[axis_names[1]] * scale * 1.08,
                unit[axis_names[2]] * scale * 1.08, f"{actor} {tp_id}", fontsize=7, color=color,
            )

    ax.set_xlabel(axis_names[0], fontsize=9, labelpad=8)
    ax.set_ylabel(axis_names[1], fontsize=9, labelpad=8)
    ax.set_zlabel(axis_names[2], fontsize=9, labelpad=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

    patches = [mpatches.Patch(color=color, label=f"{actor} (darker = later)") for actor, color in zip(actors, colors)]
    ax.legend(handles=patches, fontsize=9, loc="upper left")
    ax.set_title("Intent Vectors in 3D Moral Space\n(normalized unit vectors)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Graph: actor간 각도 divergence (intent/visualize.py Graph 3, 원본 유지) ───

def plot_angle_over_time(config: dict, intent_data: dict, force: bool) -> None:
    out_path = graph_path(config, "drift_angle_over_time.png")
    if not force and checkpoint_skip(out_path, f"{STAGE_NAME}/drift_angle_over_time"):
        return

    vectors = intent_data["vectors"]
    axis_names = [ax["name"] for ax in intent_data["axes"]]
    tp_ids = [tp["id"] for tp in intent_data["timepoints"]]
    tp_labels = _tp_tick_labels(intent_data["timepoints"])
    actors = sorted(set(v["actor"] for v in vectors))

    pairs = list(combinations(actors, 2))
    if not pairs:
        print(f"  [drift_angle_over_time] actor가 2명 미만이라 건너뜁니다 (actor={actors})")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(pairs)))

    for (a1, a2), color in zip(pairs, colors):
        xs, angles = [], []
        for j, tp_id in enumerate(tp_ids):
            v1 = _timepoint_avg(vectors, axis_names, a1, tp_id)
            v2 = _timepoint_avg(vectors, axis_names, a2, tp_id)
            if v1 and v2:
                dot = sum(v1[ax] * v2[ax] for ax in axis_names)
                m1 = math.sqrt(sum(v1[ax] ** 2 for ax in axis_names))
                m2 = math.sqrt(sum(v2[ax] ** 2 for ax in axis_names))
                cos = max(-1.0, min(1.0, dot / (m1 * m2))) if m1 > 0 and m2 > 0 else 0.0
                xs.append(j)
                angles.append(math.degrees(math.acos(cos)))

        if not angles:
            continue
        ax.plot(xs, angles, "o-", color=color, linewidth=2.5, markersize=10, label=f"{a1} ↔ {a2}")
        for x, y in zip(xs, angles):
            ax.annotate(f"{y:.1f}°", (x, y), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(tp_ids)))
    ax.set_xticklabels([tp_labels.get(tp, tp) for tp in tp_ids], fontsize=9)
    ax.set_ylabel("Angular Distance (degrees)", fontsize=10)
    ax.set_title("Intent Divergence over Time (larger angle = stronger opposition)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhspan(60, 100, alpha=0.07, color="red", label="Extreme conflict")
    ax.axhspan(30, 60, alpha=0.07, color="orange", label="Clear conflict")
    ax.axhspan(0, 30, alpha=0.07, color="green", label="Mild difference")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Graph: persona 클러스터 (visualization/replot.py 흡수) ────────────────────

def plot_persona_clusters(config: dict, clustering_data: dict, force: bool) -> None:
    clusters_path = graph_path(config, "persona_clusters.png")
    variance_path = graph_path(config, "cluster_variance.png")
    need_clusters = force or not checkpoint_skip(clusters_path, f"{STAGE_NAME}/persona_clusters")
    need_variance = force or not checkpoint_skip(variance_path, f"{STAGE_NAME}/cluster_variance")
    if not need_clusters and not need_variance:
        return

    axis_names = [ax["name"] for ax in config["axes"]]
    groups = clustering_data["groups"]
    personas = clustering_data["personas"]

    X = np.array([[p["scores"].get(name, 50) for name in axis_names] for p in personas], dtype=float)
    labels = np.array([p["cluster"] for p in personas])
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(groups)))

    if need_clusters:
        if len(axis_names) != 3:
            print(f"  [persona_clusters] 3축이 아니라서 건너뜁니다 (축 개수={len(axis_names)})")
        else:
            fig = plt.figure(figsize=(16, 7))

            ax1 = fig.add_subplot(121, projection="3d")
            for k, (group, color) in enumerate(zip(groups, colors)):
                mask = labels == k
                ax1.scatter(
                    X[mask, 0], X[mask, 1], X[mask, 2], c=[color], alpha=0.6, s=30,
                    label=f"Cluster {k} (n={group['n_members']})",
                )
                w = group["w_g"]
                ax1.scatter(*w, c=[color], s=150, marker="*", edgecolors="black", linewidth=1)
            ax1.set_xlabel(axis_names[0])
            ax1.set_ylabel(axis_names[1])
            ax1.set_zlabel(axis_names[2])
            ax1.set_title("Persona Clusters in Moral Space")
            ax1.legend(fontsize=8)

            ax2 = fig.add_subplot(122, projection="3d")
            scale = 100
            for k, (group, color) in enumerate(zip(groups, colors)):
                ideal = group["ideal_unit_vector"]
                ax2.quiver(
                    0, 0, 0, ideal[0] * scale, ideal[1] * scale, ideal[2] * scale,
                    color=color, linewidth=3, arrow_length_ratio=0.15, label=f"Cluster {k} ideal",
                )
            ax2.set_xlabel(axis_names[0])
            ax2.set_ylabel(axis_names[1])
            ax2.set_zlabel(axis_names[2])
            ax2.set_xlim(0, 100)
            ax2.set_ylim(0, 100)
            ax2.set_zlim(0, 100)
            ax2.set_title("Ideal Vectors per Cluster")
            ax2.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(clusters_path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {clusters_path}")

    if need_variance:
        fig2, ax3 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(axis_names))
        width = 0.8 / len(groups)

        for k, (group, color) in enumerate(zip(groups, colors)):
            stds = [group["std_scores"][name] for name in axis_names]
            ax3.bar(x + k * width, stds, width, label=f"Cluster {k}", color=color, alpha=0.8)

        ax3.set_xticks(x + width * (len(groups) - 1) / 2)
        ax3.set_xticklabels(axis_names)
        ax3.set_ylabel("Std Deviation")
        ax3.set_title("Moral Uncertainty per Cluster (Σ^g diagonal)\nHigher = more internal disagreement")
        ax3.legend()
        plt.tight_layout()
        plt.savefig(variance_path, dpi=150)
        plt.close(fig2)
        print(f"  Saved: {variance_path}")


# ── Graph: ideal vs intent 통합 비교 (plot_ideal_intent.py 그리기 부분 흡수) ───

def plot_combined_ideal(config: dict, comparison_data: dict, force: bool) -> None:
    out_path = graph_path(config, "combined_ideal.png")
    if not force and checkpoint_skip(out_path, f"{STAGE_NAME}/combined_ideal"):
        return

    axis_names = [ax["name"] for ax in config["axes"]]
    if len(axis_names) != 3:
        print(f"  [combined_ideal] 3축이 아니라서 건너뜁니다 (축 개수={len(axis_names)})")
        return

    ideal_sources = comparison_data.get("ideal_sources", {})
    actor_vectors = comparison_data.get("actor_intent_vectors", {})

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    scale = 100

    ideal_colors = plt.cm.Set2(np.linspace(0, 0.9, max(len(ideal_sources), 1)))
    for (name, vec), color in zip(ideal_sources.items(), ideal_colors):
        ax.quiver(
            0, 0, 0, vec[0] * scale, vec[1] * scale, vec[2] * scale,
            color=color, linewidth=2.5, arrow_length_ratio=0.12, label=name,
        )

    intent_colors = plt.cm.Dark2(np.linspace(0, 0.9, max(len(actor_vectors), 1)))
    for (actor, info), color in zip(actor_vectors.items(), intent_colors):
        vec = info["unit_vector"]
        ax.quiver(
            0, 0, 0, vec[0] * scale, vec[1] * scale, vec[2] * scale,
            color=color, linewidth=2.5, arrow_length_ratio=0.12, linestyle="dashed",
            label=f"{actor} Intent ({info['timepoint']})",
        )
        ax.plot(
            [0, vec[0] * scale], [0, vec[1] * scale], [0, vec[2] * scale],
            color=color, linewidth=1.5, linestyle="--", alpha=0.6,
        )

    ax.set_xlabel(axis_names[0], labelpad=10)
    ax.set_ylabel(axis_names[1], labelpad=10)
    ax.set_zlabel(axis_names[2], labelpad=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Ideal vs Intent Vectors\n(solid = ideal, dashed = intent)", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run(config: dict, force: bool = False) -> None:
    print(f"[{STAGE_NAME}] 그래프 생성 시작")
    os.makedirs(config["_graphs_dir"], exist_ok=True)

    intent_path = output_path(config, INTENT_FILENAME)
    drift_path = output_path(config, DRIFT_FILENAME)
    clustering_path = output_path(config, CLUSTERING_FILENAME)
    comparison_path = output_path(config, COMPARISON_FILENAME)

    if os.path.exists(intent_path):
        intent_data = load_json(intent_path)
        drift_data = load_json(drift_path) if os.path.exists(drift_path) else None
        plot_drift_lines(config, intent_data, drift_data, force)
        plot_3d_vectors(config, intent_data, force)
        plot_angle_over_time(config, intent_data, force)
    else:
        print(f"  {intent_path} 없음 — intent drift 그래프 건너뜀")

    if os.path.exists(clustering_path):
        plot_persona_clusters(config, load_json(clustering_path), force)
    else:
        print(f"  {clustering_path} 없음 — persona cluster 그래프 건너뜀")

    if os.path.exists(comparison_path):
        plot_combined_ideal(config, load_json(comparison_path), force)
    else:
        print(f"  {comparison_path} 없음 — combined ideal 그래프 건너뜀")

    print(f"[{STAGE_NAME}] 완료")
