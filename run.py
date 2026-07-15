"""CLI 진입점: 케이스별 파이프라인 실행.

Usage:
  python run.py --case dod_anthropic
  python run.py --case dod_anthropic --ideal-methods debate
  python run.py --case dod_anthropic --force stage3b
  python run.py --case 새로운_사례 --force-all
"""

import argparse

from pipeline import (
    stage1_extract_intent,
    stage2_analyze_intent,
    stage3a_ideal_debate,
    stage3b_ideal_clustering,
    stage4a_compare,
    stage4b_visualize,
)
from pipeline.common import load_case_config

STAGE_NAMES = ["stage1", "stage2", "stage3a", "stage3b", "stage4a", "stage4b"]
IDEAL_METHOD_NAMES = {"debate", "clustering"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI's Topography of Justice — 파이프라인 실행")
    parser.add_argument("--case", required=True, help="cases/<이름>/config.json 의 <이름>")
    parser.add_argument(
        "--ideal-methods",
        default=None,
        help="쉼표로 구분해 이번 실행에서 켤 ideal method만 지정 (예: debate 또는 debate,clustering). "
             "지정하지 않으면 config.json의 설정을 그대로 사용",
    )
    parser.add_argument(
        "--force",
        choices=STAGE_NAMES,
        default=None,
        help="지정한 stage만 체크포인트를 무시하고 재실행",
    )
    parser.add_argument("--force-all", action="store_true", help="모든 stage를 체크포인트 무시하고 재실행")
    return parser.parse_args()


def apply_ideal_methods_override(config: dict, raw: str | None) -> None:
    if raw is None:
        return

    selected = {m.strip() for m in raw.split(",") if m.strip()}
    invalid = selected - IDEAL_METHOD_NAMES
    if invalid:
        raise SystemExit(f"--ideal-methods에 알 수 없는 값: {sorted(invalid)} (가능한 값: debate, clustering)")

    methods = config.setdefault("ideal_methods", {})
    for name in IDEAL_METHOD_NAMES:
        methods.setdefault(name, {})["enabled"] = name in selected


def should_force(stage_name: str, args: argparse.Namespace) -> bool:
    return args.force_all or args.force == stage_name


def main() -> None:
    args = parse_args()
    config = load_case_config(args.case)
    apply_ideal_methods_override(config, args.ideal_methods)

    stage1_extract_intent.run(config, force=should_force("stage1", args))
    stage2_analyze_intent.run(config, force=should_force("stage2", args))

    if config.get("ideal_methods", {}).get("clustering", {}).get("enabled", True):
        stage3b_ideal_clustering.run(config, force=should_force("stage3b", args))
    else:
        print("[stage3b] ideal_methods.clustering.enabled=false — 건너뜀")

    if config.get("ideal_methods", {}).get("debate", {}).get("enabled", True):
        stage3a_ideal_debate.run(config, force=should_force("stage3a", args))
    else:
        print("[stage3a] ideal_methods.debate.enabled=false — 건너뜀")

    stage4a_compare.run(config, force=should_force("stage4a", args))
    stage4b_visualize.run(config, force=should_force("stage4b", args))


if __name__ == "__main__":
    main()
