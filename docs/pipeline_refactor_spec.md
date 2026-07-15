# AI's Topography of Justice — 파이프라인 리팩토링 스펙

## 1. 목표

케이스(사건) 전용으로 짜여 있던 코드를 **재사용 가능한 파이프라인 + 케이스별 설정 파일** 구조로 바꾼다.
새로운 사례를 돌릴 때는 `pipeline/` 안의 코드를 건드리지 않고 `cases/<새사례>/config.json`만 추가하면 되도록 만든다.

**이번에 함께 제거할 것:**
- 단순 평균 기반 페르소나 시뮬레이션(v1) — GMM 클러스터링(v2)으로 대체되었으므로 삭제
- 모든 하드코딩된 절대경로 (예: `/Users/parkjeongseo/Desktop/ideal/`)
- 사건 배경, 축(axes) 정의, 서베이 수치 등 코드에 박혀 있던 "이 사례만의 값"

---

## 2. 최종 폴더 구조

```
cases/
  dod_anthropic/
    config.json
    raw_sources.json           ← 원자료 (발언 텍스트 등)
    outputs/
      01_intent_extracted.json
      02_intent_drift.json
      03a_ideal_debate.json
      03b_ideal_clustering.json
      04_comparison.json
      graphs/
        combined_ideal.png
        persona_clusters.png
        cluster_variance.png
        drift_lines.png
        drift_3d_vectors.png
        drift_angle_over_time.png

  <새로운_사례>/
    config.json
    raw_sources.json
    outputs/

pipeline/
  common.py                    ← 공통 함수 (각도 계산, LLM 호출, json 입출력)
  stage1_extract_intent.py     ← 실제 발언 → 벡터화
  stage2_analyze_intent.py     ← intent 내부 일관성/drift 분석
  stage3a_ideal_debate.py      ← 멀티에이전트 토론 → ideal 벡터
  stage3b_ideal_clustering.py  ← 페르소나 생성 + GMM 클러스터링 → ideal 벡터
  stage4a_compare.py           ← intent vs 각 ideal 소스 간 각도 계산
  stage4b_visualize.py         ← 그래프 생성 (계산 없이 저장된 json만 읽음)

run.py                         ← CLI 진입점
```

---

## 3. Stage별 입출력 계약

| Stage | 입력 | 출력 | 기존 파일 대응 |
|---|---|---|---|
| `stage1_extract_intent` | `config.json` (case_context, axes), `raw_sources.json` | `01_intent_extracted.json` | `intent/extractor.py` |
| `stage2_analyze_intent` | `01_intent_extracted.json` | `02_intent_drift.json` | `intent/analyzer.py` |
| `stage3a_ideal_debate` | `config.json` (case_context, axes, debate 설정) | `03a_ideal_debate.json` | `Ideal/debate/debate.py` |
| `stage3b_ideal_clustering` | `config.json` (n_personas, k_range 등) | `03b_ideal_clustering.json` | `ideal/clustering/gmm_cluster.py` |
| `stage4a_compare` | `01_intent_extracted.json`, `03a_ideal_debate.json`, `03b_ideal_clustering.json`, `config.survey_ideal_vector` | `04_comparison.json` (pairwise 각도) | `visualization/plot_ideal_intent.py` (계산 부분) |
| `stage4b_visualize` | `04_comparison.json`, `02_intent_drift.json` + 위 3개 json | `graphs/*.png` (combined_ideal, persona_clusters, cluster_variance, drift_lines, drift_3d_vectors, drift_angle_over_time) | `visualization/plot_ideal_intent.py`(그리기 부분), `visualization/replot.py`, `intent/visualize.py` (drift 그래프 3종 흡수) |

각 stage는 **자신의 출력 파일이 이미 존재하면 재계산하지 않고 종료**한다 (체크포인트).

---

## 4. `config.json` 스키마

```json
{
  "case_id": "dod_anthropic",
  "case_context": "사건 배경 설명 (기존 case_context 텍스트 그대로)",
  "axes": [
    {
      "id": "axis_1",
      "name": "state_authority",
      "description": "...",
      "low_label": "...",
      "high_label": "..."
    }
  ],
  "sources_path": "raw_sources.json",
  "survey_ideal_vector": [0.321, 0.587, 0.743],
  "ideal_methods": {
    "debate": {
      "enabled": true,
      "models": ["gpt-4.1", "claude-sonnet-4-6", "gemini-2.5"],
      "agents_per_model": 2,
      "n_rounds": 4
    },
    "clustering": {
      "enabled": true,
      "n_personas": 200,
      "k_range": [2, 6],
      "countries": ["South Korea", "Nigeria", "..."]
    }
  },
  "llm": {
    "model": "claude-sonnet-4-6",
    "max_tokens": 1000
  }
}
```

`survey_ideal_vector`는 외부 설문(ITIF 등) 결과가 있을 때만 채우고, 없는 사례는 생략 가능 (그러면 stage4에서 서베이 비교는 자동으로 건너뜀).

---

## 5. `run.py` 동작 방식

```
python run.py --case dod_anthropic
python run.py --case dod_anthropic --ideal-methods debate
python run.py --case dod_anthropic --force stage3b
python run.py --case 새로운_사례 --force-all
```

- `--case`: `cases/<이름>/config.json`을 읽어서 파이프라인 실행 (필수)
- `--ideal-methods`: config의 `ideal_methods.*.enabled`를 일시적으로 덮어씀 (예: 이번엔 debate 생략하고 clustering만)
- `--force <stage>`: 해당 stage는 체크포인트 무시하고 재실행
- `--force-all`: 전체 재실행
- 각 stage 실행 전, 해당 출력 파일이 `outputs/`에 있는지 확인 → 있으면 `"[stage1] 이미 존재, 건너뜀"` 출력 후 스킵 → 없으면 실행 후 즉시 저장

---

## 6. 마이그레이션 매핑 (기존 → 신규)

| 기존 파일 | 신규 파일 | 주요 변경 사항 |
|---|---|---|
| `intent/extractor.py` | `pipeline/stage1_extract_intent.py` | `AXES`, `case_context`를 하드코딩 대신 config에서 로드 |
| `intent/analyzer.py` | `pipeline/stage2_analyze_intent.py` | 경로를 `outputs/` 기준 상대경로로 |
| `ideal/clustering/gmm_cluster.py` | `pipeline/stage3b_ideal_clustering.py` | `SAVE_DIR` 절대경로 제거, `n_personas`/`k_range`/`countries`를 config에서 로드 |
| `ideal/debate/debate.py` | `pipeline/stage3a_ideal_debate.py` | 사건 배경/축을 config에서 로드, 에이전트 모델 구성(`models`, `agents_per_model`, `n_rounds`)을 config로 이동 |
| `ideal/persona/simulate_persona.py` (v1, 단순 평균 기반 페르소나 시뮬레이션) | — | **삭제** (GMM 기반 `gmm_cluster.py`(v2)로 대체됨. 다른 파일에서 참조 없음) |
| `visualization/plot_ideal_intent.py` | `stage4a_compare.py` + `stage4b_visualize.py` | 서베이 수치 하드코딩 제거 → config로 이동, 계산/그리기 분리 |
| `visualization/plot_combined_ideal.py` | — | **삭제** (`plot_ideal_intent.py`와 로직 중복, 후자를 기준으로 채택) |
| `visualization/replot.py` | `stage4b_visualize.py`에 흡수 | 체크포인트 존재 시 자동으로 이 동작이 됨 (별도 스크립트 불필요) |
| `intent/visualize.py` | `stage4b_visualize.py`에 흡수 | intent drift 그래프 3종(`drift_lines`, `drift_3d_vectors`, `drift_angle_over_time`)을 `02_intent_drift.json` 기반으로 stage4b에 통합. 별도 stage로 분리하지 않음 |
| `intent/main_1.py` | `run.py` | **삭제** (구식 CLI 진입점, `run.py`가 대체) |
| (여러 파일에 중복된 각도 계산, LLM 호출 코드) | `pipeline/common.py` | 함수 통합 |

**데이터 이관** (코드 아님, `cases/dod_anthropic/`로 그대로 이동):
| 기존 | 신규 |
|---|---|
| `intent/data/sources.json` | `cases/dod_anthropic/raw_sources.json` |
| `intent/data/extracted.json` | `cases/dod_anthropic/outputs/01_intent_extracted.json` |
| `intent/data/analysis.json` | `cases/dod_anthropic/outputs/02_intent_drift.json` |

**삭제 대상 정리 (모두 마지막 단계, 다른 이관 및 검증 완료 후에만 실행)**:
- `ideal/persona/simulate_persona.py` (v1)
- `intent/main_1.py`
- `visualization/plot_combined_ideal.py`

---

## 7. 클로드 코드 작업 순서 제안

1. `git checkout -b refactor-pipeline`
2. `pipeline/common.py`부터 작성 (공통 함수 추출)
3. stage1 → stage2 → stage3b(클러스터링, `ideal/clustering/gmm_cluster.py` 기반) → stage3a(디베이트, `Ideal/debate/debate.py` 기반) → stage4a → stage4b 순서로 이관
4. 기존 `dod_anthropic` 데이터로 `cases/dod_anthropic/`을 채우고 `run.py --case dod_anthropic`이 기존과 동일한 결과를 내는지 확인
5. 확인되면 새로운 사례용 `config.json` 템플릿 작성
