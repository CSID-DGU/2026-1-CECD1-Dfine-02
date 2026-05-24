# 산출물 설계 — 목표 대비 변경 이력

원래 목표(핵심 책임·산출물 3종)와 현재 구현의 차이를 기록하고, 최종 산출물 스키마를 확정한다.

---

## 원래 목표 대비 변경 요약

| 원래 목표 | 현재 구현 | 변경 이유 |
|----------|----------|---------|
| Tier 1 하드 필터 (age 3밴드) | **age_stratified 샘플링으로 구현** (`embed_percol5.py`) | 절대 필터 대신 평탄 샘플링 — 19-29 / 30-49 / 50-69 밴드 각 n//3 목표, 한 밴드 미달 시 잔여 정원을 다른 밴드로 이월해 총 n에 도달. `config.toml` `strategy = "age_stratified"` 활성화 |
| 유사도 임계값 미만 "분류 불가" 탈락 | `noise_dist` 플래그로 대체 | 임계값 설정은 임의적 — Shannon 엔트로피 상위 10%로 경계 불확실 샘플을 플래깅하고 분석 시 선택적 제외 |
| 아키타입 단일 티어 | Tier 2(archetype) + Tier 3(소비태그) 2-tier 독립 분류 | 가치관과 소비 패턴이 다른 신호 — 동일 공간 혼합 시 해석력 저하 |
| `embeddings.npy` (단일 배열) | `embeddings_percol5.parquet` (uuid + float16[5120]) | uuid 없는 npy는 데이터셋과 조인 불가; parquet이 uuid 키 + 컬럼 압축 지원 |
| `archetype_labels.parquet` (uuid, archetype, similarity_score) | `consumption_tags_labeled_n{N}.csv` | archetype 외 소비태그 추가; similarity_score 대신 entropy(간접 신뢰도) + noise_dist 플래그; 레이블은 JSON 매핑으로 분리 |
| `anchor_texts.json` (5개 앵커) | `anchors_archetype.py` + `anchors_consumption.py` | 2-tier 분리로 파일 2개; JSON 대신 Python 상수 → 칼럼 순서와 concat 구조를 코드에서 강제 |

---

## 폐기된 설계 결정

### Tier 1 — age 균등 샘플링

원래 계획: `age` 3밴드(청년·중장년·노년) + `occupation` 분산으로 후보 풀 구성.

**현재 구현**: 절대 필터(occupation 분산 기준)는 삭제하고 `age_stratified` 평탄 샘플링으로 대체. `embed_percol5.py`의 `_load_age_stratified`가 19-29 / 30-49 / 50-69 3밴드에서 평탄 추출하되, **한 밴드 인구가 `n//3`에 미달하면 그 부족분을 다른 밴드 정원에 이월**해 최종 합이 `n`에 도달하도록 한다 (데이터셋 전체 인구 < n 인 경우만 예외, 경고 출력). 활성화 방법: `config.toml` → `[sampling] strategy = "age_stratified"`.

현재 기본값은 `strategy = "random"` (200,000건 랜덤). 연령대 편향이 분석 결과에 영향을 준다고 판단될 때 `age_stratified`로 전환한다.

### 유사도 임계값 탈락

원래 계획: `similarity_score < θ` → "분류 불가" 제외.

**대체**: `noise_dist` 플래그 (Shannon 엔트로피 상위 10%). 절대 임계값 θ는 anchor 텍스트 버전마다 달라지므로 재현성이 낮다. 엔트로피 기반 상위 10%는 상대적 기준으로 재현 가능하며, 탈락이 아닌 플래그로 남겨 분석가가 선택적으로 제외할 수 있게 한다.

---

## 현재 산출물 스키마 (확정)

### 핵심 산출물

| 파일 | 원래 대응 | 스키마 | 용도 |
|------|----------|--------|------|
| `embeddings_percol5/embeddings_percol5.parquet` | `embeddings.npy` | uuid · float16[5120] | 5 AIO 칼럼 × BGE-M3 1024-dim concat |
| `outputs/consumption_tags_labeled_n{N}.csv` | `archetype_labels.parquet` | uuid · archetype · consumption_tag · noise_dist · entropy · consumption_label · archetype_label | 최종 분류 결과 **(분석 기준 파일)** |
| `outputs/anchor_mapping_n{N}.json` | `anchor_texts.json` (일부) | `{"0": "충동·즉흥", ...}` | 소비태그 cluster_id → 레이블 |
| `outputs/archetype_mapping_n{N}.json` | `anchor_texts.json` (일부) | `{"0": "전통·자애", ...}` | archetype cluster_id → 레이블 |

### 추가 산출물 (원래 목표에 없던 것)

| 파일 | 스키마 | 용도 |
|------|--------|------|
| `outputs/consumption_emb_n{N}.parquet` | uuid · float32[2048] | culinary+hobbies 소비 임베딩 (Step 4 클러스터링 + Step 5 anchor 매칭 입력) |
| `outputs/matrix_5x5_n{N}.csv` | arch0–4 × ctag0–4 인원수 | archetype × 소비태그 교차 분석 (noise_dist=0 기준) |
| `outputs/matrix_5x5_all_n{N}.csv` | 동일 | noise_dist 포함 전체 |
| `outputs/anchor_sim_n{N}.csv` | 5×5 cosine 행렬 | 소비태그 Hungarian 레이블링 품질 진단 |
| `outputs/archetype_sim_n{N}.csv` | 5×5 cosine 행렬 | archetype Hungarian 레이블링 품질 진단 |

### 앵커 텍스트 관리 파일

| 파일 | 원래 대응 | 내용 |
|------|----------|------|
| `src/anchors_archetype.py` | `anchor_texts.json` (아키타입 절반) | `ANCHORS dict`: 5 archetype × 5 AIO 칼럼 텍스트 (25개). v2 확정. |
| `src/anchors_consumption.py` | `anchor_texts.json` (소비 절반) | `ANCHORS dict`: 5 소비태그 × 2 칼럼 텍스트 (10개). v2 확정. |

JSON 대신 Python 상수를 쓰는 이유: `EMBED_COLS` 리스트와 `ANCHORS` 딕셔너리가 같은 파일 안에 있어 칼럼 순서와 concat 구조가 코드 수준에서 강제된다. JSON은 칼럼 순서 불일치를 런타임에 감지하지 못한다.

---

## 최종 산출물 흐름 (원래 → 현재)

```
원래                                    현재
─────────────────────────────────────────────────────
embeddings.npy                    →  embeddings_percol5.parquet
  (n × 1024, float32, 인덱스 기반)       (uuid + float16[5120], parquet)

archetype_labels.parquet          →  consumption_tags_labeled_n{N}.csv
  (uuid, archetype, similarity)        (+ consumption_tag, noise_dist,
                                          entropy, consumption_label,
                                          archetype_label)

anchor_texts.json (5개)           →  anchors_archetype.py (25 texts)
                                  →  anchors_consumption.py (10 texts)

[없음]                            →  matrix_5x5_n{N}.csv (교차 분석)
[없음]                            →  *_sim_n{N}.csv (레이블링 품질 진단)
[없음]                            →  consumption_emb_n{N}.parquet (캐시)
```

---

## EDA 산출물 현황

원래 목표에 EDA(1M 칼럼별 분포, 결측, 길이 통계)가 포함되어 있었으나 **공식 EDA 보고서는 미생성**이다. `resource/outputs/` 내 실험 파일들(`signal_distributions.png`, `pca_scree.png`, `sweep_*.csv` 등)이 탐색 과정의 산물로 남아 있으며, 파이프라인 설계 결정의 근거로 사용되었다.

공식 EDA 산출물이 필요하다면 별도 스크립트 작성이 필요하다.
