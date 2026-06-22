# AIO 페르소나 분류 파이프라인

`nvidia/Nemotron-Personas-Korea` 데이터셋을 대상으로 Schwartz 가치 기반 **archetype** 5종과 소비 행동 기반 **소비태그** 5종을 부여하는 파이프라인.

**개요**: [파이프라인 기술 개요 (2p)](docs/overview.md) · [파라미터 선택 근거 (HTML)](docs/parameter_rationale.html) · [실험 로그](docs/experiment_log.md)

**상세 문서**: [스크립트 설명](docs/scripts.md) · [산출물 설명](docs/outputs.md) · [레이블 정의](docs/labels.md) · [Anchor 관리](docs/anchors.md) · [설계 결정](docs/pipeline.md) · [산출물 설계 변경](docs/design.md)

---

## 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│           nvidia/Nemotron-Personas-Korea  (HuggingFace)             │
└───────────┬──────────────────────────────────┬──────────────────────┘
            │                                  │
  ┌─────────▼──────────────┐        ┌──────────▼──────────────┐
  │  STEP 1  embed_percol5 │        │  STEP 2  embed_consumption│
  │  [~4–5h]               │        │  [~2–3h]                 │
  │  career/professional/  │        │  culinary_persona         │
  │  family/travel/hobbies │        │  hobbies_list             │
  │  → BGE-M3 × 5칼럼     │        │  → BGE-M3 × 2칼럼        │
  │  → concat 5120-dim f16 │        │  → concat 2048-dim f32   │
  └─────────┬──────────────┘        └──────────┬───────────────┘
            │ embeddings_percol5.parquet (9 GB)  │ consumption_emb_n{N}.parquet
            │                                    │
  ┌─────────▼──────────────┐        ┌────────────▼────────────┐
  │  STEP 3                │        │  STEP 4                 │
  │  archetype_cluster.py  │        │  consumption_cluster.py │
  │  percol5 → L2          │        │  cons_emb → L2          │
  │  → PCA(100) → L2       │        │  → PCA(100) → L2        │
  │  → UMAP(2D) → KMeans   │        │  → UMAP(2D) → KMeans    │
  └─────────┬──────────────┘        └────────────┬────────────┘
            │ archetype_n{N}.csv                  │ consumption_cluster_n{N}.csv
            │                                     │ (+ noise_dist / entropy)
  ┌─────────▼──────────────┐        ┌────────────▼────────────┐
  │  STEP 6                │        │  STEP 5                 │
  │  archetype_match.py    │        │  anchor_match.py        │
  │  anchors_archetype     │        │  anchors_consumption    │
  │  → cosine 5×5          │        │  → cosine 5×5           │
  │  → Hungarian           │        │  → Hungarian            │
  └─────────┬──────────────┘        └────────────┬────────────┘
            │ archetype_labeled_n{N}.csv          │ consumption_labeled_n{N}.csv
            └──────────────┬──────────────────────┘
                           │
              ┌────────────▼───────────────────┐
              │  STEP 7   matrix.py            │
              │  uuid 기준 병합                │
              │  → matrix_5x5_n{N}.csv        │
              └────────────┬───────────────────┘
                           │
         consumption_tags_labeled_n{N}.csv  ← 최종 산출물
```

| 단계 | 스크립트 | 입력 | 출력 |
|------|----------|------|------|
| Step 1 | `embed_percol5.py` | HF 데이터셋 | `embeddings_percol5.parquet` (9 GB) |
| Step 2 | `embed_consumption.py` | HF + Step 1 UUID | `consumption_emb_n{N}.parquet` |
| Step 3 | `archetype_cluster.py` | Step 1 parquet | `archetype_n{N}.csv` |
| Step 4 | `consumption_cluster.py` | Step 2 parquet | `consumption_cluster_n{N}.csv` |
| Step 5 | `anchor_match.py` | Step 4 CSV + Step 2 parquet | `consumption_labeled_n{N}.csv` |
| Step 6 | `archetype_match.py` | Step 3 CSV + Step 1 parquet | `archetype_labeled_n{N}.csv` |
| Step 7 | `matrix.py` | Step 5 + Step 6 CSV | `consumption_tags_labeled_n{N}.csv`, `matrix_5x5_n{N}.csv` |

---

## 환경 설정

```bash
# 의존성 설치
uv sync

# HuggingFace 캐시 경로 (선택 — 기본: ~/.cache/huggingface)
# config.toml → [dataset] cache_dir 에 절대 경로 입력
```

GPU (`cuda:0`) 필수. `config.toml`에서 `device`, `dtype`, `batch_size` 조정 가능.

---

## 실행

### 상태 확인

```bash
uv run main.py --status                    # n=1,000,000 기준 5개 산출물 존재 여부 및 다음 단계 안내
uv run main.py --status --sample 50000     # 소규모 검증용
```

### 전체 파이프라인 (순서대로)

```bash
uv run main.py --step 1                    # percol5 임베딩, age_stratified 1M (~4-5시간)
uv run main.py --step 2 --sample 1000000   # 소비 임베딩 culinary+hobbies (~2-3시간)
uv run main.py --step 3 --sample 1000000   # archetype 클러스터링 (BGE-M3 불필요, 빠름)
uv run main.py --step 4 --sample 1000000   # 소비태그 클러스터링 (BGE-M3 불필요, 빠름)
uv run main.py --step 5 --sample 1000000   # 소비태그 anchor 레이블링
uv run main.py --step 6 --sample 1000000   # archetype anchor 레이블링
uv run main.py --step 7 --sample 1000000   # 최종 병합 + 매트릭스
```

### 소규모 검증 (n=50,000)

```bash
uv run main.py --step 2 --sample 50000
uv run main.py --step 3 --sample 50000
uv run main.py --step 4 --sample 50000
uv run main.py --step 5 --sample 50000
uv run main.py --step 6 --sample 50000
uv run main.py --step 7 --sample 50000
uv run main.py --status  --sample 50000
```

### 정성 검증

```bash
# Tier2 archetype — percol5 공간에서 medoid + random 샘플 출력
uv run main.py --inspect archetype
uv run main.py --inspect archetype --per-cluster 3

# Tier3 소비태그 — 2048-dim 공간에서 medoid + random 샘플 출력
uv run main.py --inspect consumption
uv run main.py --inspect consumption --per-cluster 5 --sample 50000
```

---

## 산출물

상세 스키마와 해석 방법은 [docs/outputs.md](docs/outputs.md) 참조.

| 파일 | 단계 | 설명 |
|------|------|------|
| `resource/embeddings_percol5/embeddings_percol5.parquet` | Step 1 | uuid + float16[5120] percol5 임베딩 |
| `resource/outputs/consumption_emb_n{N}.parquet` | Step 2 | uuid + float32[2048] 소비 임베딩 |
| `resource/outputs/archetype_n{N}.csv` | Step 3 | uuid / archetype(0-4) |
| `resource/outputs/consumption_cluster_n{N}.csv` | Step 4 | uuid / consumption_tag(0-4) / noise_dist / entropy |
| `resource/outputs/consumption_labeled_n{N}.csv` | Step 5 | + consumption_label 열 추가 |
| `resource/outputs/archetype_labeled_n{N}.csv` | Step 6 | + archetype_label 열 추가 |
| `resource/outputs/consumption_tags_labeled_n{N}.csv` | Step 7 | 전 컬럼 병합 **(최종 산출물)** |
| `resource/outputs/matrix_5x5_n{N}.csv` | Step 7 | archetype × 소비태그 정합 매트릭스 |
| `resource/outputs/anchor_mapping_n{N}.json` | Step 5 | 소비태그 cluster_id → 레이블 매핑 |
| `resource/outputs/anchor_sim_n{N}.csv` | Step 5 | 소비태그 5×5 코사인 유사도 행렬 |
| `resource/outputs/archetype_mapping_n{N}.json` | Step 6 | archetype cluster_id → 레이블 매핑 |
| `resource/outputs/archetype_sim_n{N}.csv` | Step 6 | archetype 5×5 코사인 유사도 행렬 |

### anchor_sim_n{N}.csv 상세

Step 5(`anchor_match.py`)가 Hungarian 레이블링 직전에 계산하는 **5×5 코사인 유사도 행렬**.

- **행(row)**: 5개 소비태그 anchor 텍스트 (BGE-M3로 임베딩한 한국어 기술문)
- **열(col)**: KMeans가 만든 5개 클러스터의 centroid
- **값**: BGE-M3 이방성 보정(data_mean 차감) 후 L2 정규화된 벡터 간 cosine 유사도 → 음수 가능

`n=200,000` 실측값:

```
              cluster0   cluster1   cluster2   cluster3   cluster4
절약·계획      -0.090     +0.183     -0.089     -0.006     -0.011
경험·관계      -0.007     -0.021     +0.081     +0.002     -0.049
가족·실용      -0.205     -0.021     -0.089     +0.382     -0.063
충동·즉흥      +0.355     -0.231     +0.183     -0.191     -0.123
가치·정성      -0.272     +0.003     -0.170     +0.065     +0.411
```

Hungarian 매핑 결과 (`linear_sum_assignment` 최대 합):

| 클러스터 | 레이블 | cosine |
|---------|--------|--------|
| cluster0 | 충동·즉흥 | +0.355 |
| cluster1 | 절약·계획 | +0.183 |
| cluster2 | 경험·관계 | +0.081 |
| cluster3 | 가족·실용 | +0.382 |
| cluster4 | 가치·정성 | +0.411 |

값이 낮을수록(경험·관계 +0.081) anchor 텍스트와 클러스터 특성이 덜 명확히 분리된 것이므로, 재실행 시 anchor 텍스트 재합성 또는 클러스터 수 조정의 신호로 활용.

### archetype_sim_n{N}.csv 상세

Step 6(`archetype_match.py`)가 Hungarian 레이블링 직전에 계산하는 **5×5 코사인 유사도 행렬**.

- **행(row)**: 5개 archetype anchor 텍스트 (BGE-M3로 임베딩한 5 AIO 칼럼 concat)
- **열(col)**: KMeans가 만든 5개 클러스터의 centroid (percol5 5120-dim 공간)
- **값**: BGE-M3 이방성 보정(data_mean 차감) 후 L2 정규화된 벡터 간 cosine 유사도

`n=200,000` 실측값 **(v2 anchor — inspect 기반 재합성)**:

```
              cluster0   cluster1   cluster2   cluster3   cluster4
안전·규범      -0.037     +0.310     -0.141     -0.053     -0.120
자애·안전      +0.032     -0.143     +0.302     -0.245     -0.026
전통·자애      +0.365     -0.068     +0.090     -0.127     -0.169
성취·자율      -0.171     -0.038     -0.149     +0.373     -0.078
자율·자극      -0.076     -0.090     +0.012     -0.065     +0.297
```

Hungarian 매핑 결과:

| 클러스터 | 레이블 | cosine | margin | 신뢰도 |
|---------|--------|--------|--------|--------|
| cluster0 | 전통·자애 | +0.365 | 0.275 | ✅ |
| cluster1 | 안전·규범 | +0.310 | 0.347 | ✅ |
| cluster2 | 자애·안전 | +0.302 | 0.270 | ✅ |
| cluster3 | 성취·자율 | +0.373 | 0.411 | ✅ |
| cluster4 | 자율·자극 | +0.297 | 0.285 | ✅ |

v1(이론 기반) 대비: low-conf 4/5 → 0/5, cosine 최저 +0.015 → +0.297. 개선 원인은 텍스트 길이가 아닌 인구통계 가정 수정 — Schwartz 이론 직역 대신 `archetype_inspect.py` medoid 샘플 기반으로 재합성.

---

## 레이블 정의

**Tier 2 — Archetype (Schwartz BHV)**

| 번호 | 레이블 | 핵심 가치 |
|------|--------|----------|
| 0-4 | 안전·규범 / 자애·안전 / 전통·자애 / 성취·자율 / 자율·자극 | Schwartz 순환 모형 5분위 |

**Tier 3 — 소비태그**

| 레이블 | 특성 |
|--------|------|
| 절약·계획 | 예산 관리, 직접 요리, 절제된 외식 |
| 경험·관계 | 맛집 탐방, 사람 중심, 관계 소비 |
| 가족·실용 | 가족 식탁 우선, 건강 고려, 실용 구매 |
| 충동·즉흥 | 배달·야식, 즉흥 외식, 기분 전환형 |
| 가치·정성 | 플레이팅·품질 중시, 정성 소비 |

---

## 스크립트 구조

각 스크립트의 알고리즘·파라미터 상세는 [docs/scripts.md](docs/scripts.md) 참조.

```
main.py                        ← 파이프라인 오케스트레이터
src/
  embed_percol5.py             ← Step 1: percol5 BGE-M3 임베딩 (5칼럼 → 5120-dim)
  embed_consumption.py         ← Step 2: 소비 BGE-M3 임베딩 (culinary+hobbies → 2048-dim)
  archetype_cluster.py         ← Step 3: Tier 2 archetype 클러스터링 (percol5 → KMeans)
  consumption_cluster.py       ← Step 4: Tier 3 소비태그 클러스터링 (소비 임베딩 → KMeans)
  anchor_match.py              ← Step 5: 소비태그 Hungarian 레이블링
  archetype_match.py           ← Step 6: archetype Hungarian 레이블링
  matrix.py                    ← Step 7: 최종 병합 + archetype × 소비태그 매트릭스
  anchors_consumption.py       ← 소비태그 anchor 텍스트 상수
  anchors_archetype.py         ← archetype anchor 텍스트 상수
  archetype_inspect.py         ← Tier2 정성 검증
  consumption_inspect.py       ← Tier3 정성 검증
  legacy/                      ← 실험·비교용 스크립트
config.toml                    ← 모델·데이터셋·런타임 설정
```
