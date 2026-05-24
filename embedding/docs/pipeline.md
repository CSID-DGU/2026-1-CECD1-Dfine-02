# 파이프라인 설계 결정

Step 1–7의 알고리즘·파라미터 선택 근거와 트레이드오프를 기록한다.

---

## 전체 구조 결정

### 2-tier 분리 설계

| 티어 | 입력 | 목적 |
|------|------|------|
| Tier 2 (archetype) | 5 AIO 칼럼 전체 → percol5 5120-dim | 삶의 방향·가치관 포착 |
| Tier 3 (소비태그) | culinary + hobbies → 2048-dim | 음식·여가 소비 패턴 포착 |

archetype과 소비태그를 동일 임베딩 공간에서 클러스터링하면 두 신호가 혼합되어 해석이 어려워진다. 독립적으로 클러스터링하고 `matrix_5x5`로 교차 분석하는 방식이 해석 가능성을 높인다.

---

## Step 1 — percol5 임베딩

### BGE-M3 선택

- 한국어 포함 다국어 dense retrieval 모델로 1024-dim dense vector 출력
- `return_dense=True, return_sparse=False, return_colbert_vecs=False`: dense만 사용 (속도·메모리 최적화)
- `torch.compile(mode="reduce-overhead")`: GPU 커널 재컴파일 오버헤드 최소화

### 5칼럼 선택 (career / professional / family / travel / hobbies)

데이터셋 전체 칼럼 중 **삶의 방향·가치관을 가장 직접적으로 서술하는 5개**를 선택했다. `culinary_persona`는 소비 패턴에 편향되므로 Tier 2에서 제외하고 Tier 3 전용으로 사용한다.

### float16 저장

`float32` 대비 절반 용량(9 GB → 4.5 GB). L2 정규화를 Step 2에서 수행하므로 저장 시 정규화 생략 가능. BGE-M3 추론도 fp16으로 수행해 mantissa 정밀도를 bf16보다 높게 유지한다.

---

## Step 2 — 소비 임베딩 (embed_consumption.py)

Tier 3 클러스터링용 임베딩을 별도로 생성한다. percol5(Step 1)와 분리한 이유:

- **칼럼이 다르다**: percol5는 5 AIO 칼럼, 소비 임베딩은 `culinary_persona` + `hobbies_and_interests_list` 2개만 사용한다. 가치관 신호와 소비 신호의 임베딩 공간을 분리해 Tier 2/3 클러스터링을 독립적으로 수행.
- **출력 dtype**: float32 (2048-dim). Step 5 anchor cosine 계산 정밀도 확보. 캐시 용량(~738 MB at 200k)이 허용 범위.

---

## Step 3/4 — 클러스터링 파이프라인

Step 3(`archetype_cluster.py`)은 percol5 → archetype, Step 4(`consumption_cluster.py`)는 소비 임베딩 → consumption_tag 를 동일한 알고리즘 골격으로 처리한다.

### L2 → PCA → L2 → UMAP → KMeans

이중 L2 정규화의 이유:

1. **PCA 전 L2**: BGE-M3 dense 벡터를 단위 구면에 투영해 방향 정보만 사용. 크기 편향 제거.
2. **PCA 후 L2**: PCA는 선형 변환으로 스케일을 재분배하므로 cosine metric을 사용하는 UMAP 전에 재정규화 필요.

### PCA 차원 선택

| 티어 | PCA | 근거 |
|------|-----|------|
| Tier 2 | 100 | 5120-dim → 100. bootstrap ARI 0.954 최고 (baseline 대비 100× 가속). |
| Tier 3 | 100 | 2048-dim → 100. **sweep Exp-001로 확정** (coarse 10–500 + fine 70–150). 100→110 구간에서 min_margin 0.0184 급락. 상세: [experiment_log.md](experiment_log.md) |

### UMAP 파라미터

```python
UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine")
```

- `n_components=2`: KMeans가 2D에서 더 안정적인 볼록(convex) 클러스터를 형성함
- `n_neighbors=15`: 기본값. 로컬-글로벌 구조 균형
- `min_dist=0.1`: 기본값. 클러스터 내부 밀도 허용
- `metric="cosine"`: L2 정규화 후 벡터이므로 방향 유사도 기반 거리 사용

### KMeans k=5

Schwartz BHV 5분위와 소비태그 5종에 맞춰 고정. `n_init=10`으로 수렴 불안정성 완화.

### noise_dist — Shannon 엔트로피 상위 10%

```
각 점 → 5 centroid 역거리 softmax → Shannon H
H ≥ p90 → noise_dist = 1
```

클러스터 경계에 위치해 어느 클러스터에도 명확히 속하지 않는 샘플을 플래깅한다. 상위 10%는 경험적으로 선택했으며 `--noise-pct`로 조정 가능. 분석 시 `noise_dist=0` 행만 사용하는 것을 권장한다.

---

## Step 5/6 — Hungarian 레이블링

### BGE-M3 이방성 보정 (data_mean 차감)

BGE-M3 임베딩 공간은 등방성(isotropic)이 아니다. 공간의 평균 방향이 존재해 raw cosine 유사도가 방향보다 크기에 편향될 수 있다 (Li et al. 2020, Su et al. 2021).

```
centroid_corrected = centroid - data_mean → L2 norm
anchor_corrected   = anchor   - data_mean → L2 norm
sim = anchor_corrected @ centroid_corrected.T
```

`--no-center` 옵션으로 OFF 가능하나 기본값 ON 유지 권장.

### 메모리 순차 확보 (archetype_match.py)

percol5 9 GB를 먼저 로드해 centroid를 계산한 뒤 해제(`del emb; gc.collect()`)하고, 이후 BGE-M3를 로드한다. 두 가지를 동시에 메모리에 올리면 RAM 20 GB 이상 필요.

### Hungarian vs. greedy 매핑

`scipy.optimize.linear_sum_assignment`(Hungarian algorithm)는 5×5 cosine 행렬에서 **전체 합이 최대**인 1:1 매핑을 O(n³) 시간에 보장한다. greedy(행 최댓값 순 할당) 대비 전역 최적이므로 모든 레이블이 사용됨이 보장된다.

### low-confidence 경고 기준

```
margin = top-1 cosine - top-2 cosine < 0.05 → ⚠ low-conf
```

margin이 작으면 anchor 텍스트가 클러스터 특성을 충분히 분리하지 못한다는 신호다. anchor 재합성 또는 클러스터 수 조정이 필요하다.

---

## Step 7 — 병합 + 매트릭스 (matrix.py)

Step 5(`consumption_labeled`)와 Step 6(`archetype_labeled`)를 uuid 기준 inner merge로 합쳐 최종 산출물 `consumption_tags_labeled_n{N}.csv`를 만든다. 같은 단계에서 archetype × consumption_tag 5×5 정합 매트릭스 두 개도 생성:

- `matrix_5x5_n{N}.csv` — `noise_dist=0` 행만
- `matrix_5x5_all_n{N}.csv` — 전체 (noise_dist 포함)

병합을 별도 Step으로 분리한 이유: 클러스터링·anchor 매칭은 독립적으로 재실행될 수 있고, 그때마다 매트릭스 재생성을 자동화하기 위함.

---

## 산출물 포맷 결정

### parquet + zstd

- percol5 9 GB: Parquet columnar format + zstd 압축. `pyarrow.dataset`으로 스트리밍 로드 가능.
- consumption_emb 캐시: Step 5(`anchor_match.py`)가 BGE-M3 재추론 없이 cosine 계산 가능.

### float32 vs float16

| 파일 | dtype | 이유 |
|------|-------|------|
| percol5 parquet | float16 | 저장 용량 절반. 클러스터링은 float32로 변환 후 수행. |
| consumption_emb | float32 | cosine 계산 정밀도 확보. 캐시 파일(738 MB)이라 용량 허용. |
| 최종 CSV | int(archetype, tag), float32(entropy) | 분석 편의 |

---

## 주요 파라미터 요약

| 파라미터 | 값 | 위치 |
|---------|-----|------|
| BGE-M3 batch_size | 128 | config.toml |
| BGE-M3 max_length | 2048 | config.toml |
| BGE-M3 dtype | fp16 | `config.toml` |
| Tier2 PCA | 100 | `src/archetype_cluster.toml` `[pca] n_components` |
| Tier3 PCA | 100 | `src/consumption_cluster.toml` `[pca] n_components` (Exp-001 확정) |
| UMAP n_neighbors | 15 | `src/{archetype,consumption}_cluster.toml` `[umap] n_neighbors` |
| UMAP min_dist | 0.1 | 동상 `[umap] min_dist` |
| UMAP n_components | 2 | 동상 `[umap] n_components` |
| UMAP metric | cosine | 동상 `[umap] metric` |
| KMeans k | 5 | 동상 `[kmeans] k` (CLI `--k` 오버라이드) |
| KMeans n_init | 10 | 동상 `[kmeans] n_init` |
| noise_dist pct | 10% | `src/consumption_cluster.toml` `[noise] pct` (CLI `--noise-pct` 오버라이드) |
| Hungarian margin | 0.05 | `src/{anchor,archetype}_match.toml` `[matching] margin` (CLI `--margin` 오버라이드) |
| 이방성 보정 (center) | true | 동상 `[matching] center` (CLI `--center/--no-center` 오버라이드) |
| random seed | 42 | 전 스크립트에 하드코딩 (재현성 상수) |
