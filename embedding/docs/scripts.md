# 스크립트 상세 설명

파이프라인 활성 스크립트 8개의 역할·알고리즘·파라미터를 설명한다.

---

## main.py — 파이프라인 오케스트레이터

### 역할

5단계 파이프라인과 2개 정성 검증 스크립트를 `subprocess`로 실행하는 진입점.  
GPU나 대용량 연산을 직접 수행하지 않으며, 상태 확인·단계 실행·검증을 단일 인터페이스로 제공한다.

### CLI

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--status` | 5개 산출물 존재 여부 + 다음 단계 안내 | — |
| `--step {1,2,3,4,5}` | 해당 단계 스크립트 실행 | — |
| `--inspect {archetype,consumption}` | 정성 검증 스크립트 실행 | — |
| `--sample N` | Step 2–5 및 inspect에 전달할 샘플 크기 | 1,000,000 |
| `--per-cluster N` | inspect 클러스터당 출력 샘플 수 | 5 |

### 내부 동작

```
--step 1  →  src/embed_percol5.py        (--sample 미전달, config.toml 기준)
--step 2  →  src/embed_consumption.py    --sample N
--step 3  →  src/archetype_cluster.py   --sample N
--step 4  →  src/consumption_cluster.py --sample N
--step 5  →  src/anchor_match.py         --sample N
--step 6  →  src/archetype_match.py      --sample N
--step 7  →  src/matrix.py               --sample N
--inspect archetype   →  src/archetype_inspect.py   --sample N --per-cluster N
--inspect consumption →  src/consumption_inspect.py --sample N --per-cluster N
```

Step 1은 `config.toml`의 `n_bench`로 전량 처리하므로 `--sample` 옵션을 전달하지 않는다.

---

## src/embed_percol5.py — Step 1: percol5 BGE-M3 임베딩

### 역할

AIO 5칼럼을 각각 BGE-M3로 임베딩한 뒤 이어붙여 **5120-dim percol5 임베딩**을 생성한다.  
전체 파이프라인에서 가장 시간이 오래 걸리는 단계(약 4–5시간, GPU 기준).

### 입력 칼럼

| 칼럼 | 의미 |
|------|------|
| `career_goals_and_ambitions` | 커리어 목표 |
| `professional_persona` | 직업적 자아 |
| `family_persona` | 가족적 자아 |
| `travel_persona` | 여행 성향 |
| `hobbies_and_interests` | 취미와 관심사 |

### 알고리즘

```
각 칼럼 텍스트 → BGE-M3 dense encode → 1024-dim float16
5개 칼럼 concat → 5120-dim float16
청크 단위 ParquetWriter로 스트리밍 저장 (메모리 절약)
```

`torch.compile(mode="reduce-overhead")`로 GPU 커널 재컴파일 오버헤드 최소화.

### 샘플링 전략 (config.toml `sampling.strategy`)

| 값 | 동작 |
|----|------|
| `random` | 전체(~7M+)에서 `n_bench`건 무작위 추출 |
| `full` | 전체 데이터 사용 |
| `age_stratified` | 19–29 / 30–49 / 50–69 밴드 평탄 추출, 한 밴드가 `n_bench//3`에 미달 시 그 부족분을 다른 밴드로 이월해 총 `n_bench` 도달 (전체 인구 < n인 경우만 예외) |

현재 기본값: `age_stratified` (연령대 편향 방지).

### 주요 파라미터 (config.toml)

| 키 | 설명 |
|----|------|
| `model.batch_size` | GPU forward pass 배치 크기 (기본 128) |
| `model.encode_chunk` | encode() 1회 호출 텍스트 수 — 토큰화 오버헤드 상각 (기본 8192) |
| `model.dtype` | 추론 정밀도 — `fp16` 권장 (L2 정규화 mantissa 정밀도) |
| `benchmark.n_bench` | 처리할 행 수 (기본 1,000,000) |

### 출력

`resource/embeddings_percol5/embeddings_percol5.parquet`  
스키마: `uuid (string)`, `embedding (float16[5120])`

---

## src/embed_consumption.py — Step 2: 소비 BGE-M3 임베딩

### 역할

culinary_persona와 hobbies_and_interests_list 2개 칼럼을 각각 BGE-M3로 임베딩한 뒤 이어붙여 **2048-dim 소비 임베딩**을 생성한다. Step 1이 생성한 parquet의 UUID를 기준으로 동일 페르소나 집합을 처리한다.

### 알고리즘

```
percol5 parquet에서 UUID 목록 로드  (임베딩 벡터는 읽지 않음)
HF 데이터셋 전체 스캔 → UUID 매칭 → culinary + hobbies 텍스트 추출
culinary_persona      → BGE-M3 → 1024-dim float32
hobbies_and_interests_list → BGE-M3 → 1024-dim float32
concat → 2048-dim
청크(50,000행) 단위 ParquetWriter로 스트리밍 저장
```

UUID 목록은 Step 1 parquet에서 읽으므로 **Step 1과 동일한 페르소나 집합**이 보장된다.  
이미 출력 파일이 존재하면 실행을 중단한다(덮어쓰기 방지).

### 주요 파라미터

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--sample` | 처리할 행 수 (percol5 parquet 전체보다 작으면 동일 seed로 서브샘플링) | 1,000,000 |

### 출력

`resource/outputs/consumption_emb_n{N}.parquet`  
스키마: `uuid (string)`, `embedding (float32[2048])`

---

## src/embed_consumption2.py — (독립) hobbies 항목별 임베딩

### 역할

`hobbies_and_interests_list` 칼럼의 **각 항목을 독립적으로** BGE-M3 1024-dim 임베딩한다.  
`embed_consumption.py`(Step 2)와 달리 culinary를 제외하고, 리스트를 join하지 않고 원소별로 처리한다.  
행마다 항목 수가 다르므로 출력 임베딩 길이가 가변적이다. 메인 파이프라인(main.py)에 포함되지 않는 독립 스크립트다.

### 알고리즘

```
percol5 parquet에서 UUID 목록 로드
HF 데이터셋 스캔 → UUID 매칭 → hobbies_and_interests_list 추출 (list 타입 유지)
청크(50,000행) 내 모든 항목을 flat 배열로 묶어 BGE-M3 1회 encode (GPU 효율 극대화)
UUID별로 재그룹 → list<fixed_size_list<float32>[1024]> 형식으로 저장
항목 0개인 행 → 빈 리스트 []
```

### 주요 파라미터

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--sample` | 처리할 UUID 수 (percol5 기준) | 200,000 |
| `--config` | 공용 config 경로 (BGE-M3·dataset) | `config.toml` |

### 출력

`resource/outputs/hobby_emb_n{N}.parquet`  
스키마: `uuid (string)`, `embeddings (list<fixed_size_list<float32>[1024]>)`  
— 행마다 임베딩 개수 = 해당 UUID의 hobbies 항목 수 (가변 길이)

---

## src/archetype_cluster.py — Step 3: Tier 2 archetype 클러스터링

### 역할

percol5 임베딩으로 **Tier 2 archetype**(Schwartz BHV 가치 유형 5종)을 클러스터링한다.  
HF 데이터셋과 BGE-M3를 사용하지 않으므로 파라미터 조정 후 빠르게 재실행 가능하다.

### 알고리즘

```
percol5 5120-dim → L2 정규화(in-place)
→ PCA(100)      → L2 정규화
→ UMAP(2D, cosine metric, n_neighbors=15, min_dist=0.1)
→ KMeans k=5   → archetype 라벨 (0–4)
```

n=1M에서 `l2_norm` 복사본(20GB)을 피하기 위해 in-place 정규화를 사용한다.

### 설정 파일

`src/archetype_cluster.toml` — PCA·UMAP·KMeans 하이퍼파라미터. CLI 인자(`--k`)는 config 값을 오버라이드한다.

### 주요 파라미터

| 옵션 / 키 | 설명 | 기본값 |
|----------|------|--------|
| `--sample` | 처리할 행 수 (CLI 전용) | 50,000 |
| `--k` / `[kmeans] k` | 클러스터 수 | 5 |
| `[pca] n_components` | PCA 차원 | 100 |
| `[umap] n_neighbors` / `min_dist` | UMAP 파라미터 | 15 / 0.1 |
| `[kmeans] n_init` | KMeans 재시작 횟수 | 10 |

### 출력

- `archetype_n{N}.csv` — uuid / archetype(0-4)

---

## src/consumption_cluster.py — Step 4: Tier 3 소비태그 클러스터링

### 역할

Step 2가 생성한 소비 임베딩으로 **Tier 3 소비태그**(소비 패턴 5종)를 클러스터링한다.  
HF 데이터셋과 BGE-M3를 사용하지 않으므로 파라미터 조정 후 빠르게 재실행 가능하다.

### 알고리즘

```
consumption_emb_n{N}.parquet 로드 (2048-dim)
→ L2 정규화 → PCA(100) → L2 정규화
→ UMAP(2D) → KMeans k=5 → consumption_tag 라벨 (0–4)
```

**noise_dist** (경계 불확실 샘플 플래깅)
```
각 점 → 5 centroid까지 거리 역수 → softmax → Shannon 엔트로피 H
H 상위 noise_pct%(기본 10%) → noise_dist = 1
```

### 설정 파일

`src/consumption_cluster.toml` — PCA·UMAP·KMeans·noise 하이퍼파라미터. CLI 인자(`--k` / `--pca2` / `--noise-pct`)는 config 값을 오버라이드한다.

### 주요 파라미터

| 옵션 / 키 | 설명 | 기본값 |
|----------|------|--------|
| `--sample` | 처리할 행 수 (CLI 전용) | 50,000 |
| `--k` / `[kmeans] k` | 클러스터 수 | 5 |
| `--pca2` / `[pca] n_components` | PCA 차원 | 100 (sweep Exp-001 확정) |
| `--noise-pct` / `[noise] pct` | noise_dist 임계 백분위 | 10.0 |
| `[umap] n_neighbors` / `min_dist` | UMAP 파라미터 | 15 / 0.1 |
| `[kmeans] n_init` | KMeans 재시작 횟수 | 10 |
| `--out` | 출력 CSV 경로 (sweep 용, CLI 전용) | 자동 |

### 출력

- `consumption_cluster_n{N}.csv` — uuid / consumption_tag(0-4) / noise_dist / entropy

---

## src/anchor_match.py — Step 5: 소비태그 Hungarian 레이블링

### 역할

KMeans 클러스터 번호(0–4)에 소비태그 레이블(절약·계획 등)을 자동으로 매핑한다.  
`anchors_consumption.py`에 정의된 anchor 텍스트를 기준점으로 사용한다.

### 알고리즘

```
1. cluster centroid 계산 (2048-dim 평균)
2. data_mean 차감 (BGE-M3 이방성 보정, Li 2020 / Su 2021)
3. L2 정규화
4. anchor 텍스트 → BGE-M3 encode → data_mean 차감 → L2 정규화
5. cosine 유사도 행렬 (5 anchor × 5 cluster)
6. scipy.linear_sum_assignment — 최대 합 1:1 매핑 (Hungarian)
7. margin 검사: top-1 vs top-2 cosine 차이 < threshold → ⚠ low-conf 경고
```

**이방성 보정** 필요성: BGE-M3 임베딩은 공간이 등방성이 아니어서 raw cosine 유사도가 방향보다 크기(norm)에 편향될 수 있음. 데이터 평균을 빼면 이 편향이 상쇄된다.

### 설정 파일

- `src/anchor_match.toml` — Hungarian 레이블링 하이퍼파라미터 (`[matching] k / margin / center`)
- 루트 `config.toml` — BGE-M3·dataset 공용 설정 (`--config`로 경로 변경 가능)

CLI 인자(`--k` / `--margin` / `--center`)는 `anchor_match.toml` 값을 오버라이드한다.

### 주요 파라미터

| 옵션 / 키 | 설명 | 기본값 |
|----------|------|--------|
| `--sample` | 입력 파일 크기 (CLI 전용) | 50,000 |
| `--k` / `[matching] k` | 클러스터 수 | 5 |
| `--margin` / `[matching] margin` | low-conf 경고 임계 | 0.05 |
| `--center` / `[matching] center` | 이방성 보정 ON/OFF | ON |
| `--csv` / `--config` | 입력 CSV·공용 config 경로 (CLI 전용) | 자동 / 루트 config.toml |

### 출력

- `consumption_labeled_n{N}.csv` — Step 4 CSV + `consumption_label` 열
- `anchor_mapping_n{N}.json` — `{"0": "충동·즉흥", "1": "절약·계획", ...}`
- `anchor_sim_n{N}.csv` — 5×5 cosine 유사도 행렬 (행=anchor, 열=cluster)

---

## src/archetype_match.py — Step 6: archetype Hungarian 레이블링

### 역할

archetype 클러스터 번호(0–4)에 Schwartz BHV 레이블(안전·규범 등)을 자동으로 매핑한다.  
`anchors_archetype.py`에 정의된 anchor 텍스트(5 AIO 칼럼 × 5 archetype)를 기준으로 사용한다.

### 알고리즘

anchor_match.py와 동일한 Hungarian 방식이나 입력 공간이 다르다.

```
percol5 9GB 로드 → centroid 계산 → del + gc → BGE-M3 로드 (순차 메모리 확보)
anchor 텍스트 (5칼럼 × 5종) → BGE-M3 → 5120-dim concat → 이방성 보정 → L2
cosine 5×5 → Hungarian → archetype_label 열 추가
```

percol5(~20GB float32)와 BGE-M3(~5GB VRAM)를 동시에 메모리에 올리지 않도록  
centroid 계산 후 반드시 `del emb; gc.collect()`로 해제한 뒤 모델을 로드한다.

### 설정 파일

- `src/archetype_match.toml` — Hungarian 레이블링 하이퍼파라미터 (`[matching] k / margin / center`)
- 루트 `config.toml` — BGE-M3·dataset 공용 설정 (`--config`로 경로 변경 가능)

CLI 인자(`--k` / `--margin` / `--center`)는 `archetype_match.toml` 값을 오버라이드한다.

### 출력

- `archetype_labeled_n{N}.csv` — Step 3 CSV + `archetype_label` 열
- `archetype_mapping_n{N}.json` — `{"0": "전통·자애", ...}`
- `archetype_sim_n{N}.csv` — 5×5 cosine 유사도 행렬

---

## src/matrix.py — Step 7: 최종 병합 + 매트릭스

### 역할

Step 5 (`consumption_labeled`)와 Step 6 (`archetype_labeled`) 산출물을 uuid 기준으로 병합해 **최종 산출물**을 만들고,  
archetype × 소비태그 5×5 정합 매트릭스를 출력한다.

### 알고리즘

```
consumption_labeled_n{N}.csv + archetype_labeled_n{N}.csv → uuid 기준 inner join
archetype × consumption_tag → 5×5 카운트 행렬 (noise_dist=0 / 전체)
병합 결과 → consumption_tags_labeled_n{N}.csv
```

### 주요 파라미터

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--sample` | 입력 파일 크기 | 1,000,000 |
| `--k` | 클러스터 수 | 5 |

### 출력

- `consumption_tags_labeled_n{N}.csv` — 전 컬럼 병합 **(최종 산출물)**
- `matrix_5x5_n{N}.csv` — archetype × 소비태그 (noise_dist=0만)
- `matrix_5x5_all_n{N}.csv` — archetype × 소비태그 (전체)

---

## src/anchors_consumption.py — 소비태그 anchor 텍스트 상수

### 역할

`anchor_match.py`가 import하는 **상수 모듈**. 연산 없음.  
5개 소비태그 각각에 대해 `culinary_persona`와 `hobbies_and_interests_list` 텍스트를 정의한다.

### 구조

```python
ANCHORS: dict[str, dict[str, str]]  # {레이블: {칼럼: 텍스트}}
LABELS: list[str]                   # ANCHORS.keys() 순서 고정
EMBED_COLS: list[str]               # ["culinary_persona", "hobbies_and_interests_list"]
```

anchor 텍스트는 임베딩 공간에서 데이터와 동일한 포맷(culinary + hobbies concat)으로 비교되므로, 어조·길이·어휘가 실제 페르소나 텍스트와 유사해야 cosine 유사도가 유의미해진다. → [Anchor 관리](anchors.md) 참조.

---

## src/sweep_consumption_pca.py — PCA 차원 비교 실험

### 역할

`consumption_cluster.py`의 `--pca2` 값을 여러 개 순차 실행하고 각 설정에서 `anchor_match.py`의 Hungarian 결과를 수집해 **PCA 차원 선택 근거**를 정량적으로 비교한다.

### 평가 지표

| 지표 | 설명 |
|------|------|
| `min_cos` | Hungarian 할당 후 최저 anchor cosine (레이블 신뢰도 하한) |
| `mean_cos` | Hungarian 할당 cosine 평균 |
| `min_margin` | top-1 vs top-2 cosine 차 최솟값 (낮을수록 레이블 모호) |
| `balance` | 최소/최대 클러스터 크기 비율 (1.0 = 완전 균등) |

### 알고리즘

```
--pca 목록의 각 값에 대해:
  consumption_cluster.py --pca2 {pca} --out sweep_pca/cons_pca{pca}_n{N}.csv
  anchor_match.py --csv sweep_pca/cons_pca{pca}_n{N}.csv
  anchor_sim_n{N}.csv → 복사 보존 → Hungarian 지표 계산
최종 비교 테이블 출력
```

anchor_match.py는 `consumption_emb_n{N}.parquet`를 캐시로 재사용하므로 BGE-M3는 **첫 실행 1회**만 로드된다.

### 주요 파라미터

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--sample` | 처리할 행 수 (필수) | — |
| `--pca` | 비교할 PCA 차원 목록 | `10 30 50 100 200 500` 권장 |
| `--noise-pct` | noise_dist 임계 백분위 | 10.0 |

### 권장 실험 범위

신뢰 있는 곡선을 위해 10× 범위를 로그 간격으로 커버:

| PCA | 목적 |
|-----|------|
| 10 | 과소 축소 하한 |
| 30 | 하한 근방 |
| 50 | 현재 기본값 (기준점) |
| 100 | 2× 기본값 |
| 200 | 4× 기본값 |
| 500 | 상한 |

### 출력

- `resource/outputs/sweep_pca/cons_pca{pca}_n{N}.csv` — 각 PCA 설정의 클러스터 결과
- `resource/outputs/sweep_pca/anchor_sim_pca{pca}_n{N}.csv` — 각 PCA 설정의 5×5 cosine 유사도 행렬
- 표준 출력 — 지표 비교 테이블

### 소요 시간 (n=50,000 기준)

consumption_cluster 1회 ~2분, anchor_match 첫 실행 ~5분·이후 ~1분 → **PCA 6개 기준 약 20분**

### Usage

```bash
uv run src/sweep_consumption_pca.py --sample 50000
uv run src/sweep_consumption_pca.py --sample 50000 --pca 10 30 50 100 200 500
```

---

## src/archetype_inspect.py — Tier 2 정성 검증

### 역할

archetype 클러스터별로 **medoid 샘플**(centroid 최근접)과 **random 샘플**의 AIO 원문 5칼럼을 출력해 레이블 타당성을 수동 확인한다.  
입력: `archetype_n{N}.csv` (Step 3 산출물)

### 알고리즘

```
percol5 5120-dim → L2 → PCA(100) → L2
각 클러스터 centroid 최근접 점 = medoid
random n개 추가 선택
HF 데이터셋에서 해당 uuid의 텍스트 5칼럼 로드 후 출력
```

percol5 전체(~20GB float32)를 메모리에 로드하므로 RAM 32 GB 이상 권장.

---

## src/consumption_inspect.py — Tier 3 정성 검증

### 역할

소비태그 클러스터별로 **medoid + random 샘플**의 culinary_persona, hobbies_and_interests_list 원문을 출력한다.  
입력: `consumption_cluster_n{N}.csv` (Step 4 산출물). `anchor_mapping_n{N}.json`이 있으면 클러스터 번호 옆에 레이블도 표시한다.

### archetype_inspect.py와의 차이

| 항목 | archetype_inspect | consumption_inspect |
|------|-------------------|---------------------|
| 임베딩 소스 | percol5 (5120-dim, ~20GB) | consumption_emb parquet (2048-dim, ~8GB) |
| PCA | PCA(100) 적용 | 적용 안 함 |
| 표시 칼럼 | AIO 5칼럼 | culinary + hobbies 2칼럼 |
| anchor 매핑 표시 | 없음 | 있으면 표시 |

Step 2 산출물(consumption_emb parquet)을 그대로 사용하므로 archetype_inspect보다 빠르게 실행된다.
