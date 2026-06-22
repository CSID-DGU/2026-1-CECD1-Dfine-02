# 산출물 상세 설명

파이프라인이 생성하는 주요 산출물의 스키마·형식·해석 방법을 설명한다.  
`{N}`은 `--sample` 값(기본 200,000)을 의미한다.

---

## embeddings_percol5.parquet

**경로**: `resource/embeddings_percol5/embeddings_percol5.parquet`  
**생성**: Step 1 (`embed_percol5.py`)  
**크기**: 약 9 GB (n=1,000,000 기준)

### 스키마

| 열 | 타입 | 설명 |
|----|------|------|
| `uuid` | string | 페르소나 고유 식별자 |
| `embedding` | float16[5120] | 5 AIO 칼럼 × BGE-M3 1024-dim concat |

### 구성

```
career_goals_and_ambitions [1024-dim]
professional_persona       [1024-dim]
family_persona             [1024-dim]
travel_persona             [1024-dim]
hobbies_and_interests      [1024-dim]
─────────────────────────────────────
concat                     [5120-dim]
```

float16 저장으로 float32 대비 절반 용량. L2 정규화는 Step 2에서 수행.

---

## consumption_emb_n{N}.parquet

**경로**: `resource/outputs/consumption_emb_n{N}.parquet`  
**생성**: Step 2 (`embed_consumption.py`)  
**크기**: 약 738 MB (n=200,000), 약 3.7 GB (n=1,000,000)

### 스키마

| 열 | 타입 | 설명 |
|----|------|------|
| `uuid` | string | 페르소나 고유 식별자 |
| `embedding` | float32[2048] | culinary + hobbies BGE-M3 concat |

### 용도

- Step 4(`consumption_cluster.py`): KMeans 클러스터링 입력
- Step 5(`anchor_match.py`): anchor cosine 유사도 계산 입력 (BGE-M3 재추론 없이)

---

## archetype_n{N}.csv

**경로**: `resource/outputs/archetype_n{N}.csv`  
**생성**: Step 3 (`archetype_cluster.py`)

### 스키마

| 열 | 타입 | 설명 |
|----|------|------|
| `uuid` | string | 페르소나 고유 식별자 |
| `archetype` | int (0–4) | Tier 2 Schwartz 가치 유형 클러스터 |

레이블 문자열은 Step 6에서 추가된다.

---

## consumption_cluster_n{N}.csv

**경로**: `resource/outputs/consumption_cluster_n{N}.csv`  
**생성**: Step 4 (`consumption_cluster.py`)

### 스키마

| 열 | 타입 | 설명 |
|----|------|------|
| `uuid` | string | 페르소나 고유 식별자 |
| `consumption_tag` | int (0–4) | Tier 3 소비 패턴 클러스터 |
| `noise_dist` | int (0 or 1) | Shannon 엔트로피 상위 10% → 1 (경계 불확실 샘플) |
| `entropy` | float32 | 5 centroid 역거리 softmax Shannon H (nats) |

### 예시

```
uuid,consumption_tag,noise_dist,entropy
a2996e70...,3,0,0.848
ee5e6eae...,0,1,1.422
```

`noise_dist=1`인 행은 여러 클러스터 경계에 걸쳐 있어 분석 시 제외 또는 주의 필요.

---

## consumption_labeled_n{N}.csv

**경로**: `resource/outputs/consumption_labeled_n{N}.csv`  
**생성**: Step 5 (`anchor_match.py`)

### 스키마

`consumption_cluster_n{N}.csv`의 모든 열 + 추가 열:

| 열 | 타입 | 설명 |
|----|------|------|
| `consumption_label` | string | Hungarian 매핑 결과 레이블 (5종 중 하나) |

---

## archetype_labeled_n{N}.csv

**경로**: `resource/outputs/archetype_labeled_n{N}.csv`  
**생성**: Step 6 (`archetype_match.py`)

### 스키마

`archetype_n{N}.csv`의 모든 열 + 추가 열:

| 열 | 타입 | 설명 |
|----|------|------|
| `archetype_label` | string | Hungarian 매핑 결과 레이블 (5종 중 하나) |

---

## consumption_tags_labeled_n{N}.csv

**경로**: `resource/outputs/consumption_tags_labeled_n{N}.csv`  
**생성**: Step 7 (`matrix.py`)  
**크기**: 약 13 MB (n=200,000)

### 스키마

`consumption_labeled` + `archetype_labeled` 를 uuid 기준 inner merge:

| 열 | 타입 | 설명 |
|----|------|------|
| `uuid` | string | 페르소나 고유 식별자 |
| `consumption_tag` | int (0–4) | Tier 3 클러스터 번호 |
| `noise_dist` | int (0 or 1) | 경계 불확실 플래그 |
| `entropy` | float32 | Shannon H |
| `consumption_label` | string | Tier 3 레이블 |
| `archetype` | int (0–4) | Tier 2 클러스터 번호 |
| `archetype_label` | string | Tier 2 레이블 |

### 소비태그 레이블 예시

| consumption_tag | consumption_label |
|----------------|-------------------|
| 0 | 충동·즉흥 |
| 1 | 절약·계획 |
| 2 | 경험·관계 |
| 3 | 가족·실용 |
| 4 | 가치·정성 |

**최종 분석 대상 파일**. archetype × consumption_label 교차 분석이 이 파일을 기준으로 한다.

---

## anchor_mapping_n{N}.json

**경로**: `resource/outputs/anchor_mapping_n{N}.json`  
**생성**: Step 5 (`anchor_match.py`)

### 형식

```json
{
  "0": "충동·즉흥",
  "1": "절약·계획",
  "2": "경험·관계",
  "3": "가족·실용",
  "4": "가치·정성"
}
```

키는 KMeans 클러스터 번호(문자열), 값은 소비태그 레이블.  
`consumption_inspect.py`가 이 파일을 읽어 클러스터 번호 옆에 레이블을 표시한다.

---

## anchor_sim_n{N}.csv

**경로**: `resource/outputs/anchor_sim_n{N}.csv`  
**생성**: Step 5 (`anchor_match.py`)

### 형식

**행**: 5개 소비태그 anchor 레이블 / **열**: KMeans 클러스터 (cluster0–4) / **값**: 이방성 보정 후 cosine 유사도

### n=200,000 실측값

```
              cluster0   cluster1   cluster2   cluster3   cluster4
절약·계획      -0.090     +0.183     -0.089     -0.006     -0.011
경험·관계      -0.007     -0.021     +0.081     +0.002     -0.049
가족·실용      -0.205     -0.021     -0.089     +0.382     -0.063
충동·즉흥      +0.355     -0.231     +0.183     -0.191     -0.123
가치·정성      -0.272     +0.003     -0.170     +0.065     +0.411
```

### 해석

- 값이 **양수·크다** → anchor와 클러스터 특성이 잘 일치
- 값이 **낮다** (경험·관계 최고 +0.081) → anchor 텍스트 또는 클러스터링 재검토 신호
- **음수** 가능: BGE-M3 이방성 보정(data_mean 차감) 후에는 cosine이 [-1, 1] 전 구간 취함
- Hungarian(`linear_sum_assignment`)이 이 행렬에서 **전체 합 최대** 1:1 매핑을 선택함

---

## archetype_mapping_n{N}.json

**경로**: `resource/outputs/archetype_mapping_n{N}.json`  
**생성**: Step 6 (`archetype_match.py`)

### 형식

```json
{
  "0": "전통·자애",
  "1": "안전·규범",
  "2": "자애·안전",
  "3": "성취·자율",
  "4": "자율·자극"
}
```

키는 KMeans 클러스터 번호(문자열), 값은 Schwartz BHV 인접 쌍 레이블.

---

## archetype_sim_n{N}.csv

**경로**: `resource/outputs/archetype_sim_n{N}.csv`  
**생성**: Step 6 (`archetype_match.py`)

### 형식

**행**: 5개 archetype anchor 레이블 / **열**: KMeans 클러스터 (cluster0–4) / **값**: 이방성 보정 후 cosine 유사도

### n=200,000 실측값 — v1 (이론 기반, 참고용)

```
              cluster0   cluster1   cluster2   cluster3   cluster4
안전·규범      +0.025     +0.080     -0.043     +0.039     -0.130
자애·안전      -0.041     -0.014     +0.015     +0.016     +0.002
전통·자애      +0.091     -0.075     +0.077     +0.006     -0.117
성취·자율      -0.144     -0.000     -0.077     +0.158     +0.029
자율·자극      -0.090     -0.046     -0.066     +0.130     +0.085
```

4/5 low-conf. 원인: Schwartz 이론 직역 → 실제 클러스터 인구통계(은퇴·노년층, 현장 노동자 등)와 불일치.

### n=200,000 실측값 — v2 (inspect 기반 재합성, 확정)

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

### 해석

- v1 대비 개선: low-conf 4/5 → **0/5**, cosine 최저 +0.015 → **+0.297**
- v1 실패 원인은 텍스트 길이가 아닌 **인구통계 가정 오류** — Schwartz 이론 직역 대신 `archetype_inspect.py` medoid 샘플을 기반으로 재합성한 것이 결정적
- **값이 낮을수록** anchor 텍스트와 클러스터 특성이 덜 분리된 것 — 재실행 시 anchor 재합성 또는 클러스터 수 조정의 신호로 활용
- **음수** 가능: BGE-M3 이방성 보정(data_mean 차감) 후 cosine이 [-1, 1] 전 구간을 취함

---

## matrix_5x5_n{N}.csv / matrix_5x5_all_n{N}.csv

**경로**: `resource/outputs/matrix_5x5_n{N}.csv` (noise_dist=0만)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`resource/outputs/matrix_5x5_all_n{N}.csv` (전체 포함)  
**생성**: Step 7 (`matrix.py`)

### 형식

행: archetype (arch0–arch4) / 열: consumption_tag (ctag0–ctag4) / 값: 인원 수

### n=200,000 실측값 (noise_dist=0 기준)

```
          ctag0    ctag1    ctag2    ctag3    ctag4
arch0      3490     6362     5400     6173     2367
arch1     14478    14587    11140     3938     5311
arch2      2948     6351     6897    20264     3608
arch3     13505    11890     5016     1147     5382
arch4      5006    10745     4648     3933     5413
```

### 해석 지침

- **대각선 집중** → archetype과 소비태그가 잘 정렬됨
- **특정 셀 지배** (예: arch2–ctag3 20,264) → 해당 archetype의 소비 패턴 특성이 강함
- **균등 분포 행** (예: arch0) → 해당 archetype은 소비 패턴 이질성이 높음
- `matrix_5x5_all_n{N}.csv`와 비교해 noise_dist=1 제외 효과를 확인할 수 있음

레이블 매핑은 `anchor_mapping_n{N}.json`을 참조해야 ctag 번호를 이름으로 변환 가능.
