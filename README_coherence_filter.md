# 아키타입 × 소비 성향 정합성 필터링 파이프라인

## 개요

본 파이프라인은 Nemotron 생성 페르소나 데이터에서 **아키타입(Archetype)**과 **소비 성향(Cluster Type)** 조합의 정합성을 검증하고, 비정합 데이터를 탐지하여 학습 데이터 품질을 개선하는 시스템입니다.

**핵심 목적**: 통계적 + 의미론적 이중 필터링으로 허상 조합(spurious correlation)을 제거

---

## 주요 기능

### 1. **통계적 정합성 검증 (Chi-square Test)**
- 5×5 빈도 매트릭스에서 표준화 잔차 계산
- 기대 빈도 대비 실제 빈도의 통계적 유의성 판단
- 임계값: **< -1.5** → 통계적 비정합 신호 🔴

### 2. **의미론적 응집도 검증 (Embedding Gap)**
- 페르소나 텍스트 임베딩(BAAI/bge-m3) 기반 내부/외부 유사도 계산
- **Intra-Similarity** (그룹 내부 응집도) vs **Inter-Similarity** (그룹 간 거리)
- 임계값: **Gap < 0.012** → 임베딩 비정합 신호 🟡

### 3. **최종 판정 로직 (3-Tier System)**
```
❌ INVALID : 두 신호 모두 비정합 → 학습 데이터 제거 권장
🟡 REVIEW  : 한 신호만 비정합   → 수동 검토 또는 가중치 감소
🟢 VALID   : 두 신호 모두 정합  → 그대로 사용
```

---

## 🔍 필터링 메커니즘

### 표준화 잔차 (Chi-square Standardized Residual)
$$
	ext{Residual}_{i,j} = rac{O_{i,j} - E_{i,j}}{\sqrt{E_{i,j}}}
$$
- $O_{i,j}$: 관측 빈도 (실제 데이터 개수)
- $E_{i,j}$: 기대 빈도 (독립 가정 시 예상 개수)
- **해석**: 값이 -2.0 이하면 해당 조합이 통계적으로 과소 출현

### 임베딩 갭 (Embedding Cohesion Gap)
$$
	ext{Gap} = 	ext{Intra-Sim} - 	ext{Inter-Sim}
$$
- **Intra-Sim**: 같은 조합 내 페르소나들 간 평균 코사인 유사도
- **Inter-Sim**: 해당 조합과 전체 외부 페르소나 간 평균 코사인 유사도
- **해석**: Gap이 0.012 미만이면 내부 응집력이 약해 의미론적으로 비정합

---

##  설정 파라미터

### 임계값 설정
```python
CHI_THRESHOLD    = -1.5   # 표준화 잔차 임계값 (이하면 통계적 비정합)
GAP_THRESHOLD    = 0.012  # 임베딩 갭 임계값 (미만이면 의미론적 비정합)
MIN_SAMPLE_COUNT = 3      # 판정 가능 최소 샘플 수
```
---

##  실행 흐름

### STEP 1: 데이터 로드 및 조인
```python
from pathlib import Path

archePath  = Path("/content/drive/MyDrive/data/sampled_nemotron_personas_10k.csv")
clusterPath = Path("/content/drive/MyDrive/data/nemotron_cluster_types.csv")

df_persona = pd.read_csv(archePath)
df_cluster = pd.read_csv(clusterPath)
df = pd.merge(df_persona, df_cluster, on='uuid', how='inner')
```

### STEP 2: 카이제곱 표준화 잔차 계산
```python
count_matrix = pd.crosstab(df['archetype'], df['type1'])
chi2, p, dof, expected = chi2_contingency(count_matrix)
residual_df = (count_matrix - expected) / np.sqrt(expected)
```

**출력 예시**:
```
[ 표준화 잔차 매트릭스 ]
type1   사회적 소비  적극적 소비  자기 중심적  윤리적 소비  소극적 실속
Arch 0   -5.51    3.43   -0.23   -3.07    3.67
Arch 1    5.97   -1.32   -2.05   -0.52   -4.02
```
→ Arch 0 × 사회적 소비 조합은 -5.51로 심각한 통계적 비정합

### STEP 3: 임베딩 응집도 갭 계산
```python
model = SentenceTransformer('BAAI/bge-m3', device='cuda')
all_embeddings = model.encode(df['persona'].tolist(), normalize_embeddings=True)

for arch in archetypes:
    for cluster in CLUSTER_ORDER:
        mask = (df['archetype'] == arch) & (df['type1'] == cluster)
        t_embs = all_embeddings[mask]
        o_embs = all_embeddings[~mask]

        intra_sim = (np.sum(cosine_similarity(t_embs)) - n) / (n * (n - 1))
        inter_sim = np.mean(cosine_similarity(t_embs, o_embs))
        gap = intra_sim - inter_sim
```

**출력 예시**:
```
Archetype Cluster      Intra_Sim  Inter_Sim  Emb_Gap  Emb_Signal
Arch 0    사회적 소비    0.4884     0.4857     0.0027   BAD
Arch 1    사회적 소비    0.5052     0.4864     0.0188   OK
```

### STEP 4: 최종 판정
```python
def judge(row):
    chi_bad = row['Chi_Residual'] < CHI_THRESHOLD
    emb_bad = row['Emb_Gap'] < GAP_THRESHOLD

    if chi_bad and emb_bad:
        return '❌ INVALID'
    elif chi_bad or emb_bad:
        return '🟡 REVIEW'
    else:
        return '🟢 VALID'
```

---

##  분석 결과 해석 가이드

### 표준화 잔차 해석
| 값 범위 | 의미 | 조치 |
|---------|------|------|
| **> 2.0** | 통계적 과다 출현 | 해당 조합 특성 분석 권장 |
| **-1.5 ~ 2.0** | 정상 범위 | 정합 |
| **< -1.5** | 통계적 과소 출현 | 비정합 신호 |

### 임베딩 갭 해석
| 값 범위 | 의미 | 조치 |
|---------|------|------|
| **≥ 0.020** | 강한 내부 응집력 | 매우 정합 |
| **0.012 ~ 0.020** | 적정 응집력 | 정합 |
| **< 0.012** | 약한 내부 응집력 | 비정합 신호 |

### 최종 판정 활용
- **❌ INVALID**: 학습 데이터에서 **제외** 또는 별도 재검토
- **🟡 REVIEW**: 샘플링 가중치 **50% 감소** 또는 수동 검증
- **🟢 VALID**: 그대로 학습에 사용

---
