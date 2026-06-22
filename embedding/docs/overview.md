# AIO 페르소나 분류 파이프라인 — 기술 개요

---

## 개요

`nvidia/Nemotron-Personas-Korea` 데이터셋(한국어 LLM 페르소나 100만 건)에 두 종류의 레이블을 자동 부여하는 파이프라인이다.

| 레이블 | 티어 | 이론 기반 | 입력 칼럼 |
|--------|------|----------|----------|
| **Archetype** (5종) | Tier 2 | Schwartz Basic Human Values | career / professional / family / travel / hobbies |
| **소비태그** (5종) | Tier 3 | 소비심리학 라이프스타일 세분화 | culinary\_persona / hobbies\_list |

두 레이블은 독립적으로 도출되므로 5×5 교차 분석이 가능하다.

---

## 파이프라인 구조 (7단계)

```
[HF 데이터셋]
     │
Step 1  embed_percol5.py          BGE-M3으로 5개 AIO 칼럼 인코딩
     │  → 5 × 1024-dim concat → 5120-dim float16 parquet (9 GB)
     │
Step 2  embed_consumption.py      culinary + hobbies 임베딩
     │  → 2 × 1024-dim concat → consumption_emb_n{N}.parquet (2048-dim float32)
     │
Step 3  archetype_cluster.py      Tier 2 archetype 클러스터링
     │  percol5 5120-dim → L2→PCA(100)→L2→UMAP(2D)→KMeans(k=5) → archetype 0–4
     │
Step 4  consumption_cluster.py    Tier 3 소비태그 클러스터링
     │  consumption_emb 2048-dim → L2→PCA(100)→L2→UMAP(2D)→KMeans(k=5) → consumption_tag 0–4
     │  + noise_dist (Shannon 엔트로피 상위 10% 플래깅)
     │  ※ PCA=100은 sweep Exp-001로 확정 (docs/experiment_log.md)
     │
Step 5  anchor_match.py           소비태그 Hungarian 레이블링
     │  anchor 텍스트 → BGE-M3 → 이방성 보정 cosine 5×5 → linear_sum_assignment
     │  → consumption_labeled_n{N}.csv (+ consumption_label)
     │
Step 6  archetype_match.py        archetype Hungarian 레이블링
     │  anchor 텍스트 → BGE-M3 → 이방성 보정 cosine 5×5 → linear_sum_assignment
     │  → archetype_labeled_n{N}.csv (+ archetype_label)
     │
Step 7  matrix.py                 uuid 기준 병합 + 5×5 교차 매트릭스
        → consumption_tags_labeled_n{N}.csv  ← 최종 산출물
        → matrix_5x5_n{N}.csv / matrix_5x5_all_n{N}.csv
```

---

## 핵심 기술 선택

### BGE-M3 임베딩

- 한국어 다국어 dense retrieval 모델, 1024-dim 출력
- 5개 칼럼을 **독립 인코딩** 후 concat → 5120-dim: 칼럼별 신호 보존
- fp16 추론 + `torch.compile(mode="reduce-overhead")` 로 속도 최적화

### 이중 L2 정규화

PCA 전후 각각 L2 정규화:
1. **PCA 전**: 벡터 크기 편향 제거 → 방향 정보만 사용
2. **PCA 후**: PCA로 재분배된 스케일 보정 → UMAP cosine metric 보장

### UMAP → KMeans (2D)

고차원(5120-dim) 직접 클러스터링 대비 UMAP 2D 투영 후 클러스터링이 더 안정적인 볼록(convex) 클러스터를 형성한다. `n_neighbors=15, min_dist=0.1, metric="cosine"`.

### BGE-M3 이방성 보정 (Hungarian 전처리)

BGE-M3 공간은 비등방성(anisotropic)이므로 raw cosine이 방향보다 크기에 편향된다 (Li et al. 2020; Su et al. 2021). 공간 평균(data\_mean)을 차감 후 L2 정규화해 보정한다.

```
sim = (centroid − data_mean)/‖·‖  @  (anchor − data_mean)/‖·‖
```

### Anchor 텍스트 품질: 이론 직역 → 실제 인구통계

| 접근 | 방법 | 결과 |
|------|------|------|
| v1 | Schwartz 이론 직역 | low-conf 4/5, cosine 최저 +0.015 |
| v2 | `archetype_inspect.py` medoid 샘플 확인 후 재합성 | low-conf 0/5, cosine 최저 +0.297 |

핵심 교훈: **anchor 텍스트 길이보다 실제 클러스터 인구통계 일치가 결정적**이다.

---

## 레이블 정의

### Tier 2 — Archetype (Schwartz BHV 인접 쌍)

| 레이블 | 핵심 동기 | 실측 특성 (n=200,000) |
|--------|---------|-------------------|
| 안전·규범 | 안정·질서·규칙 준수 | 전문직(철도 관제사·항공 정비사), 매뉴얼 집착, 자연 휴양 |
| 자애·안전 | 내집단 돌봄 + 안정 기반 | 은퇴·노년층, 손주 카카오톡, 텃밭·트로트·화투 |
| 전통·자애 | 문화 계승 + 공동체 유대 | 현장 노동자(청소원·경비원), 역사 유적지, 단골 주점 소주 |
| 성취·자율 | 독립적 목표 + 역량 인정 | 분석가·마케터, 엑셀·깃허브 완벽주의, 성수동 맛집 분석 |
| 자율·자극 | 자유로운 선택 + 새 경험 | 육아맘 소품샵 꿈, 골목 산책, 수국 가꾸기 |

### Tier 3 — 소비태그

| 레이블 | 행동 시그니처 | anchor cosine |
|--------|------------|--------------|
| 절약·계획 | 식비 예산, 식재료 직접 구매, 외식 월 1–2회 절제 | +0.183 |
| 경험·관계 | 노포·맛집 탐방, 친구·가족과 식사, 동호회 | +0.081 |
| 가족·실용 | 된장찌개·나물 직접 요리, 가족 외식 단골 식당 | +0.382 |
| 충동·즉흥 | 배달 앱 치킨·족발·마라탕, 고깃집 삼겹살·소주 | +0.355 |
| 가치·정성 | 소금빵·한정식 플레이팅, 홈가드닝, 일상 블로그 | +0.411 |

---

## 결과 요약 (n=200,000)

### Archetype × 소비태그 5×5 매트릭스 (noise\_dist=0 기준)

```
                  절약·계획  경험·관계  가족·실용  충동·즉흥  가치·정성
전통·자애 (n≈26k)    6,362     5,400     6,173     3,490     2,367  ← 소비 이질성 높음
안전·규범 (n≈56k)   14,587    11,140     3,938    14,478     5,311  ← 균등, 절약 약간 우세
자애·안전 (n≈43k)    6,351     6,897    20,264     2,948     3,608  ← 가족·실용 압도
성취·자율 (n≈41k)   11,890     5,016     1,147    13,505     5,382  ← 충동·즉흥 우세
자율·자극 (n≈33k)   10,745     4,648     3,933     5,006     5,413  ← 절약 약간 우세
```

주요 해석:
- **자애·안전 × 가족·실용** (20,264명): 은퇴·노년층의 가족 식탁 중심 소비와 일치
- **성취·자율 × 충동·즉흥** (13,505명): 성취 욕구 높은 집단의 외식·배달 스트레스 해소
- **전통·자애**: 모든 소비태그에 균등 분포 → 현장 노동자 계층의 소비 다양성

---

## 실행

```bash
uv run main.py --step 1          # percol5 임베딩 (~4–5시간)
uv run main.py --step 2          # 소비 임베딩 (~2–3시간)
uv run main.py --step 3          # Tier 2 archetype 클러스터링
uv run main.py --step 4          # Tier 3 소비태그 클러스터링
uv run main.py --step 5          # 소비태그 레이블링
uv run main.py --step 6          # archetype 레이블링
uv run main.py --step 7          # 병합 + 5×5 매트릭스
uv run main.py --status          # 산출물 존재 여부 확인
```

GPU(`cuda:0`) 필수. 소규모 검증: `--sample 50000` 옵션.

---

*상세 문서: [스크립트](scripts.md) · [산출물](outputs.md) · [레이블](labels.md) · [Anchor](anchors.md) · [설계 결정](pipeline.md) · [실험 로그](experiment_log.md)*
