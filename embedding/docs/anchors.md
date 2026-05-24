# Anchor 텍스트 관리

소비태그(`anchors_consumption.py`)·archetype(`anchors_archetype.py`) anchor 텍스트의 역할, 작성 원칙, 버전 이력을 정리한다.

---

## 역할

`anchor_match.py`(Step 5)와 `archetype_match.py`(Step 6)는 KMeans 클러스터 번호(0–4)에 의미 있는 레이블을 자동 부여한다.  
이를 위해 **각 레이블을 대표하는 한국어 페르소나 텍스트**를 BGE-M3로 임베딩해 클러스터 centroid와 cosine 유사도를 비교한다.

```
anchor 텍스트 → BGE-M3 → (n_labels, dim)
                              ↓  cosine
cluster centroid ──────── (k, dim)   →  5×5 유사도 행렬 → Hungarian
```

---

## 포맷 원칙

**데이터 임베딩과 동일한 칼럼·순서로 임베딩해야 한다.**

| 티어 | 사용 칼럼 | 임베딩 dim |
|------|-----------|-----------|
| Tier 3 소비태그 | `culinary_persona` + `hobbies_and_interests_list` | 2048 (1024×2 concat) |
| Tier 2 archetype | `career_goals_and_ambitions` + `professional_persona` + `family_persona` + `travel_persona` + `hobbies_and_interests` | 5120 (1024×5 concat) |

칼럼 순서가 달라지면 concat 벡터의 의미 구조가 바뀌어 유사도가 무효가 된다.

### 텍스트 작성 지침

- **어조·어휘**: 실제 `nvidia/Nemotron-Personas-Korea` 데이터셋의 페르소나 서술 문체와 일치시킨다
- **길이**: 칼럼 실측 분포(~100–300자)와 유사하게 2–4문장
- **구체성**: 직업명·지역·음식명·활동명 등 구체적인 한국어 표현 사용 — 추상적 가치 서술은 유사도를 저하시킨다
- **`hobbies`**: 실제 데이터처럼 쉼표 구분 목록 형태로 작성

---

## 소비태그 anchor (`anchors_consumption.py`)

### v1 → v2 변경 (2026-05-21)

| 단계 | 방법 |
|------|------|
| v1 | 소비심리학·Schwartz 문헌 기반 학술 정의 직역 |
| v2 | n=200,000 정성 검증(`consumption_inspect.py`) medoid 샘플 확인 후 한국 LLM 페르소나 어조로 재합성 |

### v2 결과 (n=200,000)

| 레이블 | cosine | margin |
|--------|--------|--------|
| 절약·계획 | +0.183 | 0.188 |
| 경험·관계 | +0.081 | 0.079 |
| 가족·실용 | +0.382 | 0.403 |
| 충동·즉흥 | +0.355 | 0.172 |
| 가치·정성 | +0.411 | 0.346 |

경험·관계(+0.081)가 낮은 이유: 맛집 탐방·사람 중심 소비는 다른 클러스터와 임베딩 공간이 겹침.

---

## Archetype anchor (`anchors_archetype.py`)

### v1 실패 (2026-05-22)

Schwartz(2012) BHV 이론을 직역한 초초안. 4/5 low-conf, cosine 최저 +0.015.

**원인**: 이론이 묘사하는 행동 패턴과 실제 데이터셋의 한국 LLM 페르소나 인구통계가 불일치.

| 레이블 | v1 cosine | 문제 |
|--------|-----------|------|
| 안전·규범 | +0.080 | "노인 안정 추구" 톤 → 실제 클러스터는 전문직 매뉴얼 준수형 |
| 자애·안전 | +0.015 | "중년 돌봄 직종" 가정 → 실제는 은퇴·노년층, 손주 돌봄 |
| 전통·자애 | +0.091 | "도예·자수·제례" 중심 → 실제는 현장 노동자, 역사 유적지 탐방 |
| 성취·자율 | +0.158 | 방향 맞으나 추상적 서술 |
| 자율·자극 | +0.085 | "암벽등반·익스트림" → 실제는 소품샵 꿈, 골목 산책, 수국 가꾸기 |

### v2 확정 (2026-05-22)

`archetype_inspect.py --sample 200000` 실행 → 클러스터별 medoid + random 샘플 확인 → 실제 인구통계 기반 재합성.

| 레이블 | v2 cosine | margin | 실제 클러스터 특성 |
|--------|-----------|--------|-----------------|
| 안전·규범 | +0.310 | 0.347 | 전문직(철도 관제사·항공 정비사), 매뉴얼 집착·무사고, 자연 휴양 |
| 자애·안전 | +0.302 | 0.270 | 은퇴·노년층, 손주 카카오톡, 텃밭·트로트·화투 |
| 전통·자애 | +0.365 | 0.275 | 현장 노동자(청소원·경비원), 역사 유적지, 단골 주점 소주 |
| 성취·자율 | +0.373 | 0.411 | 분석가·마케터, 엑셀·깃허브 완벽주의, 성수동 맛집 분석 |
| 자율·자극 | +0.297 | 0.285 | 육아맘 소품샵 꿈, 골목 산책, 수국 가꾸기, 수목원 |

**v1 대비**: low-conf 4/5 → 0/5, cosine 최저 +0.015 → +0.297

---

## 핵심 교훈: 텍스트 길이가 아닌 인구통계 일치

v1과 v2의 텍스트 길이는 유사하다(칼럼당 2–4문장). 개선의 결정적 요인은 **실제 클러스터가 어떤 사람들인지**를 anchor 텍스트가 정확히 반영하느냐였다.

> 이론 직역 anchor → 낮은 cosine  
> 실제 medoid 기반 anchor → 높은 cosine

---

## Anchor 재합성 절차

새 anchor 버전이 필요할 때:

```bash
# 1. 현재 클러스터 특성 확인
uv run main.py --inspect archetype --sample 200000 --per-cluster 3
uv run main.py --inspect consumption --sample 200000 --per-cluster 3

# 2. medoid 샘플을 보며 각 클러스터의 지배적 패턴 파악
# 3. anchors_archetype.py / anchors_consumption.py 텍스트 수정
# 4. 재실행 (anchor 변경은 클러스터링 자체에는 영향 없음 → Step 5/6/7만 재실행)
uv run main.py --step 5 --sample 200000   # 소비태그 레이블링
uv run main.py --step 6 --sample 200000   # archetype 레이블링
uv run main.py --step 7 --sample 200000   # 병합 + 매트릭스
```

low-conf 경고(`margin < 0.05`)가 있는 레이블부터 우선 재합성한다.
