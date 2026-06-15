# Nemotron Korea — 5대 소비가치 선호 분포 추출 파이프라인

## 개요

본 파이프라인은 Nemotron 생성 페르소나 데이터로부터 **5대 소비가치(functional, emotional, social, epistemic, ethical)** 점수를 자동 추출하고, 사전 정의된 군집 중심점과의 유사도 기반으로 **소비자 유형(Type1, Type2)**을 분류하는 전체 워크플로우입니다.

---

## 주요 기능

### 1. **Culinary Persona → 소비가치 점수 변환**

- 식생활 텍스트를 5차원 가치 벡터로 변환
- 앵커 문장 기반 코사인 유사도 계산 → 0~6점 스케일링

### 2. **Hobby → 소비가치 점수 변환 (OpenAI Batch API)**

- 17,000개 유니크 취미를 OpenAI GPT-4o-mini로 라벨링
- 임베딩 맵(embedding + value scores) 구축
- 신규 페르소나의 취미와 기존 DB 간 유사도 매칭

### 3. **최적 가중치 탐색 (Silhouette Score)**

- `a × hobby_score + b × culinary_score` 결합
- K-Means 군집화 품질(Silhouette)을 기준으로 최적 a, b 탐색

### 4. **군집 분류 (Type1 / Type2)**

- 사전 정의된 5개 군집 중심점과 코사인 유사도 계산
- Top-2 군집을 Type1(주성향), Type2(부성향)로 할당
- 임계값 기반 노이즈 필터링 (유사도 < 0.95 제거)

---

## 파이프라인 단계별 요약

```
STEP 1  culinary 임계값   │ 1,000건 샘플 → Percentile 5%/95% 추출 → 스케일러 초기화
STEP 2  hobby 포화도      │ 유니크 취미 증가율 추적 → 전량 추출 → 유사 취미 압축
STEP 3  hobby 라벨링      │ OpenAI Batch API로 17,000개 취미 점수화
STEP 4  hobby 맵 구축     │ 임베딩 + 점수 pkl 저장 → Google Drive 업로드
STEP 5  최적 가중치       │ 5,000건 샘플로 KMeans Silhouette 그리드 탐색
STEP 6  실제 생성         │ 페르소나 데이터로부터 소비가치 추출
STEP 7  군집 분류         │ 코사인 유사도 Top-2 매칭 
```

---

## 실행 흐름

### 1단계: Culinary 스케일러 초기화

```python
# 1,000건 샘플로 임계값 도출
culinary_scaler = CulinaryValueScaler(embed_model, CULINARY_ANCHORS, min_thumb, max_thumb)
```

### 2단계: Hobby 라벨링 (Batch API)

```python
# 유니크 취미 추출 → 유사 취미 압축
repr_hobbies, alias_map = deduplicate_hobbies(all_unique_hobbies, embed_model, 0.85)

# OpenAI Batch 제출
job_log = submit_all_batches(repr_hobbies, OPENAI_API_KEY, subbatch_size, LOCAL_DIR)

# 결과 수집
hobby_df = poll_and_collect_results(job_log, OPENAI_API_KEY, HOBBY_RESULT_CSV)
```

### 3단계: 최적 가중치 탐색

```python
# 5,000건 샘플로 그리드 탐색
# → BEST_A (hobby 가중치), BEST_B (culinary 가중치) 도출
```

### 4단계: 군집 분류

```python
# 코사인 유사도 기반 Top-2 군집 할당
# → outputs/nemotron_cluster_types.csv 생성
```

---

## 결과

### Hobby 압축 효과

- 압축 전: 22,477개 → 압축 후: 9,372개 (압축률 58.3%)
- 유사도 임계값: 0.85

### 가중치 설정 결과

!image.png

- 실루엣 점수 2.8

### 군집 분류 결과 (예시)

- 전체 유저: 8,042건
- 페르소나 매칭 유저: 7,632건 (94.9%)
- 노이즈 제거: 410건 (5.1%)
- 복합 성향 유저(Type2 존재): 7,381건 (96.7%)

### 주성향(Type1) 분포

```
적극적 소비: 3,024건
사회적 소비: 2,183건
소극적 실속: 1,583건
윤리적 소비:   566건
자기 중심적:   276건
```

---