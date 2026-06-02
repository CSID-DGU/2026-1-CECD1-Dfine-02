# D-fine — 멘토 3인 오케스트레이션 파이프라인

사용자의 소비 상황을 입력받아 **서로 다른 관점의 멘토 3인**을 선택하고, 각자의 페르소나로
응답을 병렬 생성한 뒤 **응답 다양성**을 평가하는 온라인 파이프라인.

## 멘토 3인 구성

| 슬롯 | 선택 방식 |
|------|-----------|
| **최애** | 온보딩 퀴즈로 고정된 멘토 UUID |
| **맥락** | 사용자 상황과 의미적으로 가까운 멘토 (FAISS ANN 검색) |
| **반대** | 정합 점수가 가장 낮은 아키타입 (관점 다양성 확보) |

3인은 항상 **서로 다른 아키타입 + 서로 다른 소비 유형(Type 1~5)**으로 선택된다.

## 파이프라인 흐름

```
사용자 상황 입력
   → select_mentors  (최애·맥락·반대 3인 선택)
   → generate_responses  (3인 병렬 LLM 호출 + 후처리)
   → 다양성 평가 (KR-SBERT 쌍별 코사인 거리)  ── 미달 시 최대 1회 재생성
   → 결과 출력
```

## 디렉토리 구성

```
pipeline.py               # 진입점 (CLI)
select_mentors.py         # 멘토 3인 선택 로직
generate_responses.py     # 병렬 LLM 응답 생성 + 다양성 평가/재생성
prompt_builder.py         # 시스템 프롬프트 조립 (Type 1~5 가이드)
vector_store.py           # FAISS 인덱스 로드 (맥락 멘토 ANN 검색)
evaluate.py               # 시나리오 전체 응답시간/다양성 평가
ablation_embedding.py     # 임베딩 모델 3종 교차검증(ablation)
make_diagram.py           # 파이프라인 다이어그램 생성

build_faiss.py            # [데이터 준비] FAISS 인덱스 생성 (1회용)
build_consumption_meta.py # [데이터 준비] 소비성향 메타 parquet 생성 (1회용)

data/
  scenarios.json            # 시연 시나리오 (1~7, 파생 1A/1B)
  consumption_matrix.json   # 소비 유형 매트릭스
  consumption_tags.csv      # 소비성향 태그 원본
  onboarding_regulars.json  # 온보딩 최애 멘토 UUID
```

## 설치

```bash
pip install -r requirements.txt
```

`mock` 백엔드만 쓸 거면 `numpy`만 있어도 동작한다. 임베딩 기반 다양성 평가·맥락 멘토
ANN 검색을 쓰려면 `sentence-transformers`, `faiss-cpu`가 필요하다(미설치 시 자동 폴백).

## 환경 변수 (`.env`)

```
LLM_BACKEND=mock           # mock | openai | anthropic
OPENAI_API_KEY=...         # --backend openai 사용 시
OPENAI_MODEL=gpt-4o        # (선택) 모델 override
```

> `.env`는 저장소에 포함되지 않는다. 직접 생성할 것.

## 실행

```bash
# 시연 시나리오 모드 (1~7, 파생 1A/1B)
python pipeline.py --scenario 1 --backend mock

# 멀티턴 파생 시나리오
python pipeline.py --scenario 1A --backend mock

# 맥락 멘토를 벡터 ANN으로 선택 (인덱스 필요)
python pipeline.py --scenario 1 --embedding

# 수동 모드
python pipeline.py --category 굿즈 --tag SNS광고 --tag 친구추천 \
    --item "캐릭터 굿즈 세트" --price 18000

# JSON 출력
python pipeline.py --scenario 1 --json
```

## 평가

```bash
# 시나리오 전체 응답시간 + 다양성 집계
python evaluate.py

# 임베딩 모델 3종 ablation
python ablation_embedding.py
```

## 데이터 / 인덱스 안내

용량 문제로 아래 파일들은 저장소에서 **제외**되어 있다. 실행 전 별도로 준비해야 한다.

- `nemotron_raw.parquet` — 멘토 원본 데이터
- `data/mentor_cards.json` — 멘토 카드
- `data/mentor_index.faiss`, `data/mentor_index_meta.json` — 맥락 멘토 FAISS 인덱스
- `faiss_persona_meta.parquet`, `faiss_consumption_meta.parquet`

FAISS 인덱스는 `build_faiss.py` / `build_consumption_meta.py`로 재생성할 수 있다.
인덱스가 없으면 맥락 멘토는 random 폴백, 다양성 평가는 self_bleu 폴백으로 동작한다.
