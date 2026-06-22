"""
generate_responses.py — (LLM 오케스트레이션)
LLM_BACKEND = "mock" | "openai" | "anthropic"  (.env에서 로드)

다양성 평가 전략:
  1차 (임베딩 기반) : 멘토 응답 쌍별 코사인 거리 평균 → DIVERSITY_THRESHOLD 미만이면 재시도
  2차 (Self-BLEU)  : sentence-transformers 없을 때 경량 fallback
"""
from __future__ import annotations
import asyncio, os, math, sys, time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np
from select_mentors import card_archetype, card_primary_tag, card_name
from prompt_builder import build_system_prompt, build_user_prompt, card_type

# ── .env 로드 ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

LLM_BACKEND: Literal["openai", "anthropic", "mock"] = os.getenv("LLM_BACKEND", "mock")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o")   # 상위 모델 기본값(.env로 override 가능)
ANTHROPIC_MODEL   = "claude-opus-4-6"

MIN_CHARS            = 20    # 규격: <Appropriate> 두 문장 / 이탈 시 한 문장 → 짧을 수 있음
MAX_CHARS            = 300
MAX_RETRY            = 1    # 1 = 무효 슬롯(LLM 에러·빈 응답·길이 탈락)만 한 번 더 호출.
                            #     다양성 미달도 1회까지 재생성. 프롬프트 단 차별화로 첫 응답이
                            #     대체로 충분히 다양하므로 재시도는 '실패 복구' 안전망 위주로 동작
                            #     (유효 응답 2개 미만 → 다양성 0.000 으로 박제되던 문제 완화).
# 다양성 임계값: 쌍별 코사인 거리 평균이 이 값 미만이면 "너무 비슷하다" 판단 (측정용)
DIVERSITY_THRESHOLD  = 0.25   # 0 = 완전 동일, 1 = 완전 다름 (거리이므로 낮을수록 유사)

EMBED_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
_embed_model = None

# ── 프롬프트 콘텐츠 = prompt_builder.py ──────────────────────
#    Type 1~5 톤가이드·페르소나·스캐폴딩 규칙은 모두 prompt_builder에 있음.

# ── 데이터 클래스 ─────────────────────────────────────────────────────────────
@dataclass
class MentorResponse:
    uuid:          str
    name:          str
    archetype:     str            # Arch_0~4
    primary_tag:   str
    response_text: str
    token_count:   int   = 0
    latency_s:     float = 0.0   # 이 멘토 한 명의 LLM 호출에 걸린 시간(초)
    is_valid:      bool  = True
    issues:        list  = field(default_factory=list)

@dataclass
class DiversityReport:
    """다양성 평가 결과"""
    method:          str              # "embedding" | "self_bleu"
    pairwise_scores: list[float]      # 쌍별 거리 or (1 - BLEU) 값
    mean_score:      float            # 평균 다양성 점수
    passed:          bool             # DIVERSITY_THRESHOLD 통과 여부
    worst_pair:      tuple[int, int]  # 가장 유사한 쌍의 인덱스

    def summary(self) -> str:
        status = "✅ 통과" if self.passed else "⚠️ 임계값 미달"
        return (f"[다양성 {status}] 방법={self.method}  "
                f"평균거리={self.mean_score:.3f}  "
                f"임계값={DIVERSITY_THRESHOLD}  "
                f"최유사쌍=({self.worst_pair[0]},{self.worst_pair[1]})")


# ── 임베딩 유틸 ──────────────────────────────────────────────────────────────
def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def _embed_texts(texts: list[str]) -> np.ndarray:
    return _get_embed_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# ── 다양성 평가 ──────────────────────────────────────────────────────────────

def _degenerate_report(method: str) -> DiversityReport:
    """응답이 2개 미만이라 쌍을 만들 수 없을 때의 안전 리포트."""
    return DiversityReport(method=method, pairwise_scores=[],
                           mean_score=0.0, passed=False, worst_pair=(0, 0))


def _diversity_embedding(responses: list[MentorResponse]) -> DiversityReport:
    """
    방법 ①: 임베딩 코사인 거리
    정규화된 벡터끼리의 거리 = 1 - cosine_similarity
    쌍별 평균이 DIVERSITY_THRESHOLD 이상이어야 통과
    """
    if len(responses) < 2:
        return _degenerate_report("embedding")
    texts = [r.response_text for r in responses]
    embs  = _embed_texts(texts)          # (N, dim), 이미 L2 정규화됨

    n = len(embs)
    pairs, scores = [], []
    for i in range(n):
        for j in range(i + 1, n):
            sim  = float(np.dot(embs[i], embs[j]))
            dist = 1.0 - sim              # 거리 (0=동일, 2=반대)
            scores.append(dist)
            pairs.append((i, j))

    mean_score = float(np.mean(scores)) if scores else 1.0
    worst_idx  = int(np.argmin(scores)) if scores else 0  # 거리 최소 = 가장 유사한 쌍

    return DiversityReport(
        method="embedding",
        pairwise_scores=scores,
        mean_score=mean_score,
        passed=mean_score >= DIVERSITY_THRESHOLD,
        worst_pair=pairs[worst_idx],
    )


def _self_bleu_score(responses: list[MentorResponse]) -> DiversityReport:
    """
    방법 ②: Self-BLEU (어휘 겹침 기반, 경량 fallback)
    각 응답을 가설(hypothesis)로, 나머지를 참조(reference)로 삼아 BLEU 계산.
    다양성 점수 = 1 - mean(self_bleu) → 높을수록 다양함.
    """
    if len(responses) < 2:
        return _degenerate_report("self_bleu")

    def tokenize(text: str) -> list[str]:
        return text.split()

    def ngram_counts(tokens: list[str], n: int) -> Counter:
        return Counter(zip(*[tokens[i:] for i in range(n)]))

    def bleu1(hypothesis: list[str], references: list[list[str]]) -> float:
        """단순 1-gram precision (brevity penalty 생략)"""
        if not hypothesis:
            return 0.0
        ref_ngrams = Counter()
        for ref in references:
            ref_ngrams |= ngram_counts(ref, 1)
        hyp_ngrams = ngram_counts(hypothesis, 1)
        clipped = sum(min(cnt, ref_ngrams[ng]) for ng, cnt in hyp_ngrams.items())
        return clipped / len(hypothesis)

    tokens_list = [tokenize(r.response_text) for r in responses]
    self_bleus  = []
    pairs, dist_scores = [], []

    for i, hyp in enumerate(tokens_list):
        refs = [t for j, t in enumerate(tokens_list) if j != i]
        sb   = bleu1(hyp, refs)
        self_bleus.append(sb)

    # 쌍별 점수로 변환 (1 - bleu를 거리로 사용)
    n = len(responses)
    for i in range(n):
        for j in range(i + 1, n):
            dist = 1.0 - bleu1(tokens_list[i], [tokens_list[j]])
            dist_scores.append(dist)
            pairs.append((i, j))

    mean_score = float(np.mean(dist_scores)) if dist_scores else 1.0
    worst_idx  = int(np.argmin(dist_scores)) if dist_scores else 0

    return DiversityReport(
        method="self_bleu",
        pairwise_scores=dist_scores,
        mean_score=mean_score,
        passed=mean_score >= DIVERSITY_THRESHOLD,
        worst_pair=pairs[worst_idx],
    )


def evaluate_diversity(responses: list[MentorResponse]) -> DiversityReport:
    """임베딩 우선, 실패하면 Self-BLEU fallback"""
    try:
        return _diversity_embedding(responses)
    except Exception:
        return _self_bleu_score(responses)


# ── LLM 호출 (프롬프트는 prompt_builder가 조립) ──────────────────────────────
# mock: Type 1~5 별 <Appropriate> 2문장 스캐폴딩 샘플 (API 없이 흐름·형식 검증)
_MOCK = {
    1: "민지야, 다이소 진열대 앞에서 꽤 오래 고민하고 있구나? 🛍 오늘 미리 계획해둔 용돈 선 안에서 고른다면 어떤 게 딱 맞을 것 같아?",
    2: "옆에 서연이랑 같이 구경하니까 더 신나 보인다! 👭 둘이 함께 나눠 쓰거나 세트로 맞췄을 때 제일 마음이 따뜻해지는 건 뭘까?",
    3: "필통에 이미 친구들이 많은데 또 마음이 가는 게 있나 보네 ✏️ 이게 오랫동안 네 곁에서 진짜 쓸모 있게 남을 물건일지 한번 떠올려볼래?",
    4: "학원 가기 전 남은 시간이랑 토스 잔고를 딱 계산해 보고 있구나 📊 후회 없는 최고의 선택을 하려면 지금 뭘 고르는 게 제일 똑똑할까?",
    5: "남들 다 사는 거 말고 네 눈에 유독 반짝이는 게 있나 보다 🎨 그게 정말 네 방에 뒀을 때 너만의 분위기를 살려줄 특별한 아이템일까?",
}

async def _call_mock(sys_p: str, user_p: str, type_no: int = 1) -> tuple[str, int]:
    await asyncio.sleep(0.05)
    return _MOCK.get(type_no, _MOCK[1]), 0

OPENAI_MAX_TOKENS = 512   # 신모델(GPT-5 계열)은 추론 토큰이 예산을 잠식 → 넉넉히

async def _call_openai(sys_p: str, user_p: str) -> tuple[str, int]:
    from openai import AsyncOpenAI
    messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]
    # 신모델(GPT-5 계열)은 'max_tokens' 대신 'max_completion_tokens'를 요구. (gpt-4o도 호환)
    kwargs = dict(model=OPENAI_MODEL, messages=messages,
                  max_completion_tokens=OPENAI_MAX_TOKENS)
    # async with: 루프 종료 전에 HTTP 연결을 정리 → "Event loop is closed" 방지
    async with AsyncOpenAI(api_key=OPENAI_API_KEY) as client:
        try:
            resp = await client.chat.completions.create(temperature=0.9, **kwargs)
        except Exception as e:
            # 일부 신모델은 temperature 커스텀(0.9) 미지원 → 기본값(1)으로 재시도
            if "temperature" in str(e).lower():
                resp = await client.chat.completions.create(**kwargs)
            else:
                raise
    return resp.choices[0].message.content or "", resp.usage.total_tokens if resp.usage else 0

async def _call_anthropic(sys_p: str, user_p: str) -> tuple[str, int]:
    import anthropic
    async with anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY) as client:
        resp = await client.messages.create(
            model=ANTHROPIC_MODEL, system=sys_p,
            messages=[{"role": "user", "content": user_p}],
            temperature=0.9, max_tokens=300,
        )
    text = resp.content[0].text if resp.content else ""
    return text, resp.usage.input_tokens + resp.usage.output_tokens

async def _call_llm(sys_p: str, user_p: str, type_no: int = 1) -> tuple[str, int]:
    if LLM_BACKEND == "openai":    return await _call_openai(sys_p, user_p)
    if LLM_BACKEND == "anthropic": return await _call_anthropic(sys_p, user_p)
    return await _call_mock(sys_p, user_p, type_no)

async def _timed_call(sys_p: str, user_p: str, type_no: int = 1) -> tuple[str, int, float]:
    """_call_llm을 감싸 멘토 1명 호출에 걸린 시간(초)을 함께 반환."""
    t0 = time.perf_counter()
    text, tokens = await _call_llm(sys_p, user_p, type_no)
    return text, tokens, time.perf_counter() - t0


# ── 후처리 ───────────────────────────────────────────────────────────────────
def _postprocess(text: str, card: dict) -> MentorResponse:
    issues = []
    if len(text) < MIN_CHARS:
        issues.append(f"너무 짧음({len(text)}자)")
    if len(text) > MAX_CHARS:
        trunc = text[:MAX_CHARS]
        last  = max(trunc.rfind("。"), trunc.rfind("!"), trunc.rfind("?"), trunc.rfind("."))
        text  = trunc[:last + 1] if last > MIN_CHARS else trunc
    return MentorResponse(
        uuid=card["uuid"], name=card_name(card),
        archetype=card_archetype(card), primary_tag=card_primary_tag(card),
        response_text=text, is_valid=len(issues) == 0, issues=issues,
    )


# ── 메인 함수 ────────────────────────────────────────────────────────────────
async def generate_responses_async(
    cards: list[dict],
    scenario: dict,
    profile: dict,
) -> tuple[list[MentorResponse], DiversityReport]:
    """
    멘토 N명(현재 3명) 병렬 LLM 호출 → 후처리 → 다양성 평가 → 미달 시 재시도.
    각 멘토는 prompt_builder로 조립한 시스템 프롬프트(자기 Type 가이드)를 받는다.

    Returns
    -------
    (responses, diversity_report)
    """
    responses: list[MentorResponse | None] = [None] * len(cards)

    for attempt in range(MAX_RETRY + 1):
        # 유효하지 않은 슬롯만 재호출
        to_call = [i for i, r in enumerate(responses) if r is None or not r.is_valid]
        tasks   = [
            _timed_call(build_system_prompt(cards[i], scenario, profile),
                        build_user_prompt(scenario, profile),
                        card_type(cards[i]))
            for i in to_call
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in zip(to_call, results):
            if isinstance(result, Exception):
                print(f"  [LLM 호출 실패] {card_name(cards[idx])}: {result!r}", file=sys.stderr)
                responses[idx] = MentorResponse(
                    uuid=cards[idx]["uuid"], name=card_name(cards[idx]),
                    archetype=card_archetype(cards[idx]), primary_tag=card_primary_tag(cards[idx]),
                    response_text="[응답 실패]", is_valid=False, issues=[str(result)],
                )
            else:
                text, tokens, latency = result
                mr = _postprocess(text, cards[idx])
                mr.token_count = tokens
                mr.latency_s   = latency
                responses[idx] = mr

        valid = [r for r in responses if r and r.is_valid]

        # ── 다양성 평가 ──────────────────────────────────────────────────────
        if len(valid) == len(cards):
            report = evaluate_diversity(valid)
            print(f"  {report.summary()}", file=sys.stderr)

            if report.passed or attempt == MAX_RETRY:
                break

            # 미달: 가장 유사한 쌍 중 나중 인덱스 응답을 무효화 → 재호출
            _, worse_idx = report.worst_pair
            responses[worse_idx] = None
            print(f"  → 슬롯 {worse_idx} ({card_name(cards[worse_idx])}) 재생성", file=sys.stderr)
        else:
            report = DiversityReport(
                method="skipped", pairwise_scores=[], mean_score=0.0,
                passed=False, worst_pair=(0, 0),
            )

    final     = [r for r in responses if r is not None]
    final_rpt = evaluate_diversity([r for r in final if r.is_valid]) if final else report
    return final, final_rpt


def generate_responses(
    cards: list[dict],
    scenario: dict,
    profile: dict,
) -> tuple[list[MentorResponse], DiversityReport]:
    """동기 래퍼"""
    return asyncio.run(generate_responses_async(cards, scenario, profile))
