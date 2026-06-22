"""
select_mentors.py — 온라인 파이프라인

실제 데이터 스키마 (이전 작업자 전달):
  mentor_cards.json  : [{uuid, profile{name,basic_info,education}, summary,
                         details{career_goals_and_ambitions, cultural_background,
                                 culinary_persona, hobbies_and_interests_list},
                         labels{archetype(Arch_0~4), primary_tag, secondary_tag,
                                coherence_matrix}}]
  consumption_tags.csv : uuid,primary_tag,secondary_tag,coherence_score(0~2)
  coherence_matrix.json: { "Arch_0": {tag: score, ...}, ... }   # arch → tag

슬롯 구성 (3명, 모두 서로 다른 archetype + 서로 다른 primary_tag):
  최애 1명 : 온보딩 퀴즈로 확정된 고정 멘토 UUID (ctx.regular_mentor_uuids)
  맥락 1명 : 페르소나 임베딩 ANN 검색 상위 후보 중 정합(신뢰도) 높은 멘토
  반대 1명 : 사용자 정합점수 '최저' archetype 중 정합(신뢰도) 높은 멘토
※ 매 요청마다 새로 샘플링. 신뢰도 동점 후보 사이에서는 무작위 → 출력 archetype 랜덤성 확보.
※ primary_tag(소비성향 5종)는 프롬프트 Type(1~5)과 1:1 대응 → primary_tag까지 distinct하게
  뽑아 3명이 항상 서로 다른 Type 가이드를 받도록 보장 (Arch는 같아도 tag 겹칠 수 있어 추가).

정합(coherence)의 두 가지 쓰임:
  ① 매트릭스(arch×tag) × 사용자 태그 → archetype별 정합점수 → '반대' archetype 결정
  ② 멘토별 coherence_score(0~2) = 말·행동 일치 신뢰도
     → 샘플링 시 신뢰도 높은 멘토를 우선 추출 (믿을 수 있는 멘토 우대)

입력: category(필수) + purchase_tags(필수 해시태그) + item_name/price/extra_text/image(선택)
"""
from __future__ import annotations
import json, csv, random, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

def _log(*args):
    print(*args, file=sys.stderr)

BASE_DIR = Path(__file__).parent
MENTOR_CARDS_PATH  = BASE_DIR / "data" / "mentor_cards.json"
CONSUMPTION_CSV    = BASE_DIR / "data" / "consumption_tags.csv"
COHERENCE_MTX_PATH = BASE_DIR / "data" / "consumption_matrix.json"

CONSUMPTION_TAGS = {"소극적 실속", "사회적 소비", "윤리적 소비", "적극적 소비", "자기 중심적"}

# 맥락 멘토: ANN 상위 몇 명까지를 "관련 있는 후보군"으로 보고 그 안에서 신뢰도로 재정렬할지
CONTEXT_RERANK_TOP_K = 5

# ── 구매이유 해시태그 → 소비 성향 태그 매핑 ──────────────────────────────────
PURCHASE_TAG_TO_CONSUMPTION: dict[str, str] = {
    "SNS광고":      "적극적 소비",
    "친구추천":     "사회적 소비",
    "할인중":       "소극적 실속",
    "오래고민중":   "윤리적 소비",
    "그냥갖고싶음": "자기 중심적",
    "품절전에":     "소극적 실속",
    "기타":         "자기 중심적",
}
CATEGORY_TO_TAG: dict[str, str] = {
    "굿즈":     "사회적 소비",
    "패션·잡화":"적극적 소비",
    "게임":     "적극적 소비",
    "문구·다꾸":"자기 중심적",
    "뷰티":     "자기 중심적",
    "간식":     "소극적 실속",
    "도서":     "윤리적 소비",
    "기타":     "자기 중심적",
}
def _price_to_tag(price: Optional[int]) -> Optional[str]:
    if price is None:       return None
    if price < 10_000:      return "소극적 실속"
    if price < 50_000:      return None
    return "적극적 소비"


# 아키타입 코드(Arch_0~4) → 표시 이름. 내부 선택 로직은 코드 키로 비교하고,
# 사용자에게 보여줄 때만 이 이름을 쓴다. (출처: 아키타입 정의 문서)
ARCHETYPE_NAMES: dict[str, str] = {
    "Arch_0": "돌봄형",
    "Arch_1": "창작형",
    "Arch_2": "사색형",
    "Arch_3": "평온형",
    "Arch_4": "향유형",
}


# ── 카드 접근자 (실제 스키마 캡슐화) ─────────────────────────────────────────
def card_archetype(c: dict) -> str:   return c["labels"]["archetype"]
def card_archetype_name(c: dict) -> str:
    """아키타입 코드 → 한국어 표시 이름(돌봄형 등). 미지정 시 코드 그대로."""
    a = card_archetype(c)
    return ARCHETYPE_NAMES.get(a, a)
def card_primary_tag(c: dict) -> str: return c["labels"]["primary_tag"]
def card_name(c: dict) -> str:        return c["profile"]["name"]
def card_basic_info(c: dict) -> str:  return c["profile"].get("basic_info", "")
def card_coherence(c: dict) -> float: return c.get("_coherence", 0.0)

def card_persona_text(c: dict) -> str:
    """임베딩/프롬프트용 페르소나 문장."""
    d = c.get("details", {})
    parts = [
        c.get("summary", ""),
        d.get("career_goals_and_ambitions", ""),
        d.get("cultural_background", ""),
        d.get("culinary_persona", ""),
        " ".join(d.get("hobbies_and_interests_list", [])),
        card_primary_tag(c),
    ]
    return " ".join(p for p in parts if p)


# ── 사용자 컨텍스트 ──────────────────────────────────────────────────────────
@dataclass
class UserContext:
    """앱 화면에서 수집되는 전체 사용자 입력"""
    category:      str           = ""                            # 필수
    purchase_tags: list[str]     = field(default_factory=list)   # 필수 구매이유 해시태그
    item_name:     str           = ""
    price:         Optional[int] = None
    extra_text:    str           = ""                            # 자연어
    image_path:    Optional[str] = None                          # 이미지 (추후 멀티모달)
    regular_mentor_uuids: list[str] = field(default_factory=list)  # 온보딩 퀴즈 결과
    preferred_tags: list[str]    = field(default_factory=list)
    preferred_age:  Optional[str] = None                         # 선호 멘토 연령 (선택)

    @property
    def inferred_tags(self) -> list[str]:
        if self.preferred_tags:
            return self.preferred_tags
        tags: list[str] = []
        for tag in self.purchase_tags:
            mapped = PURCHASE_TAG_TO_CONSUMPTION.get(tag)
            if mapped and mapped not in tags:
                tags.append(mapped)
        cat_tag = CATEGORY_TO_TAG.get(self.category)
        if cat_tag and cat_tag not in tags:
            tags.append(cat_tag)
        price_tag = _price_to_tag(self.price)
        if price_tag and price_tag not in tags:
            tags.append(price_tag)
        return tags if tags else ["자기 중심적"]

    @property
    def context_query(self) -> str:
        parts = []
        if self.purchase_tags: parts.append(" ".join(self.purchase_tags))
        if self.item_name:     parts.append(self.item_name)
        if self.category:      parts.append(self.category)
        if self.extra_text:    parts.append(self.extra_text)
        return " ".join(parts).strip()

    @property
    def llm_context_block(self) -> str:
        lines = ["[아이의 고민]"]
        if self.item_name or self.category or self.price:
            item = self.item_name or "상품"
            cat  = f" ({self.category})" if self.category else ""
            prc  = f" — {self.price:,}원" if self.price else ""
            lines.append(f"상품: {item}{cat}{prc}")
        if self.purchase_tags:
            lines.append(f"구매 이유: {' + '.join(self.purchase_tags)}")
        if self.extra_text:
            lines.append(f"하고 싶은 말: {self.extra_text}")
        return "\n".join(lines)


# ── 데이터 로드 ──────────────────────────────────────────────────────────────
def _load_consumption_scores(path=CONSUMPTION_CSV) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not Path(path).exists():
        return scores
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                scores[row["uuid"]] = float(row["coherence_score"])
            except (KeyError, ValueError):
                continue
    return scores

def load_mentor_cards(path=MENTOR_CARDS_PATH, csv_path=CONSUMPTION_CSV):
    with open(path, encoding="utf-8") as f:
        cards = json.load(f)
    scores = _load_consumption_scores(csv_path)
    for c in cards:
        c["_coherence"] = scores.get(c["uuid"], 0.0)
    return cards

def _normalize_matrix(raw: dict) -> dict[str, dict[str, float]]:
    """어떤 방향/래퍼로 와도 canonical: archetype → {tag: score} 로 변환."""
    m = raw.get("matrix", raw) if isinstance(raw, dict) else raw
    outer = list(m.keys())
    outer_is_tag = any(k in CONSUMPTION_TAGS for k in outer)
    canon: dict[str, dict[str, float]] = {}
    if outer_is_tag:                                  # tag → {arch: score} (전치 필요)
        for tag, inner in m.items():
            for arch, val in inner.items():
                canon.setdefault(arch, {})[tag] = val
    else:                                             # arch → {tag: score} (그대로)
        for arch, inner in m.items():
            canon[arch] = dict(inner)
    return canon

def load_consumption_matrix(path=COHERENCE_MTX_PATH):
    with open(path, encoding="utf-8") as f:
        return _normalize_matrix(json.load(f))


# ── 정합 점수: 사용자 태그 × 매트릭스(arch→tag) → arch별 점수 ──────────────
def _arch_scores(user_tags: list[str], matrix: dict) -> dict[str, float]:
    return {arch: sum(tagmap.get(t, 0.0) for t in user_tags)
            for arch, tagmap in matrix.items()}


# ── 신뢰도 우선 + 동점 무작위 선택 ──────────────────────────────────────────
def _pick_reliable(cands: list[dict]):
    """후보 중 정합점수(말·행동 일치 신뢰도) 최상위 멘토를 우선.
    동점이 여럿이면 그 안에서 무작위 → 매 요청마다 다른 멘토가 뽑히게 함."""
    if not cands:
        return None
    top = max(card_coherence(c) for c in cands)
    return random.choice([c for c in cands if card_coherence(c) == top])


# ── 슬롯: 최애 1명 (온보딩 퀴즈로 확정된 UUID) ──────────────────────────────
def _select_regular(cards, uuids, n=1):
    by_uuid = {c["uuid"]: c for c in cards}
    selected = []
    for u in uuids:
        if len(selected) >= n: break
        card = by_uuid.get(u)
        if card and card not in selected:
            selected.append(card)
    return selected


# ── 슬롯: 반대 1명 (사용자 정합점수 최저 archetype) ─────────────────────────
def _select_opposing(cards, arch_scores, excl_uuid=None, excl_arch=None, excl_tag=None):
    """사용자와 가장 안 맞는(정합점수 최저) archetype을 고르고,
    그 archetype 안에서는 신뢰도(coherence) 높은 멘토를 우선 추출.
    primary_tag(=Type)가 이미 쓰인 것과 겹치지 않는 후보를 우선(1차), 없으면 허용(2차)."""
    excl_uuid = excl_uuid or set()
    excl_arch = excl_arch or set()
    excl_tag  = excl_tag or set()
    cand_arch = [a for a in arch_scores if a not in excl_arch]
    if not cand_arch:
        return None
    cand_arch.sort(key=lambda a: arch_scores[a])      # 최저 정합 archetype부터
    for require_distinct_tag in (True, False):        # tag까지 distinct 우선 → 불가 시 완화
        for arch in cand_arch:
            cands = [c for c in cards
                     if card_archetype(c) == arch and c["uuid"] not in excl_uuid
                     and (not require_distinct_tag or card_primary_tag(c) not in excl_tag)]
            if cands:
                return _pick_reliable(cands)          # 신뢰도 높은 멘토(동점 시 무작위)
    return None


# ── 슬롯: 맥락 1명 (페르소나 임베딩 ANN 검색) ───────────────────────────────
def _select_context(cards, query_text, store=None, n=1,
                    excl_uuid=None, excl_arch=None, excl_tag=None, use_embedding=True):
    """ANN으로 쿼리와 의미가 가까운 멘토를 찾고, 상위 후보군(CONTEXT_RERANK_TOP_K)
    안에서 신뢰도 높은 멘토를 우선 추출. = '관련성 + 신뢰도' 동시 고려.
    archetype·primary_tag(=Type) 모두 이미 쓰인 것과 겹치지 않게 후보를 구성."""
    excl_uuid = excl_uuid or set()
    excl_arch = excl_arch or set()
    excl_tag  = excl_tag or set()
    by_uuid = {c["uuid"]: c for c in cards}

    def _eligible(require_tag):
        return [c for c in cards
                if c["uuid"] not in excl_uuid and card_archetype(c) not in excl_arch
                and (not require_tag or card_primary_tag(c) not in excl_tag)]
    eligible = _eligible(True) or _eligible(False)    # tag distinct 우선, 불가 시 완화
    if not eligible:
        return []
    eligible_uuids = {c["uuid"] for c in eligible}

    if not (use_embedding and store and query_text):
        return _sample_distinct_arch(eligible, n, prefer_reliable=True)

    try:
        hits = store.search(query_text, k=min(len(cards), 200), exclude_uuids=excl_uuid)
        selected = []
        used_arch = set(excl_arch)
        used_tag  = set(excl_tag)
        # ANN 관련도 순서를 유지하며 archetype·primary_tag 모두 서로 다른 후보군 수집
        cand_pool = []
        for uuid, _score in hits:
            card = by_uuid.get(uuid)
            if not card or uuid not in eligible_uuids:
                continue
            if card_archetype(card) in used_arch or card_primary_tag(card) in used_tag:
                continue
            cand_pool.append(card)
            used_arch.add(card_archetype(card)); used_tag.add(card_primary_tag(card))
            if len(cand_pool) >= CONTEXT_RERANK_TOP_K + n:
                break
        # 후보군을 n명 뽑을 때마다 '상위 K + 남은 슬롯' 범위에서 신뢰도로 재정렬
        while cand_pool and len(selected) < n:
            window = cand_pool[:CONTEXT_RERANK_TOP_K]
            pick = _pick_reliable(window)
            selected.append(pick)
            cand_pool.remove(pick)
        if len(selected) < n:                         # ANN 결과 부족 시 보충
            picked_arch = {card_archetype(c) for c in selected}
            picked_tag  = {card_primary_tag(c) for c in selected}
            for c in eligible:
                if len(selected) >= n: break
                if c in selected: continue
                if card_archetype(c) in picked_arch or card_primary_tag(c) in picked_tag:
                    continue
                selected.append(c)
                picked_arch.add(card_archetype(c)); picked_tag.add(card_primary_tag(c))
        return selected
    except Exception as e:
        _log(f"  [ANN 검색 실패, 랜덤 fallback] {e}")
        return _sample_distinct_arch(eligible, n, prefer_reliable=True)


def _sample_distinct_arch(cards, n, prefer_reliable=False):
    """가능한 한 서로 다른 archetype·primary_tag(=Type)로 n명 샘플링.
    1차: arch·tag 모두 distinct → 2차: arch만 distinct → 3차: 아무나.
    prefer_reliable=True면 신뢰도 높은 멘토를 우선(동점은 무작위)."""
    shuffled = cards[:]
    random.shuffle(shuffled)
    if prefer_reliable:
        # 무작위 셔플 후 신뢰도 내림차순 안정정렬 → 동일 신뢰도 안에서는 무작위 유지
        shuffled.sort(key=card_coherence, reverse=True)
    selected, used_arch, used_tag = [], set(), set()
    for c in shuffled:                                # 1차: archetype + primary_tag 모두 distinct
        if len(selected) >= n: break
        if card_archetype(c) in used_arch or card_primary_tag(c) in used_tag: continue
        selected.append(c); used_arch.add(card_archetype(c)); used_tag.add(card_primary_tag(c))
    for c in shuffled:                                # 2차: archetype만 distinct
        if len(selected) >= n: break
        if c in selected or card_archetype(c) in used_arch: continue
        selected.append(c); used_arch.add(card_archetype(c))
    for c in shuffled:                                # 3차: 아무나
        if len(selected) >= n: break
        if c not in selected:
            selected.append(c)
    return selected


# ── 메인 함수 ────────────────────────────────────────────────────────────────
def select_mentors(
    ctx: UserContext,
    cards: list[dict] | None = None,
    matrix: dict | None = None,
    store=None,
    use_embedding: bool = True,
) -> list[dict]:
    """UserContext → 멘토 3명 카드: [최애, 맥락, 반대] (모두 서로 다른 archetype)

    선택 순서: 최애(고정) → 반대(정합 최저 archetype 확정) → 맥락(나머지 중 ANN).
    반대를 맥락보다 먼저 확정해 '사용자와 가장 안 맞는 archetype'이 항상 보장되게 함.
    반환 순서는 표시 편의상 [최애, 맥락, 반대].
    """
    if cards  is None: cards  = load_mentor_cards()
    if matrix is None: matrix = load_consumption_matrix()
    pool = cards

    tags = ctx.inferred_tags
    arch_scores = _arch_scores(tags, matrix)
    _log(f"  [태그 추론] {ctx.purchase_tags} → {tags}")
    _log(f"  [아키타입 점수] { {k: v for k, v in sorted(arch_scores.items(), key=lambda x: -x[1])} }")

    excl_uuid: set[str] = set()
    excl_arch: set[str] = set()
    excl_tag:  set[str] = set()            # 이미 쓰인 primary_tag(=Type) → 중복 회피

    # 1) 최애 1명 — 온보딩 퀴즈 고정 UUID (없으면 신뢰도 높은 멘토로 보충)
    regular = _select_regular(pool, ctx.regular_mentor_uuids, n=1)
    if not regular:
        _log("  [최애 없음] 온보딩 UUID 미지정 → 사용자 정합 상위 archetype의 신뢰도 높은 멘토로 보충")
        for arch in sorted(arch_scores, key=lambda a: arch_scores[a], reverse=True):
            cands = [c for c in pool if card_archetype(c) == arch]
            pick = _pick_reliable(cands)
            if pick:
                regular = [pick]; break
    excl_uuid.update(c["uuid"] for c in regular)
    excl_arch.update(card_archetype(c) for c in regular)
    excl_tag.update(card_primary_tag(c) for c in regular)

    # 2) 반대 1명 — 사용자 정합점수 최저 archetype 중 신뢰도 높은 멘토
    opp = _select_opposing(pool, arch_scores, excl_uuid=excl_uuid,
                           excl_arch=excl_arch, excl_tag=excl_tag)
    if opp:
        excl_uuid.add(opp["uuid"]); excl_arch.add(card_archetype(opp))
        excl_tag.add(card_primary_tag(opp))

    # 3) 맥락 1명 — 페르소나 임베딩 ANN 상위 후보 중 신뢰도 높은 멘토
    context = _select_context(pool, ctx.context_query, store=store, n=1,
                              excl_uuid=excl_uuid, excl_arch=excl_arch,
                              excl_tag=excl_tag, use_embedding=use_embedding)
    excl_uuid.update(c["uuid"] for c in context)
    excl_arch.update(card_archetype(c) for c in context)
    excl_tag.update(card_primary_tag(c) for c in context)

    # 표시 순서: 최애 → 맥락 → 반대
    selected = regular + context + ([opp] if opp else [])

    if len(selected) < 3:                             # 미달 보충 (UUID·archetype·tag 중복 최소화)
        used_arch = {card_archetype(c) for c in selected}
        used_tag  = {card_primary_tag(c) for c in selected}
        rest = [c for c in pool if c["uuid"] not in excl_uuid]
        rest.sort(key=card_coherence, reverse=True)   # 신뢰도 높은 멘토 우선 보충
        # 1차: arch·tag 모두 distinct → 2차: arch만 distinct → 3차: 아무나
        for stage in ("arch_tag", "arch", "any"):
            for c in rest:
                if len(selected) >= 3: break
                if c["uuid"] in excl_uuid: continue
                if stage in ("arch_tag", "arch") and card_archetype(c) in used_arch: continue
                if stage == "arch_tag" and card_primary_tag(c) in used_tag: continue
                selected.append(c); excl_uuid.add(c["uuid"])
                used_arch.add(card_archetype(c)); used_tag.add(card_primary_tag(c))
            if len(selected) >= 3: break

    return selected[:3]


# ── 테스트 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from vector_store import load_store

    cards  = load_mentor_cards()
    matrix = load_consumption_matrix()
    store  = load_store()
    if store is None:
        _log("[경고] 벡터 인덱스 없음 → 'python vector_store.py'로 먼저 빌드. 맥락은 랜덤 폴백.")

    reg_uuids = [cards[0]["uuid"]]                      # 데모용 최애 1명
    ctx = UserContext(
        category="굿즈",
        purchase_tags=["SNS광고", "친구추천"],
        item_name="캐릭터 굿즈 세트", price=18000,
        extra_text="친구가 한정판이래, 다음주에 영화도 보고 싶은데…",
        regular_mentor_uuids=reg_uuids,
    )
    print(f"inferred_tags : {ctx.inferred_tags}")
    print(f"context_query : {ctx.context_query[:60]}…")

    result = select_mentors(ctx, cards=cards, matrix=matrix,
                            store=store, use_embedding=store is not None)
    labels = ["최애", "맥락", "반대"]
    print("\n선택된 멘토 3명:")
    for label, c in zip(labels, result):
        print(f"  [{label}] {card_name(c)} ({card_basic_info(c)}) | "
              f"{card_archetype(c)} | {card_primary_tag(c)} | coh={card_coherence(c)}")
