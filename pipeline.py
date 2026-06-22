"""
pipeline.py — end-to-end 진입점

실행 예:
  python pipeline.py \
      --category "굿즈" --tag SNS광고 --tag 친구추천 \
      --item "캐릭터 굿즈 세트" --price 18000 \
      --regular 287c832a-... --regular ffd23254-... \
      --backend mock

필수: --category, --tag(1개 이상)
선택: --item, --price, --extra, --image, --regular(최애 UUID), --age
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from select_mentors import (
    UserContext, load_mentor_cards, load_consumption_matrix, select_mentors,
    card_name, card_basic_info, card_archetype, card_archetype_name, card_primary_tag,
)
from generate_responses import generate_responses
from vector_store import load_store

MENTOR_CARDS_PATH  = Path(__file__).parent / "data" / "mentor_cards.json"
CONSUMPTION_MTX_PATH = Path(__file__).parent / "data" / "consumption_matrix.json"
REGULARS_PATH      = Path(__file__).parent / "data" / "onboarding_regulars.json"
SCENARIOS_PATH     = Path(__file__).parent / "data" / "scenarios.json"
SLOT_LABELS = ["최애", "맥락", "반대"]

# 수동 실행(시나리오 미지정) 시 쓰는 기본 유저 프로필
DEFAULT_PROFILE = {"name": "유저", "age_grade": "초등학생", "atomic_memories": []}

def load_default_regulars(path=REGULARS_PATH) -> list[str]:
    """온보딩 퀴즈 결과(최애 UUID)를 파일에서 로드. --regular 미지정 시 기본값."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("regular_mentor_uuids", [])

def load_scenarios(path=SCENARIOS_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_scenario(data: dict, key) -> dict | None:
    """key = 정수(1~7) 또는 '1A'/'1B'(파생). 못 찾으면 None."""
    for s in data.get("scenarios", []):
        if str(s.get("id")) == str(key):
            return s
    for s in data.get("derived_from_scenario_1", []):
        if str(s.get("id")) == str(key):
            return s
    return None

def scenario_to_context(sc: dict, regulars: list[str]) -> "UserContext":
    """시나리오 select_hint → 멘토 3명 선택용 UserContext."""
    hint = sc.get("select_hint", {})
    return UserContext(
        category=hint.get("category", ""),
        purchase_tags=hint.get("tags", []),
        item_name=hint.get("item", ""),
        price=hint.get("price"),
        regular_mentor_uuids=regulars,
    )

def context_to_scenario(ctx: "UserContext") -> dict:
    """수동 모드: CLI 입력을 최소 동적 컨텍스트로 변환 (시연 시나리오 미사용 시)."""
    return {
        "geofence_zone_name": ctx.item_name or ctx.category or "매장",
        "current_datetime": "", "stay_time": "잠시 고민 중",
        "next_schedule_info": "다음 일정", "spare_time": "약간",
        "user_balance": f"{ctx.price:,}원" if ctx.price else "약간의 용돈",
        "weekly_allowance": "용돈", "memory_stream": ctx.extra_text or "",
    }

def print_result(ctx, scenario, selected, responses, report, elapsed):
    import generate_responses as gr
    print("\n" + "=" * 64)
    print("  D-fine 멘토 스캐폴딩 응답 결과")
    print("=" * 64)
    if scenario.get("title"):
        print(f"시나리오: {scenario['title']}")
    print(f"상황: {scenario.get('geofence_zone_name','')}")
    print(f"시각: {scenario.get('current_datetime','')}  |  잔고: {scenario.get('user_balance','')}")
    if scenario.get("user_utterance"):
        print(f"유저 발화: \"{scenario['user_utterance']}\"")
    print(f"소요 시간: {elapsed:.2f}s  |  백엔드: {gr.LLM_BACKEND}")
    print(f"다양성 점수: {report.mean_score:.3f} ({report.method})  {'✅' if report.passed else '⚠️'}")
    print("-" * 64)
    for label, card, resp in zip(SLOT_LABELS, selected, responses):
        print(f"\n[{label}] {card_name(card)} ({card_basic_info(card)})  —  "
              f"{card_archetype_name(card)} · {card_primary_tag(card)}")
        print(f"  {resp.response_text}")
        if resp.issues: print(f"  ⚠️  이슈: {resp.issues}")
    print("\n" + "=" * 64)

def run_pipeline(ctx: UserContext, scenario: dict, profile: dict,
                 backend="mock", use_embedding=False, output_json=False):
    import generate_responses as gr
    from prompt_builder import card_type, TYPE_GUIDES
    gr.LLM_BACKEND = backend
    cards  = load_mentor_cards(MENTOR_CARDS_PATH)
    matrix = load_consumption_matrix(CONSUMPTION_MTX_PATH)
    store  = load_store() if use_embedding else None
    if use_embedding and store is None:
        print("  [경고] 벡터 인덱스 없음 → 'python vector_store.py'로 빌드 필요. 맥락은 랜덤 폴백.")
    t0 = time.perf_counter()
    selected          = select_mentors(ctx, cards=cards, matrix=matrix,
                                        store=store, use_embedding=use_embedding)
    responses, report = generate_responses(selected, scenario, profile)
    elapsed           = time.perf_counter() - t0
    if output_json:
        result = {
            "scenario": {
                "id": scenario.get("id"), "title": scenario.get("title"),
                "geofence": scenario.get("geofence_zone_name"),
                "datetime": scenario.get("current_datetime"),
                "user_balance": scenario.get("user_balance"),
                "user_utterance": scenario.get("user_utterance"),
            },
            "diversity": {
                "method": report.method, "mean_score": round(report.mean_score, 4),
                "passed": report.passed, "threshold": 0.25,
            },
            "mentors": [
                {"slot": l.strip(), "uuid": c["uuid"], "name": r.name,
                 "archetype": card_archetype_name(c), "basic_info": card_basic_info(c),
                 "primary_tag": r.primary_tag,
                 "type_no": card_type(c), "type_name": TYPE_GUIDES[card_type(c)]["name"],
                 "response": r.response_text, "valid": r.is_valid}
                for l, c, r in zip(SLOT_LABELS, selected, responses)
            ]
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(ctx, scenario, selected, responses, report, elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D-fine 파이프라인")
    parser.add_argument("--scenario", default=None,
                        help="시연 시나리오 번호(1~7) 또는 파생(1A/1B). 지정 시 category/tag 불필요")
    parser.add_argument("--category", default=None, help="상품 카테고리 (수동 모드 필수)")
    parser.add_argument("--tag",      action="append", dest="tags", default=None,
                        help="구매이유 해시태그 (수동 모드 1개 이상)")
    parser.add_argument("--item",     default="")
    parser.add_argument("--price",    type=int, default=None)
    parser.add_argument("--extra",    default="")
    parser.add_argument("--image",    default=None, help="이미지 경로 (선택)")
    parser.add_argument("--regular",  action="append", dest="regulars", default=None,
                        help="최애 멘토 UUID (온보딩 퀴즈 결과). 여러 개 줘도 첫 번째 1명만 사용")
    parser.add_argument("--age",      default=None)
    parser.add_argument("--backend",  choices=["openai","anthropic","mock"], default="mock")
    parser.add_argument("--embedding",action="store_true", default=False,
                        help="맥락 멘토를 벡터 ANN으로 선택 (기본: 비활성=랜덤)")
    parser.add_argument("--json",     action="store_true", default=False)
    args = parser.parse_args()

    regulars = args.regulars or load_default_regulars()

    if args.scenario:                                  # ── 시연 시나리오 모드 ──
        sdata    = load_scenarios()
        scenario = get_scenario(sdata, args.scenario)
        if scenario is None:
            sys.exit(f"[오류] 시나리오 '{args.scenario}' 없음. (1~7 또는 1A/1B)")
        profile  = sdata.get("static_profile", DEFAULT_PROFILE)
        ctx      = scenario_to_context(scenario, regulars)
    else:                                              # ── 수동 모드 ──
        if not args.category or not args.tags:
            sys.exit("[오류] 수동 모드에선 --category 와 --tag(1개 이상)가 필수. "
                     "또는 --scenario N 사용.")
        ctx = UserContext(
            category=args.category, purchase_tags=args.tags,
            item_name=args.item, price=args.price,
            extra_text=args.extra, image_path=args.image,
            regular_mentor_uuids=regulars, preferred_age=args.age,
        )
        profile  = DEFAULT_PROFILE
        scenario = context_to_scenario(ctx)

    run_pipeline(ctx, scenario, profile,
                 backend=args.backend, use_embedding=args.embedding, output_json=args.json)
