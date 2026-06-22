"""
evaluate.py — D-fine 멘토 파이프라인 평가지표 측정 하니스

평가지표 2종:
  1. 응답 다양성 : 멘토 3명 응답의 쌍별 거리 평균
                   (임베딩 코사인 거리 우선, sentence-transformers 없으면 Self-BLEU 폴백)
                   = 0(완전 동일) ~ 1+(완전 다름). 높을수록 좋음. 임계값 0.25.
  2. 응답 시간   : 한 번의 요청에 걸리는 시간(초)
                   - t_select   : 멘토 3명 선택(샘플링/ANN)
                   - t_generate : 3명 LLM 응답 생성까지의 wall-clock(병렬) = 사용자 체감 시간
                   - t_total    : 선택 + 생성
                   - mentor     : 멘토 1명당 LLM 호출 시간(병렬이므로 t_generate ≈ max)

여러 시나리오 × 반복 실행에 걸쳐 평균/중앙값(p50)/p95/min/max로 집계한다.

실행 예:
  python evaluate.py --backend mock --repeat 5
  python evaluate.py --backend openai --embedding --repeat 3        # 실측 권장(VSCode 환경)
  python evaluate.py --scenario 1 --backend openai --embedding --repeat 10
  python evaluate.py --backend mock --repeat 5 --json eval_result.json
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import generate_responses as gr
from select_mentors import UserContext, load_mentor_cards, load_consumption_matrix, select_mentors
from generate_responses import generate_responses
from prompt_builder import card_type
from vector_store import load_store
from pipeline import (
    load_scenarios, get_scenario, scenario_to_context, load_default_regulars,
    MENTOR_CARDS_PATH, CONSUMPTION_MTX_PATH, DEFAULT_PROFILE,
)

# 메인 시연 시나리오(파생 1A/1B 제외). --scenario 미지정 시 전체 측정 대상.
DEFAULT_SCENARIO_IDS = ["1", "2", "3", "4", "5", "6", "7"]


def stat(xs: list[float]) -> dict:
    """리스트 → 평균/p50/p95/min/max 요약."""
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "n":    int(arr.size),
    }


def run_trial(ctx, scenario, profile, cards, matrix, store, use_embedding) -> dict:
    """1회 측정: 선택 → 생성 → 다양성·시간 기록."""
    t0 = time.perf_counter()
    selected = select_mentors(ctx, cards=cards, matrix=matrix,
                              store=store, use_embedding=use_embedding)
    t_select = time.perf_counter() - t0

    t1 = time.perf_counter()
    responses, report = generate_responses(selected, scenario, profile)
    t_generate = time.perf_counter() - t1

    return {
        "diversity":  report.mean_score,
        "method":     report.method,
        "passed":     bool(report.passed),
        "t_select":   t_select,
        "t_generate": t_generate,
        "t_total":    t_select + t_generate,
        "mentor_latencies": [r.latency_s for r in responses],
        "types":      [card_type(c) for c in selected],
        "n_distinct_types": len({card_type(c) for c in selected}),
    }


def aggregate(trials: list[dict]) -> dict:
    """여러 trial → 지표별 집계."""
    diversities = [t["diversity"] for t in trials]
    mentor_lat  = [l for t in trials for l in t["mentor_latencies"]]
    pass_rate   = sum(t["passed"] for t in trials) / len(trials) if trials else 0.0
    distinct_ok = sum(t["n_distinct_types"] == 3 for t in trials) / len(trials) if trials else 0.0
    return {
        "n_trials":            len(trials),
        "diversity":           stat(diversities),
        "diversity_pass_rate": pass_rate,           # mean_score >= 임계값 비율
        "distinct_type_rate":  distinct_ok,         # 3명 Type 모두 다른 비율
        "method":              trials[0]["method"] if trials else "-",
        "latency": {
            "t_select":   stat([t["t_select"]   for t in trials]),
            "t_generate": stat([t["t_generate"] for t in trials]),
            "t_total":    stat([t["t_total"]    for t in trials]),
            "per_mentor": stat(mentor_lat),
        },
    }


def _fmt_stat(s: dict, unit: str = "s") -> str:
    if unit == "s":
        return f"mean {s['mean']:.3f}s  p50 {s['p50']:.3f}  p95 {s['p95']:.3f}  max {s['max']:.3f}"
    return f"mean {s['mean']:.3f}  min {s['min']:.3f}  max {s['max']:.3f}"


def print_report(per_scenario: dict, overall: dict, backend: str, use_embedding: bool):
    print("\n" + "=" * 70)
    print("  D-fine 평가지표 리포트")
    print("=" * 70)
    print(f"백엔드: {backend}  |  임베딩 다양성: {'ON' if use_embedding else 'OFF'}  "
          f"|  다양성 방법: {overall['method']}  |  임계값: {gr.DIVERSITY_THRESHOLD}")
    print(f"총 trial 수: {overall['n_trials']}  ({len(per_scenario)}개 시나리오)")
    print("-" * 70)

    # 시나리오별 요약 (다양성 평균 / 체감 시간 평균)
    print(f"{'시나리오':<10}{'trial':>6}{'다양성(평균)':>14}{'통과율':>9}"
          f"{'체감시간(평균)':>16}{'Type3종':>9}")
    for sid, agg in per_scenario.items():
        print(f"{sid:<10}{agg['n_trials']:>6}{agg['diversity']['mean']:>14.3f}"
              f"{agg['diversity_pass_rate']*100:>8.0f}%"
              f"{agg['latency']['t_generate']['mean']:>15.3f}s"
              f"{agg['distinct_type_rate']*100:>8.0f}%")
    print("-" * 70)

    # 전체 집계
    print("[전체 집계]")
    print(f"  1) 응답 다양성   : {_fmt_stat(overall['diversity'], 'score')}")
    print(f"     - 임계값 통과율 : {overall['diversity_pass_rate']*100:.0f}% "
          f"(mean_score ≥ {gr.DIVERSITY_THRESHOLD})")
    print(f"     - Type 3종 보장 : {overall['distinct_type_rate']*100:.0f}% (3명 모두 다른 Type)")
    print(f"  2) 응답 시간(초)")
    print(f"     - 선택(select)  : {_fmt_stat(overall['latency']['t_select'])}")
    print(f"     - 생성(체감)    : {_fmt_stat(overall['latency']['t_generate'])}")
    print(f"     - 전체(total)   : {_fmt_stat(overall['latency']['t_total'])}")
    print(f"     - 멘토 1명당    : {_fmt_stat(overall['latency']['per_mentor'])}")
    print("=" * 70)
    if backend == "mock":
        print("※ mock 백엔드는 0.05s 고정 지연 + 캔드 응답이라 시간·다양성이 실측이 아님.")
        print("  실측은 VSCode에서 `--backend openai --embedding` 으로 측정하세요.")
    print()


def main():
    parser = argparse.ArgumentParser(description="D-fine 평가지표(다양성·시간) 측정")
    parser.add_argument("--scenario", default=None,
                        help="특정 시나리오만 측정(1~7). 미지정 시 1~7 전체")
    parser.add_argument("--repeat", type=int, default=5,
                        help="시나리오당 반복 횟수(매번 새 멘토 샘플링). 기본 5")
    parser.add_argument("--backend", choices=["openai", "anthropic", "mock"], default="mock")
    parser.add_argument("--embedding", action="store_true", default=False,
                        help="맥락 ANN + 다양성 임베딩 측정 활성화")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="집계 결과를 JSON 파일로 저장할 경로")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="선택/생성 과정의 stderr 로그 표시")
    args = parser.parse_args()

    gr.LLM_BACKEND = args.backend
    print(f"[준비] 데이터·모델 로딩 중… (백엔드={args.backend}, 임베딩={'ON' if args.embedding else 'OFF'})",
          file=sys.stderr, flush=True)
    cards   = load_mentor_cards(MENTOR_CARDS_PATH)
    matrix  = load_consumption_matrix(CONSUMPTION_MTX_PATH)
    store   = load_store() if args.embedding else None
    if args.embedding and store is None:
        print("  [경고] 벡터 인덱스 없음 → 'python vector_store.py' 빌드 필요. 맥락은 랜덤 폴백.",
              file=sys.stderr)

    sdata    = load_scenarios()
    profile  = sdata.get("static_profile", DEFAULT_PROFILE)
    regulars = load_default_regulars()
    sid_list = [args.scenario] if args.scenario else DEFAULT_SCENARIO_IDS
    total    = len(sid_list) * args.repeat
    print(f"[준비 완료] 총 {total} trial 측정 시작 "
          f"(시나리오 {len(sid_list)}개 × repeat {args.repeat})\n", file=sys.stderr, flush=True)

    per_scenario, all_trials = {}, []
    done = 0
    for sid in sid_list:
        sc = get_scenario(sdata, sid)
        if sc is None:
            print(f"  [건너뜀] 시나리오 '{sid}' 없음", file=sys.stderr)
            continue
        ctx = scenario_to_context(sc, regulars)
        trials = []
        for r in range(args.repeat):
            # trial 사이 짧은 간격(jitter): 연속 호출이 한꺼번에 몰려 OpenAI 순간
            #   rate-limit에 걸리는 것을 방지(→ 호출 실패발 다양성 0.000 완화).
            #   첫 trial은 대기 없음. mock 백엔드는 API 호출이 없으므로 생략.
            if done > 0 and args.backend != "mock":
                time.sleep(random.uniform(0.3, 0.6))
            t0 = time.perf_counter()
            if args.verbose:
                tr = run_trial(ctx, sc, profile, cards, matrix, store, args.embedding)
            else:
                with contextlib.redirect_stderr(io.StringIO()):
                    tr = run_trial(ctx, sc, profile, cards, matrix, store, args.embedding)
            trials.append(tr)
            done += 1
            print(f"  [{done}/{total}] 시나리오 {sid} #{r+1}  "
                  f"다양성={tr['diversity']:.3f}  소요={time.perf_counter()-t0:.1f}s",
                  file=sys.stderr, flush=True)
        per_scenario[sid] = aggregate(trials)
        all_trials.extend(trials)

    if not all_trials:
        sys.exit("[오류] 측정된 trial 없음.")

    overall = aggregate(all_trials)
    print_report(per_scenario, overall, args.backend, args.embedding)

    if args.json_path:
        out = {
            "backend": args.backend, "use_embedding": args.embedding,
            "repeat": args.repeat, "threshold": gr.DIVERSITY_THRESHOLD,
            "overall": overall,
            "per_scenario": per_scenario,
        }
        Path(args.json_path).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        print(f"  → 집계 결과 저장: {args.json_path}")


if __name__ == "__main__":
    main()
