"""
ablation_embedding.py — 다양성 지표의 임베딩 모델 민감도(ablation) 실험

목적:
  "응답 다양성 점수가 특정 임베딩 모델(KR-SBERT) 때문에 생긴 착시가 아니다"를
  검증한다. 동일한 멘토 응답 집합을 여러 한국어 임베딩 모델로 채점하여
  결론(평균 거리·임계값 통과율)이 모델에 관계없이 유지되는지 본다.

공정성:
  응답은 시나리오별로 '한 번만' 생성해 고정한다(temperature·샘플링 노이즈 차단).
  그 동일 텍스트를 후보 모델들로 채점 → 점수 차이는 오직 임베딩 모델 차이.

실행:
  python ablation_embedding.py --repeat 2
"""
from __future__ import annotations
import argparse, contextlib, io, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import generate_responses as gr
from select_mentors import load_mentor_cards, load_consumption_matrix, select_mentors
from generate_responses import generate_responses, DIVERSITY_THRESHOLD
from vector_store import load_store
from pipeline import (
    load_scenarios, get_scenario, scenario_to_context, load_default_regulars,
    MENTOR_CARDS_PATH, CONSUMPTION_MTX_PATH, DEFAULT_PROFILE,
)

# sentence-transformers 인터페이스와 바로 호환되는 한국어/다국어 임베딩 후보들
CANDIDATE_MODELS = [
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS",                       # 현재 사용 모델(기준)
    "jhgan/ko-sroberta-multitask",                               # 대표 한국어 SBERT
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 다국어 baseline
]
SCENARIO_IDS = ["1", "2", "3", "4", "5", "6", "7"]


def mean_pairwise_distance(model, texts: list[str]) -> float:
    """generate_responses._diversity_embedding과 동일한 계산: 쌍별 코사인 거리 평균."""
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    n = len(embs)
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(1.0 - float(np.dot(embs[i], embs[j])))
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser(description="다양성 지표 임베딩 모델 ablation")
    ap.add_argument("--repeat", type=int, default=2, help="시나리오당 응답 생성 횟수(기본 2)")
    args = ap.parse_args()

    gr.LLM_BACKEND = "openai"
    print(f"[1/2] 멘토 응답 생성 중 (시나리오 {len(SCENARIO_IDS)}개 × repeat {args.repeat}, "
          f"백엔드=openai)…", file=sys.stderr, flush=True)

    cards    = load_mentor_cards(MENTOR_CARDS_PATH)
    matrix   = load_consumption_matrix(CONSUMPTION_MTX_PATH)
    store    = load_store()
    sdata    = load_scenarios()
    profile  = sdata.get("static_profile", DEFAULT_PROFILE)
    regulars = load_default_regulars()

    # ── 응답 집합 한 번만 생성해 고정 (유효 응답 2개 이상인 것만 채점 대상) ──
    response_sets: list[list[str]] = []
    n_made = 0
    total  = len(SCENARIO_IDS) * args.repeat
    for sid in SCENARIO_IDS:
        sc = get_scenario(sdata, sid)
        if sc is None:
            continue
        ctx = scenario_to_context(sc, regulars)
        for r in range(args.repeat):
            if n_made > 0:
                time.sleep(random.uniform(0.3, 0.6))   # rate-limit 회피
            with contextlib.redirect_stderr(io.StringIO()):
                selected = select_mentors(ctx, cards=cards, matrix=matrix,
                                          store=store, use_embedding=True)
                responses, _ = generate_responses(selected, sc, profile)
            texts = [resp.response_text for resp in responses if resp.is_valid]
            n_made += 1
            if len(texts) >= 2:
                response_sets.append(texts)
                print(f"  [{n_made}/{total}] 시나리오 {sid} #{r+1}  유효응답 {len(texts)}개 수집",
                      file=sys.stderr, flush=True)
            else:
                print(f"  [{n_made}/{total}] 시나리오 {sid} #{r+1}  유효응답 부족({len(texts)}개)→제외",
                      file=sys.stderr, flush=True)

    if not response_sets:
        sys.exit("[오류] 채점할 응답 집합이 없음.")

    # ── 동일 응답 집합을 후보 모델별로 채점 ──
    print(f"\n[2/2] 임베딩 모델 {len(CANDIDATE_MODELS)}종으로 동일 응답 {len(response_sets)}세트 채점 중…\n",
          file=sys.stderr, flush=True)
    from sentence_transformers import SentenceTransformer

    rows = []
    for name in CANDIDATE_MODELS:
        try:
            model = SentenceTransformer(name)
        except Exception as e:
            print(f"  [건너뜀] {name} 로드 실패: {e!r}", file=sys.stderr)
            continue
        dists = [mean_pairwise_distance(model, ts) for ts in response_sets]
        arr = np.asarray(dists, dtype=float)
        passed = float((arr >= DIVERSITY_THRESHOLD).mean())
        rows.append((name, arr.mean(), float(np.median(arr)), arr.min(), arr.max(), passed))

    # ── 리포트 ──
    print("\n" + "=" * 88)
    print("  다양성 지표 — 임베딩 모델 민감도(ablation)")
    print("=" * 88)
    print(f"동일 응답 {len(response_sets)}세트를 임베딩 모델만 바꿔 채점  |  임계값 {DIVERSITY_THRESHOLD}")
    print("-" * 88)
    print(f"{'임베딩 모델':<52}{'평균':>8}{'중앙':>8}{'min':>7}{'max':>7}{'통과율':>8}")
    for name, mean, med, mn, mx, pr in rows:
        short = name if len(name) <= 50 else name[:47] + "…"
        print(f"{short:<52}{mean:>8.3f}{med:>8.3f}{mn:>7.3f}{mx:>7.3f}{pr*100:>7.0f}%")
    print("=" * 88)
    print("※ 절대 평균값은 모델마다 다를 수 있으나, 통과율·상대 순위가 일관되면")
    print("  다양성 결론이 특정 임베딩 모델에 의존하지 않음을 의미한다.")
    print()


if __name__ == "__main__":
    main()
