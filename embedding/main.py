"""
main.py — AIO 페르소나 분류 파이프라인 오케스트레이터
작성: 2026-05-22
입력: nvidia/Nemotron-Personas-Korea (HuggingFace, config.toml 경유)
출력: resource/outputs/consumption_tags_labeled_n{N}.csv  (uuid + archetype_label + consumption_label)
연산: 7단계 순차 실행
  Step 1  percol5 임베딩            — BGE-M3 5칼럼 → 5120-dim fp16 parquet
  Step 2  소비 임베딩               — BGE-M3 culinary+hobbies → 2048-dim fp16 parquet
  Step 3  archetype 클러스터링      — percol5 → KMeans → archetype_n{N}.csv
  Step 4  소비태그 클러스터링       — 소비 임베딩 → KMeans → consumption_cluster_n{N}.csv
  Step 5  소비태그 레이블링         — Hungarian anchor 매칭
  Step 6  archetype 레이블링        — Hungarian anchor 매칭
  Step 7  최종 병합 + 매트릭스      — 두 레이블 병합 → consumption_tags_labeled_n{N}.csv

Usage:
    uv run main.py --status
    uv run main.py --step 1                      # percol5 임베딩 (config.toml 기준)
    uv run main.py --step 2 --sample 1000000     # 소비 임베딩
    uv run main.py --step 3 --sample 1000000     # archetype 클러스터링
    uv run main.py --step 4 --sample 1000000     # 소비태그 클러스터링
    uv run main.py --step 5 --sample 1000000     # 소비태그 레이블링
    uv run main.py --step 6 --sample 1000000     # archetype 레이블링
    uv run main.py --step 7 --sample 1000000     # 최종 병합 + 매트릭스
    uv run main.py --inspect archetype           # Tier2 정성 검증
    uv run main.py --inspect consumption         # Tier3 정성 검증
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "resource" / "outputs"
EMB_DIR = ROOT / "resource" / "embeddings_percol5"

DEFAULT_SAMPLE = 1_000_000

STEPS: dict[int, tuple[str, str]] = {
    1: ("embed_percol5.py",       "percol5 BGE-M3 임베딩 생성"),
    2: ("embed_consumption.py",   "소비 BGE-M3 임베딩 생성 (culinary+hobbies)"),
    3: ("archetype_cluster.py",   "Tier 2 archetype 클러스터링 (percol5 → KMeans)"),
    4: ("consumption_cluster.py", "Tier 3 소비태그 클러스터링 (소비 임베딩 → KMeans)"),
    5: ("anchor_match.py",        "소비태그 Hungarian 자동 레이블링"),
    6: ("archetype_match.py",     "archetype Hungarian 자동 레이블링"),
    7: ("matrix.py",              "최종 병합 + archetype × 소비태그 매트릭스"),
}

INSPECT: dict[str, str] = {
    "archetype":   "archetype_inspect.py",
    "consumption": "consumption_inspect.py",
}


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "src" / script)] + extra
    print(f"\n[run] {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def status(sample: int) -> None:
    emb_path    = EMB_DIR / "embeddings_percol5.parquet"
    cons_emb    = OUT_DIR / f"consumption_emb_n{sample}.parquet"
    arch_csv    = OUT_DIR / f"archetype_n{sample}.csv"
    cons_clus   = OUT_DIR / f"consumption_cluster_n{sample}.csv"
    cons_lbl    = OUT_DIR / f"consumption_labeled_n{sample}.csv"
    arch_lbl    = OUT_DIR / f"archetype_labeled_n{sample}.csv"
    final_csv   = OUT_DIR / f"consumption_tags_labeled_n{sample}.csv"

    checks = [
        (1, "percol5 임베딩",       emb_path),
        (2, "소비 임베딩",          cons_emb),
        (3, "archetype 클러스터링", arch_csv),
        (4, "소비태그 클러스터링",  cons_clus),
        (5, "소비태그 레이블링",    cons_lbl),
        (6, "archetype 레이블링",   arch_lbl),
        (7, "최종 병합",            final_csv),
    ]

    print(f"\n[status]  n={sample:,}")
    print("─" * 66)
    for step, desc, path in checks:
        mark = "✅" if path.exists() else "❌"
        size = f"  ({path.stat().st_size / 1e6:.0f} MB)" if path.exists() else ""
        print(f"  Step {step}  {mark}  {desc:22s}  {path.name}{size}")
    print()

    if not emb_path.exists():
        print("→ 다음: uv run main.py --step 1")
    elif not cons_emb.exists():
        print(f"→ 다음: uv run main.py --step 2 --sample {sample}")
    elif not arch_csv.exists():
        print(f"→ 다음: uv run main.py --step 3 --sample {sample}")
    elif not cons_clus.exists():
        print(f"→ 다음: uv run main.py --step 4 --sample {sample}")
    elif not cons_lbl.exists():
        print(f"→ 다음: uv run main.py --step 5 --sample {sample}")
    elif not arch_lbl.exists():
        print(f"→ 다음: uv run main.py --step 6 --sample {sample}")
    elif not final_csv.exists():
        print(f"→ 다음: uv run main.py --step 7 --sample {sample}")
    else:
        print("→ 파이프라인 완료.")
        print(f"   최종 산출물: {final_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="AIO 페르소나 분류 파이프라인")
    ap.add_argument("--status",  action="store_true",
                    help="중간 산출물 존재 여부 확인")
    ap.add_argument("--step",    type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                    help="파이프라인 단계 실행 (1=percol5임베딩 / 2=소비임베딩 / 3=archetype클러스터링 / 4=소비태그클러스터링 / 5=소비태그레이블링 / 6=archetype레이블링 / 7=최종병합)")
    ap.add_argument("--inspect", choices=list(INSPECT),
                    help="정성 검증 스크립트 실행")
    ap.add_argument("--sample",  type=int, default=DEFAULT_SAMPLE,
                    help=f"샘플 크기 — step 2/3 및 inspect 적용 (기본 {DEFAULT_SAMPLE:,})")
    ap.add_argument("--per-cluster", type=int, default=5,
                    help="--inspect 클러스터당 샘플 출력 수 (기본 5)")
    args = ap.parse_args()

    if args.status:
        status(args.sample)
        return

    if args.step is not None:
        script, desc = STEPS[args.step]
        print(f"[step {args.step}] {desc}")
        extra: list[str] = []
        if args.step in (2, 3, 4, 5, 6, 7):
            extra += ["--sample", str(args.sample)]
        _run(script, extra)
        return

    if args.inspect is not None:
        script = INSPECT[args.inspect]
        extra = [
            "--sample",      str(args.sample),
            "--per-cluster", str(args.per_cluster),
        ]
        _run(script, extra)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
