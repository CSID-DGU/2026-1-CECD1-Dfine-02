"""
sweep_consumption_pca.py — 소비태그 클러스터링 PCA 차원 비교 실험
작성: 2026-05-22
목적: --pca2 값별로 consumption_cluster → anchor_match 실행 후 anchor cosine / margin 비교
결과: resource/outputs/sweep_pca/ 디렉터리에 각 PCA 값의 cluster CSV + anchor_sim CSV 저장

Usage:
    uv run src/sweep_consumption_pca.py --sample 50000
    uv run src/sweep_consumption_pca.py --sample 50000 --pca 10 30 50 100 200
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT      = Path(__file__).parent.parent
SRC_DIR   = ROOT / "src"
OUT_DIR   = ROOT / "resource" / "outputs"
SWEEP_DIR = OUT_DIR / "sweep_pca"

DEFAULT_PCA = [10, 30, 50, 100, 200]


def run(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(SRC_DIR / script)] + [str(a) for a in extra]
    print(f"\n[run] {' '.join(cmd)}\n{'─'*60}")
    return subprocess.run(cmd).returncode


def hungarian_metrics(sim_path: Path) -> dict:
    """anchor_sim CSV → Hungarian 할당 후 min_cos / min_margin 계산."""
    sim = pd.read_csv(sim_path, index_col=0).values.astype(float)  # (5, 5)
    row_idx, col_idx = linear_sum_assignment(1.0 - sim)
    assigned = [sim[r, c] for r, c in zip(row_idx, col_idx)]
    margins  = [sim[r].max() - np.partition(sim[r], -2)[-2]
                for r in row_idx]
    return {
        "min_cos":    float(min(assigned)),
        "mean_cos":   float(np.mean(assigned)),
        "min_margin": float(min(margins)),
    }


def cluster_balance(csv_path: Path) -> float:
    """최소/최대 클러스터 크기 비율 (1.0 = 완전 균등)."""
    sizes = pd.read_csv(csv_path)["consumption_tag"].value_counts()
    return float(sizes.min() / sizes.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, required=True,
                    help="처리할 행 수 (consumption_emb_n{N}.parquet 기준)")
    ap.add_argument("--pca",    type=int, nargs="+", default=DEFAULT_PCA,
                    help=f"비교할 PCA 차원 목록 (기본: {DEFAULT_PCA})")
    ap.add_argument("--noise-pct", type=float, default=10.0,
                    help="noise_dist 임계 백분위 (기본 10)")
    args = ap.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    n = args.sample
    results: list[dict] = []

    for pca in args.pca:
        print(f"\n{'='*60}")
        print(f"  PCA = {pca}  (n={n:,})")
        print(f"{'='*60}")

        cons_out = SWEEP_DIR / f"cons_pca{pca}_n{n}.csv"
        sim_dst  = SWEEP_DIR / f"anchor_sim_pca{pca}_n{n}.csv"

        # Step 4: 소비태그 클러스터링
        rc = run("consumption_cluster.py", [
            "--sample", n, "--pca2", pca,
            "--noise-pct", args.noise_pct,
            "--out", cons_out,
        ])
        if rc != 0:
            print(f"[skip] consumption_cluster 실패 (pca={pca})")
            continue

        # Step 5: anchor Hungarian 레이블링
        rc = run("anchor_match.py", [
            "--sample", n, "--csv", cons_out,
        ])
        if rc != 0:
            print(f"[skip] anchor_match 실패 (pca={pca})")
            continue

        # anchor_sim_n{N}.csv 복사 (다음 실행에 덮어쓰이기 전에 보존)
        sim_src = OUT_DIR / f"anchor_sim_n{n}.csv"
        shutil.copy(sim_src, sim_dst)

        metrics = hungarian_metrics(sim_dst)
        metrics["pca"]     = pca
        metrics["balance"] = cluster_balance(cons_out)
        results.append(metrics)

    # ── 요약 테이블 ────────────────────────────────────────────────────────────
    if not results:
        print("\n[warn] 수집된 결과 없음")
        return

    print(f"\n{'='*60}")
    print(f"  PCA 비교 요약  (n={n:,})")
    print(f"{'='*60}")
    print(f"  {'PCA':>6}  {'min_cos':>8}  {'mean_cos':>9}  {'min_margin':>11}  {'balance':>8}")
    print(f"  {'─'*50}")

    best_cos    = max(r["min_cos"]    for r in results)
    best_margin = max(r["min_margin"] for r in results)

    for r in results:
        cos_flag    = " ◀ best_cos"    if r["min_cos"]    == best_cos    else ""
        margin_flag = " ◀ best_margin" if r["min_margin"] == best_margin else ""
        flag = cos_flag or margin_flag
        print(f"  {r['pca']:>6}  {r['min_cos']:>8.4f}  {r['mean_cos']:>9.4f}"
              f"  {r['min_margin']:>11.4f}  {r['balance']:>8.3f}{flag}")

    print(f"\n  sweep 결과 저장: {SWEEP_DIR}")


if __name__ == "__main__":
    main()
