"""
matrix.py — 최종 병합 + archetype × 소비태그 매트릭스  [파이프라인 Step 7]
작성: 2026-05-22
입력: resource/outputs/consumption_labeled_n{N}.csv  — uuid / consumption_tag / noise_dist / entropy / consumption_label
      resource/outputs/archetype_labeled_n{N}.csv     — uuid / archetype / archetype_label
출력: resource/outputs/consumption_tags_labeled_n{N}.csv  — 최종 산출물 (전 컬럼 병합)
      resource/outputs/matrix_5x5_n{N}.csv            — archetype × 소비태그 (noise_dist=0만)
      resource/outputs/matrix_5x5_all_n{N}.csv        — archetype × 소비태그 (전체)

Usage:
    uv run src/matrix.py --sample 1000000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "resource" / "outputs"


def print_and_save_matrix(
    archetype: np.ndarray, cons_tag: np.ndarray, k: int, out_path: Path
) -> None:
    mat = np.zeros((k, k), dtype=int)
    for a, c in zip(archetype, cons_tag):
        mat[a, c] += 1

    print(f"\n{'':14s}" + "".join(f"  ctag{c:1d}" for c in range(k)))
    print("  " + "-" * (12 + 7 * k))
    for a in range(k):
        row_str = f"  arch{a:1d}       |" + "".join(f"  {mat[a, c]:4d}" for c in range(k))
        print(row_str)

    df = pd.DataFrame(mat, index=[f"arch{a}" for a in range(k)],
                      columns=[f"ctag{c}" for c in range(k)])
    df.to_csv(out_path)
    print(f"[save] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50_000)
    ap.add_argument("--k",      type=int, default=5)
    args = ap.parse_args()

    cons_path = OUT_DIR / f"consumption_labeled_n{args.sample}.csv"
    arch_path = OUT_DIR / f"archetype_labeled_n{args.sample}.csv"

    for p in (cons_path, arch_path):
        if not p.exists():
            sys.exit(f"[error] 파일 없음: {p}")

    cons_df = pd.read_csv(cons_path)
    arch_df = pd.read_csv(arch_path)

    df = cons_df.merge(arch_df[["uuid", "archetype", "archetype_label"]], on="uuid", how="inner")
    if len(df) < len(cons_df):
        print(f"[warn] uuid 매칭 누락: {len(cons_df) - len(df)}행")

    archetype = df["archetype"].to_numpy()
    cons_tag  = df["consumption_tag"].to_numpy()
    noise     = df["noise_dist"].to_numpy()

    print("\n" + "=" * 64)
    print("  5×5 정합 매트릭스 (archetype × 소비태그)")
    print("  → noise_dist=0 인원만 카운트")
    print("=" * 64)
    clean = noise == 0
    print_and_save_matrix(archetype[clean], cons_tag[clean], args.k,
                          OUT_DIR / f"matrix_5x5_n{args.sample}.csv")

    print("\n  (전체 포함 매트릭스)")
    print_and_save_matrix(archetype, cons_tag, args.k,
                          OUT_DIR / f"matrix_5x5_all_n{args.sample}.csv")

    out = OUT_DIR / f"consumption_tags_labeled_n{args.sample}.csv"
    df.to_csv(out, index=False)
    print(f"[save] {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
