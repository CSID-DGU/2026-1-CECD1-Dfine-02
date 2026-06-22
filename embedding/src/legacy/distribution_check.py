"""
distribution_check.py — Nemotron-Personas-Korea의 age, occupation 분포 확인.

용도: embed.py 실행 전 sampling_strategy 결정.
     ("random" 또는 "age_stratified")

전제: 데이터셋이 resource/nemotron_korea/ 에 다운로드되어 있음.
     없으면 huggingface-cli 다운로드 명령을 안내.

실행: uv run python src/distribution_check.py
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.dataset as pd_ds


DATA_DIR = Path(__file__).parent.parent / "resource" / "nemotron_korea"


def age_band(a: int) -> str:
    if 19 <= a <= 29:
        return "19-29"
    if 30 <= a <= 49:
        return "30-49"
    if 50 <= a <= 69:
        return "50-69"
    if a >= 70:
        return "70+"
    return "other"


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(
            f"데이터셋이 {DATA_DIR}에 없습니다.\n"
            f"다음 명령으로 다운로드:\n"
            f"  huggingface-cli download nvidia/Nemotron-Personas-Korea "
            f"--repo-type dataset --local-dir {DATA_DIR}"
        )

    ds = pd_ds.dataset(DATA_DIR, format="parquet")
    df = ds.to_table(columns=["age", "occupation"]).to_pandas()
    n = len(df)

    print(f"\n총 행 수: {n:,}\n")

    print("age 밴드 분포:")
    bands = (
        df["age"]
        .apply(age_band)
        .value_counts(normalize=True)
        .sort_index()
    )
    print(bands.to_string(float_format=lambda x: f"{x:.1%}"))

    print("\noccupation 상위 20 (참고용):")
    top20 = df["occupation"].value_counts(normalize=True).head(20)
    print(top20.to_string(float_format=lambda x: f"{x:.1%}"))

    print("\n--- 결정 가이드 (계획서 §11.2) ---")
    print("  age 3밴드 모두 20–45%        → sampling_strategy = 'random'")
    print("  한 밴드 ≥ 50% 또는 다른 쪽 < 10% → sampling_strategy = 'age_stratified'")
    print("  45–55% 경계선                 → 두 분기 모두 실행")


if __name__ == "__main__":
    main()
