"""
build_consumption_meta.py — consumption_tags.csv → FAISS용 메타 parquet 변환

build_faiss.py의 load_labels() + 메타 저장 방식과 동일한 구조로,
consumption_tags.csv를 FAISS 인덱스와 1:1 대응되는 parquet으로 변환합니다.

입력:
  consumption_tags.csv
    컬럼: uuid, primary_tag, secondary_tag, coherence_score

출력:
  faiss_consumption_meta.parquet
    컬럼: uuid, primary_tag, secondary_tag, coherence_score
    - 인덱스 순서(0, 1, 2, ...)가 FAISS index의 row 순서와 일치
    - uuid 기준으로 임베딩을 로드할 때 pos_map으로 활용 가능

Usage:
    python build_consumption_meta.py
    python build_consumption_meta.py --input /path/to/consumption_tags.csv
    python build_consumption_meta.py --filter-coherence  # coherence_score==1 만 포함
"""

import argparse
from pathlib import Path

import pandas as pd


def load_consumption_tags(csv_path: Path, filter_coherence: bool) -> pd.DataFrame:
    """
    consumption_tags.csv 로드 및 전처리.

    build_faiss.py의 load_labels()에 대응:
      - utf-8-sig(BOM) 인코딩 처리
      - 타입 명시 (archetype → int 처럼 coherence_score → int)
      - 필요 시 coherence_score == 1 필터링
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 타입 보정 (build_faiss의 archetype.astype(int) 패턴과 동일)
    df["coherence_score"] = df["coherence_score"].astype(int)
    df["primary_tag"]     = df["primary_tag"].astype(str).str.strip()
    df["secondary_tag"]   = df["secondary_tag"].astype(str).str.strip()

    print(f"[load] {len(df):,} rows  columns={list(df.columns)}")

    if filter_coherence:
        before = len(df)
        df = df[df["coherence_score"] == 1].reset_index(drop=True)
        print(f"[filter] coherence_score==1 → {len(df):,} rows  (제거: {before - len(df):,})")

    # uuid 중복 체크 (FAISS 인덱스와 1:1 대응을 위해 고유해야 함)
    dup = df["uuid"].duplicated().sum()
    if dup > 0:
        print(f"[warn] uuid 중복 {dup:,}개 발견 → 첫 번째만 유지")
        df = df.drop_duplicates(subset="uuid", keep="first").reset_index(drop=True)

    print(f"[ready] 최종 {len(df):,} rows — 인덱스 순서 확정")
    return df


def save_meta(df: pd.DataFrame, out_path: Path) -> None:
    """
    build_faiss.py의 meta_df.to_parquet() 방식과 동일하게 저장.
    index=False로 저장해 FAISS row 번호(0-based)와 직접 대응.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[save] {out_path}  ({size_mb:.2f} MB)")


def print_summary(df: pd.DataFrame) -> None:
    print("\n── 태그 분포 (primary_tag) ──────────────────")
    print(df["primary_tag"].value_counts().to_string())
    print("\n── coherence_score 분포 ─────────────────────")
    print(df["coherence_score"].value_counts().sort_index().to_string())
    print("\n── FAISS 인덱스 활용 예시 ───────────────────")
    print("  meta_df = pd.read_parquet('faiss_consumption_meta.parquet')")
    print("  # FAISS 검색 결과 I (shape: [query_n, k])")
    print("  D, I = index.search(query_vec, k=10)")
    print("  meta_df.iloc[I[0]]  # 유사한 10개의 uuid + 태그 조회")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path,
        default=Path("consumption_tags.csv"),
        help="입력 CSV 경로 (기본: consumption_tags.csv)"
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("faiss_consumption_meta.parquet"),
        help="출력 parquet 경로 (기본: faiss_consumption_meta.parquet)"
    )
    ap.add_argument(
        "--filter-coherence", action="store_true",
        help="coherence_score==1인 행만 포함 (노이즈 제거)"
    )
    args = ap.parse_args()

    # 1. CSV 로드 + 전처리
    df = load_consumption_tags(args.input, args.filter_coherence)

    # 2. parquet 저장 (인덱스 순서 = FAISS row 순서)
    save_meta(df, args.output)

    # 3. 요약 출력
    print_summary(df)

    print(f"\n[done] {len(df):,} rows → {args.output}")
    print("다음 단계: 이 parquet의 uuid 순서로 임베딩을 로드해 FAISS 인덱스를 빌드하세요.")
    print("  uuids = pd.read_parquet(args.output)['uuid'].tolist()")
    print("  vecs  = load_embeddings_ordered(uuids)  # build_faiss.py 참고")


if __name__ == "__main__":
    main()
