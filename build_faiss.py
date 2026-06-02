"""
build_faiss.py — 클러스터링 결과 FAISS IVFFlat 인덱스 탑재

입력:
  resource/embeddings_percol5/*.parquet   (uuid, embedding fp16 5120-dim)
  resource/outputs/archetype_labeled_n1000000.csv  (uuid, archetype)

출력:
  resource/outputs/faiss_persona.index        FAISS IVFFlat 인덱스 (IP metric)
  resource/outputs/faiss_persona_meta.parquet uuid + archetype (인덱스 순서 일치)
  resource/nemotron_raw.parquet               Nemotron 원본 데이터 (모든 컬럼)

Usage:
    uv run src/legacy/build_faiss.py
    uv run src/legacy/build_faiss.py --nlist 2048 --nprobe 128
    uv run src/legacy/build_faiss.py --skip-nemotron
"""

import argparse
import gc
from pathlib import Path

import datasets
import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

ROOT      = Path(__file__).parent.parent.parent
OUT_DIR   = ROOT / "resource" / "outputs"
EMB_DIR   = ROOT / "resource" / "embeddings_percol5"
RES_DIR   = ROOT / "resource"

ARCHETYPE_CSV = OUT_DIR / "archetype_labeled_n1000000.csv"

BATCH = 50_000


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(ARCHETYPE_CSV, usecols=["uuid", "archetype"])
    df["archetype"] = df["archetype"].astype(int)
    print(f"[labels] {len(df):,} rows loaded")
    return df


def load_embeddings_ordered(uuid_order: list[str]) -> np.ndarray:
    """uuid_order 순서대로 fp32 행렬 반환."""
    pos_map = {uid: i for i, uid in enumerate(uuid_order)}
    n       = len(uuid_order)
    ds      = pa_ds.dataset(EMB_DIR, format="parquet")

    dim = ds.schema.field("embedding").type.list_size
    out = np.empty((n, dim), dtype=np.float32)

    filled = 0
    for batch in ds.to_batches(columns=["uuid", "embedding"], batch_size=BATCH):
        uids  = batch.column("uuid").to_pylist()
        flat  = (batch.column("embedding").flatten()
                      .to_numpy(zero_copy_only=False))
        vecs  = flat.reshape(len(uids), dim)
        for local_i, uid in enumerate(uids):
            idx = pos_map.get(uid)
            if idx is not None:
                out[idx] = vecs[local_i]
                filled  += 1
        del flat, vecs

    print(f"[embed] {filled:,} / {n:,} rows loaded  dim={dim}")
    gc.collect()
    return out


def save_nemotron_raw(cache_dir: str | None = None) -> None:
    """Nemotron 원본 데이터를 모든 컬럼 그대로 저장."""
    out_path = RES_DIR / "nemotron_raw.parquet"

    if out_path.exists():
        print(f"[nemotron] skip  {out_path} (already exists)")
        return

    print(f"[nemotron] loading dataset ...")
    dataset = datasets.load_dataset(
        "nvidia/Nemotron-Personas-Korea",
        split="train",
        cache_dir=cache_dir
    )

    df = dataset.to_pandas()
    print(f"[nemotron] {len(df):,} rows  columns={list(df.columns)}")

    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    print(f"[save] {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


def build_index(vecs: np.ndarray, nlist: int) -> faiss.Index:
    dim     = vecs.shape[1]
    quant   = faiss.IndexFlatIP(dim)
    index   = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    train_n = min(len(vecs), max(nlist * 50, 100_000))
    rng     = np.random.default_rng(42)
    t_idx   = rng.choice(len(vecs), train_n, replace=False)
    print(f"[faiss] train  nlist={nlist}  n_train={train_n:,}")
    index.train(vecs[t_idx])

    print(f"[faiss] add {len(vecs):,} vectors ...")
    index.add(vecs)
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlist",  type=int, default=1024)
    ap.add_argument("--nprobe", type=int, default=64,
                    help="검색 기본 nprobe (인덱스에 저장, 쿼리 시 오버라이드 가능)")
    ap.add_argument("--skip-nemotron", action="store_true",
                    help="Nemotron 원본 저장 건너뛰기")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Nemotron 원본 저장 (선택사항)
    if not args.skip_nemotron:
        save_nemotron_raw()

    # 1. 라벨 로드 → uuid 순서 확정
    meta_df = load_labels()
    uuids   = meta_df["uuid"].tolist()

    # 2. 임베딩 로드 (uuid 순서 맞춤, fp16→fp32)
    vecs = load_embeddings_ordered(uuids)

    # L2 정규화 (저장된 임베딩이 정규화 안 된 경우 대비)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.clip(norms, 1e-10, None)

    # 3. FAISS 인덱스 빌드
    index = build_index(vecs, args.nlist)
    index.nprobe = args.nprobe
    del vecs; gc.collect()

    # 4. 저장
    idx_path  = OUT_DIR / "faiss_persona.index"
    meta_path = OUT_DIR / "faiss_persona_meta.parquet"

    faiss.write_index(index, str(idx_path))
    print(f"[save] {idx_path}  ({idx_path.stat().st_size / 1e9:.2f} GB)")

    meta_df.to_parquet(meta_path, index=False)
    print(f"[save] {meta_path}")

    print(f"\n[done] ntotal={index.ntotal:,}  nlist={args.nlist}  nprobe={args.nprobe}")
    print("쿼리 예시:")
    print("  index = faiss.read_index('resource/outputs/faiss_persona.index')")
    print("  D, I = index.search(query_vec, k=10)   # I → meta_df.iloc[I[0]]")


if __name__ == "__main__":
    main()
