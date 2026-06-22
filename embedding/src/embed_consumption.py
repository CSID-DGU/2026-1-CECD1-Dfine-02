"""
embed_consumption.py — 소비 임베딩 생성  [파이프라인 Step 2]
작성: 2026-05
입력: resource/embeddings_percol5/embeddings_percol5.parquet  (UUID 기준)
      nvidia/Nemotron-Personas-Korea (culinary_persona, hobbies_and_interests_list)
출력: resource/outputs/consumption_emb_n{N}.parquet  (uuid + float16[2048])
연산: culinary_persona + hobbies_and_interests_list 각각 BGE-M3 1024-dim → concat → 2048-dim

Usage:
    uv run src/embed_consumption.py
    uv run src/embed_consumption.py --sample 1000000
    uv run src/embed_consumption.py --sample 50000
"""

import argparse
import gc
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel

ROOT      = Path(__file__).parent.parent
OUT_DIR   = ROOT / "resource" / "outputs"
EMBED_DIR = ROOT / "resource" / "embeddings_percol5"
SEED      = 42
EMBED_DIM = 2048  # 1024 × 2 cols

CONS_COLS = ["culinary_persona", "hobbies_and_interests_list"]


def load_uuids(sample: int) -> list[str]:
    """percol5 parquet에서 UUID만 로드. consumption_tag.py와 동일한 서브샘플링 로직."""
    table = pa_ds.dataset(EMBED_DIR, format="parquet").to_table(columns=["uuid"])
    uuids_all = table.column("uuid").to_pylist()
    n_total   = len(uuids_all)
    del table; gc.collect()

    if sample < n_total:
        idx = np.sort(np.random.default_rng(SEED).choice(n_total, sample, replace=False))
        uuids = [uuids_all[i] for i in idx]
    else:
        uuids = uuids_all

    print(f"[uuid] {len(uuids):,} (from percol5, n_total={n_total:,})")
    return uuids


def load_cons_texts(uuids: list[str], cfg: dict) -> dict[str, dict]:
    """HF 데이터셋에서 UUID 매칭 → culinary + hobbies 텍스트 dict."""
    name  = cfg["dataset"]["name"]
    cache = cfg["dataset"]["cache_dir"] or None

    target = set(uuids)
    cols   = ["uuid"] + CONS_COLS
    print(f"[text] {name} 스캔 중 ({len(target):,} targets) ...")
    ds = load_dataset(name, split="train", cache_dir=cache).select_columns(cols)

    def to_str(v) -> str:
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return v or ""

    result: dict[str, dict] = {}
    for row in ds:
        u = row["uuid"]
        if u in target:
            result[u] = {c: to_str(row.get(c)) for c in CONS_COLS}
        if len(result) >= len(target):
            break

    print(f"[text] matched {len(result):,} / {len(uuids):,}")
    return result


def embed_and_save(
    uuids: list[str],
    texts: dict[str, dict],
    model: BGEM3FlagModel,
    cfg: dict,
    out_path: Path,
) -> None:
    bs    = cfg["model"]["batch_size"]
    ml    = cfg["model"]["max_length"]
    n     = len(uuids)
    comp  = cfg["output"]["compression"]

    schema = pa.schema([
        ("uuid",      pa.string()),
        ("embedding", pa.list_(pa.float16(), EMBED_DIM)),
    ])

    CHUNK = 50_000
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, schema, compression=comp)
    t0 = time.perf_counter()

    try:
        for start in range(0, n, CHUNK):
            end          = min(start + CHUNK, n)
            chunk_uuids  = uuids[start:end]

            col_vecs = []
            for col in CONS_COLS:
                col_texts = [texts.get(u, {}).get(col) or "" for u in chunk_uuids]
                out = model.encode(col_texts, batch_size=bs, max_length=ml,
                                   return_dense=True, return_sparse=False,
                                   return_colbert_vecs=False)
                col_vecs.append(np.asarray(out["dense_vecs"], dtype=np.float16))

            combined = np.concatenate(col_vecs, axis=1)  # (chunk, 2048) fp16
            flat     = combined.reshape(-1)

            tbl = pa.table({
                "uuid": pa.array(chunk_uuids, type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array(flat, type=pa.float16()), EMBED_DIM,
                ),
            }, schema=schema)
            writer.write_table(tbl)

            elapsed = time.perf_counter() - t0
            rate    = end / elapsed
            eta     = (n - end) / rate if rate > 0 else 0
            print(f"  {end:>7,}/{n}  ({rate:.0f} rows/s, ETA {eta/60:.0f}m)")
    finally:
        writer.close()

    dt   = time.perf_counter() - t0
    size = out_path.stat().st_size / 1e9
    print(f"\n[done] {dt:.1f}s  |  {out_path}  ({size:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent.parent / "config.toml")
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f"config not found: {args.config}")
    with args.config.open("rb") as f:
        cfg = tomllib.load(f)

    out_path = OUT_DIR / f"consumption_emb_n{args.sample}.parquet"
    if out_path.exists():
        print(f"[skip] 이미 존재: {out_path}")
        print("  덮어쓰려면 파일을 삭제 후 재실행하세요.")
        sys.exit(0)

    # 1. UUID 로드 (percol5 parquet 기준)
    uuids = load_uuids(args.sample)

    # 2. HF 텍스트 로드
    texts = load_cons_texts(uuids, cfg)

    # 3. BGE-M3 로드 → 임베딩 + 저장
    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = _DTYPE_MAP[cfg["model"]["dtype"]]
    print(f"\n[model] loading {cfg['model']['name']} ({cfg['model']['dtype']}) ...")
    model = BGEM3FlagModel(cfg["model"]["name"], use_fp16=False,
                           devices=[cfg["runtime"]["device"]])
    model.model = model.model.to(dtype)

    embed_and_save(uuids, texts, model, cfg, out_path)


if __name__ == "__main__":
    main()
