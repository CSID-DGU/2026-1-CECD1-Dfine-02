"""
칼럼별 개별 임베딩 테스트 — 접근 B.

기존 embed.py: 5칼럼 텍스트 concat → 단일 1024-dim
이 스크립트: 7칼럼(전략 C) 각각 독립 임베딩 → concat 7168-dim

Usage:
    uv run src/embed_percol.py
    uv run src/embed_percol.py --config custom.toml
"""

import argparse
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel

COLS: list[tuple[str, str]] = [
    ("career_goals_and_ambitions", "커리어 목표"),
    ("professional_persona",       "직업적 자아"),
    ("family_persona",             "가족적 자아"),
    ("travel_persona",             "여행 성향"),
    ("arts_persona",               "예술·문화"),
    ("culinary_persona",           "음식·미식"),
    ("hobbies_and_interests",      "취미와 관심사"),
]

EMBED_DIM = 1024
TOTAL_DIM = len(COLS) * EMBED_DIM  # 7168
OUT_DIR   = Path("resource/embeddings_percol")


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def env_check(device: str) -> None:
    assert torch.cuda.is_available(), "CUDA unavailable"
    dev_idx = int(device.split(":")[1]) if ":" in device else 0
    cap = torch.cuda.get_device_capability(dev_idx)
    name = torch.cuda.get_device_name(dev_idx)
    print(f"[env] {name} sm_{cap[0]}{cap[1]} | "
          f"torch {torch.__version__} | cuda {torch.version.cuda}")


def load_samples(cfg: dict) -> tuple[list[str], list[dict]]:
    """데이터셋 로드 → (uuids, rows). rows는 7칼럼 dict 리스트."""
    strategy = cfg["sampling"]["strategy"]
    name     = cfg["dataset"]["name"]
    seed     = cfg["dataset"]["seed"]
    n        = cfg["benchmark"]["n_bench"]
    cache    = cfg["dataset"]["cache_dir"] or None
    col_keys = ["uuid"] + [c for c, _ in COLS]

    ds = load_dataset(name, split="train", cache_dir=cache)
    ds = ds.select_columns(col_keys)

    if strategy == "random":
        ds = ds.shuffle(seed=seed).select(range(n))
    elif strategy == "full":
        pass
    else:
        sys.exit(f"[load] embed_percol은 random/full만 지원: {strategy!r}")

    print(f"[load] {strategy} — {len(ds):,} rows, {len(COLS)} cols")

    uuids, rows = [], []
    for row in ds:
        uuids.append(row["uuid"])
        rows.append({c: row[c] for c, _ in COLS})
    return uuids, rows


def embed_and_save(
    uuids: list[str],
    rows: list[dict],
    model: BGEM3FlagModel,
    cfg: dict,
) -> None:
    bs           = cfg["model"]["batch_size"]
    ml           = cfg["model"]["max_length"]
    encode_chunk = cfg["model"].get("encode_chunk", bs * 64)
    comp         = cfg["output"]["compression"]
    warmup       = cfg["benchmark"]["warmup_batches"]
    n = len(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("uuid", pa.string()),
        ("embedding", pa.list_(pa.float16(), TOTAL_DIM)),
    ])

    print(f"\n[warmup] {warmup} batches ...")
    dummy = [rows[i][COLS[0][0]] or "" for i in range(min(bs * warmup, n))]
    model.encode(dummy, batch_size=bs, max_length=ml,
                 return_dense=True, return_sparse=False,
                 return_colbert_vecs=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n[embed] {n:,} rows x {len(COLS)} cols = {n * len(COLS):,} texts")
    print(f"        bs={bs}, encode_chunk={encode_chunk:,}, "
          f"out_dim={TOTAL_DIM}")

    path = OUT_DIR / "embeddings_percol.parquet"
    writer = pq.ParquetWriter(path, schema, compression=comp)
    rows_done = 0
    t0 = time.perf_counter()

    try:
        for start in range(0, n, encode_chunk):
            end = min(start + encode_chunk, n)
            chunk_rows  = rows[start:end]
            chunk_uuids = uuids[start:end]

            col_vecs = []
            for col_name, label in COLS:
                texts = [r.get(col_name) or "" for r in chunk_rows]
                out = model.encode(
                    texts, batch_size=bs, max_length=ml,
                    return_dense=True, return_sparse=False,
                    return_colbert_vecs=False,
                )
                col_vecs.append(
                    np.asarray(out["dense_vecs"], dtype=np.float16)
                )

            combined = np.concatenate(col_vecs, axis=1)

            tbl = pa.table({
                "uuid": pa.array(chunk_uuids, type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array(combined.reshape(-1), type=pa.float16()),
                    TOTAL_DIM,
                ),
            }, schema=schema)
            writer.write_table(tbl)

            rows_done += len(chunk_rows)
            elapsed = time.perf_counter() - t0
            rate = rows_done / elapsed
            eta = (n - rows_done) / rate if rate > 0 else 0
            print(f"  {rows_done:>7,}/{n}  "
                  f"({rate:6.1f} rows/s, ETA {eta/60:.0f}m)")
    finally:
        writer.close()

    dt = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[done] {dt:.1f}s  ({n/dt:.0f} rows/s)  vram={vram:.2f}GB")
    print(f"[done] {path}  ({path.stat().st_size/1e9:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent.parent / "config.toml")
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f"config not found: {args.config}")
    cfg = load_config(args.config)
    print(f"[init] config: {args.config}")

    env_check(cfg["runtime"]["device"])
    uuids, rows = load_samples(cfg)

    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = _DTYPE_MAP[cfg["model"]["dtype"]]

    print(f"\n[model] loading {cfg['model']['name']} ({cfg['model']['dtype']}) ...")
    model = BGEM3FlagModel(
        cfg["model"]["name"],
        use_fp16=False,
        devices=[cfg["runtime"]["device"]],
    )
    model.model = model.model.to(dtype)
    model.model = torch.compile(model.model, mode="reduce-overhead")

    embed_and_save(uuids, rows, model, cfg)


if __name__ == "__main__":
    main()
