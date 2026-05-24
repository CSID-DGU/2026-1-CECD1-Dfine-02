"""
embed_percol5.py — percol5 BGE-M3 임베딩 생성  [파이프라인 Step 1]
작성: 2026-05
입력: nvidia/Nemotron-Personas-Korea (HF)  — config.toml n_bench 행
출력: resource/embeddings_percol5/embeddings_percol5.parquet  (uuid, float16[5120])
연산: 5 AIO 칼럼(career/professional/family/travel/hobbies) 각각 BGE-M3 dense 1024-dim
     → column-wise concat → 5120-dim percol 임베딩

비교 대상 (legacy/):
  embed.py       — 5칼럼 텍스트 concat → 단일 1024-dim
  embed_percol.py — 7칼럼 각각 → concat 7168-dim

Usage:
    uv run src/embed_percol5.py
    uv run src/embed_percol5.py --config custom.toml
"""

import argparse
import sys
import time
import tomllib
from pathlib import Path

import pandas as pd
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
    ("hobbies_and_interests",      "취미와 관심사"),
]

EMBED_DIM = 1024
TOTAL_DIM = len(COLS) * EMBED_DIM  # 5120
OUT_DIR   = Path("resource/embeddings_percol5")


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


def _load_age_stratified(ds, n: int, seed: int) -> tuple[list[str], list[dict]]:
    """age 3밴드 평탄 샘플링 (19-29 / 30-49 / 50-69).

    한 밴드 인구가 균등 정원(n/3)에 미달하면 그 미달분을 잔여 밴드로 이월해
    전체 합이 n에 도달하도록 한다 (데이터셋 전체 < n 인 경우만 예외).
    """
    col_keys_age = ["uuid", "age"] + [c for c, _ in COLS]
    ds = ds.select_columns(col_keys_age)
    df = ds.to_pandas()

    def age_band(a: int) -> str:
        if 19 <= a <= 29: return "19-29"
        if 30 <= a <= 49: return "30-49"
        if 50 <= a <= 69: return "50-69"
        return "other"

    df["_band"] = df["age"].apply(age_band)
    bands  = ["19-29", "30-49", "50-69"]
    groups = {b: df[df["_band"] == b] for b in bands}
    sizes  = {b: len(g) for b, g in groups.items()}
    total  = sum(sizes.values())
    if total < n:
        print(f"[warn]  19-69 합계 {total:,} < n_bench {n:,} — 전체 사용")

    # 평탄화 잔여 재할당: 작은 밴드부터 처리, 미달분을 다음 밴드 share 에 합산
    remaining_n = min(n, total)
    allocations: dict[str, int] = {}
    for i, b in enumerate(sorted(bands, key=lambda x: sizes[x])):
        share = remaining_n // (len(bands) - i)
        take  = min(share, sizes[b])
        allocations[b] = take
        remaining_n   -= take

    samples = []
    for b in bands:
        k = allocations[b]
        samples.append(groups[b].sample(n=k, random_state=seed))
        print(f"[load]   band {b}: {k:,} / {sizes[b]:,}")

    df = (pd.concat(samples)
            .drop(columns=["_band", "age"])
            .reset_index(drop=True))
    print(f"[load] age_stratified — {len(df):,} rows (target n={n:,})")

    uuids, rows = [], []
    for row in df.itertuples(index=False):
        uuids.append(row.uuid)
        rows.append({c: getattr(row, c) for c, _ in COLS})
    return uuids, rows


def load_samples(cfg: dict) -> tuple[list[str], list[dict]]:
    """데이터셋 로드 → (uuids, rows). rows는 5칼럼 dict 리스트."""
    strategy = cfg["sampling"]["strategy"]
    name     = cfg["dataset"]["name"]
    seed     = cfg["dataset"]["seed"]
    n        = cfg["benchmark"]["n_bench"]
    cache    = cfg["dataset"]["cache_dir"] or None

    ds = load_dataset(name, split="train", cache_dir=cache)

    if strategy == "age_stratified":
        return _load_age_stratified(ds, n, seed)

    col_keys = ["uuid"] + [c for c, _ in COLS]
    ds = ds.select_columns(col_keys)
    if strategy == "random":
        ds = ds.shuffle(seed=seed).select(range(n))
    elif strategy == "full":
        pass
    else:
        sys.exit(f"[load] 지원하지 않는 strategy: {strategy!r}")

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

    path = OUT_DIR / "embeddings_percol5.parquet"
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
