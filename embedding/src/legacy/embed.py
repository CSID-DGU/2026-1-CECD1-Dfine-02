"""
BGE-M3 임베딩 파이프라인 — Nemotron-Personas-Korea 1M 샘플.

계획서: AIO_임베딩_파이프라인_계획.md v2.2
원본: bge-m3-bench/bench.py (worst-case throughput 측정)에서 전환.

변경 요약:
    - row_to_text: 26컬럼 전체 → 5컬럼 prefix concat (계획서 §4.2)
    - load_texts → load_samples: sampling_strategy 분기 (§11.3–11.5)
    - schema: id (int32) → uuid (string) (§4.3)
    - stage3_extrapolate 제거 (단일 처리 — extrapolation 불필요)

Usage:
    uv run src/embed.py
    uv run src/embed.py --config custom.toml
"""

import argparse
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from FlagEmbedding import BGEM3FlagModel


# --- 5컬럼 정의 (계획서 §4.2) ---------------------------------------------

COLS: list[tuple[str, str]] = [
    ("career_goals_and_ambitions", "커리어 목표"),
    ("professional_persona",       "직업적 자아"),
    ("family_persona",             "가족적 자아"),
    ("travel_persona",             "여행 성향"),
    ("hobbies_and_interests",      "취미와 관심사"),
]


# --- config loading -------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


# --- helpers --------------------------------------------------------------

def env_check(device: str) -> None:
    assert torch.cuda.is_available(), "CUDA unavailable"
    dev_idx = int(device.split(":")[1]) if ":" in device else 0
    cap = torch.cuda.get_device_capability(dev_idx)
    name = torch.cuda.get_device_name(dev_idx)
    print(f"[env] {name} sm_{cap[0]}{cap[1]} | "
          f"torch {torch.__version__} | cuda {torch.version.cuda}")
    if cap < (12, 0):
        print(f"[env] WARN: expected Blackwell sm_120, got sm_{cap[0]}{cap[1]} "
              "— ensure torch wheel is cu130")


def row_to_text(row: dict) -> str:
    """5개 선정 컬럼을 prefix 레이블 방식으로 결합 (계획서 §4.2)."""
    parts = [f"{label}: {row[col]}" for col, label in COLS if row.get(col)]
    return " | ".join(parts)


# --- sampling dispatcher (계획서 §11.3–11.5) ------------------------------

def load_samples(cfg: dict) -> tuple[list[str], list[str]]:
    """sampling_strategy에 따라 분기. (uuids, texts) 튜플 반환."""
    strategy = cfg["sampling"]["strategy"]
    if strategy == "full":
        ds = _load_full(cfg)
    elif strategy == "random":
        ds = _load_random(cfg)
    elif strategy == "age_stratified":
        ds = _load_age_stratified(cfg)
    else:
        raise ValueError(f"unknown sampling_strategy: {strategy!r}")

    uuids: list[str] = []
    texts: list[str] = []
    for row in ds:
        uuids.append(row["uuid"])
        texts.append(row_to_text(row))
    return uuids, texts


def _load_full(cfg: dict):
    """분기 full — 전체 데이터셋 로드 (샘플링 없음)."""
    name  = cfg["dataset"]["name"]
    cache = cfg["dataset"]["cache_dir"] or None
    cols  = ["uuid"] + [c for c, _ in COLS]

    ds = load_dataset(name, split="train", cache_dir=cache)
    ds = ds.select_columns(cols)
    print(f"[load] full dataset — {len(ds):,} rows")
    return ds


def _load_random(cfg: dict):
    """분기 A — 무작위 1M (계획서 §11.4)."""
    n     = cfg["benchmark"]["n_bench"]
    name  = cfg["dataset"]["name"]
    seed  = cfg["dataset"]["seed"]
    cache = cfg["dataset"]["cache_dir"] or None
    cols  = ["uuid"] + [c for c, _ in COLS]

    if cfg["dataset"]["streaming"]:
        print("[load] WARN: streaming=true ignored — random uses full download "
              "for strict shuffle (buffered shuffle is not uniformly random)")

    print(f"[load] random sampling — {n:,} rows")
    ds = load_dataset(name, split="train", cache_dir=cache)
    ds = ds.select_columns(cols)
    return ds.shuffle(seed=seed).select(range(n))


def _load_age_stratified(cfg: dict):
    """분기 B — age 균등화 stratified 1M, 19-69세만 (계획서 §11.5)."""
    n     = cfg["benchmark"]["n_bench"]
    name  = cfg["dataset"]["name"]
    seed  = cfg["dataset"]["seed"]
    cache = cfg["dataset"]["cache_dir"] or None
    cols  = ["uuid", "age"] + [c for c, _ in COLS]

    if cfg["dataset"]["streaming"]:
        print("[load] WARN: streaming=true ignored — stratified requires full download")

    print(f"[load] age-stratified sampling — {n:,} rows (19-69, 3 bands)")
    ds = load_dataset(name, split="train", cache_dir=cache)
    ds = ds.select_columns(cols)
    df = ds.to_pandas()

    def age_band(a: int) -> str:
        if 19 <= a <= 29: return "19-29"
        if 30 <= a <= 49: return "30-49"
        if 50 <= a <= 69: return "50-69"
        return "other"

    df["_band"] = df["age"].apply(age_band)
    target = n // 3

    samples = []
    for band in ["19-29", "30-49", "50-69"]:
        group = df[df["_band"] == band]
        k = min(target, len(group))
        samples.append(group.sample(n=k, random_state=seed))
        print(f"[load]   band {band}: {k:,} rows  (group total {len(group):,})")

    balanced = (pd.concat(samples)
                  .drop(columns=["_band", "age"])
                  .reset_index(drop=True))
    return Dataset.from_pandas(balanced)


# --- stage 1: token-length distribution -----------------------------------

def stage1_distribution(texts: list[str], tokenizer, max_length: int) -> np.ndarray:
    print(f"\n[stage1] tokenize-only, {len(texts)} rows ...")
    t0 = time.perf_counter()
    lens = [len(tokenizer.encode(t, add_special_tokens=True, truncation=False))
            for t in texts]
    dt = time.perf_counter() - t0
    a = np.array(lens)
    print(f"[stage1] cpu tokenize  {dt:.2f}s ({len(texts)/dt:.0f} rows/s)")
    print(f"[stage1] tokens/row    mean={a.mean():.0f}  "
          f"p50={np.percentile(a,50):.0f}  p95={np.percentile(a,95):.0f}  "
          f"p99={np.percentile(a,99):.0f}  max={a.max()}")
    over = int((a > max_length).sum())
    print(f"[stage1] rows > {max_length}: {over} ({over/len(a)*100:.2f}%) "
          "→ truncated to max_length")
    return a


# --- stage 2: embed + write -----------------------------------------------

def stage2_throughput(uuids: list[str], texts: list[str], model, cfg: dict) -> dict:
    bs           = cfg["model"]["batch_size"]
    encode_chunk = cfg["model"].get("encode_chunk", bs * 64)
    ml           = cfg["model"]["max_length"]
    out_dir      = Path(cfg["output"]["parquet_dir"])
    chunk        = cfg["output"]["chunk_size"]
    comp         = cfg["output"]["compression"]
    warmup       = cfg["benchmark"]["warmup_batches"]
    n = len(texts)
    assert len(uuids) == n, "uuids/texts length mismatch"

    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("uuid", pa.string()),
        ("embedding", pa.list_(pa.float16(), 1024)),  # FixedSizeList — native numpy 변환
    ])

    print(f"\n[stage2] warmup ({warmup} batches) ...")
    _ = model.encode(texts[:bs * warmup], batch_size=bs, max_length=ml,
                     return_dense=True, return_sparse=False,
                     return_colbert_vecs=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    batch_times: list[float] = []
    encode_s, write_s = 0.0, 0.0
    rows_done = 0

    print(f"[stage2] embedding {n} rows, bs={bs}, encode_chunk={encode_chunk:,}, "
          f"max_len={ml}, chunk={chunk:,} rows/file")
    t_wall0 = time.perf_counter()

    writer: pq.ParquetWriter | None = None
    current_chunk_start = -1

    try:
        for start in range(0, n, encode_chunk):
            e_end   = min(start + encode_chunk, n)
            e_texts = texts[start:e_end]
            e_uuids = uuids[start:e_end]

            chunk_start = (start // chunk) * chunk
            if chunk_start != current_chunk_start:
                if writer is not None:
                    writer.close()
                path = out_dir / f"embeddings_{chunk_start:010d}.parquet"
                writer = pq.ParquetWriter(path, schema, compression=comp)
                current_chunk_start = chunk_start
                print(f"[stage2]   → new file: {path.name}")

            # encode_chunk 전체를 한 번에 전달 — FlagEmbedding이 bs 단위로 내부 배치 처리
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = model.encode(e_texts, batch_size=bs, max_length=ml,
                               return_dense=True, return_sparse=False,
                               return_colbert_vecs=False)
            ev1.record()
            torch.cuda.synchronize()
            dt_gpu = ev0.elapsed_time(ev1) / 1000.0
            encode_s += dt_gpu
            batch_times.append(dt_gpu)

            vecs = np.asarray(out["dense_vecs"], dtype=np.float16)
            t_w = time.perf_counter()
            tbl = pa.table({
                "uuid": pa.array(e_uuids, type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array(vecs.reshape(-1), type=pa.float16()), 1024
                ),
            }, schema=schema)
            writer.write_table(tbl)
            write_s += time.perf_counter() - t_w

            rows_done += len(e_texts)
            rate = rows_done / (time.perf_counter() - t_wall0)
            print(f"[stage2]   {rows_done:>7,}/{n}  ({rate:6.1f} rows/s)")
    finally:
        if writer is not None:
            writer.close()

    dt_wall = time.perf_counter() - t_wall0
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    bt = np.array(batch_times)
    other_s = dt_wall - encode_s - write_s

    print(f"\n[stage2] DONE")
    print(f"  wall          {dt_wall:8.2f} s   ({n/dt_wall:.1f} rows/s)")
    print(f"  gpu encode    {encode_s:8.2f} s   ({encode_s/dt_wall*100:5.1f}%)")
    print(f"  parquet write {write_s:8.2f} s   ({write_s/dt_wall*100:5.1f}%)")
    print(f"  cpu + io      {other_s:8.2f} s   ({other_s/dt_wall*100:5.1f}%)")
    print(f"  batch lat     p50={np.percentile(bt,50)*1000:6.0f}ms  "
          f"p95={np.percentile(bt,95)*1000:6.0f}ms  "
          f"p99={np.percentile(bt,99)*1000:6.0f}ms")
    print(f"  vram peak     {vram_peak:5.2f} GB")

    return {
        "wall_s": dt_wall,
        "rps": n / dt_wall,
        "p99_batch_s": float(np.percentile(bt, 99)),
        "vram_gb": vram_peak,
    }


# --- main -----------------------------------------------------------------

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

    print(f"\n[load] {cfg['benchmark']['n_bench']:,} rows from "
          f"{cfg['dataset']['name']}  (strategy={cfg['sampling']['strategy']})")
    uuids, texts = load_samples(cfg)
    chars = np.array([len(t) for t in texts])
    print(f"[load] chars/row  mean={chars.mean():.0f}  "
          f"p99={np.percentile(chars,99):.0f}  max={chars.max()}")

    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = _DTYPE_MAP[cfg["model"]["dtype"]]

    print(f"\n[model] loading {cfg['model']['name']} ({cfg['model']['dtype']}) ...")
    model = BGEM3FlagModel(
        cfg["model"]["name"],
        use_fp16=False,   # 정밀도는 model.dtype + .to(dtype) 한 곳에서만 제어
        devices=[cfg["runtime"]["device"]],
    )
    model.model = model.model.to(dtype)
    model.model = torch.compile(model.model, mode="reduce-overhead")

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    _ = stage1_distribution(
        texts[:cfg["benchmark"]["n_pilot"]],
        tok,
        cfg["model"]["max_length"],
    )

    stage2_throughput(uuids, texts, model, cfg)

    out_dir = Path(cfg["output"]["parquet_dir"])
    files = sorted(out_dir.glob("*.parquet"))
    total = sum(pq.read_metadata(p).num_rows for p in files)
    print(f"\n[verify] {len(files)} file(s) written, total rows: {total}")
    for p in files:
        print(f"  {p.name}  ({pq.read_metadata(p).num_rows:,} rows, "
              f"{p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
