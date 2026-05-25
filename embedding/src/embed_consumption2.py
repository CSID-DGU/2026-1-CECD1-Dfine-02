"""
embed_consumption2.py — hobbies 항목별 임베딩 생성  [파이프라인 Step 2-2]
작성: 2026-05
입력: resource/embeddings_percol5/embeddings_percol5.parquet  (UUID 기준)
      nvidia/Nemotron-Personas-Korea (hobbies_and_interests_list)
출력: resource/outputs/hobby_emb_n{N}.parquet
      스키마: uuid (string) + embeddings (list<fixed_size_list<float32>[1024]>)
연산: hobbies_and_interests_list 각 원소를 독립적으로 BGE-M3 1024-dim 임베딩
     → 행마다 가변 개수 임베딩 (항목이 0개인 행은 빈 리스트)

embed_consumption.py 와의 차이:
  - culinary_persona 제외
  - hobbies 텍스트를 join하지 않고 원소별 개별 임베딩
  - 출력 dim: 1024 (고정) × 가변 개수 → FixedSizeList wrapped in List

Usage:
    uv run src/embed_consumption2.py
    uv run src/embed_consumption2.py --sample 50000
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
EMBED_DIM = 1024
HOB_COL   = "hobbies_and_interests_list"


def load_uuids(sample: int) -> list[str]:
    table     = pa_ds.dataset(EMBED_DIR, format="parquet").to_table(columns=["uuid"])
    uuids_all = table.column("uuid").to_pylist()
    n_total   = len(uuids_all)
    del table; gc.collect()

    if sample < n_total:
        idx   = np.sort(np.random.default_rng(SEED).choice(n_total, sample, replace=False))
        uuids = [uuids_all[i] for i in idx]
    else:
        uuids = uuids_all

    print(f"[uuid] {len(uuids):,} (from percol5, n_total={n_total:,})")
    return uuids


def load_hobby_items(uuids: list[str], cfg: dict) -> dict[str, list[str]]:
    """HF 데이터셋에서 UUID 매칭 → hobbies 항목 리스트 dict.

    hobbies_and_interests_list 가 이미 list 타입이므로 join 없이 원소만 추출.
    빈 리스트·None 처리: 빈 리스트로 통일.
    """
    name   = cfg["dataset"]["name"]
    cache  = cfg["dataset"]["cache_dir"] or None
    target = set(uuids)

    print(f"[text] {name} 스캔 중 ({len(target):,} targets) ...")
    ds = load_dataset(name, split="train", cache_dir=cache).select_columns(["uuid", HOB_COL])

    result: dict[str, list[str]] = {}
    for row in ds:
        u = row["uuid"]
        if u in target:
            raw = row.get(HOB_COL) or []
            if isinstance(raw, str):
                raw = [raw] if raw else []
            result[u] = [str(x) for x in raw if x]
        if len(result) >= len(target):
            break

    matched = len(result)
    print(f"[text] matched {matched:,} / {len(uuids):,}")

    # 항목 수 통계
    counts = [len(v) for v in result.values()]
    if counts:
        print(f"[text] hobbies/row — min={min(counts)}, "
              f"mean={sum(counts)/len(counts):.1f}, max={max(counts)}")
    return result


def embed_and_save(
    uuids: list[str],
    hobby_map: dict[str, list[str]],
    model: BGEM3FlagModel,
    cfg: dict,
    out_path: Path,
) -> None:
    bs   = cfg["model"]["batch_size"]
    ml   = cfg["model"]["max_length"]
    comp = cfg["output"]["compression"]
    n    = len(uuids)

    # 가변 길이: List<FixedSizeList<float32>[1024]>
    item_type = pa.list_(pa.float32(), EMBED_DIM)   # FixedSizeList[1024]
    emb_type  = pa.list_(item_type)                  # List<FixedSizeList[1024]>
    schema = pa.schema([
        ("uuid",       pa.string()),
        ("embeddings", emb_type),
    ])

    CHUNK = 50_000
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, schema, compression=comp)
    t0 = time.perf_counter()

    try:
        for start in range(0, n, CHUNK):
            end         = min(start + CHUNK, n)
            chunk_uuids = uuids[start:end]

            # 청크 내 모든 항목을 flat 수집 + 각 UUID의 슬라이스 기록
            all_texts:  list[str]             = []
            row_slices: list[tuple[int, int]] = []
            for u in chunk_uuids:
                s = len(all_texts)
                all_texts.extend(hobby_map.get(u, []))
                row_slices.append((s, len(all_texts)))

            # 전체를 한 번에 임베딩 (GPU 효율 극대화)
            if all_texts:
                out_enc = model.encode(
                    all_texts, batch_size=bs, max_length=ml,
                    return_dense=True, return_sparse=False,
                    return_colbert_vecs=False,
                )
                flat_emb = np.asarray(out_enc["dense_vecs"], dtype=np.float32)
            else:
                flat_emb = np.empty((0, EMBED_DIM), dtype=np.float32)

            # UUID별 재그룹 → list[list[list[float]]]
            row_emb_lists = []
            for s, e in row_slices:
                row_emb_lists.append(flat_emb[s:e].tolist() if e > s else [])

            tbl = pa.table({
                "uuid":       pa.array(chunk_uuids, type=pa.string()),
                "embeddings": pa.array(row_emb_lists, type=emb_type),
            }, schema=schema)
            writer.write_table(tbl)

            elapsed = time.perf_counter() - t0
            rate    = end / elapsed
            eta     = (n - end) / rate if rate > 0 else 0
            n_items = sum(e - s for s, e in row_slices)
            print(f"  {end:>7,}/{n}  items={n_items:,}  "
                  f"({rate:.0f} rows/s, ETA {eta/60:.0f}m)")
    finally:
        writer.close()

    dt   = time.perf_counter() - t0
    size = out_path.stat().st_size / 1e9
    print(f"\n[done] {dt:.1f}s  |  {out_path}  ({size:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="hobbies_and_interests_list 항목별 BGE-M3 임베딩 생성"
    )
    ap.add_argument("--sample", type=int, default=200_000,
                    help="처리할 UUID 수 (percol5 기준)")
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent.parent / "config.toml")
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f"config not found: {args.config}")
    with args.config.open("rb") as f:
        cfg = tomllib.load(f)

    out_path = OUT_DIR / f"hobby_emb_n{args.sample}.parquet"
    if out_path.exists():
        print(f"[skip] 이미 존재: {out_path}")
        print("  덮어쓰려면 파일을 삭제 후 재실행하세요.")
        sys.exit(0)

    uuids     = load_uuids(args.sample)
    hobby_map = load_hobby_items(uuids, cfg)

    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = _DTYPE_MAP[cfg["model"]["dtype"]]
    print(f"\n[model] loading {cfg['model']['name']} ({cfg['model']['dtype']}) ...")
    model = BGEM3FlagModel(cfg["model"]["name"], use_fp16=False,
                           devices=[cfg["runtime"]["device"]])
    model.model = model.model.to(dtype)

    embed_and_save(uuids, hobby_map, model, cfg, out_path)


if __name__ == "__main__":
    main()
