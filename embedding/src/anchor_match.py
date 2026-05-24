"""
anchor_match.py — 소비태그 Hungarian 자동 레이블링  [파이프라인 Step 5]
작성: 2026-05-21
입력: resource/outputs/consumption_cluster_n{N}.csv  — uuid + consumption_tag (Step 4 산출물)
      resource/outputs/consumption_emb_n{N}.parquet   — 2048-dim 소비 임베딩 (Step 2 산출물)
      anchors_consumption.ANCHORS                      — 5개 anchor 텍스트 (한국어)
출력: resource/outputs/consumption_labeled_n{N}.csv   — + consumption_label 열 추가
      resource/outputs/anchor_mapping_n{N}.json        — cluster_id → 레이블 매핑
      resource/outputs/anchor_sim_n{N}.csv             — 5×5 코사인 유사도 행렬
연산:
  1. 소비 임베딩 cluster centroid (2048-dim 평균) - data_mean → L2 정규화
  2. anchor 텍스트 BGE-M3 percol (culinary + hobbies) - data_mean → L2
  3. 코사인 유사도 (5 anchor × 5 cluster)
  4. scipy linear_sum_assignment (Hungarian) → 최대 합 1:1 매핑
  data_mean 차감은 BGE-M3 이방성 보정 (Li 2020, Su 2021)

Usage:
    uv run src/anchor_match.py
    uv run src/anchor_match.py --sample 50000 --margin 0.05
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
from scipy.optimize import linear_sum_assignment

from anchors_consumption import ANCHORS, EMBED_COLS, LABELS

ROOT          = Path(__file__).parent.parent
OUT_DIR       = ROOT / "resource" / "outputs"
SCRIPT_CFG    = Path(__file__).with_suffix(".toml")


# ── 공통 유틸 ──────────────────────────────────────────────────────────────────

def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-10, None)


def load_cfg(cfg_path: Path) -> dict:
    import tomllib
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def load_script_cfg() -> dict:
    import tomllib
    with SCRIPT_CFG.open("rb") as f:
        return tomllib.load(f)


def load_model(cfg: dict):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[cfg["model"]["dtype"]]
    model = BGEM3FlagModel(cfg["model"]["name"], use_fp16=False,
                           devices=[cfg["runtime"]["device"]])
    model.model = model.model.to(dtype)
    return model


def encode(model, texts: list[str], cfg: dict) -> np.ndarray:
    bs, ml = cfg["model"]["batch_size"], cfg["model"]["max_length"]
    out = model.encode(texts, batch_size=bs, max_length=ml,
                       return_dense=True, return_sparse=False, return_colbert_vecs=False)
    torch.cuda.synchronize()
    return np.asarray(out["dense_vecs"], dtype=np.float32)


# ── 데이터 임베딩 (캐시 우선) ──────────────────────────────────────────────────

def load_text_rows(uuids: list[str], cfg: dict) -> dict[str, dict]:
    """percol 입력용 두 컬럼 텍스트를 uuid 기준으로 수집."""
    name  = cfg["dataset"]["name"]
    cache = cfg["dataset"]["cache_dir"] or None
    target = set(uuids)
    cols   = ["uuid"] + EMBED_COLS
    print(f"[text] {name} ({len(target):,} targets) ...")
    ds = load_dataset(name, split="train", cache_dir=cache).select_columns(cols)

    def to_str(v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return v or ""

    result: dict[str, dict] = {}
    for row in ds:
        u = row["uuid"]
        if u in target:
            result[u] = {c: to_str(row.get(c)) for c in EMBED_COLS}
        if len(result) >= len(target):
            break
    print(f"[text] matched {len(result):,} / {len(uuids):,}")
    return result


def compute_data_emb(uuids: list[str], model, cfg: dict) -> np.ndarray:
    """uuid 순서대로 (N, 2048) 임베딩 계산."""
    texts = load_text_rows(uuids, cfg)
    col_vecs = []
    for col in EMBED_COLS:
        col_texts = [texts.get(u, {}).get(col, "") or "" for u in uuids]
        t0 = time.perf_counter()
        print(f"[embed] {col} ({len(col_texts):,}) ...")
        col_vecs.append(encode(model, col_texts, cfg))
        print(f"  done {time.perf_counter()-t0:.1f}s")
    return np.concatenate(col_vecs, axis=1)


def save_cache(path: Path, uuids: list[str], emb: np.ndarray) -> None:
    dim = emb.shape[1]
    table = pa.table({
        "uuid":      pa.array(uuids, type=pa.string()),
        "embedding": pa.array([row.tolist() for row in emb],
                              type=pa.list_(pa.float32(), dim)),
    })
    pq.write_table(table, path, compression="zstd")
    print(f"[cache] saved {path}")


def load_cache(path: Path, uuids: list[str]) -> np.ndarray | None:
    """캐시에서 uuid 순서대로 임베딩 추출. uuid 누락 시 None 반환."""
    table = pq.read_table(path)
    cached_uuids = table.column("uuid").to_pylist()
    flat = (table.column("embedding").combine_chunks()
                 .flatten().to_numpy(zero_copy_only=False))
    dim = flat.size // len(cached_uuids)
    arr = flat.reshape(-1, dim).astype(np.float32)
    idx = {u: i for i, u in enumerate(cached_uuids)}

    missing = [u for u in uuids if u not in idx]
    if missing:
        print(f"[cache] {len(missing)} uuids missing → re-embed")
        return None

    out = np.zeros((len(uuids), dim), dtype=np.float32)
    for i, u in enumerate(uuids):
        out[i] = arr[idx[u]]
    print(f"[cache] {path} loaded ({out.shape})")
    return out


# ── anchor 임베딩 ──────────────────────────────────────────────────────────────

def embed_anchors(model, cfg: dict) -> np.ndarray:
    """5 anchor → percol(culinary, hobbies) → concat → (5, 2048)"""
    col_vecs = []
    for col in EMBED_COLS:
        texts = [ANCHORS[label][col] for label in LABELS]
        col_vecs.append(encode(model, texts, cfg))
    return np.concatenate(col_vecs, axis=1)


# ── 매칭 ───────────────────────────────────────────────────────────────────────

def cluster_centroids(
    emb: np.ndarray, labels: np.ndarray, k: int,
    subtract: np.ndarray | None = None,
) -> np.ndarray:
    """(k, dim) 평균 [- subtract] → L2 정규화"""
    centroids = np.zeros((k, emb.shape[1]), dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids[c] = emb[mask].mean(axis=0)
    if subtract is not None:
        centroids = centroids - subtract
    return l2_norm(centroids)


def hungarian_assign(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_anchor × n_cluster) 유사도 행렬에서 최대 합 매칭."""
    cost = 1.0 - sim
    return linear_sum_assignment(cost)


def print_sim_matrix(sim: np.ndarray, k: int) -> None:
    header = f"  {'anchor \\\\ cluster':18s}" + "".join(f"  c{c}    " for c in range(k))
    print(f"\n{header}")
    print("  " + "─" * (18 + 8 * k))
    for i, label in enumerate(LABELS):
        row = "".join(f"  {sim[i, c]:+.4f}" for c in range(k))
        print(f"  {label:16s}  {row}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    script_cfg = load_script_cfg()
    m_cfg      = script_cfg["matching"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",  type=int,   default=50_000)
    ap.add_argument("--k",       type=int,   default=m_cfg["k"])
    ap.add_argument("--csv",     type=Path,  default=None,
                    help="입력 CSV (기본: outputs/consumption_cluster_n{sample}.csv)")
    ap.add_argument("--config",  type=Path,  default=ROOT / "config.toml",
                    help="BGE-M3·dataset 공용 설정 (기본: 루트 config.toml)")
    ap.add_argument("--margin",  type=float, default=m_cfg["margin"],
                    help="top-1 vs top-2 코사인 마진 임계 (config.matching.margin)")
    ap.add_argument("--center",  action=argparse.BooleanOptionalAction, default=m_cfg["center"],
                    help="BGE-M3 이방성 보정 — 데이터 평균 빼고 cosine (config.matching.center)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. cluster assignments
    csv_path = args.csv or OUT_DIR / f"consumption_cluster_n{args.sample}.csv"
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) != args.sample:
        print(f"[warn] CSV rows ({len(df)}) ≠ --sample ({args.sample})")
    uuids   = df["uuid"].tolist()
    cluster = df["consumption_tag"].to_numpy()

    # 2. 데이터 임베딩 — 캐시 우선
    cache_path = OUT_DIR / f"consumption_emb_n{args.sample}.parquet"
    data_emb = load_cache(cache_path, uuids) if cache_path.exists() else None

    cfg   = load_cfg(args.config)
    model = load_model(cfg)

    if data_emb is None:
        data_emb = compute_data_emb(uuids, model, cfg)
        save_cache(cache_path, uuids, data_emb)

    # 3. 데이터 평균 (이방성 보정용)
    data_mean = data_emb.mean(axis=0).astype(np.float32) if args.center else None
    if args.center:
        print(f"[center]    data_mean ||·||={np.linalg.norm(data_mean):.4f}")

    # 4. 클러스터 centroid
    centroids = cluster_centroids(data_emb, cluster, args.k, subtract=data_mean)
    print(f"[centroids] {centroids.shape}")
    del data_emb; gc.collect()

    # 5. anchor 임베딩 (동일 평균으로 중심화)
    anchors_raw = embed_anchors(model, cfg)
    if data_mean is not None:
        anchors_raw = anchors_raw - data_mean
    anchors = l2_norm(anchors_raw)
    print(f"[anchors]   {anchors.shape}")

    # 6. 코사인 유사도 (cosine = 정규화 후 내적)
    sim = anchors @ centroids.T  # (5, 5)
    print_sim_matrix(sim, args.k)

    # 7. Hungarian
    a_idx, c_idx = hungarian_assign(sim)

    # 8. 매핑 + low-confidence
    print(f"\n[mapping]  (margin threshold = {args.margin:.3f})")
    print("  " + "─" * 60)
    mapping: dict[int, str] = {}
    low_conf: list[str] = []
    for ai, ci in zip(a_idx, c_idx):
        label  = LABELS[ai]
        s      = float(sim[ai, ci])
        sorted_s = np.sort(sim[ai])[::-1]
        margin = float(sorted_s[0] - sorted_s[1])
        flag   = " ⚠ low-conf" if margin < args.margin else ""
        if flag:
            low_conf.append(f"cluster {ci} ({label})")
        print(f"  cluster {ci} → {label:14s}  cos={s:+.4f}  margin={margin:.4f}{flag}")
        mapping[int(ci)] = label

    # 9. 저장
    df["consumption_label"] = df["consumption_tag"].map(mapping)
    out_csv = OUT_DIR / f"consumption_labeled_n{args.sample}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")

    map_path = OUT_DIR / f"anchor_mapping_n{args.sample}.json"
    with map_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"[save] {map_path}")

    sim_path = OUT_DIR / f"anchor_sim_n{args.sample}.csv"
    pd.DataFrame(sim, index=LABELS,
                 columns=[f"cluster{c}" for c in range(args.k)]).to_csv(sim_path)
    print(f"[save] {sim_path}")

    if low_conf:
        print(f"\n[warn] low-confidence: {len(low_conf)} — anchor 재합성 검토")
        for entry in low_conf:
            print(f"  - {entry}")


if __name__ == "__main__":
    main()
