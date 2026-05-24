"""
archetype_match.py — archetype Hungarian 자동 레이블링  [파이프라인 Step 6]
작성: 2026-05-22
입력: resource/outputs/archetype_n{N}.csv                   — uuid + archetype (Step 3 산출물)
      resource/embeddings_percol5/embeddings_percol5.parquet — 5120-dim percol5 임베딩 (Step 1 산출물)
      anchors_archetype.ANCHORS                              — 5개 archetype anchor 텍스트
출력: resource/outputs/archetype_labeled_n{N}.csv           — + archetype_label 열 추가
      resource/outputs/archetype_mapping_n{N}.json          — cluster_id → 레이블 매핑
      resource/outputs/archetype_sim_n{N}.csv               — 5×5 코사인 유사도 행렬
연산:
  1. percol5 5120-dim centroid (cluster별 평균) - data_mean → L2 정규화
  2. anchor 텍스트 BGE-M3 percol (5 AIO 칼럼) - data_mean → L2
  3. 코사인 유사도 (5 anchor × 5 cluster)
  4. scipy linear_sum_assignment (Hungarian) → 최대 합 1:1 매핑
  data_mean 차감은 BGE-M3 이방성 보정 (Li 2020, Su 2021)
  percol5 해제 후 BGE-M3 로드 — 메모리 순차 확보

Usage:
    uv run src/archetype_match.py
    uv run src/archetype_match.py --sample 200000 --margin 0.05
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import torch
from FlagEmbedding import BGEM3FlagModel
from scipy.optimize import linear_sum_assignment

from anchors_archetype import ANCHORS, EMBED_COLS, LABELS

ROOT       = Path(__file__).parent.parent
OUT_DIR    = ROOT / "resource" / "outputs"
EMBED_DIR  = ROOT / "resource" / "embeddings_percol5"
SCRIPT_CFG = Path(__file__).with_suffix(".toml")


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


# ── percol5 로드 ────────────────────────────────────────────────────────────────

def load_percol5_subset(uuids: list[str]) -> np.ndarray:
    """percol5 parquet에서 uuid 순서대로 (N, 5120) float32 추출."""
    print(f"[percol5] loading {EMBED_DIR} ...")
    table    = pa_ds.dataset(EMBED_DIR, format="parquet").to_table(
                   columns=["uuid", "embedding"])
    n_total  = len(table)
    all_uids = table.column("uuid").to_pylist()
    flat     = (table.column("embedding").combine_chunks()
                     .flatten().to_numpy(zero_copy_only=False))
    del table; gc.collect()

    dim = flat.size // n_total
    arr = flat.reshape(n_total, dim).astype(np.float32)  # float16 → float32
    del flat; gc.collect()

    idx = {u: i for i, u in enumerate(all_uids)}
    missing = [u for u in uuids if u not in idx]
    if missing:
        raise RuntimeError(
            f"{len(missing)} uuid가 percol5에 없음 — Step 1(embed_percol5.py) 먼저 실행하세요"
        )

    out = np.zeros((len(uuids), dim), dtype=np.float32)
    for i, u in enumerate(uuids):
        out[i] = arr[idx[u]]
    del arr; gc.collect()

    print(f"[percol5] subset {out.shape}")
    return out


# ── anchor 임베딩 ──────────────────────────────────────────────────────────────

def embed_anchors(model, cfg: dict) -> np.ndarray:
    """5 anchor → percol(5 AIO 칼럼) → concat → (5, 5120)"""
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
    ap.add_argument("--sample",  type=int,   default=200_000)
    ap.add_argument("--k",       type=int,   default=m_cfg["k"])
    ap.add_argument("--config",  type=Path,  default=ROOT / "config.toml",
                    help="BGE-M3·dataset 공용 설정 (기본: 루트 config.toml)")
    ap.add_argument("--margin",  type=float, default=m_cfg["margin"],
                    help="top-1 vs top-2 코사인 마진 임계 (config.matching.margin)")
    ap.add_argument("--center",  action=argparse.BooleanOptionalAction, default=m_cfg["center"],
                    help="BGE-M3 이방성 보정 — 데이터 평균 빼고 cosine (config.matching.center)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. cluster assignments
    csv_path = OUT_DIR / f"archetype_n{args.sample}.csv"
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) != args.sample:
        print(f"[warn] CSV rows ({len(df)}) ≠ --sample ({args.sample})")
    uuids   = df["uuid"].tolist()
    cluster = df["archetype"].to_numpy()

    # 2. percol5 임베딩 로드
    emb = load_percol5_subset(uuids)

    # 3. 데이터 평균 (이방성 보정용)
    data_mean = emb.mean(axis=0).astype(np.float32) if args.center else None
    if args.center:
        print(f"[center]    data_mean ||·||={np.linalg.norm(data_mean):.4f}")

    # 4. 클러스터 centroid — percol5 해제 전에 계산
    centroids = cluster_centroids(emb, cluster, args.k, subtract=data_mean)
    print(f"[centroids] {centroids.shape}")
    del emb; gc.collect()

    # 5. anchor 임베딩 (percol5 해제 후 BGE-M3 로드)
    cfg         = load_cfg(args.config)
    model       = load_model(cfg)
    anchors_raw = embed_anchors(model, cfg)
    del model; gc.collect()
    torch.cuda.empty_cache()

    if data_mean is not None:
        anchors_raw = anchors_raw - data_mean
    anchors = l2_norm(anchors_raw)
    print(f"[anchors]   {anchors.shape}")

    # 6. 코사인 유사도
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
        label    = LABELS[ai]
        s        = float(sim[ai, ci])
        sorted_s = np.sort(sim[ai])[::-1]
        margin   = float(sorted_s[0] - sorted_s[1])
        flag     = " ⚠ low-conf" if margin < args.margin else ""
        if flag:
            low_conf.append(f"cluster {ci} ({label})")
        print(f"  cluster {ci} → {label:14s}  cos={s:+.4f}  margin={margin:.4f}{flag}")
        mapping[int(ci)] = label

    # 9. 저장
    df["archetype_label"] = df["archetype"].map(mapping)
    out_csv = OUT_DIR / f"archetype_labeled_n{args.sample}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")

    map_path = OUT_DIR / f"archetype_mapping_n{args.sample}.json"
    with map_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"[save] {map_path}")

    sim_path = OUT_DIR / f"archetype_sim_n{args.sample}.csv"
    pd.DataFrame(sim, index=LABELS,
                 columns=[f"cluster{c}" for c in range(args.k)]).to_csv(sim_path)
    print(f"[save] {sim_path}")

    if low_conf:
        print(f"\n[warn] low-confidence: {len(low_conf)} — anchor 재합성 검토")
        for entry in low_conf:
            print(f"  - {entry}")


if __name__ == "__main__":
    main()
