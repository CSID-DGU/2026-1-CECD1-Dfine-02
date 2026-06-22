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
  1. percol5 청크 read → cluster별 centroid + data_mean 직접 누적 (fp64)
  2. anchor 텍스트 BGE-M3 percol (5 AIO 칼럼) - data_mean → L2
  3. 코사인 유사도 (5 anchor × 5 cluster)
  4. scipy linear_sum_assignment (Hungarian) → 최대 합 1:1 매핑
  data_mean 차감은 BGE-M3 이방성 보정 (Li 2020, Su 2021)
  풀 임베딩 (N×5120 fp32) 적재 회피 — 청크 단위 누적으로 메모리 < 1GB

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


# ── percol5 청크 누적 ───────────────────────────────────────────────────────────

def stream_centroids(
    uuids: list[str], cluster: np.ndarray, k: int, center: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """percol5 parquet 청크 read → cluster별 centroid + 전체 평균 (fp64 누적).

    풀 (N, 5120) fp32 적재를 피하고 청크당 fp64 sum 만 누적한다.
    centroid 는 (sum_c / count_c), data_mean 은 (sum_all / n_kept) 로 산출 —
    원본 `emb[mask].mean()` 과 수학적으로 동일.
    """
    print(f"[percol5] streaming {EMBED_DIR} (chunked) ...")
    uuid_to_pos = {u: i for i, u in enumerate(uuids)}
    n_subset    = len(uuids)

    BATCH    = 20_000
    ds       = pa_ds.dataset(EMBED_DIR, format="parquet")
    sum_c:   np.ndarray | None = None   # (k, dim) fp64
    count_c                     = np.zeros(k, dtype=np.int64)
    sum_all: np.ndarray | None = None   # (dim,) fp64
    n_seen                      = 0

    for batch in ds.to_batches(columns=["uuid", "embedding"], batch_size=BATCH):
        batch_uids = batch.column("uuid").to_pylist()
        n_b        = len(batch_uids)

        kept_local: list[int] = []
        kept_cls:   list[int] = []
        for j, u in enumerate(batch_uids):
            i = uuid_to_pos.get(u)
            if i is not None:
                kept_local.append(j)
                kept_cls.append(int(cluster[i]))
        if not kept_local:
            continue

        flat = (batch.column("embedding").flatten()
                     .to_numpy(zero_copy_only=False))
        dim  = flat.size // n_b
        if sum_c is None:
            sum_c   = np.zeros((k, dim), dtype=np.float64)
            sum_all = np.zeros(dim,      dtype=np.float64)

        emb_kept = flat.reshape(n_b, dim)[kept_local].astype(np.float64)
        cls_arr  = np.asarray(kept_cls)
        for c in range(k):
            m = cls_arr == c
            if m.any():
                sum_c[c]   += emb_kept[m].sum(axis=0)
                count_c[c] += int(m.sum())
        if center:
            sum_all += emb_kept.sum(axis=0)
        n_seen += len(emb_kept)

    if sum_c is None:
        raise RuntimeError(
            "percol5 에서 subset 매칭 행 0개 — Step 1(embed_percol5.py) 먼저 실행하세요"
        )
    if n_seen < n_subset:
        print(f"[warn] subset {n_subset:,} 중 {n_subset - n_seen:,}개 percol5에 없음")

    centroids = (sum_c / np.maximum(count_c[:, None], 1)).astype(np.float32)
    data_mean = ((sum_all / max(n_seen, 1)).astype(np.float32)
                 if center else None)
    print(f"[percol5] centroids {centroids.shape} (n_kept={n_seen:,})")
    return centroids, data_mean


# ── anchor 임베딩 ──────────────────────────────────────────────────────────────

def embed_anchors(model, cfg: dict) -> np.ndarray:
    """5 anchor → percol(5 AIO 칼럼) → concat → (5, 5120)"""
    col_vecs = []
    for col in EMBED_COLS:
        texts = [ANCHORS[label][col] for label in LABELS]
        col_vecs.append(encode(model, texts, cfg))
    return np.concatenate(col_vecs, axis=1)


# ── 매칭 ───────────────────────────────────────────────────────────────────────

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

    # 2. percol5 청크 read → centroid + data_mean 직접 누적
    centroids, data_mean = stream_centroids(uuids, cluster, args.k, args.center)
    if args.center:
        print(f"[center]    data_mean ||·||={np.linalg.norm(data_mean):.4f}")

    # 3. 평균 차감 + L2 정규화
    if data_mean is not None:
        centroids = centroids - data_mean
    centroids = l2_norm(centroids)
    print(f"[centroids] {centroids.shape}")

    # 4. anchor 임베딩
    cfg         = load_cfg(args.config)
    model       = load_model(cfg)
    anchors_raw = embed_anchors(model, cfg)
    del model; gc.collect()
    torch.cuda.empty_cache()

    if data_mean is not None:
        anchors_raw = anchors_raw - data_mean
    anchors = l2_norm(anchors_raw)
    print(f"[anchors]   {anchors.shape}")

    # 5. 코사인 유사도
    sim = anchors @ centroids.T  # (5, 5)
    print_sim_matrix(sim, args.k)

    # 6. Hungarian
    a_idx, c_idx = hungarian_assign(sim)

    # 7. 매핑 + low-confidence
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

    # 8. 저장
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
