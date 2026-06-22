"""
archetype_cluster.py — Tier 2 archetype 클러스터링  [파이프라인 Step 3]
작성: 2026-05-22
입력: resource/embeddings_percol5/embeddings_percol5.parquet  (percol5 임베딩, Step 1 산출물)
출력: resource/outputs/archetype_n{N}.csv  — uuid / archetype(0-4)
연산: percol5 → L2(in-place) → PCA(100) → L2 → UMAP(2D) → KMeans k=5
     --regress-out 시: percol5 → age·sex 회귀잔차 → (이하 동일)

Usage:
    uv run src/archetype_cluster.py --sample 1000000
    uv run src/archetype_cluster.py --sample 50000 --k 5
    uv run src/archetype_cluster.py --sample 1000000 --regress-out age,sex
"""

import argparse
import gc
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import umap as umap_lib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT      = Path(__file__).parent.parent
OUT_DIR   = ROOT / "resource" / "outputs"
EMBED_DIR = ROOT / "resource" / "embeddings_percol5"
RAW_PATH  = ROOT / "resource" / "nemotron_raw.parquet"
CFG_PATH  = Path(__file__).with_suffix(".toml")
SEED      = 42


def load_script_cfg() -> dict:
    with CFG_PATH.open("rb") as f:
        return tomllib.load(f)


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-10, None)


def umap2d(x: np.ndarray, umap_cfg: dict) -> np.ndarray:
    print(f"[umap] {x.shape[1]}-dim → {umap_cfg['n_components']}D ...")
    t0 = time.perf_counter()
    out = umap_lib.UMAP(
        n_components = umap_cfg["n_components"],
        n_neighbors  = umap_cfg["n_neighbors"],
        min_dist     = umap_cfg["min_dist"],
        metric       = umap_cfg["metric"],
        random_state = SEED,
        low_memory   = True,
    ).fit_transform(x)
    print(f"  done {time.perf_counter()-t0:.1f}s")
    return out.astype(np.float32)


def load_percol5(sample: int) -> tuple[list[str], np.ndarray]:
    """percol5 parquet 청크 read → 샘플 행만 (sample, dim) fp32 추출.

    원본은 풀 (n_total, dim) fp16 → fp32 더블링(~20GB+) 후 sample 슬라이스.
    여기서는 sample 인덱스 사전 결정 후 청크 순회하며 슬롯만 채워 메모리 절약.
    동일 SEED 사용 — 인덱스·순서·내용 모두 원본과 동치.
    """
    ds      = pa_ds.dataset(EMBED_DIR, format="parquet")
    n_total = ds.count_rows()

    if sample < n_total:
        sample_pos = np.sort(
            np.random.default_rng(SEED).choice(n_total, sample, replace=False)
        )
    else:
        sample_pos = np.arange(n_total)
    n_sample = len(sample_pos)

    out: np.ndarray | None = None
    uuids                    = [""] * n_sample
    dim                      = -1
    row_off                  = 0
    BATCH                    = 20_000

    for batch in ds.to_batches(columns=["uuid", "embedding"], batch_size=BATCH):
        n_b = batch.num_rows
        lo  = int(np.searchsorted(sample_pos, row_off,       side="left"))
        hi  = int(np.searchsorted(sample_pos, row_off + n_b, side="left"))
        if lo == hi:
            row_off += n_b
            continue

        local_idx = sample_pos[lo:hi] - row_off  # batch 내 row 위치
        if out is None:
            dim = batch.column("embedding").type.list_size
            out = np.zeros((n_sample, dim), dtype=np.float32)

        flat       = (batch.column("embedding").flatten()
                           .to_numpy(zero_copy_only=False))
        out[lo:hi] = flat.reshape(n_b, dim)[local_idx].astype(np.float32)

        batch_uids = batch.column("uuid").to_pylist()
        for k, j in enumerate(local_idx):
            uuids[lo + k] = batch_uids[int(j)]

        row_off += n_b

    if out is None:
        raise RuntimeError("percol5 dataset empty")

    print(f"[percol5] {n_sample:,} × {dim}")
    return uuids, out


def regress_out(emb: np.ndarray, uuids: list[str], factors: list[str]) -> np.ndarray:
    """emb에서 인구통계 factors(age·sex 등)의 선형성분을 최소제곱 회귀로 제거.

    수치형(age)은 표준화 후 z, z² 두 항(비선형 허용), 범주형(sex)은 원핫(drop_first).
    설계행렬 D (N×k, k 작음) → B=lstsq(D,emb) → emb -= D@B (in-place, peak +1배).
    표본 자체의 연령 분포는 불변 — 임베딩에서 분리축으로서의 연령만 제거한다.
    """
    raw = pd.read_parquet(RAW_PATH, columns=["uuid"] + factors)
    raw = raw.set_index("uuid").loc[uuids]

    parts = [np.ones((len(uuids), 1), dtype=np.float64)]
    for f in factors:
        col = raw[f].to_numpy()
        if col.dtype.kind in "if":
            z = (col - col.mean()) / col.std()
            parts.append(z.reshape(-1, 1))
            parts.append((z ** 2).reshape(-1, 1))
        else:
            d = pd.get_dummies(raw[f], drop_first=True).to_numpy(dtype=np.float64)
            parts.append(d)
    D = np.hstack(parts).astype(np.float32)
    print(f"[regress] factors={factors} → design {D.shape}")

    B, *_ = np.linalg.lstsq(D, emb, rcond=None)
    emb -= (D @ B).astype(np.float32)
    return emb


def run_tier2(emb: np.ndarray, k: int, cfg: dict) -> np.ndarray:
    """percol5 → L2(in-place) → PCA → L2 → UMAP → KMeans k"""
    pca_n  = cfg["pca"]["n_components"]
    n_init = cfg["kmeans"]["n_init"]

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb /= np.clip(norms, 1e-10, None)
    del norms
    print(f"[tier2] PCA {emb.shape[1]} → {pca_n} ...")
    x = l2_norm(PCA(n_components=pca_n, random_state=SEED).fit_transform(emb).astype(np.float32))
    coords = umap2d(x, cfg["umap"])
    print(f"[tier2] KMeans k={k} ...")
    return KMeans(n_clusters=k, n_init=n_init, random_state=SEED).fit_predict(coords).astype(np.int32)


def main() -> None:
    cfg = load_script_cfg()

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50_000)
    ap.add_argument("--k",      type=int, default=cfg["kmeans"]["k"])
    ap.add_argument("--regress-out", type=str, default="",
                    help="쉼표구분 인구통계 factor 회귀제거 (예: age,sex). 빈값이면 미적용")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    uuids, emb = load_percol5(args.sample)
    suffix = ""
    if args.regress_out:
        factors = [f.strip() for f in args.regress_out.split(",") if f.strip()]
        emb     = regress_out(emb, uuids, factors)
        suffix  = "_resid_" + "_".join(factors)
    archetype  = run_tier2(emb, k=args.k, cfg=cfg)
    del emb; gc.collect()

    sizes = {c: int((archetype == c).sum()) for c in range(args.k)}
    print(f"\n[sizes] {sizes}")

    df  = pd.DataFrame({"uuid": uuids, "archetype": archetype})
    out = OUT_DIR / f"archetype_n{args.sample}{suffix}.csv"
    df.to_csv(out, index=False)
    print(f"[save] {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
