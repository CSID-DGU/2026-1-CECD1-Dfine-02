"""
archetype_cluster.py — Tier 2 archetype 클러스터링  [파이프라인 Step 3]
작성: 2026-05-22
입력: resource/embeddings_percol5/embeddings_percol5.parquet  (percol5 임베딩, Step 1 산출물)
출력: resource/outputs/archetype_n{N}.csv  — uuid / archetype(0-4)
연산: percol5 → L2(in-place) → PCA(100) → L2 → UMAP(2D) → KMeans k=5

Usage:
    uv run src/archetype_cluster.py --sample 1000000
    uv run src/archetype_cluster.py --sample 50000 --k 5
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
    ).fit_transform(x)
    print(f"  done {time.perf_counter()-t0:.1f}s")
    return out.astype(np.float32)


def load_percol5(sample: int) -> tuple[list[str], np.ndarray]:
    table     = pa_ds.dataset(EMBED_DIR, format="parquet").to_table(
                    columns=["uuid", "embedding"])
    n_total   = len(table)
    uuids_all = table.column("uuid").to_pylist()
    flat      = (table.column("embedding").combine_chunks()
                      .flatten().to_numpy(zero_copy_only=False))
    del table; gc.collect()

    dim = flat.size // n_total
    arr = flat.reshape(n_total, dim).astype(np.float32)
    del flat; gc.collect()

    if sample < n_total:
        idx   = np.sort(np.random.default_rng(SEED).choice(n_total, sample, replace=False))
        arr   = arr[idx].copy()
        uuids = [uuids_all[i] for i in idx]
    else:
        uuids = uuids_all

    print(f"[percol5] {len(uuids):,} × {dim}")
    return uuids, arr


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
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    uuids, emb = load_percol5(args.sample)
    archetype  = run_tier2(emb, k=args.k, cfg=cfg)
    del emb; gc.collect()

    sizes = {c: int((archetype == c).sum()) for c in range(args.k)}
    print(f"\n[sizes] {sizes}")

    df  = pd.DataFrame({"uuid": uuids, "archetype": archetype})
    out = OUT_DIR / f"archetype_n{args.sample}.csv"
    df.to_csv(out, index=False)
    print(f"[save] {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
