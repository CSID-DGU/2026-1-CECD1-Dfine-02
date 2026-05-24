"""
consumption_cluster.py — Tier 3 소비태그 클러스터링  [파이프라인 Step 4]
작성: 2026-05-22
입력: resource/outputs/consumption_emb_n{N}.parquet  (소비 임베딩, Step 2 산출물)
출력: resource/outputs/consumption_cluster_n{N}.csv  — uuid / consumption_tag / noise_dist / entropy
연산: consumption_emb → L2 → PCA(50) → L2 → UMAP(2D) → KMeans k=5
      noise_dist: Shannon 엔트로피 상위 noise_pct% → 1

Usage:
    uv run src/consumption_cluster.py --sample 1000000
    uv run src/consumption_cluster.py --sample 50000 --noise-pct 10
"""

import argparse
import gc
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import umap as umap_lib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "resource" / "outputs"
CFG_PATH = Path(__file__).with_suffix(".toml")
SEED     = 42


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


def load_consumption_emb(sample: int) -> tuple[list[str], np.ndarray]:
    path = OUT_DIR / f"consumption_emb_n{sample}.parquet"
    if not path.exists():
        sys.exit(f"[error] 소비 임베딩 캐시 없음: {path}\n"
                 f"        먼저 Step 2 실행: uv run main.py --step 2 --sample {sample}")
    table = pa_ds.dataset(str(path), format="parquet").to_table(columns=["uuid", "embedding"])
    uuids = table.column("uuid").to_pylist()
    flat  = (table.column("embedding").combine_chunks()
                  .flatten().to_numpy(zero_copy_only=False))
    del table; gc.collect()
    dim = flat.size // len(uuids)
    arr = flat.reshape(len(uuids), dim).astype(np.float32)
    del flat; gc.collect()
    print(f"[cons_emb] {len(uuids):,} × {dim}")
    return uuids, arr


def run_tier3(emb: np.ndarray, pca_n: int, k: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """consumption_emb → L2 → PCA(pca_n) → L2 → UMAP → KMeans k"""
    n_init = cfg["kmeans"]["n_init"]
    x = l2_norm(emb)
    print(f"[tier3] PCA {x.shape[1]} → {pca_n} ...")
    x = l2_norm(PCA(n_components=pca_n, random_state=SEED).fit_transform(x).astype(np.float32))
    coords = umap2d(x, cfg["umap"])
    print(f"[tier3] KMeans k={k} ...")
    km = KMeans(n_clusters=k, n_init=n_init, random_state=SEED).fit(coords)
    return coords, km.labels_.astype(np.int32), km.cluster_centers_.astype(np.float32)


def compute_entropy(coords: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
    inv   = 1.0 / np.clip(dists, 1e-8, None)
    prob  = inv / inv.sum(axis=1, keepdims=True)
    H     = -(prob * np.log(np.clip(prob, 1e-12, None))).sum(axis=1)
    return H.astype(np.float32)


def main() -> None:
    cfg = load_script_cfg()

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",    type=int,   default=50_000)
    ap.add_argument("--k",         type=int,   default=cfg["kmeans"]["k"])
    ap.add_argument("--pca2",      type=int,   default=cfg["pca"]["n_components"],
                    help="소비 임베딩 PCA 차원 (config.pca.n_components)")
    ap.add_argument("--noise-pct", type=float, default=cfg["noise"]["pct"],
                    help="엔트로피 상위 N%% → noise_dist=1 (config.noise.pct)")
    ap.add_argument("--out", type=Path, default=None,
                    help="출력 CSV 경로 (기본: consumption_cluster_n{sample}.csv)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    uuids, emb = load_consumption_emb(args.sample)
    coords, cons_tag, centroids = run_tier3(emb, pca_n=args.pca2, k=args.k, cfg=cfg)
    del emb; gc.collect()

    sizes = {c: int((cons_tag == c).sum()) for c in range(args.k)}
    print(f"\n[sizes] {sizes}")

    H          = compute_entropy(coords, centroids)
    threshold  = np.percentile(H, 100 - args.noise_pct)
    noise_dist = (H >= threshold).astype(np.int8)
    print(f"[entropy]  mean={H.mean():.4f}  threshold(p{100-args.noise_pct:.0f})={threshold:.4f}"
          f"  noise_dist n={noise_dist.sum():,} ({noise_dist.mean()*100:.1f}%)")

    df  = pd.DataFrame({
        "uuid":            uuids,
        "consumption_tag": cons_tag,
        "noise_dist":      noise_dist,
        "entropy":         H,
    })
    out = args.out or OUT_DIR / f"consumption_cluster_n{args.sample}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[save] {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
