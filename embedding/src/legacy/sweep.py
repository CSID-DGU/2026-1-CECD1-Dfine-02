"""
sweep.py — KMeans k sweep (MiniBatch / Full / GPU 통합).

내부 지표: silhouette, davies_bouldin, calinski_harabasz (KMeans 가정에 편향)
보조 지표: inertia (elbow), bootstrap_ari (안정성 — KMeans 편향 없음)

Usage:
    uv run src/sweep.py --algo mini
    uv run src/sweep.py --algo full --embed-dir resource/embeddings_percol
    uv run src/sweep.py --algo gpu  --device cuda:0
    uv run src/sweep.py --algo mini --sample 200000
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "resource" / "outputs"

SEED        = 42
BATCH_SIZE  = 10_000
BOOT_FRAC   = 0.5   # bootstrap 서브샘플 비율


def load_embeddings(
    embed_dir: Path, sample: int | None = None
) -> tuple[list[str], np.ndarray]:
    """parquet → numpy float32. sample 지정 시 float16 단계에서 슬라이스 (메모리 절약)."""
    if not any(embed_dir.glob("*.parquet")):
        sys.exit(f"[load] 임베딩 없음: {embed_dir}")

    table = pa_ds.dataset(embed_dir, format="parquet").to_table(columns=["uuid", "embedding"])
    n_total = len(table)
    uuids   = table.column("uuid").to_pylist()
    # to_pandas() 금지: FixedSizeListArray → 1M Python numpy objects → 무한 메모리 성장
    flat = table.column("embedding").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    del table
    gc.collect()

    dim     = flat.size // n_total
    arr_f16 = flat.reshape(n_total, dim)

    if sample and sample < n_total:
        rng     = np.random.default_rng(SEED)
        idx     = np.sort(rng.choice(n_total, size=sample, replace=False))
        arr_f16 = arr_f16[idx].copy()          # flat 뷰 분리 → del flat 가능
        uuids   = [uuids[i] for i in idx]
        del flat
        gc.collect()
        n = sample
        print(f"[load] sample {sample:,} / {n_total:,}")
    else:
        n = n_total

    embeddings = arr_f16.astype(np.float32)
    del arr_f16
    gc.collect()
    print(f"[load] {n:,} rows  dim={dim}")
    return uuids, embeddings


def normalize(embeddings: np.ndarray) -> None:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.clip(norms, 1e-10, None)


def _cpu_labels_iter(
    x: np.ndarray, k_range: range, algo: str
) -> Iterator[tuple[int, np.ndarray, float, float]]:
    for k in k_range:
        if algo == "mini":
            km = MiniBatchKMeans(n_clusters=k, batch_size=BATCH_SIZE,
                                 n_init=10, random_state=SEED)
        else:  # "full"
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        t0      = time.perf_counter()
        labels  = km.fit_predict(x)
        dt      = time.perf_counter() - t0
        inertia = float(km.inertia_)
        del km
        gc.collect()
        yield k, labels, dt, inertia


def _predict_once(x: np.ndarray, k: int, algo: str, device: str) -> np.ndarray:
    if algo == "gpu":
        from gpu_kmeans import predict_once
        return predict_once(x, k, device)
    if algo == "mini":
        km = MiniBatchKMeans(n_clusters=k, batch_size=BATCH_SIZE,
                             n_init=10, random_state=SEED)
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    return km.fit_predict(x)


def _bootstrap_ari(x: np.ndarray, k: int, algo: str, device: str) -> float:
    """50% 서브샘플 2회 클러스터링 → 공통 점들의 ARI."""
    rng    = np.random.default_rng(SEED + 1)   # 데이터 로드 SEED와 분리
    n      = len(x)
    m      = int(n * BOOT_FRAC)
    idx_a  = np.sort(rng.choice(n, size=m, replace=False))
    idx_b  = np.sort(rng.choice(n, size=m, replace=False))

    labels_a = _predict_once(x[idx_a], k, algo, device)
    labels_b = _predict_once(x[idx_b], k, algo, device)

    _, ia, ib = np.intersect1d(idx_a, idx_b, return_indices=True)
    ari = adjusted_rand_score(labels_a[ia], labels_b[ib])

    del labels_a, labels_b
    gc.collect()
    return float(ari)


def sweep(
    norm_emb: np.ndarray, k_range: range, algo: str, device: str
) -> tuple[dict[int, dict], dict[int, np.ndarray]]:
    if algo == "gpu":
        from gpu_kmeans import labels_iter
        it    = labels_iter(norm_emb, k_range, device)
        label = f"GPU ({device})"
    else:
        it    = _cpu_labels_iter(norm_emb, k_range, algo)
        label = {"mini": "MiniBatchKMeans", "full": "KMeans (full)"}[algo]

    print(f"\n[sweep] {label}  k={k_range.start}..{k_range.stop - 1}"
          f"  n={len(norm_emb):,}")
    print(f"{'k':>3}  {'sil':>8}  {'db':>8}  {'ch':>11}"
          f"  {'inertia':>11}  {'ari':>6}  {'time':>8}")
    print("-" * 66)

    results:    dict[int, dict]       = {}
    all_labels: dict[int, np.ndarray] = {}
    for k, labels, dt, inertia in it:
        sil = silhouette_score(norm_emb, labels,
                               sample_size=min(5_000, len(labels)),
                               random_state=SEED)
        db  = davies_bouldin_score(norm_emb, labels)
        ch  = calinski_harabasz_score(norm_emb, labels)
        all_labels[k] = labels.astype(np.int32, copy=False)

        ari = _bootstrap_ari(norm_emb, k, algo, device)

        results[k] = {
            "silhouette":        sil,
            "davies_bouldin":    db,
            "calinski_harabasz": ch,
            "inertia":           inertia,
            "bootstrap_ari":     ari,
        }
        print(f"{k:>3}  {sil:>8.4f}  {db:>8.4f}  {ch:>11.1f}"
              f"  {inertia:>11.1f}  {ari:>6.3f}  {dt:>7.1f}s")

    best_sil = max(results, key=lambda k: results[k]["silhouette"])
    best_ari = max(results, key=lambda k: results[k]["bootstrap_ari"])
    print(f"\n[sweep] best by silhouette : k={best_sil}  "
          f"(sil={results[best_sil]['silhouette']:.4f})")
    print(f"[sweep] best by stability   : k={best_ari}  "
          f"(ari={results[best_ari]['bootstrap_ari']:.3f})")
    return results, all_labels


def save(
    results: dict[int, dict],
    all_labels: dict[int, np.ndarray],
    uuids: list[str],
    meta: dict,
    out_base: str,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / f"sweep_{out_base}.csv"
    pd.DataFrame([{"k": k, **v} for k, v in results.items()]).to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    labels_path = OUT_DIR / f"labels_{out_base}.parquet"
    label_df    = pd.DataFrame({"uuid": uuids})
    for k, labels in all_labels.items():
        label_df[f"k{k}"] = labels
    label_df.to_parquet(labels_path, index=False)
    print(f"[save] {labels_path}")

    meta_path = OUT_DIR / f"sweep_{out_base}.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[save] {meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["mini", "full", "gpu"], default="mini")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=7)
    ap.add_argument("--embed-dir", type=Path,
                    default=ROOT / "resource" / "embeddings")
    ap.add_argument("--sample", type=int, default=None,
                    help="스위핑용 샘플 수 (RAM 절약, 예: 200000)")
    ap.add_argument("--device", type=str, default="cuda:0",
                    help="--algo gpu 일 때 사용")
    args = ap.parse_args()

    uuids, embeddings = load_embeddings(args.embed_dir, sample=args.sample)
    n, dim = embeddings.shape
    normalize(embeddings)

    k_range = range(args.k_min, args.k_max + 1)
    results, all_labels = sweep(embeddings, k_range, args.algo, args.device)

    best_sil = max(results, key=lambda k: results[k]["silhouette"])
    best_ari = max(results, key=lambda k: results[k]["bootstrap_ari"])
    meta = {
        "algo":            args.algo,
        "embed_dir":       str(args.embed_dir),
        "sample":          args.sample,
        "n":               int(n),
        "dim":             int(dim),
        "k_range":         list(k_range),
        "seed":            SEED,
        "best_silhouette": {"k": int(best_sil), "value": results[best_sil]["silhouette"]},
        "best_stability":  {"k": int(best_ari), "value": results[best_ari]["bootstrap_ari"]},
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
    }
    out_base = f"{args.algo}_{args.embed_dir.name}"
    save(results, all_labels, uuids, meta, out_base)


if __name__ == "__main__":
    main()
