"""
umap_sweep.py — UMAP n_components 스윕 후 KMeans k=5 비교

파이프라인:
    percol5 (5120-dim)
    → L2 정규화
    → PCA(100)          ← 이전 실험에서 최적 확인
    → UMAP(n_components) ← 스윕 대상
    → L2 정규화
    → KMeans k=5

산출물:
    resource/outputs/umap_sweep_metrics.csv   — n_components별 지표
    resource/outputs/umap_sweep_curve.png     — 지표 변화 곡선
    resource/outputs/umap_sweep_scatter.png   — 각 n별 2D 투영 + 라벨

Usage:
    uv run src/umap_sweep.py
    uv run src/umap_sweep.py --sample 50000 --pca 100
    uv run src/umap_sweep.py --components 2 5 10 20 50
"""

import argparse
import gc
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"]      = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# cuML 가용 여부 감지 (--gpu 플래그와 함께 사용)
try:
    import cupy as cp
    import cuml.manifold
    import cuml.cluster
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "resource" / "outputs"
SEED    = 42
BOOT_FRAC = 0.5
DEFAULT_COMPONENTS = [2, 5, 10, 20, 30, 50]


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_embeddings(embed_dir: Path, sample: int) -> np.ndarray:
    """스트리밍: 사전 결정 sample 인덱스 → 청크 순회하며 fp32 슬롯 채움.

    풀 fp16(~10GB)+fp32(~20GB) 더블링 회피.
    """
    ds      = pa_ds.dataset(embed_dir, format="parquet")
    n_total = ds.count_rows()

    if sample < n_total:
        sample_pos = np.sort(
            np.random.default_rng(SEED).choice(n_total, sample, replace=False))
    else:
        sample_pos = np.arange(n_total)
    n_sample = len(sample_pos)

    out: np.ndarray | None = None
    dim     = -1
    row_off = 0
    BATCH   = 20_000

    for batch in ds.to_batches(columns=["embedding"], batch_size=BATCH):
        n_b = batch.num_rows
        lo  = int(np.searchsorted(sample_pos, row_off,       side="left"))
        hi  = int(np.searchsorted(sample_pos, row_off + n_b, side="left"))
        if lo == hi:
            row_off += n_b
            continue
        local_idx = sample_pos[lo:hi] - row_off
        if out is None:
            dim = batch.column("embedding").type.list_size
            out = np.zeros((n_sample, dim), dtype=np.float32)
        flat = (batch.column("embedding").flatten()
                     .to_numpy(zero_copy_only=False))
        out[lo:hi] = flat.reshape(n_b, dim)[local_idx].astype(np.float32)
        row_off += n_b
        del flat

    if out is None:
        raise RuntimeError(f"[load] 빈 dataset: {embed_dir}")
    gc.collect()
    print(f"[load] {n_sample:,} × {dim}")
    return out


# ── 전처리 ─────────────────────────────────────────────────────────────────────

def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-10, None)


def apply_pca(x: np.ndarray, n: int) -> np.ndarray:
    print(f"[pca] {x.shape[1]} → {n} ...")
    return PCA(n_components=n, random_state=SEED).fit_transform(x).astype(np.float32)


def apply_umap(x: np.ndarray, n: int, use_gpu: bool = False) -> np.ndarray:
    t0 = time.perf_counter()
    min_dist = 0.0 if n == 2 else 0.1   # 2D 시각화는 0.0, 클러스터링용은 0.1
    if use_gpu and CUML_AVAILABLE:
        print(f"[umap-gpu] {x.shape[1]} → {n}  (n_neighbors=30, min_dist={min_dist}) ...")
        out = cuml.manifold.UMAP(
            n_components=n, n_neighbors=30, min_dist=min_dist,
            metric="cosine", random_state=SEED,
        ).fit_transform(cp.array(x))
        out = cp.asnumpy(out).astype(np.float32)
    else:
        import umap as umap_lib
        print(f"[umap-cpu] {x.shape[1]} → {n}  (n_neighbors=30, min_dist={min_dist}) ...")
        out = umap_lib.UMAP(
            n_components=n, n_neighbors=30, min_dist=min_dist,
            metric="cosine", random_state=SEED,
        ).fit_transform(x).astype(np.float32)
    print(f"  done {time.perf_counter()-t0:.1f}s")
    return out


# ── 클러스터링 + 지표 ──────────────────────────────────────────────────────────

def run_kmeans(x_norm: np.ndarray, k: int, use_gpu: bool = False) -> tuple[np.ndarray, dict]:
    if use_gpu and CUML_AVAILABLE:
        x_gpu  = cp.array(x_norm)
        labels = cuml.cluster.KMeans(n_clusters=k, n_init=10,
                                     random_state=SEED).fit_predict(x_gpu)
        labels = cp.asnumpy(labels).astype(np.int32)
    else:
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(x_norm)

    sil = silhouette_score(x_norm, labels,
                           sample_size=min(5_000, len(labels)), random_state=SEED)
    db  = davies_bouldin_score(x_norm, labels)
    ch  = calinski_harabasz_score(x_norm, labels)
    ari = _bootstrap_ari(x_norm, k, use_gpu)

    return labels, {"silhouette": sil, "davies_bouldin": db,
                    "calinski_harabasz": ch, "bootstrap_ari": ari}


def _bootstrap_ari(x: np.ndarray, k: int, use_gpu: bool = False) -> float:
    rng = np.random.default_rng(SEED + 1)
    n, m = len(x), int(len(x) * BOOT_FRAC)
    ia = np.sort(rng.choice(n, m, replace=False))
    ib = np.sort(rng.choice(n, m, replace=False))
    if use_gpu and CUML_AVAILABLE:
        km  = cuml.cluster.KMeans(n_clusters=k, n_init=5, random_state=SEED)
        la  = cp.asnumpy(km.fit_predict(cp.array(x[ia]))).astype(np.int32)
        lb  = cp.asnumpy(km.fit_predict(cp.array(x[ib]))).astype(np.int32)
    else:
        la  = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(x[ia])
        lb  = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(x[ib])
    _, ia2, ib2 = np.intersect1d(ia, ib, return_indices=True)
    return float(adjusted_rand_score(la[ia2], lb[ib2]))


# ── 시각화 ─────────────────────────────────────────────────────────────────────

def plot_curves(df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "bootstrap_ari"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    for ax, m in zip(axes.flat, metrics):
        ax.plot(df["n_components"], df[m], marker="o", linewidth=1.5)
        best_idx = df[m].idxmin() if m == "davies_bouldin" else df[m].idxmax()
        best_n   = df.loc[best_idx, "n_components"]
        best_v   = df.loc[best_idx, m]
        ax.axvline(best_n, linestyle="--", linewidth=1, alpha=0.6, color="crimson")
        ax.set_title(f"{m}  (best n={int(best_n)})", fontsize=10)
        ax.set_xlabel("UMAP n_components")
        ax.set_ylabel(m)
        ax.grid(alpha=0.3)
        ax.annotate(f"{best_v:.4f}", (best_n, best_v),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="crimson")

    fig.suptitle(f"UMAP sweep  percol5→PCA{df['pca_n'].iloc[0]}→UMAP(n)→KMeans k=5",
                 fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_scatter(all_coords2d: dict[int, np.ndarray],
                 all_labels:   dict[int, np.ndarray],
                 all_metrics:  dict[int, dict],
                 out_path: Path) -> None:
    ns   = sorted(all_coords2d)
    cols = min(len(ns), 3)
    rows = (len(ns) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows),
                             squeeze=False, constrained_layout=True)
    for ax, n in zip(axes.flat, ns):
        m = all_metrics[n]
        ax.scatter(all_coords2d[n][:, 0], all_coords2d[n][:, 1],
                   c=all_labels[n], cmap="tab10", s=1, alpha=0.4, linewidths=0)
        ax.set_title(
            f"UMAP n={n}\nsil={m['silhouette']:.4f}  ari={m['bootstrap_ari']:.3f}",
            fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes.flat[len(ns):]:
        ax.axis("off")

    n_pts = len(next(iter(all_labels.values())))
    fig.suptitle(f"KMeans k=5  (n={n_pts:,}, 2D via UMAP-2 재투영)", fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir",   type=Path,
                    default=ROOT / "resource" / "embeddings_percol5")
    ap.add_argument("--sample",      type=int,   default=50_000)
    ap.add_argument("--k",           type=int,   default=5)
    ap.add_argument("--pca",         type=int,   default=100,
                    help="UMAP 전 PCA 차원 (0=생략)")
    ap.add_argument("--components",  type=int,   nargs="+",
                    default=DEFAULT_COMPONENTS,
                    help="스윕할 UMAP n_components 목록")
    ap.add_argument("--gpu",         action="store_true",
                    help="cuML GPU 가속 사용 (cuml 설치 필요)")
    args = ap.parse_args()

    if args.gpu and not CUML_AVAILABLE:
        print("[warn] --gpu 지정했으나 cuml 미설치 → CPU fallback")
        args.gpu = False
    if args.gpu:
        print(f"[mode] GPU (cuML {cuml.__version__})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 로드 + 정규화
    raw    = load_embeddings(args.embed_dir, args.sample)
    x      = l2_norm(raw);  del raw; gc.collect()

    # 2. PCA 선처리
    if args.pca > 0:
        x = l2_norm(apply_pca(x, args.pca))

    # 3. UMAP 스윕
    rows_list:    list[dict]              = []
    all_labels:   dict[int, np.ndarray]  = {}
    all_coords2d: dict[int, np.ndarray]  = {}
    all_metrics:  dict[int, dict]        = {}

    header = f"{'n':>4}  {'sil':>8}  {'db':>8}  {'ch':>11}  {'ari':>6}"
    print(f"\n{header}")
    print("─" * len(header))

    for n in sorted(args.components):
        x_umap = apply_umap(x, n, use_gpu=args.gpu)
        x_n    = l2_norm(x_umap)
        labels, m = run_kmeans(x_n, args.k, use_gpu=args.gpu)

        all_labels[n]  = labels
        all_metrics[n] = m

        # 2D 투영: n=2면 그대로, n>2면 UMAP-2 재투영
        if n == 2:
            all_coords2d[n] = x_umap
        else:
            all_coords2d[n] = apply_umap(x_n, 2, use_gpu=args.gpu)

        rows_list.append({"n_components": n, "pca_n": args.pca, **m})
        print(f"{n:>4}  {m['silhouette']:>8.4f}  {m['davies_bouldin']:>8.4f}"
              f"  {m['calinski_harabasz']:>11.1f}  {m['bootstrap_ari']:>6.3f}")

        del x_umap, x_n; gc.collect()

    # 4. 저장 + 시각화
    df = pd.DataFrame(rows_list)
    csv_path = OUT_DIR / "umap_sweep_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    plot_curves(df, OUT_DIR / "umap_sweep_curve.png")
    plot_scatter(all_coords2d, all_labels, all_metrics,
                 OUT_DIR / "umap_sweep_scatter.png")

    # 5. 최적 n 요약
    best_sil = df.loc[df["silhouette"].idxmax(), "n_components"]
    best_ari = df.loc[df["bootstrap_ari"].idxmax(), "n_components"]
    best_db  = df.loc[df["davies_bouldin"].idxmin(), "n_components"]
    print(f"\n[best]  silhouette → n={int(best_sil)}"
          f"  |  ari → n={int(best_ari)}"
          f"  |  davies_bouldin → n={int(best_db)}")


if __name__ == "__main__":
    main()
