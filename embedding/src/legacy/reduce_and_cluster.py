"""
reduce_and_cluster.py — 차원 축소 후 KMeans k=5 비교

방법 비교:
  baseline       : L2 정규화 5120-dim KMeans (현재 파이프라인)
  pca_50/100/200 : PCA n-dim → L2 정규화 → KMeans
  umap_20/50     : PCA(200) → UMAP n-dim → L2 정규화 → KMeans

산출물:
  resource/outputs/pca_scree.png
  resource/outputs/dim_reduction_comparison.csv
  resource/outputs/viz_dimred_k5.png

Usage:
    uv run src/reduce_and_cluster.py
    uv run src/reduce_and_cluster.py --sample 30000 --skip-umap
    uv run src/reduce_and_cluster.py --k 5 --scree-max 300
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

ROOT      = Path(__file__).parent.parent
OUT_DIR   = ROOT / "resource" / "outputs"
SEED      = 42
BOOT_FRAC = 0.5
BOOT_MAX  = 200_000  # bootstrap 서브샘플 상한 (1M 입력 시 메모리 캡)


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_embeddings(embed_dir: Path, sample: int) -> tuple[list[str], np.ndarray]:
    """스트리밍: 사전 결정 sample 인덱스 → 청크 순회하며 fp32 슬롯 채움."""
    ds      = pa_ds.dataset(embed_dir, format="parquet")
    n_total = ds.count_rows()

    if sample < n_total:
        sample_pos = np.sort(
            np.random.default_rng(SEED).choice(n_total, sample, replace=False))
    else:
        sample_pos = np.arange(n_total)
    n_sample = len(sample_pos)

    out: np.ndarray | None = None
    uuids                   = [""] * n_sample
    dim                     = -1
    row_off                 = 0
    BATCH                   = 20_000

    for batch in ds.to_batches(columns=["uuid", "embedding"], batch_size=BATCH):
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
        batch_uids = batch.column("uuid").to_pylist()
        for k, j in enumerate(local_idx):
            uuids[lo + k] = batch_uids[int(j)]
        row_off += n_b
        del flat, batch_uids

    if out is None:
        raise RuntimeError(f"[load] 빈 dataset: {embed_dir}")
    gc.collect()
    print(f"[load] {n_sample:,} × {dim}")
    return uuids, out


# ── 전처리 ─────────────────────────────────────────────────────────────────────

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-10, None)


# ── 차원 축소 ──────────────────────────────────────────────────────────────────

def reduce_pca(x: np.ndarray, n: int) -> np.ndarray:
    print(f"  [pca] {x.shape[1]} → {n} ...")
    return PCA(n_components=n, random_state=SEED).fit_transform(x).astype(np.float32)


def reduce_umap(x: np.ndarray, n: int) -> np.ndarray:
    import umap as umap_lib
    print(f"  [umap] {x.shape[1]} → {n}  (n_neighbors=30, min_dist=0.0, cosine) ...")
    reducer = umap_lib.UMAP(
        n_components=n, n_neighbors=30, min_dist=0.0,
        metric="cosine", random_state=SEED,
    )
    return reducer.fit_transform(x).astype(np.float32)


# ── 클러스터링 + 지표 ──────────────────────────────────────────────────────────

def run_kmeans(x_norm: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """L2-정규화 입력에 KMeans → (labels, metrics)"""
    t0     = time.perf_counter()
    labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(x_norm)
    dt     = time.perf_counter() - t0

    sil = silhouette_score(x_norm, labels,
                           sample_size=min(5_000, len(labels)), random_state=SEED)
    db  = davies_bouldin_score(x_norm, labels)
    ch  = calinski_harabasz_score(x_norm, labels)
    ari = _bootstrap_ari(x_norm, k)

    return labels, {
        "dim":              x_norm.shape[1],
        "silhouette":       round(sil, 4),
        "davies_bouldin":   round(db,  4),
        "calinski_harabasz": round(ch, 1),
        "bootstrap_ari":    round(ari, 4),
        "km_time_s":        round(dt, 1),
    }


def _bootstrap_ari(x: np.ndarray, k: int) -> float:
    rng = np.random.default_rng(SEED + 1)
    n   = len(x)
    m   = int(n * BOOT_FRAC)
    ia  = np.sort(rng.choice(n, size=m, replace=False))
    ib  = np.sort(rng.choice(n, size=m, replace=False))
    la  = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(x[ia])
    lb  = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(x[ib])
    _, ia2, ib2 = np.intersect1d(ia, ib, return_indices=True)
    return float(adjusted_rand_score(la[ia2], lb[ib2]))


# ── 시각화 ─────────────────────────────────────────────────────────────────────

def plot_scree(x_norm: np.ndarray, max_n: int, out_path: Path) -> None:
    n = min(max_n, x_norm.shape[1], x_norm.shape[0] - 1)
    print(f"[scree] PCA fit n_components={n} ...")
    pca    = PCA(n_components=n, random_state=SEED).fit(x_norm)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    comp   = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(comp, pca.explained_variance_ratio_, linewidth=0.8, color="steelblue")
    ax1.set_xlabel("Component"); ax1.set_ylabel("Explained variance ratio")
    ax1.set_title("Scree plot (per-component)")

    ax2.plot(comp, cumvar, linewidth=1.2, color="steelblue")
    for thr, color in [(0.5, "gray"), (0.7, "orange"), (0.8, "tomato"), (0.9, "crimson")]:
        n_needed = int(np.searchsorted(cumvar, thr)) + 1
        if n_needed <= n:
            ax2.axhline(thr, linestyle="--", linewidth=0.7, color=color, alpha=0.7)
            ax2.axvline(n_needed, linestyle=":", linewidth=0.7, color=color, alpha=0.7)
            ax2.text(n_needed + n * 0.01, thr + 0.01,
                     f"{int(thr*100)}%  n={n_needed}", fontsize=8, color=color)
    ax2.set_xlabel("n components"); ax2.set_ylabel("Cumulative explained variance")
    ax2.set_title("Cumulative variance")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] {out_path}")

    print("  누적 분산 요약:")
    for thr in [0.5, 0.7, 0.8, 0.9]:
        n_needed = int(np.searchsorted(cumvar, thr)) + 1
        print(f"    {int(thr*100)}%  → {n_needed} components")


def plot_scatter(coords2d: np.ndarray, all_labels: dict[str, np.ndarray],
                 results: dict[str, dict], out_path: Path) -> None:
    methods = list(all_labels.keys())
    n       = len(methods)
    cols    = min(n, 3)
    rows    = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows),
                             squeeze=False, constrained_layout=True)
    for ax, name in zip(axes.flat, methods):
        m = results[name]
        ax.scatter(coords2d[:, 0], coords2d[:, 1],
                   c=all_labels[name], cmap="tab10", s=1, alpha=0.4, linewidths=0)
        ax.set_title(
            f"{name}\nsil={m['silhouette']:.3f}  ari={m['bootstrap_ari']:.3f}",
            fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle("KMeans k=5 — 방법별 클러스터 라벨 (PCA-200 → UMAP-2D 투영)", fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=Path,
                    default=ROOT / "resource" / "embeddings_percol5")
    ap.add_argument("--sample",    type=int, default=50_000)
    ap.add_argument("--k",         type=int, default=5)
    ap.add_argument("--skip-umap", action="store_true",
                    help="UMAP 단계 건너뜀 (PCA만, 빠른 실행)")
    ap.add_argument("--scree-max", type=int, default=500,
                    help="scree plot 최대 PC 수 (기본 500)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 로드 + 정규화
    _, raw   = load_embeddings(args.embed_dir, args.sample)
    x_norm   = l2_normalize(raw)
    del raw; gc.collect()

    # 2. Scree plot
    plot_scree(x_norm, args.scree_max, OUT_DIR / "pca_scree.png")

    # 3. 방법별 실험
    results:    dict[str, dict]       = {}
    all_labels: dict[str, np.ndarray] = {}
    pca200: np.ndarray | None = None

    header = f"{'method':20s}  {'dim':>5}  {'sil':>7}  {'db':>7}  {'ch':>10}  {'ari':>6}  {'t(s)':>6}"
    print(f"\n{header}")
    print("─" * len(header))

    def _record(name: str, x_reduced: np.ndarray) -> None:
        x_n            = l2_normalize(x_reduced)
        labels, m      = run_kmeans(x_n, args.k)
        results[name]  = m
        all_labels[name] = labels
        print(f"  {name:20s}  {m['dim']:>5}  {m['silhouette']:>7.4f}"
              f"  {m['davies_bouldin']:>7.4f}  {m['calinski_harabasz']:>10.1f}"
              f"  {m['bootstrap_ari']:>6.3f}  {m['km_time_s']:>6.1f}")

    # baseline
    print(f"\n[baseline] 5120-dim")
    _record("baseline", x_norm)

    # PCA variants
    for n_pca in [50, 100, 200]:
        print(f"\n[pca_{n_pca}]")
        x_pca = reduce_pca(x_norm, n_pca)
        _record(f"pca_{n_pca}", x_pca)
        if n_pca == 200:
            pca200 = x_pca  # UMAP 입력 재사용

    # UMAP variants (PCA-200 → UMAP)
    if not args.skip_umap:
        for n_umap in [20, 50]:
            print(f"\n[umap_{n_umap}]  입력: PCA-200")
            x_umap = reduce_umap(pca200, n_umap)
            _record(f"umap_{n_umap}", x_umap)

    # 4. 결과 표 + 저장
    df = pd.DataFrame(results).T
    df.index.name = "method"
    print(f"\n{'─'*70}")
    print(df[["dim", "silhouette", "davies_bouldin",
              "calinski_harabasz", "bootstrap_ari"]].to_string())
    print(f"{'─'*70}")

    csv_path = OUT_DIR / "dim_reduction_comparison.csv"
    df.to_csv(csv_path)
    print(f"[save] {csv_path}")

    # 5. 시각화 — 모든 방법의 k5 라벨을 동일 2D 좌표 위에 표시
    print(f"\n[viz] 2D 투영 (PCA-200 → UMAP-2D) ...")
    base_for_viz = pca200 if pca200 is not None else reduce_pca(x_norm, 200)
    coords2d = reduce_umap(base_for_viz, 2)
    plot_scatter(coords2d, all_labels, results, OUT_DIR / "viz_dimred_k5.png")


if __name__ == "__main__":
    main()
