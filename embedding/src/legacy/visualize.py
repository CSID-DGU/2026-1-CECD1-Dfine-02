"""
visualize.py — sweep 결과(labels parquet) UMAP 2D 시각화.

산출:
    resource/outputs/umap_{base}.parquet  — uuid, x, y (재사용용 캐시)
    resource/outputs/viz_{base}.png       — k별 subplot

Usage:
    uv run src/visualize.py --base mini_embeddings_percol5
    uv run src/visualize.py --base mini_embeddings_percol5 --k 5
    uv run src/visualize.py --base mini_embeddings_percol5 --reproject
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "resource" / "outputs"

SEED = 42


def load_aligned(embed_dir: Path, uuids: list[str]) -> np.ndarray:
    """uuid 순서에 맞춘 정규화된 embeddings 배열."""
    if not any(embed_dir.glob("*.parquet")):
        sys.exit(f"[load] 임베딩 없음: {embed_dir}")

    table = pa_ds.dataset(embed_dir, format="parquet").to_table(columns=["uuid", "embedding"])
    all_uuids = table.column("uuid").to_pylist()
    flat = table.column("embedding").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    del table
    gc.collect()

    n_total = len(all_uuids)
    dim     = flat.size // n_total
    arr_f16 = flat.reshape(n_total, dim)

    uuid_to_idx = {u: i for i, u in enumerate(all_uuids)}
    sel = np.fromiter((uuid_to_idx[u] for u in uuids), dtype=np.int64, count=len(uuids))
    embeddings = arr_f16[sel].astype(np.float32)
    del flat, arr_f16
    gc.collect()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.clip(norms, 1e-10, None)
    print(f"[load] {len(embeddings):,} × {dim}  (정규화 완료)")
    return embeddings


def project(embeddings: np.ndarray) -> np.ndarray:
    import umap
    print(f"[umap] {len(embeddings):,} × {embeddings.shape[1]} → 2D  (cosine, seed={SEED}) ...")
    reducer = umap.UMAP(
        n_components=2, random_state=SEED,
        n_neighbors=15, min_dist=0.1, metric="cosine",
    )
    coords = reducer.fit_transform(embeddings)
    print(f"[umap] done")
    return coords.astype(np.float32)


def plot(coords: np.ndarray, labels_df: pd.DataFrame,
         k_cols: list[str], out_path: Path, title: str) -> None:
    n    = len(k_cols)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             squeeze=False, constrained_layout=True)

    for ax, col in zip(axes.flat, k_cols):
        labels = labels_df[col].to_numpy()
        k      = int(col[1:])
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
                   s=1, alpha=0.4, linewidths=0)
        ax.set_title(f"{col}  (n_clusters={k})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(f"UMAP — {title}", fontsize=14)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="공통 base, 예: mini_embeddings_percol5")
    ap.add_argument("--k", type=int, default=None,
                    help="특정 k만 시각화 (기본: 모든 k subplot)")
    ap.add_argument("--reproject", action="store_true",
                    help="기존 umap parquet 무시하고 다시 투영")
    args = ap.parse_args()

    labels_path = OUT_DIR / f"labels_{args.base}.parquet"
    meta_path   = OUT_DIR / f"sweep_{args.base}.json"
    umap_path   = OUT_DIR / f"umap_{args.base}.parquet"

    labels_df = pd.read_parquet(labels_path)
    print(f"[load] {labels_path}  ({len(labels_df):,} rows)")

    with meta_path.open() as f:
        meta = json.load(f)
    embed_dir = Path(meta["embed_dir"])
    if not embed_dir.is_absolute():
        embed_dir = ROOT / embed_dir

    if umap_path.exists() and not args.reproject:
        umap_df = pd.read_parquet(umap_path)
        print(f"[load] {umap_path}  (캐시 사용, --reproject로 재계산 강제)")
        labels_df = labels_df.merge(umap_df, on="uuid", how="inner")
        coords = labels_df[["x", "y"]].to_numpy()
    else:
        embeddings = load_aligned(embed_dir, labels_df["uuid"].tolist())
        coords = project(embeddings)
        del embeddings
        gc.collect()
        pd.DataFrame({
            "uuid": labels_df["uuid"].values,
            "x":    coords[:, 0],
            "y":    coords[:, 1],
        }).to_parquet(umap_path, index=False)
        print(f"[save] {umap_path}")

    k_cols = sorted(
        [c for c in labels_df.columns if c.startswith("k") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )
    if args.k is not None:
        k_cols = [f"k{args.k}"]
        if k_cols[0] not in labels_df.columns:
            sys.exit(f"[error] {k_cols[0]} 컬럼 없음. labels에 있는 k: "
                     f"{[c for c in labels_df.columns if c.startswith('k')]}")

    suffix = f"_k{args.k}" if args.k is not None else ""
    out_path = OUT_DIR / f"viz_{args.base}{suffix}.png"
    plot(coords, labels_df, k_cols, out_path, args.base)


if __name__ == "__main__":
    main()
