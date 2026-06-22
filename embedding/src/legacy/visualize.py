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
plt.rcParams["font.family"]      = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "resource" / "outputs"

SEED = 42


def load_aligned(embed_dir: Path, uuids: list[str]) -> np.ndarray:
    """uuid 순서에 맞춘 정규화된 embeddings 배열.

    스트리밍: uuid → 원본 행 인덱스 dict 1회 빌드 → 청크 순회 시 해당 행만 fp32로 슬롯 채움.
    풀 fp16(~10GB)+fp32(~10GB) 더블링 회피.
    """
    if not any(embed_dir.glob("*.parquet")):
        sys.exit(f"[load] 임베딩 없음: {embed_dir}")

    ds        = pa_ds.dataset(embed_dir, format="parquet")
    all_uuids = ds.to_table(columns=["uuid"]).column("uuid").to_pylist()
    uuid2pos  = {u: i for i, u in enumerate(all_uuids)}
    del all_uuids; gc.collect()

    target_pos   = np.fromiter((uuid2pos[u] for u in uuids),
                               dtype=np.int64, count=len(uuids))
    del uuid2pos; gc.collect()

    sorted_ord = np.argsort(target_pos)        # 출력 슬롯 인덱스
    sorted_pos = target_pos[sorted_ord]         # 정렬된 원본 위치

    T   = len(uuids)
    out: np.ndarray | None = None
    dim = -1
    row_off = 0
    BATCH   = 20_000

    for batch in ds.to_batches(columns=["embedding"], batch_size=BATCH):
        n_b = batch.num_rows
        lo  = int(np.searchsorted(sorted_pos, row_off,       side="left"))
        hi  = int(np.searchsorted(sorted_pos, row_off + n_b, side="left"))
        if lo == hi:
            row_off += n_b
            continue

        local_pos    = sorted_pos[lo:hi] - row_off
        target_slots = sorted_ord[lo:hi]
        if out is None:
            dim = batch.column("embedding").type.list_size
            out = np.zeros((T, dim), dtype=np.float16)

        flat = (batch.column("embedding").flatten()
                     .to_numpy(zero_copy_only=False))
        out[target_slots] = flat.reshape(n_b, dim)[local_pos].astype(np.float16, copy=False)
        row_off += n_b
        del flat

    if out is None:
        raise RuntimeError(f"[load] 빈 dataset: {embed_dir}")

    f32 = out.astype(np.float32)
    norms = np.linalg.norm(f32, axis=1, keepdims=True)
    f32 /= np.clip(norms, 1e-10, None)
    out[:] = f32
    del f32, norms; gc.collect()
    print(f"[load] {T:,} × {dim}  (정규화 완료)")
    return out


def project(embeddings: np.ndarray) -> np.ndarray:
    import umap
    print(f"[umap] {len(embeddings):,} × {embeddings.shape[1]} → 2D  (cosine, seed={SEED}) ...")
    reducer = umap.UMAP(
        n_components=2, random_state=SEED,
        n_neighbors=15, min_dist=0.1, metric="cosine",
    )
    coords = reducer.fit_transform(embeddings)
    print(f"[umap] done")
    return coords.astype(np.float16)


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
            "x":    coords[:, 0].astype(np.float16),
            "y":    coords[:, 1].astype(np.float16),
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
