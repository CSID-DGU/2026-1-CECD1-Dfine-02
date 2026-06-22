"""
embed_3col_compare.py — 3칼럼 3가지 임베딩 방식 비교 시각화

비교 대상:
  A. 3col-concat  : 3칼럼 합쳐 단일 텍스트 → BGE-M3 1회 → 1024-dim
  B. 3col-percol  : 3칼럼 각각 임베딩 후 concat → 3 × 1024 = 3072-dim
  C. percol5      : 기존 5칼럼 각각 임베딩 (저장된 parquet 로드) → 5120-dim

3칼럼: career_goals_and_ambitions / cultural_background / persona

Usage:
    uv run src/embed_3col_compare.py
    uv run src/embed_3col_compare.py --sample 10000
"""

import argparse
import gc
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as pa_ds
import torch
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
)

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "resource" / "outputs"
SEED    = 42

COLS_3 = [
    ("career_goals_and_ambitions", "커리어 목표"),
    ("cultural_background",        "문화적 배경"),
    ("persona",                    "페르소나"),
]

PERCOL5_DIR = ROOT / "resource" / "embeddings_percol5"
PERCOL5_DIM = 5120


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_rows(sample: int, cfg_path: Path) -> tuple[list[str], list[dict]]:
    """uuid 리스트 + 3칼럼 dict 리스트 반환 (로컬 캐시 사용)."""
    import tomllib
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    name  = cfg["dataset"]["name"]
    cache = cfg["dataset"]["cache_dir"] or None
    seed  = cfg["dataset"]["seed"]
    cols  = ["uuid"] + [c for c, _ in COLS_3]

    print(f"[load] {name}  random {sample:,} rows (cache) ...")
    ds = load_dataset(name, split="train", cache_dir=cache)
    ds = ds.select_columns(cols).shuffle(seed=seed).select(range(sample))

    uuids = ds["uuid"]
    rows  = [{col: (row.get(col) or "") for col, _ in COLS_3} for row in ds]
    print(f"[load] {len(uuids):,} rows")
    return uuids, rows


def load_percol5(uuids: list[str]) -> np.ndarray:
    """percol5 parquet에서 uuid 순서에 맞춰 (N, 5120) float32 반환."""
    table     = pa_ds.dataset(PERCOL5_DIR, format="parquet").to_table(
                    columns=["uuid", "embedding"])
    all_uids  = table.column("uuid").to_pylist()
    flat      = (table.column("embedding").combine_chunks()
                      .flatten().to_numpy(zero_copy_only=False))
    del table; gc.collect()

    n_total   = len(all_uids)
    arr       = flat.reshape(n_total, PERCOL5_DIM).astype(np.float32)
    del flat;  gc.collect()

    uid_idx   = {u: i for i, u in enumerate(all_uids)}
    missing   = [u for u in uuids if u not in uid_idx]
    if missing:
        print(f"[warn] percol5에 없는 uuid {len(missing)}개 → 0벡터 대체")

    out = np.zeros((len(uuids), PERCOL5_DIM), dtype=np.float32)
    for i, u in enumerate(uuids):
        if u in uid_idx:
            out[i] = arr[uid_idx[u]]
    del arr; gc.collect()

    print(f"[load] percol5 {len(uuids):,} rows  dim={PERCOL5_DIM}")
    return out


# ── 임베딩 ─────────────────────────────────────────────────────────────────────

def _load_model(cfg_path: Path):
    import tomllib
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[cfg["model"]["dtype"]]
    model = BGEM3FlagModel(cfg["model"]["name"], use_fp16=False,
                           devices=[cfg["runtime"]["device"]])
    model.model = model.model.to(dtype)
    return model, cfg


def _encode(model, texts: list[str], bs: int, ml: int) -> np.ndarray:
    out = model.encode(texts, batch_size=bs, max_length=ml,
                       return_dense=True, return_sparse=False, return_colbert_vecs=False)
    torch.cuda.synchronize()
    return np.asarray(out["dense_vecs"], dtype=np.float32)


def embed_concat(rows: list[dict], cfg_path: Path) -> np.ndarray:
    """3칼럼 합쳐 단일 텍스트 → 1024-dim."""
    texts = [" | ".join(f"{lbl}: {r[col]}" for col, lbl in COLS_3 if r.get(col))
             for r in rows]
    model, cfg = _load_model(cfg_path)
    bs, ml = cfg["model"]["batch_size"], cfg["model"]["max_length"]

    print(f"\n[embed-concat] 3col → 1024-dim  (bs={bs}) ...")
    _encode(model, texts[:bs], bs, ml)          # 웜업
    t0   = time.perf_counter()
    vecs = _encode(model, texts, bs, ml)
    print(f"  done {time.perf_counter()-t0:.1f}s  shape={vecs.shape}")
    return vecs


def embed_percol(rows: list[dict], cfg_path: Path) -> np.ndarray:
    """3칼럼 각각 임베딩 후 concat → 3072-dim."""
    model, cfg = _load_model(cfg_path)
    bs, ml = cfg["model"]["batch_size"], cfg["model"]["max_length"]

    print(f"\n[embed-percol] 3col × 1024 = 3072-dim  (bs={bs}) ...")
    col_vecs = []
    for col, lbl in COLS_3:
        texts = [r.get(col) or "" for r in rows]
        t0    = time.perf_counter()
        vecs  = _encode(model, texts, bs, ml)
        print(f"  {lbl:10s}  {time.perf_counter()-t0:.1f}s")
        col_vecs.append(vecs)

    combined = np.concatenate(col_vecs, axis=1)
    print(f"  concat shape={combined.shape}")
    return combined


# ── 클러스터링·지표 ────────────────────────────────────────────────────────────

def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-10, None)


def pca_reduce(x: np.ndarray, n: int) -> np.ndarray:
    return PCA(n_components=n, random_state=SEED).fit_transform(x).astype(np.float32)


def cluster_and_metrics(x_norm: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(x_norm)
    sil    = silhouette_score(x_norm, labels,
                              sample_size=min(5_000, len(labels)), random_state=SEED)

    # bootstrap ARI (50%)
    rng = np.random.default_rng(SEED + 1)
    n, m = len(x_norm), len(x_norm) // 2
    ia = np.sort(rng.choice(n, m, replace=False))
    ib = np.sort(rng.choice(n, m, replace=False))
    la = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(x_norm[ia])
    lb = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(x_norm[ib])
    _, ia2, ib2 = np.intersect1d(ia, ib, return_indices=True)
    ari = float(adjusted_rand_score(la[ia2], lb[ib2]))

    return labels, {"silhouette": sil, "bootstrap_ari": ari, "dim": x_norm.shape[1]}


# ── 시각화 ─────────────────────────────────────────────────────────────────────

def umap_2d(x: np.ndarray) -> np.ndarray:
    import umap as umap_lib
    print(f"[umap] {x.shape} → 2D ...")
    return umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                         metric="cosine", random_state=SEED).fit_transform(x)


def make_comparison(panels: list[tuple], out_path: Path) -> None:
    """panels: list of (coords, labels, metrics, title_str)"""
    n    = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), constrained_layout=True)

    for ax, (coords, labels, m, title) in zip(axes, panels):
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=labels, cmap="tab10", s=2, alpha=0.5, linewidths=0)
        ax.set_title(
            f"{title}\nsilhouette={m['silhouette']:.4f}  ari={m['bootstrap_ari']:.3f}",
            fontsize=10, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    n_pts = len(panels[0][1])
    fig.suptitle(f"KMeans k=5  (n={n_pts:,})", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",  type=int, default=10_000)
    ap.add_argument("--k",       type=int, default=5)
    ap.add_argument("--pca",     type=int, default=100,
                    help="PCA 차원 (0=비활성화)")
    ap.add_argument("--config",  type=Path, default=ROOT / "config.toml")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    uuids, rows = load_rows(args.sample, args.config)

    # 2. 임베딩 생성
    vA = embed_concat(rows, args.config)
    vB = embed_percol(rows, args.config)
    vC = load_percol5(uuids)

    # 3. L2 정규화
    nA = l2_norm(vA); del vA; gc.collect()
    nB = l2_norm(vB); del vB; gc.collect()
    nC = l2_norm(vC); del vC; gc.collect()

    # 4. PCA 차원 통일
    pca_n = args.pca
    if pca_n > 0:
        print(f"\n[pca] 모든 임베딩 → {pca_n}-dim")
        nA = l2_norm(pca_reduce(nA, pca_n))
        nB = l2_norm(pca_reduce(nB, pca_n))
        nC = l2_norm(pca_reduce(nC, pca_n))
        dim_label = f"→PCA{pca_n}"
    else:
        dim_label = ""

    # 5. KMeans + 지표
    print(f"\n[cluster] KMeans k={args.k}")
    labA, mA = cluster_and_metrics(nA, args.k)
    labB, mB = cluster_and_metrics(nB, args.k)
    labC, mC = cluster_and_metrics(nC, args.k)

    print(f"\n{'method':16s}  {'dim':>5}  {'silhouette':>10}  {'ari':>6}")
    print("─" * 44)
    for name, m in [("3col-concat", mA), ("3col-percol", mB), ("percol5", mC)]:
        print(f"{name:16s}  {m['dim']:>5}  {m['silhouette']:>10.4f}  {m['bootstrap_ari']:>6.3f}")

    # 6. UMAP 2D (각자)
    print()
    cA = umap_2d(nA)
    cB = umap_2d(nB)
    cC = umap_2d(nC)

    # 7. 비교 시각화
    panels = [
        (cA, labA, mA, f"3col-concat 1024{dim_label}"),
        (cB, labB, mB, f"3col-percol 3072{dim_label}"),
        (cC, labC, mC, f"percol5     5120{dim_label}"),
    ]
    out = OUT_DIR / f"compare_3col_pca{pca_n}_n{args.sample}.png"
    make_comparison(panels, out)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
