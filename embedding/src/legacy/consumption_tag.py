"""
consumption_tag.py — Tier2 archetype + Tier3 소비태그 클러스터링  [파이프라인 Step 3]
작성: 2026-05-20
입력: resource/embeddings_percol5/embeddings_percol5.parquet  (percol5 임베딩)
      resource/outputs/consumption_emb_n{N}.parquet           (소비 임베딩, Step 2 산출물)
출력: resource/outputs/consumption_tags_n{N}.csv        — uuid / archetype / consumption_tag / noise_dist / entropy
      resource/outputs/matrix_5x5_n{N}.csv              — archetype × 소비태그 정합 매트릭스 (noise 제외)
연산:
  Tier 2: percol5 → L2(in-place) → PCA(100) → L2 → UMAP(2D) → KMeans k=5 → archetype 라벨
  Tier 3: consumption_emb → L2 → PCA(50) → L2 → UMAP(2D) → KMeans k=5
  noise_dist: 각 점의 k centroid까지 역거리 softmax → Shannon H 상위 noise_pct% → True

Usage:
    uv run src/consumption_tag.py --sample 1000000
    uv run src/consumption_tag.py --sample 50000 --noise-pct 10
"""

import argparse
import gc
import sys
import time
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
SEED      = 42


# ── 공통 유틸 ──────────────────────────────────────────────────────────────────

def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-10, None)


def umap2d(x: np.ndarray) -> np.ndarray:
    print(f"[umap] {x.shape[1]}-dim → 2D ...")
    t0 = time.perf_counter()
    out = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=SEED).fit_transform(x)
    print(f"  done {time.perf_counter()-t0:.1f}s")
    return out.astype(np.float32)


# ── Tier 2: percol5 → archetype 라벨 ──────────────────────────────────────────

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


def run_tier2(emb: np.ndarray, k: int) -> np.ndarray:
    """percol5 → L2(in-place) → PCA(100) → L2 → UMAP(2D) → KMeans k=5"""
    # n=1M에서 l2_norm 복사본(20GB)을 피하기 위해 in-place 정규화
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb /= np.clip(norms, 1e-10, None)
    del norms
    print(f"[tier2] PCA {emb.shape[1]} → 100 ...")
    x = l2_norm(PCA(n_components=100, random_state=SEED).fit_transform(emb).astype(np.float32))
    coords = umap2d(x)
    print(f"[tier2] KMeans k={k} ...")
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(coords).astype(np.int32)


# ── 소비 임베딩 로드 (embed_consumption.py 캐시) ──────────────────────────────

def load_consumption_emb(sample: int) -> tuple[list[str], np.ndarray]:
    """Step 2(embed_consumption.py)가 생성한 consumption_emb 파일을 로드."""
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


# ── Tier 3a: 소비태그 클러스터링 ─────────────────────────────────────────────

def run_tier3(emb: np.ndarray, pca_n: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    consumption emb → L2 → PCA(pca_n) → L2 → UMAP(2D) → KMeans k
    반환: coords(N,2), labels(N,), centroids(k,2)
    """
    x = l2_norm(emb)
    print(f"[tier3] PCA {x.shape[1]} → {pca_n} ...")
    x = l2_norm(PCA(n_components=pca_n, random_state=SEED).fit_transform(x).astype(np.float32))
    coords = umap2d(x)
    print(f"[tier3] KMeans k={k} ...")
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(coords)
    return coords, km.labels_.astype(np.int32), km.cluster_centers_.astype(np.float32)


# ── 엔트로피 ───────────────────────────────────────────────────────────────────

def compute_entropy(coords: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """각 점 → 5 centroid 역거리 softmax → Shannon H (nats)"""
    dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)  # (N, k)
    inv   = 1.0 / np.clip(dists, 1e-8, None)
    prob  = inv / inv.sum(axis=1, keepdims=True)
    H     = -(prob * np.log(np.clip(prob, 1e-12, None))).sum(axis=1)
    return H.astype(np.float32)


# ── 5×5 매트릭스 ──────────────────────────────────────────────────────────────

def print_and_save_matrix(
    archetype: np.ndarray, cons_tag: np.ndarray, k: int, out_path: Path
) -> None:
    mat = np.zeros((k, k), dtype=int)
    for a, c in zip(archetype, cons_tag):
        mat[a, c] += 1

    print(f"\n{'':14s}" + "".join(f"  ctag{c:1d}" for c in range(k)))
    print("  " + "-" * (12 + 7 * k))
    for a in range(k):
        row_str = f"  arch{a:1d}       |" + "".join(f"  {mat[a, c]:4d}" for c in range(k))
        print(row_str)

    df = pd.DataFrame(mat, index=[f"arch{a}" for a in range(k)],
                      columns=[f"ctag{c}" for c in range(k)])
    df.to_csv(out_path)
    print(f"\n[save] {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",    type=int,   default=50_000)
    ap.add_argument("--k",         type=int,   default=5)
    ap.add_argument("--pca2",      type=int,   default=50,
                    help="소비 임베딩 PCA 차원 (기본 50)")
    ap.add_argument("--noise-pct", type=float, default=10.0,
                    help="엔트로피 상위 N%% → noise_dist=True (기본 10)")
    ap.add_argument("--config",    type=Path,  default=ROOT / "config.toml")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. percol5 로드 → Tier 2 archetype 라벨
    uuids, emb5 = load_percol5(args.sample)
    archetype = run_tier2(emb5, k=args.k)
    del emb5; gc.collect()

    arch_sizes = {c: int((archetype == c).sum()) for c in range(args.k)}
    print(f"\n[tier2 sizes] {arch_sizes}")

    # 2. 소비 임베딩 로드 (Step 2 산출물)
    uuids_cons, cons_emb = load_consumption_emb(args.sample)
    if uuids != uuids_cons:
        sys.exit("[error] UUID 순서 불일치 — Step 2를 같은 --sample로 재실행하세요.")
    del uuids_cons

    # 3. Tier 3: 소비태그 클러스터링
    coords, cons_tag, centroids = run_tier3(cons_emb, pca_n=args.pca2, k=args.k)
    del cons_emb; gc.collect()

    cons_sizes = {c: int((cons_tag == c).sum()) for c in range(args.k)}
    print(f"\n[tier3 sizes] {cons_sizes}")

    # 4. 엔트로피 → noise_dist
    H         = compute_entropy(coords, centroids)
    threshold = np.percentile(H, 100 - args.noise_pct)
    noise_dist = (H >= threshold).astype(np.int8)
    print(f"\n[entropy]  mean={H.mean():.4f}  p90={np.percentile(H, 90):.4f}"
          f"  threshold(p{100-args.noise_pct:.0f})={threshold:.4f}"
          f"  noise_dist n={noise_dist.sum():,} ({noise_dist.mean()*100:.1f}%)")

    # 5. 5×5 정합 매트릭스
    print("\n" + "=" * 64)
    print("  5×5 정합 매트릭스 (archetype × 소비태그)")
    print("  → noise_dist=False 인원만 카운트")
    print("=" * 64)
    clean_mask = noise_dist == 0
    print_and_save_matrix(
        archetype[clean_mask], cons_tag[clean_mask], args.k,
        OUT_DIR / f"matrix_5x5_n{args.sample}.csv",
    )

    print("\n  (전체 포함 매트릭스)")
    print_and_save_matrix(
        archetype, cons_tag, args.k,
        OUT_DIR / f"matrix_5x5_all_n{args.sample}.csv",
    )

    # 8. UUID별 결과 저장
    df = pd.DataFrame({
        "uuid":            uuids,
        "archetype":       archetype,
        "consumption_tag": cons_tag,
        "noise_dist":      noise_dist,
        "entropy":         H,
    })
    out_csv = OUT_DIR / f"consumption_tags_n{args.sample}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")
    print(f"  columns: uuid / archetype(0-{args.k-1}) / consumption_tag(0-{args.k-1})"
          f" / noise_dist(0|1) / entropy")
    print(f"  rows: {len(df):,}  (noise_dist=1: {noise_dist.sum():,})")


if __name__ == "__main__":
    main()
