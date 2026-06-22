"""
qualitative_validation.py — UMAP 2D KMeans k=5 정성 검증

파이프라인: percol5 → PCA(100) → UMAP(2D) → KMeans k=5
각 클러스터별:
  - medoid 1명 (centroid 최근접 점)
  - random 샘플 N명
원문 5칼럼 출력 → 통합 5분류 1:1 매핑 수동 확인

Usage:
    uv run src/qualitative_validation.py
    uv run src/qualitative_validation.py --sample 50000 --per-cluster 3
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pyarrow.dataset as pa_ds
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT    = Path(__file__).parent.parent
SEED    = 42

EMBED_DIR  = ROOT / "resource" / "embeddings_percol5"
SHOW_COLS  = [
    "career_goals_and_ambitions",
    "professional_persona",
    "family_persona",
    "travel_persona",
    "hobbies_and_interests",
]

# 통합 5분류 레퍼런스 (수동 매핑용)
ARCHETYPE_REF = """
통합 5분류 (매핑 기준):
  ① 계획적 절약   — 가계부·단골·가격비교·일상 효율
  ② 목표 지향 저축 — 장기 자산·자격증·내 집 마련
  ③ 자기보상·경험  — 즉시 만족·트렌드·자기 보상
  ④ 과시·자기표현  — 사회적 위치·정체성 표현·명품
  ⑤ 안정·관계     — 익숙함·가족·관계 우선·변화 회피
"""


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_embeddings(sample: int) -> tuple[list[str], np.ndarray]:
    table   = pa_ds.dataset(EMBED_DIR, format="parquet").to_table(
                  columns=["uuid", "embedding"])
    n_total = len(table)
    uuids_all = table.column("uuid").to_pylist()
    flat    = (table.column("embedding").combine_chunks()
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

    print(f"[load] {len(uuids):,} × {dim}")
    return uuids, arr


def load_texts(uuids: list[str], cfg_path: Path) -> dict[str, dict]:
    import tomllib
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    name  = cfg["dataset"]["name"]
    cache = cfg["dataset"]["cache_dir"] or None

    print(f"[text] 데이터셋 로드: {name}")
    target = set(uuids)
    cols   = ["uuid"] + SHOW_COLS
    ds     = load_dataset(name, split="train", cache_dir=cache)
    ds     = ds.select_columns(cols)

    result: dict[str, dict] = {}
    for row in ds:
        u = row["uuid"]
        if u in target:
            result[u] = {c: row.get(c) or "" for c in SHOW_COLS}
        if len(result) >= len(target):
            break

    print(f"[text] 매칭 {len(result):,} / {len(uuids):,}")
    return result


# ── 파이프라인 ─────────────────────────────────────────────────────────────────

def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-10, None)


def run_pipeline(emb: np.ndarray, pca_n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """percol5 → PCA → UMAP 2D → KMeans. (coords2d, labels) 반환."""
    import umap as umap_lib

    x = l2_norm(emb)
    print(f"[pca] → {pca_n} ...")
    x = l2_norm(PCA(n_components=pca_n, random_state=SEED).fit_transform(x).astype(np.float32))

    print(f"[umap] → 2D ...")
    coords = umap_lib.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=SEED,
    ).fit_transform(x).astype(np.float32)

    print(f"[kmeans] k={k} ...")
    labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(coords)
    return coords, labels.astype(np.int32)


# ── medoid + 샘플 선택 ─────────────────────────────────────────────────────────

def pick_representatives(
    uuids: list[str], coords: np.ndarray, labels: np.ndarray,
    k: int, n_random: int, rng_seed: int = SEED,
) -> dict[int, dict[str, list[str]]]:
    """클러스터별 medoid UUID + random UUID 목록 반환."""
    rng = np.random.default_rng(rng_seed)
    result: dict[int, dict[str, list[str]]] = {}

    for c in range(k):
        mask    = labels == c
        idx_all = np.where(mask)[0]
        coords_c = coords[idx_all]
        centroid = coords_c.mean(axis=0)
        dists    = np.linalg.norm(coords_c - centroid, axis=1)
        medoid_local = int(dists.argmin())
        medoid_uuid  = uuids[idx_all[medoid_local]]

        rand_local = rng.choice(len(idx_all),
                                size=min(n_random, len(idx_all)),
                                replace=False)
        rand_uuids = [uuids[idx_all[i]] for i in rand_local
                      if uuids[idx_all[i]] != medoid_uuid][:n_random]

        result[c] = {"medoid": [medoid_uuid], "random": rand_uuids}
    return result


# ── 출력 ───────────────────────────────────────────────────────────────────────

def print_persona(uid: str, texts: dict[str, dict], label: str) -> None:
    row = texts.get(uid, {})
    print(f"\n  [{label}] {uid[:8]}...")
    for col in SHOW_COLS:
        val = row.get(col, "(없음)")
        print(f"    {col[:28]:28s}: {val}")


def print_cluster(
    cluster_id: int, reps: dict[str, list[str]],
    texts: dict[str, dict], cluster_size: int,
) -> None:
    bar = "═" * 64
    print(f"\n{bar}")
    print(f"  CLUSTER {cluster_id}  (n={cluster_size:,})")
    print(f"  → 어느 아키타입? ① 계획적절약 ② 목표저축 ③ 자기보상 ④ 과시 ⑤ 안정관계")
    print(bar)

    for uid in reps["medoid"]:
        print_persona(uid, texts, "medoid")

    if reps["random"]:
        print(f"\n  ── random sample ──")
        for uid in reps["random"]:
            print_persona(uid, texts, "random")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",      type=int, default=50_000)
    ap.add_argument("--k",           type=int, default=5)
    ap.add_argument("--pca",         type=int, default=100)
    ap.add_argument("--per-cluster", type=int, default=3,
                    help="클러스터당 random 샘플 수")
    ap.add_argument("--config",      type=Path, default=ROOT / "config.toml")
    args = ap.parse_args()

    # 1. 임베딩 로드
    uuids, emb = load_embeddings(args.sample)

    # 2. 파이프라인
    coords, labels = run_pipeline(emb, args.pca, args.k)
    del emb; gc.collect()

    sizes = {c: int((labels == c).sum()) for c in range(args.k)}
    print(f"\n[cluster sizes] {sizes}")

    # 3. 대표 선택
    reps = pick_representatives(uuids, coords, labels, args.k, args.per_cluster)

    # 4. 텍스트 로드 (medoid + random 합집합)
    needed = []
    for v in reps.values():
        needed += v["medoid"] + v["random"]
    texts = load_texts(needed, args.config)

    # 5. 출력
    print(f"\n{ARCHETYPE_REF}")
    print(f"{'='*64}")
    print(f"  percol5 → PCA({args.pca}) → UMAP(2D) → KMeans k={args.k}")
    print(f"  n={args.sample:,}  per-cluster random={args.per_cluster}")
    print(f"{'='*64}")

    for c in range(args.k):
        print_cluster(c, reps[c], texts, sizes[c])

    print(f"\n{'='*64}")
    print("▶ 위 출력을 보고 각 클러스터에 ①~⑤ 레이블을 매핑하세요.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
