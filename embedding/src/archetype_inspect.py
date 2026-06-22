"""
archetype_inspect.py — Tier 2 archetype 정성 검증  [검증 스크립트]
작성: 2026-05-21
입력: resource/outputs/consumption_tags_n{N}.csv   — uuid + archetype (필수)
      resource/embeddings_percol5/                  — percol5 5120-dim 임베딩
      nvidia/Nemotron-Personas-Korea                 — AIO 원문 (5칼럼 텍스트)
출력: 표준 출력 — 각 archetype의 클러스터별 medoid 10~15개 AIO 텍스트
연산: percol5 → L2 → PCA(100) → L2 공간에서 cluster centroid 최근접 medoid 선정
     원본 AIO 5칼럼(cultural/career/family/travel/arts + age/sex/occupation) 텍스트 표시

Usage:
    uv run src/archetype_inspect.py --sample 200000
    uv run src/archetype_inspect.py --sample 200000 --clusters 2 3 --per-cluster 5
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
from datasets import load_dataset
from sklearn.decomposition import PCA

ROOT       = Path(__file__).parent.parent
OUT_DIR    = ROOT / "resource" / "outputs"
EMBED_DIR  = ROOT / "resource" / "embeddings_percol5"
SEED       = 42

SHOW_COLS = [
    "cultural_background",
    "career_goals_and_ambitions",
    "family_persona",
    "travel_persona",
    "arts_persona",
]
DEMO_COLS = ["age", "sex", "occupation"]


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-10, None)


def load_cfg(cfg_path: Path) -> dict:
    import tomllib
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def load_percol5_subset(uuids: list[str]) -> np.ndarray:
    """percol5 parquet에서 uuid 순서대로 (N, 5120) 추출."""
    print(f"[percol5] loading {EMBED_DIR} ...")
    table     = pa_ds.dataset(EMBED_DIR, format="parquet").to_table(
                    columns=["uuid", "embedding"])
    n_total   = len(table)
    all_uids  = table.column("uuid").to_pylist()
    flat      = (table.column("embedding").combine_chunks()
                      .flatten().to_numpy(zero_copy_only=False))
    del table; gc.collect()

    dim = flat.size // n_total
    arr = flat.reshape(n_total, dim).astype(np.float32)
    del flat; gc.collect()

    idx = {u: i for i, u in enumerate(all_uids)}
    out = np.zeros((len(uuids), dim), dtype=np.float32)
    missing = 0
    for i, u in enumerate(uuids):
        if u in idx:
            out[i] = arr[idx[u]]
        else:
            missing += 1
    del arr; gc.collect()

    if missing:
        print(f"[warn] {missing} uuids not found in percol5")
    print(f"[percol5] subset {out.shape}")
    return out


def load_texts(uuids: list[str], cfg: dict) -> dict[str, dict]:
    name  = cfg["dataset"]["name"]
    cache = cfg["dataset"]["cache_dir"] or None
    target = set(uuids)
    cols   = ["uuid"] + DEMO_COLS + SHOW_COLS
    print(f"[text] {name} ({len(target):,} targets) ...")
    ds = load_dataset(name, split="train", cache_dir=cache).select_columns(cols)

    result: dict[str, dict] = {}
    for row in ds:
        u = row["uuid"]
        if u in target:
            result[u] = {c: row.get(c) or "" for c in DEMO_COLS + SHOW_COLS}
        if len(result) >= len(target):
            break
    return result


def find_medoids_pca(
    idx_cluster: np.ndarray, pca_proj: np.ndarray, k: int,
) -> list[int]:
    """PCA(100) L2 공간 클러스터 centroid 최근접 top-k 글로벌 인덱스 (가까운 순)."""
    coords   = pca_proj[idx_cluster]
    centroid = coords.mean(axis=0)
    dists    = np.linalg.norm(coords - centroid, axis=1)
    order    = np.argsort(dists)[:k]
    return [int(idx_cluster[i]) for i in order]


def pick_samples(
    emb: np.ndarray, archetype: np.ndarray, target_c: list[int], n_medoid: int,
) -> dict[int, dict]:
    """클러스터별 centroid 최근접 medoid n_medoid개 수집. PCA(100) 공간 거리."""
    print(f"[pca] {emb.shape[1]} → 100 + L2 ...")
    x      = l2_norm(emb)
    x_pca  = l2_norm(PCA(n_components=100, random_state=SEED).fit_transform(x).astype(np.float32))
    del x; gc.collect()

    out: dict[int, dict] = {}
    for c in target_c:
        idx_all = np.where(archetype == c)[0]
        if len(idx_all) == 0:
            continue
        medoids = find_medoids_pca(idx_all, x_pca, n_medoid)
        out[c] = {"medoids": medoids, "n": int(len(idx_all))}
    return out


def print_sample(label: str, uid: str, texts: dict[str, dict]) -> None:
    row = texts.get(uid, {})
    demo = " / ".join(str(row.get(c, "?")) for c in DEMO_COLS)
    print(f"\n  [{label}] {uid[:8]}...  ({demo})")
    for col in SHOW_COLS:
        val = row.get(col, "(없음)")
        print(f"    {col:30s}: {val}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",      type=int, default=200_000)
    ap.add_argument("--clusters",    type=int, nargs="+", default=None,
                    help="대상 archetype 번호 (기본: 모두)")
    ap.add_argument("--per-cluster", type=int, default=12,
                    help="클러스터당 medoid 수 (기본 12)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="archetype csv 경로 직접 지정 (기본: archetype_n{sample}.csv)")
    ap.add_argument("--regress-out", type=str, default="",
                    help="medoid 선정 전 인구통계 factor 회귀제거 (예: age,sex). 클러스터링과 동일 공간 맞춤")
    ap.add_argument("--config",      type=Path, default=ROOT / "config.toml")
    args = ap.parse_args()

    # 1. archetype 라벨
    csv_path = args.csv if args.csv is not None else OUT_DIR / f"archetype_n{args.sample}.csv"
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path)
    if args.clusters is not None:
        df = df[df["archetype"].isin(args.clusters)].reset_index(drop=True)
        print(f"[filter] clusters={args.clusters} → {len(df):,} rows")
    uuids     = df["uuid"].tolist()
    archetype = df["archetype"].to_numpy()

    # 2. percol5 임베딩
    emb = load_percol5_subset(uuids)

    # 2b. (옵션) 잔차화 — 클러스터링과 동일 공간에서 medoid 선정
    if args.regress_out:
        from archetype_cluster import regress_out
        factors = [f.strip() for f in args.regress_out.split(",") if f.strip()]
        emb     = regress_out(emb, uuids, factors)

    # 3. 대상 클러스터
    all_c    = sorted(set(int(c) for c in archetype))
    target_c = args.clusters if args.clusters is not None else all_c

    # 4. 샘플 선택
    picks = pick_samples(emb, archetype, target_c, args.per_cluster)
    del emb; gc.collect()

    # 5. 텍스트 로드
    need_uuids: list[str] = []
    for v in picks.values():
        need_uuids += [uuids[i] for i in v["medoids"]]
    cfg   = load_cfg(args.config)
    texts = load_texts(need_uuids, cfg)

    # 6. 출력
    print(f"\n{'='*70}")
    print(f"  Tier 2 archetype 정성 검증  (n={args.sample:,})")
    print(f"  clusters: {target_c}  |  medoids/cluster={args.per_cluster}")
    print(f"{'='*70}")

    for c in target_c:
        if c not in picks:
            continue
        info = picks[c]
        bar = "─" * 70
        print(f"\n{bar}")
        print(f"  ARCHETYPE {c}   (n={info['n']:,})")
        print(bar)
        for rank, i in enumerate(info["medoids"]):
            print_sample(f"medoid{rank+1}", uuids[i], texts)


if __name__ == "__main__":
    main()
