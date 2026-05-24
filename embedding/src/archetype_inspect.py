"""
archetype_inspect.py — Tier 2 archetype 정성 검증  [검증 스크립트]
작성: 2026-05-21
입력: resource/outputs/consumption_tags_n{N}.csv   — uuid + archetype (필수)
      resource/embeddings_percol5/                  — percol5 5120-dim 임베딩
      nvidia/Nemotron-Personas-Korea                 — AIO 원문 (5칼럼 텍스트)
출력: 표준 출력 — 각 archetype의 medoid + random 샘플 AIO 텍스트
연산: percol5 → L2 → PCA(100) → L2 공간에서 cluster centroid 최근접 medoid 선정
     원본 AIO 5칼럼(career/professional/family/travel/hobbies) 텍스트 표시

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
    "career_goals_and_ambitions",
    "professional_persona",
    "family_persona",
    "travel_persona",
    "hobbies_and_interests",
]


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
    cols   = ["uuid"] + SHOW_COLS
    print(f"[text] {name} ({len(target):,} targets) ...")
    ds = load_dataset(name, split="train", cache_dir=cache).select_columns(cols)

    result: dict[str, dict] = {}
    for row in ds:
        u = row["uuid"]
        if u in target:
            result[u] = {c: row.get(c) or "" for c in SHOW_COLS}
        if len(result) >= len(target):
            break
    return result


def find_medoid_pca(
    emb: np.ndarray, idx_cluster: np.ndarray, pca_proj: np.ndarray,
) -> int:
    """PCA(100) L2 공간의 클러스터 centroid 최근접 글로벌 인덱스."""
    coords   = pca_proj[idx_cluster]
    centroid = coords.mean(axis=0)
    dists    = np.linalg.norm(coords - centroid, axis=1)
    return int(idx_cluster[dists.argmin()])


def pick_samples(
    emb: np.ndarray, archetype: np.ndarray, target_c: list[int], n_random: int,
) -> dict[int, dict]:
    """클러스터별 medoid + random idx 수집. PCA(100) 공간에서 거리 계산."""
    print(f"[pca] {emb.shape[1]} → 100 + L2 ...")
    x      = l2_norm(emb)
    x_pca  = l2_norm(PCA(n_components=100, random_state=SEED).fit_transform(x).astype(np.float32))
    del x; gc.collect()

    rng = np.random.default_rng(SEED)
    out: dict[int, dict] = {}
    for c in target_c:
        idx_all = np.where(archetype == c)[0]
        if len(idx_all) == 0:
            continue
        medoid = find_medoid_pca(emb, idx_all, x_pca)
        choice = rng.choice(idx_all, size=min(n_random + 1, len(idx_all)),
                            replace=False)
        rand_idx = [int(i) for i in choice if i != medoid][:n_random]
        out[c] = {"medoid": medoid, "random": rand_idx, "n": int(len(idx_all))}
    return out


def print_sample(label: str, uid: str, texts: dict[str, dict]) -> None:
    row = texts.get(uid, {})
    print(f"\n  [{label}] {uid[:8]}...")
    for col in SHOW_COLS:
        val = row.get(col, "(없음)")
        print(f"    {col:30s}: {val}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",      type=int, default=200_000)
    ap.add_argument("--clusters",    type=int, nargs="+", default=None,
                    help="대상 archetype 번호 (기본: 모두)")
    ap.add_argument("--per-cluster", type=int, default=3,
                    help="클러스터당 random 샘플 수 (기본 3)")
    ap.add_argument("--config",      type=Path, default=ROOT / "config.toml")
    args = ap.parse_args()

    # 1. archetype 라벨
    csv_path = OUT_DIR / f"archetype_n{args.sample}.csv"
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path)
    uuids     = df["uuid"].tolist()
    archetype = df["archetype"].to_numpy()

    # 2. percol5 임베딩
    emb = load_percol5_subset(uuids)

    # 3. 대상 클러스터
    all_c    = sorted(set(int(c) for c in archetype))
    target_c = args.clusters if args.clusters is not None else all_c

    # 4. 샘플 선택
    picks = pick_samples(emb, archetype, target_c, args.per_cluster)
    del emb; gc.collect()

    # 5. 텍스트 로드
    need_uuids: list[str] = []
    for v in picks.values():
        need_uuids.append(uuids[v["medoid"]])
        need_uuids += [uuids[i] for i in v["random"]]
    cfg   = load_cfg(args.config)
    texts = load_texts(need_uuids, cfg)

    # 6. 출력
    print(f"\n{'='*70}")
    print(f"  Tier 2 archetype 정성 검증  (n={args.sample:,})")
    print(f"  clusters: {target_c}  |  per-cluster random={args.per_cluster}")
    print(f"{'='*70}")

    for c in target_c:
        if c not in picks:
            continue
        info = picks[c]
        bar = "─" * 70
        print(f"\n{bar}")
        print(f"  ARCHETYPE {c}   (n={info['n']:,})")
        print(bar)
        print_sample("medoid", uuids[info["medoid"]], texts)
        for i in info["random"]:
            print_sample("random", uuids[i], texts)


if __name__ == "__main__":
    main()
