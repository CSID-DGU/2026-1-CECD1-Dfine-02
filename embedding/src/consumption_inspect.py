"""
consumption_inspect.py — Tier 3 소비태그 정성 검증  [검증 스크립트]
작성: 2026-05-21
입력: resource/outputs/consumption_tags_n{N}.csv      — uuid + consumption_tag (필수)
      resource/outputs/consumption_emb_n{N}.parquet    — uuid + 2048-dim 임베딩 (필수)
      resource/outputs/anchor_mapping_n{N}.json         — cluster → 레이블 (있으면 표시)
      nvidia/Nemotron-Personas-Korea                     — culinary + hobbies 원문
출력: 표준 출력 — 각 소비태그 클러스터의 medoid + random 샘플 원문
연산: 2048-dim 임베딩 공간에서 cluster 평균 최근접 medoid 선정
     culinary_persona + hobbies_and_interests_list 텍스트 표시

Usage:
    uv run src/consumption_inspect.py
    uv run src/consumption_inspect.py --clusters 1 4 --per-cluster 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset

ROOT       = Path(__file__).parent.parent
OUT_DIR    = ROOT / "resource" / "outputs"
SEED       = 42
EMBED_COLS = ["culinary_persona", "hobbies_and_interests_list"]


def load_cfg(cfg_path: Path) -> dict:
    import tomllib
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def load_emb(cache_path: Path, uuids: list[str]) -> np.ndarray:
    table = pq.read_table(cache_path)
    cached = table.column("uuid").to_pylist()
    flat   = (table.column("embedding").combine_chunks()
                   .flatten().to_numpy(zero_copy_only=False))
    dim    = flat.size // len(cached)
    arr    = flat.reshape(-1, dim).astype(np.float32)
    idx    = {u: i for i, u in enumerate(cached)}

    out = np.zeros((len(uuids), dim), dtype=np.float32)
    for i, u in enumerate(uuids):
        out[i] = arr[idx[u]]
    return out


def load_texts(uuids: list[str], cfg: dict) -> dict[str, dict]:
    name  = cfg["dataset"]["name"]
    cache = cfg["dataset"]["cache_dir"] or None
    target = set(uuids)
    cols   = ["uuid"] + EMBED_COLS
    print(f"[text] {name} ({len(target):,} targets) ...")
    ds = load_dataset(name, split="train", cache_dir=cache).select_columns(cols)

    def to_str(v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return v or ""

    result: dict[str, dict] = {}
    for row in ds:
        u = row["uuid"]
        if u in target:
            result[u] = {c: to_str(row.get(c)) for c in EMBED_COLS}
        if len(result) >= len(target):
            break
    return result


def pick_samples(
    emb: np.ndarray, cluster: np.ndarray, target_c: list[int], n_random: int,
) -> dict[int, dict]:
    """클러스터별 medoid + random idx 수집."""
    rng = np.random.default_rng(SEED)
    out: dict[int, dict] = {}
    for c in target_c:
        idx_all = np.where(cluster == c)[0]
        if len(idx_all) == 0:
            continue
        centroid = emb[idx_all].mean(axis=0)
        dists    = np.linalg.norm(emb[idx_all] - centroid, axis=1)
        medoid   = int(idx_all[dists.argmin()])

        choice   = rng.choice(idx_all, size=min(n_random + 1, len(idx_all)),
                              replace=False)
        rand_idx = [int(i) for i in choice if i != medoid][:n_random]
        out[c] = {"medoid": medoid, "random": rand_idx, "n": int(len(idx_all))}
    return out


def print_sample(label: str, uid: str, texts: dict[str, dict]) -> None:
    row = texts.get(uid, {})
    print(f"\n  [{label}] {uid[:8]}...")
    for col in EMBED_COLS:
        print(f"    {col}: {row.get(col, '(없음)')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",      type=int, default=50_000)
    ap.add_argument("--clusters",    type=int, nargs="+", default=None,
                    help="대상 클러스터 번호 (기본: 모두)")
    ap.add_argument("--per-cluster", type=int, default=3,
                    help="클러스터당 random 샘플 수 (기본 3)")
    ap.add_argument("--config",      type=Path, default=ROOT / "config.toml")
    args = ap.parse_args()

    # 1. 클러스터 라벨
    csv_path = OUT_DIR / f"consumption_cluster_n{args.sample}.csv"
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path)
    uuids   = df["uuid"].tolist()
    cluster = df["consumption_tag"].to_numpy()

    # 2. anchor mapping (있으면)
    map_path = OUT_DIR / f"anchor_mapping_n{args.sample}.json"
    anchor_map: dict[int, str] = {}
    if map_path.exists():
        with map_path.open(encoding="utf-8") as f:
            anchor_map = {int(k): v for k, v in json.load(f).items()}
        print(f"[load] {map_path}  ({len(anchor_map)} mappings)")

    # 3. 임베딩
    cache_path = OUT_DIR / f"consumption_emb_n{args.sample}.parquet"
    print(f"[load] {cache_path}")
    emb = load_emb(cache_path, uuids)
    print(f"[emb]  {emb.shape}")

    # 4. 대상 클러스터
    all_c    = sorted(set(int(c) for c in cluster))
    target_c = args.clusters if args.clusters is not None else all_c

    # 5. 샘플 선택
    picks = pick_samples(emb, cluster, target_c, args.per_cluster)
    del emb

    # 6. 텍스트 로드
    need_uuids: list[str] = []
    for v in picks.values():
        need_uuids.append(uuids[v["medoid"]])
        need_uuids += [uuids[i] for i in v["random"]]
    cfg   = load_cfg(args.config)
    texts = load_texts(need_uuids, cfg)

    # 7. 출력
    print(f"\n{'='*70}")
    print(f"  Tier 3 소비 클러스터 정성 검증")
    print(f"  clusters: {target_c}  |  per-cluster random={args.per_cluster}")
    print(f"{'='*70}")

    for c in target_c:
        if c not in picks:
            continue
        info = picks[c]
        label = anchor_map.get(c, "(미매핑)")
        bar = "─" * 70
        print(f"\n{bar}")
        print(f"  CLUSTER {c}  →  {label}   (n={info['n']:,})")
        print(bar)

        print_sample("medoid", uuids[info["medoid"]], texts)
        for i in info["random"]:
            print_sample("random", uuids[i], texts)


if __name__ == "__main__":
    main()
