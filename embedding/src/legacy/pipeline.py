"""
pipeline.py — AIO 페르소나 KMeans 클러스터링 파이프라인.

계획서: AIO_KMeans_파이프라인_계획.md
입력:  resource/embeddings/*.parquet  (embed.py 출력)
출력:  resource/outputs/

Usage:
    uv run src/pipeline.py
    uv run src/pipeline.py --apply-labels
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT      = Path(__file__).parent.parent
EMBED_DIR = ROOT / "resource" / "embeddings"
DATA_DIR  = ROOT / "resource" / "nemotron_korea"
OUT_DIR   = ROOT / "resource" / "outputs"

COLS: list[tuple[str, str]] = [
    ("career_goals_and_ambitions", "커리어 목표"),
    ("professional_persona",       "직업적 자아"),
    ("family_persona",             "가족적 자아"),
    ("travel_persona",             "여행 성향"),
    ("hobbies_and_interests",      "취미와 관심사"),
]

K_RANGE   = range(3, 8)
N_MEDOIDS = 5
SEED      = 42


# ---------------------------------------------------------------------------
# Step 1 — 임베딩 로드
# ---------------------------------------------------------------------------

def load_embeddings(embed_dir: Path = EMBED_DIR) -> tuple[list[str], np.ndarray]:
    """*.parquet → (uuids, float32 행렬). embed_dir로 디렉토리 오버라이드 가능."""
    if not any(embed_dir.glob("*.parquet")):
        sys.exit(f"[load] 임베딩 파일 없음: {embed_dir}\n"
                 "embed.py를 먼저 실행하세요.")

    ds  = pa_ds.dataset(embed_dir, format="parquet")
    df  = ds.to_table(columns=["uuid", "embedding"]).to_pandas()
    uuids      = df["uuid"].tolist()
    embeddings = np.stack(df["embedding"].to_numpy()).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[load] {len(uuids):,} rows  dim={embeddings.shape[1]}")
    print(f"[load] norm  mean={norms.mean():.4f}  "
          f"min={norms.min():.4f}  max={norms.max():.4f}")
    return uuids, embeddings


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 정규화 — cosine 거리를 euclidean KMeans로 근사."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-10, None)


# ---------------------------------------------------------------------------
# Step 5 — k Sweep
# ---------------------------------------------------------------------------

def sweep_k(norm_emb: np.ndarray) -> tuple[dict, int]:
    """k=3..7 sweep. Silhouette 기준 best_k 반환. norm_emb는 L2 정규화 완료."""
    results: dict[int, dict] = {}

    print(f"\n[sweep] k sweep (KMeans, L2-normalized)")
    print(f"{'k':>3}  {'silhouette':>12}  {'davies_bouldin':>15}  {'calinski_harabasz':>18}")
    print("-" * 55)

    for k in K_RANGE:
        km     = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = km.fit_predict(norm_emb)
        sil    = silhouette_score(norm_emb, labels,
                                  sample_size=min(20_000, len(labels)),
                                  random_state=SEED)
        db     = davies_bouldin_score(norm_emb, labels)
        ch     = calinski_harabasz_score(norm_emb, labels)
        results[k] = {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
        print(f"{k:>3}  {sil:>12.4f}  {db:>15.4f}  {ch:>18.1f}")

    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"\n[sweep] best k={best_k}  "
          f"(silhouette={results[best_k]['silhouette']:.4f})")

    if best_k == 4:
        print("[sweep] NOTE: k=4 peak — ⑤ 무관심·의존 군집이 별도 분리 안 됨")
    elif best_k == 6:
        print("[sweep] NOTE: k=6 peak — Anxious spenders 분리 가능성, 6개 라벨링 검토")

    return results, best_k


# ---------------------------------------------------------------------------
# Step 6 — 클러스터링 & Medoid 추출
# ---------------------------------------------------------------------------

def cluster(norm_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """최적 k KMeans → (labels, centers). norm_emb는 L2 정규화 완료."""
    km     = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels = km.fit_predict(norm_emb)

    sizes    = pd.Series(labels).value_counts().sort_index()
    print(f"\n[cluster] k={k} 클러스터 크기:")
    for cid, cnt in sizes.items():
        pct = cnt / len(labels) * 100
        print(f"  [{cid}]  {cnt:>6,} rows  ({pct:5.1f}%)")

    if sizes.max() / len(labels) > 0.5:
        print(f"[cluster] WARN: 최대 군집 > 50% — 분리 실패 의심")
    if sizes.min() / len(labels) < 0.05:
        print(f"[cluster] WARN: 최소 군집 < 5% — 고립 군집 의심")

    return labels, km.cluster_centers_


def extract_medoids(
    uuids: list[str],
    norm_emb: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> dict[int, list[str]]:
    """각 클러스터에서 centroid 최근접 N_MEDOIDS개 uuid 반환. norm_emb는 L2 정규화 완료."""
    medoid_uuids = {}
    for cid in range(len(centers)):
        idx_in  = np.where(labels == cid)[0]
        dists   = np.linalg.norm(norm_emb[idx_in] - centers[cid], axis=1)
        top_idx = idx_in[np.argsort(dists)[:N_MEDOIDS]]
        medoid_uuids[cid] = [uuids[i] for i in top_idx]
    return medoid_uuids


def _load_texts_for_uuids(uuid_set: set[str]) -> dict[str, str]:
    """원본 데이터셋에서 uuid 집합의 AIO 텍스트 복원."""
    if not DATA_DIR.exists():
        print(f"[medoid] WARN: 원본 데이터셋 없음 ({DATA_DIR}) — uuid만 기록")
        return {}

    cols = ["uuid"] + [c for c, _ in COLS]
    ds   = pa_ds.dataset(DATA_DIR, format="parquet")
    df   = ds.to_table(columns=cols).to_pandas()
    df   = df[df["uuid"].isin(uuid_set)]

    texts = {}
    for _, row in df.iterrows():
        parts = [f"{label}: {row[col]}" for col, label in COLS if row.get(col)]
        texts[row["uuid"]] = " | ".join(parts)
    return texts


# ---------------------------------------------------------------------------
# Step 8 — 저장
# ---------------------------------------------------------------------------

def save_sweep(results: dict, tag: str = "") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    rows = [{"k": k, **v} for k, v in results.items()]
    path = OUT_DIR / f"k_sweep_results{suffix}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[save] {path}")


def save_medoids(medoid_uuids: dict[int, list[str]], texts: dict[str, str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cid, uuids in medoid_uuids.items():
        path  = OUT_DIR / f"cluster_{cid}_medoids.txt"
        lines = []
        for uid in uuids:
            lines.append(f"uuid: {uid}")
            lines.append(texts.get(uid, "(원본 데이터셋 필요)"))
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] medoid 파일 {len(medoid_uuids)}개 → {OUT_DIR}")

    mapping_path = OUT_DIR / "cluster_to_archetype.json"
    if not mapping_path.exists():
        template = {str(cid): "" for cid in medoid_uuids}
        mapping_path.write_text(
            json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[save] {mapping_path}  ← medoid 확인 후 아키타입 채워주세요")
        print("       선택지: Rational_Achiever | Cautious_Saver | Status_Expressive"
              " | Hedonic_Experiencer | Disengaged_Reactive")


def apply_labels(uuids: list[str], labels: np.ndarray) -> None:
    """cluster_to_archetype.json 기반 라벨 컬럼 추가 후 parquet 저장."""
    mapping_path = OUT_DIR / "cluster_to_archetype.json"
    if not mapping_path.exists():
        sys.exit(f"[label] {mapping_path} 없음 — 먼저 medoid 확인 후 작성하세요.")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    empty   = [k for k, v in mapping.items() if not v]
    if empty:
        sys.exit(f"[label] 빈 항목: {empty} — 모든 클러스터에 아키타입을 입력하세요.")

    df = pd.DataFrame({"uuid": uuids, "cluster_id": labels.tolist()})
    df["archetype"] = df["cluster_id"].map(lambda c: mapping[str(c)])

    out_path = OUT_DIR / "personas_labeled.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[label] → {out_path}")
    print(df["archetype"].value_counts().to_string())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply-labels", action="store_true",
                    help="cluster_to_archetype.json 기반 라벨링 후 저장")
    ap.add_argument("--embed-dir", type=Path, default=None,
                    help="임베딩 디렉토리 오버라이드 (기본: resource/embeddings)")
    args = ap.parse_args()

    embed_dir = args.embed_dir or EMBED_DIR
    tag = embed_dir.name if args.embed_dir else ""

    uuids, embeddings = load_embeddings(embed_dir)
    norm_emb = _normalize(embeddings)
    del embeddings  # float32 1M×1024 ≈ 4GB — 정규화 후 불필요

    sweep_results, best_k = sweep_k(norm_emb)
    save_sweep(sweep_results, tag=tag)

    labels, centers = cluster(norm_emb, best_k)

    medoid_uuids = extract_medoids(uuids, norm_emb, labels, centers)
    uuid_set     = {uid for uids in medoid_uuids.values() for uid in uids}
    texts        = _load_texts_for_uuids(uuid_set)
    save_medoids(medoid_uuids, texts)

    if args.apply_labels:
        apply_labels(uuids, labels)

    if not args.apply_labels:
        print("\n[done] 다음 단계:")
        print(f"  1. {OUT_DIR}/cluster_*_medoids.txt 확인")
        print(f"  2. {OUT_DIR}/cluster_to_archetype.json 아키타입 매핑 작성")
        print(f"  3. uv run src/pipeline.py --apply-labels")


if __name__ == "__main__":
    main()
