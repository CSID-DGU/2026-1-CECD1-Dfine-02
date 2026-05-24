"""
validate_signals.py — 사전 노이즈 필터 신호 검증

신호 4종의 분포·상관·이중봉우리를 확인하여 ε threshold 선정 근거 확보.

신호 정의 (분류모델_재정의_2026-05-20.md §2.2):
  text_len        : 5칼럼 char 합 (HF 데이터셋 필요, --load-text)
  distinctiveness : 칼럼별 코퍼스 mean 벡터로부터의 L2 거리 평균
  emb_norm        : 5칼럼 임베딩 평균 벡터의 L2 norm (∈ [1/√5, 1.0])
                    ※ 설계상 "정규화 전" raw norm 의도이나, 현 파이프라인은
                      L2 정규화된 임베딩을 저장하므로 mean norm을 proxy로 사용.
                      LOW = 칼럼 방향 분산 (부분 상쇄), HIGH = 칼럼 방향 일치.
  col_consistency : 5칼럼 임베딩 간 차원별 분산의 평균 (HIGH = 칼럼 간 불일치 = noise)

합성 판정: 4개 중 ≥3개 조건 충족 시 noise_struct = True
  text_len / distinctiveness / emb_norm : 하위 ε% → 조건 충족
  col_consistency                        : 상위 ε% → 조건 충족

Usage:
    uv run src/validate_signals.py
    uv run src/validate_signals.py --load-text
    uv run src/validate_signals.py --sample 20000 --eps 0.10
    uv run src/validate_signals.py --no-emb-norm  # emb_norm 제외 3신호 검증
"""

import argparse
import gc
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
SEED    = 42

COLS = [
    "career_goals_and_ambitions",
    "professional_persona",
    "family_persona",
    "travel_persona",
    "hobbies_and_interests",
]
N_COLS    = len(COLS)
COL_DIM   = 1024
TOTAL_DIM = N_COLS * COL_DIM  # 5120

# noise 방향: True = 하위 ε (낮을수록 noise), False = 상위 ε (높을수록 noise)
SIGNAL_LOW = {
    "text_len":        True,
    "distinctiveness": True,
    "emb_norm":        True,
    "col_consistency": False,
}


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_embeddings(embed_dir: Path, sample: int) -> tuple[list[str], np.ndarray]:
    """percol5 parquet → (uuids, float32 array (N, 5120))."""
    table    = pa_ds.dataset(embed_dir, format="parquet").to_table(columns=["uuid", "embedding"])
    n_total  = len(table)
    uuids_all = table.column("uuid").to_pylist()
    flat     = table.column("embedding").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    del table; gc.collect()

    arr = flat.reshape(n_total, TOTAL_DIM).astype(np.float32)
    del flat; gc.collect()

    if sample < n_total:
        rng   = np.random.default_rng(SEED)
        idx   = np.sort(rng.choice(n_total, size=sample, replace=False))
        arr   = arr[idx].copy()
        uuids = [uuids_all[i] for i in idx]
        del uuids_all; gc.collect()
    else:
        uuids = uuids_all

    print(f"[load] {len(uuids):,} rows  dim={TOTAL_DIM}")
    return uuids, arr


def load_text_len(uuids: list[str], cfg_path: Path) -> np.ndarray:
    """HuggingFace 데이터셋 → 5칼럼 char 합. 매칭 못 찾으면 0."""
    import tomllib
    from datasets import load_dataset

    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    name  = cfg["dataset"]["name"]
    cache = cfg["dataset"]["cache_dir"] or None

    print(f"[text] 데이터셋 로드: {name}")
    uuid_set  = set(uuids)
    text_map: dict[str, int] = {}
    ds = load_dataset(name, split="train", cache_dir=cache, streaming=True)

    for row in ds:
        u = row["uuid"]
        if u in uuid_set:
            text_map[u] = sum(len(row.get(c) or "") for c in COLS)
        if len(text_map) >= len(uuids):
            break

    matched = sum(1 for u in uuids if u in text_map)
    print(f"[text] 매칭 {matched:,} / {len(uuids):,}")
    return np.array([text_map.get(u, 0) for u in uuids], dtype=np.float32)


# ── 신호 계산 ──────────────────────────────────────────────────────────────────

def compute_signals(emb: np.ndarray, text_len: np.ndarray | None,
                    include_emb_norm: bool) -> pd.DataFrame:
    """신호 행렬 계산. emb: (N, 5120) float32"""
    N = len(emb)
    E = emb.reshape(N, N_COLS, COL_DIM)  # (N, 5, 1024)

    # distinctiveness: 칼럼별 코퍼스 mean 거리 평균
    col_means       = E.mean(axis=0)                         # (5, 1024)
    diffs           = E - col_means[np.newaxis, :, :]        # (N, 5, 1024)
    col_dists       = np.linalg.norm(diffs, axis=2)          # (N, 5)
    distinctiveness = col_dists.mean(axis=1)                 # (N,)
    del diffs, col_dists; gc.collect()

    # emb_norm: 5칼럼 평균 벡터의 L2 norm ∈ [1/√5, 1]
    emb_norm = None
    if include_emb_norm:
        mean_vec = E.mean(axis=1)                            # (N, 1024)
        emb_norm = np.linalg.norm(mean_vec, axis=1)         # (N,)
        del mean_vec; gc.collect()

    # col_consistency: 차원별 분산의 평균 (높을수록 칼럼 간 불일치)
    col_var         = E.var(axis=1)                          # (N, 1024)
    col_consistency = col_var.mean(axis=1)                   # (N,)
    del col_var; gc.collect()

    cols = {}
    if text_len is not None:
        cols["text_len"] = text_len
    cols["distinctiveness"]  = distinctiveness
    if emb_norm is not None:
        cols["emb_norm"]     = emb_norm
    cols["col_consistency"]  = col_consistency

    return pd.DataFrame(cols)


# ── noise 판정 ─────────────────────────────────────────────────────────────────

def apply_noise(df: pd.DataFrame, signals: list[str], eps: float) -> np.ndarray:
    """≥(n_sig-1) 합의 규칙으로 noise_struct 마스크 반환."""
    flags = np.zeros(len(df), dtype=int)
    for sig in signals:
        vals = df[sig].values
        if SIGNAL_LOW[sig]:               # 하위 ε → noise
            thr = np.quantile(vals, eps)
            flags += (vals <= thr).astype(int)
        else:                             # 상위 ε → noise
            thr = np.quantile(vals, 1.0 - eps)
            flags += (vals >= thr).astype(int)
    return flags >= (len(signals) - 1)


def noise_rate_table(df: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    eps_list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    rows = []
    for eps in eps_list:
        mask = apply_noise(df, signals, eps)
        rows.append({
            "eps":        eps,
            "noise_rate": f"{mask.mean()*100:.1f}%",
            "n_noise":    int(mask.sum()),
            "rule":       f">={len(signals)-1}/{len(signals)}",
        })
    return pd.DataFrame(rows)


# ── 시각화 ─────────────────────────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame, signals: list[str], eps: float, out_dir: Path) -> None:
    n = len(signals)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, sig in zip(axes, signals):
        vals = df[sig].values
        ax.hist(vals, bins=80, color="steelblue", alpha=0.8, edgecolor="none")
        if SIGNAL_LOW[sig]:
            thr = np.quantile(vals, eps)
            label = f"ε={eps} lo={thr:.3f}"
        else:
            thr = np.quantile(vals, 1.0 - eps)
            label = f"ε={eps} hi={thr:.3f}"
        ax.axvline(thr, color="crimson", linewidth=1.5, label=label)
        ax.set_title(sig, fontsize=10)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    plt.suptitle(f"Signal Distributions  (n={len(df):,})", fontsize=11)
    plt.tight_layout()
    path = out_dir / "signal_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")


def plot_correlation(df: pd.DataFrame, signals: list[str], out_dir: Path) -> None:
    n = len(signals)
    corr = df[signals].corr(method="spearman")  # 비정규 분포에 Spearman

    fig, ax = plt.subplots(figsize=(n * 1.5 + 1, n * 1.5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_xticklabels(signals, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(signals, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(corr.values[i,j]) > 0.6 else "black")
    ax.set_title("Spearman Correlation (신호 독립성)", fontsize=11)
    plt.tight_layout()
    path = out_dir / "signal_correlation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")


def plot_noise_curve(df: pd.DataFrame, signals: list[str], out_dir: Path) -> None:
    eps_list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    rates    = [apply_noise(df, signals, e).mean() * 100 for e in eps_list]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps_list, rates, marker="o", color="steelblue")
    ax.axhline(10, color="orange", linestyle="--", linewidth=1, label="10% (목표)")
    ax.axhline(20, color="crimson", linestyle="--", linewidth=1, label="20% (상한)")
    ax.set_xlabel("ε (per-signal percentile threshold)")
    ax.set_ylabel("noise_struct rate (%)")
    ax.set_title(f"Noise Rate vs ε  (≥{len(signals)-1}/{len(signals)} 합의)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "noise_rate_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")


def plot_scatter_matrix(df: pd.DataFrame, signals: list[str], eps: float,
                        out_dir: Path) -> None:
    """신호 쌍별 scatter (noise 점 강조)."""
    noise_mask = apply_noise(df, signals, eps)
    n = len(signals)
    if n < 2:
        return

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            xi = df[signals[j]].values
            yi = df[signals[i]].values
            if i == j:
                ax.hist(xi[~noise_mask], bins=50, color="steelblue",
                        alpha=0.7, label="clean")
                ax.hist(xi[noise_mask],  bins=50, color="crimson",
                        alpha=0.7, label="noise")
                ax.set_title(signals[i], fontsize=8)
                if i == 0:
                    ax.legend(fontsize=7)
            else:
                sample_idx = np.random.default_rng(SEED).choice(
                    len(df), size=min(5000, len(df)), replace=False)
                ax.scatter(xi[sample_idx][~noise_mask[sample_idx]],
                           yi[sample_idx][~noise_mask[sample_idx]],
                           s=2, alpha=0.3, color="steelblue")
                ax.scatter(xi[sample_idx][noise_mask[sample_idx]],
                           yi[sample_idx][noise_mask[sample_idx]],
                           s=6, alpha=0.6, color="crimson")
            if i == n - 1:
                ax.set_xlabel(signals[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(signals[i], fontsize=8)
    plt.suptitle(f"Signal Scatter Matrix  ε={eps}", fontsize=11)
    plt.tight_layout()
    path = out_dir / "signal_scatter_matrix.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] {path}")


# ── 텍스트 정성 확인 ───────────────────────────────────────────────────────────

def print_qualitative(uuids: list[str], noise_mask: np.ndarray,
                      cfg_path: Path, n_show: int = 5) -> None:
    import tomllib
    from datasets import load_dataset

    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    name  = cfg["dataset"]["name"]
    cache = cfg["dataset"]["cache_dir"] or None

    rng        = np.random.default_rng(SEED)
    noise_uids = set(np.array(uuids)[noise_mask][
        rng.choice(noise_mask.sum(), size=min(n_show, noise_mask.sum()), replace=False)])
    clean_uids = set(np.array(uuids)[~noise_mask][
        rng.choice((~noise_mask).sum(), size=min(n_show, (~noise_mask).sum()), replace=False)])
    target     = noise_uids | clean_uids

    collected: dict[str, dict] = {}
    ds = load_dataset(name, split="train", cache_dir=cache, streaming=True)
    for row in ds:
        if row["uuid"] in target:
            collected[row["uuid"]] = {c: row.get(c, "") for c in COLS}
        if len(collected) >= len(target):
            break

    def _show(uid: str, label: str) -> None:
        print(f"\n  [{label}] {uid[:8]}...")
        for c in COLS:
            txt = (collected.get(uid, {}).get(c) or "")
            print(f"    {c[:20]:20s}: {txt!r}")

    print("\n" + "=" * 60)
    print("▶ noise_struct = True 예시")
    for uid in list(noise_uids)[:n_show]:
        if uid in collected:
            _show(uid, "noise")

    print("\n▶ noise_struct = False 예시")
    for uid in list(clean_uids)[:n_show]:
        if uid in collected:
            _show(uid, "clean")
    print("=" * 60)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=Path,
                    default=ROOT / "resource" / "embeddings_percol5")
    ap.add_argument("--sample",    type=int, default=50_000,
                    help="검증용 샘플 수 (기본 50k)")
    ap.add_argument("--eps",       type=float, default=0.10,
                    help="ε threshold 기본값 (그래프·noise_struct 계산용)")
    ap.add_argument("--load-text", action="store_true",
                    help="HuggingFace에서 text_len 로드 (느림)")
    ap.add_argument("--no-emb-norm", action="store_true",
                    help="emb_norm 신호 제외 (3신호 검증)")
    ap.add_argument("--config",    type=Path, default=ROOT / "config.toml")
    args = ap.parse_args()

    uuids, emb = load_embeddings(args.embed_dir, args.sample)

    text_len = None
    if args.load_text:
        text_len = load_text_len(uuids, args.config)

    df = compute_signals(emb, text_len, include_emb_norm=not args.no_emb_norm)
    del emb; gc.collect()

    signals = [s for s in ["text_len", "distinctiveness", "emb_norm", "col_consistency"]
               if s in df.columns]

    print(f"\n[signals] {signals}")
    print(df[signals].describe().round(4).to_string())

    # noise rate 표
    rates = noise_rate_table(df, signals)
    print(f"\n[noise rate vs ε]  (≥{len(signals)-1}/{len(signals)} 합의)")
    print(rates.to_string(index=False))

    # 그래프 생성
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_distributions(df, signals, args.eps, OUT_DIR)
    plot_correlation(df, signals, OUT_DIR)
    plot_noise_curve(df, signals, OUT_DIR)
    plot_scatter_matrix(df, signals, args.eps, OUT_DIR)

    # noise_struct 열 추가
    noise_mask = apply_noise(df, signals, args.eps)
    df["noise_struct"] = noise_mask
    print(f"\n[ε={args.eps}] noise_struct: {noise_mask.sum():,} / {len(df):,}"
          f"  ({noise_mask.mean()*100:.1f}%)")

    # 결과 저장
    df["uuid"] = uuids
    out_path = OUT_DIR / f"signal_matrix_{args.sample}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[save] {out_path}")

    # 정성 확인 (--load-text 시)
    if args.load_text:
        print_qualitative(uuids, noise_mask, args.config)


if __name__ == "__main__":
    main()
