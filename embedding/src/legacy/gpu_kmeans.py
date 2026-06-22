"""
gpu_kmeans.py — PyTorch GPU KMeans (sweep.py --algo gpu 에서 호출).

L2 정규화된 벡터에서 코사인 거리를 행렬곱으로 계산, GPU에서 assignment/update.
"""

import sys
import time
from typing import Iterator

import numpy as np
import torch

SEED     = 42
MAX_ITER = 300
TOL      = 1e-4


def env_check(device: str) -> None:
    if not torch.cuda.is_available():
        sys.exit("[error] CUDA 사용 불가")
    name = torch.cuda.get_device_name(device)
    print(f"[gpu] {name}  |  torch {torch.__version__}")


def _init_centers(x: torch.Tensor, k: int) -> torch.Tensor:
    """KMeans++ 초기화 (GPU)."""
    n = x.shape[0]
    rng = torch.Generator(device=x.device)
    rng.manual_seed(SEED)

    idx = torch.randint(n, (1,), generator=rng, device=x.device).item()
    centers = [x[idx]]

    for _ in range(1, k):
        c     = torch.stack(centers)                          # (m, d)
        sim   = x @ c.T                                       # (n, m)
        d2    = (1.0 - sim.max(dim=1).values).clamp(min=0)
        probs = d2 / d2.sum()
        idx   = torch.multinomial(probs, 1, generator=rng).item()
        centers.append(x[idx])

    return torch.stack(centers)


def _kmeans(x: torch.Tensor, k: int) -> tuple[torch.Tensor, float]:
    """반환: (labels, inertia). 최종 centroid 기준으로 labels 재할당."""
    centers = _init_centers(x, k)

    for _ in range(MAX_ITER):
        sim    = x @ centers.T
        labels = sim.argmax(dim=1)

        new_c  = torch.zeros_like(centers)
        counts = torch.zeros(k, device=x.device)
        new_c.scatter_add_(0, labels.unsqueeze(1).expand(-1, x.shape[1]), x)
        counts.scatter_add_(0, labels, torch.ones(x.shape[0], device=x.device))
        counts = counts.clamp(min=1)
        new_c  = new_c / counts.unsqueeze(1)

        norms = new_c.norm(dim=1, keepdim=True).clamp(min=1e-10)
        new_c = new_c / norms

        shift   = (new_c - centers).norm(dim=1).max().item()
        centers = new_c
        if shift < TOL:
            break

    # 최종 centroid 기준 재할당 + inertia 계산
    sim     = x @ centers.T
    labels  = sim.argmax(dim=1)
    max_sim = sim.gather(1, labels.unsqueeze(1)).squeeze(1)
    inertia = (2.0 - 2.0 * max_sim).sum().item()  # ||x-c||^2 on unit vectors
    return labels, inertia


def labels_iter(
    x_np: np.ndarray, k_range: range, device: str
) -> Iterator[tuple[int, np.ndarray, float, float]]:
    """k_range 순회 → (k, labels_np, elapsed_s, inertia) yield. 입력은 1회 업로드."""
    env_check(device)
    x = torch.from_numpy(x_np).to(device)
    try:
        for k in k_range:
            t0 = time.perf_counter()
            labels_gpu, inertia = _kmeans(x, k)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            labels = labels_gpu.cpu().numpy()
            del labels_gpu
            torch.cuda.empty_cache()
            yield k, labels, dt, inertia
    finally:
        del x
        torch.cuda.empty_cache()


def predict_once(x_np: np.ndarray, k: int, device: str) -> np.ndarray:
    """1회성 클러스터링 (bootstrap stability 용). 매 호출마다 upload/cleanup."""
    x = torch.from_numpy(x_np).to(device)
    try:
        labels_gpu, _ = _kmeans(x, k)
        torch.cuda.synchronize()
        return labels_gpu.cpu().numpy()
    finally:
        del x
        torch.cuda.empty_cache()
