"""Checkpoint merge methods as pure tensor functions (fp32, any device).

Formulas follow working_specs.md section 7.2 and 7.3. Cosine A/B, trainDifference
and extract reproduce sd-webui-supermerger's mergers.py.
"""
from __future__ import annotations

import hashlib
import math

import torch

METHODS = ("add_difference", "weighted_sum", "slerp", "cosine_a", "cosine_b",
           "ties", "dare", "train_difference", "extract")
ADVANCED = ("train_difference", "extract")
NEEDS_C = ("ties", "dare", "train_difference", "extract")
TWO_PASS = ("cosine_a", "cosine_b")

METHOD_LABELS = {
    "add_difference": "Add difference: A + w (B - C). w = how many times B's change is applied to A",
    "weighted_sum": "Weighted sum: (1 - w) A + w B. w = position between A (0) and B (1)",
    "slerp": "SLERP: spherical interpolation from A (0) to B (1) at t = w",
    "cosine_a": "Cosine A (SuperMerger): keep A where A and B agree, take B where they differ; w shifts toward B",
    "cosine_b": "Cosine B (SuperMerger): as Cosine A, computed on raw tensors with magnitude; w shifts toward B",
    "ties": "TIES: trim (A - C) and w (B - C) to their largest changes, elect signs, add lambda x result to C",
    "dare": "DARE: randomly drop p of each change, rescale the rest, add lambda x sum to C",
    "train_difference": "trainDifference (SuperMerger): A + 1.8 w s (B - C), damped where A already moved toward B",
    "extract": "Extract (SuperMerger): C + lerp(A - C, B - C, w) masked by row similarity (beta 0 = common, 1 = distinct)",
}


# ----------------------------------------------------------------------------- simple
def weighted_sum(a, b, w):
    return a * (1.0 - w) + b * w


def add_difference(a, b, c, w):
    return a + (b - c) * w


def slerp(a, b, t, eps=1e-3):
    """Spherical interpolation of the flattened tensors; falls back to LERP for tiny angles."""
    af, bf = a.reshape(-1), b.reshape(-1)
    na, nb = af.norm(), bf.norm()
    if na.item() == 0 or nb.item() == 0:
        return weighted_sum(a, b, t)
    cos = torch.clamp(torch.dot(af, bf) / (na * nb), -1.0, 1.0)
    omega = torch.arccos(cos)
    so = torch.sin(omega)
    if abs(omega.item()) < eps or so.item() < 1e-6:
        return weighted_sum(a, b, t)
    return (torch.sin((1.0 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


# ----------------------------------------------------------------------------- cosine (SuperMerger)
def _cosine_similarity_values(a, b, mode):
    """Per column (dim 0) similarity as SuperMerger computes it. Returns a 1-D tensor."""
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    if a32.dim() == 0:
        a32, b32 = a32.reshape(1), b32.reshape(1)
    sim = torch.nn.CosineSimilarity(dim=0)
    if mode == "cosine_a":
        an = torch.nn.functional.normalize(a32, p=2, dim=0)
        bn = torch.nn.functional.normalize(b32, p=2, dim=0)
        simab = sim(an, bn)
        dot = torch.dot(an.reshape(-1), bn.reshape(-1))
        mag = dot / (an.norm() * bn.norm())
    else:
        simab = sim(a32, b32)
        dot = torch.dot(a32.reshape(-1), b32.reshape(-1))
        mag = dot / (a32.norm() * b32.norm())
    combined = (simab + mag) / 2.0
    return combined.reshape(-1)


class CosineStats:
    """Stage 0 of the cosine modes: the trimmed min/max of all similarity values."""

    def __init__(self, mode: str):
        self.mode = mode
        self._chunks: list[torch.Tensor] = []
        self.min = 0.0
        self.max = 1.0

    def add(self, a, b):
        v = _cosine_similarity_values(a, b, self.mode)
        v = v[~torch.isnan(v)]
        if v.numel():
            self._chunks.append(v.detach().float().cpu())

    def finish(self):
        if not self._chunks:
            return self
        s = torch.cat(self._chunks).double()
        lo = torch.quantile(s, 0.01, interpolation="midpoint")
        hi = torch.quantile(s, 0.99, interpolation="midpoint")
        s = s[(s >= lo) & (s <= hi)]
        if s.numel() == 0:
            s = torch.cat(self._chunks).double()
        self.min, self.max = float(s.min()), float(s.max())
        self._chunks = []
        return self


def cosine_merge(a, b, w, stats: CosineStats):
    """Stage 1: k per column from similarity, shifted by w; output = B (1 - k) + A k."""
    v = _cosine_similarity_values(a, b, stats.mode)
    span = stats.max - stats.min
    k = (v - stats.min) / span if span > 0 else torch.zeros_like(v)
    k = k - (abs(w) if stats.mode == "cosine_a" else w)
    k = k.clamp(0.0, 1.0)
    if a.dim() >= 1 and k.numel() == a.shape[-1] and a.dim() >= 2:
        kk = k.reshape(*([1] * (a.dim() - 1)), -1)
    elif a.dim() == 1 and k.numel() == 1:
        kk = k.reshape(())
    else:
        kk = k.reshape(-1)[0] if k.numel() == 1 else k
    return b * (1.0 - kk) + a * kk


# ----------------------------------------------------------------------------- TIES / DARE
def trim_topk(tau: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the top density fraction of entries by magnitude, zero the rest."""
    if density >= 1.0:
        return tau
    n = tau.numel()
    k = max(1, int(round(n * density)))
    if k >= n:
        return tau
    flat = tau.abs().reshape(-1)
    thresh = torch.kthvalue(flat, n - k + 1).values
    return torch.where(tau.abs() >= thresh, tau, torch.zeros_like(tau))


def ties_merge(taus: list[torch.Tensor], density: float) -> torch.Tensor:
    """Trim, elect sign, disjoint mean of the agreeing values."""
    trimmed = [trim_topk(t, density) for t in taus]
    stacked = torch.stack(trimmed, 0)
    sign = torch.sign(stacked.sum(0))
    agree = (torch.sign(stacked) == sign.unsqueeze(0)) & (stacked != 0)
    num = (stacked * agree).sum(0)
    cnt = agree.sum(0).clamp(min=1)
    return num / cnt


def _seed_for(seed: int, key: str) -> int:
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFFFFFFFFFF


def dare_drop(tau: torch.Tensor, p: float, seed: int, key: str) -> torch.Tensor:
    """Drop each entry with probability p, rescale the survivors by 1 / (1 - p). Deterministic per key."""
    if p <= 0:
        return tau
    if p >= 1:
        return torch.zeros_like(tau)
    # the mask is drawn on the CPU so a recipe reproduces byte for byte on any device
    g = torch.Generator().manual_seed(_seed_for(seed, key))
    keep = (torch.rand(tau.shape, generator=g) >= p).to(tau.device)
    return tau * keep / (1.0 - p)


def dare_merge(taus: list[torch.Tensor], p: float, seed: int, key: str, ties_density: float | None = None) -> torch.Tensor:
    dropped = [dare_drop(t, p, seed, f"{key}#{i}") for i, t in enumerate(taus)]
    if ties_density is not None:
        return ties_merge(dropped, ties_density)
    return torch.stack(dropped, 0).sum(0)


# ----------------------------------------------------------------------------- SuperMerger extras
def train_difference(a, b, c, w):
    """SuperMerger traindiff: A + 1.8 w sign(B - C) |B - C| |B - A| / (|B - C| + |B - A|)."""
    diff_bc = b - c
    d0 = (b - c).abs()
    d1 = (b - a).abs()
    denom = d0 + d1
    scale = torch.where(denom != 0, d1 / denom, torch.zeros_like(denom))
    scale = torch.sign(diff_bc) * scale.abs()
    return a + scale * diff_bc.abs() * (w * 1.8)


def extract_super(base, a, b, alpha, beta, gamma):
    """SuperMerger extract_super with base = C: C + lerp(A - C, B - C, alpha) * lerp(d, 1 - d, beta)."""
    alpha = min(max(alpha, 0.0), 1.0)
    beta = min(max(beta, 0.0), 1.0)
    gamma = max(gamma, 0.0)
    da, db = a - base, b - base
    if da.dim() == 0:
        da, db = da.reshape(1), db.reshape(1)
    cos = torch.nn.functional.cosine_similarity(da, db, dim=-1).clamp(-1, 1).unsqueeze(-1)
    d = ((cos + 1) / 2) ** gamma
    res = torch.lerp(da, db, alpha) * torch.lerp(d, 1 - d, beta)
    return (base + res.reshape(base.shape)) if base.dim() else (base + res.reshape(()))
