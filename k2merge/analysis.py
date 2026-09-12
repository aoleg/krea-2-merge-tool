"""Merge analysis: singular values, rank needed per energy target, per module group.

For sources that are all exact (A/B LoRAs) the spectrum comes from the small
rank x rank core via QR. Otherwise the full delta is materialized and its
singular values computed on the device.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .keys import GROUPS, group_of

ENERGY_TARGETS = (0.90, 0.95, 0.99, 0.995)


def spectrum_from_factors(downs: list[torch.Tensor], ups: list[torch.Tensor]) -> torch.Tensor:
    """Singular values of sum_i ups[i] @ downs[i] via the concatenated core."""
    A = torch.cat(downs, dim=0)      # [R, in]
    B = torch.cat(ups, dim=1)        # [out, R]
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    return torch.linalg.svdvals(Rb @ Ra.T)


def spectrum_full(delta: torch.Tensor) -> torch.Tensor:
    """Singular values of a materialized delta, descending.

    Through the Gram matrix of the smaller side: eigvalsh on a 6144 x 6144
    symmetric matrix is about ten times faster than svdvals on 16384 x 6144 and
    the energy curves need no more precision than that. Values below fp32
    noise of the largest eigenvalue are clamped to zero.
    """
    d = delta.to(torch.float32)
    if d.dim() != 2:
        d = d.reshape(d.shape[0], -1)
    if min(d.shape) <= 64:
        return torch.linalg.svdvals(d)
    g = d @ d.T if d.shape[0] <= d.shape[1] else d.T @ d
    ev = torch.linalg.eigvalsh(g.double() if g.shape[0] <= 4096 else g)
    ev = torch.flip(ev, dims=[0]).to(torch.float32)
    floor = ev[0].item() * 1e-7 if ev.numel() else 0.0
    return torch.sqrt(torch.clamp(ev, min=0.0).masked_fill(ev < floor, 0.0))


def rank_for_energy(s2_cum: torch.Tensor, total: float, target: float) -> int:
    """Smallest r with cumulative energy(r) / total >= target."""
    if total <= 0:
        return 0
    frac = s2_cum / total
    idx = int(torch.searchsorted(frac, torch.tensor(target, dtype=frac.dtype, device=frac.device)).item())
    return min(idx + 1, int(frac.numel()))


@dataclass
class ModuleSpectrum:
    canon: str
    name: str                  # bare module name
    group: str
    shape: tuple
    sv: torch.Tensor           # singular values, descending, on CPU
    exact: bool                # from low rank factors (no truncation happened)
    noise_energy: float = 0.0  # expected quantization noise energy in this delta (0 = none)

    @property
    def energy(self) -> float:
        return float((self.sv ** 2).sum().item())

    @property
    def max_rank(self) -> int:
        return int(min(self.shape))

    def rank_needed(self, target: float) -> int:
        s2 = self.sv ** 2
        return rank_for_energy(torch.cumsum(s2, 0), float(s2.sum().item()), target)

    def energy_at(self, r: int) -> float:
        s2 = self.sv ** 2
        tot = float(s2.sum().item())
        return float(s2[:r].sum().item() / tot) if tot > 0 else 1.0

    def signal_rank(self) -> int:
        """Directions whose energy sits above the per direction noise level."""
        if self.noise_energy <= 0:
            return int(self.sv.numel())
        per_dir = self.noise_energy / max(self.max_rank, 1)
        return int((self.sv ** 2 > per_dir).sum().item())

    def energy_above_floor(self) -> float:
        if self.noise_energy <= 0:
            return 1.0
        per_dir = self.noise_energy / max(self.max_rank, 1)
        s2 = self.sv ** 2
        above = torch.clamp(s2 - per_dir, min=0.0).sum().item()
        tot = float(s2.sum().item())
        return above / tot if tot > 0 else 0.0


@dataclass
class GroupReport:
    group: str
    modules: list = field(default_factory=list)   # ModuleSpectrum
    per_module_rank: dict = field(default_factory=dict)   # target -> max over modules of rank needed
    weighted_rank: dict = field(default_factory=dict)     # target -> rank meeting target on pooled energy
    worst: dict = field(default_factory=dict)             # target -> module name driving per_module_rank
    total_norm: float = 0.0
    noise_fraction: float = 0.0                           # 1 - energy above floor, pooled
    signal_rank_max: int = 0

    def compute(self):
        if not self.modules:
            return
        self.total_norm = sum(m.energy for m in self.modules) ** 0.5
        max_r = max(m.sv.numel() for m in self.modules)
        pooled = torch.zeros(max_r, dtype=torch.float64)
        for m in self.modules:
            s2 = (m.sv.double() ** 2)
            pooled[: s2.numel()] += torch.cumsum(s2, 0)
            if s2.numel() < max_r:
                pooled[s2.numel():] += s2.sum()
        total = float(sum(m.energy for m in self.modules))
        for t in ENERGY_TARGETS:
            ranks = [(m.rank_needed(t), m.name) for m in self.modules]
            r, name = max(ranks)
            self.per_module_rank[t] = r
            self.worst[t] = name
            self.weighted_rank[t] = rank_for_energy(pooled, total, t)
        above = sum(m.energy_above_floor() * m.energy for m in self.modules)
        self.noise_fraction = 1.0 - (above / total if total > 0 else 1.0)
        self.signal_rank_max = max(m.signal_rank() for m in self.modules)


@dataclass
class AnalysisReport:
    groups: dict = field(default_factory=dict)   # group -> GroupReport
    modules: dict = field(default_factory=dict)  # canon -> ModuleSpectrum
    clamped: list = field(default_factory=list)  # (name, max_rank) modules smaller than 2x requested
    notes: list = field(default_factory=list)

    def add(self, ms: ModuleSpectrum):
        self.modules[ms.canon] = ms
        self.groups.setdefault(ms.group, GroupReport(ms.group)).modules.append(ms)

    def finish(self, requested_rank: int | None = None):
        for g in self.groups.values():
            g.compute()
        if requested_rank:
            self.clamped = [(m.name, m.max_rank) for m in self.modules.values() if m.max_rank < 2 * requested_rank]

    def rank_plan(self, target: float, criterion: str = "per_module", exclude: set | None = None,
                  uniform: bool = False) -> dict:
        """group -> rank (or {'*': rank} when uniform), by criterion 'per_module' | 'weighted'."""
        exclude = exclude or set()
        chosen = {}
        for g, rep in self.groups.items():
            if g in exclude:
                continue
            chosen[g] = rep.per_module_rank[target] if criterion == "per_module" else rep.weighted_rank[target]
        if uniform:
            return {"*": max(chosen.values()) if chosen else 0}
        return chosen

    def text(self) -> str:
        lines = []
        for g in GROUPS:
            rep = self.groups.get(g)
            if rep is None:
                continue
            lines.append(f"[{g}] {len(rep.modules)} modules, delta norm {rep.total_norm:.4g}"
                         + (f", noise {rep.noise_fraction * 100:.1f}% of energy, signal rank <= {rep.signal_rank_max}"
                            if rep.noise_fraction > 0 else ""))
            lines.append("   target   per-module rank   weighted rank   driven by")
            for t in ENERGY_TARGETS:
                lines.append(f"   {t * 100:5.1f}%   {rep.per_module_rank[t]:15d}   {rep.weighted_rank[t]:13d}   {rep.worst[t]}")
        if self.clamped:
            lines.append(f"{len(self.clamped)} module(s) are smaller than twice the requested rank and clamp:")
            for name, r in self.clamped[:10]:
                lines.append(f"   max rank {r:4d}  {name}")
        for n in self.notes:
            lines.append(n)
        return "\n".join(lines)


def analyze_sources(sources: list, canon_list: list, names: dict, device, noise: dict | None = None,
                    requested_rank: int | None = None, progress=None) -> AnalysisReport:
    """sources: DeltaSource list. canon_list: modules to analyze. names: canon -> bare module name.
    noise: canon -> expected noise energy (quantized inputs)."""
    rep = AnalysisReport()
    noise = noise or {}
    total = len(canon_list)
    for i, c in enumerate(canon_list):
        if progress is not None:
            progress(i, total, names.get(c, c))
        contributing = [s for s in sources if c in s.modules()]
        if not contributing:
            continue
        exact = all(s.is_exact(c) for s in contributing)
        if exact:
            downs, ups = [], []
            for s in contributing:
                d, u = s.low_rank(c, device)
                downs.append(d)
                ups.append(u)
            sv = spectrum_from_factors(downs, ups)
            shape = (ups[0].shape[0], downs[0].shape[1])
        else:
            delta = None
            for s in contributing:
                d = s.delta(c, device)
                delta = d if delta is None else delta + d
            sv = spectrum_full(delta)
            shape = tuple(delta.shape)
            del delta
        name = names.get(c, c)
        rep.add(ModuleSpectrum(c, name, group_of(name), shape, sv.detach().float().cpu(), exact,
                               noise.get(c, 0.0)))
    rep.finish(requested_rank)
    return rep


# Expected relative weight error of the storage formats (per tensor Frobenius), used for the noise floor.
FORMAT_NOISE = {
    "fp8": 0.045, "fp8_scaled_legacy": 0.045, "fp8_scaled_meta": 0.045, "fp8_scaled_cq": 0.045,
    "int8_convrot": 0.011, "int8": 0.011, "plain": 0.0,
}


def quantization_noise_energy(weight_norm: float, layouts: list[str]) -> float:
    """Noise energy in a delta between tensors stored in the given layouts."""
    return sum((FORMAT_NOISE.get(l, 0.0) * weight_norm) ** 2 for l in layouts)
