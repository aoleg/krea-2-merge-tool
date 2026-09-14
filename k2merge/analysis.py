"""Merge analysis: singular value spectra, rank needed per energy target, noise model, per module group.

For sources that are all exact (A/B LoRAs) the spectrum comes from the small
rank x rank core via QR. Otherwise the full delta is materialized and its
singular values computed on the device through the Gram matrix.

Noise model (one per element variance per tensor, from the storage layout of
each input): bf16 or fp16 rounding for plain files, the measured relative
error for fp8 and int8 layouts, zero for fp32. The variance gives both the
total noise energy and the Marchenko-Pastur edge, the largest singular value
of a matrix of that noise alone. Directions below the edge cannot be told
from noise. Elements whose delta is exactly zero carry no rounding noise
(both files hold the same value), which scales the variance down.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import torch

from . import TOOL_NAME, __version__
from .keys import GROUPS, block_index, group_of

ENERGY_TARGETS = (0.90, 0.95, 0.99, 0.995)                  # the group report
ENERGY_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.995)      # per tensor rank_at_energy
CANDIDATE_RANKS = (8, 16, 32, 64, 128, 256, 512, 1024)
GRAM_FLOOR = 1e-7                                            # eigenvalue clamp in spectrum_full, relative to the largest
SPECTRUM_SUFFIX = ".spectrum"

# Expected relative weight error of the quantized storage formats (per tensor Frobenius), used for the noise model.
FORMAT_NOISE = {
    "fp8": 0.045, "fp8_scaled_legacy": 0.045, "fp8_scaled_meta": 0.045, "fp8_scaled_cq": 0.045,
    "int8_convrot": 0.011, "int8": 0.011, "plain": 0.0,
}
MANTISSA_BITS = {"BF16": 7, "F16": 10}


# ----------------------------------------------------------------------------- spectra
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
    noise of the largest eigenvalue are clamped to zero. A delta with no
    energy returns an empty tensor without a decomposition.
    """
    d = delta.to(torch.float32)
    if d.dim() != 2:
        d = d.reshape(d.shape[0], -1)
    if not bool((d != 0).any()):
        return torch.zeros(0, dtype=torch.float32, device=d.device)
    if min(d.shape) <= 64:
        return torch.linalg.svdvals(d)
    g = d @ d.T if d.shape[0] <= d.shape[1] else d.T @ d
    ev = torch.linalg.eigvalsh(g.double() if g.shape[0] <= 4096 else g)
    ev = torch.flip(ev, dims=[0]).to(torch.float32)
    floor = ev[0].item() * GRAM_FLOOR if ev.numel() else 0.0
    return torch.sqrt(torch.clamp(ev, min=0.0).masked_fill(ev < floor, 0.0))


def gram_resolution(top: float) -> float:
    """Smallest singular value spectrum_full can resolve next to the largest one."""
    return math.sqrt(GRAM_FLOOR) * top


def rank_for_energy(s2_cum: torch.Tensor, total: float, target: float) -> int:
    """Smallest r with cumulative energy(r) / total >= target."""
    if total <= 0 or s2_cum.numel() == 0:
        return 0
    frac = s2_cum / total
    idx = int(torch.searchsorted(frac, torch.tensor(target, dtype=frac.dtype, device=frac.device)).item())
    return min(idx + 1, int(frac.numel()))


# ----------------------------------------------------------------------------- noise model
def rounding_variance(w: torch.Tensor, dtype_tag: str) -> float:
    """Per element variance of the rounding error of w stored as bf16 or fp16: mean(ulp(w) ** 2) / 12.
    ulp(w) = 2 ** (floor(log2 |w|) - mantissa bits). Zero weights round exactly and add nothing."""
    bits = MANTISSA_BITS.get(dtype_tag)
    if bits is None or w.numel() == 0:
        return 0.0
    a = w.abs().reshape(-1)
    nz = a[a > 0]
    if nz.numel() == 0:
        return 0.0
    e = torch.floor(torch.log2(nz.to(torch.float32)))
    ulp2 = torch.exp2(2.0 * (e - bits)).double()
    return float(ulp2.sum().item() / w.numel() / 12.0)


def layout_noise_variance(layout: str, dtype_tag: str, w: torch.Tensor, fro: float | None = None) -> float:
    """Per element noise variance one input contributes to a delta, from its storage layout.
    Plain files: rounding of the stored dtype. Quantized layouts: the measured relative error spread over the elements."""
    if layout == "plain":
        return rounding_variance(w, dtype_tag)
    rel = FORMAT_NOISE.get(layout, 0.0)
    if rel <= 0 or w.numel() == 0:
        return 0.0
    fro = float(w.norm().item()) if fro is None else fro
    return (rel * fro) ** 2 / w.numel()


def quantization_noise_energy(weight_norm: float, layouts: list[str]) -> float:
    """Noise energy in a delta between tensors stored in the given quantized layouts (kept for older callers)."""
    return sum((FORMAT_NOISE.get(l, 0.0) * weight_norm) ** 2 for l in layouts)


def noise_spectrum(shape, variance: float, device, seed: int = 0) -> torch.Tensor:
    """Singular values of an iid uniform noise matrix of the given per element variance."""
    g = torch.Generator(device=device).manual_seed(seed)
    a = math.sqrt(3.0 * variance)
    n = (torch.rand(tuple(shape), generator=g, device=device, dtype=torch.float32) * 2.0 - 1.0) * a
    return spectrum_full(n)


# ----------------------------------------------------------------------------- per tensor
@dataclass
class ModuleSpectrum:
    canon: str
    name: str                  # bare module name
    group: str
    shape: tuple
    sv: torch.Tensor           # singular values, descending, on CPU; empty when all_zero
    exact: bool                # from low rank factors (no truncation happened)
    noise_var: float = 0.0     # per element noise variance after the zero fraction correction (0 = no model)
    base_fro: float | None = None
    zero_fraction: float = 0.0
    layouts: tuple = ()        # (base, target) storage layouts, () for LoRA sources
    dtypes: tuple = ()         # (base, target) stored dtype tags
    resolved_by_svd: bool = False   # the Gram path could not resolve the noise edge; svdvals was used
    sigma_null: torch.Tensor | None = None

    # ---- basics
    @property
    def block(self) -> int | None:
        return block_index(self.name)

    @property
    def all_zero(self) -> bool:
        return self.sv.numel() == 0

    @property
    def energy(self) -> float:
        return float((self.sv.double() ** 2).sum().item()) if self.sv.numel() else 0.0

    @property
    def delta_fro(self) -> float:
        return math.sqrt(self.energy)

    @property
    def rel_change(self) -> float | None:
        if self.base_fro is None or self.base_fro <= 0:
            return None
        return self.delta_fro / self.base_fro

    @property
    def max_rank(self) -> int:
        return int(min(self.shape))

    # ---- noise
    @property
    def noise_edge(self) -> float:
        if self.noise_var <= 0:
            return 0.0
        m, n = self.shape[0], self.shape[1]
        return math.sqrt(self.noise_var) * (math.sqrt(m) + math.sqrt(n))

    @property
    def noise_energy(self) -> float:
        return self.noise_var * self.shape[0] * self.shape[1] if self.noise_var > 0 else 0.0

    def denoised_sv(self) -> torch.Tensor:
        e = self.noise_edge
        return self.sv[self.sv > e] if e > 0 else self.sv

    @property
    def n_above_noise(self) -> int:
        return int(self.denoised_sv().numel())

    @property
    def energy_above_noise(self) -> float:
        tot = self.energy
        if tot <= 0:
            return 1.0
        return float((self.denoised_sv().double() ** 2).sum().item()) / tot

    # ---- ranks
    def rank_needed(self, target: float, denoised: bool = False) -> int:
        sv = self.denoised_sv() if denoised else self.sv
        s2 = sv.double() ** 2
        return rank_for_energy(torch.cumsum(s2, 0), float(s2.sum().item()), target)

    def energy_at(self, r: int, denoised: bool = False) -> float:
        sv = self.denoised_sv() if denoised else self.sv
        s2 = sv.double() ** 2
        tot = float(s2.sum().item())
        return float(s2[:r].sum().item() / tot) if tot > 0 else 1.0

    def rank_at_energy(self, denoised: bool = False) -> dict:
        return {t: self.rank_needed(t, denoised) for t in ENERGY_THRESHOLDS}

    @property
    def effective_rank(self) -> float:
        s2 = self.sv.double() ** 2
        tot = float(s2.sum().item())
        if tot <= 0:
            return 0.0
        p = s2[s2 > 0] / tot
        return float(torch.exp(-(p * torch.log(p)).sum()).item())

    @property
    def stable_rank(self) -> float:
        if self.sv.numel() == 0 or float(self.sv[0]) <= 0:
            return 0.0
        return self.energy / float(self.sv[0].double() ** 2)

    # kept for older callers
    def signal_rank(self) -> int:
        return self.n_above_noise if self.noise_var > 0 else int(self.sv.numel())

    def energy_above_floor(self) -> float:
        return self.energy_above_noise

    # ---- persistence
    def to_dict(self) -> dict:
        d = {"name": self.name, "canon": self.canon, "group": self.group, "block": self.block, "shape": list(self.shape),
             "exact": self.exact, "all_zero": self.all_zero, "layouts": list(self.layouts), "dtypes": list(self.dtypes),
             "delta_fro": self.delta_fro, "base_fro": self.base_fro, "rel_change": self.rel_change,
             "zero_fraction": self.zero_fraction, "noise_var": self.noise_var, "noise_edge": self.noise_edge,
             "n_above_noise": self.n_above_noise, "energy_above_noise": self.energy_above_noise,
             "effective_rank": self.effective_rank, "stable_rank": self.stable_rank,
             "rank_at_energy": {str(t): r for t, r in self.rank_at_energy().items()},
             "rank_at_energy_denoised": {str(t): r for t, r in self.rank_at_energy(True).items()},
             "resolved_by_svd": self.resolved_by_svd, "has_null": self.sigma_null is not None}
        return d

    @classmethod
    def from_dict(cls, d: dict, sv: torch.Tensor, sigma_null: torch.Tensor | None = None) -> "ModuleSpectrum":
        return cls(d["canon"], d["name"], d["group"], tuple(d["shape"]), sv, bool(d.get("exact", False)),
                   float(d.get("noise_var", 0.0)), d.get("base_fro"), float(d.get("zero_fraction", 0.0)),
                   tuple(d.get("layouts", ())), tuple(d.get("dtypes", ())), bool(d.get("resolved_by_svd", False)), sigma_null)


@dataclass
class OtherTensor:
    """A tensor the extractor never targets: norms, modulation vectors, biases, filtered out weights."""
    name: str                  # bare key without the wrapper prefix
    shape: tuple
    group: str
    kind: str                  # "1-D" | "not a weight" | "filtered"
    base_fro: float
    delta_fro: float

    @property
    def block(self) -> int | None:
        return block_index(self.name)

    @property
    def all_zero(self) -> bool:
        return self.delta_fro <= 0.0

    @property
    def rel_change(self) -> float | None:
        return self.delta_fro / self.base_fro if self.base_fro > 0 else None

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.shape), "group": self.group, "block": self.block, "kind": self.kind,
                "base_fro": self.base_fro, "delta_fro": self.delta_fro, "rel_change": self.rel_change, "all_zero": self.all_zero}

    @classmethod
    def from_dict(cls, d: dict) -> "OtherTensor":
        return cls(d["name"], tuple(d["shape"]), d["group"], d["kind"], float(d["base_fro"]), float(d["delta_fro"]))


def _median(xs: list) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ----------------------------------------------------------------------------- per group
@dataclass
class GroupReport:
    group: str
    modules: list = field(default_factory=list)   # ModuleSpectrum
    per_module_rank: dict = field(default_factory=dict)   # target -> max over modules of rank needed
    weighted_rank: dict = field(default_factory=dict)     # target -> rank meeting target on pooled energy
    worst: dict = field(default_factory=dict)             # target -> module name driving per_module_rank
    per_module_rank_dn: dict = field(default_factory=dict)   # same three on the denoised spectra
    weighted_rank_dn: dict = field(default_factory=dict)
    worst_dn: dict = field(default_factory=dict)
    total_norm: float = 0.0
    noise_fraction: float = 0.0                           # 1 - energy above the noise edge, pooled
    signal_rank_max: int = 0
    median_effective_rank: float = 0.0
    median_stable_rank: float = 0.0
    median_rel_change: float | None = None

    @property
    def has_noise_model(self) -> bool:
        return any(m.noise_var > 0 for m in self.modules)

    @property
    def energy_above_noise(self) -> float:
        return 1.0 - self.noise_fraction

    def _pooled(self, denoised: bool):
        svs = [m.denoised_sv() if denoised else m.sv for m in self.modules]
        max_r = max((s.numel() for s in svs), default=0)
        pooled = torch.zeros(max_r, dtype=torch.float64)
        for s in svs:
            s2 = s.double() ** 2
            pooled[: s2.numel()] += torch.cumsum(s2, 0)
            if s2.numel() < max_r:
                pooled[s2.numel():] += s2.sum()
        return pooled, float(sum((s.double() ** 2).sum().item() for s in svs))

    def compute(self):
        if not self.modules:
            return
        total = float(sum(m.energy for m in self.modules))
        self.total_norm = math.sqrt(total)
        for denoised, (pm, wr, wo) in ((False, (self.per_module_rank, self.weighted_rank, self.worst)),
                                       (True, (self.per_module_rank_dn, self.weighted_rank_dn, self.worst_dn))):
            pooled, ptotal = self._pooled(denoised)
            for t in ENERGY_TARGETS:
                r, name = max((m.rank_needed(t, denoised), m.name) for m in self.modules)
                pm[t] = r
                wo[t] = name
                wr[t] = rank_for_energy(pooled, ptotal, t)
        above = sum(m.energy_above_noise * m.energy for m in self.modules)
        self.noise_fraction = 1.0 - (above / total if total > 0 else 1.0)
        self.signal_rank_max = max(m.n_above_noise for m in self.modules)
        self.median_effective_rank = _median([m.effective_rank for m in self.modules]) or 0.0
        self.median_stable_rank = _median([m.stable_rank for m in self.modules]) or 0.0
        self.median_rel_change = _median([m.rel_change for m in self.modules])

    def to_dict(self) -> dict:
        return {"group": self.group, "modules": len(self.modules), "total_norm": self.total_norm,
                "noise_fraction": self.noise_fraction, "signal_rank_max": self.signal_rank_max,
                "median_effective_rank": self.median_effective_rank, "median_stable_rank": self.median_stable_rank,
                "median_rel_change": self.median_rel_change,
                "per_module_rank": {str(t): r for t, r in self.per_module_rank.items()},
                "weighted_rank": {str(t): r for t, r in self.weighted_rank.items()},
                "worst": {str(t): n for t, n in self.worst.items()},
                "per_module_rank_denoised": {str(t): r for t, r in self.per_module_rank_dn.items()},
                "weighted_rank_denoised": {str(t): r for t, r in self.weighted_rank_dn.items()},
                "worst_denoised": {str(t): n for t, n in self.worst_dn.items()}}


# ----------------------------------------------------------------------------- the report
@dataclass
class AnalysisReport:
    groups: dict = field(default_factory=dict)   # group -> GroupReport
    modules: dict = field(default_factory=dict)  # canon -> ModuleSpectrum
    others: list = field(default_factory=list)   # OtherTensor
    clamped: list = field(default_factory=list)  # (name, max_rank) modules smaller than 2x requested
    notes: list = field(default_factory=list)
    run: dict = field(default_factory=dict)      # function, inputs, options, device, noise model, timestamp
    noise_model: dict = field(default_factory=dict)   # {"applied": bool, "base": [layout, dtype], "target": [...], "text": str}

    def add(self, ms: ModuleSpectrum):
        self.modules[ms.canon] = ms
        self.groups.setdefault(ms.group, GroupReport(ms.group)).modules.append(ms)

    def finish(self, requested_rank: int | None = None):
        for g in self.groups.values():
            g.compute()
        if requested_rank:
            self.clamped = [(m.name, m.max_rank) for m in self.modules.values() if m.max_rank < 2 * requested_rank]

    # ---- summaries
    @property
    def has_noise_model(self) -> bool:
        return any(m.noise_var > 0 for m in self.modules.values())

    @property
    def target_energy(self) -> float:
        return float(sum(m.energy for m in self.modules.values()))

    @property
    def outside_energy(self) -> float:
        return float(sum(o.delta_fro ** 2 for o in self.others))

    @property
    def outside_fraction(self) -> float:
        tot = self.target_energy + self.outside_energy
        return self.outside_energy / tot if tot > 0 else 0.0

    @property
    def label(self) -> str:
        r = self.run
        if r.get("function") == "extract":
            stem = lambda p: os.path.splitext(os.path.basename(p or ""))[0]  # noqa: E731
            return f"{stem(r.get('target'))} - {stem(r.get('base'))}"
        if r.get("function") == "lora_merge":
            return " + ".join(os.path.splitext(os.path.basename(i.get("file", "")))[0] for i in r.get("inputs", []))
        return r.get("label", "analysis")

    def candidate_table(self, ranks=CANDIDATE_RANKS) -> list:
        """[(rank, median raw, min raw, median denoised, min denoised)] over the 2-D target tensors."""
        rows = []
        mods = [m for m in self.modules.values() if not m.all_zero]
        for r in ranks:
            raw = [m.energy_at(r) for m in mods]
            dn = [m.energy_at(r, True) for m in mods]
            rows.append((r, _median(raw) if raw else None, min(raw) if raw else None,
                         _median(dn) if dn else None, min(dn) if dn else None))
        return rows

    def rank_plan(self, target: float, criterion: str = "per_module", exclude: set | None = None,
                  uniform: bool = False, denoised: bool = False) -> dict:
        """group -> rank (or {'*': rank} when uniform), by criterion 'per_module' | 'weighted'.
        denoised uses the spectra above the noise edge (no effect without a noise model)."""
        exclude = exclude or set()
        chosen = {}
        for g, rep in self.groups.items():
            if g in exclude:
                continue
            if denoised:
                chosen[g] = rep.per_module_rank_dn[target] if criterion == "per_module" else rep.weighted_rank_dn[target]
            else:
                chosen[g] = rep.per_module_rank[target] if criterion == "per_module" else rep.weighted_rank[target]
        if uniform:
            return {"*": max(chosen.values()) if chosen else 0}
        return chosen

    def text(self) -> str:
        lines = []
        noisy = self.has_noise_model
        for g in GROUPS:
            rep = self.groups.get(g)
            if rep is None:
                continue
            head = f"[{g}] {len(rep.modules)} modules, delta norm {rep.total_norm:.4g}"
            if rep.median_rel_change is not None:
                head += f", median change {rep.median_rel_change * 100:.2f}% of the weight"
            head += f", median effective rank {rep.median_effective_rank:.1f}, stable rank {rep.median_stable_rank:.2f}"
            if rep.has_noise_model:
                head += f", noise {rep.noise_fraction * 100:.1f}% of energy, directions above the edge <= {rep.signal_rank_max}"
            lines.append(head)
            if noisy:
                lines.append("   target   per-module rank   weighted rank   denoised per-module   denoised weighted   driven by")
                for t in ENERGY_TARGETS:
                    lines.append(f"   {t * 100:5.1f}%   {rep.per_module_rank[t]:15d}   {rep.weighted_rank[t]:13d}   "
                                 f"{rep.per_module_rank_dn[t]:19d}   {rep.weighted_rank_dn[t]:17d}   {rep.worst[t]}")
            else:
                lines.append("   target   per-module rank   weighted rank   driven by")
                for t in ENERGY_TARGETS:
                    lines.append(f"   {t * 100:5.1f}%   {rep.per_module_rank[t]:15d}   {rep.weighted_rank[t]:13d}   {rep.worst[t]}")
        if self.modules:
            lines.append("energy kept by a uniform rank (median / minimum over the target tensors" + (", raw and denoised)" if noisy else ")"))
            for r, med, mn, medd, mnd in self.candidate_table():
                row = f"   rank {r:5d}   {med * 100:6.2f}% / {mn * 100:6.2f}%" if med is not None else f"   rank {r:5d}   -"
                if noisy and medd is not None:
                    row += f"   denoised {medd * 100:6.2f}% / {mnd * 100:6.2f}%"
                lines.append(row)
        zero = [m.name for m in self.modules.values() if m.all_zero]
        if zero:
            lines.append(f"{len(zero)} target tensor(s) unchanged (nothing to extract): " + ", ".join(zero[:6]) + (" ..." if len(zero) > 6 else ""))
        if self.others:
            unchanged = sum(1 for o in self.others if o.all_zero)
            lines.append(f"energy outside the LoRA's reach: {self.outside_fraction * 100:.2f}% of the delta energy in "
                         f"{len(self.others)} non target tensor(s) (norms, modulation, biases, filtered weights), {unchanged} unchanged")
            top = sorted((o for o in self.others if o.rel_change is not None), key=lambda o: -o.rel_change)[:5]
            if top:
                lines.append("   largest relative change: " + ", ".join(f"{o.name} {o.rel_change * 100:.2f}%" for o in top))
        if self.clamped:
            lines.append(f"{len(self.clamped)} module(s) are smaller than twice the requested rank and clamp:")
            for name, r in self.clamped[:10]:
                lines.append(f"   max rank {r:4d}  {name}")
        if self.noise_model.get("text"):
            lines.append(self.noise_model["text"])
        for n in self.notes:
            lines.append(n)
        return "\n".join(lines)

    def median_energy_curve(self, denoised: bool = False, max_rank: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(ranks 1..R, median over the target tensors of the energy kept at that rank). A tensor whose spectrum
        is shorter than R keeps everything beyond its length."""
        mods = [m for m in self.modules.values() if not m.all_zero]
        if not mods:
            return torch.zeros(0), torch.zeros(0)
        curves = []
        R = max_rank or max(m.sv.numel() for m in mods)
        for m in mods:
            sv = m.denoised_sv() if denoised else m.sv
            s2 = sv.double() ** 2
            tot = float(s2.sum().item())
            c = torch.cumsum(s2, 0) / tot if tot > 0 else torch.ones(0, dtype=torch.float64)
            c = c[:R]
            if c.numel() < R:
                c = torch.cat([c, torch.ones(R - c.numel(), dtype=torch.float64)])
            curves.append(c)
        stack = torch.stack(curves)
        return torch.arange(1, R + 1), stack.median(dim=0).values.float()

    def compare_rows(self, other: "AnalysisReport") -> list:
        """[(metric, value for self, value for other)] as strings, the basis of the compare table and text."""
        rows = [("target tensors", str(len(self.modules)), str(len(other.modules))),
                ("delta norm (targets)", f"{math.sqrt(self.target_energy):.4g}", f"{math.sqrt(other.target_energy):.4g}"),
                ("outside the LoRA's reach", f"{self.outside_fraction * 100:.2f}%", f"{other.outside_fraction * 100:.2f}%")]
        f = lambda r, fn: fn(r) if r is not None else "-"  # noqa: E731
        for g in GROUPS:
            ra, rb = self.groups.get(g), other.groups.get(g)
            if ra is None and rb is None:
                continue
            rows.append((f"[{g}] delta norm", f(ra, lambda r: f"{r.total_norm:.4g}"), f(rb, lambda r: f"{r.total_norm:.4g}")))
            rows.append(("   median effective rank", f(ra, lambda r: f"{r.median_effective_rank:.1f}"), f(rb, lambda r: f"{r.median_effective_rank:.1f}")))
            rows.append(("   median stable rank", f(ra, lambda r: f"{r.median_stable_rank:.2f}"), f(rb, lambda r: f"{r.median_stable_rank:.2f}")))
            rows.append(("   median change of the weight", f(ra, lambda r: f"{r.median_rel_change * 100:.2f}%" if r.median_rel_change is not None else "-"),
                         f(rb, lambda r: f"{r.median_rel_change * 100:.2f}%" if r.median_rel_change is not None else "-")))
            rows.append(("   noise share", f(ra, lambda r: f"{r.noise_fraction * 100:.1f}%"), f(rb, lambda r: f"{r.noise_fraction * 100:.1f}%")))
            for t in (0.9, 0.99):
                rows.append((f"   weighted rank {t * 100:.0f}% raw", f(ra, lambda r: str(r.weighted_rank[t])), f(rb, lambda r: str(r.weighted_rank[t]))))
                rows.append((f"   weighted rank {t * 100:.0f}% denoised", f(ra, lambda r: str(r.weighted_rank_dn[t])), f(rb, lambda r: str(r.weighted_rank_dn[t]))))
        for (r, ma, _, da, _), (_, mb, _, db, _) in zip(self.candidate_table(), other.candidate_table()):
            fa = f"{ma * 100:.1f}% raw, {da * 100:.1f}% denoised" if ma is not None else "-"
            fb = f"{mb * 100:.1f}% raw, {db * 100:.1f}% denoised" if mb is not None else "-"
            rows.append((f"median energy at rank {r}", fa, fb))
        return rows

    def compare_text(self, other: "AnalysisReport") -> str:
        """Side by side group summaries of two analyses."""
        lines = [f"A: {self.label}", f"B: {other.label}", f"{'':30s} {'A':>30s} {'B':>30s}"]
        for name, fa, fb in self.compare_rows(other):
            lines.append(f"{name:30s} {fa:>30s} {fb:>30s}")
        return "\n".join(lines)

    # ---- persistence
    def to_dict(self) -> dict:
        return {"tool": f"{TOOL_NAME} {__version__}", "run": dict(self.run), "label": self.label,
                "noise_model": dict(self.noise_model), "notes": list(self.notes),
                "energy_thresholds": list(ENERGY_THRESHOLDS), "candidate_ranks": list(CANDIDATE_RANKS),
                "tensors": [m.to_dict() for m in self.modules.values()],
                "others": [o.to_dict() for o in self.others],
                "groups": {g: r.to_dict() for g, r in self.groups.items()},
                "candidates": [{"rank": r, "median": med, "min": mn, "median_denoised": medd, "min_denoised": mnd}
                               for r, med, mn, medd, mnd in self.candidate_table()],
                "outside_fraction": self.outside_fraction, "clamped": [list(c) for c in self.clamped]}

    def save(self, stem: str) -> tuple[str, str]:
        """Writes <stem>.spectrum.npz (singular values) and <stem>.spectrum.json (everything else). Returns both paths."""
        import numpy as np
        stem = spectrum_stem(stem)
        npz, js = stem + SPECTRUM_SUFFIX + ".npz", stem + SPECTRUM_SUFFIX + ".json"
        arrays = {}
        for m in self.modules.values():
            arrays[m.name] = m.sv.detach().float().cpu().numpy()
            if m.sigma_null is not None:
                arrays[m.name + "__null"] = m.sigma_null.detach().float().cpu().numpy()
        with open(npz, "wb") as f:
            np.savez_compressed(f, **arrays)
        d = self.to_dict()
        d["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(js, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        return npz, js

    @classmethod
    def load(cls, path: str) -> "AnalysisReport":
        """path: the .spectrum.json or .spectrum.npz (or the stem); both files must exist."""
        import numpy as np
        stem = spectrum_stem(path)
        npz, js = stem + SPECTRUM_SUFFIX + ".npz", stem + SPECTRUM_SUFFIX + ".json"
        if not (os.path.exists(npz) and os.path.exists(js)):
            raise FileNotFoundError(f"a saved analysis needs both {os.path.basename(js)} and {os.path.basename(npz)}")
        with open(js, encoding="utf-8") as f:
            d = json.load(f)
        rep = cls(run=dict(d.get("run", {})), noise_model=dict(d.get("noise_model", {})), notes=list(d.get("notes", [])))
        with np.load(npz) as z:
            for td in d.get("tensors", []):
                sv = torch.from_numpy(np.asarray(z[td["name"]], dtype="float32")) if td["name"] in z.files else torch.zeros(0)
                nk = td["name"] + "__null"
                sn = torch.from_numpy(np.asarray(z[nk], dtype="float32")) if nk in z.files else None
                rep.add(ModuleSpectrum.from_dict(td, sv, sn))
        rep.others = [OtherTensor.from_dict(o) for o in d.get("others", [])]
        rep.clamped = [tuple(c) for c in d.get("clamped", [])]
        rep.finish()
        return rep


def spectrum_stem(path: str) -> str:
    """'x.spectrum.json' | 'x.spectrum.npz' | 'x.spectrum' | 'x' -> 'x'."""
    p = path
    for ext in (".json", ".npz"):
        if p.lower().endswith(ext):
            p = p[: -len(ext)]
    if p.lower().endswith(SPECTRUM_SUFFIX):
        p = p[: -len(SPECTRUM_SUFFIX)]
    return p


# ----------------------------------------------------------------------------- the pass
def analyze_sources(sources: list, canon_list: list, names: dict, device, noise: dict | None = None,
                    requested_rank: int | None = None, progress=None, cancel=None,
                    null_spectrum: bool = False) -> AnalysisReport:
    """sources: DeltaSource list. canon_list: modules to analyze. names: canon -> bare module name.
    cancel: callable, raises Cancelled when true. null_spectrum: also store the spectrum of modeled noise.
    The noise model applies when a single source contributes to a module and can report its inputs' layouts
    (CheckpointDelta); noise (canon -> total noise energy) is accepted from older callers."""
    from .engine import Cancelled
    rep = AnalysisReport()
    noise = noise or {}
    total = len(canon_list)
    for i, c in enumerate(canon_list):
        if cancel is not None and cancel():
            raise Cancelled("cancelled by the user")
        if progress is not None:
            progress(i, total, names.get(c, c))
        contributing = [s for s in sources if c in s.modules()]
        if not contributing:
            continue
        name = names.get(c, c)
        exact = all(s.is_exact(c) for s in contributing)
        stats = {}
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
                if len(contributing) == 1 and hasattr(s, "read_pair"):
                    t, b = s.read_pair(c, device)
                    d = (t - b) * s.factor(c)
                    stats = _pair_stats(s, c, b, d)
                    del t, b
                else:
                    d = s.delta(c, device)
                delta = d if delta is None else delta + d
            shape = tuple(delta.shape)
            sv = spectrum_full(delta)
            if stats.get("noise_var", 0.0) > 0 and sv.numel():
                edge = math.sqrt(stats["noise_var"]) * (math.sqrt(shape[0]) + math.sqrt(shape[1]))
                if edge < 10.0 * gram_resolution(float(sv[0])):
                    sv = torch.linalg.svdvals(delta.to(torch.float32))
                    stats["resolved_by_svd"] = True
            if null_spectrum and stats.get("noise_var", 0.0) > 0:
                stats["sigma_null"] = noise_spectrum(shape, stats["noise_var"], delta.device).detach().float().cpu()
            del delta
            if device.type == "cuda" and shape[0] * shape[1] >= (1 << 24):
                torch.cuda.empty_cache()
        ms = ModuleSpectrum(c, name, group_of(name), shape, sv.detach().float().cpu(), exact,
                            noise_var=stats.get("noise_var", 0.0), base_fro=stats.get("base_fro"),
                            zero_fraction=stats.get("zero_fraction", 0.0), layouts=stats.get("layouts", ()),
                            dtypes=stats.get("dtypes", ()), resolved_by_svd=stats.get("resolved_by_svd", False),
                            sigma_null=stats.get("sigma_null"))
        if not stats and c in noise and shape[0] * shape[1] > 0:
            ms.noise_var = float(noise[c]) / (shape[0] * shape[1])
        rep.add(ms)
    rep.finish(requested_rank)
    if rep.has_noise_model:
        rep.notes.append("noise floor: the noise edge is drawn from the storage formats' rounding and quantization error; "
                         "it is an upper bound where the fine tune left elements untouched")
    if any(m.resolved_by_svd for m in rep.modules.values()):
        n = sum(1 for m in rep.modules.values() if m.resolved_by_svd)
        rep.notes.append(f"{n} tensor(s) recomputed with a full SVD because the noise edge sat below the Gram resolution")
    return rep


def _pair_stats(src, c: str, base_w: torch.Tensor, delta: torch.Tensor) -> dict:
    """Noise variance, base norm and zero fraction for a checkpoint pair module."""
    lb, lt = src.layouts(c)
    db, dt = src.dtypes(c)
    fro = float(base_w.norm().item())
    var = layout_noise_variance(lb, db, base_w, fro) + layout_noise_variance(lt, dt, base_w, fro)
    zero = float((delta == 0).float().mean().item()) if delta.numel() else 1.0
    return {"noise_var": var * (1.0 - zero), "base_fro": fro, "zero_fraction": zero, "layouts": (lb, lt), "dtypes": (db, dt)}
