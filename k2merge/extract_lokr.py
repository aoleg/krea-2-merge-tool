"""Function 3b: extract a full rank LoKr from the difference between two checkpoints.

Per module, the delta D = target - base is fitted by the nearest Kronecker product kron(w1, w2) in Frobenius norm
(Van Loan and Pitsianis, 1993): D is rearranged so that every (out/p x in/q) block becomes one row of R, the top
eigenvector u of the small Gram matrix R R^T (p*q x p*q, fp64) gives w1, and R^T u gives w2. The captured share
lambda_1 / trace(R R^T) is exact: it is the part of the delta's energy a single Kronecker term can hold.

Shapes follow LyCORIS's factorization rule, the one ai-toolkit uses, so a trainer that creates a LoKr at the same
factor accepts the file. w2 is always full rank. A template LoKr (a file the trainer wrote, or the built-in
ai-toolkit layout) supplies the key names, the module set, the alpha marker and the format metadata.

One module at a time, streamed: the base and target weights, the delta and its rearranged copy are the only
large tensors alive (about four times the module's fp32 size, 1.6 GB for Krea 2's largest).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field

import torch

from . import TOOL_NAME, __version__
from .engine import Cancelled, pick_device, recipe_for_metadata
from .extract import FILTER_PRESETS, _open, _selected_modules
from .keys import GROUPS, KREA2, ArchProfile, canon, detect_family, group_of, block_index
from .meta import redact_paths
from .st_io import StreamWriter, TensorReader, read_header, same_file

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference")
# Built-in templates: the header of a real ai-toolkit LoKr, names and metadata only (reference/*.header.json).
BUILTIN_TEMPLATES = {"krea2": "aitoolkit_krea2_lokr"}
# ai-toolkit's full rank marker as it lands in the file: alpha 1e10 rounded through bf16. With both factors full,
# LyCORIS, ai-toolkit and ComfyUI apply scale 1 whatever the alpha, so the value only has to look like theirs.
FULL_RANK_ALPHA = 9999220736.0
W1_SCALES = ("init", "balanced")
LOKR_DTYPES = {"fp32": ("F32", torch.float32), "bf16": ("BF16", torch.bfloat16), "fp16": ("F16", torch.float16)}
FACTOR_MIN, FACTOR_MAX = 2, 32
# Template metadata carried into the output; name fields are rewritten to the output's file name. Everything else
# (hashes of the template file, training progress, dataset descriptions) describes the template, not the output.
TEMPLATE_META_KEEP = ("format", "version", "ss_base_model_version", "software",
                      "modelspec.architecture", "modelspec.implementation")
TEMPLATE_META_RENAME = ("name", "ss_output_name")
GRAM_CHUNK = 1 << 24          # elements of R converted to fp64 at a time (128 MB)


class TemplateError(ValueError):
    pass


# ----------------------------------------------------------------------------- shapes
def factorization(dimension: int, factor: int) -> tuple[int, int]:
    """(m, n) with m * n == dimension and m <= n, by LyCORIS's rule (lycoris.functional.general.factorization):
    an exact divisor `factor` gives (factor, dimension / factor); otherwise m grows through the divisors of the
    dimension while m + n keeps shrinking and m stays at or below the factor."""
    if dimension < 1:
        raise ValueError(f"dimension {dimension}")
    if factor > 0 and dimension % factor == 0:
        m, n = factor, dimension // factor
        return (n, m) if m > n else (m, n)
    if factor <= 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    return (n, m) if m > n else (m, n)


def lokr_shapes(out_features: int, in_features: int, factor: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(w1 shape, w2 shape) of a full rank LoKr on an out x in linear (LyCORIS LokrModule)."""
    out_m, out_n = factorization(out_features, factor)
    in_m, in_n = factorization(in_features, factor)
    return (out_m, in_m), (out_n, in_n)


# ----------------------------------------------------------------------------- options
@dataclass
class LokrExtractOptions:
    factor: int = 4
    filter: str = "trainer"           # all | attn | blocks | trainer | custom (as the LoRA extraction)
    include: str = ""
    exclude: str = ""
    dtype: str = "fp32"
    template: str = ""                # "" built-in ai-toolkit layout, "none" generic layout, else a LoKr file
    w1_scale: str = "init"            # init: w1 at the typical size of the trainer's random start; balanced

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        o = cls()
        for k, v in (d or {}).items():
            if hasattr(o, k):
                setattr(o, k, v)
        o.factor = int(o.factor)
        return o

    def validate(self):
        if not FACTOR_MIN <= int(self.factor) <= FACTOR_MAX:
            raise ValueError(f"factor {self.factor}: choose {FACTOR_MIN} to {FACTOR_MAX}")
        if self.dtype not in LOKR_DTYPES:
            raise ValueError(f"dtype {self.dtype!r}: choose one of {', '.join(LOKR_DTYPES)}")
        if self.w1_scale not in W1_SCALES:
            raise ValueError(f"w1 scale {self.w1_scale!r}: choose one of {', '.join(W1_SCALES)}")
        if self.filter not in FILTER_PRESETS:
            raise ValueError(f"module filter {self.filter!r}: choose one of {', '.join(FILTER_PRESETS)}")


# ----------------------------------------------------------------------------- template
@dataclass
class TemplateModule:
    key_base: str                     # tensor key without .lokr_w1 / .lokr_w2 / .alpha, as the template writes it
    w1: tuple
    w2: tuple


@dataclass
class LokrTemplate:
    label: str
    prefix: str = "diffusion_model."
    modules: dict | None = None       # canon -> TemplateModule; None: no restriction, keys = prefix + module
    alpha: float | None = FULL_RANK_ALPHA
    alpha_dtype: str = "F32"
    metadata: dict = field(default_factory=lambda: {"format": "pt"})
    rename: tuple = ()                # metadata keys set to the output's file name
    dropped: list = field(default_factory=list)
    factor: int | None = None         # the factor of the template's own modules, when they agree

    def summary(self) -> str:
        mods = "any module" if self.modules is None else f"{len(self.modules)} modules"
        f = f", factor {self.factor}" if self.factor else ""
        a = "no alpha" if self.alpha is None else f"alpha {self.alpha:g}"
        return f"{self.label} ({mods}{f}, {a})"


def _generic_template(label: str) -> LokrTemplate:
    return LokrTemplate(label=label)


def _template_from_header(header: dict, label: str, alpha_reader=None) -> LokrTemplate:
    meta = dict(header.get("__metadata__") or {})
    extra = header.get("__template__") or {}
    infos = {k: v for k, v in header.items() if k not in ("__metadata__", "__template__")}
    mods: dict = {}
    has_alpha = False
    alpha_key = None
    for k, v in infos.items():
        for bad in (".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2"):
            if k.endswith(bad):
                raise TemplateError(f"template {label}: {k} is a decomposed factor; the extraction writes full rank "
                                    f"factors only (ai-toolkit: lokr_full_rank true)")
        if k.endswith(".alpha"):
            has_alpha = True
            alpha_key = alpha_key or k
    for k, v in infos.items():
        if not k.endswith(".lokr_w1"):
            continue
        base = k[: -len(".lokr_w1")]
        w2 = infos.get(base + ".lokr_w2")
        if w2 is None or len(v["shape"]) != 2 or len(w2["shape"]) != 2:
            raise TemplateError(f"template {label}: {base} has no full rank 2-D lokr_w2 next to its lokr_w1")
        mods[canon(base)] = TemplateModule(base, tuple(int(x) for x in v["shape"]), tuple(int(x) for x in w2["shape"]))
    if not mods:
        raise TemplateError(f"template {label}: no LoKr modules (lokr_w1 / lokr_w2 tensors)")
    prefix = ""
    first = next(iter(mods.values())).key_base
    for p in ("diffusion_model.", "model.diffusion_model.", "lycoris_", "lora_unet_"):
        if first.startswith(p):
            prefix = p
            break
    alpha = None
    alpha_dtype = "F32"
    if has_alpha:
        alpha_dtype = infos[alpha_key]["dtype"]
        if "alpha" in extra:
            alpha = float(extra["alpha"])
        elif alpha_reader is not None:
            alpha = float(alpha_reader(alpha_key))
        else:
            alpha = FULL_RANK_ALPHA
    shapes = Counter(m.w1 for m in mods.values())
    w1, _n = shapes.most_common(1)[0]
    factor = w1[0] if (len(shapes) == 1 and w1[0] == w1[1]) else None
    kept = {k: str(meta[k]) for k in TEMPLATE_META_KEEP if k in meta}
    rename = tuple(k for k in TEMPLATE_META_RENAME if k in meta)
    dropped = sorted(k for k in meta if k not in kept and k not in rename)
    if "format" not in kept:
        kept["format"] = "pt"
    return LokrTemplate(label=label, prefix=prefix, modules=mods, alpha=alpha, alpha_dtype=alpha_dtype,
                        metadata=redact_paths(kept), rename=rename, dropped=dropped, factor=factor)


def builtin_template_path(profile: ArchProfile) -> str | None:
    name = BUILTIN_TEMPLATES.get(profile.family)
    return os.path.join(REFERENCE_DIR, name + ".header.json") if name else None


def load_template(spec: str | None, profile: ArchProfile) -> LokrTemplate:
    """'' = the built-in ai-toolkit template of the architecture (the generic layout when there is none),
    'none' = the generic layout, else a LoKr .safetensors file or a .header.json."""
    spec = (spec or "").strip()
    if spec.lower() == "none":
        return _generic_template("generic ai-toolkit layout")
    if not spec:
        p = builtin_template_path(profile)
        if p is None or not os.path.isfile(p):
            return _generic_template(f"generic ai-toolkit layout (no built-in {profile.label} template)")
        with open(p, encoding="utf-8") as f:
            header = json.load(f)
        return _template_from_header(header, f"built-in ai-toolkit {profile.label} layout")
    if not os.path.isfile(spec):
        raise TemplateError(f"template {spec}: no such file")
    if spec.lower().endswith(".json"):
        with open(spec, encoding="utf-8") as f:
            return _template_from_header(json.load(f), os.path.basename(spec))
    header, _ = read_header(spec)
    with TensorReader(spec) as r:
        return _template_from_header(header, os.path.basename(spec),
                                     alpha_reader=lambda k: r.read(k).reshape(-1)[0].float().item())


# ----------------------------------------------------------------------------- module plan
@dataclass
class LokrModulePlan:
    canon: str
    module: str
    target_key: str
    base_key: str
    shape: tuple
    w1: tuple
    w2: tuple
    key_base: str

    @property
    def keys(self) -> tuple[str, str, str]:
        return self.key_base + ".lokr_w1", self.key_base + ".lokr_w2", self.key_base + ".alpha"


def plan_modules(fb, ft, opts: LokrExtractOptions, template: LokrTemplate):
    """([LokrModulePlan], skipped [(module, why)], notes [str]). Raises on a template that does not fit the model."""
    mods, skipped = _selected_modules(fb, ft, opts)
    plans, notes = [], []
    f = int(opts.factor)
    for c, module, tk, bk, shape in mods:
        out_f, in_f = int(shape[0]), int(shape[1])
        w1, w2 = lokr_shapes(out_f, in_f, f)
        key_base = template.prefix + module if template.prefix in ("diffusion_model.", "model.diffusion_model.") \
            else template.prefix + module.replace(".", "_")
        if template.modules is not None:
            tm = template.modules.get(c)
            if tm is None:
                skipped.append((module, "not in the template"))
                continue
            implied = (tm.w1[0] * tm.w2[0], tm.w1[1] * tm.w2[1])
            if implied != (out_f, in_f):
                raise TemplateError(f"template {template.label}: {module} is {list(implied)} in the template and "
                                    f"{[out_f, in_f]} in the checkpoint; the template is for another model")
            if template.factor == f and (tm.w1, tm.w2) != (w1, w2):
                raise TemplateError(f"template {template.label}: {module} has w1 {list(tm.w1)} and w2 {list(tm.w2)}; "
                                    f"factor {f} by the LyCORIS rule gives {list(w1)} and {list(w2)}")
            key_base = tm.key_base
        plans.append(LokrModulePlan(c, module, tk, bk, (out_f, in_f), w1, w2, key_base))
    if template.modules is not None:
        planned = {p.canon for p in plans}
        missing = [tm.key_base for cc, tm in template.modules.items() if cc not in planned]
        if missing:
            notes.append(f"{len(missing)} template module(s) are not extracted (module filter, or not a 2-D weight in "
                         f"both checkpoints), e.g. {missing[:2]}")
    return plans, skipped, notes


# ----------------------------------------------------------------------------- the fit
def _rearrange(delta: torch.Tensor, p: int, q: int) -> torch.Tensor:
    """D [p*m, q*n] -> R [p*q, m*n]; row i*q + j is block (i, j) of D, flattened. A copy."""
    P, Q = delta.shape
    m, n = P // p, Q // q
    return delta.reshape(p, m, q, n).permute(0, 2, 1, 3).reshape(p * q, m * n)


def _chunks(cols: int, rows: int):
    step = max(1, GRAM_CHUNK // max(rows, 1))
    for s in range(0, cols, step):
        yield s, min(cols, s + step)


def init_w1_rms(q: int) -> float:
    """RMS of LyCORIS's random start of w1 (kaiming uniform, a = sqrt(5), fan in q): bound 1 / sqrt(q)."""
    return 1.0 / math.sqrt(3.0 * q)


@dataclass
class KronFit:
    w1: torch.Tensor                  # fp32 [p, q]
    w2: torch.Tensor                  # fp32 [m, n]
    energy: float                     # ||D||^2
    captured: float                   # lambda_1 / ||D||^2 (1.0 for a module with no change)
    shares: list                      # every eigenvalue share, descending
    no_change: bool = False


def nearest_kronecker(delta: torch.Tensor, p: int, q: int, w1_scale: str = "init", seed_name: str = "") -> KronFit:
    """Best kron(w1 [p, q], w2) fit of a 2-D delta in Frobenius norm. A delta with no energy gets the trainer's
    own start instead: a random w1 and a zero w2, so the module still trains."""
    P, Q = delta.shape
    if P % p or Q % q:
        raise ValueError(f"delta {list(delta.shape)} does not split into {p} x {q} blocks")
    m, n = P // p, Q // q
    R = _rearrange(delta.float(), p, q)
    rows, cols = R.shape
    G = torch.zeros(rows, rows, dtype=torch.float64, device=R.device)
    for s, e in _chunks(cols, rows):
        c = R[:, s:e].double()
        G += c @ c.T
        del c
    G = G.cpu()
    trace = float(G.diagonal().sum().item())
    if not trace > 1e-30:
        g = torch.Generator().manual_seed(zlib.crc32(seed_name.encode("utf-8")))
        bound = 1.0 / math.sqrt(q)
        w1 = (torch.rand(p, q, generator=g, dtype=torch.float64) * 2 - 1) * bound
        return KronFit(w1.float(), torch.zeros(m, n), 0.0, 1.0, [1.0] + [0.0] * (rows - 1), no_change=True)
    evals, evecs = torch.linalg.eigh(G)            # ascending, fp64, on the CPU: deterministic
    lam1 = float(evals[-1].item())
    u = evecs[:, -1].clone()
    i = int(u.abs().argmax().item())
    if u[i] < 0:                                   # the sign is free; fix it so a rerun writes the same bytes
        u = -u
    shares = [max(0.0, float(x)) / trace for x in evals.flip(0).tolist()]
    s1 = math.sqrt(max(lam1, 0.0))
    if w1_scale == "balanced":
        a = math.sqrt(s1)
    else:
        a = init_w1_rms(q) * math.sqrt(rows)       # ||w1||_F of a random start of this shape
    ud = u.to(R.device)
    w2 = torch.empty(cols, dtype=torch.float64, device=R.device)
    for s, e in _chunks(cols, rows):
        w2[s:e] = R[:, s:e].double().T @ ud        # R^T u: the best w2 for this w1, before the scale split
    w1 = (u * a).reshape(p, q)
    w2 = (w2 / a).reshape(m, n)
    del R
    return KronFit(w1.float().cpu(), w2.float(), trace, lam1 / trace, shares)


def residual_share(delta: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> float:
    """||D - kron(w1, w2)||^2 / ||D||^2 without forming the product: ||D||^2 - 2 <D, w1 x w2> + ||w1||^2 ||w2||^2,
    where <D, w1 x w2> = sum_ij w1[i, j] <block_ij(D), w2>. fp64 accumulation."""
    p, q = w1.shape
    R = _rearrange(delta.float(), p, q)
    rows, cols = R.shape
    w2v = w2.reshape(-1).to(device=R.device, dtype=torch.float64)
    d2 = 0.0
    proj = torch.zeros(rows, dtype=torch.float64, device=R.device)
    for s, e in _chunks(cols, rows):
        c = R[:, s:e].double()
        d2 += float((c * c).sum().item())
        proj += c @ w2v[s:e]
        del c
    del R
    if d2 <= 1e-30:
        return 0.0
    w1d = w1.reshape(-1).to(dtype=torch.float64, device=proj.device)
    inner = float((w1d * proj).sum().item())
    fit2 = float((w1d * w1d).sum().item()) * float((w2v * w2v).sum().item())
    return max(0.0, (d2 - 2.0 * inner + fit2) / d2)


# ----------------------------------------------------------------------------- report
@dataclass
class LokrModuleResult:
    name: str
    group: str
    block: int | None
    shape: tuple
    w1: tuple
    w2: tuple
    energy: float
    captured: float
    shares: list
    no_change: bool = False
    residual: float | None = None     # measured from the stored factors (extraction only)


@dataclass
class LokrReport:
    factor: int
    template: str
    profile: str
    blocks: int
    rows: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    run: dict = field(default_factory=dict)

    @property
    def energy(self) -> float:
        return sum(r.energy for r in self.rows)

    def captured(self, rows=None) -> float:
        rows = self.rows if rows is None else rows
        e = sum(r.energy for r in rows)
        return sum(r.captured * r.energy for r in rows) / e if e > 0 else 1.0

    def term_shares(self, k: int = 4) -> list:
        e = self.energy
        if e <= 0:
            return []
        out = [0.0] * k
        for r in self.rows:
            for i, s in enumerate(r.shares[:k]):
                out[i] += s * r.energy / e
        return out

    def _zone(self, b: int | None) -> str:
        if b is None:
            return "non block"
        z = min(2, int(3 * b / max(self.blocks, 1)))
        lo = math.ceil(z * self.blocks / 3)
        hi = math.ceil((z + 1) * self.blocks / 3) - 1
        return f"blocks {lo}-{hi}"

    def text(self) -> str:
        rows = [r for r in self.rows if not r.no_change]
        E = self.energy
        lines = [f"LoKr extraction at factor {self.factor}, full rank w2, {self.profile}; template: {self.template}",
                 f"modules: {len(self.rows)}" + (f", {len(self.rows) - len(rows)} with no change (written as a fresh start: random w1, zero w2)"
                                                  if len(rows) != len(self.rows) else "")]
        if not rows:
            lines.append("no module changed between base and target")
            return "\n".join(lines + self.notes)
        caps = sorted(r.captured for r in rows)
        lines.append(f"captured share of the delta energy: {self.captured():.3f} energy weighted "
                     f"(modules: min {caps[0]:.3f}, median {caps[len(caps) // 2]:.3f}, max {caps[-1]:.3f})")
        ts = self.term_shares(4)
        if len(ts) > 1:
            lines.append("an exact copy needs more Kronecker terms per module; energy weighted share of terms 1 to 4: "
                         + ", ".join(f"{x:.3f}" for x in ts))
        lines.append(f"  {'scope':18s} {'modules':>7s} {'energy':>7s} {'captured':>9s}")

        def line(label, rs):
            if not rs:
                return
            e = sum(r.energy for r in rs)
            lines.append(f"  {label:18s} {len(rs):7d} {100 * e / E:6.1f}% {self.captured(rs):9.3f}")
        line("all", rows)
        for g in GROUPS:
            line(g, [r for r in rows if r.group == g])
        zones = []
        for r in rows:
            z = self._zone(r.block)
            if z not in zones:
                zones.append(z)
        for z in sorted(zones, key=lambda s: (s == "non block", int(s.split()[1].split("-")[0]) if s != "non block" else 0)):
            line(z, [r for r in rows if self._zone(r.block) == z])
        top = sorted(rows, key=lambda r: -r.energy)[:6]
        lines.append("largest modules: " + "; ".join(f"{r.name} {100 * r.energy / E:.1f}% captured {r.captured:.3f}" for r in top))
        worst = sorted([r for r in rows if r.energy >= 0.002 * E], key=lambda r: r.captured)[:6]
        if worst:
            lines.append("lowest captured (modules above 0.2% of the energy): "
                         + "; ".join(f"{r.name} {r.captured:.3f}" for r in worst))
        res = [r.residual for r in self.rows if r.residual is not None and not r.no_change]
        if res:
            gap = max(abs((1 - r.captured) - r.residual) for r in self.rows if r.residual is not None and not r.no_change)
            lines.append(f"check: residual measured from the stored factors differs from 1 - captured by at most {gap:.2e}")
        if self.skipped:
            lines.append(f"{len(self.skipped)} module(s) skipped: " + ", ".join(f"{m} ({why})" for m, why in self.skipped[:5]))
        lines += self.notes
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"factor": self.factor, "template": self.template, "profile": self.profile,
                "captured": self.captured(), "term_shares": self.term_shares(4),
                "modules": [{"name": r.name, "group": r.group, "block": r.block, "shape": list(r.shape),
                             "w1": list(r.w1), "w2": list(r.w2), "energy": r.energy, "captured": r.captured,
                             "shares": r.shares[:4], "no_change": r.no_change, "residual": r.residual}
                            for r in self.rows],
                "skipped": [list(s) for s in self.skipped], "notes": list(self.notes), "run": dict(self.run)}


# ----------------------------------------------------------------------------- runs
@dataclass
class LokrExtractResult:
    path: str
    modules: int
    skipped: list
    captured: float
    captured_min: float
    no_change: int
    residual_gap: float
    seconds: float
    size_bytes: int
    template: str
    dropped_metadata: list
    report: LokrReport

    def summary(self) -> str:
        return (f"wrote {self.path}: {self.modules} LoKr modules, captured {self.captured:.3f} of the delta energy "
                f"(module min {self.captured_min:.3f}), {self.size_bytes / 1e6:.1f} MB, {len(self.skipped)} skipped, "
                f"{self.seconds:.1f}s")


def _profile_of(fb, ft) -> ArchProfile:
    return detect_family(ft.reader.infos) or detect_family(fb.reader.infos) or KREA2


def _check_paths(base_path, target_path, out_path, opts):
    paths = [base_path, target_path]
    t = (opts.template or "").strip()
    if t and t.lower() != "none":
        paths.append(t)
    for p in paths:
        if out_path and same_file(p, out_path):
            raise ValueError("the output path is one of the input files")


def _fit_all(fb, ft, plans, opts, device, progress, cancel, on_fit):
    total = len(plans)
    for i, pl in enumerate(plans):
        if cancel is not None and cancel():
            raise Cancelled("cancelled by the user")
        if progress is not None:
            progress(i, total, pl.module)
        delta = ft.read_fp32(pl.target_key, device=device)
        delta -= fb.read_fp32(pl.base_key, device=device)
        fit = nearest_kronecker(delta, pl.w1[0], pl.w1[1], opts.w1_scale, seed_name=pl.canon)
        on_fit(pl, fit, delta)
        del delta, fit
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if progress is not None:
        progress(total, total, "done")


def _row(pl: LokrModulePlan, fit: KronFit) -> LokrModuleResult:
    return LokrModuleResult(pl.module, group_of(pl.module), block_index(pl.module), pl.shape, pl.w1, pl.w2,
                            fit.energy, fit.captured, fit.shares, fit.no_change)


def analyze_extract_lokr(base_path: str, target_path: str, opts: LokrExtractOptions, use_gpu=True,
                         progress=None, cancel=None) -> LokrReport:
    """The captured share of every module at the chosen factor. Writes nothing."""
    opts.validate()
    device = pick_device(use_gpu)
    rb, rt, fb, ft = _open(base_path, target_path)
    try:
        prof = _profile_of(fb, ft)
        tpl = load_template(opts.template, prof)
        plans, skipped, notes = plan_modules(fb, ft, opts, tpl)
        if not plans:
            raise ValueError("no module to extract: no common 2-D weight passes the module filter and the template")
        rep = LokrReport(int(opts.factor), tpl.summary(), prof.label, prof.block_count(ft.reader.infos) or prof.blocks,
                         skipped=skipped, notes=notes)
        _fit_all(fb, ft, plans, opts, device, progress, cancel, lambda pl, fit, d: rep.rows.append(_row(pl, fit)))
        rep.run = {"function": "extract_lokr", "base": os.path.abspath(base_path), "target": os.path.abspath(target_path),
                   "options": opts.to_dict(), "device": str(device), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        return rep
    finally:
        rb.close()
        rt.close()


def output_metadata(tpl: LokrTemplate, out_path: str, recipe: dict, captured: float | None = None) -> dict:
    meta = dict(tpl.metadata)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    for k in tpl.rename:
        meta[k] = stem
    meta["merge_tool"] = f"{TOOL_NAME} {__version__}"
    meta["merge_recipe"] = json.dumps(recipe_for_metadata(recipe), separators=(",", ":"))
    return meta


VERIFY_MODULES = 6


def extract_lokr(base_path: str, target_path: str, out_path: str, opts: LokrExtractOptions, use_gpu=True,
                 progress=None, cancel=None, log=None) -> LokrExtractResult:
    log = log or (lambda s: None)
    opts.validate()
    _check_paths(base_path, target_path, out_path, opts)
    device = pick_device(use_gpu)
    t0 = time.time()
    rb, rt, fb, ft = _open(base_path, target_path)
    try:
        prof = _profile_of(fb, ft)
        tpl = load_template(opts.template, prof)
        log(f"template: {tpl.summary()}")
        if tpl.dropped:
            log(f"template metadata not carried over: {', '.join(tpl.dropped)}")
        plans, skipped, notes = plan_modules(fb, ft, opts, tpl)
        for n in notes:
            log(n)
        if not plans:
            raise ValueError("no module to extract: no common 2-D weight passes the module filter and the template")
        recipe = {"function": "extract_lokr", "base": os.path.basename(base_path), "target": os.path.basename(target_path),
                  "options": opts.to_dict()}
        tag, tdt = LOKR_DTYPES[opts.dtype]
        writer = StreamWriter(out_path, output_metadata(tpl, out_path, recipe))
        for pl in plans:
            k1, k2, ka = pl.keys
            if tpl.alpha is not None:
                writer.add(ka, tpl.alpha_dtype, [])
            writer.add(k1, tag, list(pl.w1))
            writer.add(k2, tag, list(pl.w2))
        step = max(1, len(plans) // VERIFY_MODULES)
        sample = {pl.canon for pl in plans[::step][:VERIFY_MODULES]}
        kept: dict[str, tuple] = {}
        rep = LokrReport(int(opts.factor), tpl.summary(), prof.label, prof.block_count(ft.reader.infos) or prof.blocks,
                         skipped=skipped, notes=notes)
        alpha_t = None
        if tpl.alpha is not None:
            from .st_io import DTYPES
            alpha_t = torch.tensor(tpl.alpha, dtype=DTYPES[tpl.alpha_dtype])
        writer.begin()

        def on_fit(pl, fit, delta):
            k1, k2, ka = pl.keys
            w1 = fit.w1.to(tdt).cpu().contiguous()
            w2 = fit.w2.to(tdt).cpu().contiguous()
            if alpha_t is not None:
                writer.write(ka, alpha_t)
            writer.write(k1, w1)
            writer.write(k2, w2)
            row = _row(pl, fit)
            if not fit.no_change:
                row.residual = residual_share(delta, w1.float(), w2.float().to(delta.device))
            rep.rows.append(row)
            if pl.canon in sample:
                kept[pl.canon] = (k1, k2, w1.clone(), w2.clone())

        try:
            _fit_all(fb, ft, plans, opts, device, progress, cancel, on_fit)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        _verify_output(out_path, plans, tpl, kept, log)
        rows = [r for r in rep.rows if not r.no_change]
        gap = max((abs((1 - r.captured) - r.residual) for r in rows if r.residual is not None), default=0.0)
        if opts.dtype == "fp32" and gap > 1e-3:
            raise RuntimeError(f"verification: the stored factors miss 1 - captured by {gap:.2e}; the fit is wrong")
        rep.run = {"function": "extract_lokr", "base": os.path.abspath(base_path), "target": os.path.abspath(target_path),
                   "options": opts.to_dict(), "device": str(device), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        log(rep.text())
        return LokrExtractResult(out_path, len(plans), skipped, rep.captured(),
                                 min((r.captured for r in rows), default=1.0), len(rep.rows) - len(rows), gap,
                                 time.time() - t0, os.path.getsize(out_path), tpl.summary(), list(tpl.dropped), rep)
    finally:
        rb.close()
        rt.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _verify_output(out_path: str, plans, tpl: LokrTemplate, kept: dict, log) -> None:
    """Reopens the output: the planned key set, sampled factors byte equal to what was written, and the tool's own
    LoKr reader sees every module as a full rank LoKr applied at scale 1."""
    from .lora import LoraFile
    with TensorReader(out_path) as r:
        want = set()
        for pl in plans:
            k1, k2, ka = pl.keys
            want |= {k1, k2} | ({ka} if tpl.alpha is not None else set())
        if set(r.infos) != want:
            raise RuntimeError("verification: the output's tensor set differs from the plan")
        for k1, k2, w1, w2 in kept.values():
            if not (torch.equal(r.read(k1), w1) and torch.equal(r.read(k2), w2)):
                raise RuntimeError(f"verification: {k1} or {k2} read back differs from what was written")
    with LoraFile(out_path) as lf:
        kinds = lf.kind_counts()
        if kinds.get("lokr", 0) != len(plans) or len(lf.modules) != len(plans):
            raise RuntimeError(f"verification: the LoKr reader finds {kinds} in the output, {len(plans)} modules planned")
        bad = [m.name for m in lf.modules.values() if abs(m.scale - 1.0) > 1e-12]
        if bad:
            raise RuntimeError(f"verification: {len(bad)} module(s) would apply at a scale other than 1, e.g. {bad[:2]}")
    log(f"verify: {len(plans)} modules, key set, {len(kept)} sampled modules read back byte equal, LoKr reader: full rank at scale 1")


def plan_text(base_path: str, target_path: str, opts: LokrExtractOptions) -> str:
    """What an extraction would write: shapes per leaf, parameters, size, peak memory. Headers only."""
    from .plan import _device_line, _human
    opts.validate()
    lines = [_device_line()]
    rb, rt, fb, ft = _open(base_path, target_path)
    try:
        prof = _profile_of(fb, ft)
        tpl = load_template(opts.template, prof)
        lines.append(f"base:   {os.path.basename(base_path)} ({fb.summary()})")
        lines.append(f"target: {os.path.basename(target_path)} ({ft.summary()})")
        lines.append(f"architecture: {prof.label}; template: {tpl.summary()}")
        plans, skipped, notes = plan_modules(fb, ft, opts, tpl)
        lines.append(f"modules: {len(plans)} (filter {opts.filter}); skipped {len(skipped)}")
        for m, why in skipped[:5]:
            lines.append(f"   skipped {m}: {why}")
        lines += notes
        shapes = Counter()
        for pl in plans:
            leaf = re.sub(r"^(blocks|layers)\.\d+\.|^txtfusion\.(\w+_blocks)\.\d+\.", lambda mm: "txtfusion." if mm.group(2) else "", pl.module)
            shapes[(leaf, pl.shape, pl.w1, pl.w2)] += 1
        lines.append(f"factor {opts.factor}, LyCORIS shapes (w1 x w2 = weight):")
        for (leaf, shape, w1, w2), n in sorted(shapes.items(), key=lambda kv: -kv[1])[:16]:
            lines.append(f"   x{n:<3d} {leaf:28s} {list(shape)} = {list(w1)} x {list(w2)}")
        params = sum(pl.w1[0] * pl.w1[1] + pl.w2[0] * pl.w2[1] for pl in plans)
        bpe = {"fp32": 4, "bf16": 2, "fp16": 2}[opts.dtype]
        big = max((pl.shape[0] * pl.shape[1] for pl in plans), default=0)
        lines.append(f"output: {params / 1e6:.1f} M parameters, about {_human(params * bpe)} in {opts.dtype}; "
                     f"w1 scale {opts.w1_scale}")
        lines.append(f"working memory: about {_human(big * 4 * 4)} for the largest module "
                     f"(base, target, delta and its rearranged copy in fp32)")
        if tpl.dropped:
            lines.append(f"template metadata not carried over: {', '.join(tpl.dropped)}")
        if fb.quant or ft.quant:
            lines.append("quantized input: the extracted delta contains quantization noise")
    finally:
        rb.close()
        rt.close()
    return "\n".join(lines)
