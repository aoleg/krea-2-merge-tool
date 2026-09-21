"""Function 1: merge LoRA / LoKr files into one LoRA file.

Exact by concatenation when every input is an A/B LoRA (block shaping and
strength folded into the down matrices). SVD truncation otherwise, or when a
rank is requested.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import torch

from . import TOOL_NAME, __version__
from .analysis import AnalysisReport, analyze_sources, rank_for_energy
from .blocks import Shaping
from .engine import Cancelled, pick_device, recipe_for_metadata
from .keys import KREA2_BLOCKS, block_index, canon, ckpt_module, group_of, strip_prefixes
from .lora import LoraFile, LoraFormatError
from .meta import redact_paths
from .refcheck import load_reference_header
from .sources import LoraSource
from .st_io import StreamWriter, TensorReader, same_file

NAMING = ("comfy", "kohya", "input")
DTYPE_TAG = {"fp16": "F16", "bf16": "BF16", "fp32": "F32"}
DTYPE_TORCH = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class LoraInput:
    path: str
    strength: float = 1.0
    shaping: Shaping = field(default_factory=Shaping)

    def to_dict(self):
        return {"file": self.path, "strength": self.strength, "shaping": self.shaping.to_dict()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["file"], float(d.get("strength", 1.0)), Shaping.from_dict(d.get("shaping")))


@dataclass
class LoraMergeOptions:
    average: bool = False              # normalize strengths to sum one
    rank_mode: str = "concat"          # concat | fixed | groups | dynamic
    rank: int | None = None            # fixed rank
    group_ranks: dict = field(default_factory=dict)   # group -> rank (rank_mode groups)
    retention: float = 0.99            # dynamic: energy every module keeps, the rank is the smallest that reaches it
    rank_cap: int | None = 16          # dynamic: no module above this rank (None = no cap)
    rank_floor: int = 1                # dynamic: no module below this rank
    modules: str = "intersection"      # intersection | union
    naming: str = "comfy"              # comfy | kohya | input
    naming_input: int = 0              # for naming == input
    dtype: str = "fp16"
    write_alpha: bool = True
    keep_metadata: bool = True
    redact_inherited: bool = True      # strip paths out of the metadata inherited from the naming input
    block_count: int = KREA2_BLOCKS
    checkpoint: str | None = None      # optional checkpoint whose keys resolve module names

    def to_dict(self):
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        o = cls()
        for k, v in (d or {}).items():
            if hasattr(o, k):
                setattr(o, k, v)
        return o


def krea2_name_dictionary() -> dict:
    """canon -> bare Krea 2 module name, from the official bf16 header."""
    h = load_reference_header("bf16")
    out = {}
    for k in h:
        if k != "__metadata__" and k.endswith(".weight"):
            m = ckpt_module(k)[0]
            out[canon(m)] = m
    return out


def module_names(sources: list, checkpoint: str | None = None) -> dict:
    """canon -> bare module name for every module of the sources."""
    names = {}
    if checkpoint:
        with TensorReader(checkpoint) as r:
            for k in r.infos:
                if k.endswith(".weight"):
                    m = ckpt_module(k)[0]
                    names[canon(m)] = m
    dictionary = krea2_name_dictionary()
    for s in sources:
        for c, m in s.file.modules.items():
            if c in names:
                continue
            if c in dictionary:
                names[c] = dictionary[c]
            else:
                # dotted conventions strip cleanly; kohya names stay underscored
                names[c] = strip_prefixes(m.name)
    return names


def output_keys(naming: str, name: str, m_ref, suffix_ref: dict | None) -> tuple[str, str, str]:
    if naming == "kohya":
        base = "lora_unet_" + name.replace(".", "_")
        return base + ".lora_down.weight", base + ".lora_up.weight", base + ".alpha"
    if naming == "input" and m_ref is not None and m_ref.kind == "lora":
        return (m_ref.name + m_ref.suffix["down"], m_ref.name + m_ref.suffix["up"], m_ref.name + ".alpha")
    base = "diffusion_model." + name
    return base + ".lora_down.weight", base + ".lora_up.weight", base + ".alpha"


def dynamic_rank(S: torch.Tensor, retention: float, cap: int | None, floor: int) -> int:
    """The pruning rule on one spectrum: the smallest rank whose energy reaches the retention, raised to the floor,
    cut at the cap, never above the number of singular values (0 for an empty spectrum)."""
    n = int(S.numel())
    if n == 0:
        return 0
    s2 = S.double() ** 2
    total = float(s2.sum().item())
    if total <= 0:
        return 0                       # nothing left after shaping: the module is dropped
    r = rank_for_energy(torch.cumsum(s2, 0), total, float(retention))
    r = max(r, max(1, int(floor)))
    if cap:
        r = min(r, int(cap))
    return max(1, min(r, n))


def truncate_factors(down: torch.Tensor, up: torch.Tensor, rank: int | None = None, rank_fn=None) -> tuple[torch.Tensor, torch.Tensor, float]:
    """SVD truncation of up @ down through the small core, to `rank` or to rank_fn(singular values).
    Returns (down2, up2, kept energy fraction)."""
    Qb, Rb = torch.linalg.qr(up)
    Qa, Ra = torch.linalg.qr(down.T)
    core = Rb @ Ra.T
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)
    r = min(rank, S.numel()) if rank is not None else max(0, min(int(rank_fn(S)), int(S.numel())))
    total = float((S ** 2).sum())
    kept = float((S[:r] ** 2).sum() / total) if total > 1e-30 else 1.0
    sq = torch.sqrt(S[:r])
    up2 = (Qb @ U[:, :r]) * sq.unsqueeze(0)
    down2 = sq.unsqueeze(1) * (Vh[:r] @ Qa.T)
    return down2, up2, kept


def factor_delta(delta: torch.Tensor, rank: int | None = None, rank_fn=None) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Full SVD of a materialized delta, truncated to `rank` or to rank_fn(singular values). Returns (down, up, kept)."""
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    r = min(rank, S.numel()) if rank is not None else max(0, min(int(rank_fn(S)), int(S.numel())))
    total = float((S ** 2).sum())
    kept = float((S[:r] ** 2).sum() / total) if total > 1e-30 else 1.0
    sq = torch.sqrt(S[:r])
    return sq.unsqueeze(1) * Vh[:r], U[:, :r] * sq.unsqueeze(0), kept


@dataclass
class LoraMergeResult:
    path: str
    modules: int
    rank_min: int
    rank_max: int
    kept_min: float
    dropped: list
    seconds: float
    analysis: AnalysisReport | None = None
    rank_table: list = field(default_factory=list)   # (name, group, block, rank in, rank out, kept) per module
    size_bytes: int = 0


def open_sources(inputs: list[LoraInput], average: bool, block_count: int) -> list[LoraSource]:
    strengths = [i.strength for i in inputs]
    if average:
        s = sum(strengths)
        if abs(s) > 1e-12:
            strengths = [x / s for x in strengths]
    sources = []
    for inp, st in zip(inputs, strengths):
        lf = LoraFile(inp.path)
        sources.append(LoraSource(lf, strength=st, shaping=inp.shaping, block_count=block_count,
                                  label=os.path.basename(inp.path)))
    return sources


def select_modules(sources: list, mode: str) -> tuple[list, list]:
    sets = [s.modules() for s in sources]
    if mode == "union":
        chosen = set().union(*sets)
    else:
        chosen = set.intersection(*sets) if sets else set()
    dropped = []
    for s, ms in zip(sources, sets):
        for c in sorted(ms - chosen):
            dropped.append((s.label, s.file.modules[c].name))
    return sorted(chosen), dropped


def analyze_lora_merge(inputs: list[LoraInput], opts: LoraMergeOptions, use_gpu=True, progress=None, cancel=None) -> AnalysisReport:
    device = pick_device(use_gpu)
    sources = open_sources(inputs, opts.average, opts.block_count)
    try:
        chosen, dropped = select_modules(sources, opts.modules)
        names = module_names(sources, opts.checkpoint)
        for s in sources:
            s.module_names = names
        rep = analyze_sources(sources, chosen, names, device, requested_rank=opts.rank, progress=progress, cancel=cancel)
        rep.run = {"function": "lora_merge", "inputs": [i.to_dict() for i in inputs], "options": opts.to_dict(),
                   "device": str(device), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        rep.noise_model = {"applied": False, "text": "no noise model: LoRA spectra are exact"}
        if dropped:
            rep.notes.append(f"{len(dropped)} module(s) not present in every input are dropped")
        return rep
    finally:
        for s in sources:
            s.file.close()


def merge_loras(inputs: list[LoraInput], out_path: str, opts: LoraMergeOptions, use_gpu=True,
                progress=None, cancel=None, log=None) -> LoraMergeResult:
    log = log or (lambda s: None)
    if not inputs:
        raise ValueError("no LoRA inputs")
    for inp in inputs:
        if same_file(inp.path, out_path):
            raise ValueError("the output path is one of the input files")
    device = pick_device(use_gpu)
    t0 = time.time()
    sources = open_sources(inputs, opts.average, opts.block_count)
    try:
        chosen, dropped = select_modules(sources, opts.modules)
        if not chosen:
            raise LoraFormatError("no modules in common between the inputs; are these the same architecture?")
        names = module_names(sources, opts.checkpoint)
        for s in sources:
            s.module_names = names
        ref_src = sources[min(opts.naming_input, len(sources) - 1)]
        dynamic = opts.rank_mode == "dynamic"
        rank_fn = (lambda S: dynamic_rank(S, opts.retention, opts.rank_cap, opts.rank_floor)) if dynamic else None
        table = []

        def rank_for(c: str) -> int | None:
            if opts.rank_mode == "fixed":
                return int(opts.rank) if opts.rank else None
            if opts.rank_mode == "groups":
                from .keys import group_of
                g = group_of(names[c])
                r = opts.group_ranks.get(g, opts.group_ranks.get("*"))
                return int(r) if r else None
            return None

        # pass 1: compute factors per module (small tensors, kept on CPU until written)
        results: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        total = len(chosen)
        for i, c in enumerate(chosen):
            if cancel is not None and cancel():
                raise Cancelled("cancelled by the user")
            if progress is not None:
                progress(i, total, names[c])
            contributing = [s for s in sources if c in s.modules()]
            exact = all(s.is_exact(c) for s in contributing)
            r = rank_for(c)
            if exact:
                downs, ups = [], []
                for s in contributing:
                    d, u = s.low_rank(c, device)
                    downs.append(d)
                    ups.append(u)
                down, up = torch.cat(downs, 0), torch.cat(ups, 1)
                rank_in = int(down.shape[0])
                kept = 1.0
                if dynamic:
                    down, up, kept = truncate_factors(down, up, rank_fn=rank_fn)
                elif r is not None and r < down.shape[0]:
                    down, up, kept = truncate_factors(down, up, r)
            else:
                delta = None
                for s in contributing:
                    d = s.delta(c, device)
                    delta = d if delta is None else delta + d
                rank_in = int(min(delta.shape))
                if dynamic:
                    down, up, kept = factor_delta(delta, rank_fn=rank_fn)
                else:
                    rr = r if r is not None else max(1, min(delta.shape) // 8)
                    if r is None:
                        log(f"{names[c]}: LoKr or full delta without a rank; using rank {rr}")
                    down, up, kept = factor_delta(delta, rr)
                del delta
            if dynamic and down.shape[0] == 0:
                dropped.append(("zero after shaping", names[c]))
                table.append((names[c], group_of(names[c]), block_index(names[c]), rank_in, 0, 1.0))
                continue
            results[c] = (down.detach().cpu().contiguous(), up.detach().cpu().contiguous(), kept)
            table.append((names[c], group_of(names[c]), block_index(names[c]), rank_in, int(down.shape[0]), kept))
        if not results:
            raise LoraFormatError("nothing left to write: every module is zero after shaping")
        written = [c for c in chosen if c in results]

        # pass 2: plan + write
        ranks = [v[0].shape[0] for v in results.values()]
        kept_min = min(v[2] for v in results.values())
        meta = {}
        if opts.keep_metadata:
            inherited = {k: str(v) for k, v in ref_src.file.metadata.items()}
            meta.update(redact_paths(inherited) if opts.redact_inherited else inherited)
        recipe = {"function": "lora_merge", "inputs": [i.to_dict() for i in inputs], "options": opts.to_dict()}
        meta.update({"merge_tool": f"{TOOL_NAME} {__version__}", "merge_recipe": json.dumps(recipe_for_metadata(recipe), separators=(",", ":")),
                     "merge_output_rank": f"{min(ranks)}-{max(ranks)}" if min(ranks) != max(ranks) else str(ranks[0])})
        tag, tdt = DTYPE_TAG[opts.dtype], DTYPE_TORCH[opts.dtype]
        writer = StreamWriter(out_path, meta)
        keys_out = {}
        for c in written:
            m_ref = ref_src.file.modules.get(c)
            kd, ku, ka = output_keys(opts.naming, names[c], m_ref, None)
            down, up, _ = results[c]
            keys_out[c] = (kd, ku, ka)
            writer.add(kd, tag, down.shape)
            writer.add(ku, tag, up.shape)
            if opts.write_alpha:
                writer.add(ka, "F32", [])
        writer.begin()
        try:
            for c in written:
                kd, ku, ka = keys_out[c]
                down, up, _ = results[c]
                writer.write(kd, down.to(tdt))
                writer.write(ku, up.to(tdt))
                if opts.write_alpha:
                    writer.write(ka, torch.tensor(float(down.shape[0]), dtype=torch.float32))
            writer.close()
        except BaseException:
            writer.abort()
            raise
        return LoraMergeResult(out_path, len(written), min(ranks), max(ranks), kept_min, dropped, time.time() - t0,
                               rank_table=table, size_bytes=os.path.getsize(out_path))
    finally:
        for s in sources:
            s.file.close()


# ----------------------------------------------------------------------------- pruning plan
_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}


def _agg(mods: list, total_energy: float) -> dict:
    e = sum(m["energy"] for m in mods)
    kept = (sum(m["kept"] * m["energy"] for m in mods) / e) if e > 0 else 1.0
    return {"modules": len(mods), "energy_share": (e / total_energy) if total_energy > 0 else 0.0,
            "rank_in": sum(m["rank_in"] for m in mods), "rank": sum(m["rank"] for m in mods), "kept": kept,
            "size_in": sum(m["size_in"] for m in mods), "size": sum(m["size"] for m in mods),
            "capped": sum(1 for m in mods if m["capped"])}


def prune_plan(rep: AnalysisReport, retention: float, cap: int | None, floor: int, dtype: str = "fp16") -> dict:
    """The dynamic rank rule applied to every module of an analysis: rank in and out, the energy each keeps,
    whether the cap bound, and the tensor bytes before and after (down, up and one fp32 alpha per module).
    Totals, per group and per block. Computed from the spectra alone, so a setting change needs no new pass."""
    b = _BYTES[dtype]
    mods = []
    for ms in rep.modules.values():
        n_in = int(ms.sv.numel())
        dims = int(ms.shape[0] + ms.shape[1])
        r = dynamic_rank(ms.sv, retention, cap, floor) if n_in else 0      # 0 = dropped (nothing left after shaping)
        need = ms.rank_needed(retention) if r else 0
        mods.append({"name": ms.name, "group": ms.group, "block": ms.block, "rank_in": n_in, "rank": r,
                     "kept": ms.energy_at(r) if r else 1.0, "capped": bool(cap) and need > r, "energy": ms.energy,
                     "size_in": n_in * dims * b + 4, "size": (r * dims * b + 4) if r else 0})
    total_energy = sum(m["energy"] for m in mods)
    groups = {g: _agg([m for m in mods if m["group"] == g], total_energy) for g in rep.groups}
    blocks = {}
    for blk in sorted({m["block"] for m in mods}, key=lambda x: (x is None, x if x is not None else 0)):
        blocks[blk] = _agg([m for m in mods if m["block"] == blk], total_energy)
    ranks_in = [m["rank_in"] for m in mods] or [0]
    ranks = [m["rank"] for m in mods] or [0]
    return {"retention": float(retention), "cap": cap, "floor": int(floor), "dtype": dtype, "modules": mods,
            "groups": groups, "blocks": blocks, "total": _agg(mods, total_energy),
            "rank_in_min": min(ranks_in), "rank_in_max": max(ranks_in), "rank_min": min(ranks), "rank_max": max(ranks)}


def prune_text(p: dict) -> str:
    mb = lambda n: f"{n / 1e6:7.1f} MB"  # noqa: E731
    lines = [f"prune plan: retention {p['retention']:.3f}, cap {p['cap'] if p['cap'] else 'none'}, floor {p['floor']}, {p['dtype']}",
             f"  ranks {p['rank_in_min']}-{p['rank_in_max']} in, {p['rank_min']}-{p['rank_max']} out",
             f"  {'scope':14s} {'modules':>7s} {'energy':>7s} {'rank in':>8s} {'out':>5s} {'kept':>8s} {'size in':>10s} {'out':>10s} {'capped':>6s}"]

    def line(label, a):
        lines.append(f"  {label:14s} {a['modules']:7d} {100 * a['energy_share']:6.1f}% {a['rank_in']:8d} {a['rank']:5d} "
                     f"{100 * a['kept']:7.2f}% {mb(a['size_in'])} {mb(a['size'])} {a['capped']:6d}")
    line("all", p["total"])
    for g, a in p["groups"].items():
        line(g, a)
    for blk, a in p["blocks"].items():
        line(f"block {blk}" if blk is not None else "non block", a)
    return "\n".join(lines)
