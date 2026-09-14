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
from .analysis import AnalysisReport, analyze_sources
from .blocks import Shaping
from .engine import Cancelled, pick_device
from .keys import KREA2_BLOCKS, canon, ckpt_module, strip_prefixes
from .lora import LoraFile, LoraFormatError
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
    rank_mode: str = "concat"          # concat | fixed | groups
    rank: int | None = None            # fixed rank
    group_ranks: dict = field(default_factory=dict)   # group -> rank (rank_mode groups)
    modules: str = "intersection"      # intersection | union
    naming: str = "comfy"              # comfy | kohya | input
    naming_input: int = 0              # for naming == input
    dtype: str = "fp16"
    write_alpha: bool = True
    keep_metadata: bool = True
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


def truncate_factors(down: torch.Tensor, up: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """SVD truncation of up @ down through the small core. Returns (down2, up2, kept energy fraction)."""
    Qb, Rb = torch.linalg.qr(up)
    Qa, Ra = torch.linalg.qr(down.T)
    core = Rb @ Ra.T
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)
    r = min(rank, S.numel())
    total = float((S ** 2).sum())
    kept = float((S[:r] ** 2).sum() / total) if total > 1e-30 else 1.0
    sq = torch.sqrt(S[:r])
    up2 = (Qb @ U[:, :r]) * sq.unsqueeze(0)
    down2 = sq.unsqueeze(1) * (Vh[:r] @ Qa.T)
    return down2, up2, kept


def factor_delta(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Full SVD of a materialized delta, truncated to rank. Returns (down, up, kept)."""
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    r = min(rank, S.numel())
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
                kept = 1.0
                if r is not None and r < down.shape[0]:
                    down, up, kept = truncate_factors(down, up, r)
            else:
                delta = None
                for s in contributing:
                    d = s.delta(c, device)
                    delta = d if delta is None else delta + d
                rr = r if r is not None else max(1, min(delta.shape) // 8)
                if r is None:
                    log(f"{names[c]}: LoKr or full delta without a rank; using rank {rr}")
                down, up, kept = factor_delta(delta, rr)
                del delta
            results[c] = (down.detach().cpu().contiguous(), up.detach().cpu().contiguous(), kept)

        # pass 2: plan + write
        ranks = [v[0].shape[0] for v in results.values()]
        kept_min = min(v[2] for v in results.values())
        meta = {}
        if opts.keep_metadata:
            meta.update({k: str(v) for k, v in ref_src.file.metadata.items()})
        recipe = {"function": "lora_merge", "inputs": [i.to_dict() for i in inputs], "options": opts.to_dict()}
        meta.update({"merge_tool": f"{TOOL_NAME} {__version__}", "merge_recipe": json.dumps(recipe, separators=(",", ":")),
                     "merge_output_rank": f"{min(ranks)}-{max(ranks)}" if min(ranks) != max(ranks) else str(ranks[0])})
        tag, tdt = DTYPE_TAG[opts.dtype], DTYPE_TORCH[opts.dtype]
        writer = StreamWriter(out_path, meta)
        keys_out = {}
        for c in chosen:
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
            for c in chosen:
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
        return LoraMergeResult(out_path, len(chosen), min(ranks), max(ranks), kept_min, dropped, time.time() - t0)
    finally:
        for s in sources:
            s.file.close()
