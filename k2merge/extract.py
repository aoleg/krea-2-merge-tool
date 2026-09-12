"""Function 3: extract a LoRA from the difference between two checkpoints.

Streams one module at a time: read both weights (dequantized), SVD the
difference on the device, write the factors immediately. Output shapes are
known from the checkpoint header, so the file is written in one pass.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import torch

from . import TOOL_NAME, __version__
from .analysis import AnalysisReport, analyze_sources, quantization_noise_energy
from .engine import Cancelled, pick_device
from .formats import FileFormat
from .keys import KREA2_BLOCKS, canon, ckpt_module, group_of, in_int8_recipe
from .lora_merge import DTYPE_TAG, DTYPE_TORCH, output_keys
from .sources import CheckpointDelta
from .st_io import StreamWriter, TensorReader, same_file

FILTER_PRESETS = ("all", "attn", "blocks", "custom")
_RE_ATTN = re.compile(r"^blocks\.\d+\.attn\.")


@dataclass
class ExtractOptions:
    rank: int = 32
    group_ranks: dict = field(default_factory=dict)   # group -> rank; overrides rank when present
    filter: str = "all"               # all | attn | blocks | custom
    include: str = ""                 # regex on bare module names (custom)
    exclude: str = ""                 # regex on bare module names (custom)
    naming: str = "comfy"             # comfy | kohya
    dtype: str = "fp16"
    method: str = "randomized"        # randomized | full
    oversample: int = 16
    niter: int = 6
    write_alpha: bool = True
    keep_metadata: bool = False
    block_count: int = KREA2_BLOCKS

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        o = cls()
        for k, v in (d or {}).items():
            if hasattr(o, k):
                setattr(o, k, v)
        return o


def module_selected(module: str, opts: ExtractOptions) -> bool:
    if opts.filter == "attn":
        return bool(_RE_ATTN.match(module))
    if opts.filter == "blocks":
        return in_int8_recipe(module)
    if opts.filter == "custom":
        if opts.exclude and re.search(opts.exclude, module):
            if not (opts.include and re.search(opts.include, module)):
                return False
        if opts.include and not re.search(opts.include, module):
            return False
        return True
    return True


@dataclass
class ExtractResult:
    path: str
    modules: int
    skipped: list
    clamped: list
    kept_min: float
    kept_by_group: dict
    seconds: float


def _open(base_path, target_path):
    rb, rt = TensorReader(base_path), TensorReader(target_path)
    return rb, rt, FileFormat(rb), FileFormat(rt)


def _selected_modules(fb: FileFormat, ft: FileFormat, opts: ExtractOptions):
    """[(canon, bare module, target key, base key, shape)] for common 2-D weights passing the filter."""
    base_mods = {canon(ckpt_module(k)[0]): k for k in fb.reader.infos if k.endswith(".weight")}
    out, skipped = [], []
    for k in ft.reader.names:
        if not k.endswith(".weight"):
            continue
        module = ckpt_module(k)[0]
        c = canon(module)
        shape = ft.reader.shape(k)
        if len(shape) != 2:
            continue
        if c not in base_mods:
            skipped.append((module, "not in base"))
            continue
        if list(fb.reader.shape(base_mods[c])) != list(shape):
            skipped.append((module, f"shape {fb.reader.shape(base_mods[c])} vs {shape}"))
            continue
        if not module_selected(module, opts):
            continue
        out.append((c, module, k, base_mods[c], shape))
    return out, skipped


def _noise_map(fb: FileFormat, ft: FileFormat, mods, device, cancel=None) -> dict:
    noise = {}
    for c, module, tk, bk, shape in mods:
        if cancel is not None and cancel():
            raise Cancelled("cancelled by the user")
        lt, lb = ft.layout_of(tk), fb.layout_of(bk)
        if lt == "plain" and lb == "plain":
            continue
        wn = fb.read_fp32(bk, device=device).norm().item()
        noise[c] = quantization_noise_energy(wn, [lt, lb])
    return noise


def analyze_extract(base_path: str, target_path: str, opts: ExtractOptions, use_gpu=True, progress=None, cancel=None) -> AnalysisReport:
    device = pick_device(use_gpu)
    rb, rt, fb, ft = _open(base_path, target_path)
    try:
        mods, skipped = _selected_modules(fb, ft, opts)
        src = CheckpointDelta(ft, fb, weight=1.0, block_count=opts.block_count)
        names = {c: module for c, module, *_ in mods}
        noise = _noise_map(fb, ft, mods, device, cancel)
        rep = analyze_sources([src], [m[0] for m in mods], names, device, noise=noise,
                              requested_rank=opts.rank, progress=progress, cancel=cancel)
        if skipped:
            rep.notes.append(f"{len(skipped)} tensor(s) skipped: " + ", ".join(f"{m} ({why})" for m, why in skipped[:5]))
        if noise:
            rep.notes.append("quantized input: the noise floor is drawn from the storage format's expected error")
        return rep
    finally:
        rb.close()
        rt.close()


def _svd(delta: torch.Tensor, r: int, opts: ExtractOptions):
    """Returns (down, up, kept). A delta with no energy (block factor 0, untouched module)
    yields zero factors and kept = 1.0: there was nothing to lose."""
    total = float((delta ** 2).sum().item())
    if total <= 1e-30:
        return (torch.zeros(r, delta.shape[1], device=delta.device), torch.zeros(delta.shape[0], r, device=delta.device), 1.0)
    if opts.method == "full" or r >= min(delta.shape) // 2:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        kept = float((S[:r] ** 2).sum() / max((S ** 2).sum(), 1e-30))
        sq = torch.sqrt(S[:r])
        return sq.unsqueeze(1) * Vh[:r], U[:, :r] * sq.unsqueeze(0), kept
    q = min(r + opts.oversample, min(delta.shape))
    U, S, V = torch.svd_lowrank(delta, q=q, niter=opts.niter)
    kept = float((S[:r] ** 2).sum().item() / max(total, 1e-30))
    sq = torch.sqrt(S[:r])
    return (V[:, :r] * sq).T.contiguous(), (U[:, :r] * sq).contiguous(), kept


def extract_lora(base_path: str, target_path: str, out_path: str, opts: ExtractOptions, use_gpu=True,
                 progress=None, cancel=None, log=None) -> ExtractResult:
    log = log or (lambda s: None)
    for p in (base_path, target_path):
        if same_file(p, out_path):
            raise ValueError("the output path is one of the input files")
    device = pick_device(use_gpu)
    t0 = time.time()
    rb, rt, fb, ft = _open(base_path, target_path)
    try:
        mods, skipped = _selected_modules(fb, ft, opts)
        if not mods:
            raise ValueError("no common 2-D weights pass the module filter")

        def rank_for(module: str, shape) -> int:
            r = opts.group_ranks.get(group_of(module), opts.group_ranks.get("*", opts.rank))
            return max(1, min(int(r), min(shape)))

        meta = {}
        if opts.keep_metadata:
            meta.update({k: str(v) for k, v in rt.metadata.items() if k != "_quantization_metadata"})
        recipe = {"function": "extract", "base": os.path.basename(base_path), "target": os.path.basename(target_path),
                  "options": opts.to_dict()}
        meta.update({"merge_tool": f"{TOOL_NAME} {__version__}", "merge_recipe": json.dumps(recipe, separators=(",", ":"))})
        tag, tdt = DTYPE_TAG[opts.dtype], DTYPE_TORCH[opts.dtype]
        writer = StreamWriter(out_path, meta)
        plan = []
        clamped = []
        for c, module, tk, bk, shape in mods:
            r = rank_for(module, shape)
            if r < int(opts.group_ranks.get(group_of(module), opts.group_ranks.get("*", opts.rank))):
                clamped.append((module, r))
            kd, ku, ka = output_keys(opts.naming, module, None, None)
            writer.add(kd, tag, [r, shape[1]])
            writer.add(ku, tag, [shape[0], r])
            if opts.write_alpha:
                writer.add(ka, "F32", [])
            plan.append((c, module, tk, bk, shape, r, kd, ku, ka))
        writer.begin()
        kept_min = 1.0
        kept_group: dict[str, list] = {}
        try:
            total = len(plan)
            for i, (c, module, tk, bk, shape, r, kd, ku, ka) in enumerate(plan):
                if cancel is not None and cancel():
                    raise Cancelled("cancelled by the user")
                if progress is not None:
                    progress(i, total, module)
                delta = ft.read_fp32(tk, device=device) - fb.read_fp32(bk, device=device)
                down, up, kept = _svd(delta, r, opts)
                del delta
                kept_min = min(kept_min, kept)
                kept_group.setdefault(group_of(module), []).append(kept)
                writer.write(kd, down.to(tdt).cpu())
                writer.write(ku, up.to(tdt).cpu())
                if opts.write_alpha:
                    writer.write(ka, torch.tensor(float(r), dtype=torch.float32))
                del down, up
            if progress is not None:
                progress(total, total, "closing file")
            writer.close()
        except BaseException:
            writer.abort()
            raise
        kept_by_group = {g: min(v) for g, v in kept_group.items()}
        return ExtractResult(out_path, len(plan), skipped, clamped, kept_min, kept_by_group, time.time() - t0)
    finally:
        rb.close()
        rt.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()
