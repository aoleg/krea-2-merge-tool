"""Plan: describe what a recipe would do without writing anything."""
from __future__ import annotations

import os

import torch

from .ckpt_merge import CkptInput, CkptMergeOptions, _open_context, _close_context, _touched_keys
from .engine import plan_from_primary
from .extract import ExtractOptions, _open, _selected_modules
from .formats import FileFormat
from .lora_merge import LoraInput, LoraMergeOptions, module_names, open_sources, select_modules
from .st_io import ELEMENT_SIZE, TensorReader
from . import methods


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _device_line() -> str:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return f"device: {torch.cuda.get_device_name(0)}, {_human(free)} free of {_human(total)} VRAM"
    return "device: CPU (no CUDA); SVD analysis and quantization will be slow"


def plan_lora_merge(inputs: list[LoraInput], opts: LoraMergeOptions) -> str:
    lines = [_device_line()]
    sources = open_sources(inputs, opts.average, opts.block_count)
    try:
        for s in sources:
            lines.append(f"input {s.label}: {s.file.summary()}; effective strength {s.strength:g}"
                         + ("" if s.shaping.is_flat() else f"; shaping {s.shaping.to_dict()}"))
        chosen, dropped = select_modules(sources, opts.modules)
        names = module_names(sources, opts.checkpoint)
        exact = all(s.is_exact(c) for s in sources for c in chosen if c in s.modules())
        lines.append(f"modules: {len(chosen)} ({opts.modules}), {len(dropped)} dropped")
        if exact and opts.rank_mode == "concat":
            ranks = [sum(s.file.modules[c].rank or 0 for s in sources if c in s.modules()) for c in chosen]
            lines.append(f"exact concatenation, output rank {min(ranks)}-{max(ranks)}, no energy loss")
        elif exact:
            lines.append(f"exact concatenation then SVD truncation ({opts.rank_mode}); run Analyze for the energy retained")
        else:
            lines.append("a LoKr or full delta input forces materialization and SVD (lossy); run Analyze for the rank")
        n = len(chosen)
        lines.append(f"output: {n} modules, {opts.naming} naming, {opts.dtype}")
        unknown = [names[c] for c in chosen if "." not in names[c] and "_" in names[c]]
        if unknown:
            lines.append(f"warning: {len(unknown)} kohya module names could not be mapped to Krea 2 names (e.g. {unknown[:2]})")
    finally:
        for s in sources:
            s.file.close()
    return "\n".join(lines)


def plan_extract(base: str, target: str, opts: ExtractOptions) -> str:
    lines = [_device_line()]
    rb, rt, fb, ft = _open(base, target)
    try:
        lines.append(f"base:   {os.path.basename(base)} ({fb.summary()})")
        lines.append(f"target: {os.path.basename(target)} ({ft.summary()})")
        mods, skipped = _selected_modules(fb, ft, opts)
        lines.append(f"modules selected: {len(mods)} (filter {opts.filter}); skipped {len(skipped)}")
        for m, why in skipped[:5]:
            lines.append(f"   skipped {m}: {why}")
        big = max((m[4][0] * m[4][1] for m in mods), default=0)
        lines.append(f"largest module {_human(big * 4)} in fp32; SVD method {opts.method}, rank {opts.rank}"
                     + (f", per group {opts.group_ranks}" if opts.group_ranks else ""))
        if fb.quant or ft.quant:
            lines.append("quantized input: the extracted delta contains quantization noise; run Analyze to see the noise floor")
        params = sum(min(opts.rank, min(m[4])) * (m[4][0] + m[4][1]) for m in mods)
        lines.append(f"output: about {_human(params * {'fp16': 2, 'bf16': 2, 'fp32': 4}[opts.dtype])}, {opts.naming} naming, {opts.dtype}")
    finally:
        rb.close()
        rt.close()
    return "\n".join(lines)


def plan_ckpt_merge(A: CkptInput, B: CkptInput | None, C: CkptInput | None, loras: list[LoraInput],
                    opts: CkptMergeOptions) -> str:
    lines = [_device_line()]
    notes = []
    ctx = _open_context(A, B, C, loras, opts, notes.append)
    try:
        lines.append(f"A: {os.path.basename(A.path)} ({ctx.A.fmt.summary()})")
        if ctx.B:
            lines.append(f"B: {os.path.basename(B.path)} ({ctx.B.fmt.summary()}), weight {B.weight:g}"
                         + ("" if B.shaping.is_flat() else f", shaping {B.shaping.to_dict()}"))
        if ctx.C:
            lines.append(f"C: {os.path.basename(C.path)} ({ctx.C.fmt.summary()})")
        lines.append(f"method: {methods.METHOD_LABELS.get(opts.method, opts.method)}")
        if opts.method in methods.NEEDS_C and ctx.C is None and ctx.B is not None:
            lines.append("no reference: C = A for this run")
        for s in ctx.loras:
            lines.append(f"LoRA {s.label}: {s.file.summary()}; strength {s.strength:g}"
                         + ("" if s.shaping.is_flat() else f"; shaping {s.shaping.to_dict()}"))
        for w in ctx.warnings + notes:
            lines.append("warning: " + w)
        touched = _touched_keys(ctx)
        if ctx.only_in_a:
            lines.append(f"{len(ctx.only_in_a)} tensor(s) only in A are copied from A")
        if ctx.only_in_b:
            lines.append(f"{len(ctx.only_in_b)} tensor(s) only in B are ignored")
        if opts.vectors_from != "merge":
            nv = sum(1 for k in ctx.A.reader.infos if not ctx.A.fmt.is_consumed(k) and len(ctx.A.reader.shape(k)) != 2)
            lines.append(f"norm scales, modulation vectors and biases ({nv} tensors) copied from {opts.vectors_from}, not merged")
        if opts.output_as_lora:
            n2 = sum(1 for k in touched if k.endswith(".weight") and len(ctx.A.reader.shape(k)) == 2)
            lines.append(f"output as LoRA: {n2} changed 2-D weights, rank {opts.lora_out.get('rank')}, {opts.lora_out.get('naming')} naming")
        else:
            plan = plan_from_primary(ctx.A.fmt, opts.output_format, opts.passthrough, opts.fp8_layer_set,
                                     compute_for=lambda k: (lambda: None) if k in touched else None, touched_keys=touched)
            summ = plan.summary()
            size = sum(ELEMENT_SIZE[it.dtype_tag] * (torch.tensor(it.shape).prod().item() if it.shape else 1) for it in plan.items)
            lines.append(f"output {opts.output_format}: {summ['tensors']} tensors, {summ['touched']} recomputed, "
                         f"{summ['kinds'].get('raw', 0)} copied raw; about {_human(size)}")
            if opts.output_format in ("fp8", "fp8_scaled"):
                lines.append("fp8 plain loses small weights; fp8_scaled is the official layout" if opts.output_format == "fp8" else
                             f"fp8 scaled: {len(plan.quant_layers)} quantized layers ({opts.fp8_layer_set} set)")
        big = max((torch.tensor(ctx.A.reader.shape(k)).prod().item() for k in ctx.A.reader.infos), default=0)
        n_in = 1 + (1 if ctx.B else 0) + (1 if ctx.C else 0) + len(ctx.loras)
        lines.append(f"VRAM estimate: about {_human(big * 4 * (n_in + 3))} peak for the largest tensor")
        if ctx.B and opts.method in methods.TWO_PASS:
            lines.append("cosine mode: two passes over A and B (similarity pass first)")
    finally:
        _close_context(ctx)
    return "\n".join(lines)


def plan_recipe(r: dict, base_dir: str | None = None) -> str:
    """Plan text for any recipe document (see recipe.py)."""
    from .recipe import _resolve, validate
    validate(r)
    fn = r["function"]
    if fn == "lora_merge":
        inputs = [LoraInput.from_dict(d) for d in r["inputs"]]
        for i in inputs:
            i.path = _resolve(i.path, base_dir)
        return plan_lora_merge(inputs, LoraMergeOptions.from_dict(r.get("options")))
    if fn == "extract":
        return plan_extract(_resolve(r["base"], base_dir), _resolve(r["target"], base_dir), ExtractOptions.from_dict(r.get("options")))
    if fn == "ckpt_merge":
        A = CkptInput.from_dict(r["A"])
        B = CkptInput.from_dict(r.get("B"))
        C = CkptInput.from_dict(r.get("C"))
        for x in (A, B, C):
            if x is not None:
                x.path = _resolve(x.path, base_dir)
        loras = [LoraInput.from_dict(d) for d in r.get("loras", [])]
        for l in loras:
            l.path = _resolve(l.path, base_dir)
        return plan_ckpt_merge(A, B, C, loras, CkptMergeOptions.from_dict(r.get("options")))
    if fn == "convert":
        src = _resolve(r["inputs"][0]["file"], base_dir)
        return plan_ckpt_merge(CkptInput(src), None, None, [],
                               CkptMergeOptions(output_format=r.get("output_format", "bf16"),
                                                passthrough=r.get("passthrough", "official"),
                                                fp8_layer_set=r.get("fp8_layer_set", "official")))
    raise ValueError(fn)
