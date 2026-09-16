"""Function 4: merge one to three checkpoints and zero to four LoRAs, or convert.

out = Method(A, B, C) + sum_i strength_i * blockfactor_i(module) * delta_i
Output written through the streaming engine, or as a LoRA (delta against the
reference) through the extraction writer.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import torch

from . import methods
from .analysis import AnalysisReport, analyze_sources, quantization_noise_energy
from .blocks import Shaping
from .engine import (Cancelled, RunResult, build_metadata, execute_plan, pick_device, recipe_for_metadata,
                     plan_from_primary)
from .extract import ExtractOptions, _svd
from .formats import FLOAT_TAGS, FileFormat, FormatError
from .keys import KREA2_BLOCKS, block_index_any, canon, ckpt_module, group_of
from .lora import LoraFile, checkpoint_modules, resolve
from .lora_merge import DTYPE_TAG, DTYPE_TORCH, LoraInput, output_keys
from .sources import LoraSource
from .st_io import StreamWriter, TensorReader, same_file


@dataclass
class CkptInput:
    path: str
    weight: float = 1.0
    shaping: Shaping = field(default_factory=Shaping)   # slot B only

    def to_dict(self):
        return {"file": self.path, "weight": self.weight, "shaping": self.shaping.to_dict()}

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        return cls(d["file"], float(d.get("weight", 1.0)), Shaping.from_dict(d.get("shaping")))


VECTOR_SOURCES = ("merge", "A", "B", "C")   # where the non 2-D tensors (norm scales, modulation vectors, biases) come from


@dataclass
class CkptMergeOptions:
    method: str = "add_difference"
    params: dict = field(default_factory=lambda: {"density": 0.2, "lambda": 1.0, "p": 0.5, "seed": 0,
                                                  "beta": 0.0, "gamma": 1.0, "dare_ties": False})
    output_format: str = "bf16"
    passthrough: str = "official"
    fp8_layer_set: str = "official"
    int8_clip: str = "mse"            # mse (reproduces the official int8 file) | absmax
    vectors_from: str = "merge"       # merge | A | B | C: norm scales, modulation vectors and biases merged like the rest, or copied from one input
    keep_metadata: bool = True
    redact_inherited: bool = True     # strip paths out of the metadata inherited from A (a trainer's dataset folders, an upstream recipe)
    lora_mode: str = "after"          # after | task_vectors (TIES / DARE)
    output_as_lora: bool = False
    lora_out: dict = field(default_factory=lambda: {"rank": 32, "naming": "comfy", "dtype": "fp16", "method": "randomized"})
    block_count: int = KREA2_BLOCKS
    use_gpu: bool = True

    def to_dict(self):
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        o = cls()
        for k, v in (d or {}).items():
            if hasattr(o, k):
                setattr(o, k, dict(v) if isinstance(v, dict) else v)
        return o


class _Ckpt:
    """One open checkpoint with a lookup by bare (module, suffix)."""

    def __init__(self, path: str):
        self.path = path
        self.reader = TensorReader(path)
        self.fmt = FileFormat(self.reader)
        self.by_module: dict[tuple, str] = {}
        for k in self.reader.infos:
            if self.fmt.is_consumed(k):
                continue
            m, s = ckpt_module(k)
            self.by_module[(canon(m), s)] = k

    def key_for(self, a_key: str) -> str | None:
        m, s = ckpt_module(a_key)
        return self.by_module.get((canon(m), s))

    def close(self):
        self.reader.close()


def _block_factor(shaping: Shaping, module: str, count: int) -> float:
    return shaping.factor_for(block_index_any(module), count)


@dataclass
class MergeContext:
    A: _Ckpt
    B: _Ckpt | None
    C: _Ckpt | None
    loras: list                      # LoraSource
    lora_keys: dict                  # A weight key -> [LoraSource...]
    opts: CkptMergeOptions
    device: torch.device
    cosine: methods.CosineStats | None = None
    only_in_a: list = field(default_factory=list)
    only_in_b: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _open_context(A: CkptInput, B: CkptInput | None, C: CkptInput | None, loras: list[LoraInput],
                  opts: CkptMergeOptions, log) -> MergeContext:
    device = pick_device(opts.use_gpu)
    a = _Ckpt(A.path)
    b = _Ckpt(B.path) if B else None
    c = _Ckpt(C.path) if C else None
    if opts.method in methods.NEEDS_C and c is None and b is not None:
        log(f"{opts.method}: no reference given, using A as the reference")
    ctx = MergeContext(a, b, c, [], {}, opts, device)
    if opts.vectors_from not in VECTOR_SOURCES:
        raise FormatError(f"vectors from {opts.vectors_from!r}: choose one of {', '.join(VECTOR_SOURCES)}")
    if _vector_source(ctx) is None:
        raise FormatError(f"vectors from {opts.vectors_from}: no {opts.vectors_from} input in this run")

    if b is not None:
        for k in a.reader.infos:
            if a.fmt.is_consumed(k):
                continue
            if b.key_for(k) is None:
                ctx.only_in_a.append(k)
        a_set = {(canon(ckpt_module(k)[0]), ckpt_module(k)[1]) for k in a.reader.infos if not a.fmt.is_consumed(k)}
        ctx.only_in_b = [k for (m, s), k in b.by_module.items() if (m, s) not in a_set]
        if opts.method in methods.TWO_PASS:
            la, lb = a.fmt.layout_counts(), b.fmt.layout_counts()
            if bool(la) != bool(lb) or set(la) != set(lb):
                ctx.warnings.append("cosine modes with inputs of different storage precision mix on quantization noise; "
                                    "use the same precision (bf16) for A and B")

    # LoRAs
    mods = checkpoint_modules(a.reader.infos)
    shapes = {k: v["shape"] for k, v in a.reader.infos.items()}
    for li in loras:
        lf = LoraFile(li.path)
        res = resolve(lf, mods, shapes)
        src = LoraSource(lf, strength=li.strength, shaping=li.shaping, block_count=opts.block_count,
                         label=os.path.basename(li.path))
        src.module_names = {cn: ckpt_module(k)[0] for cn, k in res.matched.items()}
        if res.unmatched:
            ctx.warnings.append(f"{src.label}: {len(res.unmatched)} module(s) not matched (e.g. {res.unmatched[:2]})")
        if res.mismatched:
            ctx.warnings.append(f"{src.label}: {len(res.mismatched)} module(s) with a shape mismatch, skipped")
        for cn, key in res.matched.items():
            ctx.lora_keys.setdefault(key, []).append((src, cn))
        ctx.loras.append(src)
    return ctx


def _close_context(ctx: MergeContext):
    for s in ctx.loras:
        s.file.close()
    ctx.A.close()
    if ctx.B:
        ctx.B.close()
    if ctx.C:
        ctx.C.close()


def _cosine_prepass(ctx: MergeContext, progress=None, cancel=None):
    stats = methods.CosineStats(ctx.opts.method)
    keys = [k for k in ctx.A.reader.names if not ctx.A.fmt.is_consumed(k)
            and (ctx.A.reader.dtype(k) in FLOAT_TAGS or ctx.A.fmt.is_quantized(k)) and ctx.B.key_for(k)]
    for i, k in enumerate(keys):
        if cancel is not None and cancel():
            raise Cancelled("cancelled by the user")
        if progress is not None:
            progress(i, len(keys), "similarity pass: " + k)
        a = ctx.A.fmt.read_fp32(k, device=ctx.device)
        b = ctx.B.fmt.read_fp32(ctx.B.key_for(k), device=ctx.device)
        if a.shape == b.shape:
            stats.add(a, b)
    ctx.cosine = stats.finish()


def _vector_source(ctx: MergeContext):
    """The checkpoint the non 2-D tensors are copied from, None when that input is missing, or the string
    'merge' when they are merged like the rest."""
    v = ctx.opts.vectors_from
    if v == "merge":
        return "merge"
    return {"A": ctx.A, "B": ctx.B, "C": ctx.C}[v]


def _is_vector(ctx: MergeContext, key: str) -> bool:
    return len(ctx.A.reader.shape(key)) != 2


def merged_value(ctx: MergeContext, key: str) -> torch.Tensor:
    """The merged fp32 value of A's tensor `key` (method + LoRAs)."""
    opts, dev = ctx.opts, ctx.device
    if opts.vectors_from != "merge" and _is_vector(ctx, key):
        src = _vector_source(ctx)
        sk = src.key_for(key)
        if sk is not None:
            return src.fmt.read_fp32(sk, device=dev)
    a = ctx.A.fmt.read_fp32(key, device=dev)
    module, suffix = ckpt_module(key)
    bkey = ctx.B.key_for(key) if ctx.B else None
    if bkey is not None:
        b = ctx.B.fmt.read_fp32(bkey, device=dev)
        if b.shape != a.shape:
            raise FormatError(f"{key}: shape {list(a.shape)} in A, {list(b.shape)} in B; not the same architecture")
    else:
        b = None
    ckey = ctx.C.key_for(key) if ctx.C else None
    c = ctx.C.fmt.read_fp32(ckey, device=dev) if ckey is not None else a
    if c.shape != a.shape:
        raise FormatError(f"{key}: shape {list(a.shape)} in A, {list(c.shape)} in C; not the same architecture")

    lora_taus = []
    for src, cn in ctx.lora_keys.get(key, []):
        d = src.delta(cn, dev)
        if d.shape != a.shape:
            d = d.reshape(a.shape)
        lora_taus.append(d)

    out = a
    if b is not None:
        wB = ctx.opts_weight_B * _block_factor(ctx.opts_shaping_B, module, opts.block_count)
        m, p = opts.method, opts.params
        if m == "weighted_sum":
            out = methods.weighted_sum(a, b, wB)
        elif m == "add_difference":
            out = methods.add_difference(a, b, c, wB)
        elif m == "slerp":
            out = methods.slerp(a, b, wB)
        elif m in methods.TWO_PASS:
            out = methods.cosine_merge(a, b, wB, ctx.cosine)
        elif m == "ties":
            taus = [a - c, (b - c) * wB]
            if opts.lora_mode == "task_vectors":
                taus += lora_taus
                lora_taus = []
            out = c + float(p.get("lambda", 1.0)) * methods.ties_merge(taus, float(p.get("density", 0.2)))
        elif m == "dare":
            taus = [a - c, (b - c) * wB]
            if opts.lora_mode == "task_vectors":
                taus += lora_taus
                lora_taus = []
            dens = float(p.get("density", 0.2)) if p.get("dare_ties") else None
            out = c + float(p.get("lambda", 1.0)) * methods.dare_merge(taus, float(p.get("p", 0.5)), int(p.get("seed", 0)), key, dens)
        elif m == "train_difference":
            out = methods.train_difference(a, b, c, wB)
        elif m == "extract":
            out = methods.extract_super(c, a, b, wB, float(p.get("beta", 0.0)), float(p.get("gamma", 1.0)))
        else:
            raise FormatError(f"unknown method {m}")
    for d in lora_taus:
        out = out + d
    return out


def _touched_keys(ctx: MergeContext) -> set:
    touched = set(ctx.lora_keys)
    if ctx.B is not None:
        for k in ctx.A.reader.infos:
            if ctx.A.fmt.is_consumed(k):
                continue
            if ctx.B.key_for(k) is not None and (ctx.A.reader.dtype(k) in FLOAT_TAGS or ctx.A.fmt.is_quantized(k)):
                touched.add(k)
    src = _vector_source(ctx)
    if src == "merge":
        return touched
    for k in ctx.A.reader.infos:
        if ctx.A.fmt.is_consumed(k) or not _is_vector(ctx, k) or k in ctx.lora_keys:
            continue
        if src is ctx.A or src.key_for(k) is None or ctx.A.reader.dtype(k) not in FLOAT_TAGS:
            touched.discard(k)        # A's own value, copied as it is
        else:
            touched.add(k)
    return touched


def merge_checkpoints(A: CkptInput, B: CkptInput | None, C: CkptInput | None, loras: list[LoraInput],
                      out_path: str, opts: CkptMergeOptions, progress=None, cancel=None, log=None):
    """Returns RunResult (checkpoint output) or ExtractResult (output as LoRA)."""
    log = log or (lambda s: None)
    for inp in [A, B, C] + list(loras):
        if inp is not None and same_file(inp.path, out_path):
            raise FormatError("the output path is one of the input files; choose another output name")
    ctx = _open_context(A, B, C, loras, opts, log)
    ctx.opts_weight_B = B.weight if B else 0.0
    ctx.opts_shaping_B = B.shaping if B else Shaping()
    for w in ctx.warnings:
        log("warning: " + w)
    try:
        if ctx.only_in_a:
            log(f"{len(ctx.only_in_a)} tensor(s) only in A are copied from A (e.g. {ctx.only_in_a[:2]})")
        if ctx.only_in_b:
            log(f"{len(ctx.only_in_b)} tensor(s) only in B are ignored (e.g. {ctx.only_in_b[:2]})")
        if ctx.B is not None and opts.method in methods.TWO_PASS:
            _cosine_prepass(ctx, progress, cancel)
        recipe = {"function": "ckpt_merge",
                  "A": A.to_dict(), "B": B.to_dict() if B else None, "C": C.to_dict() if C else None,
                  "loras": [l.to_dict() for l in loras], "options": opts.to_dict()}
        touched = _touched_keys(ctx)
        if opts.output_as_lora:
            return _write_as_lora(ctx, out_path, recipe, touched, progress, cancel, log)

        def compute_for(key):
            if key in touched:
                return lambda k=key: merged_value(ctx, k)
            return None

        plan = plan_from_primary(ctx.A.fmt, opts.output_format, opts.passthrough, opts.fp8_layer_set,
                                 compute_for=compute_for, touched_keys=touched)
        plan.notes.extend(ctx.warnings)
        plan.int8_clip = opts.int8_clip
        meta = build_metadata(plan, ctx.A.reader.metadata, recipe, opts.keep_metadata, opts.redact_inherited)
        inputs = tuple(i.path for i in [A, B, C] + list(loras) if i is not None)
        return execute_plan(plan, out_path, meta, ctx.device, progress, cancel, log, input_paths=inputs)
    finally:
        _close_context(ctx)


def _write_as_lora(ctx: MergeContext, out_path: str, recipe: dict, touched: set, progress, cancel, log):
    """Output as LoRA: (merged - reference) per 2-D weight, SVD, written as a LoRA file."""
    from .extract import ExtractResult
    lo = ctx.opts.lora_out
    eo = ExtractOptions(rank=int(lo.get("rank", 32)), naming=lo.get("naming", "comfy"), dtype=lo.get("dtype", "fp16"),
                        method=lo.get("method", "randomized"), group_ranks=dict(lo.get("group_ranks", {})))
    ref = ctx.C if ctx.C is not None else ctx.A
    t0 = time.time()
    mods = []
    for k in ctx.A.reader.names:
        if not k.endswith(".weight") or k not in touched:
            continue
        shape = ctx.A.reader.shape(k)
        if len(shape) != 2:
            continue
        rk = ref.key_for(k)
        if rk is None:
            continue
        mods.append((k, ckpt_module(k)[0], shape, rk))
    if not mods:
        raise FormatError("nothing to extract: no 2-D weight is changed by this merge")
    from . import TOOL_NAME, __version__
    meta = {"merge_tool": f"{TOOL_NAME} {__version__}", "merge_recipe": json.dumps(recipe_for_metadata(recipe), separators=(",", ":"))}
    tag, tdt = DTYPE_TAG[eo.dtype], DTYPE_TORCH[eo.dtype]
    writer = StreamWriter(out_path, meta)
    plan, clamped = [], []
    for k, module, shape, rk in mods:
        want = int(eo.group_ranks.get(group_of(module), eo.group_ranks.get("*", eo.rank)))
        r = max(1, min(want, min(shape)))
        if r < want:
            clamped.append((module, r))
        kd, ku, ka = output_keys(eo.naming, module, None, None)
        writer.add(kd, tag, [r, shape[1]])
        writer.add(ku, tag, [shape[0], r])
        writer.add(ka, "F32", [])
        plan.append((k, module, rk, r, kd, ku, ka))
    writer.begin()
    kept_min, kept_group = 1.0, {}
    try:
        for i, (k, module, rk, r, kd, ku, ka) in enumerate(plan):
            if cancel is not None and cancel():
                raise Cancelled("cancelled by the user")
            if progress is not None:
                progress(i, len(plan), module)
            delta = merged_value(ctx, k) - ref.fmt.read_fp32(rk, device=ctx.device)
            down, up, kept = _svd(delta, r, eo)
            del delta
            kept_min = min(kept_min, kept)
            kept_group.setdefault(group_of(module), []).append(kept)
            writer.write(kd, down.to(tdt).cpu())
            writer.write(ku, up.to(tdt).cpu())
            writer.write(ka, torch.tensor(float(r), dtype=torch.float32))
        writer.close()
    except BaseException:
        writer.abort()
        raise
    return ExtractResult(out_path, len(plan), [], clamped, kept_min, {g: min(v) for g, v in kept_group.items()},
                         time.time() - t0)


# ----------------------------------------------------------------------------- pre merge report
def premerge_report(A: CkptInput, B: CkptInput, C: CkptInput | None, use_gpu=True, progress=None, cancel=None) -> str:
    """Per group norms of B - A and B - C, similarity histogram, tensors present in one input only."""
    device = pick_device(use_gpu)
    a, b = _Ckpt(A.path), _Ckpt(B.path)
    c = _Ckpt(C.path) if C else None
    try:
        norms_ba: dict[str, float] = {}
        norms_bc: dict[str, float] = {}
        norms_a: dict[str, float] = {}
        sims = []
        keys = [k for k in a.reader.names if not a.fmt.is_consumed(k) and (a.reader.dtype(k) in FLOAT_TAGS or a.fmt.is_quantized(k))]
        only_a = [k for k in keys if b.key_for(k) is None]
        for i, k in enumerate(keys):
            if cancel is not None and cancel():
                raise Cancelled("cancelled by the user")
            if progress is not None:
                progress(i, len(keys), k)
            bk = b.key_for(k)
            if bk is None:
                continue
            g = group_of(ckpt_module(k)[0])
            ta = a.fmt.read_fp32(k, device=device)
            tb = b.fmt.read_fp32(bk, device=device)
            if ta.shape != tb.shape:
                continue
            norms_a[g] = norms_a.get(g, 0.0) + ta.norm().item() ** 2
            norms_ba[g] = norms_ba.get(g, 0.0) + (tb - ta).norm().item() ** 2
            if c is not None and c.key_for(k) is not None:
                tc = c.fmt.read_fp32(c.key_for(k), device=device)
                norms_bc[g] = norms_bc.get(g, 0.0) + (tb - tc).norm().item() ** 2
            if ta.numel() > 1:
                sims.append(torch.nn.functional.cosine_similarity(ta.reshape(-1), tb.reshape(-1), dim=0).item())
        lines = [f"A: {os.path.basename(A.path)} ({a.fmt.summary()})", f"B: {os.path.basename(B.path)} ({b.fmt.summary()})"]
        if c is not None:
            lines.append(f"C: {os.path.basename(C.path)} ({c.fmt.summary()})")
        lines.append("group            |B-A| / |A|" + ("     |B-C| / |A|" if c is not None else ""))
        for g in sorted(norms_a):
            na = norms_a[g] ** 0.5
            row = f"{g:16s} {norms_ba.get(g, 0.0) ** 0.5 / max(na, 1e-12):10.4f}"
            if c is not None:
                row += f"     {norms_bc.get(g, 0.0) ** 0.5 / max(na, 1e-12):10.4f}"
            lines.append(row)
        if sims:
            s = torch.tensor(sims)
            lines.append(f"cosine(A, B) over {len(sims)} tensors: min {s.min():.5f}, median {s.median():.5f}, max {s.max():.5f}")
        lines.append(f"tensors only in A: {len(only_a)}, only in B: {len([1 for (m, s_), k in b.by_module.items() if (m, s_) not in {(canon(ckpt_module(x)[0]), ckpt_module(x)[1]) for x in keys}])}")
        return "\n".join(lines)
    finally:
        a.close()
        b.close()
        if c is not None:
            c.close()
