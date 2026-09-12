"""Streaming tensor pipeline: plan every output tensor, then write in one pass.

A plan is a list of PlanItem in output file order. Each item either copies raw
bytes from a source file or computes an fp32 tensor (a callable) and stores it
in the planned format. Quantized weights expand into weight + scale
(+ descriptor) items. The safetensors header is built from the plan before any
data is written.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from . import TOOL_NAME, __version__
from .formats import (FLOAT_TAGS, FileFormat, FormatError, OutputSpec, fp16_overflow,
                      plan_plain_tensor, plan_weight, quantize_fp8_plain, quantize_fp8_scaled,
                      quantize_int8_convrot)
from .keys import ckpt_module
from .st_io import DTYPES, TensorReader, StreamWriter, same_file


class Cancelled(RuntimeError):
    pass


@dataclass
class PlanItem:
    name: str
    dtype_tag: str
    shape: list
    kind: str                                  # raw | plain | fp8 | fp8_scaled | int8_convrot
    raw_from: tuple[TensorReader, str] | None = None   # (reader, key) for raw copies
    compute: Callable[[], torch.Tensor] | None = None  # fp32 tensor, any device
    spec: OutputSpec | None = None
    module: str = ""
    touched: bool = False                      # arithmetic happened (for verification sampling)


@dataclass
class Plan:
    items: list[PlanItem] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    quant_layers: dict = field(default_factory=dict)   # module -> layer conf (fp8 scaled metadata)
    notes: list[str] = field(default_factory=list)
    int8_clip: str = "mse"                             # int8 scale choice: mse (official) | absmax

    def add(self, item: PlanItem):
        self.items.append(item)

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for it in self.items:
            kinds[it.kind] = kinds.get(it.kind, 0) + 1
        return {"tensors": len(self.items), "kinds": kinds, "touched": sum(1 for i in self.items if i.touched)}


def pick_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- planning helpers
def plan_weight_items(plan: Plan, name: str, module: str, shape, src_tag: str, src_layout: str,
                      out_format: str, passthrough: str, fp8_layer_set: str,
                      compute: Callable[[], torch.Tensor] | None, raw_reader: TensorReader | None,
                      raw_fmt: FileFormat | None, touched: bool) -> None:
    """Adds the items for one weight: raw copy when nothing changes, else compute + store."""
    spec = plan_weight(module, shape, src_tag, src_layout, out_format, passthrough, fp8_layer_set)
    src_spec = raw_fmt.quant.get(name) if raw_fmt is not None else None

    # raw copy possible?  same storage kind and no arithmetic
    can_raw = (not touched) and raw_reader is not None and (
        (spec.kind == "raw")
        or (spec.kind == "plain" and src_layout == "plain" and spec.dtype_tag == src_tag)
        or (spec.kind == "fp8" and src_layout == "fp8")
        or (spec.kind == "fp8_scaled" and src_layout in ("fp8_scaled_legacy", "fp8_scaled_meta", "fp8_scaled_cq"))
        or (spec.kind == "int8_convrot" and src_layout == "int8_convrot"
            and src_spec is not None and src_spec.group_size == spec.group_size)
    )
    if can_raw:
        plan.add(PlanItem(name, src_tag, list(shape), "raw", raw_from=(raw_reader, name),
                          spec=spec, module=module))
        if spec.kind == "fp8_scaled":
            sk = src_spec.scale_key
            plan.add(PlanItem(module_key(name) + ".weight_scale", raw_reader.dtype(sk), raw_reader.shape(sk),
                              "raw", raw_from=(raw_reader, sk), module=module))
            plan.quant_layers[module_key(name)] = spec.layer_conf
        elif spec.kind == "int8_convrot":
            sk = src_spec.scale_key
            plan.add(PlanItem(module_key(name) + ".weight_scale", raw_reader.dtype(sk), raw_reader.shape(sk),
                              "raw", raw_from=(raw_reader, sk), module=module))
            desc = spec.descriptor
            plan.add(PlanItem(module_key(name) + ".comfy_quant", "U8", [len(desc)], "descriptor",
                              compute=lambda d=desc: d, module=module))
        return

    if spec.kind == "raw":
        # non float tensor that had arithmetic requested: cannot happen, copy raw
        plan.add(PlanItem(name, src_tag, list(shape), "raw", raw_from=(raw_reader, name), module=module))
        return

    plan.add(PlanItem(name, spec.dtype_tag, list(shape), spec.kind, compute=compute, spec=spec,
                      module=module, touched=touched))
    if spec.kind == "fp8_scaled":
        plan.add(PlanItem(module_key(name) + ".weight_scale", "F32", [], "scale", module=module))
        plan.quant_layers[module_key(name)] = spec.layer_conf
    elif spec.kind == "int8_convrot":
        plan.add(PlanItem(module_key(name) + ".weight_scale", "F32", [shape[0], 1], "scale", module=module))
        desc = spec.descriptor
        plan.add(PlanItem(module_key(name) + ".comfy_quant", "U8", [len(desc)], "descriptor",
                          compute=lambda d=desc: d, module=module))


def module_key(weight_key: str) -> str:
    return weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key


def plan_from_primary(primary: FileFormat, out_format: str, passthrough: str = "official",
                      fp8_layer_set: str = "official",
                      compute_for: Callable[[str], Callable[[], torch.Tensor] | None] | None = None,
                      touched_keys: set | None = None) -> Plan:
    """Plan an output that follows the primary file's key order.

    compute_for(key) returns a callable producing the fp32 value of a float
    tensor (after merging), or None when the tensor is untouched. For a plain
    conversion compute_for is None.
    """
    plan = Plan()
    reader = primary.reader
    touched_keys = touched_keys or set()
    for key in reader.names:
        if primary.is_consumed(key):
            continue
        info = reader.infos[key]
        tag, shape = info["dtype"], list(info["shape"])
        module, suffix = ckpt_module(key)
        is_weight = key.endswith(".weight")
        layout = primary.layout_of(key)
        touched = key in touched_keys
        comp = compute_for(key) if (compute_for is not None) else None
        if comp is None:
            # untouched: value straight from the primary (dequantized on demand)
            if tag in FLOAT_TAGS or primary.is_quantized(key):
                comp = (lambda k=key: primary.read_fp32(k, device=None))
        if is_weight and (primary.is_quantized(key) or (tag in FLOAT_TAGS and len(shape) == 2)):
            plan_weight_items(plan, key, module, shape, tag, layout, out_format, passthrough,
                              fp8_layer_set, comp, reader, primary, touched)
        elif tag in FLOAT_TAGS:
            out_tag = plan_plain_tensor(ckpt_module(key)[0] + suffix, tag, out_format, passthrough)
            if not touched and out_tag == tag:
                plan.add(PlanItem(key, tag, shape, "raw", raw_from=(reader, key), module=module))
            else:
                plan.add(PlanItem(key, out_tag, shape, "plain", compute=comp, module=module, touched=touched))
        else:
            plan.add(PlanItem(key, tag, shape, "raw", raw_from=(reader, key), module=module))
    return plan


# ----------------------------------------------------------------------------- execution
def build_metadata(plan: Plan, base_metadata: dict | None, recipe: dict | None, keep_metadata: bool) -> dict:
    meta: dict = {}
    if keep_metadata and base_metadata:
        meta.update({k: str(v) for k, v in base_metadata.items()})
    meta.pop("_quantization_metadata", None)
    if plan.quant_layers:
        meta["_quantization_metadata"] = json.dumps({"layers": plan.quant_layers})
    meta["format"] = "pt"
    meta["merge_tool"] = f"{TOOL_NAME} {__version__}"
    if recipe is not None:
        meta["merge_recipe"] = json.dumps(recipe, separators=(",", ":"))
    return meta


@dataclass
class RunResult:
    path: str
    tensors: int
    seconds: float
    verify: dict
    notes: list


def execute_plan(plan: Plan, out_path: str, metadata: dict, device: torch.device,
                 progress: Callable[[int, int, str], None] | None = None,
                 cancel: Callable[[], bool] | None = None,
                 log: Callable[[str], None] | None = None,
                 verify_samples: int = 12, input_paths: tuple = ()) -> RunResult:
    for p in input_paths:
        if same_file(p, out_path):
            raise FormatError("the output path is one of the input files; choose another output name")
    log = log or (lambda s: None)
    t0 = time.time()
    writer = StreamWriter(out_path, metadata)
    for it in plan.items:
        writer.add(it.name, it.dtype_tag, it.shape)
    writer.begin()

    # verification sample: touched tensors first, then a few others
    touched = [i for i, it in enumerate(plan.items) if it.touched and it.kind in ("plain", "fp8", "fp8_scaled", "int8_convrot")]
    others = [i for i, it in enumerate(plan.items) if not it.touched and it.kind in ("plain", "fp8", "fp8_scaled", "int8_convrot")]
    rng = random.Random(0)
    sample = set(rng.sample(touched, min(len(touched), verify_samples)))
    sample |= set(rng.sample(others, min(len(others), max(0, verify_samples - len(sample)))))
    kept: dict[str, torch.Tensor] = {}

    total = len(plan.items)
    pending_scale: torch.Tensor | None = None
    stage = _HostStage(device)
    try:
        for idx, it in enumerate(plan.items):
            if cancel is not None and cancel():
                raise Cancelled("cancelled by the user")
            if progress is not None:
                progress(idx, total, it.name)
            if it.kind == "raw":
                reader, key = it.raw_from
                writer.write(it.name, reader.raw(key))
                continue
            if it.kind == "descriptor":
                writer.write(it.name, it.compute())
                continue
            if it.kind == "scale":
                if pending_scale is None:
                    raise RuntimeError(f"{it.name}: scale planned without a preceding quantized weight")
                writer.write(it.name, pending_scale.reshape(it.shape).to(torch.float32).cpu())
                pending_scale = None
                continue
            w = it.compute()
            if not isinstance(w, torch.Tensor):
                raise RuntimeError(f"{it.name}: compute() did not return a tensor")
            w = w.to(device=device, dtype=torch.float32)
            if idx in sample:
                kept[it.name] = _subsample(w)
            if it.kind == "plain":
                dt = DTYPES[it.dtype_tag]
                if dt == torch.float16 and fp16_overflow(w):
                    raise FormatError(f"{it.name}: values exceed the fp16 range; use bf16 or fp32")
                writer.write(it.name, stage.host(w.to(dt)))
            elif it.kind == "fp8":
                writer.write(it.name, stage.host(quantize_fp8_plain(w)))
            elif it.kind == "fp8_scaled":
                q, s = quantize_fp8_scaled(w)
                writer.write(it.name, stage.host(q))
                pending_scale = s
            elif it.kind == "int8_convrot":
                q, s = quantize_int8_convrot(w, it.spec.group_size, plan.int8_clip)
                writer.write(it.name, stage.host(q))
                pending_scale = s
            else:
                raise RuntimeError(f"unknown plan kind {it.kind}")
            del w
            if device.type == "cuda" and it.shape and int(torch.tensor(it.shape).prod().item()) >= (1 << 24):
                # release the caching allocator's reserve after each large tensor: on Windows every
                # byte of reserved VRAM is also committed system memory (WDDM backing store)
                torch.cuda.empty_cache()
        if progress is not None:
            progress(total, total, "closing file")
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seconds = time.time() - t0
    verify = verify_output(out_path, plan, kept, device, log)
    return RunResult(out_path, total, seconds, verify, list(plan.notes))


class _HostStage:
    """One pinned host buffer for device to host copies of output tensors.

    A pageable .cpu() copy makes the driver commit a staging area per copy;
    copying into one persistent pinned buffer keeps private memory flat.
    """

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self._buf: torch.Tensor | None = None

    def host(self, t: torch.Tensor) -> torch.Tensor:
        if not self.cuda or t.device.type != "cuda":
            return t.cpu()
        n = t.numel() * t.element_size()
        if self._buf is None or self._buf.numel() < n:
            from .st_io import USE_PINNED
            try:
                self._buf = torch.empty(max(n, 1), dtype=torch.uint8, pin_memory=USE_PINNED)
            except RuntimeError:
                return t.cpu()
        host = self._buf[:n].view(t.dtype).reshape(t.shape)
        host.copy_(t.contiguous())
        return host


VERIFY_ELEMENTS = 1 << 18   # elements kept per sampled tensor (1 MB in fp32): memory stays flat on 900 MB tensors


def _subsample(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(indices, values) of a fixed pseudo random subset of the flattened tensor."""
    flat = w.reshape(-1)
    n = flat.numel()
    if n <= VERIFY_ELEMENTS:
        idx = torch.arange(n, device=flat.device)
    else:
        g = torch.Generator().manual_seed(n)
        idx = torch.randperm(n, generator=g)[:VERIFY_ELEMENTS].to(flat.device)
    return idx.cpu(), flat[idx].detach().cpu().clone()


def verify_output(out_path: str, plan: Plan, kept: dict, device: torch.device, log) -> dict:
    """Reopens the output and checks structure and a sample of values."""
    result = {"ok": True, "problems": [], "checked": 0, "max_rel_err": 0.0, "per_layer": []}
    with TensorReader(out_path) as r:
        fmt = FileFormat(r)
        planned = {it.name: it for it in plan.items}
        if set(r.infos) != set(planned):
            result["problems"].append("tensor set differs from the plan")
        for name, it in planned.items():
            info = r.infos.get(name)
            if info is None:
                continue
            if info["dtype"] != it.dtype_tag or list(info["shape"]) != list(it.shape):
                result["problems"].append(f"{name}: stored {info['dtype']} {info['shape']}, planned {it.dtype_tag} {it.shape}")
        for spec in fmt.quant.values():
            if spec.scale_key is None and spec.layout != "fp8":
                result["problems"].append(f"{spec.weight_key}: quantized weight without scale")
        for name, (idx, ref) in kept.items():
            try:
                got_full = fmt.read_fp32(name, device=device)
            except FormatError as e:
                result["problems"].append(f"{name}: cannot read back ({e})")
                continue
            got = got_full.reshape(-1)[idx.to(got_full.device)]
            del got_full
            ref_d = ref.to(got.device)
            denom = ref_d.norm().item()
            rel = ((got - ref_d).norm().item() / denom) if denom > 0 else 0.0
            result["per_layer"].append((name, rel))
            result["max_rel_err"] = max(result["max_rel_err"], rel)
            result["checked"] += 1
            kind = planned[name].kind
            limit = {"plain": 0.01, "fp8": 0.08, "fp8_scaled": 0.08, "int8_convrot": 0.03}.get(kind, 0.1)
            if planned[name].dtype_tag == "F32" and kind == "plain":
                limit = 1e-6
            if rel > limit:
                result["problems"].append(f"{name}: read back error {rel:.4f} exceeds {limit}")
    result["ok"] = not result["problems"]
    for p in result["problems"]:
        log(f"verify: {p}")
    return result


# ----------------------------------------------------------------------------- conversion job
def convert_checkpoint(src: str, dst: str, out_format: str, passthrough: str = "official",
                       fp8_layer_set: str = "official", keep_metadata: bool = True,
                       recipe: dict | None = None, use_gpu: bool = True,
                       progress=None, cancel=None, log=None, int8_clip: str = "mse") -> RunResult:
    """Single input, no arithmetic: format conversion."""
    device = pick_device(use_gpu)
    with TensorReader(src) as reader:
        fmt = FileFormat(reader)
        plan = plan_from_primary(fmt, out_format, passthrough, fp8_layer_set)
        plan.int8_clip = int8_clip
        rec = recipe if recipe is not None else {
            "function": "convert", "inputs": [{"role": "A", "file": os.path.basename(src)}],
            "output_format": out_format, "passthrough": passthrough, "fp8_layer_set": fp8_layer_set,
            "int8_clip": int8_clip}
        meta = build_metadata(plan, reader.metadata, rec, keep_metadata)
        return execute_plan(plan, dst, meta, device, progress, cancel, log, input_paths=(src,))
