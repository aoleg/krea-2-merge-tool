"""Storage formats: detection of quantization layouts, dequantization, quantization.

Layouts (see working_specs.md section 2.2):
  plain              bf16 / fp16 / fp32 tensors
  fp8                fp8 e4m3 without scale
  fp8_scaled_legacy  ``scaled_fp8`` marker + ``<module>.scale_weight``
  fp8_scaled_meta    ``<module>.weight_scale`` + ``__metadata__["_quantization_metadata"]``  (official Krea 2)
  fp8_scaled_cq      ``<module>.weight_scale`` + ``<module>.comfy_quant`` descriptor
  int8_convrot       int8 + ``weight_scale`` [out,1] + descriptor with convrot   (official Krea 2)
  int8               int8 + ``weight_scale``, no rotation
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch

from .hadamard import build_hadamard, rotate_weight, unrotate_weight
from .keys import (CKPT_PREFIXES, fp8_full_precision_mm, in_fp8_recipe, in_int8_recipe,
                   strip_prefixes)
from .st_io import DTYPES, FLOAT_TAGS, TensorReader

FP8_MAX = 448.0
INT8_MAX = 127.0
CONVROT_GROUP = 256

L_PLAIN = "plain"
L_FP8 = "fp8"
L_FP8_LEGACY = "fp8_scaled_legacy"
L_FP8_META = "fp8_scaled_meta"
L_FP8_CQ = "fp8_scaled_cq"
L_INT8_CONVROT = "int8_convrot"
L_INT8 = "int8"

OUTPUT_FORMATS = ("keep", "fp32", "fp16", "bf16", "fp8", "fp8_scaled", "int8_convrot")
OUTPUT_DTYPE_TAG = {"fp32": "F32", "fp16": "F16", "bf16": "BF16"}
# Passthrough dtype of non quantized float tensors in a quantized output:
#   official: mirror the official Krea 2 file of that format (int8: fp32 -> bf16 except the
#             RMSNorm ".scale" vectors; fp8 scaled: every fp32 -> bf16)
#   keep:     keep the source dtype
#   bf16:     every fp32 -> bf16
PASSTHROUGH = ("official", "keep", "bf16")


class FormatError(ValueError):
    pass


@dataclass
class QuantSpec:
    """How one quantized weight is stored."""
    layout: str
    module: str                 # module key as in the file, without ".weight"
    weight_key: str
    scale_key: str | None
    aux_keys: tuple = ()        # keys consumed together with the weight (descriptor, marker...)
    group_size: int = 0
    full_precision_mm: bool = False
    config: dict = field(default_factory=dict)


def comfy_quant_bytes(conf: dict) -> bytes:
    """Descriptor blob exactly as ComfyUI writes it (json.dumps default separators)."""
    return json.dumps(conf).encode("utf-8")


def _conf_layout(conf: dict, per_tensor: bool) -> str:
    fmt = conf.get("format")
    if fmt in ("float8_e4m3fn", "float8_e5m2"):
        return L_FP8_CQ if per_tensor else L_FP8_META
    if fmt == "int8_tensorwise":
        params = conf.get("params") if isinstance(conf.get("params"), dict) else {}
        if conf.get("convrot", params.get("convrot", False)):
            return L_INT8_CONVROT
        return L_INT8
    raise FormatError(f"unsupported quantization format {fmt!r}")


def _conf_group(conf: dict) -> int:
    params = conf.get("params") if isinstance(conf.get("params"), dict) else {}
    return int(conf.get("convrot_groupsize", params.get("convrot_groupsize", CONVROT_GROUP)))


class FileFormat:
    """Quantization map of one safetensors file."""

    def __init__(self, reader: TensorReader):
        self.reader = reader
        keys = list(reader.infos)
        self.prefix = ""
        for k in keys:
            for p in CKPT_PREFIXES:
                if k.startswith(p):
                    self.prefix = p
                    break
            break
        self.quant: dict[str, QuantSpec] = {}     # weight key -> spec
        self.consumed: set[str] = set()           # aux keys, never emitted on their own
        self.legacy_fpmm = False
        self._detect(keys)

    # ------------------------------------------------------------------ detection
    def _add(self, spec: QuantSpec):
        self.quant[spec.weight_key] = spec
        if spec.scale_key:
            self.consumed.add(spec.scale_key)
        self.consumed.update(spec.aux_keys)

    def _detect(self, keys: list[str]):
        infos = self.reader.infos
        keyset = set(keys)

        # 1. metadata layout
        meta = self.reader.metadata.get("_quantization_metadata")
        if meta:
            try:
                layers = (json.loads(meta) or {}).get("layers") or {}
            except json.JSONDecodeError as e:
                raise FormatError(f"_quantization_metadata is not valid JSON: {e}") from e
            for layer, conf in layers.items():
                cands = [layer] if layer + ".weight" in keyset else [self.prefix + layer]
                for mod in cands:
                    wk = mod + ".weight"
                    if wk not in keyset:
                        continue
                    sk = mod + ".weight_scale"
                    layout = _conf_layout(conf, per_tensor=False)
                    self._add(QuantSpec(layout, mod, wk, sk if sk in keyset else None,
                                        group_size=_conf_group(conf) if layout == L_INT8_CONVROT else 0,
                                        full_precision_mm=bool(conf.get("full_precision_matrix_mult", False)),
                                        config=dict(conf)))
                    for extra in ("input_scale",):
                        if mod + "." + extra in keyset:
                            self.consumed.add(mod + "." + extra)

        # 2. per tensor descriptors
        for k in keys:
            if not k.endswith(".comfy_quant"):
                continue
            mod = k[: -len(".comfy_quant")]
            wk = mod + ".weight"
            self.consumed.add(k)
            if wk not in keyset:
                continue
            raw = bytes(self.reader.raw(k))
            try:
                conf = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise FormatError(f"{k}: descriptor is not valid JSON: {e}") from e
            layout = _conf_layout(conf, per_tensor=True)
            sk = mod + ".weight_scale"
            self._add(QuantSpec(layout, mod, wk, sk if sk in keyset else None, aux_keys=(k,),
                                group_size=_conf_group(conf) if layout == L_INT8_CONVROT else 0,
                                full_precision_mm=bool(conf.get("full_precision_matrix_mult", False)),
                                config=dict(conf)))

        # 3. legacy scaled fp8
        marker = next((k for k in keys if k.endswith("scaled_fp8")), None)
        if marker is not None:
            p = marker[: -len("scaled_fp8")]
            self.consumed.add(marker)
            self.legacy_fpmm = int(torch.tensor(infos[marker]["shape"]).prod().item() if infos[marker]["shape"] else 1) == 2
            for k in keys:
                if k.startswith(p) and k.endswith(".scale_weight"):
                    mod = k[: -len(".scale_weight")]
                    wk = mod + ".weight"
                    if wk in keyset:
                        self._add(QuantSpec(L_FP8_LEGACY, mod, wk, k, aux_keys=(marker,),
                                            full_precision_mm=self.legacy_fpmm,
                                            config={"format": "float8_e4m3fn"}))
                    else:
                        self.consumed.add(k)
                elif k.startswith(p) and k.endswith(".scale_input"):
                    self.consumed.add(k)

        # 4. fp8 without scale, int8 without descriptor
        for k in keys:
            if k in self.quant or k in self.consumed or not k.endswith(".weight"):
                continue
            dt = infos[k]["dtype"]
            mod = k[: -len(".weight")]
            if dt in ("F8_E4M3", "F8_E5M2"):
                sk = mod + ".weight_scale"
                if sk in keyset:
                    self._add(QuantSpec(L_FP8_CQ, mod, k, sk, config={"format": "float8_e4m3fn"}))
                else:
                    self._add(QuantSpec(L_FP8, mod, k, None))
            elif dt == "I8":
                sk = mod + ".weight_scale"
                if sk not in keyset:
                    raise FormatError(f"{k}: int8 weight without a weight_scale tensor is not supported")
                self._add(QuantSpec(L_INT8, mod, k, sk, config={"format": "int8_tensorwise"}))

    # ------------------------------------------------------------------ queries
    def is_quantized(self, key: str) -> bool:
        return key in self.quant

    def is_consumed(self, key: str) -> bool:
        return key in self.consumed

    def layout_of(self, key: str) -> str:
        spec = self.quant.get(key)
        return spec.layout if spec else L_PLAIN

    def layout_counts(self) -> dict:
        c: dict[str, int] = {}
        for s in self.quant.values():
            c[s.layout] = c.get(s.layout, 0) + 1
        return c

    def dtype_counts(self) -> dict:
        c: dict[str, int] = {}
        for info in self.reader.infos.values():
            c[info["dtype"]] = c.get(info["dtype"], 0) + 1
        return c

    def summary(self) -> str:
        lc = self.layout_counts()
        if not lc:
            return "plain " + "/".join(f"{k}:{v}" for k, v in sorted(self.dtype_counts().items()))
        parts = [f"{k}:{v}" for k, v in sorted(lc.items())]
        return "quantized " + ", ".join(parts)

    def dominant_float(self) -> str:
        """Dominant non quantized float dtype tag (passthrough dtype of the file)."""
        c: dict[str, int] = {}
        for k, info in self.reader.infos.items():
            if k in self.quant or k in self.consumed:
                continue
            if info["dtype"] in ("BF16", "F16", "F32"):
                c[info["dtype"]] = c.get(info["dtype"], 0) + 1
        return max(c, key=c.get) if c else "BF16"

    # ------------------------------------------------------------------ dequant
    def read_fp32(self, key: str, device=None) -> torch.Tensor:
        """Any float tensor as fp32 on device, dequantized when needed."""
        t = self.reader.read(key, device=device)
        spec = self.quant.get(key)
        if spec is None:
            if not t.is_floating_point():
                raise FormatError(f"{key}: not a float tensor ({t.dtype})")
            return t.to(torch.float32)
        return dequantize(t, spec, self.reader, device)


def _broadcast_scale(scale: torch.Tensor, out_rows: int) -> torch.Tensor:
    s = scale.to(torch.float32)
    if s.numel() == 1:
        return s.reshape(())
    if s.numel() == out_rows:
        return s.reshape(out_rows, 1)
    raise FormatError(f"scale of {s.numel()} elements does not fit a weight with {out_rows} rows")


def dequantize(t: torch.Tensor, spec: QuantSpec, reader: TensorReader, device=None) -> torch.Tensor:
    w = t.to(device) if device is not None else t
    w = w.to(torch.float32)
    if spec.layout == L_FP8:
        return w
    scale = reader.read(spec.scale_key, device=w.device) if spec.scale_key else None
    if spec.layout in (L_FP8_LEGACY, L_FP8_META, L_FP8_CQ):
        if scale is None:
            return w
        return w * _broadcast_scale(scale, w.shape[0] if w.dim() >= 1 else 1)
    if spec.layout in (L_INT8, L_INT8_CONVROT):
        if scale is None:
            raise FormatError(f"{spec.weight_key}: int8 weight without scale")
        w = w * _broadcast_scale(scale, w.shape[0])
        if spec.layout == L_INT8_CONVROT:
            if w.dim() != 2:
                raise FormatError(f"{spec.weight_key}: convrot on a {w.dim()}-D tensor")
            w = unrotate_weight(w, spec.group_size or CONVROT_GROUP)
        return w
    raise FormatError(f"unknown layout {spec.layout}")


# ---------------------------------------------------------------------- quantizers
@torch.no_grad()
def quantize_fp8_plain(w32: torch.Tensor) -> torch.Tensor:
    return w32.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


@torch.no_grad()
def quantize_fp8_scaled(w32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per tensor scalar scale (official layout). Returns (fp8 weight, fp32 scale of shape [])."""
    amax = w32.abs().amax().to(torch.float32)
    scale = torch.clamp(amax / FP8_MAX, min=1e-12)
    q = (w32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.reshape(()).to(torch.float32)


INT8_CLIPS = ("mse", "absmax")
_CLIP_GRID = torch.linspace(0.55, 1.0, 80)


@torch.no_grad()
def quantize_int8_convrot(w32: torch.Tensor, group_size: int = CONVROT_GROUP, clip: str = "mse") -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate, per row scale, round to nearest. Returns (int8 weight, fp32 scale [out, 1]).

    clip "mse": per row scale chosen on the reference quantizer's 80 point grid
    between 0.55 and 1.0 of absmax by least squared error. This reproduces the
    official Krea 2 int8 file bit for bit (phase 8). "absmax": scale = absmax / 127.
    """
    if w32.dim() != 2:
        raise FormatError("int8 convrot needs a 2-D weight")
    h = build_hadamard(group_size, device=w32.device, dtype=torch.float32)
    wr = rotate_weight(w32.to(torch.float32), h, group_size)
    absmax = wr.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    if clip == "absmax":
        scale = (absmax / INT8_MAX).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        return q, scale.to(torch.float32)
    best_mse = torch.full_like(absmax, float("inf"))
    best_scale = absmax / INT8_MAX
    for a in _CLIP_GRID.tolist():
        sc = (absmax * a / INT8_MAX).clamp(min=1e-30)
        q = (wr / sc).round().clamp(-INT8_MAX, INT8_MAX)
        mse = ((q * sc - wr) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, sc, best_scale)
    q = (wr / best_scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, best_scale.to(torch.float32)


def convrot_eligible(shape) -> bool:
    return len(shape) == 2 and shape[1] % CONVROT_GROUP == 0 and shape[0] >= 8


def fp16_overflow(w32: torch.Tensor) -> bool:
    return bool(w32.abs().amax().item() > torch.finfo(torch.float16).max)


# ---------------------------------------------------------------------- output planning
@dataclass
class OutputSpec:
    """What one module's weight becomes in the output."""
    kind: str                       # "plain" | "fp8" | "fp8_scaled" | "int8_convrot" | "raw"
    dtype_tag: str                  # storage dtype of the weight
    layer_conf: dict | None = None  # for fp8_scaled: entry of _quantization_metadata
    descriptor: bytes | None = None  # for int8_convrot: comfy_quant bytes
    group_size: int = 0


def plan_weight(module: str, shape, src_tag: str, src_layout: str, out_format: str,
                passthrough: str = "official", fp8_layer_set: str = "official") -> OutputSpec:
    """Decides the storage of one weight tensor for the requested output format.

    module: bare module name (prefix stripped). src_layout: layout in the input.
    """
    is_float = src_tag in FLOAT_TAGS or src_layout != L_PLAIN
    if not is_float:
        return OutputSpec("raw", src_tag)
    key = module + ".weight"

    if out_format == "keep":
        if src_layout == L_INT8_CONVROT:
            return _int8_spec(CONVROT_GROUP)
        if src_layout in (L_FP8_LEGACY, L_FP8_META, L_FP8_CQ):
            return OutputSpec("fp8_scaled", "F8_E4M3", layer_conf=_fp8_conf(fp8_full_precision_mm(module)))
        if src_layout == L_FP8:
            return OutputSpec("fp8", "F8_E4M3")
        if src_layout == L_INT8:
            # unrotated int8 is read only; keep writes it as int8 convrot when eligible, else bf16
            return _int8_spec(CONVROT_GROUP) if convrot_eligible(shape) else OutputSpec("plain", "BF16")
        return OutputSpec("plain", passthrough_tag(key, src_tag, "keep", passthrough))

    if out_format in ("fp32", "fp16", "bf16"):
        return OutputSpec("plain", OUTPUT_DTYPE_TAG[out_format])

    if out_format == "fp8":
        if len(shape) == 2 and in_fp8_recipe(module, include_txtfusion=True):
            return OutputSpec("fp8", "F8_E4M3")
        return OutputSpec("plain", passthrough_tag(key, src_tag, out_format, passthrough))

    if out_format == "fp8_scaled":
        if len(shape) == 2 and in_fp8_recipe(module, include_txtfusion=(fp8_layer_set == "official")):
            return OutputSpec("fp8_scaled", "F8_E4M3", layer_conf=_fp8_conf(fp8_full_precision_mm(module)))
        return OutputSpec("plain", passthrough_tag(key, src_tag, out_format, passthrough))

    if out_format == "int8_convrot":
        if in_int8_recipe(module) and convrot_eligible(shape):
            return _int8_spec(CONVROT_GROUP)
        return OutputSpec("plain", passthrough_tag(key, src_tag, out_format, passthrough))

    raise FormatError(f"unknown output format {out_format!r}")


def passthrough_tag(key: str, src_tag: str, out_format: str, passthrough: str) -> str:
    """Storage dtype of a float tensor that is not quantized in a quantized or kept output."""
    if src_tag in ("F8_E4M3", "F8_E5M2", "I8"):
        return "BF16"
    if passthrough == "official" and out_format == "int8_convrot" and key.endswith(".scale") and src_tag in ("BF16", "F16", "F32"):
        return "F32"          # the official int8 file keeps RMSNorm scales in fp32 (upcast is exact)
    if src_tag != "F32":
        return src_tag
    if passthrough == "bf16":
        return "BF16"
    if passthrough == "official":
        if out_format in ("fp8", "fp8_scaled"):
            return "BF16"
        if out_format == "int8_convrot":
            return "BF16"
    return src_tag


def _fp8_conf(fpmm: bool) -> dict:
    conf = {"format": "float8_e4m3fn"}
    if fpmm:
        conf["full_precision_matrix_mult"] = True
    return conf


def _int8_spec(gs: int) -> OutputSpec:
    conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
    return OutputSpec("int8_convrot", "I8", descriptor=comfy_quant_bytes(conf), group_size=gs)


def plan_plain_tensor(key: str, src_tag: str, out_format: str, passthrough: str) -> str:
    """Storage dtype tag of a non weight float tensor (bias, norm, modulation)."""
    if src_tag not in FLOAT_TAGS:
        return src_tag
    if out_format in ("fp32", "fp16", "bf16"):
        return OUTPUT_DTYPE_TAG[out_format]
    return passthrough_tag(key, src_tag, out_format, passthrough)
