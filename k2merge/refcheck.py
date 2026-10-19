"""Structural comparison of a safetensors header against a reference header.

Used to check that a produced file has the structure of an official file of
its architecture (Krea 2 or MOSS-Image): same keys, same dtypes, same
quantization metadata, same tensor ranks. Shapes are ignored when the
reference is the full size file and the produced file is a mini fixture.
"""
from __future__ import annotations

import json
import os

from .keys import FAMILIES, KREA2, detect_family

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference")
REFERENCE_NAMES = dict(KREA2.references)          # Krea 2, the names before MOSS-Image support
# every official header under reference/, per family (selfcheck reads them all)
REFERENCE_FILES = {
    "krea2": ("krea2_turbo_bf16", "krea2_turbo_fp8_scaled", "krea2_turbo_int8_convrot"),
    "moss": ("moss_image_base_bf16", "moss_image_sft_bf16", "moss_image_base_int8_convrot", "moss_image_sft_int8_convrot"),
}


def reference_name(kind: str, family: str = "krea2") -> str:
    refs = FAMILIES[family].references
    if kind not in refs:
        raise KeyError(f"no official {kind} file exists for {FAMILIES[family].label}")
    return refs[kind]


def load_reference_header(kind: str, family: str = "krea2") -> dict:
    with open(os.path.join(REFERENCE_DIR, reference_name(kind, family) + ".header.json"), encoding="utf-8") as f:
        return json.load(f)


def reference_for(header: dict, kind: str) -> dict:
    """The official header of `kind` for the architecture the header's keys belong to (Krea 2 by default)."""
    p = detect_family([k for k in header if k != "__metadata__"]) or KREA2
    return load_reference_header(kind, p.family)


def _strip(key: str, prefix: str) -> str:
    return key[len(prefix):] if prefix and key.startswith(prefix) else key


def compare_structure(header: dict, reference: dict, ignore_shapes: bool = True,
                      prefix: str = "") -> list[str]:
    """Returns a list of differences (empty = structurally identical)."""
    diffs: list[str] = []
    h = {_strip(k, prefix): v for k, v in header.items() if k != "__metadata__"}
    r = {k: v for k, v in reference.items() if k != "__metadata__"}
    missing = sorted(set(r) - set(h))
    extra = sorted(set(h) - set(r))
    if missing:
        diffs.append(f"{len(missing)} keys missing, e.g. {missing[:3]}")
    if extra:
        diffs.append(f"{len(extra)} extra keys, e.g. {extra[:3]}")
    for k in sorted(set(h) & set(r)):
        if h[k]["dtype"] != r[k]["dtype"]:
            diffs.append(f"{k}: dtype {h[k]['dtype']} vs reference {r[k]['dtype']}")
        if len(h[k]["shape"]) != len(r[k]["shape"]):
            diffs.append(f"{k}: rank {len(h[k]['shape'])} vs reference {len(r[k]['shape'])}")
        elif not ignore_shapes and list(h[k]["shape"]) != list(r[k]["shape"]):
            diffs.append(f"{k}: shape {h[k]['shape']} vs reference {r[k]['shape']}")
        elif ignore_shapes and k.endswith("weight_scale"):
            # per row scales must stay per row: [out,1] keeps the trailing 1
            hs, rs = list(h[k]["shape"]), list(r[k]["shape"])
            if len(hs) == 2 and (hs[1] != rs[1]):
                diffs.append(f"{k}: scale shape {hs} vs reference {rs}")
        elif ignore_shapes and k.endswith(".comfy_quant") and list(h[k]["shape"]) != list(r[k]["shape"]):
            diffs.append(f"{k}: descriptor length {h[k]['shape']} vs reference {r[k]['shape']}")

    hm = header.get("__metadata__") or {}
    rm = reference.get("__metadata__") or {}
    hq, rq = hm.get("_quantization_metadata"), rm.get("_quantization_metadata")
    if (hq is None) != (rq is None):
        diffs.append("_quantization_metadata present in one file only")
    elif hq is not None:
        hl = (json.loads(hq).get("layers") or {})
        hl = {_strip(k, prefix): v for k, v in hl.items()}
        rl = json.loads(rq).get("layers") or {}
        if set(hl) != set(rl):
            diffs.append(f"quantization layers differ: {len(set(hl) - set(rl))} extra, {len(set(rl) - set(hl))} missing")
        for k in sorted(set(hl) & set(rl)):
            if hl[k] != rl[k]:
                diffs.append(f"layer config {k}: {hl[k]} vs {rl[k]}")
    return diffs
