"""Structural comparison of a safetensors header against a reference header.

Used to check that a produced file has the structure of an official Krea 2
file: same keys, same dtypes, same quantization metadata, same tensor ranks.
Shapes are ignored when the reference is the full size file and the produced
file is a mini fixture.
"""
from __future__ import annotations

import json
import os

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference")
REFERENCE_NAMES = {
    "bf16": "krea2_turbo_bf16",
    "fp8_scaled": "krea2_turbo_fp8_scaled",
    "int8_convrot": "krea2_turbo_int8_convrot",
}


def load_reference_header(kind: str) -> dict:
    with open(os.path.join(REFERENCE_DIR, REFERENCE_NAMES[kind] + ".header.json"), encoding="utf-8") as f:
        return json.load(f)


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
