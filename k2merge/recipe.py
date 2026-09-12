"""Recipes: one JSON document describes any run. Load, save, validate, dispatch.

Schema (function decides the rest):
  {"function": "lora_merge", "inputs": [LoraInput...], "options": LoraMergeOptions, "output": path}
  {"function": "extract", "base": path, "target": path, "options": ExtractOptions, "output": path}
  {"function": "ckpt_merge", "A": CkptInput, "B": CkptInput|null, "C": CkptInput|null,
   "loras": [LoraInput...], "options": CkptMergeOptions, "output": path}
  {"function": "convert", "inputs": [{"file": path}], "output_format": ..., "passthrough": ..., "output": path}
The same document is written into the output's __metadata__["merge_recipe"] (without "output").
"""
from __future__ import annotations

import json
import os

from .ckpt_merge import CkptInput, CkptMergeOptions, merge_checkpoints
from .engine import convert_checkpoint
from .extract import ExtractOptions, extract_lora
from .lora_merge import LoraInput, LoraMergeOptions, merge_loras
from .st_io import read_header

FUNCTIONS = ("lora_merge", "extract", "ckpt_merge", "convert")


def load_recipe(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    validate(r)
    return r


def save_recipe(recipe: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=1)


def recipe_from_file_metadata(path: str) -> dict | None:
    """The recipe stored in a produced file, or None."""
    h, _ = read_header(path)
    meta = h.get("__metadata__") or {}
    raw = meta.get("merge_recipe")
    if not raw:
        return None
    r = json.loads(raw)
    validate(r)
    return r


def validate(r: dict) -> None:
    fn = r.get("function")
    if fn not in FUNCTIONS:
        raise ValueError(f"recipe: unknown function {fn!r}")
    if fn == "lora_merge" and not r.get("inputs"):
        raise ValueError("recipe: lora_merge needs inputs")
    if fn == "extract" and not (r.get("base") and r.get("target")):
        raise ValueError("recipe: extract needs base and target")
    if fn == "ckpt_merge" and not r.get("A"):
        raise ValueError("recipe: ckpt_merge needs A")
    if fn == "convert" and not r.get("inputs"):
        raise ValueError("recipe: convert needs an input")


def _resolve(path: str | None, base_dir: str | None) -> str | None:
    if path is None:
        return None
    if base_dir and not os.path.isabs(path) and not os.path.exists(path):
        cand = os.path.join(base_dir, path)
        if os.path.exists(cand):
            return cand
    return path


def run_recipe(r: dict, output: str | None = None, base_dir: str | None = None, use_gpu: bool = True,
               progress=None, cancel=None, log=None):
    """Runs a recipe. `output` overrides r["output"]. Relative input paths resolve against base_dir."""
    validate(r)
    out = output or r.get("output")
    if not out:
        raise ValueError("recipe: no output path")
    fn = r["function"]
    if fn == "lora_merge":
        inputs = [LoraInput.from_dict(d) for d in r["inputs"]]
        for i in inputs:
            i.path = _resolve(i.path, base_dir)
        return merge_loras(inputs, out, LoraMergeOptions.from_dict(r.get("options")), use_gpu, progress, cancel, log)
    if fn == "extract":
        return extract_lora(_resolve(r["base"], base_dir), _resolve(r["target"], base_dir), out,
                            ExtractOptions.from_dict(r.get("options")), use_gpu, progress, cancel, log)
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
        opts = CkptMergeOptions.from_dict(r.get("options"))
        opts.use_gpu = use_gpu
        return merge_checkpoints(A, B, C, loras, out, opts, progress, cancel, log)
    if fn == "convert":
        src = _resolve(r["inputs"][0]["file"], base_dir)
        return convert_checkpoint(src, out, r.get("output_format", "bf16"), r.get("passthrough", "official"),
                                  r.get("fp8_layer_set", "official"), r.get("keep_metadata", True), None,
                                  use_gpu, progress, cancel, log, int8_clip=r.get("int8_clip", "mse"))
    raise ValueError(fn)
