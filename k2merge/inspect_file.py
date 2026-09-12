"""Inspect: header summary of any safetensors file (checkpoint or LoRA), no weights loaded."""
from __future__ import annotations

import json
import os

from .formats import FileFormat
from .keys import block_count, is_krea2
from .lora import LoraFile, LoraFormatError
from .st_io import TensorReader, SafetensorsError


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def inspect_path(path: str) -> dict:
    """Returns a dict with a 'text' summary and structured fields."""
    info = {"path": path, "kind": "unknown", "text": ""}
    if not os.path.isfile(path):
        info["text"] = "file not found"
        return info
    info["size"] = os.path.getsize(path)
    try:
        r = TensorReader(path)
    except SafetensorsError as e:
        info["text"] = f"not a valid safetensors file: {e}"
        return info
    try:
        lines = [f"{os.path.basename(path)}  ({_human(info['size'])}, {len(r.infos)} tensors)"]
        keys = list(r.infos)
        looks_lora = any(("lora_" in k or "lokr_" in k) for k in keys[:200])
        if looks_lora:
            try:
                lf = LoraFile(path)
            except LoraFormatError as e:
                info["kind"] = "lora"
                info["text"] = "\n".join(lines + [f"LoRA: {e}"])
                return info
            try:
                info["kind"] = "lora"
                info["convention"] = lf.convention
                info["modules"] = len(lf.modules)
                info["kinds"] = lf.kind_counts()
                info["ranks"] = lf.ranks()
                lines.append(f"LoRA ({lf.summary()})")
                blocks = {b for b in (__import__('k2merge.keys', fromlist=['block_index_any']).block_index_any(m.name) for m in lf.modules.values()) if b is not None}
                if blocks:
                    lines.append(f"blocks targeted: {min(blocks)}-{max(blocks)} ({len(blocks)} blocks)")
                if lf.unparsed:
                    lines.append(f"{len(lf.unparsed)} tensor(s) not recognized, e.g. {lf.unparsed[:2]}")
                meta_keys = [k for k in lf.metadata if not k.startswith("ss_")]
                if lf.metadata:
                    lines.append(f"metadata: {len(lf.metadata)} keys" + (f", e.g. {meta_keys[:4]}" if meta_keys else " (kohya training metadata)"))
                if "merge_recipe" in lf.metadata:
                    lines.append("made by this tool: recipe stored in metadata")
            finally:
                lf.close()
        else:
            fmt = FileFormat(r)
            info["kind"] = "checkpoint"
            info["format"] = fmt.summary()
            info["layouts"] = fmt.layout_counts()
            info["dtypes"] = fmt.dtype_counts()
            info["prefix"] = fmt.prefix
            info["krea2"] = is_krea2(keys)
            info["blocks"] = block_count(keys)
            lines.append(f"checkpoint: {fmt.summary()}")
            lines.append(f"dtypes: {fmt.dtype_counts()}")
            lines.append(f"key prefix: {fmt.prefix or '(none)'}; Krea 2 layout: {'yes' if info['krea2'] else 'no'}; blocks: {info['blocks']}")
            if fmt.quant:
                fpmm = sum(1 for s in fmt.quant.values() if s.full_precision_mm)
                lines.append(f"quantized layers: {len(fmt.quant)}" + (f", {fpmm} with full precision matmul" if fpmm else ""))
            if r.metadata:
                keys_m = [k for k in r.metadata if k not in ("_quantization_metadata",)]
                lines.append(f"metadata: {keys_m[:6]}")
                if "merge_recipe" in r.metadata:
                    try:
                        rec = json.loads(r.metadata["merge_recipe"])
                        lines.append(f"made by this tool: {rec.get('function')} recipe stored in metadata")
                    except json.JSONDecodeError:
                        pass
        info["text"] = "\n".join(lines)
        return info
    finally:
        r.close()
