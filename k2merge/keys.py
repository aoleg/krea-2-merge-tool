"""Key naming: canonical module ids, prefix handling, Krea 2 block classification."""
from __future__ import annotations

import re

# Wrapper prefixes that differ between exporters and carry no structural meaning.
# Longest first: stripping is applied repeatedly until nothing matches.
LORA_PREFIXES = (
    "model.diffusion_model.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
    "unet.",
    "lora_unet_",
    "lora_transformer_",
    "lora_te1_",
    "lora_te2_",
    "lora_te_",
)
CKPT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")

PARAM_SUFFIXES = (".weight", ".bias", ".scale", ".lin")


def strip_prefixes(name: str, prefixes=LORA_PREFIXES) -> str:
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
                changed = True
    return name


# Diffusers naming of the Krea 2 conversion (OneTrainer, ai-toolkit, diffusers trainers)
# -> ComfyUI names. Verified against the shapes in real LoRAs (working_specs.md 2.3).
# Each rule works on dotted and on kohya underscore names ([._]).
_S = r"[._]"
DIFFUSERS_RULES = (
    (re.compile(rf"^transformer_blocks({_S})(\d+)"), r"blocks\1\2"),
    (re.compile(r"^text_fusion"), "txtfusion"),
    (re.compile(rf"attn({_S})to_q$"), r"attn\1wq"),
    (re.compile(rf"attn({_S})to_k$"), r"attn\1wk"),
    (re.compile(rf"attn({_S})to_v$"), r"attn\1wv"),
    (re.compile(rf"attn({_S})to_gate$"), r"attn\1gate"),
    (re.compile(rf"attn({_S})to_out{_S}0$"), r"attn\1wo"),
    (re.compile(rf"({_S})ff({_S})(gate|up|down)$"), r"\1mlp\2\3"),
    (re.compile(rf"^final_layer({_S})linear$"), r"last\1linear"),
    (re.compile(r"^img_in$"), "first"),
    (re.compile(rf"^time_embed({_S})linear_1$"), r"tmlp\g<1>0"),
    (re.compile(rf"^time_embed({_S})linear_2$"), r"tmlp\g<1>2"),
    (re.compile(rf"^txt_in({_S})linear_1$"), r"txtmlp\g<1>1"),
    (re.compile(rf"^txt_in({_S})linear_2$"), r"txtmlp\g<1>3"),
    (re.compile(r"^time_mod_proj$"), "tproj.1"),
)


def diffusers_to_comfy(name: str) -> str:
    """Bare module name in ComfyUI naming, from any convention (prefixes stripped)."""
    n = strip_prefixes(name)
    for rx, rep in DIFFUSERS_RULES:
        n = rx.sub(rep, n)
    return n


def canon(name: str) -> str:
    """Normalized module id, comparable across exporters.

    kohya flattens the module path with underscores ("blocks_0_attn_wq"),
    diffusers style exports keep the dots ("blocks.0.attn.wq"), and the
    diffusers conversion of Krea 2 renames modules ("transformer_blocks.0.attn.to_q").
    The diffusers names are mapped first, then both separators are dropped so
    every form lands on the same string: blocks0attnwq.
    """
    return diffusers_to_comfy(name).replace(".", "").replace("_", "").lower()


def ckpt_module(key: str) -> tuple[str, str]:
    """Checkpoint tensor key -> (module name without wrapper prefix, parameter suffix)."""
    k = strip_prefixes(key, CKPT_PREFIXES)
    for s in PARAM_SUFFIXES:
        if k.endswith(s):
            return k[: -len(s)], s
    return k, ""


def ckpt_prefix(keys) -> str:
    """The wrapper prefix a checkpoint uses ('' for bare keys)."""
    for k in keys:
        for p in CKPT_PREFIXES:
            if k.startswith(p):
                return p
        return ""
    return ""


# ----------------------------------------------------------------------------- Krea 2
_RE_BLOCK = re.compile(r"^blocks\.(\d+)\.")
_RE_BLOCK_ANY = re.compile(r"^blocks[._](\d+)[._]")     # kohya underscore names too
NON_BLOCK_PREFIXES = ("txtfusion.", "first", "last.", "tmlp.", "txtmlp.", "tproj.", "pe_embedder")
_NON_BLOCK_ANY = ("txtfusion", "first", "last", "tmlp", "txtmlp", "tproj", "pe_embedder")
KREA2_BLOCKS = 28


def block_index_any(module: str) -> int | None:
    """Block number from a module name in any convention (dots, kohya underscores, diffusers names)."""
    m = diffusers_to_comfy(module)
    if m.startswith(_NON_BLOCK_ANY):
        return None
    mm = _RE_BLOCK_ANY.match(m)
    return int(mm.group(1)) if mm else None

# Module groups for the analysis report and the quantization recipes.
GROUP_ATTN = "blocks.attn"
GROUP_MLP = "blocks.mlp"
GROUP_MOD = "blocks.mod_norm"
GROUP_TXT = "txtfusion"
GROUP_PROJ = "projections"
GROUP_OTHER = "other"
GROUPS = (GROUP_ATTN, GROUP_MLP, GROUP_MOD, GROUP_TXT, GROUP_PROJ, GROUP_OTHER)


def block_index(module: str) -> int | None:
    """Block number of a bare Krea 2 module name, or None for the non block bucket."""
    if module.startswith(NON_BLOCK_PREFIXES):
        return None
    m = _RE_BLOCK.match(module)
    return int(m.group(1)) if m else None


def group_of(module: str) -> str:
    if _RE_BLOCK.match(module):
        rest = _RE_BLOCK.sub("", module)
        if rest.startswith("attn."):
            return GROUP_ATTN
        if rest.startswith("mlp."):
            return GROUP_MLP
        return GROUP_MOD
    if module.startswith("txtfusion."):
        return GROUP_TXT
    if module.startswith(("txtmlp.", "tmlp.", "tproj.", "first", "last")):
        return GROUP_PROJ
    return GROUP_OTHER


def is_krea2(keys) -> bool:
    mods = {ckpt_module(k)[0] for k in keys}
    return "blocks.0.attn.wq" in mods and any(m.startswith("txtfusion.") for m in mods)


def block_count(keys) -> int:
    mx = -1
    for k in keys:
        b = block_index(ckpt_module(k)[0])
        if b is not None:
            mx = max(mx, b)
    return mx + 1


# Official Krea 2 quantization layer sets (from the official file headers).
_RE_BLOCK_LINEAR = re.compile(r"^blocks\.\d+\.(attn\.(gate|wq|wk|wv|wo)|mlp\.(gate|up|down))$")
_RE_TXT_LINEAR = re.compile(
    r"^txtfusion\.(layerwise|refiner)_blocks\.\d+\.(attn\.(gate|wq|wk|wv|wo)|mlp\.(gate|up|down))$")
_RE_OUTPUT_PROJ = re.compile(r"\.(attn\.gate|attn\.wo|mlp\.down)$")


def in_int8_recipe(module: str) -> bool:
    """The 224 block linears of the official int8 convrot file."""
    return bool(_RE_BLOCK_LINEAR.match(module))


def in_fp8_recipe(module: str, include_txtfusion: bool = True) -> bool:
    """The 256 linears of the official fp8 scaled file (224 block + 32 txtfusion)."""
    if _RE_BLOCK_LINEAR.match(module):
        return True
    return include_txtfusion and bool(_RE_TXT_LINEAR.match(module))


def fp8_full_precision_mm(module: str) -> bool:
    """The official fp8 file flags the output projections attn.gate, attn.wo, mlp.down."""
    return bool(_RE_OUTPUT_PROJ.search(module))
