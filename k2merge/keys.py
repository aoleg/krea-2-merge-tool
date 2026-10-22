"""Key naming: canonical module ids, prefix handling, block classification per architecture (Krea 2, MOSS-Image)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

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
    "lycoris_",              # LyCORIS's own wrapper names: lycoris_blocks_0_attn_wq.lokr_w1
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


# ----------------------------------------------------------------------------- module groups
# Module groups for the analysis report and the quantization recipes. The ids are shared by every
# architecture, so a recipe's group ranks mean the same thing for each: the text group holds Krea 2's
# txtfusion and MOSS-Image's text_embedder.
GROUP_ATTN = "blocks.attn"
GROUP_MLP = "blocks.mlp"
GROUP_MOD = "blocks.mod_norm"
GROUP_TXT = "txtfusion"
GROUP_PROJ = "projections"
GROUP_OTHER = "other"
GROUPS = (GROUP_ATTN, GROUP_MLP, GROUP_MOD, GROUP_TXT, GROUP_PROJ, GROUP_OTHER)


# ----------------------------------------------------------------------------- Krea 2
_RE_BLOCK = re.compile(r"^blocks\.(\d+)\.")
_RE_BLOCK_ANY = re.compile(r"^blocks[._](\d+)[._]")     # kohya underscore names too
NON_BLOCK_PREFIXES = ("txtfusion.", "first", "last.", "tmlp.", "txtmlp.", "tproj.", "pe_embedder")
_NON_BLOCK_ANY = ("txtfusion", "first", "last", "tmlp", "txtmlp", "tproj", "pe_embedder")
KREA2_BLOCKS = 28

# Official Krea 2 quantization layer sets (from the official file headers).
_RE_BLOCK_LINEAR = re.compile(r"^blocks\.\d+\.(attn\.(gate|wq|wk|wv|wo)|mlp\.(gate|up|down))$")
_RE_TXT_LINEAR = re.compile(
    r"^txtfusion\.(layerwise|refiner)_blocks\.\d+\.(attn\.(gate|wq|wk|wv|wo)|mlp\.(gate|up|down))$")
_RE_OUTPUT_PROJ = re.compile(r"\.(attn\.gate|attn\.wo|mlp\.down)$")


def _krea2_group(module: str) -> str:
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


# ----------------------------------------------------------------------------- MOSS-Image
# 42 single stream layers with fused projections (working_specs.md section 7C).
_RE_LAYER = re.compile(r"^layers\.(\d+)\.")
_RE_LAYER_ANY = re.compile(r"^layers[._](\d+)[._]")
MOSS_NON_BLOCK_PREFIXES = ("text_embedder.", "shared_block_modulation", "video_embedder", "final_linear_video",
                           "final_norm_video", "time_embedding.", "video_head_modulation", "rope.")
_MOSS_NON_BLOCK_ANY = ("text_embedder", "shared_block_modulation", "video_embedder", "final_linear_video",
                       "final_norm_video", "time_embedding", "video_head_modulation", "rope")
MOSS_BLOCKS = 42
# The official int8 convrot file: the four projections of every layer (168 linears).
_RE_LAYER_LINEAR = re.compile(r"^layers\.\d+\.proj\.(qkv|o|ffn_up_gate|ffn_down)$")
# Split attention LoRA modules that ComfyUI slices into the fused qkv weight.
RE_MOSS_SPLIT_QKV = re.compile(r"^(.*layers[._]\d+[._]proj)([._])(q|k|v)$")
# Every layer of the official int8 file carries this U8 tensor next to its quantized projections.
MOSS_ATTENTION_CONFIG = b'{"attention": "comfy_kitchen_int8"}'


def _moss_group(module: str) -> str:
    if _RE_LAYER.match(module):
        rest = _RE_LAYER.sub("", module)
        if rest.startswith("proj.ffn"):
            return GROUP_MLP
        if rest.startswith("proj."):          # qkv, o, and the q/k/v parts of a LoRA
            return GROUP_ATTN
        return GROUP_MOD
    if module.startswith("text_embedder."):
        return GROUP_TXT
    if module.startswith(("video_embedder", "final_linear_video", "final_norm_video", "time_embedding.",
                          "video_head_modulation", "shared_block_modulation")):
        return GROUP_PROJ
    return GROUP_OTHER


# ----------------------------------------------------------------------------- architecture profiles
@dataclass(frozen=True)
class ArchProfile:
    """Everything the tool assumes about one architecture's key names."""
    family: str                       # "krea2" | "moss"
    label: str                        # for messages and the inspect report
    block_re: re.Pattern              # bare dotted module name -> block number
    block_re_any: re.Pattern          # the same for kohya underscore names
    non_block: tuple                  # bare prefixes outside the blocks
    non_block_any: tuple              # the same, separator free (kohya names)
    blocks: int                       # block count of the official files
    group_fn: Callable[[str], str]
    int8_re: re.Pattern               # the official int8 convrot layer set
    int8_clip: str                    # the clip rule that reproduces the official int8 file
    modelspec_arch: str
    implementation: str
    references: dict                  # output format -> reference header name (refcheck)
    attention_config: bytes | None = None   # per layer U8 tensor of the official int8 file
    attn_leaf_re: re.Pattern | None = None  # extract filter "attn"

    def block_index(self, module: str) -> int | None:
        if module.startswith(self.non_block):
            return None
        m = self.block_re.match(module)
        return int(m.group(1)) if m else None

    def block_index_any(self, module: str) -> int | None:
        m = diffusers_to_comfy(module)
        if m.startswith(self.non_block_any):
            return None
        mm = self.block_re_any.match(m)
        return int(mm.group(1)) if mm else None

    def group_of(self, module: str) -> str:
        return self.group_fn(module)

    def in_int8(self, module: str) -> bool:
        return bool(self.int8_re.match(module))

    def in_fp8(self, module: str, include_txtfusion: bool = True) -> bool:
        if self.int8_re.match(module):
            return True
        return self.family == "krea2" and include_txtfusion and bool(_RE_TXT_LINEAR.match(module))

    def in_trainer(self, module: str) -> bool:
        """The linears ai-toolkit trains by default. Krea 2: the 224 block linears and the 32 txtfusion linears
        (256, the module set of a real ai-toolkit LoKr). MOSS-Image: the 168 layer projections, as in its LoRAs."""
        return self.in_fp8(module, include_txtfusion=True)

    def fp8_full_precision_mm(self, module: str) -> bool:
        return self.family == "krea2" and bool(_RE_OUTPUT_PROJ.search(module))

    def keeps_f32(self, key: str, out_format: str) -> bool:
        """Float tensors the official quantized file stores in fp32 whatever the source dtype: Krea 2's int8 file
        keeps its RMSNorm scales (an upcast is exact); MOSS-Image keeps only the rope bands, in both formats."""
        if self.family == "krea2":
            return out_format == "int8_convrot" and key.endswith(".scale")
        return key == "rope.bands"

    def block_count(self, keys) -> int:
        mx = -1
        for k in keys:
            b = self.block_index(ckpt_module(k)[0])
            if b is not None:
                mx = max(mx, b)
        return mx + 1

    def layer_of_int8(self, module: str) -> int | None:
        """The layer a quantized projection belongs to (for the attention config tensors)."""
        return self.block_index(module) if self.in_int8(module) else None


KREA2 = ArchProfile(
    family="krea2", label="Krea 2", block_re=_RE_BLOCK, block_re_any=_RE_BLOCK_ANY,
    non_block=NON_BLOCK_PREFIXES, non_block_any=_NON_BLOCK_ANY, blocks=KREA2_BLOCKS, group_fn=_krea2_group,
    int8_re=_RE_BLOCK_LINEAR, int8_clip="mse", modelspec_arch="Krea-2",
    implementation="https://github.com/krea-ai/krea-2",
    references={"bf16": "krea2_turbo_bf16", "fp8_scaled": "krea2_turbo_fp8_scaled",
                "int8_convrot": "krea2_turbo_int8_convrot"},
    attn_leaf_re=re.compile(r"^blocks\.\d+\.attn\."))
MOSS = ArchProfile(
    family="moss", label="MOSS-Image", block_re=_RE_LAYER, block_re_any=_RE_LAYER_ANY,
    non_block=MOSS_NON_BLOCK_PREFIXES, non_block_any=_MOSS_NON_BLOCK_ANY, blocks=MOSS_BLOCKS, group_fn=_moss_group,
    int8_re=_RE_LAYER_LINEAR, int8_clip="absmax", modelspec_arch="MOSS-Image",
    implementation="https://github.com/OpenMOSS/MOSS-Image",
    references={"bf16": "moss_image_sft_bf16", "int8_convrot": "moss_image_sft_int8_convrot"},
    attention_config=MOSS_ATTENTION_CONFIG,
    attn_leaf_re=re.compile(r"^layers\.\d+\.proj\.(qkv|o|q|k|v)$"))
PROFILES = (KREA2, MOSS)
FAMILIES = {p.family: p for p in PROFILES}


def get_profile(family: str | None) -> ArchProfile:
    return FAMILIES.get(family or "krea2", KREA2)


def detect_family(keys) -> ArchProfile | None:
    """The architecture of a checkpoint key list, None when it is neither (keys of any wrapper prefix)."""
    mods = {ckpt_module(k)[0] for k in keys}
    if "blocks.0.attn.wq" in mods and any(m.startswith("txtfusion.") for m in mods):
        return KREA2
    if "shared_block_modulation" in mods and "layers.0.proj.ffn_up_gate" in mods:
        return MOSS
    return None


def profile_of_module(module: str) -> ArchProfile:
    """The architecture one module name belongs to (any convention). The two name spaces do not overlap:
    no Krea 2 module starts with layers or a MOSS-Image prefix, no MOSS-Image module starts with blocks."""
    m = diffusers_to_comfy(module)
    if MOSS.block_re_any.match(m) or m.startswith(MOSS.non_block_any):
        return MOSS
    return KREA2


def profile_from_modules(names) -> ArchProfile:
    """The architecture of a LoRA, from its module names (any convention); Krea 2 when nothing says otherwise."""
    votes = {"krea2": 0, "moss": 0}
    for n in names:
        m = diffusers_to_comfy(n)
        if MOSS.block_re_any.match(m) or m.startswith(MOSS.non_block_any):
            votes["moss"] += 1
        elif KREA2.block_re_any.match(m) or m.startswith(KREA2.non_block_any):
            votes["krea2"] += 1
    return MOSS if votes["moss"] > votes["krea2"] else KREA2


def profile_for(checkpoint_keys=None, lora_names=None) -> ArchProfile:
    """The profile of a run: from the checkpoint when there is one, from the LoRA module names otherwise."""
    if checkpoint_keys is not None:
        p = detect_family(checkpoint_keys)
        if p is not None:
            return p
    if lora_names:
        return profile_from_modules(lora_names)
    return KREA2


def blocks_of_keys(keys) -> int:
    """Block count of any file from its tensor keys: a checkpoint's own count, else the count of the architecture
    a LoRA's names belong to (a LoRA may skip blocks, so its highest block says nothing)."""
    keys = list(keys)
    p = detect_family(keys)
    if p is not None:
        return p.block_count(keys) or p.blocks
    return profile_from_modules(keys).blocks


def resolve_block_count(requested: int | None, profile: ArchProfile, checkpoint_keys=None) -> int:
    """The block count of a run: an explicit recipe value, else the checkpoint's own, else the profile's."""
    n = profile.block_count(checkpoint_keys) if checkpoint_keys is not None else 0
    if requested:
        if n > 0 and int(requested) != n:
            raise ValueError(f"the recipe sets block_count {requested}, the {profile.label} checkpoint has {n} blocks; "
                             f"remove block_count from the recipe")
        return int(requested)
    return n if n > 0 else profile.blocks


# ----------------------------------------------------------------------------- name dispatched helpers
# The functions below take a bare module name and use the architecture that name belongs to, so callers
# that only hold a name (reports, tables) need no profile.
def block_index_any(module: str) -> int | None:
    """Block number from a module name in any convention (dots, kohya underscores, diffusers names)."""
    return profile_of_module(module).block_index_any(module)


def block_index(module: str) -> int | None:
    """Block number of a bare module name, or None for the non block bucket."""
    if _RE_LAYER.match(module) or module.startswith(MOSS_NON_BLOCK_PREFIXES):
        return MOSS.block_index(module)
    return KREA2.block_index(module)


def group_of(module: str) -> str:
    if _RE_LAYER.match(module) or module.startswith(MOSS_NON_BLOCK_PREFIXES):
        return _moss_group(module)
    return _krea2_group(module)


def is_krea2(keys) -> bool:
    return detect_family(keys) is KREA2


def block_count(keys) -> int:
    """Number of blocks in a checkpoint key list (the architecture detected from the keys)."""
    p = detect_family(keys) or KREA2
    return p.block_count(keys)


def in_int8_recipe(module: str) -> bool:
    """The block linears of the official int8 convrot file (Krea 2: 224, MOSS-Image: 168)."""
    return bool(_RE_BLOCK_LINEAR.match(module)) or bool(_RE_LAYER_LINEAR.match(module))


def in_fp8_recipe(module: str, include_txtfusion: bool = True) -> bool:
    """The 256 linears of the official Krea 2 fp8 scaled file (224 block + 32 txtfusion); for MOSS-Image, which
    has no official fp8 file, the int8 layer set."""
    if _RE_BLOCK_LINEAR.match(module) or _RE_LAYER_LINEAR.match(module):
        return True
    return include_txtfusion and bool(_RE_TXT_LINEAR.match(module))


def fp8_full_precision_mm(module: str) -> bool:
    """The official Krea 2 fp8 file flags the output projections attn.gate, attn.wo, mlp.down."""
    return bool(_RE_OUTPUT_PROJ.search(module))
