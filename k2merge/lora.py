"""LoRA and LoKr files: parsing in every convention, resolution, deltas.

Conventions (working_specs.md 2.3): ComfyUI ``diffusion_model.<m>.lora_down/up``
+ ``.alpha``, kohya ``lora_unet_<m_with_underscores>``, diffusers ``lora_A/B``
(optional ``.default``), ai-toolkit LoKr ``lokr_w1/w2`` (full or ``_a/_b``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .keys import canon, strip_prefixes
from .st_io import TensorReader

ROLE_TOKENS = {
    "lora_A": "down", "lora_B": "up", "lora_down": "down", "lora_up": "up",
    "lokr_w1": "w1", "lokr_w1_a": "w1a", "lokr_w1_b": "w1b",
    "lokr_w2": "w2", "lokr_w2_a": "w2a", "lokr_w2_b": "w2b", "lokr_t2": "t2",
    "alpha": "alpha", "dora_scale": "dora",
}
NOISE_TOKENS = {"weight", "default", "bias"}
META_ALPHA_KEYS = ("ss_network_alpha", "network_alpha", "alpha")


class LoraFormatError(ValueError):
    pass


def parse_key(key: str) -> tuple[str | None, str | None]:
    """'diffusion_model.blocks.0.attn.wq.lora_down.weight' -> ('diffusion_model.blocks.0.attn.wq', 'down')."""
    parts = key.split(".")
    for i in range(len(parts) - 1, max(len(parts) - 4, -1), -1):
        tok = parts[i]
        if tok in NOISE_TOKENS:
            continue
        role = ROLE_TOKENS.get(tok)
        if role is not None:
            return ".".join(parts[:i]), role
        return None, None
    return None, None


@dataclass
class LoraModule:
    name: str                       # module name as written in the file
    canon: str
    kind: str                       # "lora" | "lokr"
    roles: dict = field(default_factory=dict)   # role -> tensor key
    rank: int | None = None
    alpha: float | None = None
    scale: float = 1.0              # applied scale (alpha / rank, LoKr rules)
    out_features: int = 0
    in_features: int = 0
    suffix: dict = field(default_factory=dict)  # role -> key suffix (for output naming)


class LoraFile:
    """One LoRA or LoKr file, parsed into modules."""

    def __init__(self, path: str):
        self.path = path
        self.reader = TensorReader(path)
        self.metadata = self.reader.metadata
        self.modules: dict[str, LoraModule] = {}     # canon -> module
        self.convention = "unknown"
        self.unparsed: list[str] = []
        self._parse()

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------ parsing
    def _parse(self):
        grouped: dict[str, dict] = {}
        for key in self.reader.infos:
            name, role = parse_key(key)
            if role is None:
                self.unparsed.append(key)
                continue
            grouped.setdefault(name, {})[role] = key

        dora = [n for n, r in grouped.items() if "dora" in r]
        if dora:
            raise LoraFormatError(
                f"{self.path}: DoRA file ({len(dora)} modules carry dora_scale). DoRA is not supported.")

        meta_alpha = None
        for mk in META_ALPHA_KEYS:
            if mk in self.metadata:
                try:
                    meta_alpha = float(self.metadata[mk])
                except (TypeError, ValueError):
                    meta_alpha = None
                break

        conv_votes: dict[str, int] = {}
        for name, roles in grouped.items():
            if "t2" in roles:
                raise LoraFormatError(f"{self.path}: {name} is a Tucker (conv) LoKr; not supported")
            if "w1" in roles or "w1a" in roles or "w2" in roles or "w2a" in roles:
                kind = "lokr"
            elif "down" in roles and "up" in roles:
                kind = "lora"
            else:
                self.unparsed.extend(roles.values())
                continue
            c = canon(name)
            if c in self.modules and self.modules[c].name != name:
                raise LoraFormatError(
                    f"{self.path}: '{self.modules[c].name}' and '{name}' normalize to the same module id "
                    f"({c}); the naming heuristic is too aggressive for this file")
            m = LoraModule(name=name, canon=c, kind=kind, roles=roles)
            alpha = None
            if "alpha" in roles:
                t = self.reader.read(roles["alpha"])
                if t.numel() >= 1:
                    alpha = float(t.reshape(-1)[0].float().item())
            if alpha is None:
                alpha = meta_alpha
            m.alpha = alpha
            if kind == "lora":
                dshape = self.reader.shape(roles["down"])
                ushape = self.reader.shape(roles["up"])
                m.rank = int(dshape[0])
                m.in_features = int(torch.tensor(dshape[1:]).prod().item()) if len(dshape) > 1 else 1
                m.out_features = int(ushape[0])
                m.scale = (alpha / m.rank) if (alpha is not None and m.rank) else 1.0
                for role in ("down", "up"):
                    m.suffix[role] = roles[role][len(name):]
                if roles["down"].startswith("lora_unet_") or name.startswith("lora_unet_"):
                    conv_votes["kohya"] = conv_votes.get("kohya", 0) + 1
                elif ".lora_A" in roles["down"]:
                    conv_votes["diffusers"] = conv_votes.get("diffusers", 0) + 1
                else:
                    conv_votes["comfy"] = conv_votes.get("comfy", 0) + 1
            else:
                self._lokr_geometry(m, alpha)
                conv_votes["lokr"] = conv_votes.get("lokr", 0) + 1
            self.modules[c] = m
        self.convention = max(conv_votes, key=conv_votes.get) if conv_votes else "unknown"

    def _lokr_geometry(self, m: LoraModule, alpha: float | None):
        r = m.roles
        rd = self.reader
        if "w1" in r:
            s1 = rd.shape(r["w1"])
        elif "w1a" in r and "w1b" in r:
            s1 = [rd.shape(r["w1a"])[0], rd.shape(r["w1b"])[1]]
        else:
            raise LoraFormatError(f"{self.path}: {m.name}: LoKr without w1 factors")
        if "w2" in r:
            s2 = rd.shape(r["w2"])
        elif "w2a" in r and "w2b" in r:
            s2 = [rd.shape(r["w2a"])[0], rd.shape(r["w2b"])[1]]
        else:
            raise LoraFormatError(f"{self.path}: {m.name}: LoKr without w2 factors")
        if len(s1) != 2 or len(s2) != 2:
            raise LoraFormatError(f"{self.path}: {m.name}: only Linear LoKr is supported")
        m.out_features, m.in_features = int(s1[0] * s2[0]), int(s1[1] * s2[1])
        full = ("w1" in r) and ("w2" in r)
        if full:
            m.rank, m.scale = None, 1.0      # LyCORIS: both factors full -> scale 1
        else:
            dim = rd.shape(r["w2a"])[1] if "w2a" in r else rd.shape(r["w1a"])[1]
            m.rank = int(dim)
            m.scale = (alpha / dim) if alpha is not None else 1.0

    # ------------------------------------------------------------------ queries
    def kind_counts(self) -> dict:
        c: dict[str, int] = {}
        for m in self.modules.values():
            c[m.kind] = c.get(m.kind, 0) + 1
        return c

    def ranks(self) -> list[int]:
        return sorted({m.rank for m in self.modules.values() if m.rank})

    def summary(self) -> str:
        kc = self.kind_counts()
        ranks = self.ranks()
        alphas = sorted({round(m.scale, 4) for m in self.modules.values()})
        return (f"{self.convention}, {len(self.modules)} modules {kc}, rank {ranks[:4]}, "
                f"applied scale {alphas[:4]}")

    # ------------------------------------------------------------------ tensors
    def _t(self, key: str, device) -> torch.Tensor:
        return self.reader.read(key, device=device).to(torch.float32)

    def factors(self, m: LoraModule, device=None) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(down [r,in], up [out,r]) with the applied scale folded into down. None for LoKr."""
        if m.kind != "lora":
            return None
        down = self._t(m.roles["down"], device).reshape(m.rank, -1)
        up = self._t(m.roles["up"], device).reshape(-1, m.rank)
        return down * m.scale, up

    def delta(self, m: LoraModule, device=None) -> torch.Tensor:
        """Unweighted delta [out, in] in fp32 (applied scale included)."""
        if m.kind == "lora":
            down, up = self.factors(m, device)
            return up @ down
        r = m.roles
        if "w1" in r:
            w1 = self._t(r["w1"], device)
        else:
            w1 = self._t(r["w1a"], device) @ self._t(r["w1b"], device)
        if "w2" in r:
            w2 = self._t(r["w2"], device)
        else:
            w2 = self._t(r["w2a"], device) @ self._t(r["w2b"], device)
        return torch.kron(w1, w2) * m.scale


# ---------------------------------------------------------------------- resolution
@dataclass
class Resolution:
    matched: dict            # canon -> checkpoint weight key
    unmatched: list          # LoRA module names
    mismatched: list         # (LoRA module name, lora shape, ckpt shape)


def checkpoint_modules(keys, shapes: dict | None = None) -> dict:
    """canon -> (weight key, module name) for every .weight in a checkpoint key list."""
    out = {}
    from .keys import ckpt_module
    for k in keys:
        if not k.endswith(".weight"):
            continue
        module, _ = ckpt_module(k)
        out[canon(module)] = (k, module)
    return out


def resolve(lora: LoraFile, ckpt_mods: dict, ckpt_shapes: dict | None = None) -> Resolution:
    matched, unmatched, mismatched = {}, [], []
    for c, m in lora.modules.items():
        hit = ckpt_mods.get(c)
        if hit is None:
            unmatched.append(m.name)
            continue
        key = hit[0]
        if ckpt_shapes is not None:
            shape = list(ckpt_shapes[key])
            if len(shape) == 2 and (shape[0] != m.out_features or shape[1] != m.in_features):
                mismatched.append((m.name, [m.out_features, m.in_features], shape))
                continue
        matched[c] = key
    return Resolution(matched, unmatched, mismatched)
