"""Delta sources: everything that contributes a per module delta to a merge.

LoraSource       LoRA or LoKr file with strength and block shaping
CheckpointDelta  (target - base) * weight * block factor
Each source answers delta(canon, device) -> fp32 [out, in] and, when exact,
low_rank(canon, device) -> (down, up) with everything folded into down.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .blocks import Shaping
from .formats import FileFormat
from .keys import KREA2_BLOCKS, block_index_any, canon, ckpt_module
from .lora import LoraFile, LoraModule


@dataclass
class LoraSource:
    file: LoraFile
    strength: float = 1.0
    shaping: Shaping = field(default_factory=Shaping)
    block_count: int = KREA2_BLOCKS
    label: str = ""
    # canon -> checkpoint module name (bare); set by resolution, used for block lookup
    module_names: dict = field(default_factory=dict)

    def modules(self) -> set:
        return set(self.file.modules)

    def module(self, c: str) -> LoraModule:
        return self.file.modules[c]

    def factor(self, c: str) -> float:
        """strength times the block factor of module c."""
        name = self.module_names.get(c, self.file.modules[c].name)
        b = block_index_any(name)
        return self.strength * self.shaping.factor_for(b, self.block_count)

    def is_exact(self, c: str) -> bool:
        return self.file.modules[c].kind == "lora"

    def low_rank(self, c: str, device=None):
        f = self.file.factors(self.file.modules[c], device)
        if f is None:
            return None
        down, up = f
        return down * self.factor(c), up

    def delta(self, c: str, device=None) -> torch.Tensor:
        return self.file.delta(self.file.modules[c], device) * self.factor(c)


@dataclass
class CheckpointDelta:
    target: FileFormat
    base: FileFormat
    weight: float = 1.0
    shaping: Shaping = field(default_factory=Shaping)
    block_count: int = KREA2_BLOCKS
    label: str = ""
    _mods: dict = field(default_factory=dict)

    def __post_init__(self):
        tm = {canon(ckpt_module(k)[0]): k for k in self.target.reader.infos if k.endswith(".weight")}
        bm = {canon(ckpt_module(k)[0]): k for k in self.base.reader.infos if k.endswith(".weight")}
        self._mods = {c: (tm[c], bm[c]) for c in tm if c in bm}

    def modules(self) -> set:
        return set(self._mods)

    def is_exact(self, c: str) -> bool:
        return False

    def factor(self, c: str) -> float:
        tk = self._mods[c][0]
        b = block_index_any(ckpt_module(tk)[0])
        return self.weight * self.shaping.factor_for(b, self.block_count)

    def low_rank(self, c: str, device=None):
        return None

    def keys_of(self, c: str) -> tuple[str, str]:
        """(target key, base key) of module c."""
        return self._mods[c]

    def layouts(self, c: str) -> tuple[str, str]:
        """(base layout, target layout) storage layouts of module c."""
        tk, bk = self._mods[c]
        return self.base.layout_of(bk), self.target.layout_of(tk)

    def dtypes(self, c: str) -> tuple[str, str]:
        """(base, target) stored dtype tags of module c."""
        tk, bk = self._mods[c]
        return self.base.reader.dtype(bk), self.target.reader.dtype(tk)

    def read_pair(self, c: str, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """(target, base) weights as fp32 on device, dequantized. One read each; the analysis derives
        the delta, the base norm and the noise statistics from this single pair."""
        tk, bk = self._mods[c]
        t = self.target.read_fp32(tk, device=device)
        b = self.base.read_fp32(bk, device=device)
        if t.shape != b.shape:
            raise ValueError(f"{tk}: shape {list(t.shape)} in target, {list(b.shape)} in base")
        return t, b

    def delta(self, c: str, device=None) -> torch.Tensor:
        t, b = self.read_pair(c, device)
        return (t - b) * self.factor(c)
