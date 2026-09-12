"""Block shaping: the Neo-LoraCtl block masks, reproduced exactly.

Source: Neo-LoraCtl loractl_core.py (block axis functions). The numbers here
must stay identical to the extension so that a baked LoRA equals the live one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BLOCK_PRESETS = ("FULL", "COMPOSITION", "CHARACTER", "STYLE")
MODIFIERS = ("Emphasize", "Suppress", "Isolate")
MAX_FACTOR = 2.0
DEFAULT_BLOCK_SHOULDER = 2.5


def _smoothstep(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _rise(x: float, edge: float, width: float) -> float:
    """0 -> 1 smoothstep ramp straddling `edge` (0.5 exactly at the edge)."""
    if width <= 0.0:
        return 1.0 if x >= edge else 0.0
    return _smoothstep((x - (edge - width / 2.0)) / width)


def _window(x: float, lo: float | None, hi: float | None, width: float) -> float:
    w = 1.0
    if lo is not None:
        w *= _rise(x, lo, width)
    if hi is not None:
        w *= 1.0 - _rise(x, hi, width)
    return w


def _apply_modifier(w: float, modifier: str, contrast: float) -> float:
    c = min(max(contrast, 0.0), 1.0)
    if modifier == "Suppress":
        return 1.0 - c * w
    return 1.0 - c * (1.0 - w)  # Isolate


def emphasize_amplitude(contrast: float, boost: float) -> float:
    return min(max(contrast, 0.0) * max(boost, 0.0), MAX_FACTOR)


def _emphasize_factor(w: float, p: float, a: float) -> float:
    if a <= 0.0 or p >= 1.0 - 1e-6:
        return 1.0
    p = max(p, 0.0)
    factor = 1.0 + a * (w - p) / (1.0 - p)
    return min(max(factor, 0.0), MAX_FACTOR)


def block_zone(preset: str, count: int) -> tuple[float | None, float | None]:
    third = count / 3.0
    if preset == "COMPOSITION":
        return None, third
    if preset == "CHARACTER":
        return third, 2.0 * third
    if preset == "STYLE":
        return 2.0 * third, None
    raise ValueError(f"unknown block preset: {preset}")


def build_block_mask(count: int, preset: str, modifier: str, contrast: float,
                     boost: float = 1.0, shoulder: float = DEFAULT_BLOCK_SHOULDER) -> list[float]:
    """Per block factor, evaluated at block centers (index + 0.5). Identical to Neo-LoraCtl."""
    if count < 1:
        raise ValueError("block count must be >= 1")
    if preset == "FULL" or contrast <= 0.0:
        return [1.0] * count
    if preset not in BLOCK_PRESETS:
        raise ValueError(f"unknown block preset: {preset}")
    if modifier not in MODIFIERS:
        raise ValueError(f"unknown modifier: {modifier}")
    lo, hi = block_zone(preset, count)
    windows = [_window(i + 0.5, lo, hi, shoulder) for i in range(count)]
    if modifier == "Emphasize":
        p = sum(windows) / count
        a = emphasize_amplitude(contrast, boost)
        return [_emphasize_factor(w, p, a) for w in windows]
    return [_apply_modifier(w, modifier, contrast) for w in windows]


@dataclass
class Shaping:
    """Block shaping settings of one LoRA row or of checkpoint B."""
    preset: str = "FULL"
    modifier: str = "Suppress"
    contrast: float = 0.5
    boost: float = 1.0
    custom: list[float] | None = None          # explicit per block factors (calibration)
    non_block: float | None = None             # factor for keys outside the mask; None = 1.0

    def factors(self, count: int) -> list[float]:
        if self.custom is not None:
            if len(self.custom) != count:
                raise ValueError(f"custom mask has {len(self.custom)} values, model has {count} blocks")
            return [float(x) for x in self.custom]
        return build_block_mask(count, self.preset, self.modifier, self.contrast, self.boost)

    def factor_for(self, block: int | None, count: int) -> float:
        if block is None:
            return 1.0 if self.non_block is None else float(self.non_block)
        return self.factors(count)[block]

    def is_flat(self) -> bool:
        return self.custom is None and (self.preset == "FULL" or self.contrast <= 0.0) \
            and (self.non_block is None or self.non_block == 1.0)

    def to_dict(self) -> dict:
        d = {"preset": self.preset, "modifier": self.modifier,
             "contrast": self.contrast, "boost": self.boost}
        if self.custom is not None:
            d["custom"] = list(self.custom)
        if self.non_block is not None:
            d["non_block"] = self.non_block
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "Shaping":
        if not d:
            return cls()
        return cls(preset=d.get("preset", "FULL"), modifier=d.get("modifier", "Suppress"),
                   contrast=float(d.get("contrast", 0.5)), boost=float(d.get("boost", 1.0)),
                   custom=d.get("custom"), non_block=d.get("non_block"))


# One click recipes from Neo-LoraCtl's calibration rounds.
RECIPES = {
    "character_keep_style": Shaping(preset="STYLE", modifier="Suppress", contrast=0.5),
    "style_protect_faces": Shaping(preset="CHARACTER", modifier="Suppress", contrast=0.5),
}
RECIPE_LABELS = {
    "character_keep_style": "Character LoRA: keep the checkpoint's style (STYLE + Suppress 0.5)",
    "style_protect_faces": "Style LoRA: protect faces (CHARACTER + Suppress 0.5)",
}
