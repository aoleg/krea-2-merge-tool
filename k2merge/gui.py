"""Tkinter GUI. The window only assembles recipes and hands them to the engine.

Native Windows look (ttk "vista" theme where available), DPI aware, following
the Windows text size setting (or a remembered UI scale), with the system font
one size larger than the default. Controls that do not apply are
hidden: block shaping unfolds when a preset other than FULL is chosen, method
parameters appear for the methods that use them, LoRA rows are added on
demand, and the rarely needed output options sit behind a toggle.
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from . import __version__
from .blocks import BLOCK_PRESETS, MODIFIERS, RECIPES, RECIPE_LABELS, Shaping
from .ckpt_merge import VECTOR_SOURCES, CkptInput, CkptMergeOptions
from .engine import Cancelled
from .extract import ExtractOptions
from .formats import OUTPUT_FORMATS, PASSTHROUGH
from .keys import KREA2_BLOCKS
from .lora_merge import LoraInput, LoraMergeOptions
from .methods import ADVANCED, METHODS, METHOD_LABELS, NEEDS_C
from .spectrum_tab import SpectrumTab
from .advisor_tab import AdvisorTab
from .meta_tab import MetaTab

ST_FILES = [("safetensors", "*.safetensors"), ("all files", "*.*")]
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")

# Dark palette from the original krea-2-lora-merge-tool v10. Light = the native Windows theme.
DARK = {"bg": "#12141a", "surface": "#1a1d26", "surface_2": "#232734", "border": "#2e3342",
        "fg": "#e6e9f0", "fg_muted": "#8b93a7", "accent": "#6c8cff", "accent_2": "#8aa2ff",
        "accent_fg": "#0d0f14", "ok": "#4ade80", "log_bg": "#0e1015", "trough": "#232734",
        "canvas": "#232734", "bar_up": "#8aa2ff", "bar_down": "#8b93a7", "bar_flat": "#4ade80", "link": "#8aa2ff"}
LIGHT = {"bg": "#f0f0f0", "surface": "#f0f0f0", "surface_2": "#ffffff", "border": "#c8c8c8",
         "fg": "#000000", "fg_muted": "#5f6b7a", "accent": "#1f4e9c", "accent_2": "#3b62e8",
         "accent_fg": "#ffffff", "ok": "#3a9d5c", "log_bg": "#ffffff", "trough": "#e2e7ef",
         "canvas": "#ffffff", "bar_up": "#2f6fd6", "bar_down": "#9aa4b1", "bar_flat": "#3a9d5c", "link": "#1f4e9c"}
THEMES = {"dark": DARK, "light": LIGHT}


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(d: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except OSError:
        pass
JSON_FILES = [("recipe", "*.json"), ("all files", "*.*")]
STEP = 0.05
PAD = {"padx": 6, "pady": 3}          # grid cell padding, rescaled by px() at start-up
SCALES = ("auto", "100", "125", "150", "175", "200")   # UI scale setting; auto = display DPI x Windows text size
_S = 1.0                               # the UI scale in effect (set once by MergeApp before any widget is built)


def px(n: int) -> int:
    """A pixel size from the 100 % layout, scaled to the current UI scale."""
    return int(round(n * _S))


def text_scale_factor() -> float:
    """Windows Settings > Accessibility > Text size, as a factor (1.0 when unset or not on Windows).

    Classic Win32 windows do not receive this setting; the value is a registry entry from 100 to 225."""
    if sys.platform != "win32":
        return 1.0
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Accessibility") as k:
            v, _ = winreg.QueryValueEx(k, "TextScaleFactor")
        return max(1.0, min(2.25, int(v) / 100.0))
    except (OSError, ValueError, TypeError):
        return 1.0


def ui_scale_factor(setting: str | None) -> float:
    """The scale factor for a settings value: 'auto' follows the Windows text size, '125' means 1.25."""
    if setting in (None, "", "auto"):
        return text_scale_factor()
    try:
        return max(0.5, min(3.0, int(setting) / 100.0))
    except (TypeError, ValueError):
        return text_scale_factor()
LABEL_W = 13                          # width of the label column, in characters
METHOD_PARAMS = {                     # parameters shown per method
    "ties": ("density", "lambda"),
    "dare": ("p", "seed", "lambda", "dare_ties", "density"),
    "extract": ("beta", "gamma"),
}
PARAM_DEFAULTS = (("density", "0.2"), ("lambda", "1.0"), ("p", "0.5"), ("seed", "0"), ("beta", "0.0"), ("gamma", "1.0"))
PARAM_HELP = {"density": "fraction of each change kept (TIES trim)", "lambda": "scale of the merged change",
              "p": "drop probability", "seed": "random seed", "beta": "0 = common parts, 1 = distinct parts", "gamma": "sharpness"}


def _fmt_ranks(d: dict) -> str:
    """{'blocks.attn': 12, ...} -> 'attn 12  ·  mlp 20  ·  ...' in the analysis order."""
    if not d:
        return ""
    order = ("blocks.attn", "blocks.mlp", "blocks.mod_norm", "txtfusion", "projections", "other", "*")
    names = {"blocks.attn": "attention", "blocks.mlp": "MLP", "blocks.mod_norm": "modulation", "txtfusion": "text fusion",
             "projections": "projections", "other": "other", "*": "all"}
    parts = [f"{names.get(g, g)} {d[g]}" for g in order if g in d] + [f"{g} {v}" for g, v in d.items() if g not in order]
    return "per group ranks from the analysis:  " + "  \u00b7  ".join(parts)


def _round_step(x: float) -> float:
    return round(round(float(x) / STEP) * STEP, 2)


def _enable_dpi_awareness():
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)      # per monitor v2
    except Exception:  # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


# ============================================================================== small widgets
class Collapsible(ttk.Frame):
    """A toggle line that shows or hides a body frame."""

    def __init__(self, master, title: str, open_: bool = False, on_toggle=None):
        super().__init__(master)
        self.title = title
        self.open = tk.BooleanVar(value=open_)
        self.on_toggle = on_toggle
        self.btn = ttk.Label(self, text=self._text(), style="Link.TLabel", cursor="hand2")
        self.btn.bind("<Button-1>", lambda _e: (self.open.set(not self.open.get()), self._toggle()))
        self.btn.pack(anchor="w", padx=px(6), pady=px(2))
        self.body = ttk.Frame(self)
        if open_:
            self.body.pack(fill="x", padx=(px(18), 0))

    def _text(self):
        return ("▾ " if self.open.get() else "▸ ") + self.title

    def _toggle(self):
        self.btn.configure(text=self._text())
        if self.open.get():
            self.body.pack(fill="x", padx=(px(18), 0))
        else:
            self.body.pack_forget()
        if self.on_toggle:
            self.on_toggle(self.open.get())

    def set_open(self, value: bool):
        if bool(value) != self.open.get():
            self.open.set(bool(value))
            self._toggle()


class FileSlot(ttk.Frame):
    """Label, entry, Browse, Inspect on one line."""

    def __init__(self, master, app, label: str, save: bool = False, hint: str = ""):
        super().__init__(master)
        self.app = app
        self.var = tk.StringVar()
        ttk.Label(self, text=label, width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        ttk.Entry(self, textvariable=self.var).grid(row=0, column=1, sticky="ew", **PAD)
        ttk.Button(self, text="Browse...", command=self._browse_save if save else self._browse).grid(row=0, column=2, **PAD)
        if not save:
            ttk.Button(self, text="Inspect", command=self._inspect).grid(row=0, column=3, **PAD)
        if hint:
            ttk.Label(self, text=hint, style="Hint.TLabel").grid(row=1, column=1, columnspan=3, sticky="w", padx=px(6))
        self.columnconfigure(1, weight=1)

    def _browse(self):
        p = filedialog.askopenfilename(filetypes=ST_FILES)
        if p:
            self.var.set(p)

    def _browse_save(self):
        p = filedialog.asksaveasfilename(filetypes=ST_FILES, defaultextension=".safetensors")
        if p:
            self.var.set(p)

    def _inspect(self):
        p = self.var.get().strip()
        if not p:
            return
        from .inspect_file import inspect_path
        self.app.log(inspect_path(p)["text"])

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, v: str | None):
        self.var.set(v or "")


class ValueSlider(ttk.Frame):
    """Slider -10..10 in 0.05 steps with a synchronized entry."""

    def __init__(self, master, label: str, initial: float = 1.0):
        super().__init__(master)
        self.value = tk.DoubleVar(value=initial)
        self.txt = tk.StringVar(value=f"{initial:.2f}")
        ttk.Label(self, text=label, width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        self.scale = ttk.Scale(self, from_=-10.0, to=10.0, variable=self.value, command=self._on_scale)
        self.scale.grid(row=0, column=1, sticky="ew", **PAD)
        e = ttk.Entry(self, textvariable=self.txt, width=7, justify="right")
        e.grid(row=0, column=2, **PAD)
        e.bind("<Return>", self._on_entry)
        e.bind("<FocusOut>", self._on_entry)
        self.columnconfigure(1, weight=1)

    def _on_scale(self, _v):
        v = _round_step(self.value.get())
        self.value.set(v)
        self.txt.set(f"{v:.2f}")

    def _on_entry(self, _e=None):
        try:
            v = max(-10.0, min(10.0, float(self.txt.get().replace(",", "."))))
        except ValueError:
            v = self.value.get()
        self.set(v)

    def get(self) -> float:
        return float(self.value.get())

    def set(self, v: float):
        v = _round_step(v)
        self.value.set(v)
        self.txt.set(f"{v:.2f}")


class ShapingRow(ttk.LabelFrame):
    """File, strength or weight, and block shaping that unfolds when a preset is chosen.

    kind "lora": strength. kind "ckpt": weight plus the non block weight field.
    """

    def __init__(self, master, app, title: str, kind: str = "lora", removable=None):
        super().__init__(master, text=title, padding=px(6))
        self.app, self.kind = app, kind
        self.slot = FileSlot(self, app, "file")
        self.slot.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.slider = ValueSlider(self, "weight" if kind == "ckpt" else "strength", 1.0)
        self.slider.grid(row=1, column=0, columnspan=3, sticky="ew")
        # compatibility aliases used by tests and older code
        self.value, self.value_txt, self._on_entry, self.set_value = self.slider.value, self.slider.txt, self.slider._on_entry, self.slider.set

        ttk.Label(self, text="blocks", width=LABEL_W).grid(row=2, column=0, sticky="w", **PAD)
        line = ttk.Frame(self)
        line.grid(row=2, column=1, sticky="w")
        self.preset = tk.StringVar(value="FULL")
        ttk.Combobox(line, textvariable=self.preset, values=BLOCK_PRESETS, state="readonly", width=14).pack(side="left", **PAD)
        self.preset_hint = ttk.Label(line, text="", style="Hint.TLabel")
        self.preset_hint.pack(side="left", padx=px(6))
        mb = ttk.Menubutton(line, text="Recipes")
        menu = tk.Menu(mb, tearoff=0)
        for key, label in RECIPE_LABELS.items():
            menu.add_command(label=label, command=lambda k=key: self.apply_recipe(k))
        menu.add_command(label="FULL (no shaping)", command=lambda: self.apply_recipe(None))
        mb["menu"] = menu
        mb.pack(side="left", padx=(px(12), 0))
        self.btn_remove = None
        if removable is not None:
            self.btn_remove = ttk.Button(self, text="Remove", command=removable)
            self.btn_remove.grid(row=2, column=2, sticky="e", **PAD)

        # shaping details, shown when preset != FULL
        self.detail = ttk.Frame(self)
        self.modifier = tk.StringVar(value="Suppress")
        self.contrast = tk.DoubleVar(value=0.5)
        self.boost = tk.DoubleVar(value=1.0)
        ttk.Label(self.detail, text="modifier", width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        mline = ttk.Frame(self.detail)
        mline.grid(row=0, column=1, columnspan=3, sticky="w")
        self.cb_mod = ttk.Combobox(mline, textvariable=self.modifier, values=MODIFIERS, state="readonly", width=12)
        self.cb_mod.pack(side="left", **PAD)
        self.mod_hint = ttk.Label(mline, text="", style="Hint.TLabel")
        self.mod_hint.pack(side="left", padx=px(6))
        ttk.Label(self.detail, text="contrast", width=LABEL_W).grid(row=1, column=0, sticky="w", **PAD)
        self.sc_contrast = ttk.Scale(self.detail, from_=0.0, to=1.0, variable=self.contrast, command=lambda _v: self._changed())
        self.sc_contrast.grid(row=1, column=1, columnspan=2, sticky="ew", **PAD)
        self.lbl_contrast = ttk.Label(self.detail, text="0.50", width=5)
        self.lbl_contrast.grid(row=1, column=3, sticky="w")
        self.lbl_boost_t = ttk.Label(self.detail, text="boost", width=LABEL_W)
        self.lbl_boost_t.grid(row=2, column=0, sticky="w", **PAD)
        self.sc_boost = ttk.Scale(self.detail, from_=0.25, to=2.0, variable=self.boost, command=lambda _v: self._changed())
        self.sc_boost.grid(row=2, column=1, columnspan=2, sticky="ew", **PAD)
        self.lbl_boost = ttk.Label(self.detail, text="1.00", width=5)
        self.lbl_boost.grid(row=2, column=3, sticky="w")
        ttk.Label(self.detail, text="per block", width=LABEL_W).grid(row=3, column=0, sticky="nw", **PAD)
        self.curve = tk.Canvas(self.detail, height=px(40), width=px(280), highlightthickness=1,
                               highlightbackground=app.C["border"], bg=app.C["canvas"])
        app.themed.append(self)
        self.curve.grid(row=3, column=1, columnspan=2, sticky="ew", **PAD)
        self.curve.bind("<Configure>", lambda _e: self.draw_curve())    # redraw at the real width (resize, rescale, recipe before mapping)
        self.info = ttk.Label(self.detail, text="", style="Hint.TLabel")
        self.info.grid(row=3, column=3, sticky="w")
        if kind == "ckpt":
            ttk.Label(self.detail, text="non-block", width=LABEL_W).grid(row=4, column=0, sticky="w", **PAD)
            self.non_block_txt = tk.StringVar(value="")
            ttk.Entry(self.detail, textvariable=self.non_block_txt, width=7, justify="right").grid(row=4, column=1, sticky="w", **PAD)
            ttk.Label(self.detail, text="weight for the text side and the projections (outside the block mask); empty = same as weight",
                      style="Hint.TLabel").grid(row=4, column=2, columnspan=2, sticky="w", padx=px(6))
        else:
            self.non_block_txt = None
        self.detail.columnconfigure(1, weight=1)
        self.detail.columnconfigure(2, weight=1)
        self.columnconfigure(1, weight=1)
        self.preset.trace_add("write", lambda *_: self._changed())
        self.modifier.trace_add("write", lambda *_: self._changed())
        self._changed()

    # ---- state
    def _changed(self):
        flat = self.preset.get() == "FULL"
        hints = {"FULL": "no shaping: every block at full strength", "COMPOSITION": "blocks 0-8: layout, poses, geometry",
                 "CHARACTER": "blocks 9-18: identity, faces", "STYLE": "blocks 19-27: textures, grain, rendering style"}
        self.preset_hint.configure(text=hints.get(self.preset.get(), ""))
        if flat:
            self.detail.grid_forget()
            self.cb_mod.configure(state="disabled")
            return
        self.detail.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.cb_mod.configure(state="readonly")
        mod = self.modifier.get()
        self.mod_hint.configure(text={"Suppress": "weaken the LoRA inside the zone, keep the rest",
                                      "Emphasize": "stronger inside the zone, weaker outside, same average",
                                      "Isolate": "apply inside the zone only"}.get(mod, ""))
        emph = mod == "Emphasize"
        for w in (self.lbl_boost_t, self.sc_boost, self.lbl_boost):
            w.grid() if emph else w.grid_remove()
        self.lbl_contrast.configure(text=f"{self.contrast.get():.2f}")
        self.lbl_boost.configure(text=f"{self.boost.get():.2f}")
        self.draw_curve()

    def apply_recipe(self, key: str | None):
        s = RECIPES[key] if key else Shaping()
        self.modifier.set(s.modifier)
        self.contrast.set(s.contrast)
        self.boost.set(s.boost)
        self.preset.set(s.preset)
        self._changed()

    def shaping(self) -> Shaping:
        s = Shaping(preset=self.preset.get(), modifier=self.modifier.get(),
                    contrast=round(float(self.contrast.get()), 3), boost=round(float(self.boost.get()), 3))
        if self.non_block_txt is not None and self.non_block_txt.get().strip():
            try:
                nb = float(self.non_block_txt.get().replace(",", "."))
                w = self.slider.get()
                s.non_block = (nb / w) if abs(w) > 1e-9 else 1.0   # stored as a factor on the weight
            except ValueError:
                pass
        return s

    def draw_curve(self):
        c = self.curve
        c.delete("all")
        try:
            f = self.shaping().factors(KREA2_BLOCKS)
        except ValueError:
            return
        w = int(c.winfo_width()) if c.winfo_width() > 1 else px(280)
        h = int(c.winfo_height()) if c.winfo_height() > 1 else px(40)
        bw = w / len(f)
        C = self.app.C
        c.create_line(0, h - h / 3, w, h - h / 3, fill=C["border"])
        for i, v in enumerate(f):
            bh = min(v, 2.0) / 2.0 * (h - 4)
            col = C["bar_up"] if v > 1.0 + 1e-9 else (C["bar_down"] if v < 1.0 - 1e-9 else C["bar_flat"])
            c.create_rectangle(i * bw + 1, h - bh, (i + 1) * bw - 1, h, fill=col, outline="")
        self.info.configure(text=f"min {min(f):.2f}\nmax {max(f):.2f}")

    def apply_theme(self, C: dict):
        self.curve.configure(bg=C["canvas"], highlightbackground=C["border"])
        self.draw_curve()

    # ---- conversions
    def to_lora_input(self) -> LoraInput | None:
        p = self.slot.get()
        return LoraInput(p, self.slider.get(), self.shaping()) if p else None

    def to_ckpt_input(self) -> CkptInput | None:
        p = self.slot.get()
        return CkptInput(p, self.slider.get(), self.shaping()) if p else None

    def load(self, path: str | None, value: float, shaping: Shaping | None):
        self.slot.set(path)
        self.slider.set(value)
        s = shaping or Shaping()
        self.modifier.set(s.modifier)
        self.contrast.set(s.contrast)
        self.boost.set(s.boost)
        if self.non_block_txt is not None:
            self.non_block_txt.set("" if s.non_block is None else f"{s.non_block * value:.2f}")
        self.preset.set(s.preset)
        self._changed()

    def clear(self):
        self.load(None, 1.0, None)


class RowList(ttk.Frame):
    """LoRA rows added on demand, up to a maximum."""

    def __init__(self, master, app, max_rows: int, title="LoRA", initial: int = 1):
        super().__init__(master)
        self.app, self.max_rows, self.title = app, max_rows, title
        self.rows: list[ShapingRow] = []
        self.rows_frame = ttk.Frame(self)
        self.rows_frame.pack(fill="x")
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(px(2), 0))
        self.btn_add = ttk.Button(bar, text=f"+ Add {title}", command=self.add_row)
        self.btn_add.pack(side="left", padx=px(6))
        for _ in range(initial):
            self.add_row()

    def add_row(self):
        if len(self.rows) >= self.max_rows:
            return
        r = ShapingRow(self.rows_frame, self.app, f"{self.title} {len(self.rows) + 1}", kind="lora",
                       removable=lambda: self.remove_row(r) if len(self.rows) > 1 else None)
        r.pack(fill="x", pady=px(3))
        self.rows.append(r)
        self._refresh()

    def remove_row(self, row: ShapingRow):
        if row in self.rows and len(self.rows) > 1:
            self.rows.remove(row)
            row.destroy()
            self._refresh()

    def _refresh(self):
        for i, r in enumerate(self.rows):
            r.configure(text=f"{self.title} {i + 1}")
            if r.btn_remove is not None:
                r.btn_remove.grid() if len(self.rows) > 1 else r.btn_remove.grid_remove()
        self.btn_add.configure(state="normal" if len(self.rows) < self.max_rows else "disabled")

    def set_count(self, n: int):
        n = max(1, min(n, self.max_rows))
        while len(self.rows) < n:
            self.add_row()
        while len(self.rows) > n:
            self.remove_row(self.rows[-1])

    def inputs(self) -> list[LoraInput]:
        return [i for i in (r.to_lora_input() for r in self.rows) if i is not None]

    def load(self, inputs: list[LoraInput]):
        self.set_count(len(inputs))
        for row, inp in zip(self.rows, inputs):
            row.load(inp.path, inp.strength, inp.shaping)
        if not inputs:
            self.rows[0].clear()


def _labeled(parent, row, text, widget_factory, hint: str = "", col=0):
    """label | widget | hint, on one grid row. Returns the widget."""
    ttk.Label(parent, text=text, width=LABEL_W).grid(row=row, column=col, sticky="w", **PAD)
    w = widget_factory(parent)
    w.grid(row=row, column=col + 1, sticky="w", **PAD)
    if hint:
        ttk.Label(parent, text=hint, style="Hint.TLabel").grid(row=row, column=col + 2, sticky="w", padx=px(6))
    return w


# ============================================================================== tabs
class LoraMergeTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=px(8))
        self.app = app
        self.list = RowList(self, app, 6, "LoRA", initial=2)
        self.list.pack(fill="x")
        self.rows = self.list.rows   # same list object
        self.average = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Average: normalize the strengths to sum to 1 (epochs of one training run)",
                        variable=self.average).pack(anchor="w", padx=px(6), pady=(px(6), px(2)))

        out = ttk.LabelFrame(self, text="Output", padding=px(6))
        out.pack(fill="x", pady=px(6))
        ttk.Label(out, text="rank", width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        rk = ttk.Frame(out)
        rk.grid(row=0, column=1, columnspan=2, sticky="w")
        self.rank_mode = tk.StringVar(value="concat")
        ttk.Radiobutton(rk, text="keep the concatenated rank (exact)", variable=self.rank_mode, value="concat", command=self._rank_changed).pack(side="left", padx=px(6))
        ttk.Radiobutton(rk, text="fixed", variable=self.rank_mode, value="fixed", command=self._rank_changed).pack(side="left", padx=(px(12), px(2)))
        self.rank = tk.StringVar(value="32")
        self.rank_entry = ttk.Entry(rk, textvariable=self.rank, width=6, justify="right")
        self.rank_entry.pack(side="left")
        ttk.Radiobutton(rk, text="per group, from the analysis", variable=self.rank_mode, value="groups", command=self._rank_changed).pack(side="left", padx=(px(12), px(2)))
        self.group_ranks: dict = {}
        self.group_lbl = ttk.Label(out, text="", style="Hint.TLabel")
        self.group_lbl.grid(row=6, column=0, columnspan=3, sticky="w", padx=px(6))

        self.analysis_frame = ttk.Frame(out)
        ttk.Label(self.analysis_frame, text="energy target", width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        self.target = tk.StringVar(value="0.99")
        ttk.Combobox(self.analysis_frame, textvariable=self.target, values=["0.9", "0.95", "0.99", "0.995"], width=6).grid(row=0, column=1, sticky="w", **PAD)
        self.criterion = tk.StringVar(value="weighted")
        ttk.Combobox(self.analysis_frame, textvariable=self.criterion, values=["weighted", "per_module"], state="readonly", width=11).grid(row=0, column=2, sticky="w", **PAD)
        self.denoised = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.analysis_frame, text="denoised", variable=self.denoised).grid(row=0, column=3, sticky="w", **PAD)
        ttk.Label(self.analysis_frame, text="Analyze picks one rank per group for this target; weighted = pooled energy, per_module = every module; "
                  "denoised = above the noise edge (LoRA spectra have none)", style="Hint.TLabel", wraplength=px(520), justify="left").grid(row=0, column=4, sticky="w", padx=px(6))
        self.analysis_frame.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.modules = _labeled(out, 2, "modules", lambda p: ttk.Combobox(p, values=["intersection", "union"], state="readonly", width=12),
                                "intersection = modules present in every input; union = keep modules present in any input")
        self.modules.set("intersection")
        self.naming = _labeled(out, 3, "naming", lambda p: ttk.Combobox(p, values=["comfy", "kohya", "input"], state="readonly", width=12),
                               "key convention of the output file")
        self.naming.set("comfy")
        self.dtype = _labeled(out, 4, "dtype", lambda p: ttk.Combobox(p, values=["fp16", "bf16", "fp32"], state="readonly", width=12))
        self.dtype.set("fp16")
        self.out = FileSlot(out, app, "output", save=True)
        self.out.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(px(6), 0))
        out.columnconfigure(2, weight=1)

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=px(4))
        ttk.Button(btns, text="Analyze", command=self.analyze).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Plan", command=self.plan).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Merge", style="Accent.TButton", command=self.run).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Save recipe...", command=lambda: app.save_recipe(self)).pack(side="right", padx=px(6))
        ttk.Button(btns, text="Load recipe...", command=lambda: app.load_recipe(self)).pack(side="right", padx=px(6))
        self._rank_changed()

    def _rank_changed(self):
        mode = self.rank_mode.get()
        self.rank_entry.configure(state="normal" if mode == "fixed" else "disabled")

    def add_row(self):
        self.list.add_row()

    def remove_row(self):
        if len(self.list.rows) > 1:
            self.list.remove_row(self.list.rows[-1])

    def inputs(self) -> list[LoraInput]:
        return self.list.inputs()

    def options(self) -> LoraMergeOptions:
        return LoraMergeOptions(average=self.average.get(), rank_mode=self.rank_mode.get(),
                                rank=int(self.rank.get()) if self.rank.get().strip().isdigit() else None,
                                group_ranks=dict(self.group_ranks), modules=self.modules.get(),
                                naming=self.naming.get(), dtype=self.dtype.get())

    def to_recipe(self) -> dict:
        return {"function": "lora_merge", "inputs": [i.to_dict() for i in self.inputs()],
                "options": self.options().to_dict(), "output": self.out.get()}

    def from_recipe(self, r: dict):
        self.list.load([LoraInput.from_dict(d) for d in r.get("inputs", [])])
        o = LoraMergeOptions.from_dict(r.get("options"))
        self.average.set(o.average)
        self.rank_mode.set(o.rank_mode)
        self.rank.set(str(o.rank or 32))
        self.group_ranks = dict(o.group_ranks)
        self.group_lbl.configure(text=_fmt_ranks(self.group_ranks))
        self.modules.set(o.modules)
        self.naming.set(o.naming)
        self.dtype.set(o.dtype)
        self.out.set(r.get("output"))
        self._rank_changed()

    def _check(self) -> bool:
        if not self.inputs():
            messagebox.showwarning("LoRA merge", "Add at least one LoRA file.")
            return False
        return True

    def analyze(self):
        if not self._check():
            return
        inputs, opts = self.inputs(), self.options()
        target, crit, dn = float(self.target.get()), self.criterion.get(), bool(self.denoised.get())

        gpu = self.app.use_gpu()          # read the Tk variable on the main thread, not in the worker
        def job(progress, cancel, log):
            from .lora_merge import analyze_lora_merge
            rep = analyze_lora_merge(inputs, opts, gpu, progress=progress, cancel=cancel)
            use_dn = dn and rep.has_noise_model
            plan = rep.rank_plan(target, crit, denoised=use_dn)
            return rep.text() + f"\n\nrank per group for {target:.3f} energy ({crit}, {'denoised' if use_dn else 'raw'} spectrum): {plan}", plan, rep

        def done(res):
            text, plan, rep = res
            self.app.log(text)
            self.group_ranks = {g: int(r) for g, r in plan.items()}
            self.group_lbl.configure(text=_fmt_ranks(self.group_ranks))
            self.rank_mode.set("groups")
            self._rank_changed()
            self.app.tab_spectrum.set_report(rep, {"uniform": opts.rank, "groups": dict(self.group_ranks)})
            self.app.log("the spectra are on the Spectrum tab")
        self.app.run_job("Analyzing LoRA merge", job, done)

    def plan(self):
        if not self._check():
            return
        inputs, opts = self.inputs(), self.options()
        self.app.run_job("Planning", lambda p, c, l: __import__("k2merge.plan", fromlist=["plan_lora_merge"]).plan_lora_merge(inputs, opts), self.app.log)

    def run(self):
        if not self._check():
            return
        out = self.out.get()
        if not out:
            messagebox.showwarning("LoRA merge", "Choose an output file.")
            return
        inputs, opts = self.inputs(), self.options()

        gpu = self.app.use_gpu()          # read the Tk variable on the main thread, not in the worker
        def job(progress, cancel, log):
            from .lora_merge import merge_loras
            return merge_loras(inputs, out, opts, gpu, progress, cancel, log)

        def done(res):
            self.app.log(f"wrote {res.path}: {res.modules} modules, rank {res.rank_min}-{res.rank_max}, "
                         f"energy kept >= {res.kept_min * 100:.2f}%, {len(res.dropped)} dropped, {res.seconds:.1f}s")
        self.app.run_job("Merging LoRAs", job, done)


class ExtractTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=px(8))
        self.app = app
        card = ttk.LabelFrame(self, text="Checkpoints", padding=px(6))
        card.pack(fill="x", pady=px(4))
        self.base = FileSlot(card, app, "base", hint="the model the LoRA will be applied to")
        self.base.pack(fill="x")
        self.target = FileSlot(card, app, "target", hint="the fine tuned model; the LoRA approximates target minus base")
        self.target.pack(fill="x")

        opts = ttk.LabelFrame(self, text="Extraction", padding=px(6))
        opts.pack(fill="x", pady=px(4))
        rk = ttk.Frame(opts)
        ttk.Label(opts, text="rank", width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        rk.grid(row=0, column=1, columnspan=2, sticky="w")
        self.rank = tk.StringVar(value="32")
        ttk.Entry(rk, textvariable=self.rank, width=6, justify="right").pack(side="left", padx=px(6))
        self.use_groups = tk.BooleanVar(value=False)
        ttk.Checkbutton(rk, text="use the per group ranks from the analysis", variable=self.use_groups).pack(side="left", padx=px(12))
        self.group_ranks: dict = {}
        self.group_lbl = ttk.Label(opts, text="", style="Hint.TLabel")
        self.group_lbl.grid(row=8, column=0, columnspan=3, sticky="w", padx=px(6))
        self.filter = _labeled(opts, 1, "modules", lambda p: ttk.Combobox(p, values=["all", "attn", "blocks", "custom"], state="readonly", width=12),
                               "all = every linear (264); attn = attention only (140); blocks = block linears (224); custom = regex")
        self.filter.set("all")
        self.filter.bind("<<ComboboxSelected>>", lambda _e: self._filter_changed())
        self.custom = ttk.Frame(opts)
        ttk.Label(self.custom, text="include", width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        self.include = tk.StringVar()
        ttk.Entry(self.custom, textvariable=self.include, width=28).grid(row=0, column=1, sticky="w", **PAD)
        ttk.Label(self.custom, text="exclude", width=LABEL_W).grid(row=1, column=0, sticky="w", **PAD)
        self.exclude = tk.StringVar()
        ttk.Entry(self.custom, textvariable=self.exclude, width=28).grid(row=1, column=1, sticky="w", **PAD)
        ttk.Label(self.custom, text="regular expressions on module names; include wins over exclude", style="Hint.TLabel").grid(row=0, column=2, rowspan=2, sticky="w", padx=px(6))
        self.method = _labeled(opts, 3, "SVD", lambda p: ttk.Combobox(p, values=["randomized", "full"], state="readonly", width=12),
                               "randomized is fast; full is exact and slower")
        self.method.set("randomized")
        self.naming = _labeled(opts, 4, "naming", lambda p: ttk.Combobox(p, values=["comfy", "kohya"], state="readonly", width=12))
        self.naming.set("comfy")
        self.dtype = _labeled(opts, 5, "dtype", lambda p: ttk.Combobox(p, values=["fp16", "bf16", "fp32"], state="readonly", width=12))
        self.dtype.set("fp16")
        an = ttk.Frame(opts)
        an.grid(row=6, column=0, columnspan=3, sticky="ew")
        ttk.Label(an, text="energy target", width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        self.target_e = tk.StringVar(value="0.99")
        ttk.Combobox(an, textvariable=self.target_e, values=["0.9", "0.95", "0.99", "0.995"], width=6).grid(row=0, column=1, sticky="w", **PAD)
        self.criterion = tk.StringVar(value="weighted")
        ttk.Combobox(an, textvariable=self.criterion, values=["weighted", "per_module"], state="readonly", width=11).grid(row=0, column=2, sticky="w", **PAD)
        self.denoised = tk.BooleanVar(value=True)
        ttk.Checkbutton(an, text="denoised", variable=self.denoised).grid(row=0, column=3, sticky="w", **PAD)
        self.null_spec = tk.BooleanVar(value=False)
        ttk.Checkbutton(an, text="null spectrum", variable=self.null_spec).grid(row=0, column=4, sticky="w", **PAD)
        ttk.Label(an, text="Analyze reports the rank needed per group with a noise edge from the storage formats; denoised = plan from the "
                  "spectra above the edge; null spectrum = also compute the spectrum of the modeled noise (twice the SVD time)",
                  style="Hint.TLabel", wraplength=px(420), justify="left").grid(row=0, column=5, sticky="w", padx=px(6))
        self.out = FileSlot(opts, app, "output", save=True)
        self.out.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(px(6), 0))
        opts.columnconfigure(2, weight=1)

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=px(4))
        ttk.Button(btns, text="Analyze", command=self.analyze).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Plan", command=self.plan).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Extract", style="Accent.TButton", command=self.run).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Save recipe...", command=lambda: app.save_recipe(self)).pack(side="right", padx=px(6))
        ttk.Button(btns, text="Load recipe...", command=lambda: app.load_recipe(self)).pack(side="right", padx=px(6))
        self._filter_changed()

    def _filter_changed(self):
        if self.filter.get() == "custom":
            self.custom.grid(row=2, column=0, columnspan=3, sticky="ew")
        else:
            self.custom.grid_forget()

    def options(self) -> ExtractOptions:
        return ExtractOptions(rank=int(self.rank.get()) if self.rank.get().strip().isdigit() else 32,
                              group_ranks=dict(self.group_ranks) if self.use_groups.get() else {},
                              filter=self.filter.get(), include=self.include.get().strip(), exclude=self.exclude.get().strip(),
                              method=self.method.get(), naming=self.naming.get(), dtype=self.dtype.get())

    def to_recipe(self) -> dict:
        return {"function": "extract", "base": self.base.get(), "target": self.target.get(),
                "options": self.options().to_dict(), "output": self.out.get()}

    def from_recipe(self, r: dict):
        self.base.set(r.get("base"))
        self.target.set(r.get("target"))
        o = ExtractOptions.from_dict(r.get("options"))
        self.rank.set(str(o.rank))
        self.group_ranks = dict(o.group_ranks)
        self.use_groups.set(bool(o.group_ranks))
        self.group_lbl.configure(text=_fmt_ranks(self.group_ranks))
        self.filter.set(o.filter)
        self.include.set(o.include)
        self.exclude.set(o.exclude)
        self.method.set(o.method)
        self.naming.set(o.naming)
        self.dtype.set(o.dtype)
        self.out.set(r.get("output"))
        self._filter_changed()

    def _check(self) -> bool:
        if not (self.base.get() and self.target.get()):
            messagebox.showwarning("Extract", "Choose the base and the target checkpoint.")
            return False
        return True

    def analyze(self):
        if not self._check():
            return
        b, t, o = self.base.get(), self.target.get(), self.options()
        target, crit, dn, null = float(self.target_e.get()), self.criterion.get(), bool(self.denoised.get()), bool(self.null_spec.get())

        gpu = self.app.use_gpu()          # read the Tk variable on the main thread, not in the worker
        def job(progress, cancel, log):
            from .extract import analyze_extract
            rep = analyze_extract(b, t, o, gpu, progress=progress, cancel=cancel, null_spectrum=null)
            use_dn = dn and rep.has_noise_model
            plan = rep.rank_plan(target, crit, denoised=use_dn)
            return rep.text() + f"\n\nrank per group for {target:.3f} energy ({crit}, {'denoised' if use_dn else 'raw'} spectrum): {plan}", plan, rep

        def done(res):
            text, plan, rep = res
            self.app.log(text)
            self.group_ranks = {g: int(r) for g, r in plan.items()}
            self.group_lbl.configure(text=_fmt_ranks(self.group_ranks))
            self.use_groups.set(True)
            self.app.tab_spectrum.set_report(rep, {"uniform": o.rank, "groups": dict(self.group_ranks)})
            self.app.log("the spectra are on the Spectrum tab")
        self.app.run_job("Analyzing extraction", job, done)

    def plan(self):
        if not self._check():
            return
        b, t, o = self.base.get(), self.target.get(), self.options()
        self.app.run_job("Planning", lambda p, c, l: __import__("k2merge.plan", fromlist=["plan_extract"]).plan_extract(b, t, o), self.app.log)

    def run(self):
        if not self._check():
            return
        out = self.out.get()
        if not out:
            messagebox.showwarning("Extract", "Choose an output file.")
            return
        b, t, o = self.base.get(), self.target.get(), self.options()

        gpu = self.app.use_gpu()          # read the Tk variable on the main thread, not in the worker
        def job(progress, cancel, log):
            from .extract import extract_lora
            return extract_lora(b, t, out, o, gpu, progress, cancel, log)

        def done(res):
            self.app.log(f"wrote {res.path}: {res.modules} modules, energy kept >= {res.kept_min * 100:.2f}%"
                         + (f", per group {({g: round(v, 4) for g, v in res.kept_by_group.items()})}" if res.kept_by_group else "")
                         + f", {len(res.skipped)} skipped, {res.seconds:.1f}s")
        self.app.run_job("Extracting", job, done)


class CkptTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=px(8))
        self.app = app
        cols = ttk.Frame(self)
        cols.pack(fill="both", expand=True)
        left = ttk.Frame(cols)
        left.pack(side="left", fill="both", expand=True, padx=(0, px(6)))
        right = ttk.Frame(cols)
        right.pack(side="left", fill="both", expand=True, padx=(px(6), 0))

        ck = ttk.LabelFrame(left, text="Checkpoints", padding=px(6))
        ck.pack(fill="x", pady=px(4))
        self.A = FileSlot(ck, app, "A  primary", hint="the model being changed; always weight 1")
        self.A.pack(fill="x")
        self.B = ShapingRow(ck, app, "B  secondary (optional)", kind="ckpt")
        self.B.pack(fill="x", pady=px(4))
        self.C = FileSlot(ck, app, "C  reference", hint="optional: the common ancestor of A and B, usually the official Turbo file")
        self.C.pack(fill="x")

        mt = ttk.LabelFrame(left, text="Method", padding=px(6))
        mt.pack(fill="x", pady=px(4))
        ttk.Label(mt, text="method", width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        self.method = tk.StringVar(value="add_difference")
        self.advanced = tk.BooleanVar(value=False)
        self.cb_method = ttk.Combobox(mt, textvariable=self.method, values=self._method_values(), state="readonly", width=18)
        self.cb_method.grid(row=0, column=1, sticky="w", **PAD)
        ttk.Checkbutton(mt, text="show advanced methods", variable=self.advanced, command=self._refresh_methods).grid(row=0, column=2, sticky="w", padx=px(12))
        self.method_lbl = ttk.Label(mt, text="", style="Hint.TLabel", wraplength=px(560), justify="left")
        self.method_lbl.grid(row=1, column=1, columnspan=2, sticky="w", padx=px(6), pady=(0, px(4)))
        self.params_frame = ttk.Frame(mt)
        self.params_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.params: dict[str, tk.StringVar] = {}
        self.param_widgets: dict[str, list] = {}
        for i, (name, default) in enumerate(PARAM_DEFAULTS):
            v = tk.StringVar(value=default)
            self.params[name] = v
            lbl = ttk.Label(self.params_frame, text=name, width=LABEL_W)
            ent = ttk.Entry(self.params_frame, textvariable=v, width=7, justify="right")
            hint = ttk.Label(self.params_frame, text=PARAM_HELP.get(name, ""), style="Hint.TLabel")
            lbl.grid(row=i, column=0, sticky="w", **PAD)
            ent.grid(row=i, column=1, sticky="w", **PAD)
            hint.grid(row=i, column=2, sticky="w", padx=px(6))
            self.param_widgets[name] = [lbl, ent, hint]
        self.dare_ties = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(self.params_frame, text="apply TIES after the DARE drop", variable=self.dare_ties)
        cb.grid(row=len(PARAM_DEFAULTS), column=0, columnspan=3, sticky="w", **PAD)
        self.param_widgets["dare_ties"] = [cb]
        self.lora_mode = tk.StringVar(value="after")
        self.lora_mode_widgets = [ttk.Label(self.params_frame, text="LoRAs", width=LABEL_W),
                                  ttk.Combobox(self.params_frame, textvariable=self.lora_mode, values=["after", "task_vectors"], state="readonly", width=12),
                                  ttk.Label(self.params_frame, text="after = added after the method; task_vectors = trimmed and merged with B's change", style="Hint.TLabel")]
        r = len(PARAM_DEFAULTS) + 1
        self.lora_mode_widgets[0].grid(row=r, column=0, sticky="w", **PAD)
        self.lora_mode_widgets[1].grid(row=r, column=1, sticky="w", **PAD)
        self.lora_mode_widgets[2].grid(row=r, column=2, sticky="w", padx=px(6))
        self.method.trace_add("write", lambda *_: self._method_changed())

        lf = ttk.LabelFrame(right, text="LoRAs (up to 4)", padding=px(6))
        lf.pack(fill="x", pady=px(4))
        self.lora_list = RowList(lf, app, 4, "LoRA", initial=1)
        self.lora_list.pack(fill="x")
        self.loras = self.lora_list.rows

        of = ttk.LabelFrame(left, text="Output", padding=px(6))
        of.pack(fill="x", pady=px(4))
        self.fmt = _labeled(of, 0, "format", lambda p: ttk.Combobox(p, values=list(OUTPUT_FORMATS), state="readonly", width=14),
                            "keep = same format as A; fp8_scaled and int8_convrot use the official Krea 2 layouts")
        self.fmt.set("bf16")
        self.fmt.bind("<<ComboboxSelected>>", lambda _e: self._format_changed())
        self.out = FileSlot(of, app, "output", save=True)
        self.out.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(px(4), px(2)))
        self.adv = Collapsible(of, "More output options")
        self.adv.grid(row=2, column=0, columnspan=3, sticky="ew")
        body = self.adv.body
        self.passthrough = _labeled(body, 0, "passthrough", lambda p: ttk.Combobox(p, values=list(PASSTHROUGH), state="readonly", width=10),
                                    "dtype of tensors that stay unquantized: official mirrors the official file")
        self.passthrough.set("official")
        self.fp8_set = _labeled(body, 1, "fp8 layers", lambda p: ttk.Combobox(p, values=["official", "blocks"], state="readonly", width=10),
                                "official = 256 layers incl. text fusion; blocks = the 224 block linears")
        self.fp8_set.set("official")
        self.int8_clip = _labeled(body, 2, "int8 clip", lambda p: ttk.Combobox(p, values=["mse", "absmax"], state="readonly", width=10),
                                  "mse reproduces the official int8 file; absmax is plain scaling")
        self.int8_clip.set("mse")
        self.fmt_rows = {name: list(body.grid_slaves(row=i)) for i, name in enumerate(("passthrough", "fp8", "int8"))}
        self.keep_meta = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="keep A's metadata", variable=self.keep_meta).grid(row=3, column=0, columnspan=3, sticky="w", **PAD)
        self.redact_meta = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="remove paths from the metadata A brought with it (a trainer's dataset folders, an upstream recipe)",
                        variable=self.redact_meta).grid(row=4, column=0, columnspan=3, sticky="w", **PAD)
        self.as_lora = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="write the result as a LoRA (result minus C, or minus A) instead of a checkpoint",
                        variable=self.as_lora, command=self._as_lora_changed).grid(row=5, column=0, columnspan=3, sticky="w", **PAD)
        self.lora_out_frame = ttk.Frame(body)
        ttk.Label(self.lora_out_frame, text="LoRA rank", width=LABEL_W).grid(row=0, column=0, sticky="w", **PAD)
        self.lora_rank = tk.StringVar(value="32")
        ttk.Entry(self.lora_out_frame, textvariable=self.lora_rank, width=6, justify="right").grid(row=0, column=1, sticky="w", **PAD)
        ttk.Label(self.lora_out_frame, text="naming", width=LABEL_W).grid(row=1, column=0, sticky="w", **PAD)
        self.lora_naming = tk.StringVar(value="comfy")
        ttk.Combobox(self.lora_out_frame, textvariable=self.lora_naming, values=["comfy", "kohya"], state="readonly", width=10).grid(row=1, column=1, sticky="w", **PAD)
        self.vectors_from = _labeled(body, 7, "vectors from", lambda p: ttk.Combobox(p, values=list(VECTOR_SOURCES), state="readonly", width=10),
                                     "norm scales, modulation vectors and biases: merged like the rest, or copied from A, B or C "
                                     "(C restores an official file's vectors after a fine tune or de-Turbo that skipped them)")
        self.vectors_from.set("merge")
        body.columnconfigure(2, weight=1)
        of.columnconfigure(2, weight=1)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=px(4))
        ttk.Button(btns, text="Report", command=self.report).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Plan", command=self.plan).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Merge / Convert", style="Accent.TButton", command=self.run).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Save recipe...", command=lambda: app.save_recipe(self)).pack(side="right", padx=px(6))
        ttk.Button(btns, text="Load recipe...", command=lambda: app.load_recipe(self)).pack(side="right", padx=px(6))
        self._method_changed()
        self._format_changed()
        self._as_lora_changed()

    # ---- dynamic visibility
    def _method_values(self):
        return [m for m in METHODS if self.advanced.get() or m not in ADVANCED]

    def _refresh_methods(self):
        self.cb_method.configure(values=self._method_values())
        if self.method.get() in ADVANCED and not self.advanced.get():
            self.method.set("add_difference")

    def _method_changed(self):
        m = self.method.get()
        txt = METHOD_LABELS.get(m, "")
        if m in NEEDS_C:
            txt += "  Without C, A is the reference."
        self.method_lbl.configure(text=txt)
        shown = set(METHOD_PARAMS.get(m, ()))
        for name, widgets in self.param_widgets.items():
            for w in widgets:
                w.grid() if name in shown else w.grid_remove()
        for w in self.lora_mode_widgets:
            w.grid() if m in ("ties", "dare") else w.grid_remove()

    def _format_changed(self):
        f = self.fmt.get()
        show = {"passthrough": f in ("keep", "fp8", "fp8_scaled", "int8_convrot"), "fp8": f == "fp8_scaled",
                "int8": f in ("int8_convrot", "keep")}
        for name, widgets in self.fmt_rows.items():
            for w in widgets:
                w.grid() if show[name] else w.grid_remove()

    def _as_lora_changed(self):
        if self.as_lora.get():
            self.lora_out_frame.grid(row=6, column=0, columnspan=3, sticky="ew")
        else:
            self.lora_out_frame.grid_forget()

    # ---- recipes
    def options(self) -> CkptMergeOptions:
        o = CkptMergeOptions(method=self.method.get(), output_format=self.fmt.get(), passthrough=self.passthrough.get(),
                             fp8_layer_set=self.fp8_set.get(), int8_clip=self.int8_clip.get(), keep_metadata=self.keep_meta.get(),
                             redact_inherited=self.redact_meta.get(),
                             lora_mode=self.lora_mode.get(), output_as_lora=self.as_lora.get(), vectors_from=self.vectors_from.get())
        for k, v in self.params.items():
            try:
                o.params[k] = int(v.get()) if k == "seed" else float(v.get().replace(",", "."))
            except ValueError:
                pass
        o.params["dare_ties"] = bool(self.dare_ties.get())
        o.lora_out = {"rank": int(self.lora_rank.get()) if self.lora_rank.get().strip().isdigit() else 32,
                      "naming": self.lora_naming.get(), "dtype": "fp16", "method": "randomized"}
        o.use_gpu = self.app.use_gpu()
        return o

    def inputs(self):
        A = CkptInput(self.A.get()) if self.A.get() else None
        B = self.B.to_ckpt_input()
        C = CkptInput(self.C.get()) if self.C.get() else None
        return A, B, C, self.lora_list.inputs()

    def to_recipe(self) -> dict:
        A, B, C, loras = self.inputs()
        return {"function": "ckpt_merge", "A": A.to_dict() if A else None, "B": B.to_dict() if B else None,
                "C": C.to_dict() if C else None, "loras": [l.to_dict() for l in loras],
                "options": self.options().to_dict(), "output": self.out.get()}

    def from_recipe(self, r: dict):
        A = CkptInput.from_dict(r.get("A"))
        B = CkptInput.from_dict(r.get("B"))
        C = CkptInput.from_dict(r.get("C"))
        self.A.set(A.path if A else None)
        if B:
            self.B.load(B.path, B.weight, B.shaping)
        else:
            self.B.clear()
        self.C.set(C.path if C else None)
        self.lora_list.load([LoraInput.from_dict(d) for d in r.get("loras", [])])
        o = CkptMergeOptions.from_dict(r.get("options"))
        if o.method in ADVANCED:
            self.advanced.set(True)
            self._refresh_methods()
        self.method.set(o.method)
        for k, v in self.params.items():
            if k in o.params:
                v.set(str(o.params[k]))
        self.dare_ties.set(bool(o.params.get("dare_ties", False)))
        self.lora_mode.set(o.lora_mode)
        self.fmt.set(o.output_format)
        self.passthrough.set(o.passthrough)
        self.fp8_set.set(o.fp8_layer_set)
        self.int8_clip.set(o.int8_clip)
        self.vectors_from.set(o.vectors_from if o.vectors_from in VECTOR_SOURCES else "merge")
        self.as_lora.set(o.output_as_lora)
        self.lora_rank.set(str(o.lora_out.get("rank", 32)))
        self.lora_naming.set(o.lora_out.get("naming", "comfy"))
        self.keep_meta.set(o.keep_metadata)
        self.redact_meta.set(o.redact_inherited)
        self.out.set(r.get("output"))
        self._method_changed()
        self._format_changed()
        self._as_lora_changed()
        if (o.output_as_lora or o.passthrough != "official" or o.fp8_layer_set != "official" or o.int8_clip != "mse" or not o.keep_metadata
                or not o.redact_inherited
                or o.vectors_from != "merge"):
            self.adv.set_open(True)

    # ---- actions
    def _check(self) -> bool:
        if not self.A.get():
            messagebox.showwarning("Checkpoint merge", "Choose checkpoint A.")
            return False
        return True

    def report(self):
        A, B, C, loras = self.inputs()
        if A is None or B is None:
            messagebox.showwarning("Report", "The pre merge report needs A and B.")
            return
        gpu = self.app.use_gpu()
        self.app.run_job("Pre merge report", lambda p, c, l: __import__("k2merge.ckpt_merge", fromlist=["premerge_report"]).premerge_report(A, B, C, gpu, progress=p, cancel=c), self.app.log)

    def plan(self):
        if not self._check():
            return
        A, B, C, loras = self.inputs()
        o = self.options()
        self.app.run_job("Planning", lambda p, c, l: __import__("k2merge.plan", fromlist=["plan_ckpt_merge"]).plan_ckpt_merge(A, B, C, loras, o), self.app.log)

    def run(self):
        if not self._check():
            return
        out = self.out.get()
        if not out:
            messagebox.showwarning("Checkpoint merge", "Choose an output file.")
            return
        A, B, C, loras = self.inputs()
        o = self.options()

        def job(progress, cancel, log):
            from .ckpt_merge import merge_checkpoints
            return merge_checkpoints(A, B, C, loras, out, o, progress, cancel, log)

        def done(res):
            if hasattr(res, "verify"):
                v = res.verify
                self.app.log(f"wrote {res.path}: {res.tensors} tensors in {res.seconds:.1f}s; verification "
                             + ("passed" if v["ok"] else "FAILED: " + "; ".join(v["problems"][:5]))
                             + (f" (max read back error {v['max_rel_err']:.4f} over {v['checked']} tensors)" if v["checked"] else ""))
            else:
                self.app.log(f"wrote {res.path}: {res.modules} modules as LoRA, energy kept >= {res.kept_min * 100:.2f}%")
        self.app.run_job("Merging checkpoints", job, done)


# ============================================================================== app
class MergeApp(tk.Tk):
    def __init__(self, theme: str | None = None, scale: str | None = None):
        _enable_dpi_awareness()
        super().__init__()
        self.scale_setting = scale
        self.title(f"Krea 2 Merge Tool {__version__}")
        self.msg_queue: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._start = None
        self.themed: list = []                      # widgets with an apply_theme(C) method
        self.settings = load_settings()
        if theme in (None, "native"):
            theme = self.settings.get("theme", "light")
        self.theme_name = theme if theme in THEMES else "light"
        self.C = THEMES[self.theme_name]
        self._setup_scaling_and_fonts()
        self.style = ttk.Style(self)
        self._native_theme = next((n for n in ("vista", "winnative", "clam") if n in self.style.theme_names()), "default")
        self._build()
        self.apply_theme(self.theme_name)
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        self.minsize(min(w, self.winfo_screenwidth() - 80), min(h, self.winfo_screenheight() - 120))
        self.after(80, self._poll)

    def _setup_scaling_and_fonts(self):
        """Named fonts get pixel sizes = points x display DPI x the UI scale, so a rescale can reconfigure them live
        (Tk caches point sized fonts by description and would keep the old size). Pixel paddings go through px() and PAD."""
        global _S
        self.scale_setting = self.scale_setting if self.scale_setting in SCALES else self.settings.get("scale", "auto")
        if self.scale_setting not in SCALES:
            self.scale_setting = "auto"
        _S = self.ui_scale = ui_scale_factor(self.scale_setting)
        PAD.update(padx=px(6), pady=px(3))
        try:
            self._dpi = float(self.winfo_fpixels("1i"))
        except tk.TclError:
            self._dpi = 96.0
        self._apply_tk_scaling()
        family = "Segoe UI" if sys.platform == "win32" else tkfont.nametofont("TkDefaultFont").actual()["family"]
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family=family, size=self.font_px(10))
            except tk.TclError:
                pass
        try:
            tkfont.nametofont("TkFixedFont").configure(family="Consolas" if sys.platform == "win32" else "Courier", size=self.font_px(10))
        except tk.TclError:
            pass

    def font_px(self, points: float) -> int:
        """A Tk font size in pixels (negative by Tk convention) for a point size at the current DPI and UI scale."""
        return -max(1, int(round(points * self._dpi / 72.0 * self.ui_scale)))

    def _apply_tk_scaling(self):
        try:
            self.tk.call("tk", "scaling", self._dpi / 72.0 * self.ui_scale)
        except tk.TclError:
            pass

    def rescale(self, factor: float):
        """Change the UI scale of the running window: fonts, the paddings of every placed widget, canvases and wrap
        widths are rescaled in place; the theme is reapplied for the style paddings and bold fonts."""
        global _S
        old = self.ui_scale
        if abs(factor - old) < 1e-6:
            return
        _S = self.ui_scale = factor
        PAD.update(padx=px(6), pady=px(3))
        self._apply_tk_scaling()
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkFixedFont"):
            try:
                tkfont.nametofont(name).configure(size=self.font_px(10))
            except tk.TclError:
                pass

        def re(v):
            # a padding built as px(n) under the old scale: recover n (exact for scales >= 1), rescale it
            if isinstance(v, (tuple, list)):
                return tuple(re(x) for x in v)
            try:
                n = int(round(int(v) / old))
            except (TypeError, ValueError):
                return v
            return int(round(n * factor))

        def walk(w):
            try:
                mgr = w.winfo_manager()
                if mgr == "pack":
                    info = w.pack_info()
                    w.pack_configure(padx=re(info.get("padx", 0)), pady=re(info.get("pady", 0)))
                elif mgr == "grid":
                    info = w.grid_info()
                    w.grid_configure(padx=re(info.get("padx", 0)), pady=re(info.get("pady", 0)))
            except tk.TclError:
                pass
            for opt in ("padding", "wraplength"):
                try:
                    v = w.cget(opt)
                except tk.TclError:
                    continue
                if v not in ("", 0, (), None):
                    try:
                        w.configure({opt: re(v)})
                    except tk.TclError:
                        pass
            if isinstance(w, tk.Canvas):
                try:
                    w.configure(width=re(w.cget("width")), height=re(w.cget("height")))
                except tk.TclError:
                    pass
            if isinstance(w, ttk.Entry):        # Combobox and Spinbox too: re-setting the font recomputes the text layout
                try:
                    w.configure(font=w.cget("font") or "TkTextFont")
                except tk.TclError:
                    pass
            for ch in w.winfo_children():
                walk(ch)
        walk(self)
        self.apply_theme(self.theme_name)
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        self.minsize(min(w, self.winfo_screenwidth() - 80), min(h, self.winfo_screenheight() - 120))

    # ---- theme
    def apply_theme(self, name: str):
        """Light = native Windows theme. Dark = the clam engine with the original tool's palette."""
        self.theme_name = name if name in THEMES else "light"
        C = self.C = THEMES[self.theme_name]
        st = self.style
        family = tkfont.nametofont("TkDefaultFont").actual()["family"]
        bold = (family, self.font_px(10), "bold")
        self.title_lbl.configure(font=(family, self.font_px(13), "bold"))
        if self.theme_name == "light":
            st.theme_use(self._native_theme)          # ttk keeps style settings per theme: nothing to undo
            self.configure(bg=C["bg"])
            st.configure("TLabelframe.Label", foreground=C["accent"])
            st.configure("Hint.TLabel", foreground=C["fg_muted"])
            st.configure("Link.TLabel", foreground=C["link"])
            st.configure("Accent.TButton", font=bold)
            st.configure("TNotebook.Tab", padding=(px(14), px(6)))
            st.configure("Treeview", rowheight=abs(self.font_px(10)) + px(8))
            self.option_add("*TCombobox*Listbox.background", "#ffffff")
            self.option_add("*TCombobox*Listbox.foreground", "#000000")
        else:
            st.theme_use("clam")
            self.configure(bg=C["bg"])
            st.configure(".", background=C["bg"], foreground=C["fg"], fieldbackground=C["surface_2"],
                         bordercolor=C["border"], lightcolor=C["surface"], darkcolor=C["bg"], troughcolor=C["trough"],
                         focuscolor=C["accent"], selectbackground=C["accent"], selectforeground=C["accent_fg"])
            st.configure("TFrame", background=C["bg"])
            st.configure("TLabel", background=C["bg"], foreground=C["fg"])
            st.configure("Hint.TLabel", background=C["bg"], foreground=C["fg_muted"])
            st.configure("Link.TLabel", background=C["bg"], foreground=C["link"])
            st.configure("TLabelframe", background=C["bg"], bordercolor=C["border"])
            st.configure("TLabelframe.Label", background=C["bg"], foreground=C["accent"])
            st.configure("TButton", background=C["surface_2"], foreground=C["fg"], bordercolor=C["border"], padding=(px(10), px(4)))
            st.map("TButton", background=[("active", C["border"]), ("disabled", C["surface"])], foreground=[("disabled", C["fg_muted"])])
            st.configure("Accent.TButton", background=C["accent"], foreground=C["accent_fg"], font=bold, bordercolor=C["accent"], padding=(px(10), px(4)))
            st.map("Accent.TButton", background=[("active", C["accent_2"]), ("disabled", C["surface_2"])])
            st.configure("TMenubutton", background=C["surface_2"], foreground=C["fg"], arrowcolor=C["fg"], padding=(px(8), px(4)))
            st.configure("TEntry", fieldbackground=C["surface_2"], foreground=C["fg"], insertcolor=C["fg"], bordercolor=C["border"])
            st.map("TEntry", fieldbackground=[("disabled", C["surface"]), ("readonly", C["surface"]), ("focus", C["surface_2"])],
                   foreground=[("disabled", C["fg_muted"])], bordercolor=[("focus", C["accent"])])
            st.configure("TSpinbox", fieldbackground=C["surface_2"], foreground=C["fg"], arrowcolor=C["fg"], bordercolor=C["border"])
            st.configure("TCombobox", fieldbackground=C["surface_2"], foreground=C["fg"], background=C["surface_2"],
                         arrowcolor=C["fg"], bordercolor=C["border"], selectbackground=C["surface_2"], selectforeground=C["fg"])
            st.map("TCombobox", fieldbackground=[("disabled", C["surface"]), ("readonly", C["surface_2"]), ("!readonly", C["surface_2"])],
                   foreground=[("disabled", C["fg_muted"]), ("readonly", C["fg"]), ("!readonly", C["fg"])],
                   selectbackground=[("readonly", C["surface_2"])], selectforeground=[("readonly", C["fg"])],
                   bordercolor=[("focus", C["accent"])])
            st.configure("TCheckbutton", background=C["bg"], foreground=C["fg"], indicatorbackground=C["surface_2"], indicatorforeground=C["accent"])
            st.map("TCheckbutton", background=[("active", C["bg"])])
            st.configure("TRadiobutton", background=C["bg"], foreground=C["fg"], indicatorbackground=C["surface_2"], indicatorforeground=C["accent"])
            st.map("TRadiobutton", background=[("active", C["bg"])])
            st.configure("TScale", background=C["bg"], troughcolor=C["trough"], bordercolor=C["border"], lightcolor=C["accent"], darkcolor=C["accent"])
            st.configure("Horizontal.TProgressbar", background=C["accent"], troughcolor=C["trough"], bordercolor=C["border"])
            st.configure("TNotebook", background=C["bg"], bordercolor=C["border"], tabmargins=(4, 4, 0, 0))
            st.configure("TNotebook.Tab", background=C["surface_2"], foreground=C["fg_muted"], padding=(px(14), px(6)), bordercolor=C["border"])
            st.map("TNotebook.Tab", background=[("selected", C["surface"])], foreground=[("selected", C["fg"])])
            st.configure("TScrollbar", background=C["surface_2"], troughcolor=C["bg"], bordercolor=C["border"], arrowcolor=C["fg"])
            st.configure("Treeview", background=C["surface_2"], fieldbackground=C["surface_2"], foreground=C["fg"], bordercolor=C["border"],
                         rowheight=abs(self.font_px(10)) + px(8))
            st.configure("Treeview.Heading", background=C["surface"], foreground=C["fg"], bordercolor=C["border"])
            st.map("Treeview", background=[("selected", C["accent"])], foreground=[("selected", C["accent_fg"])])
            st.map("Treeview.Heading", background=[("active", C["border"])])
            self.option_add("*TCombobox*Listbox.background", C["surface_2"])
            self.option_add("*TCombobox*Listbox.foreground", C["fg"])
            self.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
        self.txt.configure(bg=C["log_bg"], fg=C["fg"], insertbackground=C["fg"])
        self._recolor_popdowns(C)
        for w in list(self.themed):
            try:
                w.apply_theme(C)
            except tk.TclError:
                self.themed.remove(w)
        self.btn_theme.configure(text="Dark theme" if self.theme_name == "light" else "Light theme")
        self.settings["theme"] = self.theme_name
        save_settings(self.settings)

    def _recolor_popdowns(self, C: dict):
        """Combobox drop down lists are plain Tk listboxes created on first use; recolor the existing ones."""
        bg = C["surface_2"] if self.theme_name == "dark" else "#ffffff"
        fg = C["fg"] if self.theme_name == "dark" else "#000000"

        def walk(w):
            for ch in w.winfo_children():
                if isinstance(ch, ttk.Menubutton) and ch["menu"]:
                    try:
                        menu = self.nametowidget(ch["menu"])
                        menu.configure(background=bg, foreground=fg, activebackground=C["accent"], activeforeground=C["accent_fg"])
                    except (tk.TclError, KeyError):
                        pass
                if isinstance(ch, ttk.Combobox):
                    try:
                        pd = self.tk.call("ttk::combobox::PopdownWindow", ch)
                        self.tk.call(f"{pd}.f.l", "configure", "-background", bg, "-foreground", fg,
                                     "-selectbackground", C["accent"], "-selectforeground", C["accent_fg"])
                    except tk.TclError:
                        pass
                walk(ch)
        walk(self)

    def toggle_theme(self):
        self.apply_theme("dark" if self.theme_name == "light" else "light")

    @staticmethod
    def _scale_label(v: str) -> str:
        return f"Auto ({text_scale_factor() * 100:.0f}%)" if v == "auto" else f"{v}%"

    def _scale_chosen(self, _e=None):
        """Apply and remember the chosen UI scale."""
        label = self.scale_var.get()
        chosen = next((v for v in SCALES if self._scale_label(v) == label), "auto")
        self.set_scale(chosen)

    def set_scale(self, setting: str):
        self.scale_setting = setting if setting in SCALES else "auto"
        self.scale_var.set(self._scale_label(self.scale_setting))
        self.settings["scale"] = self.scale_setting
        save_settings(self.settings)
        self.rescale(ui_scale_factor(self.scale_setting))

    @staticmethod
    def _device_text() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "CUDA  \u00b7  " + torch.cuda.get_device_name(0)
            return "CPU only (no CUDA)"
        except Exception:  # noqa: BLE001
            return ""

    def _build(self):
        self.header = ttk.Frame(self, padding=(px(10), px(8), px(10), 0))
        self.header.pack(fill="x")
        self.title_lbl = ttk.Label(self.header, text="Krea 2 Merge Tool")
        self.title_lbl.pack(side="left")
        self.btn_theme = ttk.Button(self.header, text="Dark theme", command=self.toggle_theme)
        self.btn_theme.pack(side="right")
        self.scale_var = tk.StringVar(value=self._scale_label(self.scale_setting))
        self.cb_scale = ttk.Combobox(self.header, textvariable=self.scale_var, state="readonly", width=13,
                                     values=[self._scale_label(v) for v in SCALES])
        self.cb_scale.pack(side="right", padx=(0, px(8)))
        self.cb_scale.bind("<<ComboboxSelected>>", self._scale_chosen)
        ttk.Label(self.header, text="Scale", style="Hint.TLabel").pack(side="right", padx=(0, px(4)))
        ttk.Label(self.header, text=self._device_text(), style="Hint.TLabel").pack(side="right", padx=px(12))
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=px(10), pady=(px(8), px(4)))
        self.tab_lora = LoraMergeTab(self.nb, self)
        self.tab_extract = ExtractTab(self.nb, self)
        self.tab_ckpt = CkptTab(self.nb, self)
        self.tab_spectrum = SpectrumTab(self.nb, self)
        self.themed.append(self.tab_spectrum)
        self.tab_advisor = AdvisorTab(self.nb, self)
        self.tab_meta = MetaTab(self.nb, self)
        self.nb.add(self.tab_lora, text="LoRA merge")
        self.nb.add(self.tab_extract, text="Extract LoRA")
        self.nb.add(self.tab_ckpt, text="Checkpoint merge / convert")
        self.nb.add(self.tab_advisor, text="Advisor")
        self.nb.add(self.tab_spectrum, text="Spectrum")
        self.nb.add(self.tab_meta, text="Metadata")

        bottom = ttk.Frame(self, padding=(px(10), px(4), px(10), px(10)))
        bottom.pack(fill="both")
        row = ttk.Frame(bottom)
        row.pack(fill="x")
        self.progress = ttk.Progressbar(row, maximum=1000)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, px(8)))
        self.status = ttk.Label(row, text="idle", width=36)
        self.status.pack(side="left", padx=px(8))
        self.gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="use GPU", variable=self.gpu_var).pack(side="left", padx=px(8))
        self.btn_cancel = ttk.Button(row, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=px(4))
        ttk.Button(row, text="Clear log", command=lambda: self.txt.delete("1.0", "end")).pack(side="left", padx=px(4))
        logf = ttk.Frame(bottom)
        logf.pack(fill="both", expand=True, pady=(px(6), 0))
        self.txt = tk.Text(logf, height=10, wrap="none", font=tkfont.nametofont("TkFixedFont"), relief="solid", borderwidth=1)
        sb = ttk.Scrollbar(logf, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

    # ---- helpers
    def use_gpu(self) -> bool:
        return bool(self.gpu_var.get())

    def log(self, text: str):
        self.txt.insert("end", text.rstrip() + "\n")
        self.txt.see("end")

    def cancel(self):
        self.cancel_event.set()
        self.status.configure(text="cancelling...")

    def save_recipe(self, tab):
        p = filedialog.asksaveasfilename(filetypes=JSON_FILES, defaultextension=".json")
        if not p:
            return
        from .recipe import save_recipe
        save_recipe(tab.to_recipe(), p)
        self.log(f"recipe saved: {p}")

    def load_recipe(self, tab):
        p = filedialog.askopenfilename(filetypes=JSON_FILES + [("safetensors with recipe", "*.safetensors")])
        if not p:
            return
        if not self.open_recipe(p):
            messagebox.showinfo("Recipe", "This file carries no recipe.")

    def open_recipe(self, p: str) -> bool:
        """Fill the tab a recipe belongs to, from a recipe file or from a file that carries one in its metadata.
        False means there was no recipe. A stored recipe holds file names only, so the inputs are looked for next
        to the file and in the remembered search folder, and what is missing is named."""
        from .recipe import load_recipe, recipe_from_file_metadata, resolution_report, resolve_recipe_paths
        try:
            r = recipe_from_file_metadata(p) if p.lower().endswith(".safetensors") else load_recipe(p)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Recipe", str(e))
            return True
        if r is None:
            return False
        base = os.path.dirname(os.path.abspath(p))
        extra = [d for d in (self.settings.get("recipe_search_dir"),) if d]
        rows = resolution_report(r, base, extra)
        if any(q is None for _role, _name, q in rows):
            if messagebox.askyesno("Recipe", "Some inputs of this recipe are not next to the file:\n\n"
                                   + "\n".join(f"  {role}: {name}" for role, name, q in rows if q is None)
                                   + "\n\nChoose a folder to look in?"):
                d = filedialog.askdirectory(title="Where the inputs are", initialdir=self.settings.get("recipe_search_dir") or base)
                if d:
                    self.settings["recipe_search_dir"] = d
                    extra = [d]
                    rows = resolution_report(r, base, extra)
        for role, name, q in rows:
            self.log(f"  {role}: {name}" + (f" -> {q}" if q else "   NOT FOUND, fill this slot by hand"))
        r = resolve_recipe_paths(r, base, extra)   # stored recipes hold file names only
        target = {"lora_merge": self.tab_lora, "extract": self.tab_extract, "ckpt_merge": self.tab_ckpt, "convert": self.tab_ckpt}[r["function"]]
        if r["function"] == "convert":
            r = {"function": "ckpt_merge", "A": {"file": r["inputs"][0]["file"]}, "loras": [],
                 "options": {"output_format": r.get("output_format", "bf16"), "passthrough": r.get("passthrough", "official"),
                             "int8_clip": r.get("int8_clip", "mse")},
                 "output": r.get("output")}
        target.from_recipe(r)
        self.nb.select(target)
        self.log(f"recipe loaded: {p}")
        return True

    # ---- jobs
    def run_job(self, label: str, job, on_done):
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Busy", "A job is already running.")
            return
        self.cancel_event.clear()
        self.btn_cancel.configure(state="normal")
        self.status.configure(text=label)
        self.progress.configure(value=0)
        self._start = time.time()
        self.log(f"--- {label}")
        q = self.msg_queue

        def progress(cur, total, name):
            q.put(("progress", (cur, total, str(name))))

        def log(s):
            q.put(("log", s))

        def target():
            try:
                res = job(progress, self.cancel_event.is_set, log)
                q.put(("done", (on_done, res)))
            except Cancelled:
                q.put(("cancelled", None))
            except Exception as e:  # noqa: BLE001
                import traceback
                q.put(("error", (str(e), traceback.format_exc())))
        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "progress":
                    cur, total, name = payload
                    self.progress.configure(value=int(1000 * cur / max(total, 1)))
                    self.status.configure(text=f"{cur}/{total} {name[:40]}")
                elif kind == "log":
                    self.log(payload)
                elif kind == "done":
                    on_done, res = payload
                    self._finish("done")
                    try:
                        on_done(res)
                    except Exception as e:  # noqa: BLE001
                        self.log(f"error: {e}")
                elif kind == "cancelled":
                    self._finish("cancelled")
                    self.log("cancelled; partial output removed")
                elif kind == "error":
                    msg, tb = payload
                    self._finish("error")
                    self.log("ERROR: " + msg)
                    self.log(tb)
                    messagebox.showerror("Error", msg)
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _finish(self, state: str):
        el = time.time() - self._start if self._start else 0.0
        self.status.configure(text=f"{state} ({el:.1f}s)")
        self.progress.configure(value=1000 if state == "done" else 0)
        self.btn_cancel.configure(state="disabled")


def run_gui(theme: str | None = None, scale: str | None = None) -> int:
    app = MergeApp(theme, scale)
    app.mainloop()
    return 0
