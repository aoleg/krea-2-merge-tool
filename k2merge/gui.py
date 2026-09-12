"""Tkinter GUI. The window only assembles recipes and hands them to the engine.

Shell (theme, worker thread, progress and log handling) adapted from
krea-2-lora-merge-tool.py v10.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __version__
from .blocks import BLOCK_PRESETS, MODIFIERS, RECIPES, RECIPE_LABELS, Shaping, build_block_mask
from .ckpt_merge import CkptInput, CkptMergeOptions
from .engine import Cancelled
from .extract import ExtractOptions
from .formats import OUTPUT_FORMATS, PASSTHROUGH
from .keys import GROUPS, KREA2_BLOCKS
from .lora_merge import LoraInput, LoraMergeOptions
from .methods import ADVANCED, METHODS, METHOD_LABELS, NEEDS_C

THEMES = {
    "dark": {"bg": "#12141a", "surface": "#1a1d26", "surface_2": "#232734", "border": "#2e3342",
             "fg": "#e6e9f0", "fg_muted": "#8b93a7", "accent": "#6c8cff", "accent_2": "#8aa2ff",
             "accent_fg": "#0d0f14", "danger": "#ff6b6b", "warn": "#ffb454", "ok": "#4ade80",
             "log_bg": "#0e1015", "trough": "#232734"},
    "light": {"bg": "#f2f4f8", "surface": "#ffffff", "surface_2": "#eef1f6", "border": "#d3d9e3",
              "fg": "#1b1f27", "fg_muted": "#697086", "accent": "#3b62e8", "accent_2": "#5b7cf0",
              "accent_fg": "#ffffff", "danger": "#d23b3b", "warn": "#b46a00", "ok": "#1a8a4a",
              "log_bg": "#fbfcfe", "trough": "#e2e7ef"},
}
ST_FILES = [("safetensors", "*.safetensors"), ("all files", "*.*")]
JSON_FILES = [("recipe", "*.json"), ("all files", "*.*")]
STEP = 0.05


def _round_step(x: float) -> float:
    return round(round(float(x) / STEP) * STEP, 2)


# ============================================================================== widgets
class FileSlot(ttk.Frame):
    """Entry + Browse + Inspect."""

    def __init__(self, master, app, label: str, save: bool = False):
        super().__init__(master, style="Surface.TFrame")
        self.app = app
        self.var = tk.StringVar()
        ttk.Label(self, text=label, style="Surface.TLabel", width=9).grid(row=0, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.var).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(self, text="Browse", style="Ghost.TButton",
                   command=self._browse_save if save else self._browse).grid(row=0, column=2)
        if not save:
            ttk.Button(self, text="Inspect", style="Ghost.TButton", command=self._inspect).grid(row=0, column=3, padx=(2, 0))
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


class ShapingRow(ttk.LabelFrame):
    """File, strength or weight, block shaping with curve preview. Used for LoRA rows and slot B."""

    def __init__(self, master, app, title: str, kind: str = "lora"):
        super().__init__(master, text=title, style="Card.TLabelframe")
        self.app, self.kind = app, kind
        self.slot = FileSlot(self, app, "file")
        self.slot.grid(row=0, column=0, columnspan=8, sticky="ew", pady=(2, 4))
        self.value = tk.DoubleVar(value=1.0)
        self.value_txt = tk.StringVar(value="1.00")
        ttk.Label(self, text="weight" if kind == "ckpt" else "strength", style="Surface.TLabel").grid(row=1, column=0, sticky="w")
        self.scale = ttk.Scale(self, from_=-10.0, to=10.0, variable=self.value, command=self._on_scale)
        self.scale.grid(row=1, column=1, columnspan=3, sticky="ew", padx=4)
        e = ttk.Entry(self, textvariable=self.value_txt, width=7)
        e.grid(row=1, column=4, sticky="w")
        e.bind("<Return>", self._on_entry)
        e.bind("<FocusOut>", self._on_entry)
        self.preset = tk.StringVar(value="FULL")
        self.modifier = tk.StringVar(value="Suppress")
        self.contrast = tk.DoubleVar(value=0.5)
        self.boost = tk.DoubleVar(value=1.0)
        ttk.Label(self, text="blocks", style="Surface.TLabel").grid(row=2, column=0, sticky="w")
        cb = ttk.Combobox(self, textvariable=self.preset, values=BLOCK_PRESETS, state="readonly", width=13)
        cb.grid(row=2, column=1, sticky="w", padx=4)
        self.cb_mod = ttk.Combobox(self, textvariable=self.modifier, values=MODIFIERS, state="readonly", width=10)
        self.cb_mod.grid(row=2, column=2, sticky="w")
        ttk.Label(self, text="contrast", style="Surface.TLabel").grid(row=2, column=3, sticky="e")
        self.sc_contrast = ttk.Scale(self, from_=0.0, to=1.0, variable=self.contrast, command=lambda _v: self._changed())
        self.sc_contrast.grid(row=2, column=4, sticky="ew", padx=4)
        ttk.Label(self, text="boost", style="Surface.TLabel").grid(row=2, column=5, sticky="e")
        self.sc_boost = ttk.Scale(self, from_=0.25, to=2.0, variable=self.boost, command=lambda _v: self._changed())
        self.sc_boost.grid(row=2, column=6, sticky="ew", padx=4)
        mb = ttk.Menubutton(self, text="Recipes", style="Ghost.TMenubutton")
        menu = tk.Menu(mb, tearoff=0)
        for key, label in RECIPE_LABELS.items():
            menu.add_command(label=label, command=lambda k=key: self.apply_recipe(k))
        menu.add_command(label="FULL (no shaping)", command=lambda: self.apply_recipe(None))
        mb["menu"] = menu
        mb.grid(row=2, column=7, sticky="e")
        self.curve = tk.Canvas(self, height=34, width=200, highlightthickness=0, bg=app.C["surface_2"])
        self.curve.grid(row=3, column=1, columnspan=6, sticky="ew", padx=4, pady=(2, 4))
        self.info = ttk.Label(self, text="", style="Muted.TLabel")
        self.info.grid(row=3, column=7, sticky="e")
        if kind == "ckpt":
            ttk.Label(self, text="non-block weight (text side, projections)", style="Surface.TLabel").grid(row=4, column=0, columnspan=3, sticky="w")
            self.non_block_txt = tk.StringVar(value="")
            ttk.Entry(self, textvariable=self.non_block_txt, width=7).grid(row=4, column=4, sticky="w")
            ttk.Label(self, text="empty = same as weight", style="Muted.TLabel").grid(row=4, column=5, columnspan=3, sticky="w")
        else:
            self.non_block_txt = None
        for c in (1, 2, 3, 4, 6):
            self.columnconfigure(c, weight=1)
        for v in (self.preset, self.modifier):
            v.trace_add("write", lambda *_: self._changed())
        self._changed()

    # ---- value handling
    def _on_scale(self, _v):
        v = _round_step(self.value.get())
        self.value.set(v)
        self.value_txt.set(f"{v:.2f}")

    def _on_entry(self, _e=None):
        try:
            v = max(-10.0, min(10.0, float(self.value_txt.get().replace(",", "."))))
        except ValueError:
            v = self.value.get()
        v = _round_step(v)
        self.value.set(v)
        self.value_txt.set(f"{v:.2f}")

    def set_value(self, v: float):
        v = _round_step(v)
        self.value.set(v)
        self.value_txt.set(f"{v:.2f}")

    def _changed(self):
        flat = self.preset.get() == "FULL"
        state = "disabled" if flat else "normal"
        self.cb_mod.configure(state="disabled" if flat else "readonly")
        self.sc_contrast.configure(state=state)
        self.sc_boost.configure(state=state if self.modifier.get() == "Emphasize" else "disabled")
        self.draw_curve()

    def apply_recipe(self, key: str | None):
        s = RECIPES[key] if key else Shaping()
        self.preset.set(s.preset)
        self.modifier.set(s.modifier)
        self.contrast.set(s.contrast)
        self.boost.set(s.boost)
        self._changed()

    def shaping(self) -> Shaping:
        s = Shaping(preset=self.preset.get(), modifier=self.modifier.get(),
                    contrast=round(float(self.contrast.get()), 3), boost=round(float(self.boost.get()), 3))
        if self.non_block_txt is not None and self.non_block_txt.get().strip():
            try:
                nb = float(self.non_block_txt.get().replace(",", "."))
                w = self.value.get()
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
        w = max(int(c.winfo_width()), 200)
        h = 34
        n = len(f)
        bw = w / n
        c.create_line(0, h - h / 3, w, h - h / 3, fill=self.app.C["border"])
        for i, v in enumerate(f):
            bh = min(v, 2.0) / 2.0 * (h - 4)
            col = self.app.C["accent"] if v > 1.0 + 1e-9 else (self.app.C["fg_muted"] if v < 1.0 - 1e-9 else self.app.C["ok"])
            c.create_rectangle(i * bw + 1, h - bh, (i + 1) * bw - 1, h, fill=col, outline="")
        self.info.configure(text=f"min {min(f):.2f}  max {max(f):.2f}")

    # ---- conversions
    def to_lora_input(self) -> LoraInput | None:
        p = self.slot.get()
        if not p:
            return None
        return LoraInput(p, self.value.get(), self.shaping())

    def to_ckpt_input(self) -> CkptInput | None:
        p = self.slot.get()
        if not p:
            return None
        return CkptInput(p, self.value.get(), self.shaping())

    def load(self, path: str | None, value: float, shaping: Shaping | None):
        self.slot.set(path)
        self.set_value(value)
        s = shaping or Shaping()
        self.preset.set(s.preset)
        self.modifier.set(s.modifier)
        self.contrast.set(s.contrast)
        self.boost.set(s.boost)
        if self.non_block_txt is not None:
            self.non_block_txt.set("" if s.non_block is None else f"{s.non_block * value:.2f}")
        self._changed()

    def clear(self):
        self.load(None, 1.0, None)


# ============================================================================== tabs
class LoraMergeTab(ttk.Frame):
    MAX_ROWS = 6

    def __init__(self, master, app):
        super().__init__(master, style="TFrame")
        self.app = app
        self.rows: list[ShapingRow] = []
        self.rows_frame = ttk.Frame(self, style="TFrame")
        self.rows_frame.pack(fill="x", padx=8, pady=4)
        bar = ttk.Frame(self, style="TFrame")
        bar.pack(fill="x", padx=8)
        ttk.Button(bar, text="+ add LoRA", style="Ghost.TButton", command=self.add_row).pack(side="left")
        ttk.Button(bar, text="- remove last", style="Ghost.TButton", command=self.remove_row).pack(side="left", padx=4)
        self.average = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="average (weights sum to 1: epochs of one run)", variable=self.average).pack(side="left", padx=12)

        opts = ttk.LabelFrame(self, text="Output", style="Card.TLabelframe")
        opts.pack(fill="x", padx=8, pady=4)
        self.rank_mode = tk.StringVar(value="concat")
        ttk.Label(opts, text="rank", style="Surface.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(opts, text="keep concatenated (exact)", variable=self.rank_mode, value="concat").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(opts, text="fixed", variable=self.rank_mode, value="fixed").grid(row=0, column=2, sticky="w")
        self.rank = tk.StringVar(value="32")
        ttk.Entry(opts, textvariable=self.rank, width=6).grid(row=0, column=3, sticky="w")
        ttk.Radiobutton(opts, text="per group (from analysis)", variable=self.rank_mode, value="groups").grid(row=0, column=4, sticky="w")
        self.group_ranks: dict = {}
        self.group_lbl = ttk.Label(opts, text="", style="Muted.TLabel")
        self.group_lbl.grid(row=0, column=5, sticky="w")
        ttk.Label(opts, text="modules", style="Surface.TLabel").grid(row=1, column=0, sticky="w")
        self.modules = tk.StringVar(value="intersection")
        ttk.Combobox(opts, textvariable=self.modules, values=["intersection", "union"], state="readonly", width=12).grid(row=1, column=1, sticky="w")
        ttk.Label(opts, text="naming", style="Surface.TLabel").grid(row=1, column=2, sticky="e")
        self.naming = tk.StringVar(value="comfy")
        ttk.Combobox(opts, textvariable=self.naming, values=["comfy", "kohya", "input"], state="readonly", width=8).grid(row=1, column=3, sticky="w")
        ttk.Label(opts, text="dtype", style="Surface.TLabel").grid(row=1, column=4, sticky="e")
        self.dtype = tk.StringVar(value="fp16")
        ttk.Combobox(opts, textvariable=self.dtype, values=["fp16", "bf16", "fp32"], state="readonly", width=6).grid(row=1, column=5, sticky="w")
        ttk.Label(opts, text="energy target", style="Surface.TLabel").grid(row=2, column=0, sticky="w")
        self.target = tk.StringVar(value="0.99")
        ttk.Combobox(opts, textvariable=self.target, values=["0.9", "0.95", "0.99", "0.995"], width=6).grid(row=2, column=1, sticky="w")
        self.criterion = tk.StringVar(value="weighted")
        ttk.Combobox(opts, textvariable=self.criterion, values=["weighted", "per_module"], state="readonly", width=10).grid(row=2, column=2, sticky="w")
        ttk.Label(opts, text="(rank chosen from the analysis with this target and criterion)", style="Muted.TLabel").grid(row=2, column=3, columnspan=3, sticky="w")
        self.out = FileSlot(opts, app, "output", save=True)
        self.out.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(4, 2))
        opts.columnconfigure(5, weight=1)

        btns = ttk.Frame(self, style="TFrame")
        btns.pack(fill="x", padx=8, pady=4)
        ttk.Button(btns, text="Analyze", command=self.analyze).pack(side="left")
        ttk.Button(btns, text="Plan", command=self.plan).pack(side="left", padx=4)
        ttk.Button(btns, text="Merge", style="Accent.TButton", command=self.run).pack(side="left", padx=4)
        ttk.Button(btns, text="Save recipe", style="Ghost.TButton", command=lambda: app.save_recipe(self)).pack(side="right")
        ttk.Button(btns, text="Load recipe", style="Ghost.TButton", command=lambda: app.load_recipe(self)).pack(side="right", padx=4)
        self.add_row()
        self.add_row()

    def add_row(self):
        if len(self.rows) >= self.MAX_ROWS:
            return
        r = ShapingRow(self.rows_frame, self.app, f"LoRA {len(self.rows) + 1}", kind="lora")
        r.pack(fill="x", pady=2)
        self.rows.append(r)

    def remove_row(self):
        if len(self.rows) > 1:
            self.rows.pop().destroy()

    def inputs(self) -> list[LoraInput]:
        return [i for i in (r.to_lora_input() for r in self.rows) if i is not None]

    def options(self) -> LoraMergeOptions:
        o = LoraMergeOptions(average=self.average.get(), rank_mode=self.rank_mode.get(),
                             rank=int(self.rank.get()) if self.rank.get().strip().isdigit() else None,
                             group_ranks=dict(self.group_ranks), modules=self.modules.get(),
                             naming=self.naming.get(), dtype=self.dtype.get())
        return o

    def to_recipe(self) -> dict:
        return {"function": "lora_merge", "inputs": [i.to_dict() for i in self.inputs()],
                "options": self.options().to_dict(), "output": self.out.get()}

    def from_recipe(self, r: dict):
        inputs = [LoraInput.from_dict(d) for d in r.get("inputs", [])]
        while len(self.rows) < max(1, len(inputs)):
            self.add_row()
        while len(self.rows) > max(1, len(inputs)):
            self.remove_row()
        for row, inp in zip(self.rows, inputs):
            row.load(inp.path, inp.strength, inp.shaping)
        o = LoraMergeOptions.from_dict(r.get("options"))
        self.average.set(o.average)
        self.rank_mode.set(o.rank_mode)
        self.rank.set(str(o.rank or 32))
        self.group_ranks = dict(o.group_ranks)
        self.group_lbl.configure(text=json.dumps(self.group_ranks) if self.group_ranks else "")
        self.modules.set(o.modules)
        self.naming.set(o.naming)
        self.dtype.set(o.dtype)
        self.out.set(r.get("output"))

    def _check(self) -> bool:
        if not self.inputs():
            messagebox.showwarning("LoRA merge", "Add at least one LoRA file.")
            return False
        return True

    def analyze(self):
        if not self._check():
            return
        inputs, opts = self.inputs(), self.options()
        target, crit = float(self.target.get()), self.criterion.get()

        def job(progress, cancel, log):
            from .lora_merge import analyze_lora_merge
            rep = analyze_lora_merge(inputs, opts, self.app.use_gpu(), progress=progress)
            plan = rep.rank_plan(target, crit)
            return rep.text() + f"\n\nrank per group for {target:.3f} energy ({crit}): {plan}", plan

        def done(res):
            text, plan = res
            self.app.log(text)
            self.group_ranks = {g: int(r) for g, r in plan.items()}
            self.group_lbl.configure(text=json.dumps(self.group_ranks))
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
        recipe = self.to_recipe()

        def job(progress, cancel, log):
            from .lora_merge import merge_loras
            return merge_loras(inputs, out, opts, self.app.use_gpu(), progress, cancel, log)

        def done(res):
            self.app.log(f"wrote {res.path}: {res.modules} modules, rank {res.rank_min}-{res.rank_max}, "
                         f"energy kept >= {res.kept_min * 100:.2f}%, {len(res.dropped)} dropped, {res.seconds:.1f}s")
            self.app.last_recipe = recipe
        self.app.run_job("Merging LoRAs", job, done)


class ExtractTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, style="TFrame")
        self.app = app
        card = ttk.LabelFrame(self, text="Checkpoints", style="Card.TLabelframe")
        card.pack(fill="x", padx=8, pady=4)
        self.base = FileSlot(card, app, "base")
        self.base.pack(fill="x", pady=2)
        self.target = FileSlot(card, app, "target")
        self.target.pack(fill="x", pady=2)
        ttk.Label(card, text="The LoRA approximates target minus base. Both may be bf16, fp8 scaled or int8 convrot.",
                  style="Muted.TLabel").pack(anchor="w")

        opts = ttk.LabelFrame(self, text="Extraction", style="Card.TLabelframe")
        opts.pack(fill="x", padx=8, pady=4)
        ttk.Label(opts, text="rank", style="Surface.TLabel").grid(row=0, column=0, sticky="w")
        self.rank = tk.StringVar(value="32")
        ttk.Entry(opts, textvariable=self.rank, width=6).grid(row=0, column=1, sticky="w")
        self.use_groups = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="per group ranks from analysis", variable=self.use_groups).grid(row=0, column=2, sticky="w")
        self.group_ranks: dict = {}
        self.group_lbl = ttk.Label(opts, text="", style="Muted.TLabel")
        self.group_lbl.grid(row=0, column=3, columnspan=3, sticky="w")
        ttk.Label(opts, text="modules", style="Surface.TLabel").grid(row=1, column=0, sticky="w")
        self.filter = tk.StringVar(value="all")
        ttk.Combobox(opts, textvariable=self.filter, values=["all", "attn", "blocks", "custom"], state="readonly", width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(opts, text="include regex", style="Surface.TLabel").grid(row=1, column=2, sticky="e")
        self.include = tk.StringVar()
        ttk.Entry(opts, textvariable=self.include, width=18).grid(row=1, column=3, sticky="w")
        ttk.Label(opts, text="exclude regex", style="Surface.TLabel").grid(row=1, column=4, sticky="e")
        self.exclude = tk.StringVar()
        ttk.Entry(opts, textvariable=self.exclude, width=18).grid(row=1, column=5, sticky="w")
        ttk.Label(opts, text="SVD", style="Surface.TLabel").grid(row=2, column=0, sticky="w")
        self.method = tk.StringVar(value="randomized")
        ttk.Combobox(opts, textvariable=self.method, values=["randomized", "full"], state="readonly", width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(opts, text="naming", style="Surface.TLabel").grid(row=2, column=2, sticky="e")
        self.naming = tk.StringVar(value="comfy")
        ttk.Combobox(opts, textvariable=self.naming, values=["comfy", "kohya"], state="readonly", width=8).grid(row=2, column=3, sticky="w")
        ttk.Label(opts, text="dtype", style="Surface.TLabel").grid(row=2, column=4, sticky="e")
        self.dtype = tk.StringVar(value="fp16")
        ttk.Combobox(opts, textvariable=self.dtype, values=["fp16", "bf16", "fp32"], state="readonly", width=6).grid(row=2, column=5, sticky="w")
        ttk.Label(opts, text="energy target", style="Surface.TLabel").grid(row=3, column=0, sticky="w")
        self.target_e = tk.StringVar(value="0.99")
        ttk.Combobox(opts, textvariable=self.target_e, values=["0.9", "0.95", "0.99", "0.995"], width=6).grid(row=3, column=1, sticky="w")
        self.criterion = tk.StringVar(value="weighted")
        ttk.Combobox(opts, textvariable=self.criterion, values=["weighted", "per_module"], state="readonly", width=10).grid(row=3, column=2, sticky="w")
        self.out = FileSlot(opts, app, "output", save=True)
        self.out.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(4, 2))
        opts.columnconfigure(5, weight=1)

        btns = ttk.Frame(self, style="TFrame")
        btns.pack(fill="x", padx=8, pady=4)
        ttk.Button(btns, text="Analyze", command=self.analyze).pack(side="left")
        ttk.Button(btns, text="Plan", command=self.plan).pack(side="left", padx=4)
        ttk.Button(btns, text="Extract", style="Accent.TButton", command=self.run).pack(side="left", padx=4)
        ttk.Button(btns, text="Save recipe", style="Ghost.TButton", command=lambda: app.save_recipe(self)).pack(side="right")
        ttk.Button(btns, text="Load recipe", style="Ghost.TButton", command=lambda: app.load_recipe(self)).pack(side="right", padx=4)

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
        self.group_lbl.configure(text=json.dumps(self.group_ranks) if self.group_ranks else "")
        self.filter.set(o.filter)
        self.include.set(o.include)
        self.exclude.set(o.exclude)
        self.method.set(o.method)
        self.naming.set(o.naming)
        self.dtype.set(o.dtype)
        self.out.set(r.get("output"))

    def _check(self) -> bool:
        if not (self.base.get() and self.target.get()):
            messagebox.showwarning("Extract", "Choose the base and the target checkpoint.")
            return False
        return True

    def analyze(self):
        if not self._check():
            return
        b, t, o = self.base.get(), self.target.get(), self.options()
        target, crit = float(self.target_e.get()), self.criterion.get()

        def job(progress, cancel, log):
            from .extract import analyze_extract
            rep = analyze_extract(b, t, o, self.app.use_gpu(), progress=progress)
            plan = rep.rank_plan(target, crit)
            return rep.text() + f"\n\nrank per group for {target:.3f} energy ({crit}): {plan}", plan

        def done(res):
            text, plan = res
            self.app.log(text)
            self.group_ranks = {g: int(r) for g, r in plan.items()}
            self.group_lbl.configure(text=json.dumps(self.group_ranks))
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

        def job(progress, cancel, log):
            from .extract import extract_lora
            return extract_lora(b, t, out, o, self.app.use_gpu(), progress, cancel, log)

        def done(res):
            self.app.log(f"wrote {res.path}: {res.modules} modules, energy kept >= {res.kept_min * 100:.2f}%"
                         + (f", per group {({g: round(v, 4) for g, v in res.kept_by_group.items()})}" if res.kept_by_group else "")
                         + f", {len(res.skipped)} skipped, {res.seconds:.1f}s")
        self.app.run_job("Extracting", job, done)


class CkptTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, style="TFrame")
        self.app = app
        outer = ttk.Frame(self, style="TFrame")
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer, style="TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=4)
        right = ttk.Frame(outer, style="TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=4)

        ck = ttk.LabelFrame(left, text="Checkpoints", style="Card.TLabelframe")
        ck.pack(fill="x", pady=2)
        self.A = FileSlot(ck, app, "A primary")
        self.A.pack(fill="x", pady=2)
        self.B = ShapingRow(ck, app, "B secondary (optional)", kind="ckpt")
        self.B.pack(fill="x", pady=2)
        self.C = FileSlot(ck, app, "C reference")
        self.C.pack(fill="x", pady=2)
        ttk.Label(ck, text="C is the common ancestor of A and B (usually the official Turbo file). Needed by TIES, DARE, trainDifference, extract.",
                  style="Muted.TLabel").pack(anchor="w")

        mt = ttk.LabelFrame(left, text="Method", style="Card.TLabelframe")
        mt.pack(fill="x", pady=2)
        self.method = tk.StringVar(value="add_difference")
        self.advanced = tk.BooleanVar(value=False)
        self.cb_method = ttk.Combobox(mt, textvariable=self.method, values=self._method_values(), state="readonly", width=18)
        self.cb_method.grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(mt, text="advanced methods", variable=self.advanced, command=self._refresh_methods).grid(row=0, column=1, sticky="w", padx=8)
        self.method_lbl = ttk.Label(mt, text="", style="Muted.TLabel", wraplength=520, justify="left")
        self.method_lbl.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 4))
        self.params: dict[str, tk.StringVar] = {}
        pf = ttk.Frame(mt, style="Surface.TFrame")
        pf.grid(row=2, column=0, columnspan=4, sticky="w")
        for i, (name, default) in enumerate((("density", "0.2"), ("lambda", "1.0"), ("p", "0.5"), ("seed", "0"), ("beta", "0.0"), ("gamma", "1.0"))):
            ttk.Label(pf, text=name, style="Surface.TLabel").grid(row=0, column=2 * i, sticky="e", padx=(6, 2))
            v = tk.StringVar(value=default)
            self.params[name] = v
            ttk.Entry(pf, textvariable=v, width=6).grid(row=0, column=2 * i + 1, sticky="w")
        self.dare_ties = tk.BooleanVar(value=False)
        ttk.Checkbutton(pf, text="DARE then TIES", variable=self.dare_ties).grid(row=0, column=12, padx=6)
        self.lora_mode = tk.StringVar(value="after")
        ttk.Label(mt, text="LoRAs", style="Surface.TLabel").grid(row=3, column=0, sticky="w")
        ttk.Combobox(mt, textvariable=self.lora_mode, values=["after", "task_vectors"], state="readonly", width=12).grid(row=3, column=1, sticky="w")
        ttk.Label(mt, text="after = added after the method; task_vectors = joins TIES / DARE", style="Muted.TLabel").grid(row=3, column=2, columnspan=2, sticky="w")
        self.method.trace_add("write", lambda *_: self._method_changed())
        self._method_changed()

        lf = ttk.LabelFrame(right, text="LoRAs (0 to 4)", style="Card.TLabelframe")
        lf.pack(fill="x", pady=2)
        self.loras = [ShapingRow(lf, app, f"LoRA {i + 1}", kind="lora") for i in range(4)]
        for r in self.loras:
            r.pack(fill="x", pady=1)

        of = ttk.LabelFrame(left, text="Output", style="Card.TLabelframe")
        of.pack(fill="x", pady=2)
        ttk.Label(of, text="format", style="Surface.TLabel").grid(row=0, column=0, sticky="w")
        self.fmt = tk.StringVar(value="bf16")
        ttk.Combobox(of, textvariable=self.fmt, values=list(OUTPUT_FORMATS), state="readonly", width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(of, text="passthrough", style="Surface.TLabel").grid(row=0, column=2, sticky="e")
        self.passthrough = tk.StringVar(value="official")
        ttk.Combobox(of, textvariable=self.passthrough, values=list(PASSTHROUGH), state="readonly", width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(of, text="fp8 layers", style="Surface.TLabel").grid(row=0, column=4, sticky="e")
        self.fp8_set = tk.StringVar(value="official")
        ttk.Combobox(of, textvariable=self.fp8_set, values=["official", "blocks"], state="readonly", width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(of, text="int8 clip", style="Surface.TLabel").grid(row=0, column=6, sticky="e")
        self.int8_clip = tk.StringVar(value="mse")
        ttk.Combobox(of, textvariable=self.int8_clip, values=["mse", "absmax"], state="readonly", width=7).grid(row=0, column=7, sticky="w")
        self.as_lora = tk.BooleanVar(value=False)
        ttk.Checkbutton(of, text="output as LoRA (result minus C, or minus A)", variable=self.as_lora).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(of, text="rank", style="Surface.TLabel").grid(row=1, column=3, sticky="e")
        self.lora_rank = tk.StringVar(value="32")
        ttk.Entry(of, textvariable=self.lora_rank, width=6).grid(row=1, column=4, sticky="w")
        self.lora_naming = tk.StringVar(value="comfy")
        ttk.Combobox(of, textvariable=self.lora_naming, values=["comfy", "kohya"], state="readonly", width=8).grid(row=1, column=5, sticky="w")
        self.keep_meta = tk.BooleanVar(value=True)
        ttk.Checkbutton(of, text="keep A's metadata", variable=self.keep_meta).grid(row=2, column=0, columnspan=2, sticky="w")
        self.out = FileSlot(of, app, "output", save=True)
        self.out.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(4, 2))
        of.columnconfigure(5, weight=1)

        btns = ttk.Frame(left, style="TFrame")
        btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="Report", command=self.report).pack(side="left")
        ttk.Button(btns, text="Plan", command=self.plan).pack(side="left", padx=4)
        ttk.Button(btns, text="Merge / Convert", style="Accent.TButton", command=self.run).pack(side="left", padx=4)
        ttk.Button(btns, text="Save recipe", style="Ghost.TButton", command=lambda: app.save_recipe(self)).pack(side="right")
        ttk.Button(btns, text="Load recipe", style="Ghost.TButton", command=lambda: app.load_recipe(self)).pack(side="right", padx=4)

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
            txt += "\nWithout C the reference is A."
        self.method_lbl.configure(text=txt)

    def options(self) -> CkptMergeOptions:
        o = CkptMergeOptions(method=self.method.get(), output_format=self.fmt.get(), passthrough=self.passthrough.get(),
                             fp8_layer_set=self.fp8_set.get(), int8_clip=self.int8_clip.get(), keep_metadata=self.keep_meta.get(),
                             lora_mode=self.lora_mode.get(), output_as_lora=self.as_lora.get())
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
        loras = [i for i in (r.to_lora_input() for r in self.loras) if i is not None]
        return A, B, C, loras

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
        loras = [LoraInput.from_dict(d) for d in r.get("loras", [])]
        for row, inp in zip(self.loras, loras + [None] * 4):
            if inp is None:
                row.clear()
            else:
                row.load(inp.path, inp.strength, inp.shaping)
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
        self.as_lora.set(o.output_as_lora)
        self.lora_rank.set(str(o.lora_out.get("rank", 32)))
        self.lora_naming.set(o.lora_out.get("naming", "comfy"))
        self.keep_meta.set(o.keep_metadata)
        self.out.set(r.get("output"))

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
        self.app.run_job("Pre merge report", lambda p, c, l: __import__("k2merge.ckpt_merge", fromlist=["premerge_report"]).premerge_report(A, B, C, self.app.use_gpu(), progress=p), self.app.log)

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
    def __init__(self, theme: str = "dark"):
        super().__init__()
        self.title(f"Krea 2 Merge Tool {__version__}")
        self.minsize(1100, 760)
        self.theme_name = theme
        self.C = THEMES[theme]
        self.msg_queue: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.last_recipe: dict | None = None
        self._start = None
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self._apply_theme()
        self._build()
        self.after(80, self._poll)

    # ---- theme
    def _apply_theme(self):
        C, st = self.C, self.style
        self.configure(bg=C["bg"])
        st.configure(".", background=C["bg"], foreground=C["fg"], borderwidth=0, focuscolor=C["accent"])
        st.configure("TFrame", background=C["bg"])
        st.configure("Surface.TFrame", background=C["surface"])
        st.configure("TLabel", background=C["bg"], foreground=C["fg"])
        st.configure("Surface.TLabel", background=C["surface"], foreground=C["fg"])
        st.configure("Muted.TLabel", background=C["surface"], foreground=C["fg_muted"])
        st.configure("Card.TLabelframe", background=C["surface"], bordercolor=C["border"], relief="solid")
        st.configure("Card.TLabelframe.Label", background=C["surface"], foreground=C["accent"])
        st.configure("TButton", background=C["surface_2"], foreground=C["fg"], padding=(10, 5))
        st.map("TButton", background=[("active", C["border"])])
        st.configure("Accent.TButton", background=C["accent"], foreground=C["accent_fg"], padding=(14, 6))
        st.map("Accent.TButton", background=[("active", C["accent_2"]), ("disabled", C["surface_2"])])
        st.configure("Ghost.TButton", background=C["surface"], foreground=C["fg_muted"], padding=(6, 3))
        st.map("Ghost.TButton", background=[("active", C["surface_2"])], foreground=[("active", C["fg"])])
        st.configure("Ghost.TMenubutton", background=C["surface"], foreground=C["fg_muted"], padding=(6, 3))
        st.configure("TEntry", fieldbackground=C["surface_2"], foreground=C["fg"], insertcolor=C["fg"])
        st.configure("TCombobox", fieldbackground=C["surface_2"], foreground=C["fg"], background=C["surface_2"], arrowcolor=C["fg"])
        st.configure("TCheckbutton", background=C["surface"], foreground=C["fg"])
        st.configure("TRadiobutton", background=C["surface"], foreground=C["fg"])
        st.configure("TScale", background=C["surface"], troughcolor=C["trough"])
        st.configure("TNotebook", background=C["bg"], tabmargins=(4, 4, 0, 0))
        st.configure("TNotebook.Tab", background=C["surface_2"], foreground=C["fg_muted"], padding=(14, 6))
        st.map("TNotebook.Tab", background=[("selected", C["surface"])], foreground=[("selected", C["fg"])])
        st.configure("Horizontal.TProgressbar", background=C["accent"], troughcolor=C["trough"])
        self.option_add("*TCombobox*Listbox.background", C["surface_2"])
        self.option_add("*TCombobox*Listbox.foreground", C["fg"])

    # ---- layout
    def _build(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.tab_lora = LoraMergeTab(self.nb, self)
        self.tab_extract = ExtractTab(self.nb, self)
        self.tab_ckpt = CkptTab(self.nb, self)
        self.nb.add(self.tab_lora, text="  LoRA merge  ")
        self.nb.add(self.tab_extract, text="  Extract  ")
        self.nb.add(self.tab_ckpt, text="  Checkpoint merge / convert  ")

        bottom = ttk.Frame(self, style="TFrame")
        bottom.pack(fill="both", padx=8, pady=(0, 8))
        row = ttk.Frame(bottom, style="TFrame")
        row.pack(fill="x")
        self.progress = ttk.Progressbar(row, style="Horizontal.TProgressbar", maximum=1000)
        self.progress.pack(side="left", fill="x", expand=True)
        self.status = ttk.Label(row, text="idle", style="TLabel")
        self.status.pack(side="left", padx=8)
        self.gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="GPU", variable=self.gpu_var).pack(side="left", padx=4)
        self.btn_cancel = ttk.Button(row, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left")
        ttk.Button(row, text="Clear log", style="Ghost.TButton", command=lambda: self.txt.delete("1.0", "end")).pack(side="left", padx=4)
        self.txt = tk.Text(bottom, height=12, bg=self.C["log_bg"], fg=self.C["fg"], insertbackground=self.C["fg"],
                           relief="flat", font=("Consolas", 9), wrap="none")
        self.txt.pack(fill="both", expand=True, pady=(4, 0))

    # ---- helpers
    def use_gpu(self) -> bool:
        return bool(self.gpu_var.get())

    def log(self, text: str):
        self.txt.insert("end", text.rstrip() + "\n")
        self.txt.see("end")

    def cancel(self):
        self.cancel_event.set()
        self.status.configure(text="cancelling...")

    def current_tab(self):
        return self.nb.nametowidget(self.nb.select())

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
        from .recipe import load_recipe, recipe_from_file_metadata
        try:
            r = recipe_from_file_metadata(p) if p.lower().endswith(".safetensors") else load_recipe(p)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Recipe", str(e))
            return
        if r is None:
            messagebox.showinfo("Recipe", "This file carries no recipe.")
            return
        target = {"lora_merge": self.tab_lora, "extract": self.tab_extract, "ckpt_merge": self.tab_ckpt, "convert": self.tab_ckpt}[r["function"]]
        if r["function"] == "convert":
            r = {"function": "ckpt_merge", "A": {"file": r["inputs"][0]["file"]}, "loras": [],
                 "options": {"output_format": r.get("output_format", "bf16"), "passthrough": r.get("passthrough", "official")},
                 "output": r.get("output")}
        target.from_recipe(r)
        self.nb.select(target)
        self.log(f"recipe loaded: {p}")

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
                    self.status.configure(text=f"{cur}/{total} {name[:50]}")
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


def run_gui(theme: str = "dark") -> int:
    app = MergeApp(theme)
    app.mainloop()
    return 0
