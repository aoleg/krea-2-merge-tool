"""The Prune tab: one LoRA in, a smaller LoRA out.

Pruning is the single input LoRA merge with the dynamic rank rule: every module keeps the smallest rank whose
energy reaches the retention, never below the floor and never above the cap. The shaping row of the input does
what it does everywhere (a zone weight on the delta before the SVD), so a suppressed zone needs fewer components
and a zeroed one is dropped. Analyze runs the ordinary LoRA analysis once; the plan (ranks, retention achieved,
size) is recomputed from the spectra whenever the settings change, without a second pass. The output carries a
normal lora_merge recipe, so it reloads on this tab and reproduces byte for byte.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, ttk

from .lora_merge import LoraInput, LoraMergeOptions, prune_plan, prune_text

RETENTION_MIN, RETENTION_MAX = 0.5, 1.0


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


class PruneTab(ttk.Frame):
    def __init__(self, master, app):
        from .gui import PAD, FileSlot, RowList, _labeled, px
        super().__init__(master, padding=px(8))
        self.app = app
        self.rep = None
        self.plan: dict | None = None
        self._px = px
        self.list = RowList(self, app, 1, "LoRA", initial=1)
        self.list.pack(fill="x")
        ttk.Label(self, text="Strength and block shaping apply before the SVD: a suppressed zone needs fewer components, a zeroed zone is dropped.",
                  style="Hint.TLabel").pack(anchor="w", padx=px(6))

        opts = ttk.LabelFrame(self, text="Rank rule", padding=px(6))
        opts.pack(fill="x", pady=px(6))
        self.retention = tk.StringVar(value="0.990")
        self.ret_entry = _labeled(opts, 0, "retention", lambda p: ttk.Entry(p, textvariable=self.retention, width=8, justify="right"),
                                  "energy each module keeps, 0.500 to 1.000 in steps of 0.001; the rank is the smallest that reaches it")
        self.ret_entry.bind("<FocusOut>", lambda _e: self._normalize())
        self.cap = tk.StringVar(value="16")
        _labeled(opts, 1, "rank cap", lambda p: ttk.Entry(p, textvariable=self.cap, width=8, justify="right"),
                 "no module above this rank; empty = no cap. The cap decides the size, the retention decides where it does not bind")
        self.floor = tk.StringVar(value="1")
        _labeled(opts, 2, "rank floor", lambda p: ttk.Entry(p, textvariable=self.floor, width=8, justify="right"),
                 "no module below this rank, so a weak but real change is not reduced to nothing")
        self.naming = _labeled(opts, 3, "naming", lambda p: ttk.Combobox(p, values=["comfy", "kohya", "input"], state="readonly", width=12),
                               "key convention of the output file")
        self.naming.set("comfy")
        self.dtype = _labeled(opts, 4, "dtype", lambda p: ttk.Combobox(p, values=["fp16", "bf16", "fp32"], state="readonly", width=12))
        self.dtype.set("fp16")
        self.out = FileSlot(opts, app, "output", save=True)
        self.out.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(px(6), 0))
        opts.columnconfigure(2, weight=1)

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=px(4))
        ttk.Button(btns, text="Analyze", command=self.analyze).pack(side="left", padx=px(6))
        self.btn_replan = ttk.Button(btns, text="Replan", command=self.replan, state="disabled")
        self.btn_replan.pack(side="left", padx=px(6))
        ttk.Button(btns, text="Prune", style="Accent.TButton", command=self.run).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Save recipe...", command=lambda: app.save_recipe(self)).pack(side="right", padx=px(6))
        ttk.Button(btns, text="Load recipe...", command=lambda: app.load_recipe(self)).pack(side="right", padx=px(6))

        res = ttk.LabelFrame(self, text="Plan", padding=px(6))
        res.pack(fill="both", expand=True, pady=px(4))
        self.summary = ttk.Label(res, text="no analysis yet", style="Hint.TLabel", justify="left", wraplength=px(1100))
        self.summary.pack(fill="x", anchor="w")
        cols = ("scope", "modules", "energy", "rank", "kept", "size", "capped")
        frame = ttk.Frame(res)
        frame.pack(fill="both", expand=True, pady=(px(4), 0))
        self.table = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        for c, w, txt, anchor in (("scope", 150, "scope", "w"), ("modules", 70, "modules", "e"), ("energy", 90, "energy share", "e"),
                                  ("rank", 130, "rank, sum in -> out", "e"), ("kept", 90, "retention", "e"),
                                  ("size", 170, "size in -> out", "e"), ("capped", 70, "capped", "e")):
            self.table.heading(c, text=txt)
            self.table.column(c, width=px(w), anchor=anchor, stretch=c == "scope")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ---- settings
    def _normalize(self):
        try:
            v = float(self.retention.get().replace(",", "."))
        except ValueError:
            v = 0.99
        v = min(RETENTION_MAX, max(RETENTION_MIN, round(v, 3)))
        self.retention.set(f"{v:.3f}")

    def rule(self) -> tuple[float, int | None, int]:
        self._normalize()
        cap = self.cap.get().strip()
        cap_v = int(cap) if cap.isdigit() and int(cap) > 0 else None
        floor = self.floor.get().strip()
        floor_v = int(floor) if floor.isdigit() and int(floor) > 0 else 1
        return float(self.retention.get()), cap_v, floor_v

    def inputs(self) -> list[LoraInput]:
        return self.list.inputs()[:1]

    def options(self) -> LoraMergeOptions:
        ret, cap, floor = self.rule()
        return LoraMergeOptions(rank_mode="dynamic", retention=ret, rank_cap=cap, rank_floor=floor,
                                naming=self.naming.get(), dtype=self.dtype.get())

    def to_recipe(self) -> dict:
        return {"function": "lora_merge", "inputs": [i.to_dict() for i in self.inputs()],
                "options": self.options().to_dict(), "output": self.out.get()}

    def from_recipe(self, r: dict):
        inputs = [LoraInput.from_dict(d) for d in r.get("inputs", [])][:1]
        self.list.load(inputs)
        o = LoraMergeOptions.from_dict(r.get("options"))
        self.retention.set(f"{float(o.retention):.3f}")
        self.cap.set(str(o.rank_cap) if o.rank_cap else "")
        self.floor.set(str(o.rank_floor))
        self.naming.set(o.naming)
        self.dtype.set(o.dtype)
        self.out.set(r.get("output"))

    def _check(self) -> bool:
        if not self.inputs():
            messagebox.showwarning("Prune", "Choose a LoRA file.")
            return False
        return True

    # ---- actions
    def analyze(self):
        if not self._check():
            return
        inputs, opts = self.inputs(), self.options()
        gpu = self.app.use_gpu()

        def job(progress, cancel, log):
            from .lora_merge import analyze_lora_merge
            rep = analyze_lora_merge(inputs, opts, gpu, progress=progress, cancel=cancel)
            return rep

        def done(rep):
            self.rep = rep
            self.app.log(rep.text())
            self.replan()
            self.app.tab_spectrum.set_report(rep, {"uniform": opts.rank_cap, "groups": {}})
            self.app.log("the spectrum is on the Spectrum tab")
        self.app.run_job("Analyzing LoRA", job, done)

    def replan(self):
        if self.rep is None:
            return
        ret, cap, floor = self.rule()
        self.plan = prune_plan(self.rep, ret, cap, floor, self.dtype.get())
        self._fill()
        self.app.log(prune_text(self.plan))

    def _fill(self):
        t = self.table
        t.delete(*t.get_children())
        p = self.plan
        if p is None:
            self.summary.configure(text="no analysis yet")
            self.btn_replan.configure(state="disabled")
            return
        tot = p["total"]
        src = self.inputs()[0].path if self.inputs() else ""
        on_disk = os.path.getsize(src) if src and os.path.exists(src) else None
        parts = [f"{os.path.basename(src)}" if src else "LoRA",
                 f"{tot['modules']} modules",
                 f"rank {p['rank_in_min']}-{p['rank_in_max']} in, {p['rank_min']}-{p['rank_max']} out",
                 f"retention {ret_fmt(tot['kept'])} achieved for {p['retention']:.3f} asked",
                 f"{tot['capped']} modules held at the cap" if p["cap"] else "no cap",
                 f"size {_mb(tot['size_in'])} -> {_mb(tot['size'])} ({100 * (1 - tot['size'] / tot['size_in']):.0f}% smaller)" if tot["size_in"] else ""]
        if on_disk:
            parts.append(f"file on disk {_mb(on_disk)}")
        self.summary.configure(text="   ·   ".join(x for x in parts if x))

        def row(label, a):
            t.insert("", "end", values=(label, a["modules"], f"{100 * a['energy_share']:.1f}%",
                                        f"{a['rank_in']} -> {a['rank']}", ret_fmt(a["kept"]),
                                        f"{_mb(a['size_in'])} -> {_mb(a['size'])}", a["capped"]))
        row("all", tot)
        for g, a in p["groups"].items():
            row(g, a)
        for b, a in p["blocks"].items():
            row(f"block {b}" if b is not None else "non block", a)
        self.btn_replan.configure(state="normal")

    def run(self):
        if not self._check():
            return
        out = self.out.get()
        if not out:
            messagebox.showwarning("Prune", "Choose an output file.")
            return
        inputs, opts = self.inputs(), self.options()
        gpu = self.app.use_gpu()

        def job(progress, cancel, log):
            from .lora_merge import merge_loras
            return merge_loras(inputs, out, opts, gpu, progress, cancel, log)

        def done(res):
            self.app.log(f"wrote {res.path}: {res.modules} modules, rank {res.rank_min}-{res.rank_max}, "
                         f"energy kept >= {res.kept_min * 100:.2f}%, {_mb(res.size_bytes)}, {res.seconds:.1f}s; "
                         f"compare it against the source at fixed seeds and across seeds before publishing")
        self.app.run_job("Pruning LoRA", job, done)


def ret_fmt(v: float) -> str:
    return f"{100 * v:.2f}%"
