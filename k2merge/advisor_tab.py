"""The Advisor tab: A, B, C, a goal, one analysis, a table of candidate recipes, and the ways to use them.

The measurement runs on the worker thread like every other job; proposing candidates for another goal or
output format reuses the measured features without a second pass. A candidate loads into the checkpoint or
the extract tab through their from_recipe, or runs into a folder through the recipe runner; two step chains
run both steps. The spectra of both deltas land on the Spectrum tab with the compare view preloaded.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from .advisor import GOALS, GOAL_ORDER, AdvisorReport, advice_stem, compute_features, propose, run_candidate
from .formats import OUTPUT_FORMATS

ADVICE_FILES = [("saved advice", "*.advice.json"), ("all files", "*.*")]


class AdvisorTab(ttk.Frame):
    def __init__(self, master, app):
        from .gui import PAD, FileSlot, _labeled, px
        super().__init__(master, padding=px(8))
        self.app = app
        self.rep: AdvisorReport | None = None
        self._px = px
        card = ttk.LabelFrame(self, text="Checkpoints", padding=px(6))
        card.pack(fill="x", pady=px(4))
        self.A = FileSlot(card, app, "A  keep", hint="the checkpoint whose behavior is kept")
        self.A.pack(fill="x")
        self.B = FileSlot(card, app, "B  donor", hint="the checkpoint to take something from")
        self.B.pack(fill="x")
        self.C = FileSlot(card, app, "C  ancestor", hint="optional: the common ancestor of A and B, usually an official Krea 2 file; without it the relation rules are skipped")
        self.C.pack(fill="x")

        opts = ttk.LabelFrame(self, text="Advice", padding=px(6))
        opts.pack(fill="x", pady=px(4))
        self.goal_labels = {k: v[0] for k, v in GOALS.items()}
        self.goal = _labeled(opts, 0, "goal", lambda p: ttk.Combobox(p, values=[self.goal_labels[k] for k in GOAL_ORDER], state="readonly", width=34))
        self.goal.set(self.goal_labels["add_content"])
        self.goal.bind("<<ComboboxSelected>>", lambda _e: self._goal_changed())
        self.goal_hint = ttk.Label(opts, text=GOALS["add_content"][1], style="Hint.TLabel")
        self.goal_hint.grid(row=0, column=2, sticky="w", padx=px(6))
        self.depth = _labeled(opts, 1, "depth", lambda p: ttk.Combobox(p, values=["full", "quick"], state="readonly", width=10),
                              "full = spectra of both changes (minutes); quick = norms and cosines only, no structure rules")
        self.depth.set("full")
        self.fmt = _labeled(opts, 2, "output", lambda p: ttk.Combobox(p, values=list(OUTPUT_FORMATS), state="readonly", width=14),
                            "format of the candidate merges; keep = A's format")
        self.fmt.set("keep")
        self.fmt.bind("<<ComboboxSelected>>", lambda _e: self._goal_changed())
        self.keep_intermediate = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="keep the bf16 intermediate of two step candidates", variable=self.keep_intermediate).grid(row=3, column=0, columnspan=3, sticky="w", **PAD)
        opts.columnconfigure(2, weight=1)

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=px(4))
        ttk.Button(btns, text="Advise", style="Accent.TButton", command=self.advise).pack(side="left", padx=px(6))
        ttk.Button(btns, text="Load advice...", command=self.load_advice).pack(side="left", padx=px(6))
        self.btn_save = ttk.Button(btns, text="Save advice...", command=self.save_advice, state="disabled")
        self.btn_save.pack(side="left", padx=px(6))
        self.btn_run_all = ttk.Button(btns, text="Run all...", command=self.run_all, state="disabled")
        self.btn_run_all.pack(side="right", padx=px(6))
        self.btn_run = ttk.Button(btns, text="Run selected...", command=self.run_selected, state="disabled")
        self.btn_run.pack(side="right", padx=px(6))
        self.btn_load = ttk.Button(btns, text="Load into tab", command=self.load_selected, state="disabled")
        self.btn_load.pack(side="right", padx=px(6))

        res = ttk.LabelFrame(self, text="Candidates", padding=px(6))
        res.pack(fill="both", expand=True, pady=px(4))
        self.summary = ttk.Label(res, text="no advice yet", style="Hint.TLabel", justify="left", wraplength=px(1100))
        self.summary.pack(fill="x", anchor="w")
        cols = ("n", "label", "recipe", "flags")
        self.table = ttk.Treeview(res, columns=cols, show="headings", height=4)
        for c, w, txt in (("n", 30, "#"), ("label", 300, "candidate"), ("recipe", 420, "recipe"), ("flags", 200, "flags")):
            self.table.heading(c, text=txt)
            self.table.column(c, width=px(w), anchor="w", stretch=c != "n")
        self.table.pack(fill="x", pady=(px(4), 0))
        self.table.bind("<<TreeviewSelect>>", lambda _e: self._show_detail())
        self.detail = tk.Text(res, height=5, wrap="word", relief="solid", borderwidth=1, font=tkfont.nametofont("TkDefaultFont"))
        self.detail.pack(fill="both", expand=True, pady=(px(4), 0))
        self.detail.configure(state="disabled")
        app.themed.append(self)

    # ---- helpers
    def goal_key(self) -> str:
        label = self.goal.get()
        return next((k for k, v in self.goal_labels.items() if v == label), "add_content")

    def apply_theme(self, C: dict):
        self.detail.configure(bg=C["log_bg"], fg=C["fg"], insertbackground=C["fg"])

    def selected(self):
        sel = self.table.selection()
        if not sel or self.rep is None:
            return None
        i = int(self.table.item(sel[0], "values")[0]) - 1
        return self.rep.candidates[i] if 0 <= i < len(self.rep.candidates) else None

    def _goal_changed(self):
        self.goal_hint.configure(text=GOALS[self.goal_key()][1])
        if self.rep is not None and self.rep.features:
            propose(self.rep, self.goal_key(), self.fmt.get())
            self._fill()
            self.app.log(f"--- candidates for the goal '{self.goal_labels[self.goal_key()]}'")
            self.app.log("\n".join(l for l in self.rep.text().splitlines() if l.startswith("  ") and not l.startswith("  group") and not l.startswith("  blocks") and not l.startswith("  txt") and not l.startswith("  proj") and not l.startswith("  all") and not l.startswith("  reading")))

    def _fill(self):
        t = self.table
        t.delete(*t.get_children())
        if self.rep is None:
            self.summary.configure(text="no advice yet")
            return
        f = self.rep.features
        parts = [f"analysis: {self.rep.label}", f"relation: {f.get('relation', '-')}"]
        if f.get("containment_text"):
            parts.append(f["containment_text"])
        if f.get("w_eq"):
            parts.append(f"equal contribution weight {f['w_eq']:.2f}")
        for key, lab in (("dA", "A's change"), ("dB", "B's change")):
            s = f.get("structure", {}).get(key)
            if s:
                parts.append(f"{lab} {s['class']}")
        if f.get("zone_focus"):
            parts.append(f"zone focus {f['zone_focus']}")
        if f.get("precision", {}).get("mismatch"):
            parts.append("precision mismatch")
        self.summary.configure(text="   ·   ".join(parts))
        for i, c in enumerate(self.rep.candidates, 1):
            t.insert("", "end", values=(i, c.label, c.summary(), "; ".join(c.flags)[:120]))
        has = bool(self.rep.candidates)
        for b in (self.btn_run, self.btn_run_all, self.btn_load):
            b.configure(state="normal" if has else "disabled")
        self.btn_save.configure(state="normal")
        if has:
            first = t.get_children()[0]
            t.selection_set(first)
            t.focus(first)
            self._show_detail()

    def _show_detail(self):
        c = self.selected()
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        if c is not None:
            lines = [c.label, "recipe: " + c.summary()]
            lines += ["why: " + w for w in c.rationale]
            if c.expect:
                lines.append("expect: " + c.expect)
            if c.compare:
                lines.append("compare: " + c.compare)
            lines += ["flag: " + fl for fl in c.flags]
            self.detail.insert("1.0", "\n".join(lines))
        self.detail.configure(state="disabled")

    def set_report(self, rep: AdvisorReport):
        self.rep = rep
        self.A.set(rep.run.get("A"))
        self.B.set(rep.run.get("B"))
        self.C.set(rep.run.get("C"))
        g = rep.run.get("goal", "add_content")
        if g in self.goal_labels:
            self.goal.set(self.goal_labels[g])
            self.goal_hint.configure(text=GOALS[g][1])
        if rep.run.get("depth") in ("full", "quick"):
            self.depth.set(rep.run["depth"])
        self._fill()
        sp = getattr(self.app, "tab_spectrum", None)
        if sp is not None and rep.repB is not None:
            sp.set_report(rep.repB, {"uniform": None, "groups": {}})
            sp.other = rep.repA
            if rep.repA is not None:
                sp.show_view("compare")

    # ---- actions
    def _check(self) -> bool:
        if not (self.A.get() and self.B.get()):
            messagebox.showwarning("Advisor", "Choose A and B.")
            return False
        return True

    def advise(self):
        if not self._check():
            return
        a, b, c = self.A.get(), self.B.get(), self.C.get() or None
        goal, depth, fmt = self.goal_key(), self.depth.get(), self.fmt.get()
        gpu = self.app.use_gpu()

        def job(progress, cancel, log):
            from .advisor import advise
            return advise(a, b, c, goal, depth, fmt, gpu, progress=progress, cancel=cancel)

        def done(rep):
            self.app.log(rep.text())
            self.set_report(rep)
            self.app.log("the spectra of both changes are on the Spectrum tab (B's change, A's change as the comparison)")
        self.app.run_job("Advising", job, done)

    def save_advice(self):
        if self.rep is None:
            return
        p = filedialog.asksaveasfilename(title="Save advice", filetypes=ADVICE_FILES, defaultextension=".advice.json",
                                         initialdir=self.app.settings.get("advice_dir"))
        if not p:
            return
        paths = self.rep.save(advice_stem(p))
        self.app.settings["advice_dir"] = os.path.dirname(p)
        self.app.log("advice saved: " + ", ".join(os.path.basename(x) for x in paths) + f" in {os.path.dirname(paths[0])}")

    def load_advice(self):
        p = filedialog.askopenfilename(title="Load advice", filetypes=ADVICE_FILES, initialdir=self.app.settings.get("advice_dir"))
        if not p:
            return
        try:
            rep = AdvisorReport.load(p)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Advisor", str(e))
            return
        if not rep.features:
            compute_features(rep)
        if not rep.candidates:
            propose(rep, rep.run.get("goal", "add_content"), self.fmt.get())
        self.app.settings["advice_dir"] = os.path.dirname(p)
        self.set_report(rep)
        self.app.log(f"advice loaded: {p}")

    def load_selected(self):
        c = self.selected()
        if c is None or not c.steps:
            return
        step = c.steps[0]
        target = {"ckpt_merge": self.app.tab_ckpt, "extract": self.app.tab_extract}.get(step["function"])
        if target is None:
            messagebox.showinfo("Advisor", f"no tab for a {step['function']} step")
            return
        r = dict(step)
        r["output"] = None
        target.from_recipe(r)
        self.app.nb.select(target)
        msg = f"candidate '{c.label}' loaded into the {'checkpoint' if step['function'] == 'ckpt_merge' else 'extract'} tab; choose an output file"
        if len(c.steps) > 1:
            msg += f"; this is step 1 of {len(c.steps)}, the next step is: {c.summary().split(' -> ', 1)[1]}"
        self.app.log(msg)

    def _ask_dir(self):
        d = filedialog.askdirectory(title="Folder for the candidate outputs", initialdir=self.app.settings.get("advice_dir"))
        if d:
            self.app.settings["advice_dir"] = d
        return d or None

    def _run(self, cands: list):
        if self.rep is None or not cands:
            return
        d = self._ask_dir()
        if not d:
            return
        rep, keep, gpu = self.rep, bool(self.keep_intermediate.get()), self.app.use_gpu()

        def job(progress, cancel, log):
            outs = []
            for c in cands:
                if not c.steps:
                    continue
                outs += run_candidate(rep, c, d, gpu, progress=progress, cancel=cancel, log=log, keep_intermediate=keep)
            return outs

        def done(outs):
            self.app.log(f"{len(outs)} file(s) written to {d}:\n" + "\n".join("  " + os.path.basename(o) for o in outs))
        self.app.run_job(f"Running {len(cands)} candidate(s)", job, done)

    def run_selected(self):
        c = self.selected()
        if c is not None:
            self._run([c])

    def run_all(self):
        if self.rep is not None:
            self._run(list(self.rep.candidates))
