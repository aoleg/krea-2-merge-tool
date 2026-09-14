"""The Spectrum tab: plots of an analysis report.

Four views: the spectrum of one tensor, the cumulative energy of every tensor,
a layer map, and a comparison of two saved analyses. matplotlib is imported
here and only here; without it the tab shows the tables and a message. All
computation happens before the report reaches the tab, on the worker thread;
this module only draws, on the Tk thread.
"""
from __future__ import annotations

import csv
import os
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from .analysis import CANDIDATE_RANKS, AnalysisReport, spectrum_stem

SPECTRUM_FILES = [("saved analysis", "*.spectrum.json"), ("all files", "*.*")]
VIEWS = (("layer", "Per layer spectrum"), ("all", "All layers"), ("map", "Layer map"), ("compare", "Compare runs"))
MAP_VALUES = (("rel_change", "relative change"), ("effective_rank", "effective rank"), ("rank90", "rank at 90% (raw)"),
              ("rank90_dn", "rank at 90% (denoised)"), ("energy_above_noise", "energy above the noise edge"))
LEAF_ORDER = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down")
HELP = """What to look for

Knee or no knee. A few large singular values, a steep drop and a flat tail mean a low intrinsic rank; truncating at the knee loses little. A slow decay with no knee means the change is diffuse: any truncation drops real signal, and the honest options are a much higher rank, a checkpoint merge, or a LoRA that carries only part of the change.

Where the tail sits against the noise edge. Singular values at or below the dashed edge cannot be told from rounding or quantization noise. Base the rank on the denoised energy. If a large share of the raw energy sits below the edge, a "rank in the thousands" figure was mostly noise. If the edge sits far below the tail, the diffuse change is real and denoising will not rescue a small rank.

Denoised rank at 90% across layers. Look at the median and the worst layers. If most layers are fine at 64 and a few need hundreds, use the per group ranks instead of raising the uniform rank.

Stable rank against effective rank. Stable rank near 1 with a high effective rank means one dominant direction plus a diffuse cloud, typical of a global style or brightness shift on top of small changes everywhere.

Energy outside the LoRA's reach. Change in mod.lin, the norm scales and the biases cannot be carried by a LoRA at any rank. If that share is large, extraction cannot reproduce the fine tune and a checkpoint merge is the right tool.

Relative change by leaf and block. Large change in txtmlp, txtfusion and mod.lin points to prompt following or style lock in; even change across the MLP leaves of every block points to broad feature drift. Soft evidence only.

Zero delta tensors were frozen or skipped by the training harness; they never belong in the LoRA.

Comparing two runs. The same delta norm with a flatter spectrum means the more diffuse change. If raw minus unaligned is markedly flatter than unaligned minus base, the alignment stage altered the model in a less structured way, consistent with memorizing a small dataset. Not proof; the final check remains a fixed seed generation comparison."""


def _leaf(name: str) -> str:
    """'blocks.3.attn.wq' -> 'attn.wq'; 'blocks.3.attn.qknorm.qnorm.scale' -> 'attn.qknorm.qnorm.scale'."""
    parts = name.split(".")
    return ".".join(parts[2:]) if len(parts) > 2 and parts[0] == "blocks" else name


def _fmt(v, pct=False, digits=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if pct:
        return f"{v * 100:.2f}%"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def tensor_rows(rep: AnalysisReport) -> list:
    """Per tensor summary rows (the CSV and the fallback table)."""
    rows = []
    for m in rep.modules.values():
        ra, rd = m.rank_at_energy(), m.rank_at_energy(True)
        rows.append({"name": m.name, "kind": "target", "group": m.group, "block": m.block, "shape": "x".join(str(s) for s in m.shape),
                     "layouts": "/".join(m.layouts), "delta_fro": m.delta_fro, "base_fro": m.base_fro, "rel_change": m.rel_change,
                     "zero_fraction": m.zero_fraction, "effective_rank": m.effective_rank, "stable_rank": m.stable_rank,
                     "noise_edge": m.noise_edge, "n_above_noise": m.n_above_noise, "energy_above_noise": m.energy_above_noise,
                     "rank_at_0.9": ra[0.9], "rank_at_0.99": ra[0.99], "rank_at_0.9_denoised": rd[0.9], "rank_at_0.99_denoised": rd[0.99],
                     "all_zero": m.all_zero})
    for o in rep.others:
        rows.append({"name": o.name, "kind": o.kind, "group": o.group, "block": o.block, "shape": "x".join(str(s) for s in o.shape),
                     "layouts": "", "delta_fro": o.delta_fro, "base_fro": o.base_fro, "rel_change": o.rel_change, "zero_fraction": None,
                     "effective_rank": None, "stable_rank": None, "noise_edge": None, "n_above_noise": None, "energy_above_noise": None,
                     "rank_at_0.9": None, "rank_at_0.99": None, "rank_at_0.9_denoised": None, "rank_at_0.99_denoised": None,
                     "all_zero": o.all_zero})
    return rows


CSV_COLUMNS = ("name", "kind", "group", "block", "shape", "layouts", "delta_fro", "base_fro", "rel_change", "zero_fraction",
               "effective_rank", "stable_rank", "noise_edge", "n_above_noise", "energy_above_noise",
               "rank_at_0.9", "rank_at_0.99", "rank_at_0.9_denoised", "rank_at_0.99_denoised", "all_zero")


def write_csv(rep: AnalysisReport, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in tensor_rows(rep):
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS})


class SpectrumTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.configure(padding=self._px(8))
        self.app = app
        self.rep: AnalysisReport | None = None
        self.other: AnalysisReport | None = None
        self.ranks: dict = {}
        self.view = tk.StringVar(value="layer")
        self.tensor = tk.StringVar(value="")
        self.filter = tk.StringVar(value="")
        self.normalize = tk.BooleanVar(value=False)
        self.show_null = tk.BooleanVar(value=True)
        self.denoised = tk.BooleanVar(value=False)
        self.map_value = tk.StringVar(value=MAP_VALUES[0][1])
        self._names: list = []
        self._map_index: dict = {}
        self.has_mpl = False
        try:
            import matplotlib
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure
            self.has_mpl = True
        except Exception:  # noqa: BLE001
            Figure = FigureCanvasTkAgg = NavigationToolbar2Tk = None
        P = self._px

        top = ttk.Frame(self)
        top.pack(fill="x")
        for key, label in VIEWS:
            ttk.Radiobutton(top, text=label, variable=self.view, value=key, command=self.redraw).pack(side="left", padx=(0, P(10)))
        ttk.Button(top, text="Help", command=self._help).pack(side="right", padx=P(3))
        ttk.Button(top, text="Export CSV...", command=self.export_csv).pack(side="right", padx=P(3))
        self.btn_png = ttk.Button(top, text="Export PNG...", command=self.export_png)
        self.btn_png.pack(side="right", padx=P(3))
        ttk.Button(top, text="Load comparison...", command=self.load_comparison).pack(side="right", padx=P(3))
        ttk.Button(top, text="Save analysis...", command=self.save_analysis).pack(side="right", padx=P(3))
        ttk.Button(top, text="Load analysis...", command=self.load_analysis).pack(side="right", padx=P(3))

        self.header = ttk.Label(self, text="no analysis yet: run Analyze on the LoRA merge or the Extract tab, or load a saved one",
                                style="Hint.TLabel", justify="left", wraplength=P(1200))
        self.header.pack(fill="x", pady=(P(4), P(2)))

        # per view controls, one frame each, shown by redraw()
        self.ctl = ttk.Frame(self)
        self.ctl.pack(fill="x")
        self.ctl_layer = ttk.Frame(self.ctl)
        ttk.Label(self.ctl_layer, text="tensor").pack(side="left", padx=(0, P(4)))
        self.cb_tensor = ttk.Combobox(self.ctl_layer, textvariable=self.tensor, width=34, state="readonly")
        self.cb_tensor.pack(side="left", padx=(0, P(8)))
        self.cb_tensor.bind("<<ComboboxSelected>>", lambda _e: self.redraw())
        ttk.Label(self.ctl_layer, text="filter").pack(side="left", padx=(0, P(4)))
        e = ttk.Entry(self.ctl_layer, textvariable=self.filter, width=16)
        e.pack(side="left", padx=(0, P(8)))
        e.bind("<KeyRelease>", lambda _e: self._refresh_names())
        ttk.Checkbutton(self.ctl_layer, text="normalize by the largest", variable=self.normalize, command=self.redraw).pack(side="left", padx=P(6))
        self.cb_null = ttk.Checkbutton(self.ctl_layer, text="null spectrum overlay", variable=self.show_null, command=self.redraw)
        self.cb_null.pack(side="left", padx=P(6))
        self.ctl_all = ttk.Frame(self.ctl)
        ttk.Checkbutton(self.ctl_all, text="denoised (spectra above the noise edge)", variable=self.denoised, command=self.redraw).pack(side="left")
        self.ctl_map = ttk.Frame(self.ctl)
        ttk.Label(self.ctl_map, text="cell value").pack(side="left", padx=(0, P(4)))
        cb = ttk.Combobox(self.ctl_map, textvariable=self.map_value, values=[l for _, l in MAP_VALUES], state="readonly", width=28)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self.redraw())
        ttk.Label(self.ctl_map, text="click a cell to open that tensor in the per layer view", style="Hint.TLabel").pack(side="left", padx=P(10))
        self.ctl_compare = ttk.Frame(self.ctl)
        self.compare_lbl = ttk.Label(self.ctl_compare, text="load a second saved analysis with \"Load comparison...\"", style="Hint.TLabel")
        self.compare_lbl.pack(side="left")

        # figure + side panel + table
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True, pady=(P(4), 0))
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)
        if self.has_mpl:
            # Logical 96 dpi and a small figure: matplotlib's Tk backend multiplies the dpi by the "tk scaling" ratio
            # (which carries the display DPI and the UI scale) and sets the canvas request to the figure's physical
            # size when the tab is mapped, so the request stays small and the figure grows with the window.
            self.fig = Figure(figsize=(6.4, 2.6), dpi=96)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.body)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.grid(row=0, column=0, sticky="nsew")
            # The backend's <Map> handler re-requests the figure's current physical size, which an earlier <Configure>
            # set to the whole allocated area; that would inflate the window's requested size. Shrink it back after it.
            self._canvas_req = (P(640), P(240))
            self.canvas_widget.bind("<Map>", lambda _e: self.after_idle(self._shrink_request), add="+")
            self._shrink_request()
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.body, pack_toolbar=False)
            self.toolbar.grid(row=1, column=0, sticky="ew")
            self.canvas.mpl_connect("button_press_event", self._on_click)
        else:
            self.fig = self.canvas = self.toolbar = None
            self.btn_png.configure(state="disabled")
            ttk.Label(self.body, text="plots need matplotlib (pip install matplotlib); the tables below still work",
                      style="Hint.TLabel").grid(row=0, column=0, sticky="nw")
        self.side = ttk.Label(self.body, text="", justify="left", anchor="nw", font=tkfont.nametofont("TkFixedFont"))
        self.side.grid(row=0, column=1, sticky="nsew", padx=(P(8), 0))
        self.table = ttk.Treeview(self.body, show="headings", height=5)
        self.table.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(P(4), 0))
        self.tsb = ttk.Scrollbar(self.body, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=self.tsb.set)
        self.tsb.grid(row=2, column=2, sticky="ns", pady=(P(4), 0))
        self.table.bind("<Double-1>", self._table_open)
        self.redraw()

    # ---- helpers
    def _px(self, n: int) -> int:
        from .gui import px
        return px(n)

    def _shrink_request(self):
        try:
            self.canvas_widget.configure(width=self._canvas_req[0], height=self._canvas_req[1])
        except tk.TclError:
            pass

    @property
    def C(self) -> dict:
        return self.app.C

    def _help(self):
        w = tk.Toplevel(self)
        w.title("Spectrum: what to look for")
        w.configure(bg=self.C["bg"])
        t = tk.Text(w, wrap="word", width=90, height=32, bg=self.C["log_bg"], fg=self.C["fg"], relief="flat",
                    font=tkfont.nametofont("TkDefaultFont"), padx=self._px(10), pady=self._px(8))
        t.insert("1.0", HELP)
        t.configure(state="disabled")
        t.pack(fill="both", expand=True)

    # ---- data in and out
    def set_report(self, rep: AnalysisReport, ranks: dict | None = None, mark: bool = True):
        """Show a report. ranks: {'uniform': int | None, 'groups': {group: rank}} of the producing tab."""
        self.rep = rep
        self.ranks = dict(ranks or {})
        self._refresh_names(select_largest=True)
        self.denoised.set(bool(rep.has_noise_model))
        self.header.configure(text=f"analysis: {rep.label}   ({len(rep.modules)} target tensors, {len(rep.others)} other tensors)"
                              + (f"\n{rep.noise_model.get('text', '')}" if rep.noise_model.get("text") else ""))
        if mark and hasattr(self.app, "nb"):
            try:
                self.app.nb.tab(self, text="Spectrum ●")
            except tk.TclError:
                pass
        self.redraw()

    def _refresh_names(self, select_largest: bool = False):
        if self.rep is None:
            self._names = []
            self.cb_tensor.configure(values=[])
            return
        flt = self.filter.get().strip().lower()
        mods = [m for m in self.rep.modules.values() if not flt or flt in m.name.lower()]
        self._names = [m.name for m in mods]
        self.cb_tensor.configure(values=self._names)
        if select_largest and mods:
            self.tensor.set(max(mods, key=lambda m: m.delta_fro).name)
        elif self.tensor.get() not in self._names:
            self.tensor.set(self._names[0] if self._names else "")

    def select_tensor(self, name: str):
        self.filter.set("")
        self._refresh_names()
        if name in self._names:
            self.tensor.set(name)
        self.view.set("layer")
        self.redraw()

    def _ask_stem(self, title: str) -> str | None:
        init = None
        if self.rep is not None:
            src = self.rep.run.get("target") or (self.rep.run.get("inputs") or [{}])[0].get("file")
            if src:
                init = os.path.dirname(src)
        p = filedialog.asksaveasfilename(title=title, filetypes=SPECTRUM_FILES, defaultextension=".spectrum.json",
                                         initialdir=init or self.app.settings.get("spectrum_dir"))
        return p or None

    def save_analysis(self):
        if self.rep is None:
            messagebox.showinfo("Spectrum", "Nothing to save yet.")
            return
        p = self._ask_stem("Save analysis")
        if not p:
            return
        npz, js = self.rep.save(spectrum_stem(p))
        self.app.settings["spectrum_dir"] = os.path.dirname(js)
        self.app.log(f"analysis saved: {js} and {os.path.basename(npz)}")

    def _load(self, title: str) -> AnalysisReport | None:
        p = filedialog.askopenfilename(title=title, filetypes=SPECTRUM_FILES, initialdir=self.app.settings.get("spectrum_dir"))
        if not p:
            return None
        try:
            rep = AnalysisReport.load(p)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Spectrum", str(e))
            return None
        self.app.settings["spectrum_dir"] = os.path.dirname(p)
        self.app.log(f"analysis loaded: {p} ({rep.label})")
        return rep

    def load_analysis(self):
        rep = self._load("Load analysis")
        if rep is not None:
            self.set_report(rep, mark=False)

    def load_comparison(self):
        rep = self._load("Load comparison")
        if rep is not None:
            self.other = rep
            self.view.set("compare")
            self.redraw()

    def export_png(self):
        if self.fig is None or self.rep is None:
            return
        p = filedialog.asksaveasfilename(title="Export PNG", filetypes=[("PNG", "*.png")], defaultextension=".png")
        if not p:
            return
        self.fig.savefig(p, dpi=150, facecolor=self.fig.get_facecolor())
        self.app.log(f"figure exported: {p}")

    def export_csv(self):
        if self.rep is None:
            messagebox.showinfo("Spectrum", "Nothing to export yet.")
            return
        p = filedialog.asksaveasfilename(title="Export CSV", filetypes=[("CSV", "*.csv")], defaultextension=".csv")
        if not p:
            return
        write_csv(self.rep, p)
        self.app.log(f"per tensor summary exported: {p}")

    # ---- theme
    def apply_theme(self, C: dict):
        if self.fig is not None:
            self._recolor_toolbar(C)
        self.redraw()

    def _recolor_toolbar(self, C: dict):
        if self.toolbar is None:
            return
        bg = C["bg"] if self.app.theme_name == "dark" else "#f0f0f0"
        fg = C["fg"] if self.app.theme_name == "dark" else "#000000"

        def walk(w):
            try:
                w.configure(bg=bg)
            except tk.TclError:
                pass
            for opt in (("fg", fg), ("activebackground", C["surface_2"]), ("activeforeground", fg), ("highlightbackground", bg)):
                try:
                    w.configure({opt[0]: opt[1]})
                except tk.TclError:
                    pass
            for ch in w.winfo_children():
                walk(ch)
        walk(self.toolbar)

    def _style_axes(self, ax, title: str = ""):
        C = self.C
        ax.set_facecolor(C["surface_2"])
        for sp in ax.spines.values():
            sp.set_color(C["border"])
        ax.tick_params(colors=C["fg"], labelcolor=C["fg"], labelsize=8)
        ax.xaxis.label.set_color(C["fg"])
        ax.yaxis.label.set_color(C["fg"])
        ax.title.set_color(C["fg"])
        ax.grid(True, color=C["border"], alpha=0.7, linewidth=0.6)
        if title:
            ax.set_title(title, fontsize=9)

    def _legend(self, ax):
        C = self.C
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=7, facecolor=C["surface_2"], edgecolor=C["border"], labelcolor=C["fg"])

    # ---- drawing
    def show_view(self, key: str):
        self.view.set(key)
        self.redraw()

    def redraw(self):
        for f in (self.ctl_layer, self.ctl_all, self.ctl_map, self.ctl_compare):
            f.pack_forget()
        key = self.view.get()
        {"layer": self.ctl_layer, "all": self.ctl_all, "map": self.ctl_map, "compare": self.ctl_compare}[key].pack(fill="x", pady=(0, self._px(2)))
        self.side.configure(text="")
        self.side.grid_remove()
        self._map_index = {}
        if self.fig is not None:
            self.fig.clear()
            self.fig.set_facecolor(self.C["bg"])
        if self.rep is None:
            self._fill_table(("info",), [("run Analyze on the LoRA merge or the Extract tab, or load a saved analysis",)])
            if self.fig is not None:
                self.canvas.draw_idle()
            return
        if key == "layer":
            self._draw_layer()
        elif key == "all":
            self._draw_all()
        elif key == "map":
            self._draw_map()
        else:
            self._draw_compare()
        if self.fig is not None:
            try:
                self.fig.tight_layout()
            except Exception:  # noqa: BLE001
                pass
            self.canvas.draw_idle()

    def _fill_table(self, columns: tuple, rows: list, widths: dict | None = None):
        t = self.table
        t.delete(*t.get_children())
        t.configure(columns=columns)
        for i, c in enumerate(columns):
            t.heading(c, text=c)
            t.column(c, width=self._px((widths or {}).get(c, 260 if i == 0 else 120)), anchor="w", stretch=True)
        for r in rows:
            t.insert("", "end", values=r)

    def _current(self):
        if self.rep is None:
            return None
        return next((m for m in self.rep.modules.values() if m.name == self.tensor.get()), None)

    def _configured_rank(self, m) -> int | None:
        g = self.ranks.get("groups") or {}
        if g:
            r = g.get(m.group, g.get("*"))
            if r:
                return int(r)
        u = self.ranks.get("uniform")
        return int(u) if u else None

    def _draw_layer(self):
        import numpy as np
        m = self._current()
        C = self.C
        self.cb_null.configure(state="normal" if (m is not None and m.sigma_null is not None) else "disabled")
        if m is None:
            self._fill_table(("info",), [("no tensor selected",)])
            return
        ra, rd = m.rank_at_energy(), m.rank_at_energy(True)
        side = [f"{m.name}", f"shape      {m.shape[0]} x {m.shape[1]}", f"group      {m.group}",
                f"layouts    {' / '.join(m.layouts) if m.layouts else 'LoRA factors'}",
                f"dtypes     {' / '.join(m.dtypes) if m.dtypes else '-'}",
                f"delta      {m.delta_fro:.4g}", f"base       {_fmt(m.base_fro)}", f"change     {_fmt(m.rel_change, pct=True)}",
                f"zero frac  {m.zero_fraction * 100:.2f}%", f"eff. rank  {m.effective_rank:.1f}", f"stable rk  {m.stable_rank:.2f}",
                f"noise edge {_fmt(m.noise_edge)}", f"above edge {m.n_above_noise}  ({m.energy_above_noise * 100:.1f}% energy)",
                f"rank@90%   {ra[0.9]} raw, {rd[0.9]} denoised", f"rank@99%   {ra[0.99]} raw, {rd[0.99]} denoised"]
        if m.resolved_by_svd:
            side.append("full SVD used (edge below Gram resolution)")
        self.side.configure(text="\n".join(side))
        self.side.grid()
        rows = [(f"{t * 100:g}%", ra[t], rd[t]) for t in ra]
        self._fill_table(("energy target", "rank raw", "rank denoised"), rows)
        if self.fig is None:
            return
        if m.all_zero:
            ax = self.fig.add_subplot(1, 1, 1)
            self._style_axes(ax, f"{m.name}: unchanged (no spectrum)")
            return
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2, sharex=ax1)
        sv = m.sv.numpy().astype("float64")
        top = sv[0] if sv[0] > 0 else 1.0
        scale = top if self.normalize.get() else 1.0
        x = np.arange(1, sv.size + 1)
        pos = sv > 0
        ax1.semilogy(x[pos], sv[pos] / scale, color=C["accent"], lw=1.2, label="singular values")
        if m.noise_edge > 0:
            ax1.axhline(m.noise_edge / scale, ls="--", color=C["bar_down"], lw=1.0, label=f"noise edge {m.noise_edge:.3g}")
        if m.sigma_null is not None and self.show_null.get():
            sn = m.sigma_null.numpy().astype("float64")
            p2 = sn > 0
            ax1.semilogy(np.arange(1, sn.size + 1)[p2], sn[p2] / scale, color=C["fg_muted"], lw=0.9, alpha=0.8, label="modeled noise alone")
        r = self._configured_rank(m)
        if r:
            for ax in (ax1, ax2):
                ax.axvline(r, color=C["fg_muted"], ls=":", lw=1.0, label=f"configured rank {r}" if ax is ax1 else None)
        self._style_axes(ax1, f"{m.name}  ({m.shape[0]} x {m.shape[1]})")
        ax1.set_ylabel("sigma / sigma[0]" if self.normalize.get() else "sigma", fontsize=8)
        self._legend(ax1)
        s2 = sv ** 2
        tot = s2.sum()
        ax2.plot(x, np.cumsum(s2) / tot, color=C["accent"], lw=1.2, label="cumulative energy, raw")
        if m.noise_edge > 0:
            dn = sv[sv > m.noise_edge] ** 2
            if dn.size:
                ax2.plot(np.arange(1, dn.size + 1), np.cumsum(dn) / dn.sum(), color=C["bar_flat"], lw=1.2, label="denoised")
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("rank", fontsize=8)
        ax2.set_ylabel("energy kept", fontsize=8)
        self._style_axes(ax2)
        self._legend(ax2)

    def _draw_all(self):
        import numpy as np
        C = self.C
        rep = self.rep
        dn = self.denoised.get()
        rows = [(r, _fmt(med, pct=True), _fmt(mn, pct=True), _fmt(medd, pct=True), _fmt(mnd, pct=True)) for r, med, mn, medd, mnd in rep.candidate_table()]
        self._fill_table(("rank", "median raw", "min raw", "median denoised", "min denoised"), rows)
        if self.fig is None:
            return
        ax = self.fig.add_subplot(1, 1, 1)
        mods = [m for m in rep.modules.values() if not m.all_zero]
        for m in mods:
            sv = (m.denoised_sv() if dn else m.sv).numpy().astype("float64")
            if sv.size == 0:
                continue
            s2 = sv ** 2
            ax.plot(np.arange(1, sv.size + 1), np.cumsum(s2) / s2.sum(), color=C["accent"], lw=0.6, alpha=0.12)
        x, med = rep.median_energy_curve(dn)
        if x.numel():
            ax.plot(x.numpy(), med.numpy(), color=C["fg"], lw=1.8, label="median over tensors")
        for r in CANDIDATE_RANKS:
            ax.axvline(r, color=C["fg_muted"], ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("rank (log)", fontsize=8)
        ax.set_ylabel("energy kept", fontsize=8)
        self._style_axes(ax, f"cumulative energy of every target tensor, {'denoised' if dn else 'raw'}; dotted = candidate ranks")
        self._legend(ax)

    def _map_value(self, m) -> float | None:
        key = next((k for k, l in MAP_VALUES if l == self.map_value.get()), "rel_change")
        if key == "rel_change":
            return m.rel_change
        if key == "effective_rank":
            return m.effective_rank
        if key == "rank90":
            return float(m.rank_needed(0.9))
        if key == "rank90_dn":
            return float(m.rank_needed(0.9, True))
        return m.energy_above_noise

    def _draw_map(self):
        import numpy as np
        C = self.C
        rep = self.rep
        blocks = sorted({m.block for m in rep.modules.values() if m.block is not None} | {o.block for o in rep.others if o.block is not None})
        nb = (max(blocks) + 1) if blocks else 0
        leaves = sorted({_leaf(m.name) for m in rep.modules.values() if m.block is not None},
                        key=lambda l: (LEAF_ORDER.index(l) if l in LEAF_ORDER else 99, l))
        M = np.full((max(len(leaves), 1), max(nb, 1)), np.nan)
        for m in rep.modules.values():
            if m.block is None:
                continue
            v = self._map_value(m)
            if v is not None:
                M[leaves.index(_leaf(m.name)), m.block] = v
        oleaves = sorted({_leaf(o.name) for o in rep.others if o.block is not None})
        O = np.full((max(len(oleaves), 1), max(nb, 1)), np.nan)
        zeros = []
        for o in rep.others:
            if o.block is None:
                continue
            i, j = oleaves.index(_leaf(o.name)), o.block
            O[i, j] = o.rel_change if o.rel_change is not None else np.nan
            if o.all_zero:
                zeros.append((i, j))
        # the tensors outside the blocks
        rows = []
        for m in rep.modules.values():
            if m.block is None:
                rows.append((m.name, _fmt(m.rel_change, pct=True), f"{m.effective_rank:.1f}", m.rank_needed(0.9), m.rank_needed(0.9, True),
                             _fmt(m.energy_above_noise, pct=True)))
        for o in rep.others:
            if o.block is None:
                rows.append((o.name + f"  ({o.kind})", _fmt(o.rel_change, pct=True), "-", "-", "-", "unchanged" if o.all_zero else "-"))
        self._fill_table(("tensor outside the blocks", "relative change", "effective rank", "rank@90% raw", "rank@90% denoised", "energy above edge"),
                         rows, {"tensor outside the blocks": 260})
        if self.fig is None:
            return
        gs = self.fig.add_gridspec(2, 1, height_ratios=[max(len(leaves), 1), max(len(oleaves), 1) * 0.8 + 0.5])
        ax = self.fig.add_subplot(gs[0])
        im = ax.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(leaves)))
        ax.set_yticklabels(leaves, fontsize=7)
        ax.set_xticks(range(nb))
        ax.set_xticklabels([str(b) for b in range(nb)], fontsize=7)
        self._style_axes(ax, f"target linears: {self.map_value.get()}")
        ax.grid(False)
        cb = self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cb.ax.tick_params(colors=C["fg"], labelsize=7)
        self._map_index["top"] = (ax, leaves, "blocks.{b}.{leaf}")
        ax2 = self.fig.add_subplot(gs[1])
        im2 = ax2.imshow(O, aspect="auto", cmap="magma", interpolation="nearest")
        for i, j in zeros:
            ax2.text(j, i, "0", ha="center", va="center", fontsize=6, color=C["fg"])
        ax2.set_yticks(range(len(oleaves)))
        ax2.set_yticklabels(oleaves, fontsize=7)
        ax2.set_xticks(range(nb))
        ax2.set_xticklabels([str(b) for b in range(nb)], fontsize=7)
        ax2.set_xlabel("block", fontsize=8)
        self._style_axes(ax2, "not extractable (norms, modulation): relative change; 0 = unchanged")
        ax2.grid(False)
        cb2 = self.fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.01)
        cb2.ax.tick_params(colors=C["fg"], labelsize=7)

    def _on_click(self, event):
        top = self._map_index.get("top")
        if top is None or event.inaxes is not top[0] or event.xdata is None:
            return
        ax, leaves, pattern = top
        j, i = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= i < len(leaves):
            self.select_tensor(pattern.format(b=j, leaf=leaves[i]))

    def _table_open(self, _e=None):
        sel = self.table.selection()
        if not sel or self.rep is None:
            return
        name = str(self.table.item(sel[0], "values")[0]).split("  (")[0]
        if any(m.name == name for m in self.rep.modules.values()):
            self.select_tensor(name)

    def _draw_compare(self):
        import numpy as np
        C = self.C
        rep, other = self.rep, self.other
        if other is None:
            self.compare_lbl.configure(text="load a second saved analysis with \"Load comparison...\"")
            self._fill_table(("info",), [("no comparison loaded",)])
            return
        self.compare_lbl.configure(text=f"A: {rep.label}      B: {other.label}")
        rows = [(n, a, b) for n, a, b in rep.compare_rows(other)]
        self._fill_table(("metric", "A", "B"), rows, {"metric": 240, "A": 220, "B": 220})
        if self.fig is None:
            return
        ax1 = self.fig.add_subplot(2, 2, 1)
        for r, lab, col in ((rep, "A", C["accent"]), (other, "B", C["bar_flat"])):
            x, med = r.median_energy_curve(False)
            if x.numel():
                ax1.plot(x.numpy(), med.numpy(), color=col, lw=1.4, label=f"{lab} raw")
            x, med = r.median_energy_curve(True)
            if x.numel() and r.has_noise_model:
                ax1.plot(x.numpy(), med.numpy(), color=col, lw=1.2, ls="--", label=f"{lab} denoised")
        ax1.set_xscale("log")
        ax1.set_ylim(0, 1.02)
        self._style_axes(ax1, "median cumulative energy")
        self._legend(ax1)
        for k, (title, fn) in enumerate((("effective rank", lambda m: m.effective_rank), ("stable rank", lambda m: m.stable_rank),
                                         ("relative change", lambda m: m.rel_change))):
            ax = self.fig.add_subplot(2, 2, k + 2)
            va = np.array([v for v in (fn(m) for m in rep.modules.values() if not m.all_zero) if v is not None], dtype="float64")
            vb = np.array([v for v in (fn(m) for m in other.modules.values() if not m.all_zero) if v is not None], dtype="float64")
            if va.size or vb.size:
                allv = np.concatenate([va, vb]) if va.size and vb.size else (va if va.size else vb)
                bins = np.linspace(float(allv.min()), float(allv.max()) or 1.0, 25)
                if va.size:
                    ax.hist(va, bins=bins, color=C["accent"], alpha=0.6, label="A")
                if vb.size:
                    ax.hist(vb, bins=bins, color=C["bar_flat"], alpha=0.6, label="B")
            self._style_axes(ax, title)
            self._legend(ax)
