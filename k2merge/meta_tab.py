"""The Metadata tab: what a file says about itself, and the commands that change it.

Reading is free, so the tab loads a file as soon as its path resolves: the summary of the Inspect button, the
metadata as a table with the JSON blobs readable in the detail pane, and the findings a publisher should see.
Redaction patches the header in place, which is instant on a 26 GB file and keeps the data untouched; the
original header goes to a sidecar first, so Undo is exact. Stripping everything writes a new file, because it
is the irreversible one and the point of it is a file with nothing left to trace.

The tab writes no weights of its own: every path goes through k2merge.meta.
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from .meta import (ACTIONS, LOAD_BEARING, MODELSPEC_FIELDS, HeaderTooLong, apply_findings, modelspec_defaults,
                   backup_path, data_sha256, fits_in_place, lineage_text, metadata_rows, pretty, read_metadata,
                   read_modelspec, restore_header, scan_metadata, set_modelspec, strip_all, write_metadata)
from .st_io import SafetensorsError

JSON_EXPORT = [("metadata", "*.json"), ("all files", "*.*")]


class MetaTab(ttk.Frame):
    def __init__(self, master, app):
        from .gui import PAD, ST_FILES, Collapsible, FileSlot, px
        super().__init__(master, padding=px(8))
        self.app = app
        self.px = px
        self.st_files = ST_FILES
        self.path: str | None = None
        self.meta: dict = {}
        self.findings: list = []
        self.is_lora = False
        self._loading = False

        card = ttk.LabelFrame(self, text="File", padding=px(6))
        card.pack(fill="x", pady=px(4))
        self.slot = FileSlot(card, app, "file", hint="a checkpoint or a LoRA, read as soon as the path exists; nothing is written until you ask")
        self.slot.pack(fill="x")
        self.slot.var.trace_add("write", lambda *_: self._path_changed())

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=px(2))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(body, text="What this file is", padding=px(6))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, px(4)))
        self.summary = tk.Text(left, height=5, width=46, wrap="word", relief="solid", borderwidth=1,
                               font=tkfont.nametofont("TkDefaultFont"))
        self.summary.pack(fill="both", expand=True)
        self.summary.configure(state="disabled")
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(px(4), 0))
        ttk.Button(row, text="Data hash", command=self.hash_data).pack(side="left")
        ttk.Button(row, text="Lineage", command=self.show_lineage).pack(side="left", padx=px(4))
        self.send = ttk.Menubutton(row, text="Send to Advisor")
        menu = tk.Menu(self.send, tearoff=0)
        for slot in ("A", "B", "C"):
            menu.add_command(label=f"as {slot}", command=lambda s=slot: self.send_to_advisor(s))
        self.send.configure(menu=menu)
        self.send.pack(side="left", padx=px(4))

        right = ttk.LabelFrame(body, text="Metadata", padding=px(6))
        right.grid(row=0, column=1, sticky="nsew")
        cols = ("key", "size", "value")
        self.table = ttk.Treeview(right, columns=cols, show="headings", height=5)
        for c, w, txt in (("key", 150, "key"), ("size", 50, "chars"), ("value", 260, "value")):
            self.table.heading(c, text=txt)
            self.table.column(c, width=px(w), anchor="w", stretch=c == "value")
        self.table.pack(fill="x")
        self.table.bind("<<TreeviewSelect>>", lambda _e: self._show_value())
        self.detail = tk.Text(right, height=4, width=46, wrap="none", relief="solid", borderwidth=1,
                              font=tkfont.nametofont("TkFixedFont"))
        self.detail.pack(fill="both", expand=True, pady=(px(4), 0))
        self.detail.configure(state="disabled")

        fnd = ttk.LabelFrame(self, text="Findings: what a published file would reveal", padding=px(6))
        fnd.pack(fill="x", pady=px(4))
        opts = ttk.Frame(fnd)
        opts.pack(fill="x")
        cats = ttk.Frame(fnd)
        cats.pack(fill="x")
        ttk.Label(opts, text="paths become").pack(side="left")
        self.policy = ttk.Combobox(opts, values=["basename", "placeholder", "drop"], state="readonly", width=12)
        self.policy.set("basename")
        self.policy.pack(side="left", padx=px(4))
        self.policy.bind("<<ComboboxSelected>>", lambda _e: self._rescan())
        self.backup = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="save the original header next to the file, so Undo is exact",
                        variable=self.backup).pack(side="left", padx=px(12))
        ttk.Label(cats, text="also remove").pack(side="left")
        self.ss = tk.BooleanVar(value=False)
        ttk.Checkbutton(cats, text="the trainer's metadata (ss_*, ot_*)", variable=self.ss,
                        command=self._rescan).pack(side="left", padx=px(6))
        self.thumb = tk.BooleanVar(value=False)
        ttk.Checkbutton(cats, text="an embedded thumbnail", variable=self.thumb,
                        command=self._rescan).pack(side="left", padx=px(6))
        self.workflow = tk.BooleanVar(value=False)
        ttk.Checkbutton(cats, text="an embedded ComfyUI workflow and prompt", variable=self.workflow,
                        command=self._rescan).pack(side="left", padx=px(6))
        fcols = ("key", "what", "action")
        self.ftable = ttk.Treeview(fnd, columns=fcols, show="headings", height=3)
        for c, w, txt in (("key", 150, "key"), ("what", 300, "found"), ("action", 90, "action")):
            self.ftable.heading(c, text=txt)
            self.ftable.column(c, width=px(w), anchor="w", stretch=c == "what")
        self.ftable.pack(fill="x", pady=(px(4), 0))
        self.ftable.bind("<Double-1>", lambda _e: self.cycle_action())
        ttk.Label(fnd, text="double click a finding to change what happens to it; skip leaves it alone",
                  style="Hint.TLabel").pack(anchor="w", pady=(px(2), 0))

        act = ttk.Frame(self)
        act.pack(fill="x", pady=px(4))
        self.btn_redact = ttk.Button(act, text="Redact paths", style="Accent.TButton", command=self.redact_in_place)
        self.btn_redact.pack(side="left", padx=(0, px(6)))
        ttk.Button(act, text="Save redacted copy...", command=self.redact_as_copy).pack(side="left", padx=px(6))
        ttk.Button(act, text="Strip all metadata...", command=self.strip_everything).pack(side="left", padx=px(6))
        self.btn_undo = ttk.Button(act, text="Undo", command=self.undo, state="disabled")
        self.btn_undo.pack(side="left", padx=px(6))
        ttk.Button(act, text="Export JSON...", command=self.export_json).pack(side="right", padx=px(6))
        self.btn_recipe = ttk.Button(act, text="Load recipe into tab", command=self.load_recipe, state="disabled")
        self.btn_recipe.pack(side="right", padx=px(6))

        self.editor = Collapsible(self, "Metadata editor (modelspec fields)")
        self.editor.pack(fill="x", pady=px(4))
        grid = self.editor.body
        self.fields: dict[str, tk.StringVar] = {}
        for i, (name, label, required, hint) in enumerate(MODELSPEC_FIELDS):
            var = tk.StringVar()
            var.trace_add("write", lambda *_: self._refresh_fit())
            self.fields[name] = var
            ttk.Label(grid, text=label + (" *" if required else ""), width=16).grid(row=i, column=0, sticky="w", **PAD)
            ttk.Entry(grid, textvariable=var, width=44).grid(row=i, column=1, sticky="we", **PAD)
            if hint:
                ttk.Label(grid, text=hint, style="Hint.TLabel").grid(row=i, column=2, sticky="w", padx=px(6))
        grid.columnconfigure(1, weight=1)
        erow = ttk.Frame(grid)
        erow.grid(row=len(MODELSPEC_FIELDS), column=0, columnspan=3, sticky="w", **PAD)
        ttk.Button(erow, text="Apply", command=self.apply_editor).pack(side="left")
        ttk.Button(erow, text="Apply to a copy...", command=lambda: self.apply_editor(copy=True)).pack(side="left", padx=px(6))
        ttk.Button(erow, text="Fill the hash", command=self.hash_into_editor).pack(side="left", padx=px(6))
        self.fit = ttk.Label(erow, text="", style="Hint.TLabel")
        self.fit.pack(side="left", padx=px(8))
        ttk.Label(grid, text="* the standard calls these required; an empty field removes its key",
                  style="Hint.TLabel").grid(row=len(MODELSPEC_FIELDS) + 1, column=1, sticky="w", **PAD)
        app.themed.append(self)

    # ---- theme
    def apply_theme(self, C: dict):
        for t in (self.summary, self.detail):
            t.configure(bg=C["log_bg"], fg=C["fg"], insertbackground=C["fg"])

    # ---- loading
    def _path_changed(self):
        p = self.slot.get()
        if p and os.path.isfile(p) and os.path.abspath(p) != (os.path.abspath(self.path) if self.path else None):
            self.load(p)

    def set_file(self, path: str):
        self.slot.set(path)
        if self.path != path:
            self.load(path)

    def load(self, path: str):
        from .inspect_file import inspect_path
        self.path = path
        try:
            self.meta = read_metadata(path)
        except (SafetensorsError, OSError) as e:
            self.meta = {}
            self._set_text(self.summary, f"{os.path.basename(path)}\n{e}")
            self._fill_table()
            self._rescan()
            return
        info = inspect_path(path)
        self.is_lora = info.get("kind") == "lora"
        lines = [info["text"]]
        if not self.meta:
            lines.append("no metadata at all, as in the official bf16 and int8 convrot files")
        else:
            kept = [k for k in self.meta if k in LOAD_BEARING]
            if kept:
                lines.append(f"{', '.join(kept)} is load bearing: the file does not load without it, and it is never removed")
        self._set_text(self.summary, "\n".join(lines))
        self._fill_table()
        self._rescan()
        self._fill_editor()
        self.btn_recipe.configure(state="normal" if "merge_recipe" in self.meta else "disabled")
        self.btn_undo.configure(state="normal" if os.path.isfile(backup_path(path)) else "disabled")

    def _set_text(self, widget, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _fill_table(self):
        self.table.delete(*self.table.get_children())
        for key, size, preview in metadata_rows(self.meta):
            self.table.insert("", "end", values=(key, size, preview))
        self._set_text(self.detail, "")

    def _show_value(self):
        sel = self.table.selection()
        if not sel:
            return
        key = self.table.item(sel[0], "values")[0]
        self._set_text(self.detail, pretty(self.meta.get(key, "")))

    def _fill_editor(self):
        cur = read_modelspec(self.meta)
        for k, v in modelspec_defaults(self.is_lora).items():
            cur[k] = cur.get(k) or v
        if not cur.get("merged_from"):
            cur["merged_from"] = self._merged_from()
        self._loading = True
        try:
            for name, var in self.fields.items():
                var.set(cur.get(name, ""))
        finally:
            self._loading = False
        self._refresh_fit()

    def _refresh_fit(self):
        """Adding a field moves the header, and a file this tool wrote has no room in its own: say so before
        the user presses Apply, because the difference is a millisecond against a full copy of the file."""
        if not self.path or self._loading:
            return
        new = set_modelspec(self.meta, {k: v.get() for k, v in self.fields.items()})
        if new == self.meta:
            self.fit.configure(text="")
        elif fits_in_place(self.path, new):
            self.fit.configure(text="Apply writes the header in place")
        else:
            self.fit.configure(text="the header has no room: Apply will offer a new file instead")

    def _merged_from(self) -> str:
        """The inputs of this file's own recipe, which is what the field is for."""
        try:
            from .recipe import recipe_from_file_metadata, recipe_inputs
            r = recipe_from_file_metadata(self.path)
        except Exception:                                   # noqa: BLE001
            return ""
        return ", ".join(os.path.basename(n) for _role, n in recipe_inputs(r)) if r else ""

    # ---- findings
    def _rescan(self):
        self.findings = scan_metadata(self.meta, self.policy.get(), training=self.ss.get(),
                                      thumbnail=self.thumb.get(), workflow=self.workflow.get())
        self.ftable.delete(*self.ftable.get_children())
        for i, f in enumerate(self.findings):
            what = f.text + (f"  (x{f.count})" if f.count > 1 else "")
            self.ftable.insert("", "end", iid=str(i), values=(f.key, what, f.action))
        self.btn_redact.configure(state="normal" if self.findings else "disabled")

    def cycle_action(self):
        sel = self.ftable.selection()
        if not sel:
            return
        i = int(sel[0])
        f = self.findings[i]
        f.action = ACTIONS[(ACTIONS.index(f.action) + 1) % len(ACTIONS)]
        self.ftable.item(sel[0], values=(f.key, self.ftable.item(sel[0], "values")[1], f.action))

    def _accepted(self):
        return [f for f in self.findings if f.action != "skip"]

    def _describe(self, findings) -> str:
        lines = [f"  {f.key}: {f.describe()}" for f in findings[:12]]
        if len(findings) > 12:
            lines.append(f"  ... and {len(findings) - 12} more")
        return "\n".join(lines)

    # ---- commands
    def _check(self) -> bool:
        if not (self.path and os.path.isfile(self.path)):
            messagebox.showwarning("Metadata", "Choose a file first.")
            return False
        return True

    def redact_in_place(self, ):
        if not self._check():
            return
        acc = self._accepted()
        if not acc:
            messagebox.showinfo("Metadata", "Nothing to redact.")
            return
        if not messagebox.askokcancel("Redact paths", f"Change {os.path.basename(self.path)} in place:\n\n"
                                      f"{self._describe(acc)}\n\nOnly the header is rewritten; the weights are not touched."
                                      + ("\nThe original header is saved next to the file." if self.backup.get() else "")):
            return
        new = apply_findings(self.meta, acc)
        try:
            written, how = write_metadata(self.path, new, None, backup=self.backup.get(), log=self.app.log)
        except HeaderTooLong:
            messagebox.showinfo("Redact paths", "This file's header has no room for the change, which cannot happen "
                                                "for a redaction. Use Save redacted copy instead.")
            return
        except SafetensorsError as e:
            messagebox.showerror("Redact paths", str(e))
            return
        self.app.log(f"redacted {len(acc)} finding(s) in {written} ({how})")
        self.load(self.path)

    def redact_as_copy(self):
        if not self._check():
            return
        acc = self._accepted()
        if not acc:
            messagebox.showinfo("Metadata", "Nothing to redact.")
            return
        out = self._ask_out("redacted")
        if not out:
            return
        self._write_job(apply_findings(self.meta, acc), out, f"Writing a redacted copy of {os.path.basename(self.path)}")

    def strip_everything(self):
        if not self._check():
            return
        kept = strip_all(self.meta)
        lost = [k for k in self.meta if k not in kept]
        if not lost:
            messagebox.showinfo("Strip all metadata", "This file carries nothing that could be stripped.")
            return
        keep_text = ("\n\nKept, because the file does not load without it: " + ", ".join(kept)) if kept else \
                    "\n\nNothing is kept: the result has no metadata at all, like the official bf16 and int8 files."
        if not messagebox.askokcancel("Strip all metadata",
                                      f"Remove {len(lost)} key(s) from a new copy: {', '.join(list(lost)[:8])}"
                                      + (" ..." if len(lost) > 8 else "") + keep_text
                                      + "\n\nThis cannot be undone from the new file: the recipe, the training "
                                        "metadata and the modelspec fields are gone from it for good."):
            return
        out = self._ask_out("stripped")
        if not out:
            return
        self._write_job(kept, out, f"Stripping the metadata of {os.path.basename(self.path)}")

    def apply_editor(self, copy: bool = False):
        if not self._check():
            return
        values = {k: v.get() for k, v in self.fields.items()}
        new = set_modelspec(self.meta, values)
        if new == self.meta:
            messagebox.showinfo("Metadata editor", "Nothing changed.")
            return
        if copy:
            out = self._ask_out("described")
            if out:
                self._write_job(new, out, f"Writing {os.path.basename(out)}")
            return
        try:
            written, how = write_metadata(self.path, new, None, backup=self.backup.get(), log=self.app.log)
        except HeaderTooLong as e:
            if not messagebox.askokcancel("Metadata editor", f"{e}.\n\nWrite a new file instead?"):
                return
            out = self._ask_out("described")
            if out:
                self._write_job(new, out, f"Writing {os.path.basename(out)}")
            return
        except SafetensorsError as e:
            messagebox.showerror("Metadata editor", str(e))
            return
        self.app.log(f"metadata written to {written} ({how})")
        self.load(self.path)

    def undo(self):
        if not self._check():
            return
        try:
            restore_header(self.path)
        except SafetensorsError as e:
            messagebox.showerror("Undo", str(e))
            return
        self.app.log(f"restored the saved header of {os.path.basename(self.path)}")
        self.load(self.path)

    def hash_data(self):
        if not self._check():
            return
        p = self.path

        def job(progress, cancel, log):
            return data_sha256(p, progress=progress, cancel=cancel)

        def done(h):
            self.app.log(f"{os.path.basename(p)}: data sha256 {h}")
            self._set_text(self.summary, self.summary.get("1.0", "end").rstrip() + f"\ndata sha256: {h}")
        self.app.run_job("Hashing the tensor data", job, done)

    def hash_into_editor(self):
        if not self._check():
            return
        p = self.path

        def job(progress, cancel, log):
            return data_sha256(p, progress=progress, cancel=cancel)

        def done(h):
            self.fields["hash_sha256"].set(h)
            self.editor.set_open(True)
            self.app.log(f"{os.path.basename(p)}: data sha256 {h}")
        self.app.run_job("Hashing the tensor data", job, done)

    def show_lineage(self):
        if not self._check():
            return
        text = lineage_text(self.path)
        self._set_text(self.detail, text)
        self.app.log(text)

    def load_recipe(self):
        if not self._check():
            return
        if not self.app.open_recipe(self.path):
            messagebox.showinfo("Recipe", "This file carries no recipe.")

    def send_to_advisor(self, slot: str):
        if not self._check():
            return
        tab = self.app.tab_advisor
        {"A": tab.A, "B": tab.B, "C": tab.C}[slot].set(self.path)
        self.app.nb.select(tab)
        self.app.log(f"{os.path.basename(self.path)} -> Advisor slot {slot}")

    def export_json(self):
        if not self._check():
            return
        p = filedialog.asksaveasfilename(title="Export metadata", filetypes=JSON_EXPORT, defaultextension=".json",
                                         initialfile=os.path.basename(self.path) + ".metadata.json")
        if not p:
            return
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=1, ensure_ascii=False)
        self.app.log(f"metadata exported: {p}")

    # ---- helpers
    def _ask_out(self, suffix: str) -> str | None:
        stem, ext = os.path.splitext(os.path.basename(self.path))
        return filedialog.asksaveasfilename(title="Write to", filetypes=self.st_files, defaultextension=".safetensors",
                                            initialdir=os.path.dirname(self.path),
                                            initialfile=f"{stem}_{suffix}{ext}") or None

    def _write_job(self, meta: dict, out: str, label: str):
        src = self.path

        def job(progress, cancel, log):
            return write_metadata(src, meta, out, progress=progress, cancel=cancel, log=log)

        def done(res):
            written, how = res
            self.app.log(f"wrote {written} ({how}); the tensor data is a byte for byte copy")
            self.set_file(written)
        self.app.run_job(label, job, done)
