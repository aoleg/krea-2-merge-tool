"""Advisor: measure how two fine tunes relate in weight space and propose merge recipes for a goal.

A is the checkpoint to keep, B the donor, C their common ancestor (optional). One streaming pass reads
each file once per module and yields two spectrum reports (dA = A - C, dB = B - C; dB = B - A without C)
plus a per tensor relation table. Features derived from those feed a small rule set that returns
candidates: complete recipes in the tool's recipe format, in chains of one or two steps, each with its
rationale, its expected effect and the comparison to make. The advisor ranks and scales; it never judges
images, and every candidate says so.

Thresholds are calibrated on the cases measured on 2026-09-14 and 2026-09-15 (anteros Turbo, Unaligned and
Raw, the official distillation, Kroma 0.3) and the report names that calibration.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import torch

from . import TOOL_NAME, __version__
from .analysis import (CANDIDATE_RANKS, AnalysisReport, OtherTensor, delta_stats, make_module_spectrum,
                       spectrum_with_guard)
from .blocks import Shaping
from .ckpt_merge import CkptMergeOptions, _Ckpt
from .engine import Cancelled, pick_device
from .extract import ExtractOptions
from .formats import FLOAT_TAGS, FormatError
from .keys import GROUPS, KREA2_BLOCKS, block_index, ckpt_module, group_of

# ----------------------------------------------------------------------------- goals and thresholds
GOALS = {
    "add_content": ("Add B's content, keep A stable", "A plus a dose of B's change; the dose scaled to A's own change"),
    "composition": ("Take B's composition, keep A's style", "B's change shaped onto the first third of the blocks"),
    "style": ("Take B's style, keep A's subjects", "B's change shaped onto the last third of the blocks"),
    "blend": ("Blend two siblings evenly", "both changes in full when they are independent, a cosine or TIES merge when aligned"),
    "deturbo": ("De-Turbo a Turbo fine tune", "A = the fine tune, B = official Raw, C = official Turbo; removes a dose of the distillation"),
    "lora": ("Distill B minus C into a LoRA", "an extraction at the rank the energy table picks, or the reason it cannot work"),
}
GOAL_ORDER = tuple(GOALS)

COS_INDEPENDENT = 0.10      # below: the changes share no direction
COS_ALIGNED = 0.50          # above: siblings
CONTAIN_RANGE = (0.85, 1.15)  # projection coefficient near 1 = one delta carries the other in full
CONTAIN_MIN_COS = 0.30
EFF_CONCENTRATED = 0.08     # median effective rank over the smaller dimension
EFF_DIFFUSE = 0.30            # the stable Unaligned change sits at 0.22, the collapsing anteros Raw extra at 0.48
ZONE_FOCUS = 2.0            # energy share over parameter share
NON_TARGET_WARN = 0.05      # share of the donor's energy outside the linears
NB_FACTOR_TRIGGER = 2.0     # per zone equal contribution weights differing by more than this set the non block weight
W_SWEEP = (0.5, 1.0, 2.0)
W_MIN, W_MAX = 0.05, 2.0        # add difference weights above 1 apply B's change more than once; the slider allows up to 10
LORA_ENERGY = 0.85
DETURBO_W = (0.4, 0.6, 1.0)
DETURBO_X = 0.7
ZONES = ("composition", "character", "style", "non_block")
CALIBRATION = ("thresholds calibrated on: anteros Turbo, Unaligned and Raw, the official Turbo distillation and Kroma 0.3 "
               "(2026-09-14 and 2026-09-15); they are weight space priors, and the image test sets the point")


def zone_of(block: int | None, count: int = KREA2_BLOCKS) -> str:
    if block is None:
        return "non_block"
    return ZONES[min(2, int(3 * block / count))]


def round_step(w: float, step: float = 0.05) -> float:
    return round(max(W_MIN, min(W_MAX, round(w / step) * step)), 2)


def _cls_structure(eff_over_dim: float | None) -> str:
    if eff_over_dim is None:
        return "unknown"
    if eff_over_dim < EFF_CONCENTRATED:
        return "concentrated"
    if eff_over_dim > EFF_DIFFUSE:
        return "diffuse"
    return "mixed"


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ----------------------------------------------------------------------------- data model
@dataclass
class Candidate:
    label: str
    goal: str
    steps: list                      # recipe dicts; "@prev" in a file field means the previous step's output
    rationale: list = field(default_factory=list)
    expect: str = ""
    compare: str = ""
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "goal": self.goal, "steps": self.steps, "rationale": list(self.rationale),
                "expect": self.expect, "compare": self.compare, "flags": list(self.flags)}

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(d["label"], d.get("goal", ""), list(d.get("steps", [])), list(d.get("rationale", [])),
                   d.get("expect", ""), d.get("compare", ""), list(d.get("flags", [])))

    def summary(self) -> str:
        parts = []
        for s in self.steps:
            if s["function"] == "ckpt_merge":
                b = s.get("B") or {}
                c = s.get("C")
                sh = Shaping.from_dict(b.get("shaping"))
                txt = f"{s['options']['method']}: A + {b.get('weight', 1.0):g} (B - {'C' if c else 'A'})" \
                    if s["options"]["method"] == "add_difference" else f"{s['options']['method']} w {b.get('weight', 1.0):g}"
                if not sh.is_flat():
                    txt += f", shaping {sh.preset}:{sh.modifier}:{sh.contrast:g}" + (f" @{sh.non_block:g}" if sh.non_block is not None else "")
                if s.get("loras"):
                    txt += ", + LoRA " + ", ".join(f"{os.path.basename(str(l['file']))} x {l.get('strength', 1.0):g}" for l in s["loras"])
                parts.append(txt)
            elif s["function"] == "extract":
                parts.append(f"extract rank {s['options'].get('rank')} of target - base")
            else:
                parts.append(s["function"])
        return " -> ".join(parts)


@dataclass
class AdvisorReport:
    run: dict = field(default_factory=dict)
    repA: AnalysisReport | None = None      # spectra of dA = A - C (None without C, or in quick depth)
    repB: AnalysisReport | None = None      # spectra of dB (None in quick depth)
    relation: dict = field(default_factory=dict)   # canon -> {name, group, block, shape, eA, eB, dot, wn, n}
    others: list = field(default_factory=list)     # (name, group, block, nA, nB, wn) of the non target tensors
    features: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def has_c(self) -> bool:
        return bool(self.run.get("C"))

    @property
    def label(self) -> str:
        stem = lambda p: os.path.splitext(os.path.basename(p or ""))[0]  # noqa: E731
        return f"{stem(self.run.get('A'))} + {stem(self.run.get('B'))}" + (f" vs {stem(self.run.get('C'))}" if self.has_c else "")

    # ---- persistence
    def to_dict(self) -> dict:
        return {"tool": f"{TOOL_NAME} {__version__}", "run": dict(self.run), "label": self.label,
                "features": self.features, "relation": list(self.relation.values()), "others": list(self.others),
                "candidates": [c.to_dict() for c in self.candidates], "notes": list(self.notes)}

    def save(self, stem: str) -> list:
        stem = advice_stem(stem)
        paths = []
        with open(stem + ".advice.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        paths.append(stem + ".advice.json")
        if self.repA is not None:
            paths += list(self.repA.save(stem + ".dA"))
        if self.repB is not None:
            paths += list(self.repB.save(stem + ".dB"))
        return paths

    @classmethod
    def load(cls, path: str) -> "AdvisorReport":
        stem = advice_stem(path)
        with open(stem + ".advice.json", encoding="utf-8") as f:
            d = json.load(f)
        rep = cls(run=dict(d.get("run", {})), features=dict(d.get("features", {})), notes=list(d.get("notes", [])))
        rep.relation = {r["canon"]: r for r in d.get("relation", [])}
        rep.others = [tuple(o) for o in d.get("others", [])]
        rep.candidates = [Candidate.from_dict(c) for c in d.get("candidates", [])]
        for attr, suffix in (("repA", ".dA"), ("repB", ".dB")):
            if os.path.exists(stem + suffix + ".spectrum.json"):
                setattr(rep, attr, AnalysisReport.load(stem + suffix + ".spectrum.json"))
        return rep

    # ---- text
    def text(self) -> str:
        r, f = self.run, self.features
        lines = [f"advisor: {self.label}", f"A: {r.get('A')} ({r.get('A_format')})", f"B: {r.get('B')} ({r.get('B_format')})"]
        if self.has_c:
            lines.append(f"C: {r.get('C')} ({r.get('C_format')})")
        else:
            lines.append("no C: the deltas are measured against A, the relation rules are skipped")
        lines.append(f"goal: {GOALS.get(r.get('goal', ''), ('?',))[0]}; depth {r.get('depth')}; {r.get('targets')} target tensors, {r.get('others')} other tensors")
        if f:
            lines.append("relation per group" + (" (dA = A - C, dB = B - C)" if self.has_c else " (dB = B - A)"))
            head = "  group          |dA| (% of W)   |dB| (% of W)     rho     cos   dA on dB   dB on dA"
            lines.append(head)
            for g, v in f["groups"].items():
                lines.append(f"  {g:14s} {_fmt_norm(v.get('nA'), v.get('relA'))}   {_fmt_norm(v['nB'], v['relB'])}  "
                             f"{_f(v.get('rho'), '6.2f')}  {_f(v.get('cos'), '6.3f')}  {_f(v.get('c_a_on_b'), '8.3f')}   {_f(v.get('c_b_on_a'), '8.3f')}")
            o = f["overall"]
            lines.append(f"  {'all':14s} {_fmt_norm(o.get('nA'), o.get('relA'))}   {_fmt_norm(o['nB'], o['relB'])}  "
                         f"{_f(o.get('rho'), '6.2f')}  {_f(o.get('cos'), '6.3f')}  {_f(o.get('c_a_on_b'), '8.3f')}   {_f(o.get('c_b_on_a'), '8.3f')}")
            if self.has_c:
                lines.append(f"  reading: the changes are {f['relation']}" + (f"; {f['containment_text']}" if f.get("containment_text") else "")
                             + (f"; equal contribution weight for B is {f['w_eq']:.2f}" if f.get("w_eq") else ""))
            for key, lab in (("dA", "A's change"), ("dB", "B's change")):
                s = f.get("structure", {}).get(key)
                if s:
                    lines.append(f"structure of {lab}: {s['class']} (median effective rank above the noise edge {(s['eff_over_dim'] or 0.0) * 100:.1f}% of the dimension"
                                 + (f", {(s['eff_over_dim_raw'] or 0.0) * 100:.1f}% raw" if s.get('eff_over_dim_raw') is not None and s['noise'] > 0.01 else "") + ", "
                                 f"stable rank {s['stable']:.1f}, rank 64 keeps {s['e64'] * 100:.0f}%, rank 256 keeps {s['e256'] * 100:.0f}%, "
                                 f"noise {s['noise'] * 100:.1f}%)")
            z = f.get("zones", {})
            if z:
                lines.append("zone profile of B's change: " + ", ".join(
                    f"{k} {v['share'] * 100:.0f}% of the energy on {v['param_share'] * 100:.0f}% of the parameters (x{v['conc']:.1f})"
                    for k, v in z.items()) + (f"; focus: {f['zone_focus']}" if f.get("zone_focus") else "; no zone focus"))
            lines.append(f"outside the linears (norms, modulation, biases): {f.get('non_target_share', 0.0) * 100:.2f}% of B's change")
            p = f.get("precision", {})
            lines.append(f"precision: A {p.get('A')}, B {p.get('B')}" + (f", C {p.get('C')}" if self.has_c else "")
                         + ("; mismatch: the cosine methods are excluded (decision 15)" if p.get("mismatch") else "; equal"))
            if f.get("nb_factor"):
                lines.append(f"non block weight factor {f['nb_factor']:.2f}: B's text side and projections moved {'more' if f['nb_factor'] < 1 else 'less'} "
                             "than its blocks relative to A's")
        if self.candidates:
            lines.append(f"candidates for the goal ({len(self.candidates)}):")
            for i, c in enumerate(self.candidates, 1):
                lines.append(f"  {i}. {c.label}")
                lines.append(f"     recipe: {c.summary()}")
                for why in c.rationale:
                    lines.append(f"     why: {why}")
                if c.expect:
                    lines.append(f"     expect: {c.expect}")
                if c.compare:
                    lines.append(f"     compare: {c.compare}")
                for fl in c.flags:
                    lines.append(f"     flag: {fl}")
        for n in self.notes:
            lines.append(n)
        lines.append(CALIBRATION)
        return "\n".join(lines)


def _f(v, spec: str) -> str:
    """A number in the given format, or a right aligned dash of the same width."""
    if isinstance(v, (int, float)):
        return format(v, spec)
    width = int(spec.split(".")[0].lstrip("0") or "1")
    return format("-", f">{width}s")


def _fmt_norm(n, rel) -> str:
    if n is None:
        return f"{'-':>13s}"
    return f"{n:7.2f} ({rel * 100:4.1f}%)"


def advice_stem(path: str) -> str:
    p = path
    if p.lower().endswith(".json"):
        p = p[:-5]
    if p.lower().endswith(".advice"):
        p = p[:-7]
    return p


# ----------------------------------------------------------------------------- measurement pass
def measure(a_path: str, b_path: str, c_path: str | None = None, depth: str = "full", use_gpu: bool = True,
            progress=None, cancel=None) -> AdvisorReport:
    """One pass over the common tensors of A, B and C. depth 'full' computes the spectra of both deltas,
    'quick' only the norms and inner products."""
    device = pick_device(use_gpu)
    A, B = _Ckpt(a_path), _Ckpt(b_path)
    C = _Ckpt(c_path) if c_path else None
    rep = AdvisorReport()
    full = depth == "full"
    repA = AnalysisReport() if (full and C is not None) else None
    repB = AnalysisReport() if full else None
    try:
        keys = [k for k in A.reader.names if not A.fmt.is_consumed(k) and (A.reader.dtype(k) in FLOAT_TAGS or A.fmt.is_quantized(k))]
        targets, others = [], []
        for k in keys:
            bk = B.key_for(k)
            ck = C.key_for(k) if C is not None else None
            if bk is None or (C is not None and ck is None):
                continue
            shape = A.reader.shape(k)
            if list(B.reader.shape(bk)) != list(shape) or (C is not None and list(C.reader.shape(ck)) != list(shape)):
                continue
            m, suf = ckpt_module(k)
            (targets if suf == ".weight" and len(shape) == 2 else others).append((k, bk, ck, m + ("" if suf == ".weight" else suf)))
        total = len(targets) + len(others)
        base_fmt = C.fmt if C is not None else A.fmt
        for i, (k, bk, ck, name) in enumerate(targets):
            if cancel is not None and cancel():
                raise Cancelled("cancelled by the user")
            if progress is not None:
                progress(i, total, name)
            a = A.fmt.read_fp32(k, device=device)
            b = B.fmt.read_fp32(bk, device=device)
            c = C.fmt.read_fp32(ck, device=device) if C is not None else None
            base = c if c is not None else a
            dB = b - base
            dA = (a - c) if c is not None else None
            eB = float((dB * dB).sum().item())
            eA = float((dA * dA).sum().item()) if dA is not None else None
            dot = float((dA * dB).sum().item()) if dA is not None else None
            wn = float((base * base).sum().item())
            cn = ckpt_canon(name)
            rep.relation[cn] = {"canon": cn, "name": name, "group": group_of(name), "block": block_index(name), "shape": list(a.shape),
                                "eA": eA, "eB": eB, "dot": dot, "wn": wn, "n": int(a.numel())}
            if full:
                layB = (base_fmt.layout_of(ck if C is not None else k), B.fmt.layout_of(bk))
                dtB = ((C.reader.dtype(ck) if C is not None else A.reader.dtype(k)), B.reader.dtype(bk))
                stB = delta_stats(layB, dtB, base, dB)
                sv, stB = spectrum_with_guard(dB, stB)
                repB.add(make_module_spectrum(cn, name, sv, tuple(dB.shape), stB))
                if dA is not None:
                    stA = delta_stats((C.fmt.layout_of(ck), A.fmt.layout_of(k)), (C.reader.dtype(ck), A.reader.dtype(k)), base, dA)
                    sv, stA = spectrum_with_guard(dA, stA)
                    repA.add(make_module_spectrum(cn, name, sv, tuple(dA.shape), stA))
            del a, b, c, dB, dA
            if device.type == "cuda" and shape and int(torch.tensor(shape).prod().item()) >= (1 << 24):
                torch.cuda.empty_cache()
        for j, (k, bk, ck, name) in enumerate(others):
            if cancel is not None and cancel():
                raise Cancelled("cancelled by the user")
            if progress is not None:
                progress(len(targets) + j, total, "other: " + name)
            try:
                a = A.fmt.read_fp32(k, device=device)
                b = B.fmt.read_fp32(bk, device=device)
                c = C.fmt.read_fp32(ck, device=device) if C is not None else None
            except FormatError:
                continue
            base = c if c is not None else a
            nB = float((b - base).norm().item())
            nA = float((a - c).norm().item()) if c is not None else None
            wn = float(base.norm().item())
            g = group_of(name)
            rep.others.append((name, g, block_index(name), nA, nB, wn))
            kind = "1-D" if a.dim() != 2 else "not a weight"
            if repB is not None:
                repB.others.append(OtherTensor(name, tuple(a.shape), g, kind, wn, nB))
            if repA is not None:
                repA.others.append(OtherTensor(name, tuple(a.shape), g, kind, wn, nA))
            del a, b, c
        if repB is not None:
            repB.finish()
            repB.run = {"function": "extract", "base": c_path or a_path, "target": b_path, "device": str(device)}
            repB.noise_model = _noise_text(base_fmt, B.fmt)
        if repA is not None:
            repA.finish()
            repA.run = {"function": "extract", "base": c_path, "target": a_path, "device": str(device)}
            repA.noise_model = _noise_text(C.fmt, A.fmt)
        rep.repA, rep.repB = repA, repB
        rep.run = {"function": "advise", "A": os.path.abspath(a_path), "B": os.path.abspath(b_path),
                   "C": os.path.abspath(c_path) if c_path else None,
                   "A_format": A.fmt.summary(), "B_format": B.fmt.summary(), "C_format": C.fmt.summary() if C else None,
                   "A_quantized": bool(A.fmt.layout_counts()), "B_quantized": bool(B.fmt.layout_counts()),
                   "C_quantized": bool(C.fmt.layout_counts()) if C else None,
                   "A_dtype": A.fmt.dominant_float(), "B_dtype": B.fmt.dominant_float(), "C_dtype": C.fmt.dominant_float() if C else None,
                   "depth": depth, "targets": len(targets), "others": len(others), "device": str(device),
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        return rep
    finally:
        A.close()
        B.close()
        if C is not None:
            C.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def ckpt_canon(name: str) -> str:
    from .keys import canon
    return canon(name)


def _noise_text(fb, ft) -> dict:
    from .extract import _describe_noise
    return _describe_noise(fb, ft)


# ----------------------------------------------------------------------------- features
def compute_features(rep: AdvisorReport) -> dict:
    """Energy weighted relation, structure, zone profile, precision. Fills rep.features and returns it."""
    has_c = rep.has_c
    groups: dict = {}
    zones: dict = {}
    tot = {"eA": 0.0, "eB": 0.0, "dot": 0.0, "wn": 0.0, "n": 0}
    for v in rep.relation.values():
        for key, store in ((v["group"], groups), (zone_of(v["block"]), zones)):
            a = store.setdefault(key, {"eA": 0.0, "eB": 0.0, "dot": 0.0, "wn": 0.0, "n": 0})
            a["eB"] += v["eB"]; a["wn"] += v["wn"]; a["n"] += v["n"]
            if has_c:
                a["eA"] += v["eA"]; a["dot"] += v["dot"]
        tot["eB"] += v["eB"]; tot["wn"] += v["wn"]; tot["n"] += v["n"]
        if has_c:
            tot["eA"] += v["eA"]; tot["dot"] += v["dot"]

    def derive(a: dict) -> dict:
        d = {"nB": math.sqrt(a["eB"]), "relB": math.sqrt(a["eB"] / a["wn"]) if a["wn"] > 0 else 0.0, "n": a["n"]}
        if has_c:
            d["nA"] = math.sqrt(a["eA"])
            d["relA"] = math.sqrt(a["eA"] / a["wn"]) if a["wn"] > 0 else 0.0
            d["rho"] = math.sqrt(a["eB"] / a["eA"]) if a["eA"] > 0 else None
            d["cos"] = a["dot"] / math.sqrt(a["eA"] * a["eB"]) if a["eA"] > 0 and a["eB"] > 0 else 0.0
            d["c_a_on_b"] = a["dot"] / a["eB"] if a["eB"] > 0 else 0.0
            d["c_b_on_a"] = a["dot"] / a["eA"] if a["eA"] > 0 else 0.0
        return d
    f: dict = {"groups": {g: derive(groups[g]) for g in GROUPS if g in groups}, "overall": derive(tot)}
    zsum = {"eB": 0.0, "n": 0}
    for z in ZONES:
        if z in zones:
            zsum["eB"] += zones[z]["eB"]; zsum["n"] += zones[z]["n"]
    f["zones"] = {}
    focus, best = None, 0.0
    for z in ZONES:
        if z not in zones:
            continue
        share = zones[z]["eB"] / zsum["eB"] if zsum["eB"] > 0 else 0.0
        pshare = zones[z]["n"] / zsum["n"] if zsum["n"] > 0 else 0.0
        conc = share / pshare if pshare > 0 else 0.0
        f["zones"][z] = {"share": share, "param_share": pshare, "conc": conc, **derive(zones[z])}
        if z != "non_block" and conc >= ZONE_FOCUS and conc > best:
            focus, best = z, conc
    f["zone_focus"] = focus
    # relation reading
    o = f["overall"]
    if has_c:
        cos = o["cos"]
        f["relation"] = "independent" if cos < COS_INDEPENDENT else ("aligned" if cos > COS_ALIGNED else "related")
        f["containment"] = None
        if cos >= CONTAIN_MIN_COS:
            if CONTAIN_RANGE[0] <= o["c_a_on_b"] <= CONTAIN_RANGE[1] and CONTAIN_RANGE[0] <= o["c_b_on_a"] <= CONTAIN_RANGE[1]:
                f["containment"] = "equal"
            elif CONTAIN_RANGE[0] <= o["c_a_on_b"] <= CONTAIN_RANGE[1]:
                f["containment"] = "A_contains_B"
            elif CONTAIN_RANGE[0] <= o["c_b_on_a"] <= CONTAIN_RANGE[1]:
                f["containment"] = "B_contains_A"
        f["containment_text"] = {None: "", "equal": "the two changes are the same change",
                                 "A_contains_B": "A's change already carries B's change in full (coefficient near 1) plus more",
                                 "B_contains_A": "B's change carries A's change in full plus an extra part"}[f["containment"]]
        f["w_eq"] = (1.0 / o["rho"]) if o.get("rho") else None
        # residual of B beyond A when B contains A: |dB - dA|
        if f["containment"] == "B_contains_A":
            e_r = max(tot["eB"] - 2 * tot["dot"] + tot["eA"], 0.0)
            f["w_eq_extra"] = math.sqrt(tot["eA"] / e_r) if e_r > 0 else None
        # non block factor: equal contribution weights of the blocks against the rest
        zb = [zones[z] for z in ("composition", "character", "style") if z in zones]
        if zb and "non_block" in zones:
            eA_b = sum(z["eA"] for z in zb); eB_b = sum(z["eB"] for z in zb)
            w_b = math.sqrt(eA_b / eB_b) if eA_b > 0 and eB_b > 0 else None
            nb = zones["non_block"]
            w_nb = math.sqrt(nb["eA"] / nb["eB"]) if nb["eA"] > 0 and nb["eB"] > 0 else None
            if w_b and w_nb:
                ratio = w_nb / w_b
                f["nb_factor"] = round(max(0.25, min(4.0, ratio)), 2) if (ratio > NB_FACTOR_TRIGGER or ratio < 1.0 / NB_FACTOR_TRIGGER) else None
    else:
        f["relation"] = "unknown (no C)"
        f["containment"] = None
        f["containment_text"] = ""
        f["w_eq"] = None
    # structure
    f["structure"] = {}
    for key, r in (("dA", rep.repA), ("dB", rep.repB)):
        if r is None or not r.modules:
            continue
        mods = [m for m in r.modules.values() if not m.all_zero]
        # the class comes from the directions above the noise edge: a quantized input adds a full rank noise
        # tail that would read as diffuse structure (the int8 Unaligned case, 2026-09-15)
        eff = _median([m.effective_rank_denoised() / m.max_rank for m in mods]) if mods else None
        tab = {row[0]: row for row in r.candidate_table()}
        e64 = tab.get(64, (None, None, None, None))[3] or 0.0
        e256 = tab.get(256, (None, None, None, None))[3] or 0.0
        noise = 0.0
        tot_e = sum(m.energy for m in mods)
        if tot_e > 0:
            noise = sum((1.0 - m.energy_above_noise) * m.energy for m in mods) / tot_e
        f["structure"][key] = {"eff_over_dim": eff, "stable": _median([m.stable_rank_denoised() for m in mods]) or 0.0,
                               "energy_at_rank": {str(rk): (row[3] or 0.0) for rk, row in tab.items()},
                               "eff_over_dim_raw": _median([m.effective_rank / m.max_rank for m in mods]) if mods else None,
                               "e64": e64, "e256": e256, "noise": noise, "class": _cls_structure(eff)}
    f["non_target_share"] = rep.repB.outside_fraction if rep.repB is not None else _others_share(rep)
    r = rep.run
    f["precision"] = {"A": ("quantized " if r.get("A_quantized") else "plain ") + str(r.get("A_dtype")),
                      "B": ("quantized " if r.get("B_quantized") else "plain ") + str(r.get("B_dtype")),
                      "C": (("quantized " if r.get("C_quantized") else "plain ") + str(r.get("C_dtype"))) if has_c else None,
                      "mismatch": bool(r.get("A_quantized")) != bool(r.get("B_quantized"))}
    rep.features = f
    return f


def _others_share(rep: AdvisorReport) -> float:
    e_o = sum(o[4] ** 2 for o in rep.others)
    e_t = sum(v["eB"] for v in rep.relation.values())
    return e_o / (e_o + e_t) if (e_o + e_t) > 0 else 0.0


# ----------------------------------------------------------------------------- recipes
def _ckpt_step(A, B, C, weight, method="add_difference", shaping: Shaping | None = None, params: dict | None = None,
               loras: list | None = None, output_format="keep") -> dict:
    opts = CkptMergeOptions(method=method, output_format=output_format)
    if params:
        opts.params.update(params)
    return {"function": "ckpt_merge", "A": {"file": A, "weight": 1.0, "shaping": Shaping().to_dict()},
            "B": {"file": B, "weight": float(weight), "shaping": (shaping or Shaping()).to_dict()} if B else None,
            "C": {"file": C, "weight": 1.0, "shaping": Shaping().to_dict()} if C else None,
            "loras": list(loras or []), "options": opts.to_dict(), "output": None}


def _extract_step(base, target, rank, output_format=None) -> dict:
    return {"function": "extract", "base": base, "target": target, "options": ExtractOptions(rank=int(rank), method="randomized").to_dict(), "output": None}


def _sweep(w_eq: float | None, factors=W_SWEEP, default=(0.2, 0.35, 0.5)) -> list:
    """Three distinct weights around the equal contribution weight, clipped to the allowed range; when the clip
    merges values the sweep is filled downward so three points remain."""
    if not w_eq:
        return list(default)
    out = []
    for k in factors:
        w = round_step(w_eq * k)
        if w not in out:
            out.append(w)
    while len(out) < len(factors) and min(out) > W_MIN:
        w = round_step(min(out) * 0.5)
        if w in out:
            break
        out.append(w)
    return sorted(out)


def _shaped(preset: str, modifier: str = "Emphasize", contrast: float = 0.5, non_block: float | None = None) -> Shaping:
    return Shaping(preset=preset, modifier=modifier, contrast=contrast, non_block=non_block)


# ----------------------------------------------------------------------------- rules
def propose(rep: AdvisorReport, goal: str, output_format: str = "keep") -> list:
    """Candidates for a goal from the features. Fills rep.candidates and returns them."""
    if goal not in GOALS:
        raise ValueError(f"unknown goal {goal!r}; one of {', '.join(GOALS)}")
    f = rep.features or compute_features(rep)
    rep.run["goal"] = goal
    A, B, C = rep.run["A"], rep.run["B"], rep.run.get("C")
    has_c = C is not None
    sB = f["structure"].get("dB")
    diffuse = bool(sB and sB["class"] == "diffuse")
    concentrated = bool(sB and sB["class"] == "concentrated")
    mismatch = f["precision"]["mismatch"]
    nb = f.get("nb_factor")
    rel, cont = f["relation"], f.get("containment")
    w_eq = f.get("w_eq")
    cands: list = []
    common_flags = []
    if diffuse:
        common_flags.append("B's change is diffuse (memorization shaped): expect seed collapse at high doses; weights halved, watch identity variety")
    if f.get("non_target_share", 0.0) > NON_TARGET_WARN:
        common_flags.append(f"{f['non_target_share'] * 100:.1f}% of B's change sits in norms, modulation and biases: a checkpoint merge carries it, a LoRA cannot")
    if not has_c:
        common_flags.append("no C: weights are a default sweep, not scaled to A's own change")
    scale = 0.5 if diffuse else 1.0
    expect_keep = "A's subjects, settings and rendering on every seed, with B's change appearing on the same seeds in proportion to the weight"
    compare = "fixed seed grids of A alone and of each weight; if faces or settings converge on one type the weight is too high"

    def add(label, steps, why, expect=expect_keep, cmp=compare, flags=()):
        cands.append(Candidate(label, goal, steps, list(why), expect, cmp, list(common_flags) + list(flags)))

    if goal in ("add_content", "composition", "style"):
        shaping_for = {"composition": ("COMPOSITION", "STYLE"), "style": ("STYLE", "COMPOSITION")}.get(goal)
        if has_c and cont == "A_contains_B":
            add("B's change is already in A: top up only", [_ckpt_step(A, B, C, round_step(0.25 * (w_eq or 0.5)), output_format=output_format)],
                [f"A's change carries B's change with coefficient {f['overall']['c_a_on_b']:.2f}; adding B again mostly doubles what A has"],
                "little visible change; a larger dose would over apply B's direction", flags=["consider goal 'blend' or no merge"])
        elif has_c and cont == "B_contains_A":
            w2 = _sweep(f.get("w_eq_extra"), default=(0.15, 0.3, 0.5))
            if diffuse:
                w2 = [round_step(w * 0.5) for w in w2]
            for w in w2:
                add(f"add B minus A at {w:g} (B's extra part only)", [_ckpt_step(A, B, None, w, shaping=_shaped(shaping_for[0]) if shaping_for else None, output_format=output_format)],
                    [f"B's change carries A's change in full (coefficient {f['overall']['c_b_on_a']:.2f}) plus an extra part; B minus A isolates that part",
                     "the weights are scaled to the size of the extra part against A's own change"])
        else:
            method = "add_difference"
            ws = [round_step(w * scale) for w in _sweep(w_eq)] if has_c else _sweep(None)
            ws = sorted(set(ws))
            why_rel = {"independent": "the changes are independent (cosine near 0): adding B's change does not disturb A's, and the interference methods have nothing to resolve",
                       "related": "the changes share some direction; add difference keeps A's part and adds B's",
                       "aligned": "the changes are aligned (siblings); add difference adds B's on top of A's, the cosine and TIES candidates below blend instead",
                       "unknown (no C)": "without C the tool cannot separate A's change from B's; A + w (B - A) moves toward B"}[rel]
            if has_c and rel == "aligned":
                if mismatch:
                    add(f"TIES at {round_step((w_eq or 1.0) * scale):g}", [_ckpt_step(A, B, C, round_step((w_eq or 1.0) * scale), "ties", params={"density": 0.2, "lambda": 1.0}, output_format=output_format)],
                        ["aligned changes with mismatched storage precision: TIES trims and elects signs, the cosine methods would mix on quantization noise (decision 15)"])
                else:
                    for w in (0.5, round_step((w_eq or 1.0) * scale)):
                        add(f"cosine B at {w:g}", [_ckpt_step(A, B, C, w, "cosine_b", output_format=output_format)],
                            ["aligned changes, equal precision: cosine B keeps A where A and B agree and takes B where they differ"])
            for w in ws:
                sh = _shaped(shaping_for[0], non_block=nb) if shaping_for else (Shaping(non_block=nb) if nb else None)
                lab = f"add difference at {w:g}" + (f", {shaping_for[0]} emphasized" if shaping_for else "")
                why = [why_rel, f"weights: {'the equal contribution weight ' + format(w_eq, '.2f') + ' halved and doubled' if w_eq else 'a default sweep'}"]
                if shaping_for:
                    why.append(f"shaping: B's change on the {shaping_for[0].lower()} blocks emphasized at contrast 0.5"
                               + (f" (B's zone focus: {f['zone_focus']})" if f.get("zone_focus") else ""))
                if nb:
                    why.append(f"non block factor {nb:.2f} keeps B's text side and projections at their own equal contribution weight")
                add(lab, [_ckpt_step(A, B, C if has_c else None, w, method, shaping=sh, output_format=output_format)], why)
            if shaping_for:
                w = ws[len(ws) // 2]
                add(f"control: add difference at {w:g}, {shaping_for[1]} emphasized", [_ckpt_step(A, B, C if has_c else None, w, method, shaping=_shaped(shaping_for[1], non_block=nb), output_format=output_format)],
                    ["the opposite zone, to check that the effect follows the shaping and not the weight alone"],
                    "the opposite trade: if this one shows the wanted variety too, the zones are not where the change lives")
                add(f"isolate: add difference at {w:g}, {shaping_for[0]} only", [_ckpt_step(A, B, C if has_c else None, w, method, shaping=_shaped(shaping_for[0], "Isolate", 1.0, non_block=nb), output_format=output_format)],
                    ["B's change applied inside the zone only; the strongest form of the shaping"])
            if goal == "add_content" and (diffuse or f.get("zone_focus")):
                w = ws[len(ws) // 2]
                preset = "CHARACTER" if diffuse else {"composition": "COMPOSITION", "style": "STYLE", "character": "CHARACTER"}[f["zone_focus"]]
                mod = "Suppress" if diffuse else "Emphasize"
                add(f"shaped: add difference at {w:g}, {preset} {'suppressed' if mod == 'Suppress' else 'emphasized'}", [_ckpt_step(A, B, C if has_c else None, w, method, shaping=_shaped(preset, mod, 0.5, non_block=nb), output_format=output_format)],
                    ["a diffuse change collapses identities first; suppressing the character blocks keeps B's other effects" if diffuse
                     else f"B's change concentrates on the {f['zone_focus']} blocks (x{f['zones'][f['zone_focus']]['conc']:.1f}); emphasizing them spends the dose where the change is"])
    elif goal == "blend":
        if not has_c:
            for w in (0.35, 0.5):
                add(f"weighted sum at {w:g}", [_ckpt_step(A, B, None, w, "weighted_sum", output_format=output_format)],
                    ["without C an even blend is the plain average of the two checkpoints"], "a midpoint between A and B", compare)
        elif rel == "aligned":
            if mismatch:
                add("TIES at 1.0", [_ckpt_step(A, B, C, 1.0, "ties", params={"density": 0.2, "lambda": 1.0}, output_format=output_format)],
                    ["aligned siblings with mismatched precision: TIES merges the two changes by elected sign"])
            else:
                for m in ("cosine_a", "cosine_b"):
                    add(f"{m.replace('_', ' ')} at 0.5", [_ckpt_step(A, B, C, 0.5, m, output_format=output_format)],
                        ["aligned siblings, equal precision: the cosine methods keep the shared part and blend the rest"], "a model between A and B that keeps what they agree on")
            add("add difference at 1.0 (both changes in full)", [_ckpt_step(A, B, C, 1.0, output_format=output_format)],
                ["C plus both changes; with aligned changes this over applies the shared direction, kept as the reference point"])
        else:
            add("add difference at 1.0 (both changes in full)", [_ckpt_step(A, B, C, 1.0, output_format=output_format)],
                ["independent changes do not interfere: C plus A's change plus B's change is the even blend"], "both fine tunes' effects at full strength")
            add("weighted sum at 0.5", [_ckpt_step(A, B, None, 0.5, "weighted_sum", output_format=output_format)],
                ["the plain average halves both changes; B dominates the average when its change is larger" if (f["overall"].get("rho") or 1) > 1 else "the plain average halves both changes"])
            if w_eq:
                add(f"add difference at {round_step(w_eq):g} (equal contribution)", [_ckpt_step(A, B, C, round_step(w_eq), output_format=output_format)],
                    ["B's change scaled to the size of A's change"])
    elif goal == "deturbo":
        why = ["de-Turbo is add difference with A = the fine tune, B = official Raw, C = official Turbo: A minus w (Turbo - Raw)",
               "0.4 of the distillation matched the rank 64 Turbo LoRA at 0.5 seed for seed; 0.6 keeps few step sampling; 1.0 gives a Raw family model (about 28 to 52 steps, CFG about 3.5)"]
        for w in DETURBO_W:
            add(f"remove {w:g} of the distillation", [_ckpt_step(A, B, C, w, output_format=output_format)], why,
                "the fine tune's look with the distillation dialed back; more steps needed as w grows",
                "fixed seed grids at the step count each dose needs; check coherence at 8 steps for 0.4 and 0.6")
        add(f"attenuate the fine tune's own change to {DETURBO_X:g}, then remove 0.6 of the distillation",
            [_ckpt_step(C, A, None, DETURBO_X, output_format="bf16"), _ckpt_step("@prev", B, C, 0.6, output_format=output_format)],
            ["seed collapse was found in the fine tune's residual, not only in the distillation (2026-09-14); a second dial on A's own change",
             "step 1 is C + x (A - C), step 2 adds w (B - C); together C + x (A - C) + w (B - C)"],
            "more seed variety at the cost of some of the fine tune's identity", "against candidate 2 at the same seeds",
            flags=["two steps: the intermediate is written in bf16 and removed after the second step"])
        if has_c and f["structure"].get("dB") and f["structure"]["dB"]["class"] == "diffuse":
            cands[-1].flags.append("B minus C does not look like a distillation (diffuse); check that B is Raw and C is Turbo")
    elif goal == "lora":
        base = C if has_c else A
        if not sB:
            add("no spectra (quick depth)", [], ["run the advisor at full depth to size the rank"], "", "")
        elif diffuse or sB["e256"] < 0.5:
            add("no LoRA: B's change is diffuse", [_ckpt_step(A, B, C if has_c else None, round_step((w_eq or 0.5) * 0.5), output_format=output_format)],
                [f"rank 256 keeps {sB['e256'] * 100:.0f}% of the change's energy and the effective rank is {sB['eff_over_dim'] * 100:.0f}% of the dimension: no rank below the weights reproduces it",
                 "a checkpoint merge at a low weight carries the change instead"], expect_keep, compare)
        else:
            energy = sB.get("energy_at_rank", {})
            rank = next((r for r in CANDIDATE_RANKS if energy.get(str(r), 0.0) >= LORA_ENERGY), CANDIDATE_RANKS[-1])
            e = energy.get(str(rank), 0.0)
            add(f"extract at rank {rank}", [_extract_step(base, B, rank)],
                [f"rank {rank} is the smallest candidate rank whose median denoised energy reaches {LORA_ENERGY * 100:.0f}% ({e * 100:.0f}%)",
                 f"B's change is {sB['class']} (effective rank {sB['eff_over_dim'] * 100:.1f}% of the dimension)"],
                "a LoRA that applies B's change at strength 1 with the listed energy", "the LoRA at 1.0 on A against the add difference merge at 1.0")
            w = round_step((w_eq or 0.5) * scale) if has_c else 0.35
            add(f"extract at rank {rank}, bake into A at {w:g}",
                [_extract_step(base, B, rank), _ckpt_step(A, None, None, 1.0, loras=[{"file": "@prev", "strength": w, "shaping": Shaping().to_dict()}], output_format=output_format)],
                ["the low rank filter of B's change applied to A; compare with the full change at the same weight to see whether the tail matters"],
                expect_keep, "against the add difference candidate of goal 'add content' at the same weight")
    rep.candidates = cands
    return cands


def advise(a_path: str, b_path: str, c_path: str | None = None, goal: str = "add_content", depth: str = "full",
           output_format: str = "keep", use_gpu: bool = True, progress=None, cancel=None) -> AdvisorReport:
    rep = measure(a_path, b_path, c_path, depth, use_gpu, progress, cancel)
    compute_features(rep)
    propose(rep, goal, output_format)
    return rep


# ----------------------------------------------------------------------------- running candidates
def candidate_slug(label: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9.]+", "_", label).strip("_")
    return s[:60]


def run_candidate(rep: AdvisorReport, cand: Candidate, out_dir: str, use_gpu: bool = True, progress=None, cancel=None,
                  log=None, keep_intermediate: bool = False) -> list:
    """Runs the steps of a candidate into out_dir. Returns the output paths (the final one last)."""
    from .recipe import run_recipe
    log = log or (lambda s: None)
    stem = os.path.splitext(os.path.basename(rep.run["A"]))[0]
    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    prev = None
    n = len(cand.steps)
    for i, step in enumerate(cand.steps):
        r = json.loads(json.dumps(step))
        for slot in ("A", "B", "C"):
            if r.get(slot) and r[slot].get("file") == "@prev":
                r[slot]["file"] = prev
        for l in r.get("loras", []):
            if l.get("file") == "@prev":
                l["file"] = prev
        if r.get("base") == "@prev":
            r["base"] = prev
        if r.get("target") == "@prev":
            r["target"] = prev
        last = i == n - 1
        suffix = ".safetensors" if r["function"] != "extract" else ".lora.safetensors"
        name = f"{stem}__{candidate_slug(cand.label)}" + ("" if last else f"__step{i + 1}") + suffix
        out = os.path.join(out_dir, name)
        log(f"candidate '{cand.label}' step {i + 1} of {n}: {r['function']} -> {name}")
        run_recipe(r, out, use_gpu=use_gpu, progress=progress, cancel=cancel, log=log)
        outputs.append(out)
        if prev is not None and not keep_intermediate and not prev.endswith(".lora.safetensors"):
            try:
                os.remove(prev)
                outputs.remove(prev)
                log(f"removed the intermediate {os.path.basename(prev)}")
            except OSError:
                pass
        prev = out
    return outputs
