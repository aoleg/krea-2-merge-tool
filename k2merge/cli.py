"""Command line interface. `python krea2_merge_tool.py --help`."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from . import __version__
from .blocks import BLOCK_PRESETS, MODIFIERS, Shaping
from .formats import INT8_CLIPS, OUTPUT_FORMATS, PASSTHROUGH
from .methods import METHODS, METHOD_LABELS
from .advisor import GOAL_ORDER
from .ckpt_merge import VECTOR_SOURCES
from .meta import MODELSPEC_FIELDS


def _shaping_arg(text: str | None) -> Shaping:
    """PRESET[:MODIFIER[:CONTRAST[:BOOST]]][@NONBLOCK], e.g. STYLE:Suppress:0.5 or FULL."""
    if not text:
        return Shaping()
    nb = None
    if "@" in text:
        text, nb_s = text.split("@", 1)
        nb = float(nb_s)
    parts = text.split(":")
    s = Shaping(preset=parts[0].upper())
    if len(parts) > 1:
        s.modifier = parts[1].capitalize()
    if len(parts) > 2:
        s.contrast = float(parts[2])
    if len(parts) > 3:
        s.boost = float(parts[3])
    s.non_block = nb
    if s.preset not in BLOCK_PRESETS or s.modifier not in MODIFIERS:
        raise argparse.ArgumentTypeError(f"bad shaping {text!r}")
    return s


def _lora_arg(text: str):
    """FILE[:STRENGTH[:SHAPING]] where SHAPING is PRESET:MODIFIER:CONTRAST[:BOOST][@NONBLOCK]."""
    from .lora_merge import LoraInput
    head, sep, tail = text.partition("|")
    # allow drive letters: FILE|STRENGTH|SHAPING
    parts = text.split("|")
    path = parts[0]
    strength = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
    shaping = _shaping_arg(parts[2]) if len(parts) > 2 else Shaping()
    return LoraInput(path, strength, shaping)


def _progress_printer():
    last = {"n": -1}

    def p(cur, total, name):
        if cur != last["n"]:
            last["n"] = cur
            sys.stdout.write(f"\r[{cur}/{total}] {str(name)[:60]:60s}")
            sys.stdout.flush()
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="krea2_merge_tool", description="Krea 2 merge tool: LoRA merge, analysis, extraction, checkpoint merge and conversion")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--gui", action="store_true", help="start the GUI (default when no command is given)")
    ap.add_argument("--theme", default=None, choices=["light", "dark"], help="GUI theme (default: the last one used)")
    ap.add_argument("--scale", default=None, choices=["auto", "100", "125", "150", "175", "200"],
                    help="GUI scale in percent; auto follows Windows text size (default: the last one used)")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run a recipe JSON file")
    r.add_argument("recipe")
    r.add_argument("-o", "--output", help="override the recipe's output path")
    r.add_argument("--plan", action="store_true", help="describe the run, write nothing")

    i = sub.add_parser("inspect", help="summarize safetensors files")
    i.add_argument("files", nargs="+")

    mt = sub.add_parser("meta", help="show, redact, strip, set or restore the metadata of a file")
    msub = mt.add_subparsers(dest="meta_cmd", required=True)
    m_show = msub.add_parser("show", help="the metadata, and what a publisher should know about it")
    m_show.add_argument("files", nargs="+", help="files or folders")
    m_show.add_argument("--key", default=None, help="print this one key's value in full")
    m_show.add_argument("--lineage", action="store_true", help="also trace what the file was made from")
    m_red = msub.add_parser("redact", help="paths out of the metadata, in place by default")
    m_red.add_argument("files", nargs="+", help="files or folders")
    m_red.add_argument("--policy", default="basename", choices=("basename", "placeholder", "drop"),
                       help="what a path becomes: its file name (default), a placeholder, or the whole key removed")
    m_red.add_argument("--ss", action="store_true", help="also remove the trainer's ss_* and ot_* metadata (dataset names, tag frequencies, OneTrainer config)")
    m_red.add_argument("--thumbnail", action="store_true", help="also remove an embedded modelspec thumbnail")
    m_red.add_argument("--workflow", action="store_true", help="also remove an embedded ComfyUI workflow and prompt")
    m_red.add_argument("-o", "--out", default=None, help="write copies into this folder instead of patching in place")
    m_red.add_argument("--no-backup", action="store_true", help="do not save the original header next to the file")
    m_red.add_argument("--dry-run", action="store_true", help="list what would change and write nothing")
    m_str = msub.add_parser("strip", help="remove all metadata except what the file needs in order to load")
    m_str.add_argument("file")
    m_str.add_argument("-o", "--out", required=True, help="the new file; stripping never writes in place")
    m_set = msub.add_parser("set", help="set the modelspec fields")
    m_set.add_argument("file")
    for _f, _label, _req, _hint in MODELSPEC_FIELDS:
        m_set.add_argument("--" + _f.replace("_", "-"), default=None, help=_hint or _label)
    m_set.add_argument("--compute-hash", action="store_true", help="fill hash_sha256 from the tensor data")
    m_set.add_argument("-o", "--out", default=None, help="write a new file instead of patching in place")
    m_diff = msub.add_parser("diff", help="the metadata differences between two files")
    m_diff.add_argument("a")
    m_diff.add_argument("b")
    m_res = msub.add_parser("restore", help="put back the header saved before the last in place change")
    m_res.add_argument("files", nargs="+")

    c = sub.add_parser("convert", help="convert a checkpoint between storage formats")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--format", default="bf16", choices=OUTPUT_FORMATS)
    c.add_argument("--passthrough", default="official", choices=PASSTHROUGH)
    c.add_argument("--fp8-layers", default="official", choices=["official", "blocks"])
    c.add_argument("--int8-clip", default="mse", choices=INT8_CLIPS, help="int8 scale: mse reproduces the official file, absmax is plain")

    lm = sub.add_parser("lora-merge", help="merge LoRA / LoKr files into one LoRA")
    lm.add_argument("inputs", nargs="+", help="FILE|STRENGTH|SHAPING (SHAPING = PRESET:MODIFIER:CONTRAST[:BOOST][@NONBLOCK])")
    lm.add_argument("-o", "--output", required=True)
    lm.add_argument("--average", action="store_true")
    lm.add_argument("--rank", type=int, default=None, help="SVD truncate to this rank")
    lm.add_argument("--dynamic", action="store_true", help="prune: every module keeps the smallest rank that holds --retention of its energy, within --rank-floor and --rank-cap")
    lm.add_argument("--retention", type=float, default=0.99, help="with --dynamic: energy kept per module, 0.5 to 1.0 (default 0.99)")
    lm.add_argument("--rank-cap", type=int, default=16, help="with --dynamic: no module above this rank; 0 = no cap (default 16)")
    lm.add_argument("--rank-floor", type=int, default=1, help="with --dynamic: no module below this rank (default 1)")
    lm.add_argument("--union", action="store_true", help="keep modules present in any input")
    lm.add_argument("--naming", default="comfy", choices=["comfy", "kohya", "input"])
    lm.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    lm.add_argument("--analyze", action="store_true", help="print the rank / energy report and exit")
    _analysis_args(lm)
    lm.add_argument("--plan", action="store_true")

    ex = sub.add_parser("extract", help="extract a LoRA from two checkpoints")
    ex.add_argument("base")
    ex.add_argument("target")
    ex.add_argument("-o", "--output", required=True)
    ex.add_argument("--rank", type=int, default=32)
    ex.add_argument("--filter", default="all", choices=["all", "attn", "blocks", "custom"])
    ex.add_argument("--include", default="")
    ex.add_argument("--exclude", default="")
    ex.add_argument("--method", default="randomized", choices=["randomized", "full"])
    ex.add_argument("--naming", default="comfy", choices=["comfy", "kohya"])
    ex.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ex.add_argument("--analyze", action="store_true", help="print the rank / energy report and exit")
    _analysis_args(ex)
    ex.add_argument("--null-spectrum", action="store_true", help="with --analyze: also store the spectrum of the modeled noise")
    ex.add_argument("--plan", action="store_true")

    cm = sub.add_parser("ckpt-merge", help="merge 1-3 checkpoints and 0-4 LoRAs, or convert")
    cm.add_argument("-A", required=True, help="primary checkpoint")
    cm.add_argument("-B", help="secondary checkpoint: FILE|WEIGHT|SHAPING")
    cm.add_argument("-C", help="reference checkpoint")
    cm.add_argument("--lora", action="append", default=[], help="FILE|STRENGTH|SHAPING (repeatable, up to 4)")
    cm.add_argument("-o", "--output", required=True)
    cm.add_argument("--method", default="add_difference", choices=METHODS)
    cm.add_argument("--param", action="append", default=[], help="method parameter NAME=VALUE (density, lambda, p, seed, beta, gamma, dare_ties)")
    cm.add_argument("--lora-mode", default="after", choices=["after", "task_vectors"])
    cm.add_argument("--format", default="bf16", choices=OUTPUT_FORMATS)
    cm.add_argument("--passthrough", default="official", choices=PASSTHROUGH)
    cm.add_argument("--fp8-layers", default="official", choices=["official", "blocks"])
    cm.add_argument("--int8-clip", default="mse", choices=INT8_CLIPS)
    cm.add_argument("--vectors-from", default="merge", choices=VECTOR_SOURCES,
                    help="norm scales, modulation vectors and biases: merged like the rest (merge), or copied from A, B or C")
    cm.add_argument("--as-lora", type=int, default=None, metavar="RANK", help="write the result as a LoRA of this rank")
    cm.add_argument("--report", action="store_true", help="print the pre merge report and exit")
    cm.add_argument("--plan", action="store_true")

    m = sub.add_parser("methods", help="list the checkpoint merge methods and what the weight means")

    ad = sub.add_parser("advise", help="measure how two checkpoints relate and propose merge recipes for a goal")
    ad.add_argument("-A", required=True, help="the checkpoint to keep")
    ad.add_argument("-B", required=True, help="the donor")
    ad.add_argument("-C", default=None, help="the common ancestor (optional)")
    ad.add_argument("--goal", default="add_content", choices=list(GOAL_ORDER))
    ad.add_argument("--depth", default="full", choices=["full", "quick"])
    ad.add_argument("--format", default="keep", choices=OUTPUT_FORMATS, help="output format of the candidate merges")
    ad.add_argument("--save", default=None, metavar="STEM", help="save STEM.advice.json and the two spectrum pairs")
    ad.add_argument("--run", default=None, metavar="DIR", help="run the candidates into DIR")
    ad.add_argument("--pick", type=int, action="append", default=[], help="with --run: only these candidate numbers (repeatable)")
    ad.add_argument("--keep-intermediate", action="store_true", help="keep the bf16 intermediate of two step candidates")

    sp = sub.add_parser("spectrum", help="show a saved analysis (.spectrum.json), or compare two")
    sp.add_argument("analysis")
    sp.add_argument("other", nargs="?")
    sp.add_argument("--tensor", default=None, help="per tensor detail for this module name")
    return ap


def _analysis_args(p):
    p.add_argument("--spectrum", default=None, metavar="STEM", help="with --analyze: save STEM.spectrum.npz and STEM.spectrum.json")
    p.add_argument("--energy", type=float, default=0.99, help="energy target of the printed rank plan (default 0.99)")
    p.add_argument("--criterion", default="weighted", choices=["weighted", "per_module"])
    p.add_argument("--raw-plan", action="store_true", help="rank plan from the raw spectrum instead of the denoised one")


def _finish_analysis(rep, args) -> int:
    print(rep.text())
    denoised = not args.raw_plan and rep.has_noise_model
    plan = rep.rank_plan(args.energy, args.criterion, denoised=denoised)
    print(f"\nrank per group for {args.energy:.3f} energy ({args.criterion}, {'denoised' if denoised else 'raw'} spectrum): {plan}")
    if args.spectrum:
        npz, js = rep.save(args.spectrum)
        print(f"saved {js} and {npz}")
    return 0


def _spectrum_command(args) -> int:
    from .analysis import AnalysisReport
    rep = AnalysisReport.load(args.analysis)
    if args.other:
        other = AnalysisReport.load(args.other)
        print(rep.compare_text(other))
        return 0
    print(f"analysis: {rep.label}")
    print(rep.text())
    if args.tensor:
        ms = next((m for m in rep.modules.values() if m.name == args.tensor), None)
        if ms is None:
            print(f"no tensor named {args.tensor}")
            return 1
        d = ms.to_dict()
        for k in ("name", "group", "block", "shape", "layouts", "dtypes", "delta_fro", "base_fro", "rel_change", "zero_fraction",
                  "noise_edge", "n_above_noise", "energy_above_noise", "effective_rank", "stable_rank", "rank_at_energy",
                  "rank_at_energy_denoised", "resolved_by_svd"):
            print(f"  {k:24s} {d[k]}")
    return 0


def _expand_files(paths) -> list[str]:
    """Folders stand for every safetensors file in them, so a scrub before an upload is one command."""
    import glob
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "*.safetensors"))) if os.path.isdir(p) else [p]
    return [p for p in out if not p.endswith(".header.bak")]


def _meta_command(args, prog, log) -> int:
    from .inspect_file import inspect_path
    from .meta import (MODELSPEC_FIELDS, HeaderTooLong, apply_findings, data_sha256, modelspec_defaults,
                       diff_metadata, lineage_text, metadata_rows, pretty, read_metadata, read_modelspec,
                       restore_header, scan_metadata, set_modelspec, strip_all, write_metadata)

    def put(path, meta, out):
        """Write, and say plainly when the header has no room for a metadata that grew."""
        try:
            return write_metadata(path, meta, out, backup=not getattr(args, "no_backup", False), progress=prog, log=log)
        except HeaderTooLong as e:
            print(f"\n{e}.\nThe data would have to move, so write a new file: add -o NEWFILE.")
            raise SystemExit(1) from None

    cmd = args.meta_cmd
    if cmd == "show":
        for p in _expand_files(args.files):
            print(inspect_path(p)["text"])
            meta = read_metadata(p)
            if args.key:
                print(pretty(meta.get(args.key, "")) if args.key in meta else f"no key {args.key!r}")
            elif not meta:
                print("  no metadata")
            else:
                for k, n, prev in metadata_rows(meta):
                    print(f"  {k}  ({n} chars)  {prev}")
            found = scan_metadata(meta, "basename", training=True, thumbnail=True, workflow=True)
            for kind, title in (("path", "paths"), ("training", "training metadata"), ("thumbnail", "thumbnail"),
                                ("workflow", "embedded workflow")):
                rows = [f for f in found if f.kind == kind]
                if rows:
                    print(f"  {title}:")
                    for f in rows:
                        print(f"    {f.key}: {f.describe()}" + (f"  (x{f.count})" if f.count > 1 else ""))
            if not found:
                print("  nothing a publisher would want removed")
            if args.lineage:
                print(lineage_text(p))
            print()
        return 0
    if cmd == "redact":
        changed = 0
        for p in _expand_files(args.files):
            meta = read_metadata(p)
            found = scan_metadata(meta, args.policy, training=args.ss, thumbnail=args.thumbnail, workflow=args.workflow)
            if not found:
                print(f"{os.path.basename(p)}: nothing to redact")
                continue
            print(f"{os.path.basename(p)}: {len(found)} finding(s)")
            for f in found:
                print(f"  {f.key}: {f.describe()}")
            if args.dry_run:
                continue
            out = os.path.join(args.out, os.path.basename(p)) if args.out else None
            written, how = put(p, apply_findings(meta, found), out)
            print(f"  -> {written} ({how})")
            changed += 1
        if not args.dry_run:
            print(f"{changed} file(s) changed")
        return 0
    if cmd == "strip":
        meta = read_metadata(args.file)
        kept = strip_all(meta)
        written, how = put(args.file, kept, args.out)
        print(f"\n{written} ({how}): {len(meta)} metadata key(s) -> {len(kept)}"
              + (f" ({', '.join(kept)} kept: the file does not load without it)" if kept else " (no metadata at all, as in the official files)"))
        return 0
    if cmd == "set":
        meta = read_metadata(args.file)
        cur = read_modelspec(meta)
        vals = {f: (getattr(args, f) if getattr(args, f) is not None else cur[f]) for f, _l, _r, _h in MODELSPEC_FIELDS}
        if args.compute_hash:
            vals["hash_sha256"] = data_sha256(args.file, progress=prog)
        for k, v in modelspec_defaults(inspect_path(args.file).get("kind") == "lora").items():
            vals[k] = vals.get(k) or v
        written, how = put(args.file, set_modelspec(meta, vals), args.out)
        print(f"\n{written} ({how})")
        for k, v in sorted(read_metadata(written).items()):
            if k.startswith("modelspec."):
                print(f"  {k} = {v}")
        return 0
    if cmd == "diff":
        rows = diff_metadata(read_metadata(args.a), read_metadata(args.b))
        if not rows:
            print("the metadata is identical")
        for k, x, y in rows:
            print(f"{k}:\n  {os.path.basename(args.a)}: {'(absent)' if x is None else ' '.join(x.split())[:200]}"
                  f"\n  {os.path.basename(args.b)}: {'(absent)' if y is None else ' '.join(y.split())[:200]}")
        return 0
    if cmd == "restore":
        for p in _expand_files(args.files):
            print(f"restored the saved header of {restore_header(p)}")
        return 0
    raise ValueError(cmd)


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    use_gpu = not args.cpu

    if args.cmd is None or args.gui:
        from .gui import run_gui
        return run_gui(theme=args.theme, scale=args.scale)

    prog = _progress_printer()
    log = lambda s: print("\n" + s)  # noqa: E731
    try:
        if args.cmd == "methods":
            for k in METHODS:
                print(f"{k:17s} {METHOD_LABELS[k]}")
            return 0
        if args.cmd == "spectrum":
            return _spectrum_command(args)
        if args.cmd == "advise":
            from .advisor import advise, run_candidate
            rep = advise(args.A, args.B, args.C, args.goal, args.depth, args.format, use_gpu, progress=prog)
            print("\n" + rep.text())
            if args.save:
                print("saved " + ", ".join(rep.save(args.save)))
            if args.run:
                chosen = [rep.candidates[i - 1] for i in args.pick if 1 <= i <= len(rep.candidates)] if args.pick else list(rep.candidates)
                for c in chosen:
                    if not c.steps:
                        continue
                    outs = run_candidate(rep, c, args.run, use_gpu, progress=prog, log=log, keep_intermediate=args.keep_intermediate)
                    print(f"\ncandidate '{c.label}': wrote {outs[-1]}")
            return 0
        if args.cmd == "meta":
            return _meta_command(args, prog, log)
        if args.cmd == "inspect":
            from .inspect_file import inspect_path
            for f in args.files:
                print(inspect_path(f)["text"])
                print()
            return 0
        if args.cmd == "run":
            from .recipe import load_recipe, run_recipe
            rec = load_recipe(args.recipe)
            if args.plan:
                from .plan import plan_recipe
                print(plan_recipe(rec, base_dir=os.path.dirname(os.path.abspath(args.recipe))))
                return 0
            res = run_recipe(rec, args.output, base_dir=os.path.dirname(os.path.abspath(args.recipe)),
                             use_gpu=use_gpu, progress=prog, log=log)
            print(f"\nwrote {res.path}")
            return 0
        if args.cmd == "convert":
            from .engine import convert_checkpoint
            res = convert_checkpoint(args.src, args.dst, args.format, args.passthrough, args.fp8_layers,
                                     use_gpu=use_gpu, progress=prog, log=log, int8_clip=args.int8_clip)
            print(f"\nwrote {res.path} ({res.tensors} tensors, {res.seconds:.1f}s); verify {'ok' if res.verify['ok'] else 'FAILED'}")
            return 0 if res.verify["ok"] else 1
        if args.cmd == "lora-merge":
            from .lora_merge import LoraMergeOptions, analyze_lora_merge, merge_loras
            inputs = [_lora_arg(t) for t in args.inputs]
            opts = LoraMergeOptions(average=args.average, rank_mode="dynamic" if args.dynamic else ("fixed" if args.rank else "concat"),
                                    rank=args.rank, modules="union" if args.union else "intersection", naming=args.naming, dtype=args.dtype,
                                    retention=min(1.0, max(0.5, args.retention)), rank_cap=args.rank_cap or None, rank_floor=max(1, args.rank_floor))
            if args.plan:
                from .plan import plan_lora_merge
                print(plan_lora_merge(inputs, opts))
                return 0
            if args.analyze:
                rep = analyze_lora_merge(inputs, opts, use_gpu, progress=prog)
                rc = _finish_analysis(rep, args)
                if args.dynamic:
                    from .lora_merge import prune_plan, prune_text
                    print("\n" + prune_text(prune_plan(rep, opts.retention, opts.rank_cap, opts.rank_floor, opts.dtype)))
                return rc
            res = merge_loras(inputs, args.output, opts, use_gpu, progress=prog, log=log)
            print(f"\nwrote {res.path}: {res.modules} modules, rank {res.rank_min}-{res.rank_max}, "
                  f"energy kept >= {res.kept_min * 100:.2f}%, {len(res.dropped)} dropped, {res.size_bytes / 1e6:.1f} MB")
            return 0
        if args.cmd == "extract":
            from .extract import ExtractOptions, analyze_extract, extract_lora
            opts = ExtractOptions(rank=args.rank, filter=args.filter, include=args.include, exclude=args.exclude,
                                  method=args.method, naming=args.naming, dtype=args.dtype)
            if args.plan:
                from .plan import plan_extract
                print(plan_extract(args.base, args.target, opts))
                return 0
            if args.analyze:
                return _finish_analysis(analyze_extract(args.base, args.target, opts, use_gpu, progress=prog,
                                                        null_spectrum=args.null_spectrum), args)
            res = extract_lora(args.base, args.target, args.output, opts, use_gpu, progress=prog, log=log)
            print(f"\nwrote {res.path}: {res.modules} modules, energy kept >= {res.kept_min * 100:.2f}%"
                  + (f", per group {({g: round(v, 4) for g, v in res.kept_by_group.items()})}" if res.kept_by_group else ""))
            return 0
        if args.cmd == "ckpt-merge":
            from .ckpt_merge import CkptInput, CkptMergeOptions, merge_checkpoints, premerge_report
            A = CkptInput(args.A)
            B = None
            if args.B:
                parts = args.B.split("|")
                B = CkptInput(parts[0], float(parts[1]) if len(parts) > 1 and parts[1] else 1.0,
                              _shaping_arg(parts[2]) if len(parts) > 2 else Shaping())
            C = CkptInput(args.C) if args.C else None
            loras = [_lora_arg(t) for t in args.lora]
            if len(loras) > 4:
                ap.error("at most 4 LoRAs")
            opts = CkptMergeOptions(method=args.method, output_format=args.format, passthrough=args.passthrough,
                                    fp8_layer_set=args.fp8_layers, int8_clip=args.int8_clip, lora_mode=args.lora_mode, use_gpu=use_gpu,
                                    vectors_from=args.vectors_from)
            for p in args.param:
                k, _, v = p.partition("=")
                opts.params[k] = (v.lower() == "true") if k == "dare_ties" else (int(v) if k == "seed" else float(v))
            if args.as_lora:
                opts.output_as_lora = True
                opts.lora_out["rank"] = args.as_lora
            if args.report:
                if B is None:
                    ap.error("--report needs -B")
                print(premerge_report(A, B, C, use_gpu, progress=prog))
                return 0
            if args.plan:
                from .plan import plan_ckpt_merge
                print(plan_ckpt_merge(A, B, C, loras, opts))
                return 0
            res = merge_checkpoints(A, B, C, loras, args.output, opts, progress=prog, log=log)
            if hasattr(res, "verify"):
                print(f"\nwrote {res.path} ({res.tensors} tensors, {res.seconds:.1f}s); verify {'ok' if res.verify['ok'] else 'FAILED'}")
                return 0 if res.verify["ok"] else 1
            print(f"\nwrote {res.path}: {res.modules} modules, energy kept >= {res.kept_min * 100:.2f}%")
            return 0
        ap.error(f"unknown command {args.cmd}")
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {e}")
        if os.environ.get("K2MERGE_DEBUG"):
            traceback.print_exc()
        return 1
    return 0
