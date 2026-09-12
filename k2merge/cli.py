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
    ap.add_argument("--theme", default="native", choices=["native"], help="the GUI uses the native Windows theme")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run a recipe JSON file")
    r.add_argument("recipe")
    r.add_argument("-o", "--output", help="override the recipe's output path")
    r.add_argument("--plan", action="store_true", help="describe the run, write nothing")

    i = sub.add_parser("inspect", help="summarize safetensors files")
    i.add_argument("files", nargs="+")

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
    lm.add_argument("--union", action="store_true", help="keep modules present in any input")
    lm.add_argument("--naming", default="comfy", choices=["comfy", "kohya", "input"])
    lm.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    lm.add_argument("--analyze", action="store_true", help="print the rank / energy report and exit")
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
    ex.add_argument("--analyze", action="store_true")
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
    cm.add_argument("--as-lora", type=int, default=None, metavar="RANK", help="write the result as a LoRA of this rank")
    cm.add_argument("--report", action="store_true", help="print the pre merge report and exit")
    cm.add_argument("--plan", action="store_true")

    m = sub.add_parser("methods", help="list the checkpoint merge methods and what the weight means")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    use_gpu = not args.cpu

    if args.cmd is None or args.gui:
        from .gui import run_gui
        return run_gui(theme=args.theme)

    prog = _progress_printer()
    log = lambda s: print("\n" + s)  # noqa: E731
    try:
        if args.cmd == "methods":
            for k in METHODS:
                print(f"{k:17s} {METHOD_LABELS[k]}")
            return 0
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
            opts = LoraMergeOptions(average=args.average, rank_mode="fixed" if args.rank else "concat", rank=args.rank,
                                    modules="union" if args.union else "intersection", naming=args.naming, dtype=args.dtype)
            if args.plan:
                from .plan import plan_lora_merge
                print(plan_lora_merge(inputs, opts))
                return 0
            if args.analyze:
                print(analyze_lora_merge(inputs, opts, use_gpu, progress=prog).text())
                return 0
            res = merge_loras(inputs, args.output, opts, use_gpu, progress=prog, log=log)
            print(f"\nwrote {res.path}: {res.modules} modules, rank {res.rank_min}-{res.rank_max}, "
                  f"energy kept >= {res.kept_min * 100:.2f}%, {len(res.dropped)} dropped")
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
                print(analyze_extract(args.base, args.target, opts, use_gpu, progress=prog).text())
                return 0
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
                                    fp8_layer_set=args.fp8_layers, int8_clip=args.int8_clip, lora_mode=args.lora_mode, use_gpu=use_gpu)
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
