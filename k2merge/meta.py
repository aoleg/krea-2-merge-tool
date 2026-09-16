"""File metadata: what a safetensors file says about itself, and the two ways to change it.

A safetensors file is an 8 byte header length N, N bytes of UTF-8 JSON, then the data, and every tensor's
``data_offsets`` are relative to the end of the header. Metadata that only shrinks (redaction, stripping) can
therefore be written back by rebuilding the header, padding it with spaces to exactly N and rewriting the first
8 + N bytes: the data is never read or written, which turns scrubbing a 26 GB checkpoint into a few
milliseconds. Padding a header with spaces is what the format does for alignment and what StreamWriter already
does for every file this tool writes. Metadata that grows past N takes the full rewrite instead, which raw
copies every tensor into a new file.

Before an in place patch the original 8 + N header bytes go to NAME.header.bak, so an interrupted write is
recoverable (the tensor table is in those bytes too) and undo is exact, because N never changes.

The keep list matters. The official fp8 scaled file carries ``_quantization_metadata``, a 17 KB layer table
without which the file does not load; the official bf16 and int8 convrot files carry no metadata at all, so a
stripped file of those formats is structurally identical to the official one.

Redaction is a regex over the raw value, never a parse and re-serialize: the JSON blobs other tools store
(``ss_datasets``, ComfyUI configs) would come back with a different key order and different escaping. Inside
such a blob the separators are already escaped, so the pattern accepts a doubled backslash, and every
replacement this module produces is separator free, which keeps the surrounding JSON valid.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass

from .st_io import MAX_HEADER, SafetensorsError, StreamWriter, TensorReader, read_header, same_file

# ----------------------------------------------------------------------------- what may never be removed
LOAD_BEARING = ("_quantization_metadata",)      # the fp8 scaled layer table: the file does not load without it
KEEP_ON_STRIP = LOAD_BEARING + ("format",)      # "format": two characters, not private, kept when the source had it
BACKUP_SUFFIX = ".header.bak"

# ----------------------------------------------------------------------------- finding paths
_SEP = r"(?:\\{1,2}|/)"                                  # one separator, or its escaped form inside a JSON blob
_SEG = r"[^\\/:*?\"<>|\r\n\t;,']+"                       # one path segment; spaces are legal in a file name
_DIR = rf"{_SEG}(?<! ){_SEP}"                            # a directory: no space before the separator, so prose
_HOST = r"[A-Za-z0-9][A-Za-z0-9._-]+"                    # between two paths does not join them into one match
_POSIX_ROOTS = "home|Users|mnt|media|content|workspace|root|kaggle|notebooks|data"
PATH_RE = re.compile(
    rf"file://[^\s\"<>|\r\n]+"                               # file:// URL
    rf"|(?<![A-Za-z0-9_])[A-Za-z]:{_SEP}(?:{_DIR})*(?:{_SEG})?"   # C:\dir\file or C:/dir/file, never the s:/ of https://
    rf"|\\{{2,4}}{_HOST}(?<! ){_SEP}(?:{_DIR})*(?:{_SEG})?"  # \\server\share\file; the host rule keeps an escaped
    rf"|(?<![\w:/.])/(?:{_POSIX_ROOTS})/(?:{_DIR})*(?:{_SEG})?"  # regex such as blocks\\.\\d+ from reading as one
)                                                        # the lookbehind keeps a URL's path out of it
_SEP_END_RE = re.compile(rf"(?:{_SEP})+$")
PLACEHOLDER = "<path>"

ACTIONS = ("basename", "placeholder", "drop", "skip")
TRAINING_PREFIXES = ("ss_", "sshs_", "ot_")     # kohya and musubi-tuner (ss_, sshs_), OneTrainer (ot_config, 22 KB
THUMBNAIL_KEYS = ("modelspec.thumbnail",)       # of settings with three of the author's drives in it)
WORKFLOW_KEYS = ("workflow", "prompt")          # a ComfyUI conversion leaves the graph and the prompt behind


def path_basename(match: str) -> str:
    """The last segment of a matched path, whatever mixture of separators and escaping it uses."""
    s = _SEP_END_RE.sub("", match.rstrip(" ."))
    parts = [p for p in re.split(_SEP, s) if p]
    return parts[-1] if parts else ""


@dataclass
class Finding:
    """One thing in the metadata that a published file should probably not carry."""
    key: str
    kind: str                    # path | training | thumbnail
    text: str                    # the exact substring found, or a description for a whole key finding
    action: str = "basename"     # one of ACTIONS
    count: int = 1

    @property
    def replacement(self) -> str:
        if self.action == "placeholder":
            return PLACEHOLDER
        base = path_basename(self.text)
        return PLACEHOLDER if (not base or base.endswith(":")) else base      # a bare drive or share root has no file name

    def describe(self) -> str:
        if self.kind == "path":
            return f"{self.text} -> {self.replacement}" if self.action in ("basename", "placeholder") else f"{self.text} ({self.action})"
        return f"{self.text} ({self.action})"


def scan_metadata(meta: dict, policy: str = "basename", training: bool = False, thumbnail: bool = False,
                  workflow: bool = False) -> list[Finding]:
    """Everything worth redacting in one metadata dict, in the order a user should read it."""
    out: list[Finding] = []
    for key, value in (meta or {}).items():
        if key in LOAD_BEARING:
            continue
        text = value if isinstance(value, str) else str(value)
        if training and key.startswith(TRAINING_PREFIXES):
            out.append(Finding(key, "training", f"{len(text)} chars of training metadata", "drop"))
            continue
        if thumbnail and key in THUMBNAIL_KEYS:
            out.append(Finding(key, "thumbnail", f"embedded image, {len(text)} chars", "drop"))
            continue
        if workflow and key in WORKFLOW_KEYS:
            out.append(Finding(key, "workflow", f"embedded ComfyUI {key}, {len(text)} chars", "drop"))
            continue
        seen: dict[str, int] = {}
        for m in PATH_RE.finditer(text):
            hit = m.group(0).rstrip(" .")
            if not hit or len(_SEP_END_RE.sub("", hit)) < 3:
                continue
            seen[hit] = seen.get(hit, 0) + 1
        for hit, n in seen.items():
            out.append(Finding(key, "path", hit, policy, n))
    return out


def apply_findings(meta: dict, findings) -> dict:
    """A copy of the metadata with the accepted findings applied. Longest match first, so a path that contains
    another path is replaced as a whole."""
    dropped = {f.key for f in findings if f.action == "drop"}
    out = {k: (v if isinstance(v, str) else str(v)) for k, v in (meta or {}).items() if k not in dropped or k in LOAD_BEARING}
    for f in sorted((f for f in findings if f.kind == "path" and f.action in ("basename", "placeholder")),
                    key=lambda f: len(f.text), reverse=True):
        if f.key in out:
            out[f.key] = out[f.key].replace(f.text, f.replacement)
    return out


def redact_paths(meta: dict, policy: str = "basename") -> dict:
    """Paths out, everything else untouched. Used by the writers on metadata inherited from their inputs."""
    if not meta:
        return dict(meta or {})
    return apply_findings(meta, scan_metadata(meta, policy))


def paths_in(meta: dict) -> list[str]:
    """The path-like strings a metadata dict still carries. The verification pass warns on these."""
    return [f"{f.key}: {f.text}" for f in scan_metadata(meta or {}, "basename")]


def strip_all(meta: dict) -> dict:
    """Everything gone except what the file cannot load without. An empty result means no __metadata__ at all,
    which is what the official bf16 and int8 convrot files look like."""
    return {k: v for k, v in (meta or {}).items() if k in KEEP_ON_STRIP}


# ----------------------------------------------------------------------------- the model spec fields
MODELSPEC_VERSION = "1.0.0"
MODELSPEC_PREFIX = "modelspec."
# (field, label, required by the standard, hint)
MODELSPEC_FIELDS = (
    ("title", "title", True, "the name a model browser shows"),
    ("architecture", "architecture", True, "Krea-2 for a checkpoint, Krea-2/lora for a LoRA"),
    ("implementation", "implementation", True, ""),
    ("resolution", "resolution", True, "Krea 2 is native at 1024x1024"),
    ("author", "author", False, ""),
    ("description", "description", False, ""),
    ("license", "license", False, ""),
    ("usage_hint", "usage hint", False, "how to use it: steps, cfg, strength"),
    ("trigger_phrase", "trigger phrase", False, "for a LoRA trained with one"),
    ("tags", "tags", False, "comma separated"),
    ("date", "date", False, "ISO date, e.g. 2026-09-16"),
    ("merged_from", "merged from", False, "filled from the recipe when there is one"),
    ("hash_sha256", "data hash", False, "sha256 of the tensor data, which metadata edits do not change"),
)
MODELSPEC_KEYS = tuple(MODELSPEC_PREFIX + f for f, _l, _r, _h in MODELSPEC_FIELDS)


# The spelling two independent real trainers use (a OneTrainer LoRA and a musubi one, both in the phase 11c
# survey), so the tool follows the ecosystem instead of inventing a third variant.
MODELSPEC_DEFAULTS = {"implementation": "https://github.com/krea-ai/krea-2", "resolution": "1024x1024"}


def architecture_default(is_lora: bool) -> str:
    return "Krea-2/lora" if is_lora else "Krea-2"


def modelspec_defaults(is_lora: bool) -> dict:
    """What to offer for the fields the standard calls required, when the file carries nothing."""
    return dict(MODELSPEC_DEFAULTS, architecture=architecture_default(is_lora))


def read_modelspec(meta: dict) -> dict:
    return {f: (meta or {}).get(MODELSPEC_PREFIX + f, "") for f, _l, _r, _h in MODELSPEC_FIELDS}


def set_modelspec(meta: dict, values: dict) -> dict:
    """A copy with the given model spec fields set; an empty value removes its key. The spec version is written
    whenever any field remains, so the block identifies itself."""
    out = {k: (v if isinstance(v, str) else str(v)) for k, v in (meta or {}).items()}
    for f, _l, _r, _h in MODELSPEC_FIELDS:
        key, val = MODELSPEC_PREFIX + f, str(values.get(f, "") or "").strip()
        if val:
            out[key] = val
        else:
            out.pop(key, None)
    if any(k.startswith(MODELSPEC_PREFIX) and k != MODELSPEC_PREFIX + "sai_model_spec" for k in out):
        out[MODELSPEC_PREFIX + "sai_model_spec"] = MODELSPEC_VERSION
    else:
        out.pop(MODELSPEC_PREFIX + "sai_model_spec", None)
    return out


# ----------------------------------------------------------------------------- reading
def read_metadata(path: str) -> dict:
    header, _ = read_header(path)
    return {k: (v if isinstance(v, str) else str(v)) for k, v in (header.get("__metadata__") or {}).items()}


def metadata_rows(meta: dict) -> list[tuple[str, int, str]]:
    """(key, length, one line preview) for a table, the bulky machine written keys last."""
    def order(k: str):
        return (1 if k in LOAD_BEARING or k == "merge_recipe" or k.startswith(TRAINING_PREFIXES) else 0, k)
    rows = []
    for k in sorted(meta or {}, key=order):
        v = meta[k] if isinstance(meta[k], str) else str(meta[k])
        rows.append((k, len(v), " ".join(v.split())[:160]))
    return rows


def pretty(value: str) -> str:
    """A metadata value as it should be read: JSON blobs indented, everything else as stored."""
    s = value.strip()
    if s[:1] in "{[":
        try:
            return json.dumps(json.loads(s), indent=1, ensure_ascii=False)
        except ValueError:
            pass
    return value


def diff_metadata(a: dict, b: dict) -> list[tuple[str, str, str]]:
    """(key, left, right) for every key that differs; a missing side is None."""
    out = []
    for k in sorted(set(a or {}) | set(b or {})):
        x, y = (a or {}).get(k), (b or {}).get(k)
        if x != y:
            out.append((k, x, y))
    return out


def data_sha256(path: str, progress=None, cancel=None, chunk: int = 8 << 20) -> str:
    """sha256 of the data region. Metadata edits do not change it, so it identifies the weights themselves."""
    from .engine import Cancelled
    size = os.path.getsize(path)
    _, header_len = read_header(path)
    h = hashlib.sha256()
    done = 0
    total = max(size - header_len, 1)
    with open(path, "rb", buffering=0) as f:
        f.seek(header_len)
        while True:
            if cancel is not None and cancel():
                raise Cancelled("cancelled")
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if progress is not None:
                progress(done, total, "hashing")
    return h.hexdigest()


# ----------------------------------------------------------------------------- writing
class HeaderTooLong(SafetensorsError):
    """The new metadata does not fit in the file's header. The caller writes a new file instead."""


def _header_bytes(header: dict) -> bytes:
    return json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _with_metadata(header: dict, meta: dict | None) -> dict:
    out = {k: v for k, v in header.items() if k != "__metadata__"}
    clean = {str(k): (v if isinstance(v, str) else str(v)) for k, v in (meta or {}).items()}
    if clean:
        out["__metadata__"] = clean
    return out


def backup_path(path: str) -> str:
    return path + BACKUP_SUFFIX


def patch_metadata(path: str, meta: dict | None, backup: bool = True) -> int:
    """Rewrite only this file's header, keeping its length. Returns the number of spaces the header is padded
    with. Raises HeaderTooLong when the new metadata does not fit, which leaves the file untouched."""
    header, header_len = read_header(path)
    n = header_len - 8
    new_header = _with_metadata(header, meta)
    raw = _header_bytes(new_header)
    if len(raw) > n:
        raise HeaderTooLong(f"{os.path.basename(path)}: the new metadata needs {len(raw) - n} bytes more than the file's header has")
    padded = raw + b" " * (n - len(raw))
    check = json.loads(padded.decode("utf-8"))            # the padding must still parse, and nothing else may move
    if {k: v for k, v in check.items() if k != "__metadata__"} != {k: v for k, v in header.items() if k != "__metadata__"}:
        raise SafetensorsError(f"{path}: rebuilding the header changed a tensor entry; the file was not touched")
    if backup:
        with open(path, "rb") as f:
            original = f.read(header_len)
        with open(backup_path(path), "wb") as f:
            f.write(original)
            f.flush()
            os.fsync(f.fileno())
    try:
        with open(path, "r+b") as f:
            f.write(struct.pack("<Q", n))
            f.write(padded)
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        raise SafetensorsError(f"{path}: cannot write the header ({e}). A file that a running ComfyUI or Forge "
                               f"has mapped cannot be changed until that process releases it.") from e
    back, _ = read_header(path)                            # it parses, and it says what we wrote
    if (back.get("__metadata__") or {}) != (new_header.get("__metadata__") or {}):
        raise SafetensorsError(f"{path}: the header read back differently than it was written")
    return n - len(raw)


def fits_in_place(path: str, meta: dict | None) -> bool:
    """Whether this metadata can be patched into the file's header. Redaction and stripping always fit; adding
    a field fits only when the header has room, which a file this tool wrote has none of."""
    try:
        header, header_len = read_header(path)
    except (SafetensorsError, OSError):
        return False
    return len(_header_bytes(_with_metadata(header, meta))) <= header_len - 8


def rewrite_metadata(src: str, dst: str, meta: dict | None, progress=None, cancel=None, log=None) -> str:
    """A new file with the same tensors, byte for byte, and the given metadata."""
    from .engine import Cancelled
    if same_file(src, dst):
        raise SafetensorsError("the output must not be the input file")
    with TensorReader(src) as r:
        names = r.names
        w = StreamWriter(dst, meta)
        for name in names:
            w.add(name, r.dtype(name), r.shape(name))
        if len(w.header_bytes()) > MAX_HEADER:
            raise SafetensorsError("header exceeds the safetensors 100 MB limit")
        w.begin()
        try:
            for i, name in enumerate(names, 1):
                if cancel is not None and cancel():
                    raise Cancelled("cancelled")
                w.write(name, r.raw(name))
                if progress is not None:
                    progress(i, len(names), name)
            w.close()
        except BaseException:
            w.abort()
            raise
    if log is not None:
        log(f"wrote {dst}: {len(names)} tensors, data copied unchanged")
    return dst


def write_metadata(path: str, meta: dict | None, out_path: str | None = None, backup: bool = True,
                   progress=None, cancel=None, log=None) -> tuple[str, str]:
    """Put this metadata on this file. In place when out_path is None and the header has room, a new file
    otherwise. Returns (written path, "patch" | "rewrite")."""
    if out_path is None or same_file(path, out_path or path):
        pad = patch_metadata(path, meta, backup)
        if log is not None:
            log(f"header rewritten in place, {pad} bytes of padding; the data was not touched")
        return path, "patch"
    rewrite_metadata(path, out_path, meta, progress, cancel, log)
    return out_path, "rewrite"


def restore_header(path: str) -> str:
    """Put back the header saved before the last in place patch, even if the live one is unreadable."""
    bak = backup_path(path)
    if not os.path.isfile(bak):
        raise SafetensorsError(f"no saved header next to {os.path.basename(path)}")
    with open(bak, "rb") as f:
        original = f.read()
    if len(original) < 8:
        raise SafetensorsError(f"{bak}: too short to be a header")
    n = struct.unpack("<Q", original[:8])[0]
    if 8 + n != len(original):
        raise SafetensorsError(f"{bak}: says {n} header bytes, holds {len(original) - 8}")
    with open(path, "r+b") as f:
        f.write(original)
        f.flush()
        os.fsync(f.fileno())
    read_header(path)
    os.remove(bak)
    return path


# ----------------------------------------------------------------------------- provenance
def lineage(path: str, depth: int = 4, _seen: set | None = None) -> list[tuple[int, str, str]]:
    """(indent, role, text) lines describing what a file was made from, following the recipes of the inputs
    that sit next to it. Recipes hold file names only, so resolution is per folder, and a file that is not
    there simply ends its branch."""
    from .recipe import recipe_from_file_metadata, recipe_inputs
    seen = _seen if _seen is not None else set()
    key = os.path.abspath(path).lower()
    if key in seen or depth <= 0:
        return []
    seen.add(key)
    try:
        r = recipe_from_file_metadata(path)
    except Exception:                                      # noqa: BLE001 - an unreadable input ends its branch
        return []
    if r is None:
        return []
    out: list[tuple[int, str, str]] = [(0, r["function"], _recipe_line(r))]
    base = os.path.dirname(os.path.abspath(path))
    for role, name in recipe_inputs(r):
        child = os.path.join(base, os.path.basename(name or ""))
        here = bool(name) and os.path.isfile(child)
        rows = lineage(child, depth - 1, seen) if here else []
        out.append((1, role, name + ("" if here else "  (not next to this file)")))
        out += [(d + 2, ro, tx) for d, ro, tx in rows]
    return out


def _recipe_line(r: dict) -> str:
    fn = r.get("function")
    o = r.get("options") or {}
    if fn == "ckpt_merge":
        w = (r.get("B") or {}).get("weight")
        return f"{o.get('method', '?')}" + (f" at {w}" if w is not None else "") + f", out {o.get('output_format', '?')}"
    if fn == "lora_merge":
        return f"{len(r.get('inputs', []))} input(s), rank {o.get('rank_mode')}{'' if o.get('rank') is None else ' ' + str(o.get('rank'))}"
    if fn == "extract":
        return f"rank {o.get('rank')}, {o.get('filter')} modules"
    if fn == "convert":
        return f"to {r.get('output_format')}"
    return ""


def lineage_text(path: str, depth: int = 4) -> str:
    rows = lineage(path, depth)
    if not rows:
        return "no recipe in this file, so nothing to trace"
    out = [os.path.basename(path)]
    for indent, role, text in rows:
        out.append("  " * (indent + 1) + f"{role}: {text}")
    return "\n".join(out)
