"""safetensors input and output without loading whole files.

TensorReader: header parse plus buffered reads, one tensor at a time.
StreamWriter: header computed before any data is written, then single pass data.
"""
from __future__ import annotations

import json
import os
import struct
from typing import Iterable

import numpy as np
import torch

# safetensors dtype tag <-> torch dtype
DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
DTYPES_REV: dict[torch.dtype, str] = {v: k for k, v in DTYPES.items()}
FLOAT_TAGS = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}
ELEMENT_SIZE: dict[str, int] = {k: torch.empty((), dtype=v).element_size() for k, v in DTYPES.items()}

MAX_HEADER = 100 * 1024 * 1024  # safetensors limit
# Staging buffers for GPU copies are pinned unless K2MERGE_PINNED=0 (measurement switch).
USE_PINNED = os.environ.get("K2MERGE_PINNED", "1") != "0"


class SafetensorsError(ValueError):
    pass


def read_header(path: str) -> tuple[dict, int]:
    """Returns (header dict including __metadata__ if present, header_len). Validates."""
    size = os.path.getsize(path)
    if size < 8:
        raise SafetensorsError(f"{path}: file too short to be a safetensors file ({size} bytes)")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > MAX_HEADER or 8 + n > size:
            raise SafetensorsError(f"{path}: invalid header length {n}")
        raw = f.read(n)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SafetensorsError(f"{path}: corrupted header ({e})") from e
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path}: header is not a JSON object")
    data_len = size - 8 - n
    for k, v in header.items():
        if k == "__metadata__":
            continue
        try:
            a, b = v["data_offsets"]
            dt, shape = v["dtype"], v["shape"]
        except (KeyError, TypeError, ValueError) as e:
            raise SafetensorsError(f"{path}: bad tensor entry {k!r}") from e
        if dt not in DTYPES:
            raise SafetensorsError(f"{path}: unsupported dtype {dt!r} on {k!r}")
        if a > b or b > data_len:
            raise SafetensorsError(f"{path}: tensor {k!r} offsets {a}-{b} exceed data length {data_len}")
        n_el = 1
        for s in shape:
            n_el *= int(s)
        if n_el * ELEMENT_SIZE[dt] != b - a:
            raise SafetensorsError(f"{path}: tensor {k!r} byte length does not match shape {shape} {dt}")
    return header, 8 + n


def tensor_infos(header: dict) -> dict:
    return {k: v for k, v in header.items() if k != "__metadata__"}


class TensorReader:
    """Reads tensors individually from a .safetensors file with buffered file reads.

    No mmap: a mapped 26 GB file inflates the process working set by the whole
    file as it is traversed and keeps the file locked against replacement on
    Windows. A seek + readinto per tensor costs one copy into a private buffer
    that becomes the tensor's storage, and nothing else stays resident.
    """

    def __init__(self, path: str):
        self.path = path
        self.header, self.header_len = read_header(path)
        self.metadata: dict = self.header.get("__metadata__") or {}
        self.infos: dict = tensor_infos(self.header)
        # names in file (data offset) order: sequential reads
        self.names: list[str] = sorted(self.infos, key=lambda k: self.infos[k]["data_offsets"][0])
        self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __contains__(self, name: str) -> bool:
        return name in self.infos

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self.path, "rb", buffering=0)

    def dtype(self, name: str) -> str:
        return self.infos[name]["dtype"]

    def shape(self, name: str) -> list[int]:
        return list(self.infos[name]["shape"])

    def nbytes(self, name: str) -> int:
        a, b = self.infos[name]["data_offsets"]
        return b - a

    def raw(self, name: str) -> bytearray:
        """The tensor bytes in a private buffer."""
        self._ensure_open()
        a, b = self.infos[name]["data_offsets"]
        n = b - a
        buf = bytearray(n)
        if n:
            self._file.seek(self.header_len + a)
            view = memoryview(buf)
            got = 0
            while got < n:
                k = self._file.readinto(view[got:])
                if not k:
                    raise SafetensorsError(f"{self.path}: short read on {name!r}")
                got += k
        return buf

    def _read_into(self, name: str, view: memoryview) -> None:
        a, b = self.infos[name]["data_offsets"]
        n = b - a
        self._file.seek(self.header_len + a)
        got = 0
        while got < n:
            k = self._file.readinto(view[got:n])
            if not k:
                raise SafetensorsError(f"{self.path}: short read on {name!r}")
            got += k

    def read(self, name: str, device=None) -> torch.Tensor:
        """Tensor in its stored dtype, owning its buffer (no copy on CPU, one copy to a GPU).

        For a CUDA device the bytes are read into a persistent pinned staging
        buffer and copied from there: a pageable host to device copy makes the
        driver commit a staging area of the tensor's size per copy, which showed
        up as several GB of private memory on 26 GB files.
        """
        info = self.infos[name]
        dt = DTYPES[info["dtype"]]
        shape = [int(s) for s in info["shape"]]
        n = self.nbytes(name)
        cuda = device is not None and torch.device(device).type == "cuda"
        if n == 0:
            t = torch.empty(shape, dtype=dt)
            return t.to(device) if device is not None else t
        if cuda:
            self._ensure_open()
            pin = self._pinned(n)
            self._read_into(name, memoryview(pin.numpy()))
            return pin[:n].view(dt).reshape(shape).to(device)
        buf = self.raw(name)
        return torch.frombuffer(buf, dtype=torch.uint8).view(dt).reshape(shape)

    def _pinned(self, n: int) -> torch.Tensor:
        pin = getattr(self, "_pin", None)
        if pin is None or pin.numel() < n:
            # grow to the largest tensor of the file at once, so the buffer is allocated once
            biggest = max(self.nbytes(k) for k in self.infos)
            size = max(n, biggest)
            try:
                pin = torch.empty(size, dtype=torch.uint8, pin_memory=USE_PINNED)
            except RuntimeError:
                pin = torch.empty(size, dtype=torch.uint8)
            self._pin = pin
        return pin

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        self._pin = None


def tensor_bytes(t: torch.Tensor) -> np.ndarray:
    """uint8 numpy view of a CPU tensor's bytes (little endian, C order)."""
    t = t.detach().contiguous()
    if t.device.type != "cpu":
        t = t.cpu()
    if t.numel() == 0:
        return np.empty(0, dtype=np.uint8)
    return t.reshape(-1).view(torch.uint8).numpy()


class StreamWriter:
    """Writes a safetensors file in one pass.

    Usage: plan every tensor with add(name, dtype_tag, shape), then begin(),
    then write(name, tensor_or_bytes) in any order that is a permutation of the
    plan (file order is the plan order), then close(). The header is computed
    from the plan, so nothing is buffered and no temp copy is made.
    """

    def __init__(self, path: str, metadata: dict | None = None):
        self.path = path
        self.metadata = {str(k): str(v) for k, v in (metadata or {}).items()} or None
        self._plan: dict[str, dict] = {}
        self._order: list[str] = []
        self._offset = 0
        self._f = None
        self._next = 0
        self._written = 0
        self._tmp = path + ".part"

    def add(self, name: str, dtype_tag: str, shape: Iterable[int]) -> None:
        if self._f is not None:
            raise RuntimeError("cannot add tensors after begin()")
        if name in self._plan:
            raise ValueError(f"duplicate tensor name {name!r}")
        if dtype_tag not in DTYPES:
            raise ValueError(f"unsupported dtype tag {dtype_tag!r}")
        shape = [int(s) for s in shape]
        n = 1
        for s in shape:
            n *= s
        nbytes = n * ELEMENT_SIZE[dtype_tag]
        self._plan[name] = {"dtype": dtype_tag, "shape": shape,
                            "data_offsets": [self._offset, self._offset + nbytes]}
        self._order.append(name)
        self._offset += nbytes

    def planned(self, name: str) -> bool:
        return name in self._plan

    @property
    def order(self) -> list[str]:
        return list(self._order)

    @property
    def data_size(self) -> int:
        return self._offset

    def header_bytes(self) -> bytes:
        header = dict(self._plan)
        if self.metadata:
            header["__metadata__"] = self.metadata
        raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        pad = (8 - len(raw) % 8) % 8
        return raw + b" " * pad

    def begin(self) -> None:
        if self._f is not None:
            raise RuntimeError("begin() called twice")
        hb = self.header_bytes()
        if len(hb) > MAX_HEADER:
            raise SafetensorsError("header exceeds the safetensors 100 MB limit")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self._f = open(self._tmp, "wb")
        self._f.write(struct.pack("<Q", len(hb)))
        self._f.write(hb)
        self._next = 0

    def write(self, name: str, data) -> None:
        """data: torch.Tensor (any device; dtype must match the plan) or a bytes-like object."""
        if self._f is None:
            raise RuntimeError("begin() must be called before write()")
        expected = self._order[self._next] if self._next < len(self._order) else None
        if name != expected:
            raise RuntimeError(f"write order mismatch: expected {expected!r}, got {name!r}")
        info = self._plan[name]
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        if isinstance(data, torch.Tensor):
            if DTYPES_REV.get(data.dtype) != info["dtype"]:
                raise TypeError(f"{name}: dtype {data.dtype} does not match plan {info['dtype']}")
            if list(data.shape) != info["shape"]:
                raise ValueError(f"{name}: shape {list(data.shape)} does not match plan {info['shape']}")
            buf = tensor_bytes(data)
        else:
            buf = memoryview(data)
        if len(buf) != nbytes:
            raise ValueError(f"{name}: {len(buf)} bytes written, plan says {nbytes}")
        if nbytes:
            self._f.write(buf)
        self._written += nbytes
        self._next += 1

    def abort(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
        if os.path.exists(self._tmp):
            try:
                os.remove(self._tmp)
            except OSError:
                pass

    def close(self) -> None:
        if self._f is None:
            raise RuntimeError("close() without begin()")
        if self._next != len(self._order):
            self.abort()
            raise RuntimeError(f"only {self._next} of {len(self._order)} planned tensors were written")
        self._f.flush()
        self._f.close()
        self._f = None
        os.replace(self._tmp, self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
        elif self._f is not None:
            self.close()
        return False


def same_file(a: str, b: str) -> bool:
    try:
        return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a).lower() == os.path.abspath(b).lower()
