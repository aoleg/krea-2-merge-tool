"""Post-install self check, run by install.bat: python -m k2merge.selfcheck"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    ok = True
    print(f"python  : {sys.version.split()[0]} ({sys.executable})")
    try:
        import torch
        print(f"torch   : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"cuda    : {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
        else:
            print("cuda    : NOT AVAILABLE. The tool will run on the CPU, which is much slower.")
            print("          If this machine has an NVIDIA GPU, the installed torch is a CPU build;")
            print("          delete the venv folder and run install.bat again.")
            ok = False
    except ImportError as e:
        print(f"torch   : import failed: {e}")
        ok = False
    for mod in ("safetensors", "numpy", "tkinter"):
        try:
            __import__(mod)
            print(f"{mod:8s}: ok")
        except ImportError as e:
            print(f"{mod:8s}: import failed: {e}")
            ok = False

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref = os.path.join(here, "reference")
    names = ("krea2_turbo_bf16", "krea2_turbo_fp8_scaled", "krea2_turbo_int8_convrot")
    for n in names:
        p = os.path.join(ref, n + ".header.json")
        try:
            with open(p, encoding="utf-8") as f:
                h = json.load(f)
            print(f"reference {n}: {len([k for k in h if k != '__metadata__'])} tensors")
        except Exception as e:  # noqa: BLE001
            print(f"reference {n}: FAILED ({e})")
            ok = False

    try:
        from k2merge.hadamard import build_hadamard, rotate_weight
        import torch
        w = torch.randn(8, 256)
        h = build_hadamard(256)
        back = rotate_weight(rotate_weight(w, h, 256), h, 256)
        assert torch.allclose(w, back, atol=1e-4), "hadamard round trip failed"
        print("hadamard: ok")
    except Exception as e:  # noqa: BLE001
        print(f"hadamard: FAILED ({e})")
        ok = False

    print("SELF CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
