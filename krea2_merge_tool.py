#!/usr/bin/env python3
"""Krea 2 Merge Tool entry point. No arguments: GUI. See --help for the CLI."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from k2merge.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
