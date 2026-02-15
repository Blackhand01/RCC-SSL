#!/usr/bin/env python3
from __future__ import annotations

"""
Helper to run package modules as scripts without breaking relative imports.

Usage (top of a file):
  from explainability.utils.bootstrap import bootstrap_package
  bootstrap_package(__file__, globals())
"""

from pathlib import Path
import sys
from typing import Dict


def bootstrap_package(file: str, g: Dict) -> None:
    """
    If executed as a script (no __package__), add .../src to sys.path and set __package__
    so relative imports work.
    """
    if g.get("__package__"):
        return
    this = Path(file).resolve()

    src_dir = this
    while src_dir.name != "src" and src_dir.parent != src_dir:
        src_dir = src_dir.parent
    if src_dir.name != "src":
        return

    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    rel = this.relative_to(src_dir).with_suffix("")   # explainability/...
    pkg = ".".join(rel.parts[:-1])                    # explainability.<...>
    g["__package__"] = pkg
