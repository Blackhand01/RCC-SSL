from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_TORCH_OPS_PATH = _THIS_DIR / "torch_ops.py"

if "src.training.utils.torch_ops" not in sys.modules and _TORCH_OPS_PATH.exists():
    spec = importlib.util.spec_from_file_location("src.training.utils.torch_ops", _TORCH_OPS_PATH)
    module = importlib.util.module_from_spec(spec) if spec and spec.loader else None
    if module and spec and spec.loader:
        spec.loader.exec_module(module)
        sys.modules.setdefault("src.training.utils.torch_ops", module)
