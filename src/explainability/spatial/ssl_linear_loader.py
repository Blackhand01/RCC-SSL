# src/explainability/spatial/ssl_linear_loader.py  (DEPRECATED)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: use src.evaluation.ssl_linear_loader instead.

Kept as a thin compatibility wrapper for existing XAI codepaths.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "src.explainability.spatial.ssl_linear_loader is deprecated; "
    "use src.evaluation.ssl_linear_loader instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Try absolute import first (package usage), then relative (namespace usage)
try:
    from src.evaluation.ssl_linear_loader import (  # type: ignore
        SSLLinearClassifier,
        ResNetBackbone,
        VitPool,
    )
except Exception:
    from ...evaluation.ssl_linear_loader import (  # type: ignore
        SSLLinearClassifier,
        ResNetBackbone,
        VitPool,
    )

__all__ = ["SSLLinearClassifier", "ResNetBackbone", "VitPool"]
