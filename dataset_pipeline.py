"""Compatibility shim.

Keeps the original import path `import dataset_pipeline` working after refactor.
Implementation lives in `src/stock_pred/dataset_pipeline.py`.

NOTE: This file must not change runtime behavior; it only forwards symbols.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stock_pred.dataset_pipeline import *  # noqa: F401,F403
