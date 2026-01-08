"""Thin wrapper (backward compatible entrypoint).

Original script path preserved: LightGBM/trinary_LightGBM.py
The full implementation lives in: stock_pred.models.lightgbm_trinary_LightGBM

Running this file executes the implementation module with `__name__ == "__main__"`,
so behavior (including any sweep logic) is preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path
import runpy

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __name__ == "__main__":
    runpy.run_module("stock_pred.models.lightgbm_trinary_LightGBM", run_name="__main__")
