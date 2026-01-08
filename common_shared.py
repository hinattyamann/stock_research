"""Compatibility shim.

Keeps the original import path `import common_shared` working after refactor.
Implementation lives in `src/stock_pred/common_shared.py`.

NOTE: This file must not change runtime behavior; it only forwards symbols.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Re-export everything
from stock_pred.common_shared import *  # noqa: F401,F403
