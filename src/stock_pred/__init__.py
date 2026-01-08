"""Stock price prediction graduation-research project (refactored, behavior-preserving).

This package intentionally keeps algorithmic behavior identical to the original research scripts.
"""

from .common_shared import LogConfig, PWFESplit, set_seeds  # re-export (public convenience)

__all__ = ["LogConfig", "PWFESplit", "set_seeds"]
