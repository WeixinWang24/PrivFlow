"""Internal dependency loader."""
from __future__ import annotations

try:  # pragma: no cover - thin wrapper
    import numpy as np
except ImportError as exc:  # pragma: no cover - executed only when numpy missing
    raise ImportError(
        "PrivFlow requires the 'numpy' package. Install it via 'pip install numpy'."
    ) from exc

__all__ = ["np"]
