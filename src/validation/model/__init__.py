"""Model input validation required by BM1 and WM4."""

from .bm import bm_validation
from .bm_1 import bm_1_validation
from .wm_1 import wm_1_validation
from .wm_4 import wm_4_validation

__all__ = ["bm_validation", "bm_1_validation", "wm_1_validation", "wm_4_validation"]
