"""Generic solver-layer public API."""

from .active_set import active_set
from .finite_volume import finite_volume_2d

__all__ = ["active_set", "finite_volume_2d"]
