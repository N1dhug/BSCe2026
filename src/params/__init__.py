"""Parameter-estimation strategies."""

from src.params.WM2 import BaseParams, BruteParams as WM2BruteParams, ProxyParams
from src.params.WM3 import BruteParams as WM3BruteParams

BruteParams = WM2BruteParams

__all__ = ["BaseParams", "BruteParams", "WM2BruteParams", "WM3BruteParams", "ProxyParams"]
