"""Parameter-estimation strategies."""

from src.params.wm_2 import base_params, brute_params as WM2BruteParams, proxy_params
from src.params.wm_3 import brute_params as WM3BruteParams
from src.params.wm_4 import qp_params as WM4QPParams

brute_params = WM2BruteParams

__all__ = ["base_params", "brute_params", "WM2BruteParams", "WM3BruteParams", "WM4QPParams", "proxy_params"]
