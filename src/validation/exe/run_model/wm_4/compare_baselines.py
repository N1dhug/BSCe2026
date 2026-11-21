import numpy as np
import pandas as pd

from src.method.moving_horizon import GROUPS


class compare_baselines_validation:
    """CLI and matched-evaluation checks for WM4 baseline comparisons."""

    @staticmethod
    def arguments(args, candidates):
        if pd.Timestamp(args.end) < pd.Timestamp(args.start):
            raise ValueError("end must not precede start.")
        if args.stride < 1 or args.steps_per_hour < 2 or args.max_origins < 0:
            raise ValueError("stride must be >=1, steps-per-hour >=2 and max-origins >=0.")
        if args.max_state_iter < 1 or not np.isfinite(args.state_tol) or args.state_tol <= 0:
            raise ValueError("Invalid sequential fit iteration count or convergence tolerance.")
        if args.progress_every < 1: raise ValueError("progress-every must be >=1.")
        groups = [value.strip().upper() for value in args.groups.split(",")]
        if not groups or len(groups) != len(set(groups)) or any(group not in GROUPS for group in groups):
            raise ValueError("groups must be a unique comma-separated subset of: " + ", ".join(GROUPS))
        names = list(candidates) if args.baselines.lower() == "all" else [name.strip() for name in args.baselines.split(",")]
        if not names or len(names) != len(set(names)) or any(name not in candidates for name in names):
            raise ValueError("baselines must be a unique comma-separated subset of: " + ", ".join(candidates))
        return groups, names
