import numpy as np


class moving_horizon_validation:
    @staticmethod
    def arguments(args, cases):
        from src.exe.run_model.wm_4.moving_horizon_common import GROUPS
        from pandas import Timestamp

        if Timestamp(args.end) < Timestamp(args.start):
            raise ValueError("end must not precede start.")
        if args.stride < 1 or args.steps_per_hour < 2 or args.max_origins < 0:
            raise ValueError("stride must be positive, steps-per-hour >=2 and max-origins >=0.")
        if args.max_state_iter < 1 or not np.isfinite(args.state_tol) or args.state_tol <= 0:
            raise ValueError("Invalid sequential QP convergence settings.")
        if args.oracle_starts < 1 or args.oracle_nfev < 1:
            raise ValueError("Oracle starts and max function evaluations must be positive.")
        groups = [g.strip().upper() for g in args.groups.split(",")]
        if not groups or len(groups) != len(set(groups)) or any(g not in GROUPS for g in groups):
            raise ValueError(f"groups must be a non-repeating comma-separated subset of {GROUPS}.")
        variants = list(cases) if args.variants.lower() == "all" else [v.strip() for v in args.variants.split(",")]
        if not variants or len(variants) != len(set(variants)) or any(v not in cases for v in variants):
            raise ValueError("variants must select unique response configurations from: " + ", ".join(cases))
        return groups, variants
