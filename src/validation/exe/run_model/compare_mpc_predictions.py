"""Validation for matched 1-8h BM1 / WM4 MPC evaluation."""

from pathlib import Path
from hashlib import sha256
from uuid import uuid4
import json

import numpy as np
import pandas as pd

from src.validation.exe.run_model.bm_1.predict_mpc import bm_1_mpc_validation


class compare_mpc_validation:
    @staticmethod
    def date_range(start, end, horizon, steps_per_hour, x_steps, fp_max_step, prior_bandwidth, substeps):
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None or end < start:
            raise ValueError("MPC comparison requires start_day <= end_day, without timezone.")
        bm_1_mpc_validation.options(horizon, x_steps, fp_max_step, prior_bandwidth, substeps, 5)
        if isinstance(steps_per_hour, bool) or not isinstance(steps_per_hour, int) or steps_per_hour < 2:
            raise ValueError("steps_per_hour must be an integer >=2 for WM4.")
        return start, end

    @staticmethod
    def wm4_saved(source, sigma_fit_mode="mpc_replay", r0_variance_mode="baseline_residual"):
        if sigma_fit_mode not in ("mpc_replay", "historical_p0", "zero_p0"):
            raise ValueError("WM4 sigma fitting must use mpc_replay, historical_p0 or zero_p0.")
        if r0_variance_mode not in ("baseline_residual", "zero"):
            raise ValueError("WM4 R0 initial variance mode must be baseline_residual or zero.")
        if source is None: return None
        frame = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()
        if not isinstance(frame, pd.DataFrame): raise TypeError("WM4 saved forecasts must be a DataFrame/CSV path.")
        required = {"origin", "target_time", "lead_hours", "I_mpc", "lower_68", "upper_68", "lower_95", "upper_95"}
        if missing := required - set(frame.columns):
            raise ValueError("WM4 saved forecast missing " + ", ".join(sorted(missing)))
        if "sigma_fit_mode" not in frame.columns:
            # Old saved WM4 forecasts used residual-only (zero-P0) sigma fits.
            if sigma_fit_mode != "zero_p0":
                raise ValueError("Saved WM4 forecasts predate historical-P0 sigma fitting. "
                    "Regenerate the WM4 forecasts or explicitly request --wm4-sigma-fit-mode zero_p0.")
            frame["sigma_fit_mode"] = "zero_p0"
        if not frame["sigma_fit_mode"].eq(sigma_fit_mode).all():
            raise ValueError(f"Saved WM4 sigma_fit_mode does not match requested {sigma_fit_mode}.")
        if sigma_fit_mode == "mpc_replay":
            required_replay = {"process_kernel_unit", "initial_variance",
                "calibration_mpc_count", "calibration_oracle_count"}
            if missing_replay := required_replay - set(frame.columns):
                raise ValueError("Saved WM4 MPC-replay forecasts missing " + ", ".join(sorted(missing_replay)))
        if "r0_variance_mode" not in frame.columns:
            # Previously saved WM4 MPC forecasts set R0 variance to zero.
            if r0_variance_mode != "zero":
                raise ValueError("Saved WM4 forecasts predate baseline-derived R0 variance. "
                    "Regenerate WM4 forecasts or explicitly request --wm4-r0-variance-mode zero.")
            if sigma_fit_mode == "historical_p0":
                raise ValueError("Saved WM4 forecasts used full historical P0 for sigma fitting, "
                    "not the new diagonal MPC covariance. Regenerate WM4 forecasts.")
            frame["r0_variance_mode"] = "zero"
        if not frame["r0_variance_mode"].eq(r0_variance_mode).all():
            raise ValueError(f"Saved WM4 r0_variance_mode does not match requested {r0_variance_mode}.")
        frame["origin"] = pd.to_datetime(frame["origin"])
        frame["target_time"] = pd.to_datetime(frame["target_time"])
        if frame["origin"].dt.tz is not None or frame["target_time"].dt.tz is not None:
            raise ValueError("WM4 saved timestamps must be timezone-naive.")
        frame["lead_hours"] = pd.to_numeric(frame["lead_hours"]).astype(int)
        if frame.duplicated(subset=("origin", "lead_hours")).any():
            raise ValueError("WM4 saved forecasts contain duplicate origin/lead rows.")
        return frame.set_index(["origin", "lead_hours"], drop=False).sort_index()


    @staticmethod
    def observed(value, timestamp):
        value = float(value)
        if not np.isfinite(value): raise ValueError(f"Missing actual consumption at {timestamp}.")
        return value

    @staticmethod
    def complete_pair(bm_rows, wm_rows, origin, horizon):
        if len(bm_rows) != horizon or len(wm_rows) != horizon: raise ValueError(f"{origin}: incomplete {horizon}-lead predictions.")
        b, w = pd.DataFrame(bm_rows).sort_values("lead_hours"), pd.DataFrame(wm_rows).sort_values("lead_hours")
        expected = np.arange(1, horizon + 1, dtype=int)
        if not np.array_equal(b["lead_hours"].to_numpy(dtype=int), expected) or not np.array_equal(w["lead_hours"].to_numpy(dtype=int), expected):
            raise ValueError(f"{origin}: lead indices do not cover 1..{horizon}.")
        if not pd.to_datetime(b["target_time"]).reset_index(drop=True).equals(pd.to_datetime(w["target_time"]).reset_index(drop=True)):
            raise ValueError(f"{origin}: target timestamps disagree across BM1 and WM4.")
        return b, w

    REQUIRED = {"origin", "origin_day", "target_time", "lead_hours", "I_observed", "I_BM1", "I_WM4",
        "BM1_lower_68", "BM1_upper_68", "BM1_lower_95", "BM1_upper_95",
        "WM4_lower_68", "WM4_upper_68", "WM4_lower_95", "WM4_upper_95"}

    @staticmethod
    def checkpoint_directory(folder, options):
        """Separate journals for distinct model settings; never overwrite a running CSV."""
        fingerprint = sha256(json.dumps(options, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        return Path(folder) / "origin_checkpoints" / fingerprint

    @classmethod
    def checked_rows(cls, frame, horizon, *, day=None):
        if frame.empty: raise ValueError("Checkpoint has no forecast rows.")
        missing = cls.REQUIRED - set(frame.columns)
        if missing: raise ValueError("Checkpoint lacks " + ", ".join(sorted(missing)))
        if frame.duplicated(subset=("origin", "lead_hours")).any():
            raise ValueError("Checkpoint has duplicated origin/lead rows.")
        frame = frame.copy()
        for col in ("origin", "origin_day", "target_time"): frame[col] = pd.to_datetime(frame[col])
        frame["lead_hours"] = pd.to_numeric(frame["lead_hours"], errors="raise").astype(int)
        for col in ("I_observed", "I_BM1", "I_WM4", "BM1_lower_68", "BM1_upper_68",
            "BM1_lower_95", "BM1_upper_95", "WM4_lower_68", "WM4_upper_68",
            "WM4_lower_95", "WM4_upper_95"):
            if not np.all(np.isfinite(pd.to_numeric(frame[col], errors="coerce"))):
                raise ValueError(f"Checkpoint has invalid {col}.")
        for origin, group in frame.groupby("origin"):
            if len(group) != horizon or not np.array_equal(
                np.sort(group["lead_hours"].to_numpy(dtype=int)), np.arange(1, horizon + 1)
            ): raise ValueError(f"Checkpoint {origin} has incomplete prediction leads.")
            if day is not None and pd.Timestamp(origin).normalize() != pd.Timestamp(day).normalize():
                raise ValueError(f"Checkpoint belongs to a different day: {origin}.")
            expected = origin + pd.to_timedelta(group["lead_hours"], unit="h")
            if not np.array_equal(group["target_time"].to_numpy(), expected.to_numpy()):
                raise ValueError(f"Checkpoint {origin} has inconsistent target timestamps.")
        return frame.sort_values(["origin", "lead_hours"]).reset_index(drop=True)

    @classmethod
    def write_origin(cls, folder, options, origin, rows):
        """Commit one immutable origin CSV; Windows-locked aggregate files are untouched."""
        folder = Path(folder)
        frame = cls.checked_rows(pd.DataFrame(rows), int(options["horizon"]), day=origin)
        if frame["origin"].nunique() != 1 or frame["origin"].iloc[0] != pd.Timestamp(origin):
            raise ValueError(f"Invalid origin checkpoint for {origin}.")
        directory = cls.checkpoint_directory(folder, options)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{pd.Timestamp(origin):%H}_{uuid4().hex}.csv"
        # A unique filename with exclusive creation avoids Windows os.replace on
        # a destination held open by Excel, OneDrive or another process.
        with path.open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
        return path

    @staticmethod
    def write_settings(folder, options):
        """Keep provenance for an older consolidated CSV when settings change."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        info = folder / "settings.json"
        marker = folder / "matched_forecasts.settings.json"
        previous = json.loads(info.read_text(encoding="utf-8")) if info.exists() else None
        if previous is not None and not marker.exists():
            marker.write_text(json.dumps(previous, indent=2), encoding="utf-8")
        if previous != options:
            info.write_text(json.dumps(options, indent=2), encoding="utf-8")

    @staticmethod
    def consolidated_settings(folder, options):
        marker = Path(folder) / "matched_forecasts.settings.json"
        marker.write_text(json.dumps(options, indent=2), encoding="utf-8")

    @classmethod
    def saved_day(cls, folder, options):
        """Recover the latest valid version of each origin from legacy and journal files.

        The previous implementation can leave matched_forecasts.pending.csv
        when Windows prevents replacement of an opened matched_forecasts.csv.
        Both files are recoverable. Journal files are validated independently,
        so an interrupted write cannot masquerade as a complete forecast.
        """
        folder = Path(folder)
        info = folder / "settings.json"
        if not info.is_file(): return None
        with info.open(encoding="utf-8") as handle: previous = json.load(handle)
        if previous != options: return None
        sources = []
        marker = folder / "matched_forecasts.settings.json"
        legacy_options = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else previous
        if legacy_options == options:
            sources.extend((folder / "matched_forecasts.csv", folder / "matched_forecasts.pending.csv"))
        sources += sorted(cls.checkpoint_directory(folder, options).glob("*.csv"))
        latest = {}
        for path in sources:
            if not path.is_file(): continue
            try:
                frame = pd.read_csv(path)
                validated = cls.checked_rows(frame, int(options["horizon"]), day=folder.name)
            except (OSError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                print(f"WARNING: ignored incomplete MPC checkpoint {path.name}: {exc}", flush=True)
                continue
            # Prefer a committed origin journal over an older daily CSV,
            # even if Excel modifies the latter after the forecast completed.
            priority = 1 if path.parent == cls.checkpoint_directory(folder, options) else 0
            version = (priority, path.stat().st_mtime_ns)
            for origin, group in validated.groupby("origin"):
                if origin not in latest or version >= latest[origin][0]:
                    latest[origin] = (version, group)
        if not latest: return None
        return pd.concat([part for _, part in latest.values()], ignore_index=True).sort_values(
            ["origin", "lead_hours"]).reset_index(drop=True)
