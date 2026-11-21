"""Regression checks for the canonical method-layer moving-horizon helpers."""
import numpy as np
import pandas as pd

from src.method.moving_horizon import (
    _origin, _intervals, historical_weekdays, historical_calendar_days,
    origins, metrics, BASELINES,
)
from src.method.arm_variance import fit_variances, variance_kernel


def test_hourly_weekly_origins_and_calendar():
    start = _origin("2022-08-22", 7)
    assert start == pd.Timestamp("2022-08-22 07:00")
    expected = [start-pd.Timedelta(weeks=i) for i in (3,2,1)]
    assert _intervals(start, 12, 3) == expected
    assert historical_weekdays(start, 3) == expected
    assert list(origins(start,start+pd.Timedelta(hours=2))) == [
        start,start+pd.Timedelta(hours=1),start+pd.Timedelta(hours=2)]
    days = historical_calendar_days(expected, 12)
    assert days.is_monotonic_increasing
    assert len(days) == 3
    assert set(BASELINES) == {"statistical_output","weekly_average","weekly_temperature"}


def test_moving_horizon_metrics_keep_signed_bias():
    observed = np.array([0.2,0.3,0.4])
    predicted = np.array([0.3,0.4,0.5])
    result = metrics(observed,predicted)
    assert np.isclose(result["mse"],0.01)
    assert np.isclose(result["rmse"],0.1)
    assert np.isclose(result["mae"],0.1)
    assert np.isclose(result["bias"],0.1)


def test_arm_variance_fit_and_kernel_remain_nonnegative():
    K = variance_kernel(-2.0,-3.0,4,6)
    assert K.shape == (6,)
    assert K[0] == 0.0
    assert np.all(np.diff(K) >= -1e-12)
    tau2,sigma2,identifiable = fit_variances(
        np.sqrt(0.01+0.1*K),K)
    assert identifiable
    assert np.isclose(tau2,0.01,atol=1e-10)
    assert np.isclose(sigma2,0.1,atol=1e-10)
