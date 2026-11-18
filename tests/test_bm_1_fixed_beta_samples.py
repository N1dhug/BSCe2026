"""Synthetic checks for the 12-day fixed-beta BM1 evaluation and scoring."""
import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.evaluate_fixed_beta_samples import (
    DATES, N, K, historical_origins, interval_score, pooled,
    plot_best_worst, plot_timings,
)
from src.params.bm_1_fixed_beta import bm_1_fixed_beta_params


def test_selected_dates_and_history_lags():
    assert len(DATES) == 12
    assert N == 6 and K == 24
    day = pd.Timestamp(DATES[0])
    history = historical_origins(day)
    assert len(history) == 6
    assert history == sorted(history)
    assert all((day - date).days % 7 == 0 for date in history)


def test_pooled_scores_aggregate_squares_and_likelihoods():
    a = {"I":np.array([.10,.20]),"B":np.array([.10,.10]),"U":np.array([.5,.5])}
    b = {"I":np.array([.50]),"B":np.array([.5]),"U":np.array([.5])}
    sim_a = {"mean":np.array([.20,.20]),"loglik":np.array([-1.,-2.])}
    sim_b = {"mean":np.array([.30]),"loglik":np.array([-3.])}
    s1 = interval_score(a["I"],sim_a,np.array([0.,0.]),np.array([.3,.3]))
    s2 = interval_score(b["I"],sim_b,np.array([.2]),np.array([.4]))
    combined = pooled([s1,s2])
    assert combined["observations"] == 3
    assert np.isclose(combined["nll_total"],6.)
    assert np.isclose(combined["nll_mean"],2.)
    assert np.isclose(combined["rmse"],np.sqrt((.1**2+.2**2)/3))
    assert np.isclose(combined["coverage_95_pct"],100.*2./3)


def test_historical_initial_guess_uses_simplex():
    estimator = bm_1_fixed_beta_params(degree=4,quadrature=28,steps_per_hour=1)
    sample = {"I":np.array([.3,.32]),"B":np.array([.4,.41]),"U":np.array([.5,.6])}
    seed = estimator.initial_guess([sample,sample])
    states = seed["initial_distributions"]
    assert states.shape == (2,5)
    assert np.all(states > 0)
    assert np.allclose(states.sum(axis=1),1)


def test_plot_ranking_and_mean_timings(tmp_path):
    results = []
    for idx in range(6):
        date = DATES[idx]
        mean = -1.5 + idx*.2
        result = {
            "row":{"day":date},
            "target_score":{"nll_mean":mean},
            "target_rows":[{"observed":.35} for _ in range(K)],
            "oracle":{"prediction":{"mean":np.full(K,.36)},
                      "lower_95":np.full(K,.2),"upper_95":np.full(K,.5)},
            "timings":{"initial_guess":1.,"stochastic_mle":2.,
                       "oracle_optimization":3.,"complete":7.},
        }
        results.append(result)
    best,worst = plot_best_worst(results,tmp_path)
    means = plot_timings(results,tmp_path)
    assert best == list(DATES[:3])
    assert worst == list(reversed(DATES[3:6]))
    assert means["complete"] == 7.
    assert (tmp_path/"best_worst_nll.png").exists()
    assert (tmp_path/"stage_timings.png").exists()
