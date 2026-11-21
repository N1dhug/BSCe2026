"""Fast synthetic smoke tests for the Nielsen fixed-basis BM1 solver."""
import numpy as np

from src.params.bm_1_fixed_beta import bm_1_fixed_beta


def test_fixed_beta_probability_and_likelihood():
    solver = bm_1_fixed_beta(degree=4, quadrature=28, steps_per_hour=1)
    p = {"alpha":2.0,"beta":2.0,"k":1.0,"Xi":0.5,"C":3.0,"sigma_x":0.1,"sigma_y":0.05}
    data = {"B":np.array([0.45,0.48,0.47]),"U":np.array([0.5,0.6,0.4]),
            "I":np.array([0.45,0.46,0.44])}
    initial = np.ones(5)/5.
    open_result = solver.simulate(data,p,initial,observations=False)
    filtered = solver.simulate(data,p,initial,observations=True)
    assert np.all(np.isfinite(open_result["mean"]))
    assert np.all(np.isfinite(filtered["loglik"]))
    assert np.all(filtered["coefficients"] >= -1e-7)
    assert np.allclose(filtered["coefficients"].sum(axis=1),1.,atol=1e-7)
    assert np.isfinite(open_result["nll"])
    lower, upper = solver.intervals(open_result,p["sigma_y"])
    assert np.all(lower < upper)
    assert np.max(np.abs(solver.phi.T @ solver.qw - 1.)) < 1e-10


def test_constant_zero_dynamics():
    solver = bm_1_fixed_beta(degree=4,quadrature=28,steps_per_hour=1)
    p = {"alpha":2.0,"beta":2.0,"k":1.0,"Xi":0.0,"C":3.0,"sigma_x":0.0,"sigma_y":0.05}
    c = np.ones(5)/5.
    updated = solver.propagate(c,.5,.5,.5,p)
    assert np.allclose(updated,c,atol=1e-6)
