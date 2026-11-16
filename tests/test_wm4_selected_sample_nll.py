"""Regression tests for ARM stochastic fitting/scoring on hours 1..23."""
import numpy as np
import pytest

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import variance_kernel
from src.exe.run_model.wm_4.selected_sample_nll import (
    K,N,DEFAULT_MIN_TAU,positive_variances,score_likelihoods,
)


def synthetic_history(errors,steps_per_hour=3):
    """Construct six historical trajectories whose exact residuals are supplied."""
    residual=np.asarray(errors,dtype=float)
    assert residual.shape==(N,K)
    idx=np.arange(K)*steps_per_hour
    inputs,states=[],[]
    for row in residual:
        observed=np.zeros(K*steps_per_hour+1)
        predicted=np.zeros_like(observed)
        predicted[idx]=-row
        inputs.append({"I":observed})
        states.append({"I":predicted})
    return {"tau2":0.,"sigma2":0.2,"theta":np.array([-2.,-0.1,-0.5]),
            "historical_inputs":inputs,"historical_states":states}


def test_historical_variance_fit_excludes_all_six_initial_observations():
    residual=np.full((N,K),0.05)
    residual[:,0]=10.0  # Deliberately huge hour-0 outliers must have NO effect.
    result=synthetic_history(residual)
    tau2,sigma2,regularized=positive_variances(result,3)
    assert not regularized
    assert tau2==pytest.approx(0.05**2,rel=1e-10,abs=1e-10)
    assert sigma2==pytest.approx(0.,abs=1e-10)
    residual[:,0]=0.
    unaffected=positive_variances(synthetic_history(residual),3)
    np.testing.assert_allclose([tau2,sigma2],unaffected[:2],atol=1e-10)


def test_reestimate_both_parameters_even_if_old_variance_fit_was_positive():
    residual=np.full((N,K),0.035)
    result=synthetic_history(residual)
    result["tau2"],result["sigma2"]=1.,1.  # Old hour-0-inclusive values ignored.
    tau2,sigma2,regularized=positive_variances(result,3)
    assert tau2==pytest.approx(0.035**2,abs=1e-10)
    assert sigma2==pytest.approx(0.,abs=1e-10)
    assert not regularized


def test_optional_tau_floor_applied_only_after_hour_zero_exclusion():
    residual=np.full((N,K),0.01)
    result=synthetic_history(residual)
    tau2,sigma2,regularized=positive_variances(result,3,0.02)
    assert regularized
    assert tau2>=0.02**2
    assert sigma2>=0.


def test_zero_tau_admissible_with_nonzero_process_variance():
    sph=4
    kernel=variance_kernel(-2.,-0.5,sph,K)
    residual=np.tile(np.sqrt(0.05*kernel),(N,1))
    result=synthetic_history(residual,sph)
    tau2,sigma2,regularized=positive_variances(result,sph)
    assert tau2==pytest.approx(0.,abs=1e-9)
    assert sigma2==pytest.approx(0.05,rel=1e-8)
    assert not regularized
    mu=np.full(K,0.5)
    marginal,sequential,var,filtered=score_likelihoods(
        mu+residual[0],mu,result["theta"],tau2,sigma2,sph)
    assert marginal.shape==sequential.shape==var.shape==filtered.shape==(K-1,)
    assert (var>0).all()
    assert np.isfinite(np.r_[marginal,sequential,var,filtered]).all()


def test_marginal_equals_sequential_without_process_noise():
    mean=np.linspace(0.3,0.6,K)
    observed=mean+0.02*np.sin(np.arange(K))
    marginal,sequential,var,conditional=score_likelihoods(
        observed,mean,[-2.,-0.1,-0.4],0.002,0.,3)
    assert len(marginal)==K-1
    np.testing.assert_allclose(sequential,marginal,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(conditional,mean[1:],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(var,np.full(K-1,0.002))


def test_hour_zero_observation_neither_scored_nor_used_in_filter():
    mean=np.full(K,0.4)
    observed=mean+0.03*np.cos(np.arange(K))
    args=(mean,[-2.,-0.1,-0.4],0.001**2,0.005,4)
    first=score_likelihoods(observed,*args)
    changed=observed.copy()
    changed[0]=1000.  # Cannot affect the fixed-state conditional forecast.
    second=score_likelihoods(changed,*args)
    for a,b in zip(first,second): np.testing.assert_array_equal(a,b)
    marginal,sequential,var,_=first
    assert np.isfinite(np.r_[marginal,sequential,var]).all()
    assert sequential[0]==pytest.approx(marginal[0])
    assert not np.allclose(sequential,marginal)


def test_both_zero_variances_rejected_at_first_scored_hour():
    with pytest.raises(ValueError,match="Zero/nonfinite predictive variance"):
        score_likelihoods(np.ones(K),np.ones(K),[-2.,-0.1,-0.4],0.,0.,2)


def test_negative_variance_rejected():
    with pytest.raises(ValueError,match="nonnegative"):
        score_likelihoods(np.ones(K),np.ones(K),[-2.,-0.1,-0.4],-1e-5,0.001,2)
