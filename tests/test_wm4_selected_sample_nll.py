"""Unit checks for selected ARM/IM comparable NLL evaluation."""
import numpy as np
import pytest

from src.exe.run_model.wm_4.selected_sample_nll import (
    K,N,DEFAULT_MIN_TAU,positive_variances,score_likelihoods,
)


def test_zero_observation_variance_is_refitted_with_im_noise_bound():
    sph=2
    idx=np.arange(K)*sph
    samples,states=[],[]
    for day in range(N):
        observed=np.zeros(K*sph+1)
        prediction=np.zeros(K*sph+1)
        prediction[idx]=-(0.015+0.005*np.cos(np.arange(K)+day))
        samples.append({"I":observed})
        states.append({"I":prediction})
    result={"tau2":0.,"sigma2":0.0004,"theta":np.array([-2.,-0.1,-0.5]),
            "historical_inputs":samples,"historical_states":states}
    tau2,sigma2,regularized=positive_variances(result,sph,DEFAULT_MIN_TAU)
    assert regularized
    assert tau2>=DEFAULT_MIN_TAU**2
    assert sigma2>=0
    assert np.isfinite([tau2,sigma2]).all()
    assert result["tau2"]==0  # Prior fitted values remain unchanged.


def test_positive_variances_preserves_already_admissible_qp():
    result={"tau2":0.002,"sigma2":0.003}
    assert positive_variances(result,2,DEFAULT_MIN_TAU)==(0.002,0.003,False)


def test_marginal_equals_sequential_when_process_noise_vanishes():
    mean=np.linspace(0.3,0.6,K)
    observed=mean+0.02*np.sin(np.arange(K))
    marginal,sequential,var,conditional=score_likelihoods(
        observed,mean,[-2.,-0.1,-0.4],0.002,0.,3)
    np.testing.assert_allclose(sequential,marginal,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(conditional,mean,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(var,np.full(K,0.002))


def test_kalman_innovations_are_finite_with_positive_observation_noise():
    mean=np.full(K,0.4)
    observed=mean+0.03*np.cos(np.arange(K))
    marginal,sequential,var,conditional=score_likelihoods(
        observed,mean,[-2.,-0.1,-0.4],DEFAULT_MIN_TAU**2,0.005,4)
    assert np.isfinite(np.r_[marginal,sequential,var,conditional]).all()
    assert (var>0).all()
    assert sequential[0]==pytest.approx(marginal[0])
    assert not np.allclose(sequential,marginal)


def test_nonpositive_observation_noise_is_rejected():
    with pytest.raises(ValueError,match="strictly positive"):
        score_likelihoods(np.ones(K),np.ones(K),[-2.,-0.1,-0.4],0.,0.001,2)
