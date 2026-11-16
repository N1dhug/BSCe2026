"""Regression tests for ARM historical and target marginal NLL (hours 1..23)."""
import numpy as np
import pandas as pd
import pytest

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import variance_kernel
from src.exe.run_model.wm_4 import selected_sample_nll as script
from src.exe.run_model.wm_4.selected_sample_nll import (
    K,N,DEFAULT_MIN_TAU,positive_variances,score_marginal,
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
    residual[:,0]=10.0  # Hour-0 outliers must not influence the variance QP.
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
    result["tau2"],result["sigma2"]=1.,1.  # Ignore old t=0-inclusive estimates.
    tau2,sigma2,regularized=positive_variances(result,3)
    assert tau2==pytest.approx(0.035**2,abs=1e-10)
    assert sigma2==pytest.approx(0.,abs=1e-10)
    assert not regularized


def test_optional_tau_floor_applied_after_hour_zero_exclusion():
    result=synthetic_history(np.full((N,K),0.01))
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
    target_nll,var=score_marginal(mu+residual[0],mu,result["theta"],tau2,sigma2,sph)
    hist_nll,hist_var=score_marginal(mu+residual,mu+np.zeros((N,K)),
                                     result["theta"],tau2,sigma2,sph)
    assert target_nll.shape==(K-1,)
    assert hist_nll.shape==(N,K-1)
    assert var.shape==hist_var.shape==(K-1,)
    np.testing.assert_allclose(hist_nll[0],target_nll)
    assert (var>0).all()
    assert np.isfinite(np.r_[target_nll,hist_nll.ravel()]).all()


def test_single_day_marginal_matches_normal_logpdf():
    mean=np.linspace(0.3,0.6,K)
    observed=mean+0.02*np.sin(np.arange(K))
    nll,var=score_marginal(observed,mean,[-2.,-0.1,-0.4],0.002,0.,3)
    assert len(nll)==K-1
    np.testing.assert_allclose(var,np.full(K-1,0.002))
    expected=0.5*(np.log(2*np.pi*0.002)+(observed[1:]-mean[1:])**2/0.002)
    np.testing.assert_allclose(nll,expected,rtol=1e-12,atol=1e-12)


def test_initial_observation_excluded_from_historical_and_target_scores():
    mean=np.full(K,0.4)
    observed=mean+0.03*np.cos(np.arange(K))
    first,var=score_marginal(observed,mean,[-2.,-0.1,-0.4],0.,0.005,4)
    changed=observed.copy()
    changed[0]=1000.
    second,second_var=score_marginal(changed,mean,[-2.,-0.1,-0.4],0.,0.005,4)
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(var,second_var)
    hist=np.tile(observed,(N,1))
    hist[:,0]=np.arange(N)*1000.
    historical,_=score_marginal(hist,np.tile(mean,(N,1)),[-2.,-0.1,-0.4],0.,0.005,4)
    np.testing.assert_allclose(historical,np.tile(first,(N,1)))
    assert historical.size==N*(K-1)


def test_both_zero_variances_rejected():
    with pytest.raises(ValueError,match="Zero/nonfinite Gaussian variance"):
        score_marginal(np.ones(K),np.ones(K),[-2.,-0.1,-0.4],0.,0.,2)


def test_negative_variance_rejected():
    with pytest.raises(ValueError,match="nonnegative"):
        score_marginal(np.ones(K),np.ones(K),[-2.,-0.1,-0.4],-1e-5,0.001,2)


def test_report_has_historical_and_target_marginal_only(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(script,"group_frame",lambda group:None)
    history=[pd.Timestamp("2022-01-01")+pd.Timedelta(weeks=j) for j in range(N)]

    def sample(frame,day,steps_per_hour,min_tau):
        result={
            "observed":np.zeros(K),"predicted":np.zeros(K),
            "initial":[0.,0.],"theta":[-2.,-0.1,-0.5]
        }
        return (result,history,{"imputed_hourly_slots":0},
                np.zeros((N,K)),np.zeros((N,K)),
                np.full((N,K-1),-1.),np.full(K-1,-2.),
                np.ones(K-1),0.01,0.04,False)

    monkeypatch.setattr(script,"score_day",sample)
    daily=script.run(2022,4,tmp_path,True,DEFAULT_MIN_TAU)
    output=capsys.readouterr().out
    hourly=pd.read_csv(tmp_path/"hourly_nll.csv")
    historical=pd.read_csv(tmp_path/"historical_hourly_nll.csv")
    assert len(daily)==len(script.DAYS)==12
    assert daily["historical_observations"].eq(N*(K-1)).all()
    assert daily["target_observations"].eq(K-1).all()
    assert daily["historical_fit_marginal_nll"].eq(-N*(K-1)).all()
    assert daily["target_marginal_nll"].eq(-2*(K-1)).all()
    assert len(hourly)==12*(K-1)
    assert len(historical)==12*N*(K-1)
    assert hourly["hour"].min()==historical["hour"].min()==1
    assert hourly["hour"].max()==historical["hour"].max()==23
    assert "sequential" not in output.lower()
    assert "sequential" not in " ".join(daily.columns)
    assert "historical marginal NLL" in output
    assert "target marginal NLL" in output
