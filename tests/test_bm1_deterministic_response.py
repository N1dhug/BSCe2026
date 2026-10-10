"""G response and physical-hour energy conservation."""
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from src.method.bm_1_deterministic_fit import BM1DeterministicInputs
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.params.bm_1_response import BM1PiecewiseResponse

class ResponseNumericsTests(unittest.TestCase):
    def test_linear_delegation_is_exact_and_hourly_mean_balances_energy(self):
        time = np.linspace(0,24,289); baseline = .4+.1*np.sin(time/3)
        price = .5+.3*np.cos(np.arange(24)/2); p = [2.,2.,.5,.5]
        for contract in ('hourly-point','hourly-mean'):
            original = BM1DeterministicInputs(time,baseline,price,observation_contract=contract)
            adapter = BM1ResponseInputs(time,baseline,price,observation_contract=contract)
            before, after = original.solve(p,keep_fine=True), adapter.solve(p,keep_fine=True)
            for name in ('state','demand','fine_state','fine_demand'): np.testing.assert_array_equal(before[name],after[name])
        g = BM1PiecewiseResponse.from_inner_values([.35],pieces=2)
        adapter.price_response = g
        curved = adapter.solve(p)
        np.testing.assert_allclose(curved['demand'],adapter.b_mean+p[0]*np.diff(curved['state']),rtol=0,atol=1e-15)

    def test_five_piece_state_and_price_response_match_independent_integration(self):
        f = BM1PiecewiseResponse.from_inner_values([.85,.4,-.35,-.8], pieces=5)
        g = BM1PiecewiseResponse.from_inner_values([.75,.3,-.2,-.9], pieces=5)
        prices = np.array([.05,.95,.1,.8,.2,.7,.3,.9])
        p = [1.5,2.,.5,.45]
        inputs = BM1ResponseInputs(np.arange(9.),np.full(9,.5),prices,
                                  observation_contract='hourly-mean',steps_per_hour=120)
        result = inputs.solve(p,state_response=f,price_response=g)
        states, means = [p[3]], []
        for u in prices:
            def rhs(t,y):
                demand = .5+.5*p[2]*np.tanh(.5*p[1]*(float(f(y[0]))+float(g(u))))
                return [(demand-.5)/p[0], demand]
            sol = solve_ivp(rhs,(0,1),[states[-1],0],rtol=1e-10,atol=1e-12,max_step=.03)
            states.append(sol.y[0,-1]); means.append(sol.y[1,-1])
        np.testing.assert_allclose(result['state'],states,atol=2e-6,rtol=0)
        np.testing.assert_allclose(result['demand'],means,atol=2e-6,rtol=0)

if __name__=='__main__':unittest.main()
