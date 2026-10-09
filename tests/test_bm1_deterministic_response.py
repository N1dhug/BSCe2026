"""G response and physical-hour energy conservation."""
import unittest
import numpy as np
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

if __name__=='__main__':unittest.main()
