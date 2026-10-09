"""Process scheduling preserves raw numerical units and bounded work."""
from time import perf_counter
import unittest
import numpy as np
from src.exe.run_model.bm_1.curves import _NumericalJobs
from src.validation.bm_1.response import run_h1_job

class HParallelTests(unittest.TestCase):
    def specs(self):
        numerics={level:dict(cells=11,max_step=1/24,quadrature_order=1) for level in ('base','time','spatial','quadrature')}
        return [dict(unit=f'no_response_1766_{candidate}',case='no_response',record_index=0,
                     candidate=candidate,numerics=numerics,backend='fitted') for candidate in ('linear','nonlinear')]

    def test_spawn_matches_serial_arrays_exactly(self):
        specs=self.specs();pool=_NumericalJobs(specs,2,perf_counter()+120)
        try:
            for spec in specs:
                parallel=pool.take(spec['unit']);serial=run_h1_job(spec)
                for level in serial:
                    for key in serial[level]:
                        np.testing.assert_array_equal(parallel[level][key],serial[level][key])
            with self.assertRaises(StopIteration):pool.take('no_more')
        finally:pool.close()

    def test_expired_budget_submits_nothing(self):
        pool=_NumericalJobs(self.specs(),2,perf_counter()-1)
        try:
            self.assertEqual(pool.pending,[])
            with self.assertRaises(StopIteration):pool.take(self.specs()[0]['unit'])
        finally:pool.close()

if __name__=='__main__':unittest.main()
