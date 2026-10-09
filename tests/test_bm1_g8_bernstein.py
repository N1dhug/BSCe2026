"""Only new risks: exact prior coordinates and typed own-state continuation."""
import json
import unittest
import numpy as np
from src.method.bm_1_g8 import context, bank_filter, G8BankOrigin
from src.method.bm_1_g8 import BERNSTEIN_BASIS_KIND as BASIS_KIND, prior_coefficients
from src.params.bm_1 import BM1ObservationSeries

class BernsteinTests(unittest.TestCase):
    def test_prior_and_typed_checkpoint(self):
        _,fixed,_,edges,prior,p=context(257,basis_kind=BASIS_KIND)
        c=prior_coefficients()
        self.assertEqual(p.basis.shape,(255,257))
        self.assertEqual(np.count_nonzero(c),217)
        self.assertGreaterEqual(c.min(),0.)
        np.testing.assert_allclose(p.basis @ np.diff(edges),1.,atol=2e-14,rtol=0.)
        np.testing.assert_allclose(c @ p.basis*np.diff(edges),prior,atol=2e-15,rtol=0.)
        origin=G8BankOrigin(0.,fixed,[.25],[0.],edges,[prior],backend='mixture',
                           conditional_coefficients=[c],basis=p.basis,basis_kind=BASIS_KIND)
        record=origin.to_record()
        restored=G8BankOrigin.from_record(json.loads(json.dumps(record)))
        np.testing.assert_array_equal(restored.conditional_coefficients,origin.conditional_coefficients)
        record['basis_kind']='other'
        with self.assertRaises(ValueError):G8BankOrigin.from_record(record)
        record['basis_kind']=None
        untyped=G8BankOrigin.from_record(record)
        tail=BM1ObservationSeries([.0002],[np.nan],'hour',[False])
        with self.assertRaises(ValueError):bank_filter(257,tail,1/96,nodes=[.25],basis_kind=BASIS_KIND,origin=untyped)
        with self.assertRaises(ValueError):context(257,254,basis_kind=BASIS_KIND)

    def test_actual_json_resume(self):
        obs=BM1ObservationSeries([.0001,.0002],[.54,np.nan],'hour',[True,False])
        full,_,_,_=bank_filter(257,obs,1/96,nodes=[.25],basis_kind=BASIS_KIND)
        restored=G8BankOrigin.from_record(json.loads(json.dumps(full[0]['origin'].to_record())))
        tail=BM1ObservationSeries([.0002],[np.nan],'hour',[False])
        resumed,_,_,_=bank_filter(257,tail,1/96,nodes=[.25],basis_kind=BASIS_KIND,origin=restored)
        for field in ('conditional_coefficients','conditional_mass','log_weights'):
            np.testing.assert_array_equal(resumed[-1][field],full[-1][field])
        np.testing.assert_array_equal(resumed[-1]['prior_log_weights'],resumed[-1]['log_weights'])
