"""One cheap portability check: owned M254 bytes, no transport or filter replay."""
import hashlib
import json
import unittest
import numpy as np
from src.method.bm_1_g8 import G8BankOrigin,RHO_VALUES
from src.method.bm_1_g8_basis import BERNSTEIN_BASIS_KIND as BASIS_KIND,bernstein_basis,prior_coefficients
from src.method.bm_1_distribution_panel import build_problem


class OwnedBasisReloadTests(unittest.TestCase):
    def test_verified_reload_preserves_bytes_and_rejects_changed_family(self):
        edges=np.linspace(0.,1.,258)
        basis=bernstein_basis(edges)
        flat=np.argmax(basis)
        # Stand-in for platform-dependent last-bit special-function evaluation.
        basis.flat[flat]=np.nextafter(basis.flat[flat],np.inf)
        c=np.tile(prior_coefficients(),(5,1))
        fixed=build_problem().parameters
        record=dict(schema='g8-rho-origin-v1',observation_kind='point',time=12.,time_unit='hour',
            price_side='right',backend='mixture',basis_kind=BASIS_KIND,
            parameters=dict(C=fixed.C,k=fixed.k,Xi=fixed.Xi,sigma_x=fixed.sigma_x,
                f=fixed.state_response.to_record(),g=fixed.price_response.to_record()),
            sigma_y=.04,rho_values=RHO_VALUES.tolist(),log_weights=np.log(np.full(5,.2)).tolist(),
            x_edges=edges.tolist(),conditional_mass=(c@basis*np.diff(edges)).tolist(),
            conditional_coefficients=c.tolist(),basis=basis.tolist())
        with self.assertRaises(ValueError):G8BankOrigin.from_record(record)
        payload=json.dumps(record).encode();fingerprint=hashlib.sha256(payload).hexdigest()
        origin=G8BankOrigin.from_verified_bytes(payload,fingerprint)
        np.testing.assert_array_equal(origin.basis,basis)
        np.testing.assert_array_equal(origin.conditional_coefficients,c)
        restored=G8BankOrigin.from_record(origin.to_record())
        np.testing.assert_array_equal(restored.basis,basis)
        with self.assertRaises(ValueError):G8BankOrigin.from_verified_bytes(payload,'0'*64)
        altered=origin.to_record();altered['basis'][0][0]+=1e-4
        with self.assertRaises(ValueError):G8BankOrigin.from_record(altered)
        # Even a newly declared record hash cannot certify a different family.
        bad=json.dumps(altered).encode()
        with self.assertRaises(ValueError):G8BankOrigin.from_verified_bytes(bad,hashlib.sha256(bad).hexdigest())


if __name__=='__main__':unittest.main()
