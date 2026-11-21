import unittest

import numpy as np

from src.solver.joint_drift_qp import joint_drift_qp


class test_joint_drift_qp(unittest.TestCase):
    def test_recover_known_linear_drift_and_f0(self):
        dt, n = 0.05, 201
        U = np.sin(np.arange(n)*0.17)
        theta = np.array([-0.25, -0.2, -0.8])
        F0, R0 = 0.1, 0.0
        F, R = np.empty(n), np.empty(n)
        F[0], R[0] = F0, R0
        for k in range(n-1):
            F[k+1] = F[k]+dt*R[k]
            R[k+1] = R[k]+dt*(theta[0]*F[k]+theta[1]*U[k]+theta[2]*R[k])
        # Preferred equilibrium bound: |theta_U| <= -theta_F.
        A = np.array([[1, -1, 0], [-1, -1, 1]], dtype=float)
        b = np.zeros(2)
        lo = np.array([-np.inf, -np.inf, -np.inf])
        hi = np.array([-1e-8, 0.0, -1e-8])
        result = joint_drift_qp.solve(R, U, dt, U[:-1, None], R[:-1, None],
                                      (A, b, lo, hi), enforce_path=False, ridge=0.0)
        np.testing.assert_allclose(result["coefficients"], theta, atol=1e-5)
        self.assertAlmostEqual(result["init_cond"][0], F0, places=4)


if __name__ == "__main__": unittest.main()
