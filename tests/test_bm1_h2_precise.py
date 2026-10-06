"""Concrete new risks: log-state continuity, candidate isolation and crash budget."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.method.bm_1_interval_precise import (
    candidate_panel, digest, replay_precise_candidate, validate_checkpoint)


class PreciseH2Tests(unittest.TestCase):
    def test_exact_scalar_evidence_with_missing_gap_midnight_and_subfloat_state_resume(self):
        # Xi=0 and sigma_x=0: frozen state and state-independent hourly reward.
        # Gaussian evidence is exact; no FV reference is needed. Gaps cross
        # midnight, the missing observation advances time without evidence.
        edges = (np.arange(50)/24).tolist()
        baseline = [.3]*24+[.8]*24+[.6]
        spec = dict(record_id='exact', horizon=[0., 49/24], input_edges=edges,
            baseline=baseline, price=[.5]*49,
            physical=dict(C=.15, k=3., Xi=0., sigma_x=0., sigma_y=.2),
            starts=[23/24, 1., 2.], ends=[1., 25/24, 49/24],
            values=[.7, None, .8], observed=[True, False, True],
            initial_log_mass=[0., -900., -950.], time_unit='day', midnight_resets=0,
            settings=dict(cells=3, max_step=1/16, frequencies=64, span=4.,
                centers=[.5, .25, .75, 0., 1.], relative_guard=1e-8, overlap_log_tolerance=1e-7))
        candidate = candidate_panel()[0]
        prefix = replay_precise_candidate(spec, candidate, limit_units=24)
        self.assertEqual(prefix['payload']['status'], 'prefix_complete')
        self.assertEqual(sum(r['observed'] for r in prefix['payload']['rows']), 1)
        resumed = replay_precise_candidate(spec, candidate, saved=prefix)
        uninterrupted = replay_precise_candidate(spec, candidate)
        expected = sum(-np.log(.2*np.sqrt(2*np.pi))-.5*((y-b)/.2)**2
                       for y, b in ((.7, .3), (.8, .6)))
        self.assertAlmostEqual(resumed['payload']['log_likelihood'], expected, delta=2e-8)
        self.assertEqual(resumed['payload']['log_mass'], uninterrupted['payload']['log_mass'])
        np.testing.assert_allclose(resumed['payload']['log_mass'], [0., -900., -950.], atol=2e-8, rtol=0.)
        rows = resumed['payload']['rows']
        self.assertEqual(len(rows), 49)
        self.assertIsNone(rows[24]['log_evidence'])
        self.assertEqual(rows[24]['observation_index'], 1)
        # Same-shaped alternative candidate must never inherit this posterior.
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_checkpoint(prefix, spec, candidate_panel()[1])
        corrupt = deepcopy(prefix); corrupt['payload']['log_mass'][1] = None
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_checkpoint(corrupt, spec, candidate)
        changed = deepcopy(spec); changed['values'][0] += .01
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_checkpoint(prefix, changed, candidate)

    def test_crashed_reservation_stays_charged_and_validation_reserve_cannot_be_spent(self):
        from src.exe.run_model.bm_1_h2_precise import campaign_reservation, PRIOR_COMPUTE_SECONDS
        from src.tools.bm_1_reports import _atomic_report
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'campaign.json'
            # Represents an abruptly lost invocation: no orderly refund exists.
            ledger = dict(schema='f4-h2-campaign-budget-v1', ceiling_seconds=36000.,
                target_seconds=21600., validation_reserve_seconds=7200.,
                charged_seconds=PRIOR_COMPUTE_SECONDS+900.,
                entries=[dict(label='lost-parent', status='reserved', charged_seconds=900.)])
            _atomic_report(path, ledger)
            with campaign_reservation(path, 15., 'resume') as reservation:
                self.assertEqual(reservation['before'], PRIOR_COMPUTE_SECONDS+900.)
                inside = json.loads(path.read_text())
                self.assertEqual(inside['charged_seconds'], PRIOR_COMPUTE_SECONDS+915.)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['entries'][0]['charged_seconds'], 900.)
            self.assertGreaterEqual(saved['charged_seconds'], PRIOR_COMPUTE_SECONDS+900.)
            saved['entries'].append(dict(label='spent', status='closed',
                charged_seconds=28790.-saved['charged_seconds']))
            saved['charged_seconds'] = PRIOR_COMPUTE_SECONDS+sum(e['charged_seconds'] for e in saved['entries'])
            _atomic_report(path, saved, replace_existing=True)
            with self.assertRaisesRegex(ValueError, 'Remaining H budget'):
                with campaign_reservation(path, 15., 'extra'): pass


if __name__ == '__main__':
    unittest.main()
