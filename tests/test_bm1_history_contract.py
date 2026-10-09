"""Historical H rejects data leakage, changing populations and invented metadata."""

from dataclasses import replace
import unittest
import numpy as np
import pandas as pd

from src.control.bm_1 import BM1Parameters
from src.params.bm_1_history import BM1ChronologicalSplit, BM1HistoricalCurveContract
from src.params.bm_1_response import BM1PiecewiseResponse
from src.tools.bm_1_data import prepare_historical_curve_training


class HistoryContractTests(unittest.TestCase):
    def contract(self):
        t = pd.Timestamp('2022-10-29', tz='Europe/Copenhagen')
        split = BM1ChronologicalSplit(t, t+pd.DateOffset(days=2), t+pd.DateOffset(days=4), t+pd.DateOffset(days=6))
        return BM1HistoricalCurveContract('declared test population', ('a', 'b'), split,
            'kW', 10., -100., 100., 'DKK/MWh', 'spot_proxy', 'point',
            'synthetic test fixture, not workbook metadata', 'declared history-only baseline', split.selection_start)

    def data(self):
        contract = self.contract()
        times = pd.date_range(contract.split.train_start, contract.split.selection_start, freq='h', inclusive='left')
        readings = pd.DataFrame({'a': np.full(len(times), 4.), 'b': np.full(len(times), 6.)}, index=times)
        inputs = pd.date_range(times[0]-pd.Timedelta(hours=1), times[-1], freq='h')
        baseline = pd.Series(5., index=inputs)
        prices = pd.Series(0., index=inputs)
        curve = BM1PiecewiseResponse.linear()
        parameters = BM1Parameters(curve, curve, 1., 2., .75, .5)
        return contract, readings, baseline, prices, parameters

    def test_contiguous_dst_training_keeps_population_mask_and_physical_time(self):
        contract, data, baseline, price, parameters = self.data()
        data.iloc[5, 0] = np.nan
        data.iloc[8] = 12.  # Gaussian Y is not clipped to the latent demand range.
        record, metadata = prepare_historical_curve_training(data, baseline, price,
            contract=contract, parameters=parameters, sigma_y=.03, initial_mass_builder=np.diff, origin=baseline.index[0])
        self.assertEqual(len(data), 49)  # The autumn day contains 25 elapsed hours.
        np.testing.assert_array_equal(record.observations.times, np.arange(1, 50)/24)
        self.assertFalse(record.observations.observed[5])
        self.assertTrue(np.isnan(record.observations.values[5]))
        self.assertEqual(record.observations.values[8], 1.2)
        self.assertEqual(metadata['midnight_resets'], 0)
        self.assertFalse(metadata['initial_observation_excluded'])
        self.assertEqual(metadata['scored_observations'], 48)
        self.assertEqual(metadata['training_rows'], 49)
        self.assertEqual(metadata['missing_aggregate_rows'], 1)
        self.assertEqual(record.horizon, (0., 50/24))
        self.assertEqual(metadata['terminal_posterior_time'], data.index[-1].isoformat())
        data.iloc[:] = 999.
        self.assertEqual(record.observations.values[0], .5)

    def test_split_is_half_open_and_blindness_requires_exposure_audit(self):
        split = self.contract().split
        times = pd.DatetimeIndex([split.train_start, split.selection_start, split.holdout_start, split.holdout_end])
        np.testing.assert_array_equal(split.role_mask(times, 'TRAIN'), [True, False, False, False])
        with self.assertRaisesRegex(ValueError, 'exposure audit'): split.require_blind_holdout()
        audited = replace(split, holdout_audit_reference='independent synthetic test exposure record')
        audited.require_blind_holdout()
        seen = replace(audited, exposure_intervals=((split.holdout_start, split.holdout_end, 'already plotted'),))
        with self.assertRaisesRegex(ValueError, 'Previously exposed'): seen.require_blind_holdout()
        train_seen = replace(audited, exposure_intervals=((split.train_start, split.selection_start, 'training'),))
        train_seen.require_blind_holdout()
        with self.assertRaises(ValueError): replace(split, selection_start=split.train_start)

    def test_hour_start_energy_preserves_dst_intervals_and_end_posterior(self):
        from src.params.bm_1 import BM1IntervalObservationSeries
        contract, data, baseline, prices, parameters = self.data()
        contract = replace(contract, observation_kind='interval_energy', measurement_unit='kWh',
                           timestamp_label='hour_start', metadata_status='working_assumptions')
        data.iloc[5, 0] = np.nan
        def prepare(c=contract, d=data):
            return prepare_historical_curve_training(d, baseline, prices, contract=c,
                parameters=parameters, sigma_y=.03, initial_mass_builder=np.diff, origin=baseline.index[0])
        record, metadata = prepare()
        self.assertIsInstance(record.observations, BM1IntervalObservationSeries)
        np.testing.assert_array_equal(record.observations.starts, np.arange(1, 50)/24)
        np.testing.assert_array_equal(record.observations.ends, np.arange(2, 51)/24)
        self.assertEqual(record.observations.values[0], .5)
        self.assertFalse(record.observations.observed[5])
        self.assertEqual(metadata['terminal_posterior_time'], contract.split.selection_start.isoformat())
        self.assertEqual(metadata['observation_noise_unit'], 'normalized average power')
        wh_record, _ = prepare(replace(contract, measurement_unit='Wh'), data*1000.)
        np.testing.assert_array_equal(record.observations.values, wh_record.observations.values)
        average, _ = prepare(replace(contract, observation_kind='interval_average', measurement_unit='kW'))
        np.testing.assert_array_equal(record.observations.values, average.observations.values)
        self.assertEqual(contract.normalize_measurements([10.], duration_hours=2.)[0], .5)
        for updates in ({'measurement_unit': None}, {'timestamp_label': 'hour_end'}):
            with self.assertRaises(ValueError): replace(contract, **updates)

    def test_missing_metadata_interval_energy_future_scales_and_clipping_rejected(self):
        contract = self.contract()
        for updates in ({'power_unit': 'kWh'}, {'metadata_reference': ''},
                        {'scale_fitted_before': contract.split.holdout_start},
                        {'measurement_columns': ('a', 'a')}, {'price_role': 'unknown'}):
            with self.assertRaises(ValueError): replace(contract, **updates)
        for kind in ('unknown', 'interval_mean', 'hourly-point-approximation'):
            with self.assertRaises(NotImplementedError): replace(contract, observation_kind=kind)
        with self.assertRaisesRegex(ValueError, 'no clipping'): contract.normalize_prices([0., 101.])
        np.testing.assert_array_equal(contract.normalize_prices([-100., 100.]), [0., 1.])

    def test_first_training_point_is_scored_and_prior_inputs_are_explicit(self):
        contract, data, baseline, prices, parameters = self.data()
        def prepare(origin, b=baseline, p=prices):
            return prepare_historical_curve_training(data, b, p, contract=contract,
                parameters=parameters, sigma_y=.03, initial_mass_builder=np.diff, origin=origin)
        record, _ = prepare(baseline.index[0])
        self.assertEqual(len(record.observations.times), len(data))
        self.assertEqual(record.observations.times[0], 1/24)
        self.assertEqual(record.observations.values[0], .5)
        with self.assertRaisesRegex(ValueError, 'strictly before'): prepare(data.index[0])
        with self.assertRaisesRegex(ValueError, 'hourly input'): prepare(baseline.index[0]-pd.Timedelta(minutes=30))
        with self.assertRaisesRegex(ValueError, 'explicitly cover'): prepare(baseline.index[0], baseline.iloc[1:], prices.iloc[1:])

    def test_whole_declared_training_only_without_silent_subsets_or_population_changes(self):
        contract, data, baseline, prices, parameters = self.data()
        def prepare(d=data, b=baseline, p=prices):
            return prepare_historical_curve_training(d, b, p, contract=contract,
                parameters=parameters, sigma_y=.03, initial_mass_builder=np.diff, origin=baseline.index[0])
        for bad in (data.iloc[:-1], data.iloc[1:], data[['b', 'a']],
                    data.set_axis(data.index.tz_localize(None))):
            with self.assertRaises(ValueError): prepare(d=bad)
        extra = pd.DataFrame({'a': [4.], 'b': [6.]}, index=[contract.split.selection_start])
        with self.assertRaises(ValueError): prepare(d=pd.concat([data, extra]))
        missing = baseline.copy(); missing.iloc[1] = np.nan
        with self.assertRaises(ValueError): prepare(b=missing)
        with self.assertRaises(ValueError): prepare(p=prices.iloc[:-1])


if __name__ == '__main__': unittest.main()
