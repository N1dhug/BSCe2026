"""H2 Excel boundary: excluded observations cannot affect decoded TRAIN data."""

from dataclasses import replace
from datetime import datetime, time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from openpyxl import Workbook

from src.params.bm_1_history import BM1ChronologicalSplit, BM1HistoricalCurveContract
from src.tools import bm_1_history_io as io


class HistoryWorkbookTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input, self.prices = self.root/'input.xlsx', self.root/'prices.xlsx'
        self.start = pd.Timestamp('2022-02-01T00:00:00+01:00')
        self.end = self.start+pd.Timedelta(hours=2)
        self.origin = self.start-pd.Timedelta(hours=1)
        self.write_power()
        self.write_prices()

    def write_power(self, rows=None):
        book = Workbook()
        sheet = book.active
        sheet.title = io.INPUT_SHEET
        sheet.append([None, None, 'meter', None])
        sheet.append(['Dato', 'Time', 'A', 'B'])
        sheet.merge_cells('C1:D1')
        for row in rows or [
            ['31-01-2022', 23, 'do not decode previous power', '=1/0'],
            ['01-02-2022', 0, 4., 6.],
            [None, 1, None, 8.],
            [None, 2, 'do not decode selection/holdout', '=1/0'],
        ]:
            sheet.append(row)
        book.save(self.input)
        book.close()

    def write_prices(self, rows=None):
        book = Workbook()
        sheet = book.active
        sheet.append(['HourUTC', 'PriceArea', 'SpotPriceDKK'])
        for row in rows or [
            ['2022-01-31T21:00:00', 'DK1', 'do not decode earlier price'],
            ['2022-01-31T22:00:00', 'DK1', -20.],
            ['2022-01-31T23:00:00', 'DK1', 10.],
            ['2022-02-01T00:00:00', 'DK2', '=1/0'],
            ['2022-02-01T00:00:00', 'DK1', 30.],
            ['2022-02-01T01:00:00', 'DK1', 'do not decode later price'],
        ]:
            sheet.append(row)
        book.save(self.prices)
        book.close()

    def contract(self):
        split = BM1ChronologicalSplit(self.start, self.end,
            self.end+pd.Timedelta(days=1), self.end+pd.Timedelta(days=2))
        return BM1HistoricalCurveContract('synthetic fixture', ('meter_A', 'meter_B'), split,
            'kW', 10., -100., 100., 'DKK/MWh', 'spot_proxy', 'point',
            'synthetic fixture metadata', 'caller-supplied history-only baseline')

    def test_header_inspection_never_decodes_measurements(self):
        with patch.object(io, '_numeric', side_effect=AssertionError('measurement decoded')):
            report = io.inspect_history_workbooks(self.input, self.prices)
        self.assertEqual(report['input']['measurement_columns'], ['meter_A', 'meter_B'])
        self.assertEqual(report['prices']['headers'], ['HourUTC', 'PriceArea', 'SpotPriceDKK'])
        self.assertEqual(report['observation_values_read'], 0)
        self.assertFalse(report['historical_fit_ready'])
        self.assertEqual(len(report['input']['sha256']), 64)

    def test_train_boundary_ignores_poisoned_excluded_values_and_keeps_missing(self):
        power, prices, metadata = io.load_historical_training_workbooks(self.input, self.prices,
            contract=self.contract(), origin=self.origin, timezone='Europe/Copenhagen', price_area='DK1')
        np.testing.assert_array_equal(power.iloc[0], [4., 6.])
        self.assertTrue(np.isnan(power.iloc[1, 0]))
        self.assertEqual(power.iloc[1, 1], 8.)
        np.testing.assert_array_equal(prices, [-20., 10., 30.])
        self.assertEqual(power.index[0], self.start)
        self.assertEqual(prices.index[0], self.origin)
        self.assertEqual(metadata['contract']['measurement_columns'], ['meter_A', 'meter_B'])
        # Verify which cells actually reach measurement conversion.
        with patch.object(io, '_numeric', wraps=io._numeric) as decoded:
            io.read_raw_power_window(self.input, columns=('meter_A', 'meter_B'),
                start=self.start, end=self.end, timezone='Europe/Copenhagen')
        self.assertEqual(decoded.call_count, 4)

    def test_absent_rows_remain_missing_no_interpolation_and_contract_not_guessed(self):
        self.write_power([['01-02-2022', 0, 4., 6.]])
        self.write_prices([['2022-01-31T22:00:00', 'DK1', -20.],
                           ['2022-02-01T00:00:00', 'DK1', 30.]])
        power, prices, _ = io.load_historical_training_workbooks(self.input, self.prices,
            contract=self.contract(), origin=self.origin, timezone='Europe/Copenhagen', price_area='DK1')
        self.assertTrue(power.iloc[1].isna().all())
        self.assertTrue(np.isnan(prices.iloc[1]))
        for contract in (replace(self.contract(), price_role='consumer_price'),
                         replace(self.contract(), price_unit='DKK/kWh')):
            with self.assertRaisesRegex(ValueError, 'own input'):
                io.load_historical_training_workbooks(self.input, self.prices,
                    contract=contract, origin=self.origin, timezone='Europe/Copenhagen', price_area='DK1')

    def test_selected_formula_and_invalid_number_rejected_not_filled(self):
        for value in ('=1+1', 'bad', float('inf')):
            self.write_power([['01-02-2022', 0, value, 6.]])
            # openpyxl represents infinity as a missing numeric cell; this is
            # explicitly missing, not a valid value or imputed measurement.
            if value == float('inf'):
                data = io.read_raw_power_window(self.input, columns=('meter_A',),
                    start=self.start, end=self.end, timezone='Europe/Copenhagen')
                self.assertTrue(np.isnan(data.iloc[0, 0]))
            else:
                with self.assertRaises(ValueError):
                    io.read_raw_power_window(self.input, columns=('meter_A',),
                        start=self.start, end=self.end, timezone='Europe/Copenhagen')

    def test_duplicate_and_off_grid_times_fail_and_population_order_is_preserved(self):
        for hours in ((0, 0), (0, .5/24)):
            self.write_power([['01-02-2022', hours[0], 4., 6.],
                              [None, hours[1], 3., 7.]])
            with self.assertRaisesRegex(ValueError, 'duplicate or off-grid'):
                io.read_raw_power_window(self.input, columns=('meter_B', 'meter_A'),
                    start=self.start, end=self.end, timezone='Europe/Copenhagen')
        self.write_power()
        data = io.read_raw_power_window(self.input, columns=('meter_B', 'meter_A'),
            start=self.start, end=self.end, timezone='Europe/Copenhagen')
        np.testing.assert_array_equal(data.iloc[0], [6., 4.])

    def test_repeated_clock_metadata_is_cached_without_caching_observations(self):
        from src.tools.import_data.import_input import input_data
        rows = [['31-01-2022', 1, 'excluded earlier value', '=1/0'],
                ['01-02-2022', 1, 4., 6.],
                ['02-02-2022', 1, 'excluded later value', '=1/0']]
        self.write_power(rows)
        with patch.object(input_data, '_time_delta', wraps=input_data._time_delta) as parsed, \
                patch.object(io, '_numeric', wraps=io._numeric) as decoded:
            data = io.read_raw_power_window(self.input, columns=('meter_B', 'meter_A'),
                start=self.start, end=self.end, timezone='Europe/Copenhagen')
        self.assertEqual(parsed.call_count, 1)
        self.assertEqual(decoded.call_count, 2)
        expected = pd.DataFrame([[np.nan, np.nan], [6., 4.]],
            index=pd.date_range(self.start, self.end, freq='h', inclusive='left'), columns=['meter_B', 'meter_A'])
        pd.testing.assert_frame_equal(data, expected)
        # A second selected occurrence remains a duplicate, even when its
        # clock conversion came from the cache; it is never collapsed.
        self.write_power([*rows[:2], ['01-02-2022', 1, 7., 8.], rows[2]])
        with patch.object(input_data, '_time_delta', wraps=input_data._time_delta) as parsed:
            with self.assertRaisesRegex(ValueError, 'duplicate or off-grid'):
                io.read_raw_power_window(self.input, columns=('meter_B', 'meter_A'),
                    start=self.start, end=self.end, timezone='Europe/Copenhagen')
        self.assertEqual(parsed.call_count, 1)

    def test_ambiguous_demand_dst_is_not_guessed_and_utc_price_dst_is_preserved(self):
        start = pd.Timestamp('2022-10-30T01:00:00+02:00').tz_convert('Europe/Copenhagen')
        end = pd.Timestamp('2022-10-30T04:00:00+01:00').tz_convert('Europe/Copenhagen')
        self.write_power([['30-10-2022', 2, 4., 6.]])
        with self.assertRaises(Exception) as caught:
            io.read_raw_power_window(self.input, columns=('meter_A',),
                start=start, end=end, timezone='Europe/Copenhagen')
        self.assertIn('ambiguous', str(caught.exception).lower())
        self.write_prices([[t.isoformat(), 'DK1', i] for i, t in enumerate(
            pd.date_range(start.tz_convert('UTC'), end.tz_convert('UTC'), freq='h', inclusive='left'))])
        prices = io.read_raw_spot_window(self.prices, start=start, end=end, price_area='DK1')
        self.assertEqual(len(prices), 4)
        self.assertFalse(prices.index.has_duplicates)
        np.testing.assert_array_equal(prices, [0., 1., 2., 3.])

    def test_excel_numeric_dates_and_times_and_metadata_columns_are_explicit(self):
        from openpyxl.utils.datetime import to_excel
        self.write_power([[to_excel(datetime(2022, 2, 1)), 0., 4., 6.],
                          [None, time(1), 3., 7.]])
        data = io.read_raw_power_window(self.input, columns=('meter_A',),
            start=self.start, end=self.end, timezone='Europe/Copenhagen')
        np.testing.assert_array_equal(data['meter_A'], [4., 3.])
        for columns in ('meter_A', ('Dato',), ('missing_meter',)):
            with self.assertRaises(ValueError):
                io.read_raw_power_window(self.input, columns=columns,
                    start=self.start, end=self.end, timezone='Europe/Copenhagen')

    def test_cli_writes_inspection_without_running_scientific_study(self):
        from src.exe.run_model.bm_1 import curves
        output = self.root/'report.json'
        with patch.object(curves, 'run_numerics', side_effect=AssertionError('study started')):
            self.assertEqual(curves.main(['history-inspect', '--data-dir', str(self.root), '--output', str(output)]), 0)
            self.assertEqual(curves.main(['history-inspect', '--data-dir', str(self.root), '--output', str(output)]), 0)
        import json
        self.assertEqual(json.loads(output.read_text())['observation_values_read'], 0)

    def test_raw_loader_handoff_keeps_first_train_observation_and_fixed_population_mask(self):
        from src.control.bm_1 import BM1Parameters
        from src.method.bm_1_response import BM1PiecewiseResponse
        from src.tools.bm_1_data import prepare_historical_curve_training
        contract = self.contract()
        readings, prices, _ = io.load_historical_training_workbooks(self.input, self.prices,
            contract=contract, origin=self.origin, timezone='Europe/Copenhagen', price_area='DK1')
        baseline = pd.Series(5., index=prices.index)
        curve = BM1PiecewiseResponse.linear()
        parameters = BM1Parameters(curve, curve, 1., 2., .75, .5)
        record, metadata = prepare_historical_curve_training(readings, baseline, prices,
            contract=contract, parameters=parameters, sigma_y=.03,
            initial_mass_builder=np.diff, origin=self.origin)
        np.testing.assert_array_equal(record.observations.times, [1/24, 2/24])
        np.testing.assert_array_equal(record.observations.observed, [True, False])
        self.assertEqual(record.observations.values[0], .5)
        self.assertEqual(metadata['scored_observations'], 1)
        self.assertFalse(metadata['initial_observation_excluded'])


if __name__ == '__main__':
    unittest.main()
