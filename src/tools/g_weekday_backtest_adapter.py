"""Existing bounded G/WM weekday mathematics behind the shared backtest API."""
from src.method import g_weekday_rolling as model
from src.method import g_weekday_export as numerical
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast


class GWeekdayAdapter:
    name = 'G/WM'
    warmup_hours = 24
    input_columns = ('Price', 'Temp')
    target_column = 'Demand'
    prediction_columns = {'G':'g_prediction_output_mean', 'WM':'wm_prediction_output_mean', 'baseline':'baseline'}

    @property
    def metadata(self):
        return {'fit_budget':{'per_weekday':{'G_seconds':30, 'G_calls':750, 'WM_seconds':10, 'WM_calls':400},
                              'numerical_export_seconds':15},
                'initialization_policy':'existing G input-only warmup / WM variable projection; separate weekday median states',
                'population':'caller MED mean over available measured meters',
                'target_unit':'raw average-power development units; physical unit unverified',
                'source_sha256':numerical.source_hashes(), 'baseline_policy':'weekday',
                'uncertainty_policy':'existing separate weekday empirical residuals; no automatic band recalibration',
                'input_scenario_policy':'caller supplied Price/Temp; historical realized inputs are conditional scenarios'}

    def forecast_week(self, data, directory):
        if data.window.zone != model.interval.ZONE or data.history.index.tz is not None or data.inputs.index.tz is not None:
            raise ValueError('Existing G adapter requires unique local-naive Copenhagen labels; autumn ambiguity unsupported.')

        def checkpoint(record):
            _atomic_report(directory/f"weekday_{record['weekday']}_{record['stage']}.json",
                           model.interval.point._plain(record), replace_existing=True)

        fitted = model.fit_rolling(data.history, data.window.origin, baseline_policy='weekday', checkpoint=checkpoint)
        _atomic_report(directory/'fit_report.json', model.interval.point._plain(
            {k:v for k,v in fitted.items() if k not in ('snapshot', 'historical_replay')}))
        if fitted['status'] != 'complete':
            raise RuntimeError('Incomplete G/WM weekday fit; checkpoints retained, no automatic retry or extension.')
        _atomic_report(directory/'weekday_snapshot.json', fitted['snapshot'])
        fitted['historical_replay'].to_csv(directory/'historical_replay_normalized.csv', index_label='Datetime')
        exported = numerical.refine_for_export(fitted['snapshot'], data.history)
        _atomic_report(directory/'weekday_export_snapshot.json', exported['snapshot'])
        exported['historical_replay'].to_csv(directory/'export_replay_normalized.csv', index_label='Datetime')
        result = numerical.predict_comparison(exported['snapshot'], data.inputs)
        write_forecast(directory, result)
        return result['predictions']


def make_adapter():
    return GWeekdayAdapter()
