"""WM statistical feature family fitted to hourly-average design rows.

The robust regression, ridge and bounded baseline continuation are unchanged.
The training design integrates features on the same 60-sample/hour physical
grid used for G's exogenous baseline. No forecast temperature enters training.
"""
from copy import deepcopy
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.output_model import statistical_output
from src.exe.run_model.bm_1.deterministic_strategies import ZONE
from src.tools.bm_1_deterministic_snapshot import restore_baseline

IMPLEMENTATION='statistical_output_hourly_mean_features_v1'


def mean_features(model,index,temperatures):
    source=temperatures.index.tz_localize(ZONE,ambiguous='raise',nonexistent='NaT')
    source=source[~source.isna()]
    values=temperatures.loc[temperatures.index.tz_localize(ZONE,ambiguous='raise',nonexistent='NaT').notna()].to_numpy(float)
    if not len(source) or not np.isfinite(values).all():raise ValueError('Finite historical temperature grid required.')
    start=source[0].tz_convert('UTC')
    source_hours=np.asarray((source.tz_convert('UTC')-start).total_seconds())/3600
    physical=pd.DatetimeIndex(index).tz_localize(ZONE,ambiguous='raise',nonexistent='raise').tz_convert('UTC')
    hours=np.asarray((physical-start).total_seconds())/3600
    grid=(hours[:,None]+np.arange(61)[None,:]/60).reshape(-1)
    # Exact minute timestamps avoid float-hour conversion rounding just below
    # a minute edge. WM's feature function uses integer minutes/seconds.
    stamps=pd.to_datetime(np.repeat(physical.asi8,61)+
        np.tile(np.arange(61,dtype=np.int64)*60_000_000_000,len(index)),utc=True).tz_convert(ZONE)
    temp=np.interp(grid,source_hours,values)
    features=statistical_output._features(model,stamps,temp).reshape(len(index),61,-1)
    return (.5*(features[:,:-1]+features[:,1:])).sum(axis=1)/60


class IntervalStatisticalBaseline(statistical_output):
    def __init__(self,**settings):
        super().__init__(**settings);self._training_temperature=None

    def _features(self,index,temperature):
        if self._training_temperature is not None:return mean_features(self,index,self._training_temperature)
        return super()._features(index,temperature)

    def fit(self,frame,end):
        training=frame.loc[frame.index<pd.Timestamp(end)]
        physical=training.index.tz_localize(ZONE,ambiguous='raise',nonexistent='NaT')
        if np.any(physical.isna()&np.isfinite(training.I.to_numpy(float))):
            raise ValueError('Finite demand at nonexistent training interval start is invalid.')
        self._training_temperature=training.Temp.copy()
        try:super().fit(frame,end)
        finally:self._training_temperature=None
        self.metadata.update(feature_observation_operator='hourly-average trapezoid on 60 samples/hour',
            temperature_policy='physical-time interpolation of historical inputs only; last value held at cutoff',
            clipping_policy='unchanged WM bounded continuation; regression integrates unclipped linear features')
        return self


def restore_interval_baseline(record):
    if (record.get('implementation')!=IMPLEMENTATION or record['metadata'].get('feature_observation_operator')!=
            'hourly-average trapezoid on 60 samples/hour'):
        raise ValueError('Own hourly-mean-feature baseline snapshot required.')
    common=deepcopy(record);common['implementation']='statistical_output'
    validated=restore_baseline(common)
    model=IntervalStatisticalBaseline(**record['settings'])
    for name in ('coef','feature_mean','feature_scale','metadata'):
        setattr(model,name,deepcopy(getattr(validated,name)))
    return model
