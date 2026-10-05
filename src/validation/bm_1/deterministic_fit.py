"""Small reproducible G development inputs; not historical January evidence."""
import numpy as np
import pandas as pd
from src.method.bm_1_deterministic_fit import BM1DeterministicInputs


def synthetic_january():
    hours = 38*24
    time = np.arange(hours*60+1)/60
    baseline = .5+.15*np.sin(2*np.pi*time/24)
    price = np.random.default_rng(43).uniform(.05,.95,hours)
    demand = BM1DeterministicInputs(time,baseline,price,steps_per_hour=60).solve((12.,2.,.6,.4))['demand']
    return pd.DataFrame({'Demand':2+2*demand, 'Price':100+40*price,
                         'Temp':5+np.cos(np.arange(hours)/100)},
                        index=pd.date_range('2022-01-01',periods=hours,freq='h',name='Datetime'))
