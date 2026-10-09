"""Small deterministic synthetic history shared by active G tests."""
import numpy as np
import pandas as pd

def synthetic_history():
    origin=pd.Timestamp('2022-02-07')
    index=pd.date_range('2022-01-01','2022-02-14',freq='h',inclusive='left')
    t=np.arange(len(index))
    raw=pd.DataFrame({'Demand':2+.3*np.sin(t/5)+.01*np.cos(t/2),
        'Price':100+30*np.sin(t/4),'Temp':5+np.cos(t/8)},index=index)
    best={'parameters':[4.,2.,.6,.5],'g_mid':.2,'converged':True,'mse':.1}
    return raw,origin,best
