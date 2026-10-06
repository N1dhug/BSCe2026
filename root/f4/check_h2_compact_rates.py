"""Focused new pilot orchestration control; no old FV/JIT/forecast replay."""


def control(packet):
    import copy
    import tempfile
    from pathlib import Path
    import numpy as np
    from openpyxl import Workbook
    from root.f4.measure_h2_compact_rates import load_observations, work_graph, sha
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'boundary.xlsx'
        book=Workbook(); sheet=book.active
        sheet.title=packet['policy']['header_evidence']['input_sheet']
        names=packet['policy']['population']['measurement_columns']
        sheet.append([None]*4+[n.rsplit('_',1)[0] for n in names])
        sheet.append(['År','Måned','Dato','Time']+['MED']*13)
        for hour in range(6):
            row=[2022,2,'2022-02-18',f'{hour:02d}:00-']+[float(hour+1)]*13
            if hour%2:
                row[-1]=None
            sheet.append(row)
        # Out-of-window measurement formulas must never be decoded. A future
        # formula would fail the raw decoder if the selection boundary moved.
        sheet.append([2022,2,'2022-02-21','00:00-']+['=1/0']*13)
        book.save(path)
        example=copy.deepcopy(packet); example['input_sha256']['input']=sha(path)
        observed=load_observations(example,path)
        assert observed['train_complete']==3 and observed['selection_complete']==0
        assert observed['selected_train_indices']==[0,2,4] and observed['pilot_end_hour']==29
        assert np.isnan(observed['values'][1]) and np.isnan(observed['values'][3])
        assert observed['group_kW'][4]==5. and observed['values'][4]==5./packet['frozen_scales']['power_kW']
        assert observed['final_values_decoded']==observed['prior_values_decoded']==observed['imputed_values']==0
        changed=copy.deepcopy(example); changed['input_sha256']['input']='0'*64
        try:
            load_observations(changed,path)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed workbook was accepted.')
    rates=dict(base=dict(observed=1.,prediction=.1,moments=.3,cdf=.5),
        fine=dict(observed=2.,prediction=.2,moments=.4,cdf=.6))
    graph=work_graph(rates,48,24,100.)
    # Independent sum of approved graph units, all observed; no /8 factor.
    expected=1.5*(2760.+552.*2.+1176.*.1+72.*.2+744.*(.3+.4)+14.*(.5+.6))
    assert abs(graph['projected_total_seconds']-(100.+expected+600.+60.))<1e-10
    sparse=work_graph(rates,3,0,100.)
    assert sparse['jobs'][1]['base_units']==49*45 and sparse['jobs'][6]['fine_units']==3*69
    assert graph['runtime_admitted'] is False and graph['historical_fit_ready'] is False
    assert graph['protected_validation_seconds']>=7200.
    try:
        work_graph(dict(base={'observed':1.},fine=rates['fine']),48,24,100.)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete rate types were accepted.')
    return dict(status='completed',passed=True,raw_boundary_and_missing_population=True,
        changed_workbook_rejected=True,chronological_first_three_complete_hours=True,
        independent_graph_arithmetic=True,missing_hours_priced=True,
        full_fit_admission_rejected=True,model_propagations=0)
