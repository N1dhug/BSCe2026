from pathlib import Path
import ast
import hashlib
import json
import sys
import time
import numpy as np

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
from src.model.wm_4 import wm_4
from src.params.wm_4 import constant_diffusion_qp
from src.method.bm_1_linear_noise import fit_constant_sigma

started = time.perf_counter()
m = wm_4((0., 1.), 33, lambda t: .5)
r = np.array([-1., -.2, -2.])
indices = np.arange(33)
K = m.process_variance_kernel({'r': r}, indices)
A = np.array([[1., m.dt], [m.dt*r[0], 1.+m.dt*r[2]]])
q_by_step = np.linspace(.1, .4, 33)
P = np.zeros((2, 2))
expected = [0.]
for q in q_by_step[:-1]:
    P = A @ P @ A.T + m.dt*q*np.diag([0., 1.])
    expected.append(P[1, 1])
kernel_error = float(np.max(np.abs(K @ q_by_step - expected)))
assert kernel_error < 1e-14
k = K @ np.ones(33)
target = .23*k
fit = constant_diffusion_qp(m).fit(K, target)
closed = float(k @ target / (k @ k + len(k)*m._FIT_RIDGE/2))
qp_error = abs(fit['q']-closed)
assert qp_error < 1e-9
v0 = np.full(33, .03)
fit_with_initial = constant_diffusion_qp(m).fit(K, target+v0, initial_variance=v0)
assert abs(fit_with_initial['q']-closed) < 1e-9
negative = constant_diffusion_qp(m).fit(K, np.zeros(33), initial_variance=v0)
assert abs(negative['q']) < 1e-9
w = np.array([1., 2.])
s = np.array([1., 1.])
std_fit = fit_constant_sigma(w, s)['sigma_per_sqrt_hour']
variance_fit = float(np.sqrt((w*w) @ (s*s) / ((w*w) @ (w*w))))
assert abs(std_fit-.6) < 1e-14 and abs(std_fit-variance_fit) > .05
source = repo/'src/exe/run_model/wm_4/predict_initial_state_uncertainty.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
loads = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id=='deterministic' and isinstance(n.ctx, ast.Load)]
bindings = [n for n in ast.walk(tree) if (isinstance(n, ast.Name) and n.id=='deterministic' and isinstance(n.ctx, ast.Store)) or (isinstance(n, ast.arg) and n.arg=='deterministic') or (isinstance(n, ast.alias) and (n.asname or n.name)=='deterministic')]
assert loads == [80] and not bindings
paths = ['src/model/wm_4.py','src/params/wm_4.py','src/exe/run_model/wm_4/fit_constant_diffusion_qp.py','src/exe/run_model/wm_4/predict_all_days_uncertainty.py','src/exe/run_model/wm_4/predict_initial_state_uncertainty.py','src/exe/run_model/wm_4/analyze_observation_uncertainty.py','src/exe/run_model/wm_4/predict_daily.py','src/exe/run_model/wm_4/predict_all_days_optimized.py','src/exe/run_model/wm_3/fit_weekly.py','src/method/bm_1_linear_noise.py','src/exe/run_model/bm_1/linear_noise.py']
receipt = {'date':'2026-10-08','base_commit':'a06a2fcd3640b141dc89edf9d07554bc97d3cd33','scope':'small algebra checks and static source review; no historical fit or G stochastic acceptance','checks':{'wm_kernel_matches_direct_covariance_recursion':{'max_abs_error':kernel_error},'wm_qp_matches_scalar_ridge_solution':{'absolute_error':qp_error},'wm_initial_variance_subtracted_in_core':True,'wm_nonnegative_q_boundary':True,'std_fit_is_not_variance_fit':{'sigma_std_fit':std_fit,'sigma_variance_fit':variance_fit},'wm_wrapper_unbound_name_static_only':{'name':'deterministic','line':80,'full_wrapper_executed':False}},'seconds_excluding_imports':time.perf_counter()-started,'sources':{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}}
receipt['setup_correction'] = 'First synthetic r=[-1,-.2,-.5] was correctly rejected by WM domain validation; changed to admissible [-1,-.2,-2]. No production code changed.'
(repo/'root/g_wm_alignment_checks_20261008.json').write_text(json.dumps(receipt, indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt, indent=2))
