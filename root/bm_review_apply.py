"""Temporary hash-guarded BM-only consolidation; removed after application."""
from pathlib import Path
import hashlib

def blob_sha(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

expected_original = {'src/exe/run_model/bm_1/beta_dynamic_reference.py': 'e77c688428cdcc6e7f71ceebcd33ced7b13084fd', 'src/exe/run_model/bm_1/em_convergence.py': '960cde48a8e8afb6b528a67f74b1abe659f34846', 'src/exe/run_model/bm_1/em_price_step.py': 'a374ba580025c2be86de2916fb00102cf1613a45', 'src/exe/run_model/bm_1/em_reference.py': 'f23857ecc27f2940911ee25b8c77be75af21092d', 'src/exe/run_model/bm_1/f2_reference_case.py': '542e48f99399809185325de458389a83eff6ce1b', 'src/exe/run_model/bm_1/fp_price_step.py': 'c7c47c9146f3a9026dfdb904bc3cf693704b13ea', 'src/exe/run_model/bm_1/fp_reference.py': '387d430f3261329d1afb3ca9d7281ee53d118df4', 'src/exe/run_model/bm_1/stochastic_objective_reference.py': '61ede754d29f69a4a1e93fef04f67b5434c91b94', 'src/exe/run_model/bm_1/stochastic_optimization_reference.py': '3643a6556e2258b1403cd53bd2d5545f6b7e8bf6', 'src/method/bm_1_stochastic_objective.py': '859d9bc96a7dedc317cb4c627f6b81ab86c71bbb', 'src/method/bm_1_stochastic_optimization.py': '99a2f352fe70e79e299dee8ab7a3ed745e844d3a'}
for relative, digest in expected_original.items():
    if blob_sha(Path(relative)) != digest:
        raise RuntimeError("Refusing to overwrite a changed input: " + relative)

import ast, textwrap
R=Path.cwd()

def read(p): return (R/p).read_text()
def write(p,s): (R/p).parent.mkdir(parents=True,exist_ok=True); (R/p).write_text(s)
def getnode(s,name): return next(n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def block(s,name):
 n=getnode(s,name); return '\n'.join(s.splitlines()[n.lineno-1:n.end_lineno])+'\n'
def remove(s,names):
 lines=s.splitlines(keepends=True)
 for name in sorted(names,key=lambda x:getnode(s,x).lineno,reverse=True):
  n=getnode(s,name); del lines[n.lineno-1:n.end_lineno]
 return ''.join(lines)

p='src/exe/run_model/bm_1/beta_dynamic_reference.py'; source=read(p)
names=['_projection_matrices','_project_with_warm_start','_positive_explicit_step_limit','_validate_dynamic_controls','_evolve_projected_density']
core='''"""BM-only projected Beta-mixture evolution, shared by experiments and pricing.

The F2-F numerical recursion is unchanged: conservative explicit prediction,
then a simplex-constrained weighted L2 projection after each internal step.
Shared WM solvers are imported, never modified. Output times remain hard
integration boundaries; they are not an independent projection schedule.
"""\n\nimport numpy as np\n\nfrom src.solver.active_set import active_set\n\n\n_NEGATIVE_PREDICTION_TOL = 1.0e-10\n_MASS_TOL = 1.0e-9\n\n\n'''
core+='\n\n'.join(block(source,name).rstrip() for name in names)+'\n'
core=core.replace('def _evolve_projected_density(', 'def evolve_projected_density(')
core=core.replace('    max_step,\n    cfl,\n):','    max_step,\n    cfl,\n    projection_data=None,\n):')
core=core.replace('    density = coefficients @ basis\n\n    matrix, hessian, condition = _projection_matrices(basis, dx)', '''    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite one-dimensional grid.")
    dx = float(dx)
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be finite and positive.")
    if not np.allclose(np.diff(x), dx, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("x must be uniformly spaced with spacing dx.")
    if (basis.ndim != 2 or basis.shape[1] != x.size
            or basis.shape[0] != coefficients.size
            or not np.all(np.isfinite(basis))):
        raise ValueError("basis and coefficients must match the state grid.")
    if (coefficients.ndim != 1 or not np.all(np.isfinite(coefficients))
            or np.min(coefficients) < -_MASS_TOL
            or abs(float(np.sum(coefficients)) - 1.0) > _MASS_TOL):
        raise ValueError("initial_coefficients must satisfy the probability simplex.")
    if np.min(basis) < 0.0 or not np.allclose(
        np.sum(basis, axis=1) * dx, 1.0, rtol=0.0, atol=_MASS_TOL
    ):
        raise ValueError("each basis component must have non-negative unit mass.")
    density = coefficients @ basis

    # An optimizer may reuse these fixed matrices across candidate prices.
    if projection_data is None:
        projection_data = _projection_matrices(basis, dx)
    matrix, hessian, condition = projection_data
    if matrix.shape != basis.T.shape or hessian.shape != (basis.shape[0],) * 2:
        raise ValueError("projection_data does not match the supplied basis.")''')
core=core.replace('np.empty((len(t_eval), len(x)), dtype=float)','np.full((len(t_eval), len(x)), np.nan, dtype=float)')
core=core.replace('np.empty((len(t_eval), basis.shape[0]), dtype=float)','np.full((len(t_eval), basis.shape[0]), np.nan, dtype=float)')
core=core.replace('            coefficients, density = _project_with_warm_start(','''            if raw_mass_error > _MASS_TOL:
                raise RuntimeError("Raw mixture prediction failed probability conservation.")

            coefficients, density = _project_with_warm_start(''')
core=core.replace('            n_steps += 1', '''            if (projected_mass_error > _MASS_TOL
                    or coefficient_sum_error > _MASS_TOL
                    or minimum_coefficient < -_MASS_TOL):
                raise RuntimeError("Projected mixture failed probability constraints.")

            n_steps += 1''')
write('src/method/bm_1_beta_mixture.py',core)
source=remove(source,names)
source=source.replace('from src.solver.active_set import active_set', '''# Compatibility imports retain the F2-F helper names used by existing tests.
from src.method.bm_1_beta_mixture import (
    _projection_matrices,
    _project_with_warm_start,
    _positive_explicit_step_limit,
    _validate_dynamic_controls,
    evolve_projected_density as _evolve_projected_density,
)''')
while '\n\n\n\n' in source: source=source.replace('\n\n\n\n','\n\n\n')
write(p,source)

op='src/method/bm_1_stochastic_optimization.py'; opt=read(op)
objp='src/method/bm_1_stochastic_objective.py'; obj=read(objp)
demand=block(opt,'demand_grid')
obj += '\n\n'+demand
write(objp,obj)
opt=remove(opt,['demand_grid'])
opt=opt.replace('    discrete_demand_moments,','    discrete_demand_moments,\n    demand_grid,')

ep='src/exe/run_model/bm_1/stochastic_optimization_reference.py'; exe=read(ep)
cls=block(exe,'DynamicMixturePriceEvaluator')
cls=cls.replace('cfl=DEFAULT_CFL','cfl=0.40')
cls=cls.replace('        initial_density = support.initial_mass / support.dx', '''        # Basis and weighted Hessian do not depend on candidate prices.
        self.projection_data = _projection_matrices(self.basis, support.dx)
        initial_density = support.initial_mass / support.dx''')
cls=cls.replace('            cfl=self.cfl,\n        )','            cfl=self.cfl,\n            projection_data=self.projection_data,\n        )')
opt += '\n\n'+cls
opt=opt.replace('from scipy.optimize import minimize','from scipy.optimize import minimize\nfrom time import perf_counter\n\nfrom src.method.bm_1_beta_mixture import (\n    _projection_matrices,\n    evolve_projected_density as _evolve_projected_density,\n)')
exe=remove(exe,['DynamicMixturePriceEvaluator'])
exe=exe.replace('    _evolve_projected_density,\n','')
exe=exe.replace('    FVStochasticPriceEvaluator,','    FVStochasticPriceEvaluator,\n    DynamicMixturePriceEvaluator,')
opt=opt.replace('return edges\n','return edges.copy()\n')
opt=opt.replace('self.initial_mass = np.asarray(initial_mass, dtype=float).reshape(-1)','self.initial_mass = np.asarray(initial_mass, dtype=float).reshape(-1).copy()')

opt=opt.replace('    starts: np.ndarray\n','''    starts: np.ndarray
    objective_evaluations: int = 0
    cache_hits: int = 0
    elapsed_seconds: float = 0.0
''')
opt=opt.replace('    options=None,\n):','    options=None,\n    progress=None,\n    cache=True,\n):')
opt=opt.replace('    bounds = [(lower, upper)] * starts.shape[1]\n    results = []\n    for start in starts:', '''    bounds = [(lower, upper)] * starts.shape[1]
    results = []
    memo = {}
    objective_evaluations = 0
    cache_hits = 0
    started = perf_counter()

    def evaluate(candidate):
        nonlocal objective_evaluations, cache_hits
        key = tuple(float(value) for value in candidate)
        if cache and key in memo:
            cache_hits += 1
            return memo[key]
        value = float(objective(candidate))
        if not np.isfinite(value):
            raise RuntimeError("The deterministic price objective returned a non-finite value.")
        objective_evaluations += 1
        if cache:
            memo[key] = value
        if progress is not None and objective_evaluations % 50 == 0:
            progress(f"  {objective_evaluations} objective evaluations; "
                     f"{perf_counter() - started:.1f} s elapsed")
        return value

    for index, start in enumerate(starts, start=1):
        if progress is not None:
            progress(f"Powell start {index}/{len(starts)}: {start}")''')
opt=opt.replace('            lambda candidate: float(objective(candidate)),','            evaluate,')
opt=opt.replace('        results.append(result)\n','''        results.append(result)
        if progress is not None:
            progress(f"  finished: success={result.success}, J={result.fun:.8e}")
''')
opt=opt.replace('        starts=starts.copy(),\n','''        starts=starts.copy(),
        objective_evaluations=objective_evaluations,
        cache_hits=cache_hits,
        elapsed_seconds=perf_counter() - started,
''')
opt=opt.replace('        solver_max_step,\n    ):','        solver_max_step,\n        breakpoints=None,\n    ):')
opt=opt.replace('        self.breakpoints = self.interval_edges[1:-1].copy()', '''        extra_breakpoints = np.asarray(
            [] if breakpoints is None else breakpoints, dtype=float
        )
        if (extra_breakpoints.ndim != 1
                or not np.all(np.isfinite(extra_breakpoints))
                or np.any(extra_breakpoints <= model.T[0])
                or np.any(extra_breakpoints >= model.T[1])):
            raise ValueError("breakpoints must be finite points inside the model interval.")
        self.breakpoints = np.unique(np.concatenate(
            (self.interval_edges[1:-1], extra_breakpoints)
        ))''')
exe=exe.replace('    mc_paths=MC_PATHS,\n):','    mc_paths=MC_PATHS,\n    progress=None,\n):')
exe=exe.replace('    target, target_t, target_demand = build_deterministic_target()', '''    def report(message):
        if progress is not None:
            progress(message)

    report("[1/6] Building deterministic target and FV objective surface")
    target, target_t, target_demand = build_deterministic_target()''')
exe=exe.replace('    fv_optimization = multistart_powell(', '    report("[2/6] Pure-FV multistart Powell optimization")\n    fv_optimization = multistart_powell(')
exe=exe.replace('        fv_starts,\n        options=', '        fv_starts,\n        progress=progress,\n        options=')
exe=exe.replace('    if optimize_mixture:\n', '    report("[3/6] Dynamic-mixture optimizer-model comparison")\n    if optimize_mixture:\n')
exe=exe.replace('            mixture_starts,\n            options=', '            mixture_starts,\n            progress=progress,\n            options=')
exe=exe.replace('    fv_evaluations = {', '    report("[4/6] Reevaluating candidates with pure FV")\n    fv_evaluations = {')
exe=exe.replace('    final_mixture = {}', '    report("[5/6] High-resolution mixture reevaluations")\n    final_mixture = {}')
exe=exe.replace('    mc = None\n', '    report("[6/6] Paired Monte Carlo validation")\n    mc = None\n')
exe=exe.replace('    result = run_optimization_reference()', '    result = run_optimization_reference(\n        progress=lambda message: print(message, flush=True)\n    )')
exe=exe.replace('    print()\n\n    if result["mixture_optimization"] is not None:', '''    successful_costs = [float(item.fun) for item in fv_opt.results if item.success]
    print(f"successful-start objective spread: {np.ptp(successful_costs):.3e}")
    print(f"actual solves/cache hits: {fv_opt.objective_evaluations}/{fv_opt.cache_hits}")
    print()

    if result["mixture_optimization"] is not None:''')
for path,s in [(op,opt),(ep,exe)]:
 while '\n\n\n\n' in s: s=s.replace('\n\n\n\n','\n\n\n')
 write(path,s)

hp='src/exe/run_model/bm_1/stochastic_objective_reference.py'; h=read(hp)
h=remove(h,['_demand_grid'])
h=h.replace('    discrete_demand_moments,','    discrete_demand_moments,\n    demand_grid as _demand_grid,')
h=h.replace('    ax_cumulative.plot(\n        hours,','    ax_cumulative.plot(\n        24.0 * result["edges"][1:],')
write(hp,h)

for fname,marker in [('em_reference','model, parameters = build_reference_case()'),('fp_reference','model, parameters = build_reference_case()'),('em_convergence','method = euler_maruyama()'),('em_price_step','# Configure the time grid'),('fp_price_step','# Configure the time grid')]:
 path=f'src/exe/run_model/bm_1/{fname}.py'; s=read(path); i=s.index(marker)
 s=s[:i]+ 'def main():\n    """Run the reference experiment and save its diagnostic figure."""\n'+textwrap.indent(s[i:],'    ').rstrip()+'\n\n\nif __name__ == "__main__":\n    main()\n'
 write(path,s)

p='src/exe/run_model/bm_1/f2_reference_case.py'; s=read(p)
s=s.replace('    if n_paths < 1:', '''    if float(n_paths) != float(original_n_paths):
        raise ValueError("n_paths must be an integer, not a fractional value.")
    if n_paths < 1:''')
s=s.replace('    try:\n        n_paths = int(n_paths)', '    original_n_paths = n_paths\n    try:\n        n_paths = int(n_paths)')
s=s.replace('if not np.isclose(edges[0], 0.0) or not np.isclose(edges[-1], 1.0):','if edges[0] != 0.0 or edges[-1] != 1.0:')
write(p,s)

for p in (R/'src').rglob('*.py'): ast.parse(p.read_text())
print('Refactor applied; all Python files parse.')

p=R/'src/method/bm_1_stochastic_optimization.py'
s=p.read_text().replace('from dataclasses import dataclass\n','from dataclasses import dataclass\nfrom time import perf_counter\n').replace('from scipy.optimize import minimize\nfrom time import perf_counter','from scipy.optimize import minimize')
s=s.replace('        self.degree = int(degree)','        self.degree = degree')
s=s.replace('        self.basis = support.model.fokker_planc._normalise_basis_discretely(', '        self.degree = int(degree)  # beta_basis has validated the integer degree.\n        self.basis = support.model.fokker_planc._normalise_basis_discretely(')
s=s.replace('    """Minimize a deterministic price objective from several feasible starts."""', '    """Minimize a deterministic price objective from several feasible starts.\n\n    Exact duplicate price vectors may reuse scalar values within this call.\n    No prices are rounded. Set cache=False for a stateful/non-deterministic\n    objective (such objectives are not the intended reference use). Progress\n    is an optional callback and never changes stopping tolerances.\n    """')
p.write_text(s)
p=R/'src/method/bm_1_stochastic_objective.py'
p.write_text(p.read_text()+'\n\ndef deterministic_midpoint_objective(\n    model, init_cond, parameters, *, objective_max_step,\n    breakpoints=None, solver_max_step=np.inf, rtol=1.0e-6, atol=1.0e-8,\n):\n    """Evaluate deterministic tracking with the same jump-safe quadrature as H.\n\n    This is an opt-in BM-only alternative to the historical F1 sampled-Simpson\n    objective. Supply every known jump in U, B or A through breakpoints. The\n    legacy shared tracking_error function and model.optimize are unchanged.\n    """\n    _, midpoints, widths, output_t = midpoint_quadrature(\n        model.T, breakpoints=breakpoints, max_step=objective_max_step,\n    )\n    result = model.deterministic_solver(\n        init_cond, parameters, t_eval=output_t, breakpoints=breakpoints,\n        rtol=rtol, atol=atol, max_step=solver_max_step,\n    )\n    demand = np.asarray(result[7], dtype=float)[1:-1]\n    target = np.asarray(model.A(midpoints), dtype=float)\n    return stochastic_tracking_objective(widths, demand, demand**2, target)\n')
expected_result = {'src/exe/run_model/bm_1/beta_dynamic_reference.py': '2ecc61138b37c63aed8c77d51d1758689d09e30f', 'src/exe/run_model/bm_1/em_convergence.py': '25c24c250db9e762f964f6af3e314297bc9e2b8f', 'src/exe/run_model/bm_1/em_price_step.py': 'fab5d032519eeffaad55d705769fc6b580b9dfe4', 'src/exe/run_model/bm_1/em_reference.py': 'a34fa60f34b2a9226015c02d760f67e6ce7a227c', 'src/exe/run_model/bm_1/f2_reference_case.py': '8dac71538c1115a82dc96a4d451538e5d87640d6', 'src/exe/run_model/bm_1/fp_price_step.py': '63231bbf5c3a9e40ef4b56e65c829f0e3456876f', 'src/exe/run_model/bm_1/fp_reference.py': '5ea5ec4cd133d1eea634c9049d8a08330ee56924', 'src/exe/run_model/bm_1/stochastic_objective_reference.py': 'a0aaa2b44a0f8cc6826bd42ce5897d395a7142aa', 'src/exe/run_model/bm_1/stochastic_optimization_reference.py': 'd32b432553bd9baa093a30806091edb034136681', 'src/method/bm_1_stochastic_objective.py': 'b22a0531f50ef6ce46e15b695dad5439ac285282', 'src/method/bm_1_stochastic_optimization.py': '60538c5b0f108dd033eb6eb1becc7af91f4fa6ee', 'src/method/bm_1_beta_mixture.py': '5c442a180c4d7fc4c4c3446f695050c2b5b3c83c'}
for relative, digest in expected_result.items():
    if blob_sha(Path(relative)) != digest:
        raise RuntimeError("Generated content failed its reviewed hash: " + relative)
print("All generated BM files match the locally reviewed and tested hashes.")
