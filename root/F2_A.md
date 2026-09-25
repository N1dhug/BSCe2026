# F2-A: stochastic BM1 model contract

F2-A fixes the stochastic model assumptions and repairs the existing demonstration
entry points. It does **not** implement the later EM/FP reference improvements.
The deterministic F1 formulas, optimizer, return tuples and solver defaults are
unchanged. No additional dependency is required.

## 1. Mathematical contract

Keep the existing BM1 demand map `D(X, U, B)` and parameter sequence
`[U, f, g, C, k, Xi]`. For deterministic exogenous baseline `B(t)` and a fixed
price path `U(t)`, the stochastic model is the **Ito** equation

$$dX_t=a(t,X_t)\,dt+b(X_t)\,dW_t,$$
$$a(t,x)=\frac{D(x,U(t),B(t))-B(t)}{C},\qquad b(x)=\sigma_Xx(1-x).$$

`W` is standard Brownian motion: independent increments satisfy
`Delta W ~ N(0, Delta t)`. The diffusion method returns **b**, not `b**2` and
not `b*sqrt(Delta t)`. The numerical method supplies the Brownian increment.
There is no factor `C` or `Xi` multiplying/dividing `b` in this model.
In particular `Xi=0` gives `D=B` and zero drift, but not zero process noise.
`stochastic=True` in the legacy `dXdt` signature does not generate Brownian noise.

Assume `C>0`, `k>0`, `0<=Xi<=1`, `sigma_X>=0`, `X0 in [0,1]`, and bounded
inputs `B(t), U(t) in [0,1]`. Use the canonical normalized response functions

$$f(0)=g(0)=1,\qquad f(1)=g(1)=-1,$$

with `f` and `g` nonincreasing and valued in `[-1,1]`. They must be compatible
with scalar and array evaluation in the existing model. The chosen functions
must have enough regularity for a well-posed SDE (in particular, a drift locally
Lipschitz in the state with suitable uniform bounds on the simulation horizon).
Piecewise-continuous deterministic inputs and the existing monotone PCHIP
construction are the intended starting point. With random initial conditions,
the initial state is independent of future Brownian increments.

At the lower boundary, `f(0)+g(U)>=0`, hence `D-B>=0`; at the upper boundary,
`f(1)+g(U)<=0`, hence `D-B<=0`. Thus

$$a(t,0)\geq0,\qquad a(t,1)\leq0,\qquad b(0)=b(1)=0.$$

Under the regularity assumptions these are the inward-drift/vanishing-diffusion
conditions for the **continuous** process to remain in `[0,1]`. They do not
prove that a finite Euler-Maruyama proposal remains in the interval.

### What validation checks (and what it does not)

`bm_validation.response_endpoints(f, g)` evaluates both endpoints once per
stochastic solver call. It checks finite scalar outputs and the canonical
normalization to absolute tolerance `1e-10` (zero relative tolerance). It
returns the original callables: no rescaling, clipping or replacement.
`bm_1_validation.stochastic_inputs` calls this helper for both EM and FP.

The endpoint check does **not** prove monotonicity, regularity or the range
between the endpoints for arbitrary callables. These remain assumptions;
`response_values` still checks ranges at evaluated points. Functions with a
noncanonical endpoint normalization now fail at the stochastic entry even if
they were previously accepted. A different admissible normalization requires
an explicit model-contract decision, not automatic rescaling by a solver.

The deterministic validation route is deliberately unchanged. In particular,
zero-noise calls through the stochastic API still use the F2 endpoint contract.

## 2. Units

`X`, `D`, `B`, `U` and `Xi` are normalized. With a fixed power scale `P_scale`,
`C=C_phys/P_scale` has units of time, `a` has units `time**(-1)` and `sigma_X`
has units `time**(-1/2)`. Use the same time unit throughout each experiment.

For a change from hours to days:

$$t_d=t_h/24,\quad C_d=C_h/24,\quad \sigma_{X,d}=\sqrt{24}\,\sigma_{X,h}.$$

Also transform every time-dependent input, e.g. `B_d(t)=B_h(24*t)` and
`U_d(t)=U_h(24*t)`, and all interval edges. Plotting `24*t` does not itself
change the model's time unit. The two repaired demos retain days as the model
unit, with `C=1` day and `sigma_X=0.5` day**(-1/2); their numerical parameters
have not been changed.

## 3. Integration with the existing layout

| Location | Responsibility in F2-A |
| --- | --- |
| `src/model/base/bm.py` | Existing demand/drift/diffusion; contract docstrings only. |
| `src/validation/model/bm.py` | Shared endpoint-validation helper, reusing existing scalar/callable checks. |
| `src/validation/model/bm_1.py` | One common stochastic entry check for EM and FP. |
| `src/tools/input_generator/input_builder.py` | Existing input builders, reused without modification. |
| `src/exe/run_model/bm_1/em_price_step.py` | Correct imports, interval prices, units, output path and terminal labels. |
| `src/exe/run_model/bm_1/fp_price_step.py` | Same input/path fixes and an explicit warning about the hybrid method. |
| `tests/test_bm_1_f2_sde.py` | Model/validation, zero-noise, one-step and unit-scaling regression tests. |

Both demos use `price_intervals` with two prices and three interval edges over
the full four-day horizon. The baseline is defined over the full horizon too.
No separate price-interpolation or demand implementation has been introduced.
Plots go under `files/output/graphs/bm_1/`, matching the current data layout.

## 4. Run and review

From the repository root, in the existing project environment:

```powershell
python -B -m unittest discover -s tests -p "test_bm_1_f2_sde.py" -v
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.em_price_step
python -B -m src.exe.run_model.bm_1.fp_price_step
```

The full test command is important: it also runs the existing F1 and other
regression tests. The new unit tests do not produce figures. The demo commands
write/replace their named PNG files in the BM1 output folder. The EM demo still
uses 100,000 paths; it is not a lightweight unit test.

Expected demo filenames: `bm_1_em_price_step.png` and `bm_1_fp_price_step.png`.
The terminal time is 96 h, not the old hard-coded 24 h.

### Current limitations are not hidden

- EM still uses the existing `boundary="reflect"` demonstration setting.
  Reflection is a numerical correction, not an invariant-SDE proof. F2-B must
  add raw-proposal diagnostics and controlled internal stepping.
- FP is still finite volume followed by beta projection, with internal clipping
  and normalization. Its printed post-correction mass is not evidence that the
  underlying uncorrected method conserved probability.
- The EM initial state is a point; the FP initializer replaces that point by a
  spread-out beta mixture. The demos do not yet solve the same initial-law
  problem and must not be presented as an MC-versus-FP validation result.
- The current demo price jump is aligned with its time grid. General breakpoint
  handling, independent output/integration grids and coupled Brownian increments
  are F2-B work, not delivered by this patch.
- F2-A tests do not establish strong/weak convergence, a validated FP reference,
  or a validated stochastic optimizer.

## 5. Next steps

**F2-B:** first add supplied Wiener increments, independent integration/output
control, breakpoint splitting and uncorrected boundary diagnostics. Then add
shared initial distributions and reproducible Monte Carlo uncertainty summaries.
Keep model equations in `src/model`, algorithms in `src/method`, validation in
`src/validation`, and experiments under `src/exe/run_model/bm_1`.

**F2-C/D:** extract a pure 1D finite-volume reference and compare it with MC
using the same initial law, inputs and evaluation times. Later isolate the
mixture error and evaluate the expected objective. No solver rewrite or global
renaming of `fokker_planc` is included in F2-A; WM callers are left untouched.

## Sources and convention

Junker et al., *Stochastic nonlinear modelling and application of price-based
energy flexibility*, Applied Energy 275 (2020), 115096, equations (1)--(4),
DOI: 10.1016/j.apenergy.2020.115096. The article's flexibility parameter Delta
is named Xi in this repository.

Higham, *An Algorithmic Introduction to Numerical Simulation of Stochastic
Differential Equations*, SIAM Review 43(3) (2001), 525--546, section 4.
DOI: 10.1137/S0036144500378302.

For later FP work, standard Brownian variance gives
`partial_t p = -partial_x(a*p) + (1/2)*partial_xx(b**2*p)`.
The factor `1/2`, diffusion sign and derivative of the full product must not
be lost when transferring formulas from the project notes.
