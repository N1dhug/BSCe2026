# H2 interval characteristic control: development branch

5 October 2026. This follow-up to PR #72 is **not ready as a historical
likelihood backend**. It reuses exactly the saved synthetic one-day inputs;
no workbook or historical measurement is opened, no optimizer is run.
The conservative state/accumulator method remains the main reference.

The experiment propagates the terminal-state characteristic vector
E[exp(iw A) 1(X_end=x)] with the existing real fitted-FV implicit solver.
Real and imaginary right-hand sides are kept separately. Frozen reward phases
are exact; accumulator nodal deposition is absent. State/time approximation
and a truncated periodic Gaussian Fourier sum still require controls.

Plain Fourier inversion fails the declared relative series/roundoff guard
1e-8 at interval 13: the guard is 1.621e-8 for evidence 4.527e-6. No cost
is substituted and no guard is relaxed. Exponential tilting uses the identity

\[
\phi_\sigma(y-A)=e^{-(y-c)^2/(2\sigma^2)}
 e^{(y-c)(A-c)/\sigma^2}\phi_\sigma(c-A).
\]

The kernel evolves the transform with exponent s+iw, s=(y-c)/sigma², and
retains a neutral state column for the ordinary probability/mass check.
This changes the numerical representation, not the observation model.
The scalar Gaussian factor is handled in log space. Alias/truncation bounds
and the explicit heuristic floating-point allowance remain guarded at 1e-8.
The modulus diagnostic includes accumulated subnormal rounding without
altering any coefficient; all negativity guards remain strict.

The registered panel uses 161/321/641/1281/2561 cells, time steps from
1/768 to 1/12288 day, and 64/96 frequencies with period 4. Runtime was
71.31 s. Tilted 161/321-cell replays completed in 0.635/2.507 s, with total
scores -19.578/-21.378. Their difference remains much larger than 0.05.
The finer four runs abort with negative Fourier densities between -5e-324
and -1.5e-323 in small tails. No clipping, truncation or posterior repair is
performed. There is therefore no fine reference, frequency comparison,
accepted speedup or continuous-state/numerical readiness claim.

The report retains every partial score, stopping interval and diagnostic.
Expected numerical guard failures are explicitly reported and the CLI returns
exit status 2. Programming/input errors are not converted into study results.
Four short tests check an independent finite-state matrix-exponential reference,
plain/tilted equivalence, an exact rare constant Gaussian, unresolved series/
negative-density rejection and subnormal transform failure without repair.

Next work: stabilize the transform representation for tiny state tails while
retaining strict positive posterior/mass guards, then complete the fixed
state/time/frequency comparison. This is a focused numerical prerequisite;
it does not justify a large historical optimizer run. Population, split/
exposure and frozen historical inputs also remain prerequisites.

```powershell
python -B -m src.exe.run_model.bm_1_interval_characteristic --development-report files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json --output files/output/bm_1/runs/estimation/interval_characteristic_control.json --blas-threads 1
python -B -m unittest tests.test_bm1_interval_characteristic -q
```

Use a new output path. The development report is saved even when a numerical
guard fails; exit 2 means negative numerical evidence, not successful fitting.
Source and input hashes are checked before publication. Saved evidence:
[characteristic control](../files/output/bm_1/runs/estimation/h2_interval_characteristic_control_20261005.json).
Baseline method and references: [hourly interval model](H2_HOURLY_INTERVALS.md).
