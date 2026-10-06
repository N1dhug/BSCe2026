# F4-H2: native interval kernel, 6 October 2026

**Current handoff:** [PC raw review and native v2](H2_NATIVE_PC_REVIEW.md).
The first PC run passes in 118.67 s, but the full 49-pair plan fails the time
gate at that synthetic rate. V2 preserves the method and passes its 40-hour
Linux check in 80.72 s. Use the separate bounded v2 PC command, not the original
completed run. H ledger 1352.86 s; native allowance remaining 571.77 s.
The first implementation/command/budget snapshot below is historical.

The isolated native log-Thomas kernel passes the bounded synthetic equivalence
check. It implements [the Astra decision](H2_METHOD_REDESIGN.md), not historical
fitting or H2/H3 acceptance. G and WM source is unchanged.

`src/method/bm_1_interval_native.py` uses float64/complex128 Numba loops with
`fastmath=False`. Positive explicit CN and implicit Thomas steps retain finite
log-masses, including sub-float tails, and true zeros separately. Existing FV
fields/time grids, held B/U inputs and terminal Fourier inversion are reused.
Rewards retain the original observation duration across input boundaries.
The existing guards and error policy are preserved.

One candidate runs at a time with eight persistent frequency workers, strict
duplicated zero-frequency/neutral/log-amplitude agreement and canonical column
assembly. BLAS/Numba threads are one per process. There are no simultaneous
candidate/day replays, historical reads or optimizer command.

## Evidence actually obtained

The focused independent check covers an asymmetric dense solve, disconnected
true-zero cells, finite sub-float tails revived by Gaussian Bayes updates and
a positive one-jump witness. Dense-transform error is 8.89e-16; finite-tail
error zero; largest Gaussian log-state error 2.09e-10; one-jump log-joint
error 9.38e-7. All declared bounds pass.

Only the new kernel replayed the matched saved references:

| Case | Complete units | Observed | Maximum cumulative score error | Maximum finite log-state error |
|---|---:|---:|---:|---:|
| Narrow two-piece, sigma_y=.04 | 24 | 22 | 3.42e-13 | 9.77e-10 |
| Windows linear prefix | 8 | 7 | 1.46e-13 | 1.45e-10 |
| First Windows four-piece, p4_c0_plus | 8 | 7 | 1.28e-13 | 9.15e-11 |

The equivalence gate remains 1e-7 with identical true-zero support. The narrow
day compares every intermediate full log-state. Windows references contain
only final full states, so their prefixes compare each hourly score/mean and
the final full log-state. Original references were provenance-audited, never
rerun or rewritten. Earlier .05/.002/.05 finer-reference evidence is preserved.

The 40-hour invocation took **104.43 s** in this Linux environment, including
reference audit, worker startup/shutdown and checkpoint I/O. Its per-hour
`elapsed_seconds` ends before checkpoint writing; those fields alone cannot
establish end-to-end admission rates. The final runner separately records
`hour_wall_times.json`, including operator creation, IPC and checkpoint writes.
Timing missing after a crash stays missing and cannot establish fit throughput.
This is not historical throughput or a paired PC speedup measurement.

After adding STOP/rate accounting, a focused process check saved one hour,
stopped unfinished work, resumed with strict source/input/environment identity
and saved the next hour. The first row was bitwise unchanged; both checkpoints
validate. Ledger checks cover newer PC/seed histories, divergent-history
rejection, larger validation reserve, startup charge and cumulative-cap stop.
No broad suite, extra grid panel, Windows run, historical observation read or
LaTeX build was performed.

Raw evidence and exact executed source snapshots for the 40-hour run and the
STOP/resume run are in
`files/output/bm_1/runs/estimation/f4_h2_native_development_20261006.zip`.
The initial runner subsequently changed only for STOP, timing and budgeting;
checked numerical-kernel/control hashes are unchanged. Its old manifest must
not resume under changed source. Archive SHA256:
`104ebffd8d830c3aa58e3888d56a9669ff3853baebe6cd97f84719a25d62ea04`.

## One shared budget

H is **1147.3761251858778 s** charged; native development uses
**122.74500454600275 s**, including all invocations and a conservative 5-second
startup/import allowance before the final full-invocation clock was added.
**777.2549954539973 s** remains of the cumulative native 900-second allowance
across PC/chat/resumes. Each invocation preserves 60 seconds for stopping,
reserves its full grant before work and refunds unused time only on orderly
exit. A crash retains its reservation. Parent/worker deadlines stop incomplete
blocks; STOP is polled while waiting for frequency results.

The new `f4_h2_native_budget_seed_20261006.json` carries the closed ledger to
PC. Under the campaign lock it is adopted only as an exact extension of known
history. Newer PC history is retained; divergent histories block work. The old
468.09-second seed is not reused. Validation reserve stays at least 7200 s and
can increase. Target 4–6 hours and overall 10-hour ceiling remain binding.

## PC command

Wait for other model runs to stop before switching/pulling source. Use the
existing checkout/Python environment, not another worktree. In PowerShell from
the updated repository root:

```powershell
python -m pip install --no-deps numba==0.62.1 llvmlite==0.45.1
if ($LASTEXITCODE -ne 0) { throw 'Numba installation failed.' }
powershell -ExecutionPolicy Bypass -File .\root\f4\run_h2_native.ps1
```

The wrapper checks active model processes/committed source. The runner charges
imports/JIT, rechecks references and performs only the declared synthetic
comparisons. NumPy/SciPy are not upgraded. Default `--seconds 900` is clipped
by both current ledgers; it never grants a fresh 900 s. On this seed the whole
invocation is at most about 13 minutes, including its final minute for stopping.
Return `f4_h2_native_pc_review_20261006.zip`, including the updated ledger.

For an intentionally paused run only, preserve source/environment, remove STOP
when continuation is wanted and use the wrapper's `-Resume` switch. It creates
a separate dated ZIP. Completed/numerically failed runs cannot automatically
repeat. Do not edit hashes/checkpoints/ledgers to create allowance. Pause is
not acceptance.

## Next gate

PC throughput is unmeasured. MED and the three February windows remain fixed.
Remaining F inputs, prior/baseline/scales/noise, final-week exposure and the
complete forecast/validation task graph still need completion. Price actual
lead/base/fine jobs with maximum end-to-end rates and the 1.5 factor from the
redesign. Unknown work is not zero. Synthetic success does not enable COBYQA,
shorten TRAIN, remove a curve family or relax a gate. If the complete job cannot
fit below 10 hours, stop admission and revise the plan before historical fit.
