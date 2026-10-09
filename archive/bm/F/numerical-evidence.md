# F: numeriske fejl, efterkontrol og scopegrænser

<a id="h2-terminal-tilt-fine-reference"></a>
<a id="h2-terminal-tilt-cross-control"></a>
<a id="h2-terminal-bank-fine-result-20261007"></a>
<a id="h2-search-next-handoff"></a>
<a id="h2-search-feasibility"></a>
<a id="h2-precise-batch"></a>
<a id="h2-native-v2-next-review"></a>
<a id="h2-native-pc-review"></a>
<a id="h2-native-development"></a>
<a id="h2-compact-rate-review"></a>
<a id="h2-compact-rate-pilot"></a>
Dette er en samlet faglig fremstilling af bevaret udviklingsevidens. Tidligere arbejdsordrer og budgetter er historik, ikke nye kørselsordrer. Aktuel brug, accept og opgaver står i [BM-guiden](../../../docs/bm/README.md) og [status](../../../docs/bm/STATUS.txt). Originaler, hashes og genskabelse: [arkivindeks](../INDEX.md).

Historiske statusord som “åben”, “endnu ikke” og “Remaining scope” gælder den pågældende delundersøgelse 5.–8. oktober 2026. De beskriver ikke automatisk den aktuelle F4-kommando eller senere G-timeleverance.

Dette register samler de forskellige fejlmekanismer. Senere lokale passes
ophæver ikke tidligere negative fund eller udvider deres scope. Alle rå
registreringer og budgetter findes uændret i originaltræet/resultatpakkerne.

## Joint accumulator: massebevaring var utilstrækkelig
## Development evidence

The short panel uses 24 constructed hourly observations (two missing), fixed
physical/noise parameters and Beta(2,3). It is not simulated truth or recovery.
Two-/four-piece fits have ceilings 6/8; both are negative budget stops. No
TRAIN winner is selected. Saved runtime was 10.87 s with BLAS=1; daily replays
ranged from 0.23 s (81 cells, h=1/384 day, da=1/128) to 5.97 s
(321,1/1536,1/512). Maximum mass error was below 3.34e-15.

**All initial BM comparisons failed** the predeclared development tolerances:
total log-likelihood 0.05, terminal mean 0.002, terminal mass L1 0.05. Base
score differed by 6.55 and the combined 161-cell replay by 2.18. Mass
conservation does not establish likelihood accuracy; coarse settings cannot
be recommended for historical fitting. The extra bounded refinement uses
321/641/1281 cells, h=1/1536,1/3072,1/6144, da=1/512,1/1024,1/2048, without
another optimizer run. Reports are linked from the H2 contract.

The finer panel took 407.88 s. Daily 321/641/1281-cell replays took
5.49/49.60/352.54 s; maximum mass error was 1.05e-14. Against 1281, total
score errors were 1.368 and 0.416, both above 0.05. Terminal mean/L1 passed
their development tolerances (641: 0.000243 / 0.00897). This is a negative
likelihood-readiness result. The next task is a more efficient accurate
interval kernel/numerical design, not a full historical optimizer run.

An independent three-state tilted-matrix/Fourier Gaussian reference agrees
at quadrature orders 96/160. Joint time/accumulator resolutions 64/256/512
give score errors 0.0151/0.00781/0.00434 and posterior maximum errors
0.0241/0.00699/0.00358. This checks the additive law on a fixed finite state
space, not the continuous-state historical problem. Tests also cover constant
Gaussian evidence, within-hour jumps, missing/gap causality, full/likelihood-only
identity, DST hour-start and energy conversion, and bounded pause/resume.
No historical values were read; no H2/H3/model-selection/forecast acceptance.


## Transform: tidsfejl og conditional cancellation er forskellige fejl
## Isolating the first-order time error

All six balanced implicit-Euler controls now complete. The saved
[balanced panel](../../../files/output/bm_1/runs/estimation/h2_interval_balanced_characteristic_control_20261005.json)
took 453.72 s. Its 1281/2561-cell total-score difference is 0.133850,
above the existing 0.05 limit. At 1281 cells, 64 versus 96 frequencies changes
the score by 1.78e-14. Stable transforms alone do not establish time accuracy.

Two fixed-state controls reuse those results instead of rerunning the panel.
The [time control](../../../files/output/bm_1/runs/estimation/h2_interval_balanced_time_control_20261005.json)
took 306.14 s:

| Comparison | Total log-likelihood difference | Development tolerances |
|---|---:|---|
| 1281 cells, h=1/6144 versus 1/12288 | 0.112179 | Fail |
| 1281 cells, h=1/12288 versus 1/24576 | 0.056413 | Fail |
| 1281/2561 cells at the same h=1/12288 | 0.021671 | Pass |

The last spatial reference uses 96 frequencies rather than 64; the independent
frequency difference above is recorded. These results identify time
integration as the dominant unresolved error on this fixed case.

All five wide-noise comparisons pass:

| Comparison | Score difference | Terminal mean difference | Terminal mass L1 |
|---|---:|---:|---:|
| Fixed state, time | 0.000004166 | 1.30e-10 | 3.89e-7 |
| Fixed time, state | 0.002234488 | 3.19e-6 | 0.001858843 |
| Combined | 0.002238654 | 3.19e-6 | 0.001858841 |
| Frequencies 48/96 | 0 | 0 | 0 |
| Periods 3/4, frequencies 48/64 | 2.19e-14 | 0 | 2.41e-15 |

Maximum mass error 1.74e-13 and relative-series guard 3.47e-14. The candidate
took 65.78 s; all five replays 574.48 s.
[Raw wide-noise control](../../../files/output/bm_1/runs/estimation/h2_interval_four_wide_control_20261005.json).

All five narrow-noise replays abort in the first hour before any observation
update: conditional densities have 184–304 negative cells and minima about
-1e-14 to -1e-13. Finer grids, doubled frequencies and the larger period do not
resolve it. The panel took 38.89 s. No failed comparison is relabeled a pass.
[Raw narrow-noise control](../../../files/output/bm_1/runs/estimation/h2_interval_two_narrow_control_20261005.json).

An analytic three-state witness with zero dynamics confirms exact Gaussian
conditional tails about 3.28e-27 while the integrated Fourier densities are
negative about -6.66e-15. Direct closed-form phases with 128 frequencies and
period 2 remove state/time error and reduce the truncation bound to 9.94e-58;
computed tails are positive about 1e-15 and inaccurate. The existing relative
guard rejects that weighted case (0.01397 > 1e-8). Log-amplitude storage solves
underflow, not conditional Fourier cancellation; increasing frequencies alone
does not fix floating-point cancellation.
The [integrated witness](../../../files/output/bm_1/runs/estimation/h2_interval_cancellation_witness_20261005.json)
and [exact-phase witness](../../../files/output/bm_1/runs/estimation/h2_interval_exact_phase_cancellation_20261005.json)
are preserved.

## Terminal-tilt: lokal kontrol og finere reference
## Measured results, 5 October 2026

All results below use BLAS=1 and the frozen experiment source. The three new
focused unit tests passed: exact Gaussian/recursive-tail/disconnected-zero
behavior, positive one-jump integration with time refinement, and explicit
failure without prior mutation when the contour/frequency budget is inadequate.
No existing numerical kernel changed. No full suite, old accepted panel,
Windows run, historical fit or LaTeX build was repeated.

| Control | Observed error or result | Requirement |
|---|---:|---:|
| Original exact Gaussian witness, log score | 4.44e-16 | <2e-8 |
| Original witness, maximum log-posterior error | 1.21e-10 | <2e-8 |
| Two recursive below-float updates, maximum log-posterior error | 2.06e-10 | <2e-8 |
| Positive one-jump reference, fine log-joint error | 9.37e-7 | <2e-4 |
| One-jump coarse/fine error ratio | 4.00 | >3.5 |
| First BM hour, score / final mean / mass L1 | 2.51e-6 / 5.35e-7 / .00216338 | .05 / .002 / .05 |
| Full BM day, score / final mean / mass L1 | **.08754758** / 3.993e-5 / .00256929 | **.05** / .002 / .05 |

The exact-reference report took 0.462 s and the paired first hour 40.69 s.
Both full-day replays completed 24 intervals, including the original two
missing observations (indices 7 and 19). The coarse/fine replay times were
225.70/630.87 s; total numerical work was 856.57 s. Their log likelihoods were
-95.291376353943 and -95.378923934561. Each observed interval used two or
three contours. The largest selected relative conditional guard was 2.10e-9
(limit 1e-8), largest resolved overlap discrepancy 2.93e-9 (limit 1e-7),
and largest mass discrepancy 1.39e-13. Finite log-posterior values down to
-5065.72 survived; no tail clipping or probability floor was used.

**The full-day combined time/state precision gate fails.** Successful
completion resolves the saved cancellation witness on this fixture; it does
not establish continuous-state likelihood precision or fit readiness. The
largest signed per-hour coarse-minus-fine score contributions occur at indices
13 (.02221122) and 8 (.02148109). These include propagated prior differences;
they do not isolate a local error or distinguish time from space.

The actual day run was deliberately paused after its first coarse interval
(exit 3), then resumed with unchanged source, inputs and environment. The
completed report records `resumed_intervals=1`. Its saved first log posterior
and score are bitwise identical to the separate first-hour control. All 48
completed interval units retain their log states in the final report; the
failed precision gate returns exit 2 with `status=completed`, not a crash.

## Results, 5 October 2026

Only B was run: 24 intervals, including the original missing observations
at indices 7 and 19. A and C were reused; zero reference intervals were
recomputed. B's total log likelihood is -95.293536448336. Its replay took
431.77 s, and the complete invocation 433.43 s (BLAS=1). No resume was needed.

| Comparison | Absolute score difference | Final mean difference | Final mass L1 | Pass |
|---|---:|---:|---:|---|
| A/B: time at 1281 cells | .0021600944 | 8.778e-7 | 3.216e-5 | Yes |
| B/C: space at h=1/24576 | **.0853874862** | 3.905e-5 | .00255857 | **No** |
| Saved A/C: combined | **.0875475806** | 3.993e-5 | .00256929 | **No** |

The limits remain .05/.002/.05. Both signed score differences are negative,
so .0021600944+.0853874862=.0875475806: approximately 97.5% of the observed
A/C change belongs to the fixed-time spatial comparison. This diagnoses the
dominant difference between these settings; it is not a percentage breakdown
of the unknown continuous-model error. The finer time step alone cannot
resolve the failed score gate.

All intervals passed the terminal-contour diagnostics. Maximum selected
conditional guard was 2.075e-9 (limit 1e-8), maximum resolved overlap
discrepancy 3.670e-9 (limit 1e-7), and maximum mass discrepancy 9.015e-14.
Finite log-posterior values down to -3581.97 were retained. No clipping/floor
or observation/model change was used. The completed negative precision
result returns exit 2; `numerical_fit_readiness` remains false.

The focused reuse test passed; no unchanged numerical test
or old panel was repeated. Changed-Python syntax passed locally; the actual
full replay also exercised the new stage, checkpoint writes and final
comparisons. Source remained clean and byte-frozen throughout. After the run,
input-path separators were normalized for Linux/Windows evidence reuse; the
same focused test covers backslash-separated keys. That compatibility-only
guard change does not change any fingerprint value or numerical kernel, and
the numerical experiment source above remains archived. No numerical replay
was repeated.

## Completed result

The complete 24-hour replay passed on 5 October 2026. The original missing
hours, indices 7 and 19, propagated the state without a score contribution.
All 22 observed updates resolved every reachable terminal cell. No floor,
clipping, historical observation or parameter fit was introduced.

| Quantity | Saved C: 1921 cells, h=1/24576 | New D: 2561 cells, h=1/49152 |
|---|---:|---:|
| Total log likelihood | -95.37892393456096 | -95.40947855361053 |
| Replay runtime, seconds | 630.8714348370 | 1503.2969252920 |

| C/D comparison | Absolute difference | Unchanged gate | Result |
|---|---:|---:|---|
| Total log likelihood | .030554619049567577 | .05 | pass |
| Final state mean | .000013957999987379566 | .002 | pass |
| Conservatively transferred final mass L1 | .0011751598181428788 | .05 | pass |

The signed score change D-minus-C is -.030554619049567577. The maximum
state-mean difference over all 24 endpoints is .00012120064806198094; this
is descriptive and does not replace the declared final-mean gate.
Maximum conditional guard is 2.1066210961633687e-9 (<1e-8), maximum resolved
contour overlap log discrepancy 6.75709088682197e-9 (<1e-7), and maximum
absolute mass error 2.282618538629322e-13. Finite log masses down to
-6534.720975137246 were retained and used by subsequent updates.

This passes one joint state/time comparison on one fixed development fixture.
It does not isolate state error at 2561 cells, establish a continuum error
bound or convergence order, validate a parameter domain, or overturn the
older A/C and B/C failures. C can serve as the less costly local development
reference supported by this comparison. It is not a historical-fit default.
`numerical_fit_readiness`, `scientific_acceptance` and
`rigorous_roundoff_certificate` remain false; historical values read: zero.


## Billig surrogate: prefixpass gav ikke døgnaccept
Joint321 implicit Euler stoppede ved time6: posterior-L1 .0503623 mod .05.
CN ved samme accumulatorgrid gav .0503307 og løste ikke problemet. Halveret
accumulatorspacing gav seks timers pass (.04978562 maksimalL1), men den
efterfølgende dag fejlede allerede ved første prisændring; sammenfaldet
identificerer ikke årsagen.
## Final daily result and revised scope

The cheap day continued from the six preserved endpoint rows and computed
18 new intervals in 9.028 seconds. It handled both missing observations and
hourly input changes. The first failed prefix is hour 7, coincident with the
first price change; that coincidence does not identify the cause. All 24
hours completed, so the report retains the complete negative evidence.

| Quantity | Error | Unchanged gate |
|---|---:|---:|
| Final cumulative score | 3.58901219 | .05 |
| Maximum endpoint state mean over the day | .00822776 | .002 |
| Maximum endpoint posterior L1 over the day | .26784242 | .05 |
| Final state mean | .00069221 | .002 |
| Final posterior L1 | .03133849 | .05 |

The final mean/L1 happen to pass while important intermediate states do not.
Do not infer recursive validity from the terminal state alone. Neither the
first-hour nor the six-hour local pass admits this surrogate for historical
search. Numerical/historical fit readiness and H2/H3 acceptance remain false.
Production numerical kernels, historical fitter and G were not modified.

## Native kernel: ækvivalens og throughput er forskellige evidenser
## Evidence to reuse

| Evidence | Result |
|---|---:|
| First native PC invocation | 118.6677913 s |
| Native v2 PC invocation | 104.3140997 s |
| Invocation reduction | 12.10% |
| Same v2 matched Linux invocation | 80.7241724 s |
| Units / observed updates | 40 / 36 |
| Maximum v2 finite-log-state difference | 9.19499e-10, limit 1e-7 |
| Maximum conditional guard | 2.09603e-9, limit 1e-8 |
| Maximum overlap difference | 1.53580e-8, limit 1e-7 |
| Maximum neutral mass error | 3.33067e-14, limit 1e-10 |
| Warm observed maximum, excluding only first worker-start hour | 4.0682129 s |
| Startup-inclusive observed maximum | 4.7247306 s |

These are separate invocations, not a controlled paired benchmark. The
median end-to-end unit is lower in v2 (2.54215 versus 3.01055 s), but one
later four-piece unit is slower than the first run's warm maximum. Do not
exclude it or substitute a median to manufacture admission. The prescribed
maximum-times-1.5 planning rule still applies; synthetic rates are not
historical rates or rigorous bounds.


De40units havde36 observerede opdateringer og4 missing predictions. Den
første PC-audit måtte rekonstruere CPython3.11's ordnede binary64-sum:
3.12's sum ændrede den gemte score med1.42e-14. Auditoren bevarede original
aritmetik og afviste en forkert ulp; checkpoint-/sourceguards blev ikke lempet.
En warmmaximum4.0682129 s/unit gav15.37784h for alene9072basejobs i det
gamle syvdagesforslag.104.314s for det korte syntetiske panel var ingen
historisk ugeprognose eller adgang til en større kampagne.

## Den faktiske historiske pilot og kandidat28
## Råresultater og provenance

Original PC-source: `2846a5474027101a45b43aa66e9ec111a8727628`.
Råreceipt: `files/output/bm_1/runs/estimation/f4_h2_compact_rates_pc_review_20261007.zip`.
SHA256 `cb7a30a2712116082337248a35ec2644850585caf8b2abc97396a421eb91cffb`.
Aktivt indeks: `f4/h2_compact_rate_reviewed_20261007.json`.

Read-only auditor `f4/audit_h2_compact_rates.py` verificerer alle64 sourcehashes
mod den præcise Git-commit, med kun eksakt Windows LF/CRLF-checkout tilladt.
Manifest/packet/observation/checkpoint-hashes matcher. Frosne audit-bytes
bevares nu korrekt efter PR100. Alle64 raw jobs og fulde egne1921/2561-
logstates har korrekt kronologi, manifestidentitet, videreført prior og
normalisering. Total-score/momenter rekonstrueres; gemte CF'er rekonstruerer
CDF/kvantiler uden native/FV-replay eller ny JIT-propagation. Den samlede
12-jobgraf er prissat ved uafhængig aritmetik, uden at dividere candidates
med8 frekvensworkers. PC-Python3.11 ordered-ledger-sum og prefix matcher.

| Råmål | Største fejl | Uændret gate |
|---|---:|---:|
| Neutral masse | 1.670e-13 | 1e-10 |
| Conditional guard | 1.633e-9 | 1e-8 |
| Contour overlap log | 4.209e-9 | 1e-7 |
| Base/fine state mean | 2.358e-7 | .002 |
| Konservativ state L1 | .000493025 | .05 |
| Total prefix log-score | 3.520e-6 | .05 for hele vinduet |
| Pre-update interval mean | 2.419e-7 | .002 |
| Pre-update interval variance | 3.683e-8 | .002 |
| CDF | 3.766e-7 | .002 |
| Kvantil | .0001220703125 | .002 |

Workbooken med de13 rå husholdningsserier følger ikke med. Dens hash matcher
den auditerede preparation, og decoded72-hour group/mask/normalization er
konsistente med den verificerede runner. Rå household means kan ikke
uafhængigt genberegnes her; provenance-/databegrænsningen er eksplicit bevaret.

## Faktiske rates og samlet budget

| Varm maksimumrate, s/job | Base | Fine |
|---|---:|---:|
| Observeret interval/Bayes-opdatering | 3.865074 | 8.988610 |
| Prediction-only interval | .463259 | 1.185170 |
| Intervalmomenter | .509354 | .796035 |
| CDF/kvantiler | 1.322576 | 2.852026 |

49-kandidatplanens søgearbejde, alle deklarerede base/fine replays, winner+
linear reference, prequential/open-week momenter og CDF-prober, faktor1.5,
600 s imports/JIT/wrappers/export samt60 s stopreserve giver **7.84909 h**
med PC-ledgeren. Efter råreviewets egne charges er overslaget **7.85130 h**.
Det er under10 h, over4–6 h-målet. Beskyttet validation er **11840.47191 s**
(3.289 h); den fælles ledger hæver reserven til dette niveau uden dobbelttælling.
Cold jobs23.98924 s ligger inden for det særskilte300 s auxiliary-cap.
Denne pris er en lineær prefix-ekstrapolation, ikke et nonlinear worst-case
bound eller et løfte om optimizerkonvergens. Ingen pilotarbejde fratrækkes.

Alle PC120.03943 s, forberedende råreview4 s, to afbrudte read-only auditor-
forsøg1.22532/1.29270 s og bestået audit1.44895 s er bevaret. Første auditor
krævede normalisering af Windows ZIP-checkpoint-separatorer. Den anden
brugte for stram platformsammenligning af Beta-CDF-logtal (9e-12 forskel);
initiale Beta-cellemasser kontrolleres nu mod1e-13, derefter bevares de
originale fulde logstates og bitvis state-continuation. Ingen modelgate
eller PC-numerik ændres. Ingen gamle modelchecks eller pilot genkøres.

Ny H-ledger **1671.04085909 s**, native **646.40973845/900 s**, rest
**253.59026155 s** samlet på PC/chat/resumes. Seed er en præcis forlængelse
med38 entries, aldrig et reset. Original PC-ledger og alle reviewcharges er
bevarede. Reviewets source/environment og resultater må ikke ommærkes som
fuldt fit, blind evaluering eller validering af andre response-familier.


### Kandidat 28: egen fine-kontrol
The separate fine control completed in 41.78445 seconds. Candidate 28 used
2561 cells and step 1/65536 with its own Beta prior and 24 prediction-only
hours, followed by observed hour 24. No base prior interpolation was used.
All nine full contours, all 24 prior checkpoints and the posterior are saved.

| Check | Result | Existing limit |
|---|---:|---:|
| Base/fine one-hour log-score difference | 0.0000106323474 | 0.05 |
| Posterior mean difference | 0.00000011589485 | 0.002 |
| Conservative posterior L1 difference | 0.000834816732 | 0.05 |
| Maximum relative conditional guard | 2.38537e-11 | 1e-8 |
| Maximum contour overlap log difference | 1.94623e-8 | 1e-7 |
| Maximum neutral mass error | 3.01981e-14 | 1e-10 |

Fine log evidence: 0.8592776768548576; base: 0.859288309202249.
The original five-center prefix leaves 32 fine cells unresolved. Center
0.875 resolves these, matching the base diagnostic's mechanism. All 2561
active cells pass the original guards. All 69 frozen numerical sources
remain unchanged. The optional extended-bank adapter is still separate.

Read-only review verified every raw SHA256, the exact own-prior chain,
recomputed each conditional candidate from saved characteristic arrays,
and reconstructed the score and full posterior exactly. It used zero native
propagations. Its first attempt had an additional relative-per-cell mass
assertion that was unsuitable for negligible tails: absolute differences
were at most 5.43e-16 and total difference below 9.41e-14. That failure and
its charge are preserved. The corrected review uses the existing total-mass
scale 1e-10; no model or numerical guard was changed.

Campaign after computation, review and ZIP export: 40472.90629690291/64800 s.
Native development: 942.3103597460481/1080 s. This approval used 44.22399 s
of its 180 s scope. All old reservations and charges remain in the ledger.

This validates one failed hour, not optimizer convergence or the selected
family's full TRAIN/selection validation. Original 3655 jobs, 27 completed
four-piece calls and terminal failure markers remain preserved. No fit
restart, model selection, final-Y reading or forecast occurred.

Kandidat28's base-time blev løst med udvidet konturbank, efterfulgt af denne
egen fine-kontrol. Senere kandidat49 nåede budgetloftet uden konvergens;
four-piece-fit blev ikke dermed færdigt. Den fortsatte snævre lineær/tostykke-
scope og700åbne forecastjobs samt345/1344finaljobs er bevaret særskilt i
[slutvalg og H→I/J-kontrakt](h2-selection-and-handoff.md).

## Ingen ny accept af gammel timing
De gamle rates angår deres frosne grids, environment og konkrete jobtyper.
Moderne kommandotid skal måles med import, input, filtering, forecast og export;
en600s pause er ikke en færdig uge. Aktuelle målinger føres kun i status.


## Hvorfor krydskontrol opdeler tids- og rumfejl
For A=(1281,h12288), B=(1281,h24576), C=(1921,h24576) er
$$\Delta_t=\ell_B-\ell_A,\quad\Delta_x=\ell_C-\ell_B,\quad
\ell_C-\ell_A=\Delta_t+\Delta_x.$$
Modsatte fortegn kan skjule fejl i den samlede forskel. Derfor rapporteres
de to komponenter sammen med state-mean og L1; et kombineret pass alene er
ingen isoleret tids-/rumorden eller universel likelihoodgaranti.

## Konkrete kode- og kontrolindgange

[interval reference](../../../src/validation/bm_1/interval_characteristic.py), [characteristic-tests](../../../tests/test_bm1_interval_characteristic.py), [joint-CN-tests](../../../tests/test_bm1_interval_joint_cn.py). Historiske CLI- og controllerkommandoer findes kun i det restaurerede originaltræ; disse links peger på de bevarede matematiske komponenter. Tests er links til tidligere eller løbende kontroller, ikke påstået genkørt ved denne dokumentændring.
