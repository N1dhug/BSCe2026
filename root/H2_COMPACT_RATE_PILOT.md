# F4-H2: faktisk historisk rate-/præcisionsprøve, 7. oktober 2026

## F4-H2: fejlforsøgets ledger samlet, ny PC-prøve klar, 7. oktober

Den komplette PC-ledger er modtaget som tekst og matcher alle 30 tidligere
entries nøjagtigt; fejlforsøgets 6.964884199900553 s er bevaret som entry31.
Den afsluttede eksportkontrol på 0.3752458700000716 s tilføjes uden genkørsel,
og ledgerens råreview får 2 s konservativ charge. Ny seed er et præcist
prefix-superset med 33 entries: H **1543.03445744 s**, native
**518.40333680/900 s**, rest **381.59666320 s**. Original fejlforsøgsledger,
ny samlet ledger og reconciliation-rapport er bevaret. Windows ordered-sum
og den hidtidige historik er verificeret. Fejl-ZIP er ikke modtaget; ledger-
verifikation er ikke et raw source/checkpoint-review eller numerisk accept.

PR100's afgrænsede -text Git-attributter bevarer frosne audit/packet-bytes
under Windows-eksport; hashgate og numerik er uændrede. Én ny invocation må
måle den oprindeligt planlagte historiske lineære pilot, højst 300 s inkl.
60 s stopreserve, med source-commit der også indeholder denne nye seed.
Launcheren forlænger automatisk PC-ledgerens identiske prefix; ingen lokal
manuel ledgeroverskrivning, ny datapreparation eller gamle modelchecks.
Sol High; 8 frekvensworkers/BLAS1; review-ZIP også ved fejl/stop, ingen auto-
retry. Fit/admission/scientific/blind-H3 flags er fortsat false. G/WM er
uændrede. Aktiv overdragelse: [historisk pilot](H2_COMPACT_RATE_PILOT.md).
Følgende F-statusafsnit er historik.

## F4-H2: Windows-exportfejl rettet, PC-ledger afventer råreview, 7. oktober

PC-forsøget `f4_h2_compact_rates_pc_20261007_014819` stoppede ved
`Raw data audit changed.` i frozen_packet, før observationslæsning/modeljobs.
Rapport og terminal oplyser 6.964884199900553 s invocation og
1540.6592115679418 s H; review-ZIP/fuld PC-ledger er endnu ikke modtaget.
Ingen ny pilot eller seed-reset før den komplette ledger er reconcileret.

Årsagen er reproduceret: Git archive med core.autocrlf=true ændrer audit-LF
til CRLF. Afgrænset `.gitattributes` -text bevarer de frosne packet-/audit-
bytes; strict SHA256 og alle numeriske gates er uændrede. Launcheren og
source-manifestet inkluderer nu attributfilen. Ét Git-only check reproducerer
fejlen og bevarer alle fire frosne filer i seks autocrlf/eol-konfigurationer.
Check 0.3752458700000716 s, ingen model/JIT/workbook. Råreceipt:
`files/output/bm_1/runs/estimation/f4_h2_frozen_export_check_20261007.json`.
Dette check-charge afventer sammenfletning efter PC-entry; budgetseed er ikke
ændret. Sol High fortsætter. G/WM/frosne data/native kerner er uændrede.
Aktiv overdragelse: [historisk pilot](H2_COMPACT_RATE_PILOT.md).
Følgende F-afsnit er historik.

Sol High-implementering. **Klar til én afgrænset PC-kørsel, ikke kørt på
faktisk forbrugsdata endnu.** F's datagrundlag er allerede frosset/råreviewet;
brug `f4/h2_compact_data_ready_20261007.json`. Forbrugs-workbooken findes
ikke her. Dette mellemtrin prøver det eksisterende native likelihood- og
forecast-interface på faktisk F-condition uden at starte en optimizer.
G/WM, native kerner og tidligere evidence ændres ikke.

## Fast prøve og betydning

| Felt | Prøvens afgrænsning |
|---|---|
| Population/condition | Samme faste 13 MED-huse, frosne B/U/skalaer, C=1/k=2/Xi=1/sigma_x=.5, nominal sigma_y=.04, Beta(20,20) |
| Kurver | Én lineær f/g-reference, repræsenteret med eksisterende to-piece zero logits; ingen kurve-/parameterfit eller tuning |
| Prefix | Prediction-only 17.–18. februar 2022 (24 h), derefter alle kronologiske timer frem til og med tredje komplette TRAIN-time |
| Observationer | Kun 18.–21. februar (TRAIN+selection) dekodes; komplette counts bruges i jobgrafen. Ingen prior-/final-Y |
| Missing | Én manglende MED-husværdi maskerer hele timen. Den time får kun prediction; ingen interpolation eller delvist gennemsnit |
| Grids | Base1921/h=1/32768, fine2561/h=1/65536; egne priors, posteriors og kontinuerte fulde log-states, ingen daglige resets |
| Forecast-prober | Pre-update intervalmomenter på de tre komplette timer; CDF/kvantiler på anden/tredje komplette time, fra samme pre-hour state |
| Parallelisme | Én persistent pool med otte frekvensworkers; BLAS/Numba=1. Scalar prediction/momenter i hovedprocessen. Ingen kandidat-pool |
| Budget | Højst300 s inklusive60 s stopreserve, aldrig over resterende900 s native eller36000 s H. Imports/JIT/checkpoints/kildekomprimering tælles |
| Output | Immutable manifest/source, observations.npz, fulde raw hourly NPZ-checkpoints, timings, comparisons, report, ledger og review-ZIP |

Prøven er fastlagt før målte Y åbnes. Valget af de første tre **komplette**
timer er et availability-valg, ikke en søgning efter gunstige values. Alle
forudgående manglende timer propagateres for begge grids. Kun den fælles
lineære curve pair anvendes; fine-state starter aldrig fra base-posterior.
Baseline/priser genberegnes ikke, og fremtidige realiserede priser læses ikke.

`measure_h2_compact_rates.py` kontrollerer canonical packet identity,
datapolicy, audit-/inputhashes og workbookhash før og efter datalæsning.
Den eksisterende timestamp-first XML-reader udvælger observationernes
vindue før demand-celler dekodes. Energi kWh/time bliver kW og divideres
med den frosne effektskala uden klipning eller genestimation.
De 72 observationsslots holdes særskilt fra de 24 prediction-only prior-timer.

## Matematik, genbrug og accept

Ingen ny solver eller observationslov. Likelihood-opdateringen er den
eksisterende terminal-tilt update, med

\[
q_i=K_i(y_i)p_{i-1},\qquad Z_i=\mathbf1^Tq_i,
\qquad p_i=q_i/Z_i,\qquad \ell=\sum_{i\in O}\log Z_i.
\]

Missing/prior-timer bruger native-kernens **én zero-frequency block**:
ved omega=tilt=0 er reward-fasen identiteten og log-amplituden netop den
eksisterende positive CN-state-transition. Den generelle frekvensbank kræver
mindst én positiv frekvens, så scalar-kaldet bruger det eksisterende
`advance_frequency_block`, ikke `native_prediction(frequencies=0)`.
Ingen almindelig sandsynlighed genbruges som næste log-state.

Moment-/CDF-matematikken er allerede udledt i
[kompakt review](H2_COMPACT_ESTIMATION_REVIEW.md#cheap-interval-moments-derived-from-the-same-discrete-operator)
og kontrolleret i [forecast-implementeringen](H2_INTERVAL_FORECAST_IMPLEMENTATION.md).
Her anvendes disse interfaces på faktisk B/U og den faktiske pre-update
posterior. For hver CDF-probe kontrolleres moment-/CF-log-state-agreement
med samme støtte og højst1e-7 finite-log-difference. CDF-proben modtager ingen
aktuel eller fremtidig observation. Disse er one-step prequential probes;
de er ikke den åbne 168-timers slutuge.

De eksisterende guards består uændret: mass1e-10, conditional1e-8,
contour overlap1e-7; base/fine state-mean.002 og konservativ cell-overlap
mass-L1.05. Prefixets **samlede** log-score-difference skal være <=.05,
aldrig .05 per time. Interval mean/variance og CDF/quantiles <=.002.
CDF-grid, alias/truncation, quantile brackets og full log-state kommer fra
det eksisterende forecast-interface; ingen range-/monotonicity-repair.

En bestået prøve accepterer **kun dette lineære prefix**. Alle flags
`historical_fit_ready`, `numerical_likelihood_ready`, `runtime_admitted`,
`blind_h3_acceptance`, `scientific_acceptance` forbliver false. Hele
TRAIN/selection/final-windowens score-/state-gates og de faktisk valgte
nonlineære curves er stadig nødvendige. Tidligere ukendt slutugeeksponering
bevares: ingen ny blindhedsclaim eller nye datoer.

## Faktiske rates og hele jobgrafen

Hver ratesample inkluderer operator-build, kernel, IPC og almindelig raw
checkpoint-export. Operator-build prissættes igen, når et genbrugt operator
betjener en særskilt moment/CDF-probe: konservativt i forhold til fremtidige
selvstændige gemte passes. Første prediction/moment/observed-job per grid
registreres som cold; dette arbejde er charged, men ikke en varm time-rate.
CDF bruger allerede varmede frequency workers. Der kræves mindst to varme
samples for alle fire jobtyper på begge grids, og maksimum anvendes.
Normal scalar prior-propagation måles gennem hele det kronologiske prefix.
Cold-job-sum gemmes særskilt; ingen JIT-opvarmning køres bare for at måle.

`work_graph.json` priser hele den godkendte graf med den eksisterende faktor
1.5 **én gang**, uden at dele antal curve candidates med8:

| Job | Base | Fine |
|---|---:|---:|
|49 TRAIN-candidates|49×48 timer, observed/missing efter fast mask|0|
|Candidate-priors|49×24 prediction|0|
|Tre selection-continuations|3×24 observed/missing|0|
|Tre fine TRAIN+selection|0|3×72 observed/missing|
|Fine priors|0|3×24 prediction|
|Winner+linear final-filter|2×168|2×168|
|Selection/final predictive moments|408|408|
|Open-week moments|336|336|
|Syv open-CDF-horisonter, to curves|14|14|

Final-Y er ukendt; hver final-time prises med maksimum af observed/prediction
rate. De to modeller antages forskellige, og ingen påstået genbrug af
pilotarbejde fratrækkes. Alle timers counts og begge fine-/basepasses er med.

Derudover reserveres eksplicit300 s til kommende search-imports/JIT/
optimizer-wrappers og300 s til validation-imports/diagnostics/export/handoff,
samt60 s orderly stop. Auxiliary-tiderne er caps, ikke målte completion-beviser.
Summen bruger den **afsluttede** nyeste ledger; den fulde validation-graf
beskyttes som reserve, mindst7200 s, uden at blive lagt til totalen to gange.

En lineær prefix-rate er **ikke** et worst-case bound over alle nonlinear
curves eller senere posteriors. Selv et overslag under10 h giver derfor ikke
fit-tilladelse. Et overslag over10 h afviser denne plan som den står; ingen
automatisk mindre kandidatcap, løsere tolerancer eller ændret TRAIN. Prøven
leverer de målte tal, der skal bruges til den næste konkrete beslutning.

## Kontrol og tidsregnskab

Én fokuseret ny data-boundary/jobgraf-kontrol består på **1.39394814 s**
inklusive imports/kilde-export; ingen modelpropagation, FV/JIT eller gamle
numeriske cases genkøres. En syntetisk workbook har complete/missing MED-
timer og poison-formler fra21. februar. Kontrollen afviser ændret workbookhash,
bevarer fixed-group NaNs, finder første tre complete timer kronologisk og
reproducerer den godkendte grafsum ved uafhængig aritmetik. Incomplete rates
afvises, og ingen kontrol eller kort rate-estimate kan åbne fit.

Original rapport/manifest/kilde/ledger ligger i
`files/output/bm_1/runs/estimation/f4_h2_compact_rate_control_20261007_review.zip`.
Efter kontrollen ændres kun startup-regnskab, shutdown (Ctrl-C),
bevaring af netop afsluttede scalar-jobs og review-packaging; de
kontrollerede boundary-/graph-funktioners AST er verificeret identisk.
To nye Python-filer består syntaks; PowerShell statisk kontrolleret, ikke
eksekveret i chatmiljøet. Ingen ekstra bred suite.

Ny ledger: **1533.69432737 s H**, **509.06320673/900 s native**, rest
**390.93679327 s** på tværs af PC/chat/resumes. Seed udvider alle kendte
entries. Datapakkens1532.30 s er dens historiske snapshot, ikke en budgetreset.
Den kommende PC-invocation tager højst300 s af den resterende allowance.

## PC-kørsel og overdragelse

Brug den præcise source-commit i chatkommandoen. Launcheren eksporterer `src`,
`root/f4`, de auditerede packet-/array-/auditfiler og den nye budgetseed til
en ny TEMP-folder uden at skifte din aktive branch eller worktree. Eksisterende
Python-miljø med NumPy/pandas/SciPy/openpyxl og de allerede anvendte
numba0.62.1/llvmlite0.45.1 kræves; der installeres intet automatisk.

Det nye PowerShell-process flag `-ExecutionPolicy Bypass` gælder kun processen
og bruges som ved den afsluttede datapreparation. Launcherens egen Git archive
peger på samme præcise commit. Output og den fælles ledger skrives i din
eksisterende repo-folder. Modellerne kører kun i det isolerede source-snapshot. Launcherens
eksport/startup måles også og trækkes fra samme300 s grant; dens tid
overføres med invariant decimalformat til Python-ledgeren.

Ved deadline, STOP eller Ctrl-C standses workerpoolen; raw completed jobs
bevares. Review-ZIP eksporteres også ved et rent stop eller numerisk failure.
Ved watchdog/crash kan en reservation forblive charged, og arkivet kan være
ufuldstændigt: send output/ledger, genkør ikke. Ingen automatisk retry,
overwriting eller budget-reset. Prøven har ikke en automatisk resume-flow;
en eventuel fortsættelse skal vurderes ud fra de bevarede jobs og budgettet.

Send den ZIP-sti, som launcheren udskriver, også hvis den afslutter med fejl.
Råreview skal kontrollere source/inputs, checkpoint-states, målte maksimumtider,
prefix-præcision og hele jobgrafens pris før nogen længere kørsel overvejes.
