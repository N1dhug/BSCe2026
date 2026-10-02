# BM — arbejdsvejviser

**F4-status og næste opgave:** [F4.md](F4.md).
**Sammenhængende matematik:** [f4/main.tex](f4/main.tex) + [f4/notes.tex](f4/notes.tex).
**Afsluttet F1–F3-evidens:** [BM_REFERENCE.md](BM_REFERENCE.md).
Der vedligeholdes ikke parallelle audit-/closure-rapporter i `root`.

## Almindelig kontrol fra projektmappen

```powershell
python -B -m unittest discover -s tests -q
python -B root/f4/build.py
```

PDF'en skrives til `root/f4/build/main.pdf`. Buildscriptet gentager selv LaTeX,
indtil referencerne er stabile; ingen `latexmk`/Perl og ingen manuel PATH-ændring
kræves. MiKTeX i brugerens normale installationsmappe kan findes automatisk.
Manglende LaTeX giver en klar fejl; Python-tests kræver ikke LaTeX. CI bygger også
PDF og gemmer logs/artifacts. Underfull-linjer er layoutdiagnostik; uløste labels,
overflow og egentlige buildfejl skal undersøges.

## Kør kun det relevante forsøg

Fælles indgang: `python -B -m src.exe.run_model.bm_1.estimate <kommando>`.
Hver kommando har `--help`. De tunge studier nedenfor er **ikke** daglige checks.

| Formål | Kommando |
|---|---|
| F4-0 uafhængig kontinuert kvadratur/Bayes | `preparation` |
| F4-B syntetisk sekvens / koblet EM | `synthetic`, `synthetic --refine` |
| F4-C sanity / numerik | `filter`, `filter-refine` |
| F4-C population / diagnostik | `filter-study`, `filter-calibration`, `filter-truth-refine` |
| Historisk D1 / D1b / D1c basisdesign | `mixture-static`, `mixture-static-adaptive`, `mixture-static-shape` |
| D2 statisk holdout | `mixture-holdout` |
| D3a / D3b / D3c | `mixture-filter`, `mixture-filter-refine`, `mixture-filter-converge` |
| D3d population | `mixture-filter-population` |
| F4-E E1 likelihood-refinement | `likelihood-refine` |
| F4-E E2 en-parameter screening | `parameter-screen --e1-report <E1-JSON>` |
| F4-E E3 diagnostiske parflader | `pair-surfaces --e2-report <E2-JSON>` |
| F4-F F0 rekursiv FV-rho-bank og prefix-identiteter | `bank-identity` |
| F4-F F1 FV-bank numerik for rho og sigma_x | `bank-refine` |
| F4-F F2 grid, støtte og parameterprior | `bank-sensitivity` |
| F4-F F3 rekursiv rho-mixture-pilot | `bank-mixture-check` |
| F4-F F4 rho FV/mixture og separat sigma_x FV-population | `bank-population` |
| F4-F F4 FV-rho-reference med finere tid, genbrug af gemt population | `bank-population-time-refine` |
| F4-G G0 fælles rho/Xi- og sigma_x/sigma_y-bank, prefix-identiteter | `joint-bank-identity` |
| F4-G G1a FV-tid, state-grid og quadrature for de to frie par | `joint-bank-refine --g0-report <G0-JSON>` |

`preparation` erstatter det tidligere `root/f4/verify_preparation.py`: samme
uafhængige regneregler, nu i validation og med fælles CLI. Ingen kontrol er slettet.
F3-kommandoer findes stadig via `verify`, `optimize`, `refine` og `real_data` i samme
BM1 entrypoint-mappe. F3-data, WM-kode og `environment.yml` er bevaret.

## Lange kørsler og reproduktion

Populationen viser nu løbende antal færdige sekvenser, elapsed og grov ETA.
`--quiet` skjuler kun progress, ikke slutresultatet. `--output` gemmer hele rapporten,
alle sekvensrækker, settings, source-commit og dirty-flag i en **ny** JSON-fil;
eksisterende output overskrives ikke. En `.gz`-endelse vælger tabsfri gzip;
E- og F-dependencies kan læses i begge formater, og SHA256 identificerer de gemte
inputbytes. `synthetic --save` afviser også eksisterende filer før beregning.
Uden eksplicit output skrives ingen resultatfiler.

```powershell
python -B -m src.exe.run_model.bm_1.estimate mixture-filter-population --workers 8 --blas-threads 1 --output files/output/bm_1/runs/f4_population.json
```

`--blas-threads 1` er opt-in: BLAS/OpenMP-miljøvariabler sættes for denne Python-proces
**før** NumPy/SciPy importeres og arves af worker-processer. Det ændrer ikke permanent
Windows-miljøet, tolerancer, seeds eller worker-fordelingen af sekvenser. Det undgår, at
mange processer hver starter mange BLAS-tråde. Flere workers er ikke automatisk hurtigere;
mål på egen maskine. Uden flaget beholdes bibliotekernes eksisterende thread-indstillinger.
I en allerede startet notebook kræver threadkontrol en runtime-mekanisme som threadpoolctl.
D3d/E/F bruger eksplicit spawn. Egne scripts, der starter flere workers, skal
derfor kalde runneren under `if __name__ == '__main__':`, ligesom projektets CLI.

E2 bruger den låste population 1212..1261 og alle fem 41-node-grids; de officielle
settings kan ikke ændres gennem CLI. E1-filen skal være den beståede registrerede
rho-refinement på ID1211. Eksempel efter E1:

```powershell
python -B -m src.exe.run_model.bm_1.estimate parameter-screen `
  --e1-report files/output/bm_1/runs/estimation/f4_e_e1_rho_refinement.json `
  --workers 7 `
  --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_e_e2_parameter_screening.json
```

Der udføres 50 x 5 x 41 = 10250 selvstændige kandidat-replays, med løbende progress.
Output må ikke overskrives. LR-mængder er shape-diagnostik, og de øvrige parametre
holdes på truth i hvert snit; parameterudvælgelse kræver stadig E3/E4.

E3 bruger præcis de gemte observationer fra E2's første ti records, ID1212..1221,
og hver anden gridnode. CLI kræver det komplette registrerede E2-artifact;
sekvenser, grids og numerik kan ikke ændres gennem CLI. Der udføres
10 x 3 x 21 x 21 = 13230 kandidat-replays fordelt på 30 uafhængige process-opgaver.
Kør efter E2 med samme process-/BLAS-indstillinger:

```powershell
python -B -m src.exe.run_model.bm_1.estimate pair-surfaces `
  --e2-report files/output/bm_1/runs/estimation/f4_e_e2_parameter_screening.json `
  --workers 7 `
  --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_e_e3_pair_surfaces.json `
  --figure-prefix files/output/bm_1/runs/estimation/f4_e_e3
```

JSON gemmer alle rå/centrerede flader, predictive log-evidence, maxima/ties,
grid-profiler, conditional ridges og E2-digest. Truth-snit kontrolleres mod E2-kurverne
til `1e-10`; ingen ny truth eller målingsstøj genereres. Det valgfrie figure-prefix
giver `_rho_Xi.png`, `_k_Xi.png` og `_sigma_x_sigma_y.png`, hver med samlet flade
og ti sekvensflader. Både JSON og PNG afvises før beregningen, hvis de allerede findes.
Summen bruger kun de samme ti E2-records; likelihood-contours er ikke konfidensområder.

Batchgenerering bruger samme pr.-ID RNG-strømme og samme EM-grid som enkeltgenerering.
Caches indeholder kun target-uafhængig geometri eller eksplicitte, validerede startgæt;
FV-referenceprojektioner starter aldrig mixture-filteret. Den registrerede D3d-protokol
er uændret; rapporten markerer, om en kørsels settings faktisk matcher den.

F0 bruger nyt ID1262, 41 rho-noder i 0.5..1.5 med prior 1/41, 401 celler,
h<=1/768 dag og q=3. De øvrige parametre holdes kendte. Kommandoen kræver
ingen E2/E3-genkørsel og har ingen CLI-overrides af det registrerede design:

```powershell
python -B -m src.exe.run_model.bm_1.estimate bank-identity `
  --workers 7 `
  --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_f_f0_rho_bank_identity.json
```

Der køres en online bank og 41 uafhængige offline-replays; alle 24 prefixes
kontrolleres. Poolen består gennem hele kørslen og fordeler hypoteser ved samme
tidspunkt, ikke observationstidspunkter. JSON indeholder alle scores/log-vægte,
prefix-fejl, diagnostik, kildecommit og endelige betingede state-masser. Eksisterende
output afvises før beregningen; en fejlet gate gemmes og giver derefter en fejl.
F0 tester metoden, ikke parameter-recovery, mixture eller ugeforecast.

F-studierne er kørt og rå-reviewet i rækkefølge. F1/F2, F3 version 2 og F4's
FV-tidsopfølgning er gennemgået; første F3 og den oprindelige F4-population har
bevarede state-L1-fejl. F4-F er afsluttet; kommandoerne nedenfor er reproduktion,
ikke nye kørsler der skal startes nu. Alle har
default 7 workers, fast videnskabeligt design og løbende progress.
F1 genundersøger begge enparametergrids; sigma_x får finere state-numerik efter
det første udviklingsforsøgs fejl. Forsøgsdefinition og tilpasning står i `F4.md`.

```powershell
python -B -m src.exe.run_model.bm_1.estimate bank-refine `
  --workers 7 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_f_f1_bank_refinement.json.gz
if ($LASTEXITCODE -ne 0) { throw "F1 numerik fejlede; behold rapporten til review." }
```

Send F1-resumé og rå `.json.gz` til review før længere studier. Gzip-kompression er
tabsfri og holder rapporterne håndterlige; ingen scores afrundes. Et fejlresultat
gemmes og giver nonzero exit. Eksisterende output afvises før læsning/beregning.
F1-input til senere studier skal have source-commit og ren working tree; input og
alle rå kurver genvalideres, ikke blot `passes`-flaget.

Efter F1-review kan disse kommandoer bruges; de er ikke en instruktion om at starte
alle kørsler før review. Planen må tilpasses dokumenteret efter resultaterne.
F1/F2, F3 version 2, den oprindelige F4-population og `bank-population-time-refine`
er nu gennemgået. Tidsopfølgningen løste de to state-fejl med finere FV-reference;
alle 40 valgte kontroller består. Den tog 406.30 s med 8 workers og BLAS=1,
og alle 396 regressionstests bestod lokalt. De gemte F1/F3/F4-filer er dependencies;
mixture- og sigma_x-beregningerne blev genbrugt. Bevar også den fejlede F3-pilot.
PR #49 er merged på main `59b6cf3`. Mid-development cleanup og dens kontroller
står i F4-statusfilen; næste udvikling er F4-G's udvalgte parameterpar og robusthed.

```powershell
python -B -m src.exe.run_model.bm_1.estimate bank-sensitivity `
  --refinement-report files/output/bm_1/runs/estimation/f4_f_f1_bank_refinement.json.gz `
  --workers 7 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_f_f2_bank_sensitivity.json.gz
if ($LASTEXITCODE -ne 0) { throw "F2 fejlede; behold rapporten til review." }

python -B -m src.exe.run_model.bm_1.estimate bank-mixture-check `
  --refinement-report files/output/bm_1/runs/estimation/f4_f_f1_bank_refinement.json.gz `
  --workers 7 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_f_f3_rho_mixture_check.json.gz
if ($LASTEXITCODE -ne 0) { throw "F3 mixture-kontrol fejlede; behold rapporten til review." }

python -B -m src.exe.run_model.bm_1.estimate bank-mixture-check `
  --spatial-refine `
  --refinement-report files/output/bm_1/runs/estimation/f4_f_f1_bank_refinement.json.gz `
  --workers 8 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_f_f3_rho_mixture_spatial_refinement.json.gz
if ($LASTEXITCODE -ne 0) { throw "F3 refinement fejlede; behold begge rapporter til review." }

python -B -m src.exe.run_model.bm_1.estimate bank-population `
  --refinement-report files/output/bm_1/runs/estimation/f4_f_f1_bank_refinement.json.gz `
  --mixture-report files/output/bm_1/runs/estimation/f4_f_f3_rho_mixture_spatial_refinement.json.gz `
  --workers 8 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_f_f4_bank_population.json.gz
if ($LASTEXITCODE -ne 0) { throw "F4 population fejlede; behold rapporten til review." }

python -B -m src.exe.run_model.bm_1.estimate bank-population-time-refine `
  --refinement-report files/output/bm_1/runs/estimation/f4_f_f1_bank_refinement.json.gz `
  --mixture-report files/output/bm_1/runs/estimation/f4_f_f3_rho_mixture_spatial_refinement.json.gz `
  --population-report files/output/bm_1/runs/estimation/f4_f_f4_bank_population.json.gz `
  --workers 8 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_f_f4_population_time_refinement.json.gz
if ($LASTEXITCODE -ne 0) { throw "F4 tidsrefinement fejlede; behold begge populationsrapporter til review." }
```

Mixture-piloten undersøger rho med frossen `shape_D`, ikke en ny sigma_x-mixture-bank.
F4-populationen omfatter også sigma_x, men med den separate FV-reference. Den har
60 banks / 2460 replays på 20 nye, uafhængige dage, hver med ny Beta(20,20)-state.
Mixture-numerik læses og valideres fra den beståede version 2-rapport. Et groft
runtime-skøn fra de registrerede replaytider er 60--90 minutter med 8 workers;
faktisk tid afhænger af belastning og de nye records. Lange
uafhængige hypotesereplays fordeles over en vedvarende Windows-compatible spawn-pool;
hver replay følger observationsrækkefølgen og sin egen state-posterior. Priorvariationer
genbruger kun conditional replays med uændrede hypoteser, observationsmodel og state-prior.
Ingen nye filer eller studier påbegyndes ved import. Ugeforecast hører stadig til I.

Den målrettede tidsopfølgning laver 40 nye FV-rho-banks / 1640 replays ved
hmax=1/1536 og 1/3072, med samme 401 celler og q=3. Den genvaliderer alle 60
oprindelige banks fra rå data og kontrollerer dependency-digests før beregningen.
40 gemte mixture-/sigma_x-banks genbruges i resuméet; gamle FV-rækker bruges som
diagnostik. Den nye rapport linker den gamle i stedet for at kopiere den store
sigma_x-statehistorik. Upload derfor også originalen ved senere review, hvis den
ikke allerede er tilgængelig. Populationens oprindelige fejl bevares.

Den rå-reviewede tidsopfølgning vælger FV-rho401/hmax=1/1536/q=3. Gemte rho-mixture
bruger 401/hmax=1/32768/q=3, og sigma_x-FV bruger 3201/hmax=1/1536/q=3.
F1's tidligere rho-reference ved hmax=1/768 bevares til reproduktion af F1 og
den oprindelige F4. Numeriske kontroller er bestået; rho er fortsat svagt bestemt
over ét døgn. F4-F-resultater og artifact-digests står i `F4.md`.

## F4-G0: registreret kørsel afsluttet

F4-F er afsluttet, og cleanup fra PR #50 er merged. Ingen F-studier skal
gentages. G0 på ren `6670969` er gennemført og rå-reviewet: 416 tests består,
alle 100 conditional forløb gennemfører på 86.6074 s med 8 workers og BLAS=1,
og begge par består metodeidentiteten. Artifact-digest og kvantitative resultater
står i F4-statusfilen. Ingen G0-genkørsel er nødvendig; PR #51 afventer brugerens
merge. G1a's par-numerik er også gennemført og rå-reviewet nedenfor.

G0 kontrollerer begge valgte par på ID1348/1349 med 25 joint-hypoteser hver,
50 conditional online-forløb og 50 friske offline-replays. Designet er fast i CLI;
default er **8 workers**, og `--blas-threads 1` skal angives ved denne kørsel.
En pool fordeler uafhængige hypoteser; der er ingen nested workers.

Hent det dokumenterede review fra projektmappen på feature-branchen
(PR'en merger brugeren selv):

```powershell
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }
git switch feature/f4-g-joint-bank-identity
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin feature/f4-g-joint-bank-identity
if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }
git status
git log -1 --oneline
if (git status --porcelain) { throw "Working tree er ikke ren." }
```

Den allerede gennemførte kørsel brugte nedenstående kommandoer. De skal ikke
gentages for at hente dokumentationsopdateringen. En diagnostisk genkørsel kræver
ren kilde og et nyt outputnavn; originalrapporten må ikke overskrives.

```powershell
python -B -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw "Tests fejlede." }

python -B -m src.exe.run_model.bm_1.estimate joint-bank-identity `
  --workers 8 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_g_g0_joint_bank_identity.json.gz
if ($LASTEXITCODE -ne 0) { throw "G0 fejlede; behold rapporten til review." }
```

Working tree skal være ren før G0; output ligger under ignoreret run-mappe.
Eksisterende output afvises uden beregning. JSON/gzip er tabsfri og bevarer
source-commit, dirty-flag, alle prefix-scores/log-vægte, inputs og state-masser.
En fejlet gate gemmes og giver nonzero exit; fejlende replays frasorteres ikke.
Progress/ETA er grov, fordi støjparret koster mere end rho/Xi-parret.
Terminalresumé og rå `.json.gz` fra den registrerede kørsel er gennemgået.
Runtime er målt på denne dagskontrol, uden et løfte om ugeforecast eller måneds-throughput.
G1a's par-numerik er afsluttet nedenfor; G1b's parametergrid og G2's længere
records/sampling/mismatch er endnu ikke implementeret.

## F4-G1a: registreret kørsel afsluttet

Separat branch `feature/f4-g-joint-refinement` bygger på det rå-reviewede G0.
G0-rapporten ovenfor genbruges som valideret dependency; den skal ikke genkøres.
Så længe PR #51 er åben, har G1-PR'en den som base; brugeren merger G0 først,
og PR #52 retargetes til main, før G1a merges.

Ren `de6fcd5` er gennemført og rå-reviewet: **425 tests** består på 212.081 s,
alle **42 banker / 1.050 replays og 18 aktive kontroller** består med
**8 workers / BLAS=1**. Rapporten måler **990.9608 s** før lagring, mens den
komplette PowerShell-stopwatch måler **1007.74 s** inklusive opstart og output.
Max auditafvigelse er 1.7054e-13; fire grove fejl bevares som diagnostik.
G1a accepterer rho/Xi FV401 og støjpar FV3201, hmax=1/1536, q=3 alene på
G0's faste 25-node-parametermål og disse records. Digest og rå review står i
F4-statusfilen. **Ingen G0/G1a-genkørsel er nødvendig.**

Hent resultatdokumentationen fra projektmappen:

```powershell
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }
git switch feature/f4-g-joint-refinement
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin feature/f4-g-joint-refinement
if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }
git log -1 --oneline
if (git status --porcelain) { throw "Working tree er ikke ren." }
```

G1a fastholder G0's 25 hypoteser per par og joint-prior og låser seks nye records,
syv FV-konfigurationer per record og **1.050** selvstændige replays. Alle scores,
vægte, state-historier og prefix-fejl gemmes; en fejlet numerisk gate giver en
rapport og nonzero exit. Rå gzip og terminalresumé er gennemgået.
Runtime/ETA måles på dette studie; uge- og månedsbenchmark følger særskilt.
Rapportens `elapsed_seconds` måles før JSON/gzip-lagring. Stopwatch nedenfor
måler hele Python-kørslen inklusive opstart, dependency-review og output;
behold også dette terminaltal ved runtime-review.

Den allerede gennemførte kørsel brugte nedenstående kommandoer. De skal ikke
gentages for at hente dokumentationen; en diagnostisk reproduktion kræver ren
kilde og et nyt outputnavn, så originalrapporten bevares. Fra projektmappen:

```powershell
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }
git switch feature/f4-g-joint-refinement
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin feature/f4-g-joint-refinement
if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }
git log -1 --oneline
if (git status --porcelain) { throw "Working tree er ikke ren." }

python -B -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw "Tests fejlede." }

$g1aTimer = [System.Diagnostics.Stopwatch]::StartNew()
python -B -m src.exe.run_model.bm_1.estimate joint-bank-refine `
  --g0-report files/output/bm_1/runs/estimation/f4_g_g0_joint_bank_identity.json.gz `
  --workers 8 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_g_g1a_joint_bank_refinement.json.gz
$g1aExit = $LASTEXITCODE
$g1aTimer.Stop()
Write-Host ("G1a wall time inkl. opstart/output: {0:N2} s" -f $g1aTimer.Elapsed.TotalSeconds)
if ($g1aExit -ne 0) { throw "G1a fejlede; behold rapporten til review." }
```

## Kodeplacering

| Placering | Ansvar |
|---|---|
| `src/model/` | Fysiske ligninger og modelinterfaces. |
| `src/method/`, `src/solver/` | Estimatorkerner, numeriske metoder og lavniveau-solvers. |
| `src/params/bm_1.py` | Observations-, Bayes-, likelihood- og remap-primitiver. |
| `src/control/` | F3-problem, fælles cost og prisoptimering. |
| `src/validation/estimation.py` | Offentlige inputkontroller, ingen eksperimenter. |
| `src/validation/bm_1/` | Referencecases, syntetisk data og videnskabelige studier. |
| `src/diagnostics/`, `src/tools/` | Metrikker henholdsvis data/preprocessing. |
| `src/exe/run_model/bm_1/` | CLI, progress, rapporter og eksplicit filoutput. |
| `tests/` | Regression og små integrationstests; ikke hele populationer ved import. |

Udvid eksisterende ansvar før nye moduler oprettes. Ingen `_new`-/compatibility-kopier,
omvendte core→CLI-imports eller skjulte normaliseringer. FV-reference og fast FV er
bevidst uafhængige implementeringer af samme scheme; de skal ikke deduplikeres væk.
Shared WM ændres kun i en særskilt begrundet og testet ændring.

## Levering og filer

Én afgrænset branch/PR med matematik, kode, tests og nødvendig dokumentation. Test og
sammenlign efter hver selvstændig rettelse; benchmark på samme settings/hardware og
rapportér også uændrede output eller relevante numeriske afvigelser. Brugeren merger;
derefter sync, branch-delete og prune. Ingen automatisk merge.

Lokale resultater hører i ignoreret `files/output/bm_1/runs/`, PDF-build i
`root/f4/build/`. Bevar inputworkbooks og afsluttet evidens i
`files/output/bm_1/f3_b/` og `files/output/bm_1/real_data/`. Slet ikke brugerens
lokale arbejde, stashes eller miljøer for at opnå en ren status.
