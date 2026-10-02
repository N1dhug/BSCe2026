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

`preparation` erstatter det tidligere `root/f4/verify_preparation.py`: samme
uafhængige regneregler, nu i validation og med fælles CLI. Ingen kontrol er slettet.
F3-kommandoer findes stadig via `verify`, `optimize`, `refine` og `real_data` i samme
BM1 entrypoint-mappe. F3-data, WM-kode og `environment.yml` er bevaret.

## Lange kørsler og reproduktion

Populationen viser nu løbende antal færdige sekvenser, elapsed og grov ETA.
`--quiet` skjuler kun progress, ikke slutresultatet. `--output` gemmer hele rapporten,
alle sekvensrækker, settings, source-commit og dirty-flag i en **ny** JSON-fil;
eksisterende output overskrives ikke. Uden eksplicit output skrives ingen resultatfiler.

```powershell
python -B -m src.exe.run_model.bm_1.estimate mixture-filter-population --workers 8 --blas-threads 1 --output files/output/bm_1/runs/f4_population.json
```

`--blas-threads 1` er opt-in: BLAS/OpenMP-miljøvariabler sættes for denne Python-proces
**før** NumPy/SciPy importeres og arves af worker-processer. Det ændrer ikke permanent
Windows-miljøet, tolerancer, seeds eller worker-fordelingen af sekvenser. Det undgår, at
mange processer hver starter mange BLAS-tråde. Flere workers er ikke automatisk hurtigere;
mål på egen maskine. Uden flaget beholdes bibliotekernes eksisterende thread-indstillinger.
I en allerede startet notebook kræver threadkontrol en runtime-mekanisme som threadpoolctl.

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

De resterende F-studier køres og gennemgås i rækkefølge. F1/F2 og F3 version 2
er gennemgået; første F3 og den oprindelige F4-population har bevarede state-L1-fejl.
Næste kørsel er alene F4's `bank-population-time-refine`, beskrevet nedenfor. Alle har
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
F1/F2, F3 version 2 og den oprindelige F4-population er nu gennemgået. Næste
lokale kørsel er alene `bank-population-time-refine`: to state-fejl peger på for
groft FV-tidsskridt i referencen. De gemte F1/F3/F4-filer bruges som dependencies;
mixture- og sigma_x-beregningerne gentages ikke. Bevar også den fejlede F3-pilot.

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

## Hvor kode hører hjemme

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
