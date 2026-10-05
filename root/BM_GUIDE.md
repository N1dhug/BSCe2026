# BM — arbejdsvejviser

**F4-status og næste opgave:** [F4.md](F4.md).
Den aktive prioritering fra 3. oktober 2026 er I-baseline på det afsluttede
F-spor (integration nu implementeret), derefter obligatorisk H-f/g inden den endelige samlede I/J-leverance;
G2b-v2 er særskilt nat-sidearbejde uden for den kritiske vej; G2c kræver en ny
kompatibel registrering og G2b-accept. Bevar arkiveret G2a-v2
på dens oprindelige kildecommit. Nyt arbejde bruger branches i eksisterende
checkout; opret kun et worktree, hvis det er strengt nødvendigt. Bevar aktive
kørselsmapper og nødvendige ignorerede rådata/logs før oprydning. Se `AGENTS.md`
og arbejdsformen i F4-planen.
**Sammenhængende matematik:** [f4/main.tex](f4/main.tex) + [f4/notes.tex](f4/notes.tex).
**Afsluttet F1–F3-evidens:** [BM_REFERENCE.md](BM_REFERENCE.md).
Der vedligeholdes ikke parallelle audit-/closure-rapporter i `root`.

**Fælles beslutning for alle projektchats, 3. oktober 2026:** Fleksible f/g-kurver
skal estimeres og valideres i **H-f/g** før endelig H→I-handoff/J. Arbejdet er
obligatorisk; kurve-API og pooled fitprototype er implementeret, fuldt H-fit og
videnskabelig validering mangler stadig.
Læs repoets [../AGENTS.md](../AGENTS.md) og den aktuelle H-f/g-plan før arbejdet.
Kurverne deles på tværs af hele den erklærede træningspopulation, aldrig per dag.
Kandidaten med fire argumentintervaller `[0,0.25]`, `[0.25,0.5]`, `[0.5,0.75]`,
`[0.75,1]` er én kontinuert, ikke-voksende funktion med fælles estimerede knudeværdier;
den sammenlignes med en enklere to-stykkers og den lineære reference.
Frosne kurver/skalaer bruges ved validering og forecast. Det eksisterende G-design
forbliver den lineære reference; denne beslutning ændrer ikke gamle run/checkpoints.

## Genbrug i F4–F6

Læs F4-planens **Yderligere effektivisering** før ny implementering: ét fælles
I/F5/F6-interface, tidlig f/g-pilot og forberedelse af økonomi/sløjfe under fits.

* `run_fv_filter(..., keep_history=False)` returnerer `BM1FVLikelihoodResult` med
  `log_evidence`, `observed`, `total_log_likelihood`, `final_step` og samlede
  numeriske diagnoser. Ingen mellemliggende state-historik lagres. Alle observationer
  og steps beregnes stadig med den eksisterende kernel. E's kandidat-likelihood
  bruger dette automatisk; den nye pooled H-f/g-fitprototype bruger også denne vej.
  Brug default `keep_history=True` til state-/forecast-validering og figurer.
* `evaluator.evaluate_targets(prices, targets)` på FV-evaluators vurderer hver
  target-række på `evaluator.midpoints` med én fælles propagation ved fast pris.
  Returnerer én `FVCostEvaluation` per række, inklusive uændrede F3-straffe.
  Andre priser, priorer, kurver, baseline/process-inputs eller numerik kan ikke
  genbruge denne propagation. Det er kvadratisk tracking, ikke fuld F5-økonomi.

Parret implementation-/hukommelsesbenchmark fra en opdateret branch:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -B -m tests.benchmark_bm_efficiency --repeats 3 --output files/output/bm_1/benchmarks/f4_f6_efficiency_local.json
```

Kommandoen afviser eksisterende output og gemmer rå tider, miljø, kilders SHA256
og numeriske forskelle. Tracemalloc køres særskilt fra tidsmålingen; sporet
peak-allokering er ikke hele processens RAM. Målingen fra 3. oktober gav 20.06×
for et sweep med 21 A-mål og 96.34% mindre sporet hukommelse for likelihood-only
filteret, men **ingen målt walltime-gevinst for selve filteret**. Tallene gælder
de konkrete benchmarks, ikke hele projektet. Kørende/resumérbar G2a-v2 beholder
altid kildecommitten fra sit manifest.

## BM-figurer som i WM-eksemplerne

Figureksport er nu tilgængelig i `src/tools/bm_1_outputs.py` og CLI'en
`src/exe/run_model/bm_1/outputs.py`. Den genbruger F1/F2-beregninger og gemte
fordelinger; den udfører ingen ny estimation. Kør fra en opdateret branch,
aldrig ved at opdatere kilden under G2a-v2.

Lav de første figurer med det eksisterende F2-referencescenarie:

```powershell
python -B -m src.exe.run_model.bm_1.outputs reference
if ($LASTEXITCODE -ne 0) { throw "BM-figureksport fejlede." }
```

Kommandoen skriver en ny mappe under `files/output/bm_1/runs/outputs/` og udskriver
den fulde sti. Der bruges ét FV-forløb samt en deterministisk sammenligning fra
initialfordelingens middelværdi. Eksisterende referenceinputs/parametre og solver
genbruges; figurens outputgrid er 193 tidspunkter plus eventuelle prisbreakpoints.
Dette er en syntetisk reference, ikke et fit på målerdata eller afsluttet H-f/g.

| Fil | Indhold |
|---|---|
| `bm_state_density.png` | Tæthed p(X,t) med E[X] ovenpå, som WM2's tæthedsplot. |
| `bm_response.png` | State/pris, latent state-usikkerhed, netto-opladningsdemand E[D]-B og demand/baseline; reference indeholder også deterministiske kurver. |
| `bm_demand.png` | Demand/baseline, pris og BM-state; målinger vises kun, når kaldet indeholder dem. |
| `moments.csv` | Tid, eventuelle tidsstempler, B/U, middelværdi, standardafvigelse, 5/95%-kvantiler og eventuelle målinger. |
| `distribution.npz` | De fulde vægtede conditional masser og hver parameterhypoteses egne demandværdier samt inputs. |
| `summary.json` | Fortolkning, datakontrakt/metadata, massefejl og eventuelle MSE/RMSE/MAE/bias. Bias er prediction minus observation. |

Tegn den allerede gemte I-demonstration **uden nyt FV-replay eller optimizer**:

```powershell
python -B -m src.exe.run_model.bm_1.outputs i-report --input files/output/bm_1/runs/estimation/i_baseline_development_v1.json.gz
if ($LASTEXITCODE -ne 0) { throw "I-figureksport fejlede." }
```

Kun de syv gemte horisonter tegnes; der opfindes ingen timeprognose mellem dem.
`--belief known` vælger den kendte parameterreference; default er `rho_map`.
Den oprindelige I-demo og udeladt parameterusikkerhed fremgår stadig af metadata.
Et tæthedsplots tidsfelter viser nærmeste gemte slice, ikke nye beregnede laws.

Tegn en allerede eksporteret fordeling igen:

```powershell
python -B -m src.exe.run_model.bm_1.outputs redraw --input "STIEN_TIL_DEN_TIDLIGERE_OUTPUTMAPPE"
```

Alle kommandoer accepterer `--output-dir`; en eksisterende mappe afvises.
Eksporten bruger marginale **latente** 5–95%-kvantiler af X/D på det gemte support,
ikke konfidensintervaller, målestøjsintervaller eller samtidig 90%-ugecoverage.
Parameter/state-afhængigheden bevares i den fælles lov; ingen middelparameter-
substitution, massereparation eller ny observation update. BM-state og E[D]-B
sidestilles ikke med WM2/WM4's V/R/Q. Forecast, fit, filter og counterfactual
control skal deklareres særskilt. Observeret forbrug under spotprisen bruges
ikke til fejlscoring af et kontrafaktisk optimeret prisforløb.

Februar-testbranchen `preview/bm-f4-february` tilsluttes samme eksport og får
`forecast_distribution/` samt, ved kontrol, `control_distribution/` med ovenstående
filer. Det sker fra dens allerede beregnede joint laws uden ekstra solverforløb.
Det er fortsat et foreløbigt real-data-preview; den accepterer ikke H-f/g/I/J.
Opdatér først en lokal preview-mappe, når dens aktuelle kørsel er stoppet/færdig.

### Målt demand og én figur pr. dag

Når `timestamps` følger de gemte slices, tilføjer eksporten automatisk
`daily/bm_demand_YYYY-MM-DD.png`: én figur pr. lokal kalenderdato. Februar-
previewets 168 timer giver syv figurer. Demandpanelet viser **Observed demand I**
(sort), **Predicted demand E[D]** (blå), B-baseline og det latente 5–95%-bånd;
pris og state vises nedenunder. Alle dagsfigurer fra samme eksport bruger samme
demand-skala. Målingerne vises også i `bm_response.png`. `moments.csv` indeholder
`observed` og `D_mean` på de samme timestamps.

Dette er syv udsnit af **den samme frosne ugeprognose**, ikke syv nye daglige
fits eller prognoser. Manglende målinger forbliver NaN. Båndet omfatter fortsat
latent D, ikke målestøj. Tidsstempler skal være ISO-datetimes og stemme med den
gemte numeriske tidsakse; gentagne offset-aware klokketimer bevares. Et
kontrafaktisk kontrolforløb får ikke spotprisforløbets målte demand som overlay.

For en allerede færdig februar-preview: opdatér den eksisterende preview-mappe,
og tegn dens `forecast_distribution` igen. Eksempel for den gemte lokale kørsel:

```powershell
Set-Location -LiteralPath "C:\Users\jvils\OneDrive\Skrivebord\Til VS\fag\7 semester\BSCe2026-bm-preview" -ErrorAction Stop
git pull --ff-only origin preview/bm-f4-february
if ($LASTEXITCODE -ne 0) { throw "Preview-opdatering fejlede." }

$bmRun = ".\files\output\bm_1\runs\f4_bm_preview\2022-02-07_20261003T181009354096Z"
$bmFigures = Join-Path $bmRun ("daily_review_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
python -B -m src.exe.run_model.bm_1.outputs redraw --input (Join-Path $bmRun "forecast_distribution") --output-dir $bmFigures
if ($LASTEXITCODE -ne 0) { throw "Figureksport fejlede." }
Invoke-Item -LiteralPath (Join-Path $bmFigures "daily")
```

Ingen workbook-genimport, FV-solver eller optimizer køres ved `redraw`; målinger,
prognose, parametervægte og f/g genbruges fra NPZ. Originalrun og rådata overskrives
ikke. Nye preview-kørsler får dagsfigurerne direkte under deres
`forecast_distribution/daily/`.

## H1: nye kurver og numerisk kontrol

**Integration og genoptagelse, 5. oktober:** Denne version samler H1-koden
med de efterfølgende Windows-checkpointrettelser. Report-I/O ligger i
`src/tools/bm_1_reports.py`, og `curves.py` er registreret som entrypoint.
V2-kørslen og det uafhængige arrayreview bestod på original `cfde153` i
sessionen, men råmapperne skal gendannes eller reproduceres før næste
dependency-gate. Genoptag eksisterende rådata på deres oprindelige source;
en anden source kræver en ny run-mappe, uden ændring af gamle manifests.
H1b-`fit` og den historiske datakontrakt er gendannet på recovery-branchen.
H1b kræver komplette råvaliderede v2-units; et sessionsresumé er utilstrækkeligt.
Den tidligere layoutfejl er rettet i denne integration. Se F4-status for
testresultater og præcis genoptagelsesrækkefølge.


Det nye entrypoint holder H adskilt fra den store G-CLI og genbruger samme
atomiske checkpoint-I/O. Kør fra en **ren branch**, uden at ændre source under
kørslen. Gemte G-resultater er ikke en dependency. Lav først én kort unit:

```powershell
python -B -m src.exe.run_model.bm_1.curves numerics --run-dir files/output/bm_1/runs/estimation/h1_numerics_v1 --limit-units 1 --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "H1-piloten fejlede." }
```

Fortsæt samme kilde og samme mappe:

```powershell
python -B -m src.exe.run_model.bm_1.curves numerics --run-dir files/output/bm_1/runs/estimation/h1_numerics_v1 --resume --max-hours 8 --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "H1-kørslen fejlede eller panelets numerik blev afvist; behold output." }
```

V1 er gennemført og **afvist** (se F4-status). En særskilt v2-opfølgning bruger
samme cases og grænser med eksisterende fitted-flux og finere tid/rum. Start i
en ny mappe med `--design v2`; samme flag skal bruges ved resume:

```powershell
python -B -m src.exe.run_model.bm_1.curves numerics --design v2 --run-dir files/output/bm_1/runs/estimation/h1_numerics_v2 --blas-threads 1
```

96 units à fire FV-replays gemmes som komprimeret NPZ. `manifest.json` fastlåser
source, kurver, records, numerik og grænser; `summary.json` giver alle prefix-
kontroller og SHA256 for hver råunit. Resume verificerer rådata og genkører
ikke færdige units. En pause er `completed=false` og ikke accept. Et komplet
afvist panel giver exit=2 og bevarer alle negative resultater. State-L1 beregnes
på begge grids' samlede cellekanter, så forskelle ikke skjules ved grov remapping.
Send `summary.json` og behold hele mappen til råreview. En bestået screening
accepterer kun det faste panel, ikke hele optimizerboksen, recovery eller H3.

## H1b: genoptageligt pooled fit

Kør først den registrerede H1a-v2-pilot og fortsæt på **samme rene source**:

```powershell
$hNumerics = "files/output/bm_1/runs/estimation/h1_v2_recovery"
python -B -m src.exe.run_model.bm_1.curves numerics --design v2 --run-dir $hNumerics --limit-units 1 --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "H1a-pilot fejlede." }
python -B -m src.exe.run_model.bm_1.curves numerics --design v2 --run-dir $hNumerics --resume --max-hours 8 --workers 8 --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "H1a fejlede." }
```

Start kun H1b efter `completed=True` og `numerical_panel_passes=True`:

```powershell
$hFit = "files/output/bm_1/runs/estimation/h1_fit_recovery"
python -B -m src.exe.run_model.bm_1.curves fit --numerics-run $hNumerics --run-dir $hFit --limit-candidates 3 --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "H1b-pilot fejlede." }
python -B -m src.exe.run_model.bm_1.curves fit --numerics-run $hNumerics --run-dir $hFit --resume --max-hours 8 --workers 3 --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "H1b fejlede." }
```

H1a understøtter `--workers 8` med bounded spawn-pool og BLAS=1.
H1b understøtter `--workers 3` (eller 8, som begrænses til tre cases).
To-/fire-stykkers familierne inden for en case køres fortsat i samme rækkefølge.
`--limit-candidates` kræver `--workers 1`, så pilotens globale loft er entydigt.
BLAS=1 bevares. Pause er exit=0 og `paused=True`, aldrig convergence/accept.
Alle 96 numeriske råunits valideres ved hver invocation uden nye FV-replays.
Fitmanifestet låser source, dependency-hashes, Python/NumPy/SciPy, numerik og
optimizerbudget. Gemte kandidater genbruges som scalar-scores, ikke som state
for nye kurver. Trace gemmes efter hver kandidat. De seks familier får hver
højst 250 forskellige kandidater samlet over deres to erklærede starter.
Valgte kurver kræver egne tid/rum/q-kontroller; TRAIN-gevinst vælger ikke model.

Gem `manifest.json`, `summary.json`, `*_trace.json.gz`, `*_result.json.gz` og
H1a's `units/*.npz` sammen med source-commit og filhashes. Send komplette
artefakter, ikke kun terminalteksten. Historiske H2/H3-kørsler er ikke frigivet;
dataklassen er en kontrakt, ikke dokumentation af de faktiske målerdata.

Når de seks fitfamilier er afsluttet (budgetstop må gerne være et negativt resultat):

```powershell
python -B -m src.exe.run_model.bm_1.curves validate-fit --numerics-run $hNumerics --fit-run $hFit --run-dir files/output/bm_1/runs/estimation/h1_selected_checks --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "Valgte H1-kurver fejlede deres numeriske kontrol." }
```

Denne kommando gemmer 12 nye råunits med alle prefixer på base/tid/rum/q.
Resume bruger `--resume` og samme source. Den ændrer ingen fitkurver efter
kontrollerne. Budgetstoppede fits forbliver budgetstoppede, og H2/H3 frigives ikke.

## I-baseline og H1-forberedelse

Den lille integration bruger gammelt F3-ID1326, ikke en ny population. Kendt
FV-filter og 41-node-rho-bank assimilerer kun Y1..Y24. `BM1ForecastOrigin`
eksporterer MAP-nodens egne parametre og conditional posterior konservativt til
F3. Samme origin bruges til 1/6/12/24/48/72/168-timers forecast med ét forward-
forløb; fra dag 1 går 168 timer til dag 8. Fremtidige inputs deklareres før
filteret: konstant B=0.5 og det gentagne syntetiske daglige U-mønster.

Kommandoen virker fra en opdateret branch, uden G2a-dependencies:

```powershell
$iTag = Get-Date -Format "yyyyMMdd_HHmmss"
python -B -m src.exe.run_model.bm_1.estimate i-baseline --blas-threads 1 --output "files/output/bm_1/runs/estimation/i_baseline_$iTag.json.gz"
```

Vent med et ekstra lokalt CPU-job, hvis G2a allerede bruger din PC fuldt.
Eksisterende output afvises. Rå scores, vægte, origins og forecast-masser,
numeriske diagnoser, kildecommit og kendt/estimeret F3-policy-sammenligning gemmes.
F3 bruger næste døgn, to priser og samme tracking-/terminal-/prisstraf; kendt
reference bruger filtreret state, ikke truth. Parameterusikkerhed udelades.
Demo og tids-/rumdiagnoser accepterer ingen populations-/ugekalibreringsclaim;
`forecast_scientific_acceptance=False` er tilsigtet.

Den gennemførte Linux-demo på ren `df904fa` tog 10.94 s med BLAS=1.
[Rårapport](../files/output/bm_1/runs/estimation/i_baseline_development_v1.json.gz)
og [uafhængigt rå review](../files/output/bm_1/runs/estimation/i_baseline_development_v1_raw_review.json)
ligger i repoet. Rho-MAP rammer den øvre støtte, med effektivt hypotesetal 38.73/41;
det fungerende handoff er ikke parameter-recovery. Se tal og begrænsninger i F4.

`src/control/bm_1_handoff.py` er det fælles origin/F3/forecast-interface til I/F5/F6;
modellen og solverkernels genbruges. Nye tider bruger den erklærede tidsenhed,
og ingen fremtidige observationer indgår. Et længere prisprogram kan genbruges
til en kortere forecast; senere breakpoints ændrer ikke dens state-forløb.

`BM1PiecewiseResponse` i `src/params/bm_1_response.py` forbereder H1: ét fælles
kurveobjekt med to/fire stykker, ordnede indre værdier og endpoints 1,-1. Den
samme f/g eksporteres til `BM1Parameters`, F3 og forecast. Kurvebeskrivelsen
gemmer kun geometri; det endelige H-artifact skal også deklarere træning,
population, normalisering, øvrige parametre og modelvalg. Fuld erklæret træning,
ny representativ FV-kurve-numerik og blind validering mangler stadig.

Den lille pooled-fitprototype kører i det eksisterende entrypoint:

```powershell
$curveTag = Get-Date -Format "yyyyMMdd_HHmmss"
python -B -m src.exe.run_model.bm_1.estimate curve-fit-pilot --blas-threads 1 --output "files/output/bm_1/runs/estimation/h_curve_pilot_$curveTag.json.gz"
```

Den bruger kun gamle udviklingsrecords ID1326/1327, samme f/g i begge records,
kendte fysiske parametre og højst 250 forskellige kandidater per familie.
To og fire stykker sammenlignes med lineær reference; kvartknuderne er faktisk
implementeret. To-stykkers fit leverer én start til fire-stykkers fit.
Budgetloftet er samlet over starter, ikke 250 per start. Ved budgetstop gemmes
bedste evaluerede kandidat og hele scalar-tracen; det markeres ikke som optimizer-
convergence. Probability-/model-fejl dropper ingen records og skjules ikke som
en optimeringsomkostning. Alle fremtidige kandidater får korrekt fuldt replay.

`CurveTrainingRecord` accepterer kun TRAIN og eksplicit initiallov/tidsenhed;
sammenhængende dage skal videreføre state inden for ét record. Objektet afgør
ikke selv, hvilke data der er lovlige TRAIN-data; dataaudit, split og normalisering
skal fastlægges først. Logitbounds [-4,4] giver monotone positive fald med sum 2,
men eksakt flade delstykker er uden for den afgrænsede fitfamilie. Lambda=0.1
er fast før piloten. Ingen automatisk modelvinder vælges efter træningsscore.

Rapporten gemmer observationer, alle kandidat-koordinater/scalar-scores,
valgte fælles kurver, record-scores/final states, tids-/rum-/q-diagnoser og
kilders SHA256. `scientific_acceptance=False` er tilsigtet. Budget-ETA er tid
til det maksimale fitbudget; early convergence kan forkorte fit, og numeriske
diagnoser følger bagefter. Piloten er ikke genoptagelig og må ikke bruges som
en lang produktionstræning. Start ikke ekstra tunge lokale jobs under G2a.

Den gemte Linux-pilot på ren `e6d3441` tog **44.79 s** med BLAS=1 og 694
conditional record-replays. To-stykkers fit brugte 87 kandidater og begge
starter konvergerede; fire-stykkers fit stoppede ved 250 kandidater i første
start. Budgetstoppet bevares, og ingen modelvinder eller ekstra budgetrunde
følger af TRAIN-forbedringen. [Rårapport](../files/output/bm_1/runs/estimation/h_shared_curve_development_v1.json.gz)
og [uafhængigt rå review](../files/output/bm_1/runs/estimation/h_shared_curve_development_v1_raw_review.json)
ligger i repoet. Dette er et lille lokalt runtime-budget, ikke en prognose for
hele H-træningen eller for Windows-maskinen.

De to gamle prisniveauer kan kun identificere to kombinationer af de tre indre
fire-stykkers g-værdier. Mere fitbudget løser ikke den manglende prisvariation;
regulariseringen kan styre den resterende retning uden dataevidens. En gemt
[pris-supportdiagnose](../files/output/bm_1/runs/estimation/h_curve_price_support_development_v1.json)
viser to forskellige monotone g-kurver med identisk likelihood/slutfordeling.
Næste H1-design skal kontrollere input-support før længere fit; et forslag med
fire spredte priser står i F4. Det eksisterende F/G-design ændres ikke.

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
| F4-G G1b0 parametergrid/prior-pilot med FV-kontrol af nye noder | `joint-bank-grid --g1a-report <G1a-JSON>` |
| F4-G G1b1 v2 parret 9x9→17x17-kontrol efter afvist G1b0 | `joint-bank-grid-refine --g1b0-report <G1b0-JSON>` |
| F4-G G1b2 v3 samlet 17x17→33x33-kontrol, LAPACK og løbende reduktion | `joint-bank-grid-fine --g1b0-report <G1b0-JSON>` |
| F4-G G1b3 v4 samlet 33x33→65x65-kontrol | `joint-bank-grid-dense --g1b0-report <G1b0-JSON>` |
| F4-G G1c0 særskilt kontinuert hat-pilot | `joint-bank-reconstruct --grid-report <G1b3> --g1b0-report <G1b0>` |

`preparation` erstatter det tidligere `root/f4/verify_preparation.py`: samme
uafhængige regneregler, nu i validation og med fælles CLI. Ingen kontrol er slettet.
F3-kommandoer findes stadig via `verify`, `optimize`, `refine` og `real_data` i samme
BM1 entrypoint-mappe. F3-data, WM-kode og `environment.yml` er bevaret.

## Lange kørsler og reproduktion

De lange D/E/F/G-kørsler i `estimate` viser løbende antal færdige opgaver samt
`elapsed`, `remaining~` og `total~` i timer/minutter/sekunder. Eksempel:
`elapsed=12m 30s; remaining~1h 10m 00s; total~1h 22m 30s`.
`total~` er den forventede samlede varighed fra denne kørsels progress-timer,
ikke summen af tidligere forsøg eller pauser. Python-opstart og endnu ikke
udført slutreview/output kan ligge udenfor estimatet. Før en hastighed kan
måles, vises `estimating`; tidlige estimater er særligt usikre.

Estimatet beregnes ved de eksisterende statusopdateringer: konstant arbejde,
ingen nye modelkørsler, arraykopier eller ekstra filskrivninger. Den rektangulære
G2a-v2 og G2b/G2c vægter resterende arbejde efter celler, time-step og
observationsquadrature; øvrige progress-runners bruger gennemførte opgaver.
Gridvægtene er en grov prisproxy, ikke et løfte om runtime. Ved genoptagelse
udelades tidligere gemt arbejde fra den aktuelle hastighedsmåling.
G2a-v2-estimatet dækker **hele det resterende studie**, også ved `--limit-chunks`;
den afgrænsede invocation kan derfor stoppe tidligere. G2b/G2c-estimatet dækker
de valgte pending units, inklusive en eventuel `--limit-units`-afgrænsning.
`units`-parallelism viser først et tal efter den første worker-bølge og opdaterer
det ved færdige units; workers får ingen nye statuskanaler.

Progress-ændringerne påvirker hverken design, gates, seeds, resultater eller
checkpoint-format. **Opdatér ikke kildekoden under en igangværende kørsel.**
Et eksisterende checkpoint kræver fortsat sin oprindelige source-commit ved
resume, også hvis den nye kode kun ændrer statuslinjer.

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
står i F4-statusfilen. Ingen G0-genkørsel er nødvendig; PR #51 er korrekt merged
på main 44c6f3c. G1a's par-numerik er også gennemført og rå-reviewet nedenfor.

G0 kontrollerer begge valgte par på ID1348/1349 med 25 joint-hypoteser hver,
50 conditional online-forløb og 50 friske offline-replays. Designet er fast i CLI;
default er **8 workers**, og `--blas-threads 1` skal angives ved denne kørsel.
En pool fordeler uafhængige hypoteser; der er ingen nested workers.

Hent det dokumenterede review fra projektmappen på main:

```powershell
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }
git switch main
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin main
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
G1a's par-numerik og G1b0's parametergrid-pilot er afsluttet og rå-reviewet
nedenfor. G1b1's finere parametergrid er preregistreret nedenfor; en større population
og G2 er endnu ikke implementeret.

## F4-G1a: registreret kørsel afsluttet

PR #52 er korrekt merged på main `0426b50` efter G0; det testede merge-tree
er identisk med det gennemgåede head. G0-rapporten blev genbrugt som valideret
dependency; den skal ikke genkøres.

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
git switch main
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin main
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
git switch main
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin main
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

## F4-G1b0: gennemført, parametergrid afvist

Ren `990b050` er gennemført og rå-reviewet: **434 tests** består på **442.790 s**,
og alle **648 replays / 26 reduktioner** er afsluttet med **8 workers / BLAS=1**.
De **12 FV-kontroller består**, mens begge aktive **5x5→9x9-parameterkontroller
fejler**. Joint-TV er 0.325197/0.362580 mod grænsen 0.02; også unit-momenter,
evidence og begge state-L1-gates overskrides. Max massefejl er 2.4061e-12.
Uafhængig rå genberegning af alle banker og 18 sammenligninger afviger højst
1.5632e-13. Digest, samtlige maxima og reviewets omfang står i F4-statusfilen.

Rapporten måler **757.9881 s** før lagring; stopwatch måler **773.05 s** inklusive
opstart og output. Rapportens **passes=False** og efterfølgende nonzero exit er
den korrekte afvisning af kandidatgridet. **Behold originalrapporten; genkør ikke
G0, G1a eller G1b0 nu.** 9x9-gridet er heller ikke accepteret som parameter-
approksimation; det kræver en separat kontrol mod et finere grid.

PR #53 dokumenterer den afsluttede pilot med negativt resultat. Hent kun
resultatdokumentationen fra projektmappen; ingen studiekørsel følger denne blok:

```powershell
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }
git switch feature/f4-g-parameter-grid
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin feature/f4-g-parameter-grid
if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }
git log -1 --oneline
if (git status --porcelain) { throw "Working tree er ikke ren." }
```

PR #53 er korrekt merged på main `aa98beb`; kode og resultatdokumentation er med.
Den særskilt preregistrerede G1b1-opfølgning og dens nye CLI står nedenfor.
Større population og uge-/månedsbenchmark følger efter den nødvendige grid-accept.

### G1b0-design og allerede gennemførte kommandoer

Branch **feature/f4-g-parameter-grid** bygger på merged main `0426b50`.
Prior-målet er uniformt i rho/Xi og log-uniformt i støjparametrene, med
tensor-trapezvægte på 3/5/9 nested noder per akse. Nye **ID1356/1357**
kontrollerer 5x5→9x9; både 5x5- og 9x9-gridet får tre FV-kontroller.
**648 replays / 26 bankreduktioner**, alle 14 aktive gates skal bestå.
Coarse-grid og prior bridge er diagnostik. Dette er en pilot; større population,
recovery og uge-/månedsmål følger særskilt. Design og tolerancer står i F4/noterne.

**Genbrug den rå-reviewede G1a-fil; ingen G0/G1a-genkørsel.** CLI råvaliderer
G1a's 42 banker og 36 kontroller før arbejdet og gemmer digest/source.
Windows-default er **8 workers, BLAS=1**. Rapportens tid inkluderer
dependency-review, men slutter før JSON/gzip-lagring; stopwatch måler hele
processen. Replays koster forskelligt, så ETA er grov.

Den allerede gennemførte kørsel brugte nedenstående kommandoer. De skal ikke
gentages for at hente dokumentationen. En diagnostisk reproduktion kræver ren
kilde og et nyt outputnavn, så originalrapporten bevares. Fra projektmappen:

```powershell
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }
git switch feature/f4-g-parameter-grid
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin feature/f4-g-parameter-grid
if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }
git log -1 --oneline
if (git status --porcelain) { throw "Working tree er ikke ren." }

python -B -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw "Tests fejlede." }

$g1bTimer = [System.Diagnostics.Stopwatch]::StartNew()
python -B -m src.exe.run_model.bm_1.estimate joint-bank-grid `
  --g1a-report files/output/bm_1/runs/estimation/f4_g_g1a_joint_bank_refinement.json.gz `
  --workers 8 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_g_g1b0_joint_parameter_grid.json.gz
$g1bExit = $LASTEXITCODE
$g1bTimer.Stop()
Write-Host ("G1b0 wall time inkl. opstart/output: {0:N2} s" -f $g1bTimer.Elapsed.TotalSeconds)
if ($g1bExit -ne 0) { throw "G1b0 fejlede; behold rapporten til review." }
```

Rå gzip og terminalresumé er gennemgået, inklusive de fejlede gates. Output overskrives aldrig.
Originale fejl beholdes før en ny designversion; ingen automatisk tilpasning af
grid, støtte, numerik eller records. Brugeren merger selv efter review.
Development API-overrides er eksplicit uregistrerede.

## F4-G1b1 v2: selvstændig reproduktion

Kommandoen bevares til reproduktion. Den næste samlede Windows-kørsel er
G1b3 nedenfor; G1b1 skal ikke køres først.

Branch **feature/f4-g-grid-refinement** bygger på main `aa98beb`.
Samme ID1356/1357, inputs/truth, prior-mål, støtter og tolerancer som G1b0.
**5/9/17 noder per akse**, kandidat 9x9 mod kontrol 17x17. Alle 289 hypoteser
per par får fire friske FV-konfigurationer: **2.312 replays / 24 banker**.
Tre FV-kontroller på både 9x9 og 17x17 samt to parameterkontroller giver
**14 aktive gates**, alle på samtlige 24 prefixer. Tidligere 5x5→9x9-fejl
gemmes som to diagnostikker; prior bridges bevares i originalrapporten.

Genbrug **den fejlede, rå-reviewede G1b0-fil**. CLI genvaliderer dens 26 banker /
18 kontroller og G1a/G0-dependency-kæde; **ingen G0/G1a/G1b0-genkørsel**.
En separat reproduktionsgate sammenligner alle 16 fælles 5-/9-grid-banker med
originalens scores, vægte/evidence, marginale states og actual steps, tolerance
1e-10. Fejlede gates gemmes før nonzero exit; output overskrives aldrig.

Kør fra projektmappen på clean feature-branch **før merge**; brugeren merger
selv efter registreret kørsel og rå review. Default **8 workers, BLAS=1**.
Rapportens tid inkluderer dependency-review og reproduktionsgate, men slutter
før lagring. Stopwatch inkluderer også opstart/output. ETA er grov; flere
replays er ikke dokumentation for praktisk uge-/månedsruntime.

```powershell
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }

git switch feature/f4-g-grid-refinement
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }

git pull --ff-only origin feature/f4-g-grid-refinement
if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }

git log -1 --oneline
if (git status --porcelain) { throw "Working tree er ikke ren." }

python -B -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw "Tests fejlede." }

$g1b1Timer = [System.Diagnostics.Stopwatch]::StartNew()
python -B -m src.exe.run_model.bm_1.estimate joint-bank-grid-refine `
  --g1b0-report files/output/bm_1/runs/estimation/f4_g_g1b0_joint_parameter_grid.json.gz `
  --workers 8 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_g_g1b1_joint_parameter_grid_refinement_v2.json.gz
$g1b1Exit = $LASTEXITCODE
$g1b1Timer.Stop()
Write-Host ("G1b1 wall time inkl. opstart/output: {0:N2} s" -f $g1b1Timer.Elapsed.TotalSeconds)

if ($g1b1Exit -ne 0) { throw "G1b1 fejlede; behold begge rapporter til review." }
```

Send den nye rå gzip og terminalresumé inklusive stopwatch, også ved fejl.
Behold G1b0-artifactet. Der loves ingen grid-accept; større population og
recovery kræver senere nye records. En bestået pilot accepterer kandidatens
9x9-bank på disse to records, ikke automatisk 17x17 eller ugeforecast.

## F4-G1b2 v3: selvstændig reproduktion

Udviklingskørslen er rå-reviewet: FV/reproduktion består, begge parameter-TV-gates
fejler. Denne kommando bevares; næste samlede Windows-handoff er G1b3 nedenfor.
Du skal ikke først køre G1b2.

Den samlede kontrol bruger **5/9/17/33 noder**, kandidat 17x17 mod 33x33,
**8.712 replays / 32 banker og 20 aktive gates**. Den genskaber også G1b1's
5/9/17-banker og bevarer de grovere fejl som diagnostik. G1b0's rå Windows-fil
valideres og de 16 fælles banker reproduceres til 1e-10. **Du skal ikke først
køre G1b1**, og ingen G0/G1a/G1b0-genkørsel er nødvendig.

Opt-in LAPACK løser samme FV-system med uændret numerik. Den løbende reduktion
bevarer hver hypoteses egen state og bruger de korrekte gamle/nye Bayes-vægte.
Windows bruger fortsat **8 workers / BLAS=1**, nu med begrænset kø/hukommelse.
Design, fejlbevaring og udviklingsevidens står i F4-status/noter.

Fra projektmappen, før merge (brugeren merger selv efter rå review):

```powershell
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }
git switch feature/f4-g-grid-refinement
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin feature/f4-g-grid-refinement
if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }
git log -1 --oneline
if (git status --porcelain) { throw "Working tree er ikke ren." }

python -B -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw "Tests fejlede." }

$g1b2Timer = [System.Diagnostics.Stopwatch]::StartNew()
python -B -m src.exe.run_model.bm_1.estimate joint-bank-grid-fine `
  --g1b0-report files/output/bm_1/runs/estimation/f4_g_g1b0_joint_parameter_grid.json.gz `
  --workers 8 --blas-threads 1 `
  --output files/output/bm_1/runs/estimation/f4_g_g1b2_joint_parameter_grid_fine_v3.json.gz
$g1b2Exit = $LASTEXITCODE
$g1b2Timer.Stop()
Write-Host ("G1b2 wall time inkl. opstart/output: {0:N2} s" -f $g1b2Timer.Elapsed.TotalSeconds)
if ($g1b2Exit -ne 0) { throw "G1b2 fejlede; behold rå rapport og terminalresumé til review." }
```

Send den nye rå gzip og terminalresumé med stopwatch, også ved gatefejl.
Output overskrives aldrig. En bestået pilot certificerer kun kandidatgridet på
to records; større population, G2 og praktisk uge-/månedsbenchmark følger særskilt.

## F4-G1b3 og G1c0: afsluttet Windows-kørsel og reproduktion

Windows-kørslen og rå review er afsluttet på clean `1336e35`:
461 tests består; G1b3 udførte 33.800 replays / 40 banker på **3643.67 s**
inklusive opstart/output, og G1c0 **15.73 s** med nul nye replays.
Alle 24 FV-kontroller består. Original dualcelle-TV er afvist;
den særskilte 33x33-hat-lov består på de to records. Digests, råreview og
fulde resultater er registreret i [F4.md](F4.md).

**Næste handling er at hente resultatdokumentationen og merge PR #54 selv,
når dens nye CI er grøn. Ingen flere lokale studiekørsler er nødvendige nu.**
Efter merge synkroniseres `main`; G2's nye forsøgsdesign følger særskilt.

**Kommandoerne nedenfor bevares alene til reproduktion.** De afviser eksisterende
output, så de allerede gennemgåede rapporter ikke overskrives.

Kør på clean **feature/f4-g-grid-refinement før merge**. Brugeren merger selv
efter Windows-kørsel og rå review. **Ingen genkørsel af G0/G1a/G1b0/G1b1/G1b2**
er nødvendig; den eksisterende originale Windows-G1b0 er input.

G1b3 udfører **33.800 replays / 40 banker** på 5/9/17/33/65-gridene og bevarer
alle grove fejl. Samme FV-scheme, opt-in LAPACK og løbende Bayes-reduktion.
**8 workers / BLAS=1** på Windows. G1c0 læser derefter rårapporten og udfører
**ingen nye replays**; alle FV-kontroller og reproduktion mod originalen skal bestå.

Udviklingskørslen: G1b3 **40.57 min** før lagring; G1c0 **10.65 s** før lagring.
Windows-rapporterne: G1b3 **3609.6909 s** og G1c0 **12.4247 s** før lagring;
stopwatch-tiderne inklusive opstart/output er angivet ovenfor.
461 tests og PDF-build består. G1b3's oprindelige dualcelle-TV fejler fortsat;
G1c0's særskilte positive hat-lov består 33→65 på to records. Samme prior,
støtte, scores, evidence og integreret state-mixture; ændret kontinuert
rekonstruktion, eksplicit dokumenteret i F4-status/noter. Dette er ikke recovery.

**En G1b3-gatefejl med gemt komplet rapport er forventet ud fra Linux-resultatet.**
Kommandoerne nedenfor fortsætter til G1c0, som genvaliderer hele rapporten og
stopper ved manglende data, FV- eller reproduktionsfejl. Begge originale
resultater bevares, og output overskrives aldrig.

```powershell
$gStatus = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Git status fejlede." }
if ($gStatus) { throw "Working tree er ikke ren." }

git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }
git switch feature/f4-g-grid-refinement
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin feature/f4-g-grid-refinement
if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }
git log -1 --oneline
if ($LASTEXITCODE -ne 0) { throw "Git log fejlede." }
$gStatus = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Git status fejlede." }
if ($gStatus) { throw "Working tree er ikke ren." }

$gOriginal = "files/output/bm_1/runs/estimation/f4_g_g1b0_joint_parameter_grid.json.gz"
$gDense = "files/output/bm_1/runs/estimation/f4_g_g1b3_joint_parameter_grid_dense_v4.json.gz"
$gHat = "files/output/bm_1/runs/estimation/f4_g_g1c0_joint_hat_reconstruction_v1.json.gz"
if (-not (Test-Path -LiteralPath $gOriginal -PathType Leaf)) { throw "Den oprindelige G1b0-rapport mangler." }
foreach ($gFile in @($gDense, $gHat)) {
    if (Test-Path -LiteralPath $gFile) { throw "Output findes allerede; behold filen til review: $gFile" }
}

python -B -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw "Tests fejlede." }

$gTimer = [System.Diagnostics.Stopwatch]::StartNew()
python -B -m src.exe.run_model.bm_1.estimate joint-bank-grid-dense `
  --g1b0-report $gOriginal --workers 8 --blas-threads 1 --output $gDense
$gDenseExit = $LASTEXITCODE
$gTimer.Stop()
Write-Host ("G1b3 wall time inkl. opstart/output: {0:N2} s; exit={1}" -f $gTimer.Elapsed.TotalSeconds, $gDenseExit)
if (-not (Test-Path -LiteralPath $gDense -PathType Leaf)) { throw "G1b3 gav ingen rapport; stop her og send terminaloutput." }
if ($gDenseExit -ne 0) {
    Write-Host "G1b3 er ikke accepteret. G1c0 validerer nu hele rapporten; FV- eller reproduktionsfejl stopper."
}

$gTimer.Restart()
python -B -m src.exe.run_model.bm_1.estimate joint-bank-reconstruct `
  --grid-report $gDense --g1b0-report $gOriginal --blas-threads 1 --output $gHat
$gHatExit = $LASTEXITCODE
$gTimer.Stop()
Write-Host ("G1c0 wall time inkl. validering/output: {0:N2} s; exit={1}" -f $gTimer.Elapsed.TotalSeconds, $gHatExit)
if ($gHatExit -ne 0) { throw "G1c0 fejlede; behold begge rapporter og terminaloutput til review." }
Write-Host "Kontrollerne er afsluttet. Send begge nye .json.gz-filer og terminaloutput til raw review foer merge."
```

Send **begge nye `.json.gz`-filer** og terminaloutput med testantal/stopwatch,
også hvis en gate fejler. Behold den gamle G1b0-fil uændret. Et bestået G1c0
gælder alene 33x33-hat-loven på de to records; original dualcelle-accept,
held-out population, G2 og uge-/månedsruntime følger ikke automatisk.

## G2a v1: afsluttet syvdages Windows-kørsel

PR #54 er merged på `main` som `014446c`. G2a udvikles separat på
**feature/f4-g-robustness-pilot** / PR #55; brugeren merger efter Windows-kørsel
og rå review. Designet er preregistreret før recordgenerering i `e058299`.
Éndags Linux-piloten fra clean `67968a9` består alle 12 FV- og 2 hat-kontroller;
uafhængigt råreview er afsluttet. Det accepterer kun ét døgn på Linux.
Læs [F4.md](F4.md) for evidens, låste IDs, truth-/FV-/hat-gates og afgrænsning.

**Windows-resultatet er rå-reviewet:** clean `86dc765`, alle 33.800 replays færdige,
28646.89 s samlet, truth/raw/probability består, men rum- og hat-gates fejler.
Exit 1 følger efter lagring af den komplette rapport. **Ingen genkørsel på grund
af manglende terminalhistorik. G2b/G2c må ikke startes med denne kandidat.**
Alle kontroller består gennem dag 1; dag 3 og 7 afvises. Bevar originalfilen.
PowerShell-blokken nedenfor er reproduktionsvejledning, ikke næste kørsel.

Kørslen bruger de **tre eksisterende** G1b0/G1b3/G1c0-rapporter som validerede
dependencies, uden at gentage deres replays. Nye ID1358/1359 køres med
**33.800 replays / 16 banker**, 8 workers og BLAS=1. Hver hypotese fortsætter
egen state gennem midnat. Syv døgn leverer dag 1/3/7 som prefixer.

**Det oprindelige estimat var 6–9 timer; den afsluttede Windows-kørsel målte
7 timer og 57 minutter inklusive opstart og output.**
En valgfri éndagskontrol (`--days 1`, eget nyt outputnavn) estimeres til
45–90 minutter og certificerer kun ét døgn. Den er ikke påkrævet før en
direkte syvdageskørsel; særskilte kortere kørsler giver ekstra arbejde.
Hold maskinen vågen og repoet uændret under kørslen, så source-provenance bevares.

```powershell
$g2Status = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Git status fejlede." }
if ($g2Status) { throw "Working tree er ikke ren." }
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }
git switch feature/f4-g-robustness-pilot
if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
git pull --ff-only origin feature/f4-g-robustness-pilot
if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }
git log -1 --oneline
if ($LASTEXITCODE -ne 0) { throw "Git log fejlede." }
$g2Status = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Git status fejlede." }
if ($g2Status) { throw "Working tree er ikke ren." }

$g2Original = "files/output/bm_1/runs/estimation/f4_g_g1b0_joint_parameter_grid.json.gz"
$g2Dense = "files/output/bm_1/runs/estimation/f4_g_g1b3_joint_parameter_grid_dense_v4.json.gz"
$g2Hat = "files/output/bm_1/runs/estimation/f4_g_g1c0_joint_hat_reconstruction_v1.json.gz"
$g2Output = "files/output/bm_1/runs/estimation/f4_g_g2a_continued_7d_v1.json.gz"
foreach ($g2File in @($g2Original, $g2Dense, $g2Hat)) {
    if (-not (Test-Path -LiteralPath $g2File -PathType Leaf)) { throw "Input mangler: $g2File" }
}
if (Test-Path -LiteralPath $g2Output) { throw "Output findes allerede; behold filen til review." }
python -B -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) { throw "Tests fejlede." }

$g2Timer = [System.Diagnostics.Stopwatch]::StartNew()
python -B -m src.exe.run_model.bm_1.estimate joint-bank-continued-pilot `
  --g1b0-report $g2Original --grid-report $g2Dense --hat-report $g2Hat `
  --days 7 --workers 8 --blas-threads 1 --output $g2Output
$g2Exit = $LASTEXITCODE
$g2Timer.Stop()
Write-Host ("G2a samlet tid: {0:N2} s; exit={1}" -f $g2Timer.Elapsed.TotalSeconds, $g2Exit)
if (-not (Test-Path -LiteralPath $g2Output -PathType Leaf)) { throw "Ingen rapport; send terminaloutput til review." }
Write-Host "Send den nye G2a-rapport og terminaloutput til raw review før merge, også ved fejlet gate."
```

En fejlet truth-gate gemmes før replays, og komplette numeriske afvisninger
gemmes før nonzero exit. Et replay, der fejler midt i afviklingen, afbryder;
ingen overlevende delpopulation kaldes komplet. Originale dependencies
og deres afvisninger bevares. G2a er en numerisk pilot på to nye records;
sampling/mismatch, population, recovery og ugeforecast følger særskilt.

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

## Posthoc G2a-grid-diagnose efter den negative syvdagespilot

Diagnosen i PR #57 er allerede kørt på Linux fra clean `69848458`, BLAS=1:
67 syvdages-replays og 64 parameterakse-sammenligninger på de gamle ID1358/1359.
Samlet 183.2609 s før lagring; detaljer og selvstændigt review står i F4-status.
**Ingen ny lokal kørsel kræves alene for review/merge af denne PR.**

Forslaget til næste pilot er rho/Xi 9×65→17×129 med 1601/3201 rumceller,
og støjpar 33×65→65×129 med 6401/12801 celler. Dette er nu fastlagt og implementeret som særskilt v2 nedenfor, men endnu ikke
accepteret: hele den nye pilot skal køres og rå-reviewes. Den gamle G2a-CLI har fortsat uændret 33/65-design;
brug ikke dens numerics/counts-overrides som registreret accept.

Følgende er kun reproduktion af diagnosen fra et rent checkout af PR #57
eller senere merged source. Behold den negative originalfil:

```powershell
$gridDiagRoot = "files/output/bm_1/runs/estimation"
$gridDiagOutput = "$gridDiagRoot/f4_g_g2a_grid_diagnosis_v1.json"
$gridDiagLog = "$gridDiagRoot/f4_g_g2a_grid_diagnosis_v1.txt"
foreach ($gridDiagPath in @($gridDiagOutput, $gridDiagLog)) {
    if (Test-Path $gridDiagPath) { throw "Behold eksisterende fil: $gridDiagPath" }
}
if (git status --porcelain) { throw "Working tree er ikke ren; behold dine ændringer." }
New-Item -ItemType Directory -Force -Path $gridDiagRoot | Out-Null
python -B -m src.exe.run_model.bm_1.estimate joint-bank-grid-diagnose `
  --g2a-report "$gridDiagRoot/f4_g_g2a_continued_7d_v1.json.gz" `
  --rho-levels 4 --blas-threads 1 --output $gridDiagOutput `
  2>&1 | Tee-Object -FilePath $gridDiagLog
$gridDiagExit = $LASTEXITCODE
if ($gridDiagExit -ne 0) { throw "Diagnosen fejlede; behold rapport/log til review." }
```

Output er eksplicit `registered_design=False`, `certifies_g2a=False`.
Negativ G2a v1 er et tilladt input til denne diagnose; dens acceptance ændres
ikke. De oprindelige scores/observationer genvalideres, og gamle conditional
scores reproduceres. State-L1 er diagnostik; conditional state-historier er
ikke gemt til uafhængig genberegning. G2b/G2c-prerequisites er uændrede.

## Genoptagelig rektangulær G2a v2

**Afsluttet 3. oktober 2026:** Windows-run
`f4_g_g2a_rectangular_v2_20261003_135421` på ren `01c7c153` er komplet og
rå-reviewet: 8/8 units, 16 banker, 42.312 replays, alle 14 kontroller bestået,
exit=0. Ingen ny G-kørsel er nødvendig. De følgende start/resume-kommandoer er
reproduktionsvejledning; det afsluttede run-arkiv og dets source bevares.
Næste aktive arbejde er H1 og H's datakontrakt; G2b/G2c forbliver blokeret.

Designet er fastlagt efter den negative v1 og diagnosen. CLI'en låser rho/Xi
9×65→17×129 på 1601/3201 rumceller og støjpar 33×65→65×129 på 6401/12801
celler; h=1/1536 mod 1/3072 og q=3 mod 5 kontrolleres på begge grids.
42.312 replays med cached FV; samme gamle records, prior, støtte og gates.
Dette er udvikling, ikke held-out recovery eller forecast-certificering.

Start med én blok efter merge fra et rent checkout. Behold hele run-mappen og
den afviste v1-rapport. Send `summary.json` og terminalfilen til review af lokal
bloktid, før hele piloten køres. Denne blok bliver genbrugt ved fortsættelse.
`chunk-size` og `limit-chunks` ændrer kun afvikling/checkpointfrekvens.

```powershell
if (git status --porcelain) { throw "Working tree er ikke ren; behold dine ændringer." }
$rectRoot = "files/output/bm_1/runs/estimation"
$rectInput = "$rectRoot/f4_g_g2a_continued_7d_v1.json.gz"
$rectTag = Get-Date -Format "yyyyMMdd_HHmmss"
$rectDir = "$rectRoot/f4_g_g2a_rectangular_v2_$rectTag"
$rectLog = "$rectRoot/f4_g_g2a_rectangular_v2_$rectTag.txt"
if (!(Test-Path $rectInput)) { throw "Den komplette negative v1-rapport mangler: $rectInput" }
if (Test-Path $rectDir) { throw "Behold eksisterende run-mappe: $rectDir" }
python -B -m src.exe.run_model.bm_1.estimate joint-bank-rectangular-pilot `
  --g2a-report $rectInput --run-dir $rectDir --workers 8 `
  --chunk-size 128 --limit-chunks 1 --blas-threads 1 `
  2>&1 | Tee-Object -FilePath $rectLog
$rectExit = $LASTEXITCODE
if ($rectExit -ne 0) { throw "Blokkørsel fejlede; behold run-mappe/log til review." }
Write-Host "Send: $rectDir/summary.json og $rectLog"
```

Fortsæt samme mappe med samme rene source; ændr ikke git-version under en
igangværende kørsel. Hvis en ny PowerShell-session er åbnet, sæt `$rectDir`,
`$rectInput` og `$rectLog` til de eksisterende stier før følgende kommando.

```powershell
if (git status --porcelain) { throw "Working tree er ikke ren." }
$rectSource = (Get-Content "$rectDir/manifest.json" -Raw | ConvertFrom-Json).source.source_commit
if ((git rev-parse HEAD) -ne $rectSource) { throw "Genstart kræver source $rectSource" }
python -B -m src.exe.run_model.bm_1.estimate joint-bank-rectangular-pilot `
  --g2a-report $rectInput --run-dir $rectDir --workers 8 `
  --chunk-size 128 --blas-threads 1 `
  2>&1 | Tee-Object -FilePath $rectLog -Append
$rectExit = $LASTEXITCODE
Write-Host "Summary: $rectDir/summary.json; exit=$rectExit"
if ($rectExit -ne 0) { throw "Numerisk/raw gate eller kørselsfejl; behold alle data til review." }
```

`--collect-only` med samme input og mappe genvaliderer færdige råunits og det
rullende checkpoint uden nye FV-replays. En fuld negativ pilot returnerer
exit≠0, efter alle tilgængelige units og summary er gemt. En begrænset kørsel
returnerer normalt exit=0 og `completed=False`, `passes=False`; det er ikke i
sig selv en numerisk afvisning. Se `failed_comparisons` særskilt.

Run-mappen har `manifest.json`, `summary.json`, ét atomisk rullende
`checkpoint.npz` og otte endelige `units/*.npz`. Formatet er float64 NPY + JSON
metadata i ZIP uden pickle. Behold hele mappen til raw review; summary alene
er ikke videnskabelig accept. Der gemmes kun det seneste checkpoint med alle hidtil reducerede hypoteser;
en afbrydelse mister højst igangværende blok. OS-lås hindrer samtidige
skrivere. Store v1-rådata genvalideres ved hver invocation; opstart kan derfor
være længere end selve den første blok.

Linux-kilde `502d0506` har målt to blokke med reel restore (256 hypoteser),
men denne tidlige Linux-måling gav ingen fuld unit eller accept. Det komplette
Windows-resultat er nu registreret i F4-status. **G2b/G2c forbliver blokeret**, selv om v2 består,
indtil deres grid/settings og dependency-kontrakt er registreret særskilt.

Det præcise slutresumé er gemt som
`files/output/bm_1/runs/estimation/g2a_rectangular_windows_v2_summary.json.gz`;
det uafhængige review hedder `g2a_rectangular_windows_v2_raw_review.json`.
Råreviewets 510 arraykontroller og alle 168 prefixer kræver nul nye FV-replays.
De otte NPZ-hashes står i reviewet; behold også originalt manifest og terminalfil.
Den uploadede gamle `summary.json` og partial checkpoint er overhalet af de
komplette units og må ikke bruges som slutstatus. Sidste fortsættelses ydre
timer var 144.75 min; totalen på tværs af alle invocations er ikke registreret.

Kun hvis reviewet skal reproduceres, kan følgende køres fra ny `main` med adgang
til det **uændrede afsluttede arkiv**. Scriptet er afgrænset til dette registrerede
Windows-resultat på `01c7c153`; output må ikke findes i forvejen. Det reproducerer
truth/observationer og råaritmetik, uden conditional FV-replays eller ændringer
i run-mappen. Individuelle conditional state-historier er ikke gemt og kan
derfor ikke rekonstrueres fra arkivet alene.

```powershell
python -B root/f4/review_g2a_rectangular.py `
  --run-dir "files/output/bm_1/runs/estimation/f4_g_g2a_rectangular_v2_20261003_135421" `
  --v1-report "files/output/bm_1/runs/estimation/f4_g_g2a_continued_7d_v1.json.gz" `
  --log "files/output/bm_1/runs/estimation/f4_g_g2a_rectangular_v2_20261003_135421.txt" `
  --output "files/output/bm_1/runs/estimation/g2a_rectangular_windows_v2_raw_review_reproduced.json"
if ($LASTEXITCODE -ne 0) { throw "Råreview fejlede; behold alle originalfiler." }
```

## G2b-v2 som nat-sidearbejde: cirka otte timer per invocation

H1/H's datakontrakt, H2/H3 og endelig I/J er fortsat hovedsporet. Dette er en
særskilt registreret G2b-v2-kørsel, som genbruger de accepterede rektangulære
G2a-v2-grids og kontrollerer dem igen på alle fem scenarier og nye ID1360/1361.
De gamle kvadratiske G2b/G2c-kommandoer nedenfor forbliver blokeret.
G2c startes ikke automatisk og kræver senere kompatibelt design og G2b-accept.

Behold original G2a-v1-rapport og hele afsluttede G2a-v2-run-mappen. Python
råvaliderer alle otte gamle units uden nye FV-replays før G2b; summary alene
accepteres ikke. Første opstart kan derfor tage nogle minutter.

**Første korte lokale afviklingspilot:** brug kommandoen nedenfor med
`-LimitChunks 1` på launcher-scriptet. Den gemte blok genbruges ved natkørslen;
undlad flaget for fuld afvikling. En enkelt blok tester lokal afvikling, ikke numerisk
G2b-accept. Default `--chunk-size 64` giver regelmæssige checkpoints.

**Indlæs launcheren fra opdateret main**, efter den tidligere Python-kørsel er
stoppet. Scriptet opretter kun en lokal `night/g2b-v2`-branch ved første invocation;
ved resume skifter det til manifestets oprindelige source før Python. Dermed kan
nyere launcher-rettelser bruges uden at ændre eksisterende beregninger, manifests
eller råunits. Der oprettes intet worktree. Skift eller pull aldrig under en
aktiv Python-kørsel. Kommandoen nedenfor er en fuld 7,5-timers fortsættelse med
12 workers; tilføj `-LimitChunks 1` til sidste kommando for en kort pilot.

```powershell
& {
    if (git status --porcelain) { throw "Lokale ændringer fundet; behold dem." }
    git switch main
    if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
    git pull --ff-only origin main
    if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }

    powershell -NoProfile -ExecutionPolicy Bypass `
        -File .\root\f4\run_g2b_night.ps1 -MaxHours 7.5 -Workers 12
    if ($LASTEXITCODE -ne 0) { throw "Kørslen fejlede; behold alle filer og logs." }
}
```

Efter en pause viser summary normalt `completed=False`, `passes=False` og
`stop_reason=time_budget`. Det er en gemt delkørsel, ikke en numerisk fejl.
`failed_comparisons` viser reelle afvisninger; de stopper nyt studiearbejde.
En komplet bestået kørsel kræver 40 units, 80 banker og alle 70 kontroller.
Behold manifest, summary, checkpoint, log og alle units. En ny invocation
starter ved første ikke-gemte hypotese. Ved hård afbrydelse kan den igangværende
ikke-publicerede blok skulle gentages. Otte timer kan overskrides af sidste
blok, lagring og slutopsamling; computeren slukkes ikke automatisk.

Ved Windows sharing violations (32/33) på checkpointet kan den nye launcher
starte den samme pinned source igen, højst tre ekstra gange (`-MaxIoRetries`).
Der genvalideres før nyt arbejde, gemte blokke genbruges, alle forsøg har egne
logs, og det resterende fælles budget bruges. En pilots allerede gemte blokke
tælles med i `-LimitChunks`; retry giver ikke et nyt blokbudget. Andre fejl
genstartes ikke automatisk. Nye beregningsversioner bruger korte I/O-retries
og tillader udsat oprydning af et overflødigt fuldt checkpoint efter sikker
publicering; et delvist checkpoint eller en fejlet publicering droppes aldrig.

Når Python er stoppet, kan dagsarbejdet fortsætte på main:

```powershell
& {
    git switch main
    if ($LASTEXITCODE -ne 0) { throw "Branchskift fejlede." }
    git pull --ff-only origin main
    if ($LASTEXITCODE -ne 0) { throw "Git pull fejlede." }
}
```

Lad nat-branchen blive på sin oprindelige commit. Næste nat bruges samme script
og samme run-mappe, ikke en ny kopi. `--collect-only` med samme paths kan
råvalidere gemt arbejde uden nye FV-replays. Progress viser grov resterende
og samlet tid for det resterende studie; tidligere gemt arbejde indgår ikke
i hastighedsmålingen. Fuld numerisk accept og et efterfølgende råreview mangler,
selv om afviklings- og genoptagelsestests består.

## G2b/G2c og hurtigere FV (efter syvdages G2a-review)

**G2a v1 på `86dc765` er færdig, rå-reviewet og numerisk afvist. G2b/G2c er
blokeret på denne kandidat.** Der kræves ingen ekstra fuld G2a-kørsel alene for
performanceændringer eller manglende terminalhistorik. Den korte backendkontrol
nedenfor kan udføres på performancebranchen efter brugerens PR-review; den giver
ingen accept af syvdages numerik. En ny numerisk pilot skal fastlægges særskilt.

**Windows-backendkontrollen er afsluttet og reviewet** på clean `b5e5e4d`:
490 tests består, alle 40 parrede kombinationer består med registrerede
nul-afvigelser, og målt serielt speedup er **3.9103x**. Samlet benchmarktid
409.82 s. Kør den ikke igen alene for merge af PR #56; resultater og digests
står i F4-status. G2a v1's numeriske afvisning og G2b/G2c-blokering er uændret.

Følgende er reproduktionsvejledning på de gamle G2a-records, med fire hjørner
og centrum per par, alle fire FV-settings og to gentagelser. Native Python-output
fanges eksplicit, da `Start-Transcript` ikke fangede det i brugerens PowerShell:

```powershell
$fvWindowsRoot = "files/output/bm_1/runs/estimation"
$fvWindowsOutput = "$fvWindowsRoot/fv_backend_windows_v1.json"
$fvWindowsLog = "$fvWindowsRoot/fv_backend_windows_v1.txt"
foreach ($fvWindowsFile in @($fvWindowsOutput, $fvWindowsLog)) {
    if (Test-Path -LiteralPath $fvWindowsFile) { throw "Behold eksisterende fil: $fvWindowsFile" }
}
New-Item -ItemType Directory -Force -Path $fvWindowsRoot | Out-Null
python -B -m src.exe.run_model.bm_1.estimate benchmark-fv `
  --days 7 --repeats 2 --blas-threads 1 --output $fvWindowsOutput `
  2>&1 | Tee-Object -FilePath $fvWindowsLog
$fvWindowsExit = $LASTEXITCODE
if ($fvWindowsExit -ne 0) { throw "FV-backendkontrollen fejlede; behold begge filer." }
```

Rapporten måler seriel replaytid. Parallel walltime, opstart, reduktion, råreview
og output har også en pris; et speedup-tal er ikke et løfte om samlet Windows-tid.
`joint-bank-continued-pilot --fv-backend cached` findes til eksplicit sammenligning,
men default er fortsat den gamle LAPACK-backend.

**De følgende G2b/G2c-blokke er betinget reproduktionsvejledning, ikke aktuelle
arbejdsordrer. De må ikke startes med den afviste G2a v1-fil.** Den rektangulære
v2 ændrer ikke automatisk disse kommandoers gamle grid/dependency-kontrakt.
Et kompatibelt registreret design og råreview er påkrævet før frigivelse.
**Kør derefter først én G2b-unit og review runtime/resultat**, inden
resten startes. Det samlede G2b-design er ti units; G2c er 560. Det er forsknings-
studier og indgår ikke i den almindelige testsuite.

```powershell
$g2Root = "files/output/bm_1/runs/estimation"
$g2a = "$g2Root/f4_g_g2a_continued_7d_v1.json.gz"
$g2b = "$g2Root/f4_g_g2b_robustness_v1"
$g2c = "$g2Root/f4_g_g2c_population_v1"

python -B -m src.exe.run_model.bm_1.estimate joint-bank-robustness-pilot `
  --g2a-report $g2a --run-dir $g2b --limit-units 1 --workers 8 --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "G2b-unit fejlede; behold hele mappen til review." }
```

Samme kommando fortsætter fra næste manglende unit. Udelad `--limit-units 1`
for at færdiggøre resten efter review. G2c kræver alle ti beståede rå G2b-units:

```powershell
python -B -m src.exe.run_model.bm_1.estimate joint-bank-robustness-population `
  --g2a-report $g2a --pilot-dir $g2b --run-dir $g2c `
  --limit-units 10 --parallelism units --workers 8 --blas-threads 1
if ($LASTEXITCODE -ne 0) { throw "G2c fejlede; behold hele mappen til review." }
```

`--limit-units` begrænser nye beregninger, ikke studiedesignet. Et succesfuldt
chunk-exit er ikke samlet accept: se `complete` og `passes` i `summary.json`.
`--collect-only` genvaliderer og opsummerer eksisterende units uden replays.
`manifest.json`, `units/*.npz` og `summary.json` skal bevares sammen; de store
arrays er tabsfrie NPY-medlemmer plus metadata i ZIP, ikke pickle. En unit er én
record/scenarie med alle sine banker. Ved afbrydelse genbruges komplette units;
den igangværende unit beregnes igen. Skift ikke kildecommit under en serie.
En efterladt `run.lock` blokerer ikke resume; låsen holdes af operativsystemet.
Ingen fejlede eller manglende records fjernes for at få et grønt resumé.

**Procesplacering:** `--parallelism units` fordeler komplette record/scenarier
over én persistent pool. Hver worker udfører hypoteserne serielt i samme orden
og sender kun de reducerede banker tilbage. `--workers` er det samlede budget,
ikke et antal ydre gange indre workers. G2c bruger dette som default. G2b's
default `hypotheses` bevarer parallelismen i en enkelt pilot-unit; `units` kan
vælges ved flere units efter lokal måling. Råvalidering og lagring sker i
hovedprocessen. Progress i `units` vises ved færdige units; ingen output imellem
betyder ikke nødvendigvis et hængt job. Resuméets rækkefølge er uafhængig af,
hvilken worker der afslutter først.

Ved en fejlet gate starter ingen nye units, men allerede indsendte units får
lov at afslutte og gemmes. Kommandoen returnerer derefter fejl. Ved hård
afbrydelse kan op til `workers` igangværende units kræve genkørsel. Eksisterende
source-/dependency-kontrol, lås og atomisk lagring er uændret.

En kort, selvstændig afviklingsmåling kræver ingen G2a-fil og accepterer ingen
studier. Kør fra en branch med **ny source**. Bevar source, hvis en aktiv låst
kørsel stadig bruger checkoutet; kun da kan et separat worktree være nødvendigt:

```powershell
$executionTag = Get-Date -Format 'yyyyMMdd_HHmmss'
python -B -m src.exe.run_model.bm_1.estimate benchmark-execution `
  --workers 2 --repeats 2 --blas-threads 1 `
  --output "files/output/bm_1/runs/estimation/execution_$executionTag.json"
if ($LASTEXITCODE -ne 0) { throw "Afviklingskontrollen fejlede; behold output/log." }
```

Benchmarken bruger to identiske syvdages-banker fra gammelt ID1359, hver med
3×3 hypoteser, FV3201/h=1/1536/q=3, og skifter kørselsrækkefølge mellem repeats.
Den måler spawn, beregning, reduktion og overførsel; truth-setup, efterfølgende
råvalidering og filskrivning er udenfor de sammenlignede tider. Ni rå numeriske
felter sammenlignes med tolerance 1e-12. To units kan højst udnytte to samtidige
unit-workers; denne diagnostik er ikke en måling af otte-worker skalering.
Se måleresultatet og dets begrænsninger i [F4.md](F4.md).

H har forberedelsesfunktioner i `src/tools/bm_1_data.py`:
`previous_weekday_mean`, `EstimationPowerScale` og `historical_point_observations`.
De kræver eksplicit timezone, population, power-unit og observationstype.
De er ikke et historisk fit. Afklar først punkt versus interval/energi, faktisk
forbrugerpris, historik og kronologisk split. F3's retrospektive eksperiment er bevaret.


## Sigma_y-batching til kommende forsøg

PR #59 tilføjer en opt-in multiple-RHS-filtervej og en særskilt benchmark.
De eksisterende G2a-v2/G2b/G2c-kommandoer ændrer ikke backendafvikling. G2a-v2
skal færdiggøres på manifestets oprindelige revision, før checkout opdateres.
Efter merge og afsluttet igangværende kørsel kan backendpanelet måles lokalt:

```powershell
if (git status --porcelain) { throw "Working tree er ikke ren." }
$batchRoot = "files/output/bm_1/runs/estimation"
$batchTag = Get-Date -Format "yyyyMMdd_HHmmss"
$batchOutput = "$batchRoot/fv_sigma_y_batch_windows_$batchTag.json.gz"
$batchLog = "$batchRoot/fv_sigma_y_batch_windows_$batchTag.txt"
New-Item -ItemType Directory -Force -Path $batchRoot | Out-Null
python -B -m src.exe.run_model.bm_1.estimate benchmark-fv-batch `
  --days 7 --repeats 2 --batch-size 4 --blas-threads 1 --output $batchOutput `
  2>&1 | Tee-Object -FilePath $batchLog
$batchExit = $LASTEXITCODE
Write-Host "Rapport: $batchOutput; terminaloutput: $batchLog; exit=$batchExit"
if ($batchExit -ne 0) { throw "Batchkontrol fejlede; behold rapport/log." }
```

Panelet sammenligner alle prefix-scores og state-masser, actual steps og
uniform-bank evidence/vægte på samme gamle syvdages-record. Det gemmer alle
scores samt sidste prefix-state for begge backends i sidste gentagelse;
fulde tidligere state-historier indgår i eksekveringskontrollen, men lagres ikke.
Rapporten kræver ren source og overskriver ikke output. Et bestået panel
certificerer backendækvivalens på panelet, ikke G2-konvergens eller population.
Serial speedup er målt mod den allerede cached backend, ikke den gamle Thomas-
eller uncached-LAPACK-vej. Ingen Windows-/parallel-runtime kan udledes alene
fra Linux-panelet.

Intern streaming kan bruge `batch_sigma_y=4` på en cached sigma_x/sigma_y-gruppe.
Grupper har samme record, fysisk processoperator, grid og observationstider.
Hver hypotese beholder sin egen conditional state og Bayes-update. Hypoteseorden
og den eksisterende bankreduktion er uændret. Batches krydser ikke sigma_x og
queue er bounded til én batch per worker; maksimal batchbredde er fire.
Dette flag er ikke tilføjet til de gamle videnskabelige CLI'er.

`run_fv_sigma_y_batch` og `collapse_sigma_y_for_process_forecast` findes i
`src/method/bm_1_batch_filter.py`. Forecast-hjælperen summerer kun sigma_y ved
fast sigma_x i en repræsenteret node-bank efter sidste assimilation. Den er
eksakt for state/demand og forventet F3-objective under fælles fremtidig policy,
men kan ikke erstatte den fulde bank til Y-predictive density eller nye målinger.
En marginal state alene er heller ikke nok til helperen: de betingede
state–parameter-laws skal være bevaret ved origin.

`finite_volume_1d_fitted` er desuden en eksperimentel eksponentielt fitted
rumflux for samme Ito-FPE, inklusive D′. Den kan bruges i interne diagnostiske
grupper med `backend='fitted'`, men eksisterende videnskabelige CLI'er beholder
deres registrerede numerik. Se `root/F4.md` for gamle-data-panel, bevarede
all-node-fejl og de manglende tids-/quadrature-/parameterkontroller. Færre
celler er en ny numerisk kandidat, ikke en backendækvivalens fra batchpanelet.
