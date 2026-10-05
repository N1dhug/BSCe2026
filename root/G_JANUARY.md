# G0–G3: januar-fit, WM-lignende start og valgfri ugeprognose

Status 5. oktober 2026: kode implementeret, fokuserede tests består, syntetisk
månedsintegration gennemført. Historisk januar-fit er ikke kørt her: connectoren
gav ikke adgang til bytes i den store `input.xlsx`. Det er et datatilgængeligheds-
problem, ikke et negativt numerisk fitresultat. [G-planen](G.md) er det samlede spor.

## Vis fittet for januar

Seneste bestilling: vis det historiske januar-fit med virkelig DK1-pris og en
start valgt efter WM's neutrale konvention. Kommandoen er:

```powershell
python -B -m src.exe.run_model.bm_1.deterministic --month 2022-01 --tag MED --fit-only --initial-state wm-neutral
```

`--fit-only` kræver kun januarens forbrug, pris og temperatur; ingen februar-
inputs, scoring eller forecast-solves. De historiske rå DK1-priser læses fra
`prices.xlsx`, holdes konstante inden for hver time og optimeres ikke. Råprisen
gemmes som `Price_DKK_MWh`; figuren viser samme signed U som WM.

Den aktuelle WM-kilde `fit_daily.INIT_F` er 0 (main e6075c04). Vi genbruger
`fitted_initial_condition`: F0=INIT_F og R0=I0-O0. BM starter fast fra
x0=(1-INIT_F)/2=0.5, så signed koordinat 1-2x0=0 også er neutral. Det er en
fælles neutral konvention, ikke en påstand om identiske fysiske tilstande.
WM4-noternes F0=0.5 er ældre end den aktuelle kode. Ingen optimering af BM's
x0 i denne variant: kun C,k,Xi estimeres samlet over månedens 744 timer.
`--initial-state fit` bevarer det tidligere fælles fireparameter-fit som en
udtrykkelig alternativ udviklingsvariant.

Output: `fit.png` for hele januar, `fit_week_01.png`–`fit_week_05.png` for
læselige udsnit af det samme månedlige fit. Figurerne følger WM's fire-panel-
format dér hvor muligt: observationer og fine fitted kurver/baseline,
baseline og pris, forbrugskorrektioner, separate states. WM's acceleration
har ingen identisk BM-state; R_dot gemmes i CSV uden at blive kaldt BM-rate.
BM's signed 1-2x og WM's F vises med tydelige separate betegnelser.

`fit_predictions.csv` indeholder rå forbrugsenheder og råpris;
`fit_predictions_normalized.csv` indeholder timeprøver på WM-lignende skala.
`fit_signals_normalized.csv` indeholder den fine displaybane, BM x/signed state,
BM x_dot (per time), WM F/R/R_dot (WM's modeltid i dage) og råpris.
Den fine BM-displaybane beholdes kun for det valgte fit; optimizerkald
allokerer ikke fuld plot-historik. Refined demand skal fortsat bestå tolerancen.

Terminalen viser C,k,Xi,x0, WM's r og initialstate samt samme
MSE/RMSE/MAE/bias-tabel som WM, både normaliseret og i rå forbrugsenheder.
Daily/performance/weekday-CSV'er genbruger WM's analysehelpers med prefix
`fit_`. Alle disse januarfejl er in-sample: de må ikke kaldes prognosekvalitet.
Der er ét parameterfit for måneden, ingen daglige resets eller genfit for figurer.

## Kørsel på eksisterende checkout

PR #66 og #68 er merged. Den efterfølgende datadækning/audit-rettelse ligger
i `feature/g-january-data-audit`. Når ingen aktiv kørsel bruger checkoutet,
kan hele januarvisningen med rettelsen køres direkte:

```powershell
git fetch origin
if ($LASTEXITCODE -ne 0) { throw 'Fetch fejlede.' }
git switch feature/g-january-data-audit
if ($LASTEXITCODE -ne 0) { throw 'Branchskift fejlede.' }
git pull --ff-only origin feature/g-january-data-audit
if ($LASTEXITCODE -ne 0) { throw 'Git-opdatering fejlede.' }

$outputDir = Join-Path 'files/output/bm_1/runs/deterministic' ('g_january_fit_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
python -B -m src.exe.run_model.bm_1.deterministic --month 2022-01 --tag MED --fit-only --initial-state wm-neutral --output $outputDir
$fitExitCode = $LASTEXITCODE
if (Test-Path (Join-Path $outputDir 'fit.png')) { Start-Process (Join-Path $outputDir 'fit.png') }
if ($fitExitCode -ne 0) { throw 'Fit/sammenligning er uafsluttet eller afvist; BM-figuren åbnes hvis den er tilgængelig. Se den viste outputmappe.' }
```

Efter merge af audit-rettelsen kan både fit-only og fit+prognose køres på main:

```powershell
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw 'Git-opdatering fejlede.' }

python -B -m src.exe.run_model.bm_1.deterministic --month 2022-01 --tag MED
if ($LASTEXITCODE -ne 0) { throw 'G-kørslen er uafsluttet eller afvist; se checkpoint/result/failure i den viste outputmappe.' }
```

Ingen nye worktrees. Input er eksisterende `files/input/data/input.xlsx` og
`prices.xlsx`, plus DMI-timevis temperatur/cache. DMI-fetch omfatter tilgængelig
historik før origin og forecastperioden i separate requests, så eventuel
weather-imputation ikke krydser cutoff; ikke hele sæsonen. Ekstern hentning og
workbooklæsning har eget setup; femminuttersloftet gælder BM-parameterfit,
ikke netværksventetid. Input-workbookens rå observed_data bruges før imputation.

Kørslen scorer alle 744 timer i januar og forudsiger 1.–7. februar (168 timer
fra 1. februar kl. 00). Starttilstanden er som standard fast WM-neutral og
videreføres ved origin; warm-up er 0 i månedsvarianten. `--week 2022-01-24`
er særskilt én-fitugesvariant med 24 timers warm-up og samme protokol for BM/WM.

Kort dataaudit uden fit:

```powershell
python -B -m src.exe.run_model.bm_1.deterministic --month 2022-01 --tag MED --fit-only --audit-only
```

Dataaudit fungerer også når input ikke består fitkravet: `data_audit.json`
gemmer for Demand/Price/Temp antal manglende timer, de første otte tidspunkter
og den tilgængelige daterange. Terminalen viser samme korte oversigt.
`--audit-only` udfører ingen parameterfit og returnerer 0, når rapporten er
skrevet; `fit_ready` i rapporten afgør, om data er komplette. Den tidligere
tekst om mindst 168 observationer var misvisende for en 744-timers måned med
et enkelt hul. Rå huller er stadig ikke erstattet med opdigtede observationer.
Vejrhentningen dækker nu den ønskede start, også når første rå demand-time
ligger senere; dette skaber ingen manglende demand-observation.

Pris-workbooken på repoet blev kontrolleret: alle 744 januartimer har en DK1-pris.
Johans lokale stop 5. oktober vedrører endnu uafklaret Demand/Temp-dækning;
rapporten skal afklare det før en ændret datakontrakt eller et nyt fit.

Reproducerbar syntetisk integration uden workbook/DMI:

```powershell
python -B -m src.exe.run_model.bm_1.deterministic --month 2022-01 --synthetic --starts 2 --max-calls 500
```

`--input-csv path.csv` accepterer raw `Datetime,Demand,Price,Temp` med sorteret,
unik, lokal-naiv Europe/Copenhagen-tid. Den lille adapter afviser DST-overgange;
januar/første februaruge har ingen. Prognoseforberedelsen kræver U/Temp, men
ingen fremtidig demand. Den efterfølgende scoring kræver demand.

## Fit og beregningslofter

BM bruger f(x)=1−2x og g(u)=1−2u. Standardfit estimerer
(C_hours,k,Xi) med fast x0=0.5. Med `--initial-state fit` estimeres også x0.
De fælles bounds er (1,0.05,0,0) til (744,10,1,1).
Det er deklarerede udviklingsbounds, ikke fysisk identificerede kapaciteter.
C/k optimeres i logkoordinater med SciPy least_squares/TRF og two-point-Jacobian.
Én residual per rå timeobservation; fine interpolation skaber ikke nye målinger.

Fire faste starter, højst 250 residualkald per start, 1.000 i alt, 300 sekunder.
Alle finite-difference-kald tæller med. Budgetstop giver uafsluttet start;
kun en konvergeret TRAIN-kandidat kan vælges. Foreløbig bedste kandidat
gemmes særskilt og bruges ikke som færdig model. Ingen automatisk ekstra fitrunde.
BLAS=1 og én proces; call-baseret ETA er et groft budgetestimat, ikke et løfte.

Forward bruger skalær BM1-RK4, samme ligninger som eksisterende model, held pris
og fælles lineært interpoleret baseline. 60 trin/time rammer baselinens knæk;
120 trin/time kontrollerer valgt fit og forecast. Kræv maksimal demandforskel
≤1e-5 normaliserede enheder. Separat 60-sekunders final/numerikkvote og en
kort højst tre-solves/60-sekunders pilot. Ingen clipping af state eller observationer.
Den eksakte konstant-input-reference bruges kun i tests, ikke til at ændre baseline.

WM genbruger qp_params og eksisterende deterministisk solver på hele samme
træningsmåned, med én sammenhængende state og videreført simuleret sluttilstand.
Ingen oracle-fit på testugen. Hvis WM's eksisterende solver afviser koefficienter
eller initialstate, gemmes BM/baseline-resultater stadig, med status wm-rejected
og exit 2; det er ikke en fuldført WM-sammenligning. Dets krav løsnes ikke skjult.

## Data og evidensgrænse

Genbrug det samme statistical_output-objekt, robust Huber/ridge, features og
baselinehistorik før origin. Baseline er fittet in-sample på den tilgængelige
historik; normaliseringsskalaer stammer fra januar og bruges identisk for BM/WM.
Original WM's sæson-/årsskala bruges ikke, da den læser senere data. Den ændrede
fælles kontrakt fremgår af manifestet; det er ikke en score for original WM-default.

Signed WM-pris U omregnes u=(U+1)/2 i BM. Hvis en senere pris ligger uden for
træningsdomænet, stopper opsætningen. `--price-bounds LOW HIGH` kan registrere
en fælles ex-ante prisskala i DK1 DKK/MWh; den må ikke vælges ved at kigge på
sluttestens udfald. Demand uden for træningsskalaen scores uden clipping.
Baselinens eksisterende [0,1]-clipping auditeres separat.

Punkt-/intervalbetydning og rå forbrugsenhed er endnu uafklaret. Historisk
kørsel mærkes derfor hourly-point-approximation/development, uden fysisk
kapacitetsclaim eller kausal elasticitetsclaim. Den separate BM-forward-API
understøtter også hourly-mean med energibalance; WM-månedsbenchmark bruger
foreløbig punktoperatoren. Faktisk fremtidig pris/temperatur giver en betinget
historisk prognose. Februar er allerede inspiceret og er udviklingsdata.

Syntetiske data er genereret med BM1, og bruges kun til integration/runtime.
Resultaterne kan ikke bruges til at rangere BM og WM eller som januar-estimater.
Et fit kan have svag identifikation: gem aktive bounds, Jacobianens singularværdier
i log-C/log-k/Xi/x0-koordinater og lokal rang. F/g og støjfit er næste G-trin. Ved fast x0 omfatter Jacobian/rang kun
de tre frie koordinater log-C/log-k/Xi; x0 tælles ikke som identificeret parameter.

## Output og kontroller

Ny runmappe under `files/output/bm_1/runs/deterministic/`:
audit, checkpoint, manifest, fit og result som JSON; fit-tabeller/figurer ovenfor;
ved forecast også predictions og normaliserede predictions; daglige metrics; BM/WM-performance, summary og weekday-CSV; week.png.
Fejl gemmes i failure.json. Kør aldrig igen i samme ikke-tomme mappe;
checkpointet er en bevaret kandidat/status, ikke et eksakt SciPy-resume.

MSE/RMSE/MAE/bias, baselinegevinst og performanceanalyser genbruges fra WM.
Forbrugsmetrics vises i rå workbookenheder; normaliserede resultater gemmes også.
Pooled fejl og gennemsnit af dagsmål er forskellige opgørelser. Figurer viser
observation, fælles baseline, begge prognoser, korrektion, pris og modelstates.

### Minimumskontrol for G0–G3

Standardkørslen er 10 små tests (tidligere 16 G-tests plus 32 layouttests):

```powershell
python -B -m unittest tests.test_bm1_deterministic_fit -q
```

| Test | Fejlrisiko |
| --- | --- |
| 1 | Dynamik og prisskift afviger fra eksisterende adaptiv BM-solver. |
| 2 | Konstantinterval-reference, energibalance eller punkt/interval-operator er forkert. |
| 3 | Nul fleksibilitet, tilstandsgrænser eller stabilitetsværn brydes. |
| 4 | Fit kan ikke genfinde kendte parametre med enten fast eller fittet x0. |
| 5 | Svag identifikation skjules som fire identificerede parametre. |
| 6 | Kald-/tidsbudget overskrides uden korrekt stop/checkpoint. |
| 7 | Prognose kræver fremtidigt forbrug, eller fit-only kræver februardata. |
| 8 | Huller i træningsdata eller ugyldige fremtidige priser accepteres. |
| 9 | Vejrinterpolation krydser trænings-cutoff. |
| 10 | Fremtidigt forbrug påvirker skala, baseline, fit eller prognose; WM-fejl taber BM-resultat. |

Seks særskilte tests er fjernet: scalar-kernel-paritet, punkt/interval-forskel,
generelle argumentchecks, separat deadlinecheck, separat baseline/skala-cutoff
og ex-ante-skalaens happy path. De væsentlige overlappende assertions dækkes
i ovenstående cases; overflødige gentagne setup/solves er fjernet.

Layouttests køres kun ved ændringer i entrypoints/importstruktur. Ved ændringer
i fælles BM/WM-kode køres derudover de berørte regressioner. Fuld projekt-suite
er ikke en standardgate for G. F4-tests bevares til F4-arbejde.
Den tidligere fulde suite tog cirka otte minutter; G-testene er små, så det er
primært den smallere standardkørsel, der sparer tid.

Gentag ikke det syntetiske månedsforsøg eller historisk januar-fit ved rene
tekst-/testændringer. Ved ændret solver, fit eller tidsopdeling køres ét
syntetisk integrationsforsøg; historisk fit køres kun når nye estimater er
nødvendige. Numerisk refinement, datavalidering og budgetværn inde i selve
januarkørslen bevares: de afgør, om dens resultater kan accepteres.
Det tidligere syntetiske 744+168-timers forsøg tog cirka 22 s med to starter og 130 kald i denne
Linux-session; ingen garanti om samme tid/konvergens på historiske Windows-data.

Den nye fit-only-pilot med fast x0=0.5, to starter og 104 residualkald
fuldførte modeldelen på 17.54 s (fit 16.44 s); to refinement-solves bestod.
10 fokuserede tests og 2 berørte import/layoutchecks bestod.
`files/output/bm_1/benchmarks/g_january_fit_preview_20261005.json` registrerer
syntetisk evidens; historisk januar er fortsat ikke kørt her.
