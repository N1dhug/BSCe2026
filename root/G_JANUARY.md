# G0–G3: januar-fit og den efterfølgende uge

Status 5. oktober 2026: kode implementeret, fokuserede tests består, syntetisk
månedsintegration gennemført. Historisk januar-fit er ikke kørt her: connectoren
gav ikke adgang til bytes i den store `input.xlsx`. Det er et datatilgængeligheds-
problem, ikke et negativt numerisk fitresultat. [G-planen](G.md) er det samlede spor.

## Kørsel på eksisterende checkout

Efter merge af PR #66 på main, og når ingen aktiv kørsel bruger checkoutet:

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
fra 1. februar kl. 00). Starttilstanden estimeres én gang den 1. januar og
videreføres ved origin; warm-up er 0 i månedsvarianten. `--week 2022-01-24`
er særskilt én-fitugesvariant med 24 timers warm-up og samme protokol for BM/WM.

Kort dataaudit uden fit:

```powershell
python -B -m src.exe.run_model.bm_1.deterministic --month 2022-01 --audit-only
```

Reproducerbar syntetisk integration uden workbook/DMI:

```powershell
python -B -m src.exe.run_model.bm_1.deterministic --month 2022-01 --synthetic --starts 2 --max-calls 500
```

`--input-csv path.csv` accepterer raw `Datetime,Demand,Price,Temp` med sorteret,
unik, lokal-naiv Europe/Copenhagen-tid. Den lille adapter afviser DST-overgange;
januar/første februaruge har ingen. Prognoseforberedelsen kræver U/Temp, men
ingen fremtidig demand. Den efterfølgende scoring kræver demand.

## Fit og beregningslofter

BM bruger f(x)=1−2x og g(u)=1−2u. Fittede parametre er
(C_hours,k,Xi,x0), med bounds (1,0.05,0,0) til (744,10,1,1).
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
i log-C/log-k/Xi/x0-koordinater og lokal rang. F/g og støjfit er næste G-trin.

## Output og kontroller

Ny runmappe under `files/output/bm_1/runs/deterministic/`:
audit, checkpoint, manifest, fit og result som JSON; predictions og normaliserede
predictions; daglige metrics; BM/WM-performance, summary og weekday-CSV; week.png.
Fejl gemmes i failure.json. Kør aldrig igen i samme ikke-tomme mappe;
checkpointet er en bevaret kandidat/status, ikke et eksakt SciPy-resume.

MSE/RMSE/MAE/bias, baselinegevinst og performanceanalyser genbruges fra WM.
Forbrugsmetrics vises i rå workbookenheder; normaliserede resultater gemmes også.
Pooled fejl og gennemsnit af dagsmål er forskellige opgørelser. Figurer viser
observation, fælles baseline, begge prognoser, korrektion, pris og modelstates.

Fokuserede regressionschecks:

```powershell
python -B -m unittest tests.test_bm1_deterministic_fit tests.test_bm1_layout -q
```

De kontrollerer modelparitet, adaptive-ODE-reference, konstant intervalreference,
energibalance, rande, punkt/interval, syntetisk parameterfit, svag identifikation,
budgetstop inklusive afledte, historisk cutoff og bevaret BM-resultat ved WM-fejl.
Det syntetiske januarforsøg tog cirka 22 s med to starter og 130 kald i denne
Linux-session; ingen garanti om samme tid/konvergens på historiske Windows-data.
