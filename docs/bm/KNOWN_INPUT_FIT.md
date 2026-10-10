# BM1-fit med kendt demand, pris og baseline samt ukendt eller kendt X0

Denne indgang fitter C, k, Xi og som standard X0 sammen.
Som standard er f(X)=1-2X og g(u)=1-2u faste. Med --fit-response estimeres
begge funktioner samtidig med de frie modelparametre. Baseline og pris er kendte input.
X0 begrænses til [0,1]; en kendt X0 kan alternativt fastholdes.
Ét fælles parametersæt og én sammenhængende tilstandsbane bruges over perioden.
Der er ingen dagsreset, warmup eller stateopdatering med den observerede demand.

## Matematik

delta = tanh(k*(1-X-u)).
For delta >= 0: D = B + Xi*(1-B)*delta.
For delta < 0: D = B + Xi*B*delta.
dX/dt = (D-B)/C, X(0)=X0.

SciPy least_squares(method="trf", loss="linear") minimerer som standard

J = MSE_normaliseret + lambda * [log(C/1)^2 + log(k/2)^2 + log(Xi/0.5)^2].

Kun frie C/k/Xi straffes. X0 straffes aldrig; fastholdt C straffes heller ikke.
Standardvægten er lambda=0.01, et justerbart valg, ikke en kalibreret styrke.
Med `--prior-center C K XI` ændres centrene, og `--prior-weight` ændrer vægten.
Log-straffen giver samme bidrag for faktor 10 og faktor 1/10 fra et center.
MSE beregnes kun over målte timer på den normaliserede demand-skala, så
straffens relative vægt ikke automatisk svækkes ved flere observationer.
RMSE og residualer indeholder aldrig parameterstraffen.
`--prior-weight 0` bruger den tidligere rene sum af kvadrerede datafejl.
C og k optimeres fortsat i logkoordinater.
Standardgrænserne følger den eksisterende G-kode: C i [1,744] timer,
k i [0.05,10], Xi i [0,1] og ukendt X0 i [0,1].
Med positiv straf kræves Xi>0; den numeriske nedre grænse sættes til den
mindste positive normale float, fordi log-straffen divergerer ved Xi=0.
Startgættet for Xi skal da også være positivt. Uden straf tillades Xi=0.
C=1 er stadig den nedre grænse, og k=20 er uden for standardgrænserne,
selvom straffens formel er defineret der. Grænserne kan ændres i Python-kaldet.
Fitteren er lokal; konvergens garanterer hverken et globalt minimum eller
entydige parametre. Eksempelvis kan C/k/X0 ikke bestemmes fra demand, hvis Xi=0.
En estimeret X0 kan mindske koefficienternes kompensation for en forkert
fastholdt starttilstand. Bedre træningsfit er ikke i sig selv dokumentation
for bedre prognoser. Starttilstanden hører til begyndelsen af fitperioden;
ingen separat prognoseinitialisering eller støjestimation udføres her.

Pris holdes konstant inden for hver time. Baseline interpoleres lineært
mellem de angivne støttepunkter. Den eksisterende kontrollerede RK4-simulator
genbruges. Ved timegennemsnit bruges
D_mean = B_mean + C*(X_end-X_start)/1 time.
Til sidst kontrolleres demand én gang med dobbelt så mange integrationstrin.
Kun konvergens med bestået numerisk kontrol får status "converged".

## Python

Kør fra repoets rod med projektets afhængigheder installeret:

```python
from src.exe.run_model.bm_1.bm_1_known_input_fit import fit_known_inputs

result = fit_known_inputs(
    baseline_time, baseline, prices, observed_demand,
    initial=(1.0, 2.0, 0.5),
    prior_center=(1.0, 2.0, 0.5), prior_weight=0.01,
    initial_x0=0.5,
    observation_contract="hourly-mean",
)
print(result["parameters"], result["x0"], result["rmse"], result["status"])
```

Udeladt x0 (eller x0=None) estimeres sammen med koefficienterne.
initial_x0 er kun et startgæt for optimeringen, ikke en fastholdt tilstand.
Med x0=0.23 fastholdes denne kendte værdi, og kun C/k/Xi estimeres.

prices og observed_demand har N værdier for N sammenhængende fysiske timer.
baseline_time og baseline dækker [0,N] og indeholder alle hele timegrænser;
finere støttepunkter er tilladt. Alle modelinput er normaliseret til [0,1].
Demand og baseline skal bruge samme skala. Tid måles i timer, X0 er dimensionsløs.
Manglende input og observationer afvises; der udføres ingen udfyldning.
Ved punktmålinger vælges observation_contract="hourly-point": demand ved
timebegyndelsen med den nye times pris. Det er et andet observationsmål.

Resultatet indeholder parametre, x0, x0_estimated, demand, state (N+1 værdier), residuals
(model minus observation), rmse, initial_rmse, optimizer og numerics.
RMSE fra Python-kaldet er i normaliserede demand-enheder.
max_nfev begrænser SciPys evalueringer; residual_calls tæller også
evalueringerne til numerisk differentiering. Der er ét startgæt pr. kald.
Et andet initial/initial_x0 kan bruges i et særskilt kald ved behov.
fit_settings.parameter_names angiver rækkefølgen for startgæt, grænser og
optimizer.active_bounds. Kun frie parametre optræder i disse lister.
`regularization` gemmer centre, vægt, straffede parametre, normaliseret data-MSE,
strafbidrag og summen MSE+straf. Startgættet er som standard (1,2,0.5).

## CSV og kommando

En færdig syntetisk funktionsprøve følger med repoet:

```powershell
python -m src.exe.run_model.bm_1.bm_1_known_input_fit --input files/input/bm_1/known_input_synthetic.csv --prior-weight 0 --initial 12 2 0.5 --output bm-fit-synthetic
```

Den indeholder 72 timer med kendt baseline 0.43 og demand fra den analytiske
lineære BM1-løsning, med C=7.3, k=2.6 og Xi=0.67. Fitteren skal genfinde
disse parametre samt X0=0.23 med en meget lille RMSE. Dette er syntetiske data,
ikke historiske observationer. Til et rigtigt fit bruges din egen input-CSV.

CSV har time_hours,baseline,price,demand. Hver række angiver en timebegyndelse.
Sidste række angiver periodens slutgrænse og kun baseline:

```csv
time_hours,baseline,price,demand
0,0.40,0.20,0.50
1,0.42,0.70,0.35
2,0.41,0.40,0.45
3,0.40,,
```

Tallene viser formatet, ikke et komplet fælles fit. Mindst fire observationer
kræves ved ukendt X0, tre ved kendt X0; brug en længere periode
med varieret pris for at give parametrene et bedre grundlag.
Baseline i CSV er værdier ved timegrænser, ikke timegennemsnit.
time_hours skal være sammenhængende fysiske timer; starten kan være forskellig fra 0.

```powershell
python -m src.exe.run_model.bm_1.bm_1_known_input_fit --input bm_inputs.csv --output bm-fit
```

X0 estimeres med startgæt 0.5. Brug --initial-x0 0.8 for et andet startgæt.
Kun hvis X0 er kendt, angives eksempelvis --x0 0.23 for at fastholde den.

Standardinput er allerede normaliseret. Rå enheder kan bruges med eksplicitte
skalaer: D_norm=(D_raw-demand_offset)/demand_scale, og tilsvarende for baseline;
u=(price_raw-price_offset)/price_scale. Eksempel:

```powershell
python -m src.exe.run_model.bm_1.bm_1_known_input_fit --input bm_inputs.csv --demand-offset 0 --demand-scale 1000 --price-offset -500 --price-scale 6500 --output bm-fit
```

Skalaerne er kendte input og estimeres ikke. Vælg dem så modelinput ligger i [0,1].
Demand, baseline og price klippes ikke. Modellen bruger forudbestemte skalaer,
og C_hours er en kapacitets-/tidsskala i denne normalisering.

Outputmappen skal være ny. fit.json indeholder parametre, X0 og om den blev
estimeret (x0_estimated), startgæt og grænser, normalisering,
sluttilstand, RMSE, stopårsag og numerisk kontrol. trajectory.csv indeholder
observeret/fitted demand, baseline, pris, X og residualer i de angivne inputenheder.
Ved hourly-mean er outputbaseline et timegennemsnit. Den sidste række viser X
ved periodens slutning. Exitkode 0 betyder "converged"; 1 betyder et ufærdigt
eller numerisk utilstrækkeligt fit, hvis resultater stadig gemmes til inspektion.

Grafer gemmes automatisk som `fit_overview.png` (observeret/fitted demand og
baseline, pris, X og residualer) og `fit_daily_errors.png` (BM/baseline-RMSE
pr. dag, kun målte timer). Kurverne bruger samme observationsmål og rå enheder
som trajectory.csv; et historisk fit er ikke en prognoseevaluering.
Grafer af eksisterende resultater kan laves uden et nyt fit:

```powershell
python -m src.exe.run_model.bm_1.plot_fit --run files/output/bm_1/runs/fit_historik_YYYYMMDD_HHMMSS
python -m src.exe.run_model.bm_1.plot_fit --run files/output/bm_1/runs/fit_historik_YYYYMMDD_HHMMSS --day 2022-01-03
```

Erstat mappenavnet med din gemte kørsel. Dagvalget gemmer en detaljeret
`fit_2022-01-03.png`. Tegneren genskaber timeaksen for ældre automatiske
baseline-kørsler, hvis CSV-tidskolonnen brugte finere offsets; beregningerne
genkøres ikke. Nye trajectory.csv-filer skriver de korrekte hele timer.

## Historisk import fra de samme Excel-filer som WM

Den nye BM-specifikke datafil er `src/exe/run_model/bm_1/data.py`.
Den bruger de fælles `input_data`/`spot_price_data`-importører med WM4's
observerede gruppe-gennemsnit som reference. Som standard estimeres baseline
med WM4's uændrede `statistical_output` fra
`src/exe/run_model/wm_4/output_model.py`. Temperatur importeres med den fælles
`weather_data`-importør fra DMI eller lokal cache, kun til baselineberegningen.
S3's hidtidige hjælpere er særskilt bevaret i
`src/exe/run_model/bm_1/s3_data.py`; den nye datafil bruger dem ikke.
S3's kildehashes følger de ændrede filplaceringer; gamle snapshots kræver
fortsat deres oprindelige kildeversion og omskrives ikke.

```powershell
python -m src.exe.run_model.bm_1.bm_1_known_input_fit --start 2022-01-01 --end 2022-02-05 --tag MED --output bm-fit-historik
```

Perioden er [start,end). Standardfilerne er `files/input/data/input.xlsx`
og `files/input/data/prices.xlsx`; de kan vælges med `--input-workbook` og
`--prices-workbook`. Prisområdet er DK1 som standard (`--price-area`).
Demand er gennemsnittet af gruppens tilgængelige originale målinger, ikke en
sum og ikke gennemsnittet af interpolerede målere. En helt umålt time bevares
i tilstandsbanen og udelades fra fit/RMSE med en observationsmaske. Den fælles
prisimportørs eksisterende udfyldning af pristimer genbruges.

Baseline-regressionen bruger gruppens tilgængelige målte forbrug før `--end`,
også historik før `--start`, hvis den findes. Den kræver mindst 168 gyldige
forbrugs-/temperaturobservationer og bruger WM4's standardindstillinger:
døgn-/ugedags-/sæsonled, temperaturled, Huber-vægtning og ridge.
Pris indgår ikke. Ved dette historiske parameterfit er `--end` skæringsdatoen;
der bruges intet demand ved eller efter den dato. Baseline må gerne estimeres
på selve træningsperiodens demand. Dette er et fit, ikke en prognoseevaluering.

Baseline beregnes på et grid med `--steps-per-hour` støttepunkter pr. fysisk
time, inklusive slutgrænsen. Temperaturen interpoleres på gridet som i WM4.
BM bruger den samme estimerede funktion som B(t), holder den fast under
C/k/Xi/X0-fittet og beregner timegennemsnit med sin eksisterende simulator.
WM4's regressionsmål genbruges uændret; der tilføjes ikke en ny estimator.
Baseline-fitmetadata og skæringsdato gemmes i fit.json.

En allerede kendt baseline kan stadig leveres med `--baseline-csv baseline.csv`.
Da springes baselineestimation og temperaturimport over.
`baseline.csv` skal have `Datetime,baseline` ved hver timegrænse, inklusive
slutgrænsen. Baseline skal bruge samme rå enhed og gruppe-gennemsnit som demand.
Den interpoleres lineært mellem timegrænserne. Naive tidsstempler fortolkes
som Europe/Copenhagen. Tvetydige eller ikke-eksisterende lokale timer afvises;
de sammenlægges ikke stiltiende. Tid til BM måles i fysisk forløbne timer.
Demand mærkes som timebegyndelse som standard; vælg `--demand-label hour_end`,
hvis målingerne beskriver den time, der slutter ved tidsstemplet. Dette er en
erklæret fortolkning, ikke en verificering af Excel-filens målekonvention.

Ved automatisk baseline bruger demand og baseline den samme [0,1]-skala fra
forbrugshistorikken før `--end`. WM4's estimator forudsiger allerede på denne
normaliserede skala og klipper baseline til [0,1], ligesom i WM4. Den færdige
baseline normaliseres derfor ikke igen. Med leveret baseline vælges fortsat
den fælles demand/baseline-skala fra den valgte periode og CSV-baseline.
Prisskalaen kommer i begge tilfælde fra periodens priser. Eksisterende
offset/scale-flags kan i stedet levere forudbestemte skalaer. Skalaerne gemmes
i fit.json og har ingen afhængighed af WM's normalisering for hele 2022.
Med `--input` gælder fortsat CSV-indgangens standardoffset 0 og skala 1.

Python-indgangen er `aligned_bm_data(start,end,...)` efterfulgt af
`fit_inputs(frame,...)` for automatisk WM4-baseline eller
`fit_inputs(frame,baseline_path,...)` for en leveret baseline. Resultatet indeholder baseline_time,
baseline, prices, demand og observation_mask til `fit_known_inputs(...)`,
samt normalisering og metadata. Fitteren gemmer importmetadata i fit.json.

### Fixed capacity

Add `--c 5` to hold capacity at exactly 5 hours while fitting `k`, `Xi` and
`X0`. This is different from `--initial 5 2 0.5`, which only sets a starting
guess and still estimates capacity. `--x0` can also fix the initial state.
The existing data import, baseline, solver and plots are reused. The JSON
records `c_estimated: false`; optimizer parameter names and bounds list only
free parameters. Use a new output directory for each fit.

### Fælles fit af f og g med fem linjestykker

Tilføj `--fit-response --response-pieces 5` til den eksisterende kommando.
Fitteren genbruger `BM1PiecewiseResponse` og `BM1ResponseInputs`. Hver funktion
har faste knuder [0,0.2,0.4,0.6,0.8,1], endepunkter 1 og -1 og fire frie
indre værdier. Funktionerne er kontinuerte og ikke-voksende; flade stykker
er tilladt. Ét fælles par f/g bruges gennem hele perioden.

Denne tilstand slår ALLE strafled fra, også hvis --prior-weight er positiv.
Målet er alene summen af kvadrerede normaliserede demand-fejl på målte timer.
X0, C, k og Xi estimeres som før, medmindre --c eller --x0 fastholder dem.
Med alle parametre frie er der 12 fit-parametre og mindst 12 målte timer kræves.
De eksisterende fysiske parametergrænser gælder fortsat.

Indre kurveværdier parameteriseres ved fire andele i [0,1] af det resterende
fald fra 1 til -1. For andele q_i er v_i=-1+2*produkt(1-q_j, j<=i).
Det sikrer monotonien uden ekstra straf eller sortering. Startandelene
[1/5,1/4,1/3,1/2] giver lineære kurver. Optimizer-metadata gemmer disse andele;
`response.f` og `response.g` gemmer de faktiske knuder/værdier i fit.json.
`response_fitted` er true, og `regularization.weight` er 0.

Stejle f-stykker kræver finere RK4-trin. Fitteren vælger et stabilt multiplum
af --steps-per-hour ud fra grænserne (standard: 60 trin/time for fem stykker).
Både det ønskede og anvendte antal gemmes under numerics. Den eksisterende
kontrol ved dobbelt opløsning gælder stadig; en fejlet kontrol er ikke konvergens.
Grafmodulet gemmer desuden `fit_response.png` med de estimerede f/g-kurver.

Dette er et lokalt datafit uden regularisering, ikke garanti for entydige
funktioner eller et globalt optimum. Kurvedele uden tilstrækkelige besøg i
X/u kan være svagt bestemte. Vurder prognoser på en efterfølgende periode,
hvor baseline, normalisering og modelparametre er fastlagt fra træningen.

### Glatte f/g-funktioner og fælles sammenligning

`--fit-response --response-family pchip` vælger glatte monotone kubiske
kurvestykker gennem de samme støttepunkter. Den eksisterende
`input_builder.md` genbruges. `--response-family bernstein` vælger ét
Bernstein-polynomium; `--response-pieces 5` angiver her graden. De seks
værdier er ordnede Bernstein-koefficienter, ikke punkter kurven passerer.
Begge familier har fire frie formparametre pr. funktion ved standardvalget.
`--response-family piecewise-linear` er fortsat standard med --fit-response.

`--response-family beta-cdf` bruger h(z)=1-2*I_z(a,b), hvor I er beta-CDF.
Hver funktion har to formparametre, a og b, i [1,10]. Disse grænser holder
endepunktshældningerne endelige; former med a<1 eller b<1 indgår ikke.
a=b=1 giver 1-2z. --response-pieces bruges ikke af denne familie.
Alle familier går fra 1 til -1 og er ikke-voksende, og alle kurvefits
slår parameterstraffen helt fra. C/k/Xi/X0 estimeres sammen med kurverne.

Solveren genbruges med den valgte kurves evaluering og maksimale hældning.
Ved standardgrænser og 12 ønskede trin/time bruges 60 for stykvis lineær
og Bernstein, 84 for PCHIP og 108 for beta-CDF. Gridet er fast gennem
optimeringen af hver familie. Den efterfølgende kontrol med dobbelt opløsning
skal bestås. De oprindelige lineære f/g genbruger den oprindelige solver.

Prøv alle fem familier med én import og én baselineestimation:

```powershell
$outputDir = "files/output/bm_1/runs/fit_compare_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python -m src.exe.run_model.bm_1.bm_1_known_input_fit --start 2022-01-01 --end 2022-02-05 --tag MED --compare-responses --output $outputDir
```

Kørslen laver fem fits sekventielt med de samme input, normalisering,
observationsmaske, parametergrænser og startgæt. Alle strafled er slået fra,
også for den lineære reference. Hver kandidat har ét startgæt og sin egen
mappe med fit.json, trajectory.csv og grafer. Resultaterne bevares til inspektion.
En kandidat der ikke konvergerer, eller ikke består numerikkontrollen, kan
ikke vinde, selv hvis den viser lavere RMSE. En kandidat med beregningsfejl
rapporteres i sammenligningen, og de resterende kandidater forsøges stadig.

`comparison.csv` og `comparison.json` opdateres efter hver kandidat.
`comparison.png` sammenligner RMSE og de faktiske f/g-kurver; skravering
markerer resultater, som ikke kan vælges. `comparison.json.best_run` angiver
den relative mappe for det bedste gyldige træningsfit. Exitkode 0 betyder,
at mindst én gyldig kandidat blev fundet; kontrollér også de øvriges status.
Der er ingen garanti for et globalt optimum, og vinderen er valgt på
træningsfejl, ikke på prognosefejl. Et nyt fit skifter ikke automatisk
andre modelindganges standardfunktioner.

Åbn vinderens dagsgraf uden at fitte igen:

```powershell
$comparison = Get-Content (Join-Path $outputDir "comparison.json") -Raw | ConvertFrom-Json
if (-not $comparison.best_run) { throw "Ingen gyldig vinder; se comparison.csv." }
$bestRun = Join-Path $outputDir $comparison.best_run
python -m src.exe.run_model.bm_1.plot_fit --run $bestRun --day 2022-01-03
Invoke-Item (Join-Path $bestRun "fit_2022-01-03.png")
Invoke-Item (Join-Path $outputDir "comparison.png")
```

Gemte kurver rekonstrueres med `response_from_record`; gamle lineære
kurvebeskrivelser læses fortsat. Grafer tegner kurverne fra den gemte familie.
