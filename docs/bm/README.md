# Kør BM-F og BM-G

Kør fra repositoryets rod med Python 3.11/3.12 og pakkerne i
[requirements-f4.txt](../../requirements-f4.txt). Input-workbooks ligger lokalt
i `files/input/data/`. Der installeres eller downloades intet automatisk.
Gemte fits tilhører deres erklærede målergruppe, liste, skala, baseline og cutoff;
MED og UDEN må ikke udskiftes i en eksisterende model.

| Opgave | Indgang |
|---|---|
| F: filtrér og lav åben prognose med gemt fit | `python -m src.exe.run_model.bm_1_f4` |
| G: brug gemt weekday-model | `python -m src.exe.run_model.bm_1_g` |
| Nyt ugedagsfit | [Kalender, CSV-kontrakt og fitbudget](WEEKDAY_BACKTEST.md) |
| F: genlæs eksisterende resultat | `python -m src.exe.run_model.bm_1_f4 --read-result RESULTATMAPPE` |
| G: genlæs eksisterende resultat | `python -m src.exe.run_model.bm_1_g show --result RESULTATMAPPE` |

## F: start og ændr periode

```powershell
python -m src.exe.run_model.bm_1_f4 --help
python -m src.exe.run_model.bm_1_f4 --check
python -m src.exe.run_model.bm_1_f4 --output files/output/bm_1/runs/my_f_example
```

Standardeksemplet bruger én filtertime og tre forecasttimer med et gemt
tostykkefit; det estimerer ingen parametre. `--check` validerer input uden
modelpropagation. Terminalen viser fase, tid og færdig/pauset/fejlet status.
`run_f4.cmd` videresender de samme argumenter fra projektets rod.

Datoer og normale valg kan angives direkte:

```powershell
python -m src.exe.run_model.bm_1_f4 --filter-start 2022-02-21T00:00:00+01:00 --forecast-origin 2022-02-21T02:00:00+01:00 --forecast-hours 6 --model fitted --grid base --output files/output/bm_1/runs/my_f_six_hours --check
```

Fjern `--check` for beregning. Brug ny outputmappe ved ændret periode/model.
`--model fitted linear` tilføjer den lineære reference. Cutoff og datagrænser
kontrolleres før modelarbejdet; denne indgang er forskellig fra et nyt weekday-fit.
`--group MED` skal matche den gemte model; `--group UDEN` afvises for MED-fittet.

Kopiér en konfiguration fra [configs/bm](../../configs/bm/f4_quick.json) og brug
`--config din_config.json` til avancerede valg. CLI-argumenter har forrang.
F4's relative input/model/outputstier regnes fra repoets rod; `--config`
regnes fra terminalens aktuelle mappe. Weekday-adapterens modelsti er derimod
relativ til dens configfil og kontrolleres med SHA256.

| Konfiguration/felt | Betydning |
|---|---|
| [f4_quick.json](../../configs/bm/f4_quick.json) | Én filtertime, tre forecasttimer. |
| [f4_open_week.json](../../configs/bm/f4_open_week.json) | Én filtertime, 168 åbne forecasttimer uden scoring. |
| [f4_week.json](../../configs/bm/f4_week.json) | Oprindelig 24-timers filtercase; har kendte negative MED-målinger. |
| `filter_start`, `forecast_origin` | Frisk deklareret Beta-prior ved start; kun målinger før origin assimileres. |
| `forecast_hours`, `quantile_hours` | 1–168 timer; kvantiler kun ved de valgte timer inden for horisonten. |
| `models`, `grid` | `fitted`/`linear` og egne `base`/`fine`-grids. |
| `workers`, `max_seconds` | 1–8 arbejdere; tidsloft for modeldelen, ikke hele opstart/eksport. |
| `evaluate_forecast` | Læs slutmålinger til scoring efter alle forecasts er gemt. |
| `output_dir` | Ny resultatmappe ved ændrede model-/dataforudsætninger. |

Datoer skal være hele timer. Naive datoer fortolkes i Europe/Copenhagen;
tvetydige eller ikke-eksisterende sommertidstimer afvises. Brug eksplicitte
UTC-offsets ved behov. Det medfølgende fits informationscutoff er 21. februar
2022 kl. 00; baseline/skala er frosset frem til 16. februar. Tidligere origin
eller filterstart afvises efter hver sin grænse.

Prisplanen `previous_week` gentager de foregående syv døgns tilsvarende timer
uden fremtidige realiserede priser. Alternativt vælges `price_policy: supplied`
og `future_prices_csv: din_prisplan.csv`, med præcis én række pr. prognosetime:

```csv
timestamp,spot_price_DKK_MWh
2022-02-21T01:00:00+01:00,1000
2022-02-21T02:00:00+01:00,1100
2022-02-21T03:00:00+01:00,900
```

Priser uden for frossen skala afvises. Leverede priser er et erklæret scenarie;
senere realiserede priser er ikke automatisk information tilgængelig ved origin.

## F: output, pause og genbrug

`predictions.csv` skelner fysiske timeintervaller, prognose/baseline/observation,
state X og varianser/kvantiler. `comparison.png` viser efterprøvelige kurver,
`summary.json` status/fejlmål/fasetider, og `manifest.json` binder config,
model, data og kildeidentiteter. `checkpoint.npz` gemmer seneste hele time;
`origin.npz` bevarer den faktisk filtrerede origin og fulde log-state.
`--read-result RESULTATMAPPE` læser en eksisterende pakke uden dataindlæsning,
fit eller propagation; nye pakkers outputhashes kontrolleres.

Filterrækken viser prognosen før den aktuelle observation og endpoint-posterior
efter den. Forecast er åben propagation uden fremtidige targets. En manglende
måler gør hele gruppetimen manglende; state fortsætter uden Bayes-opdatering.
Hvert negativt råmål afvises før aggregering med timestamp og målernavn;
positive gruppemidler kan ikke skjule negative medlemmer. En signeret
netload-model er ikke implementeret. Dette er inputpolitikken, selv om
Gaussian-likelihooden matematisk tillader negative observationer.
Der klippes ikke, og populationen ændres ikke. Se [matematik og data](MATHEMATICS.md).

F's 5/50/95-kvantiler er marginaler med nominel Gaussian-observationsstøj;
5–95% er et ukalibreret 90%-interval, ikke samtidig ugedækning. Tomme
kvantilfelter ved øvrige timer betyder ikke nul usikkerhed. Parameter-/input-
usikkerhed er udeladt. Enhedsnavne og hour-start bygger fortsat på erklærede
antagelser. RMSE/MSE/MAE/bias bruger fælles måltemaske; bias er prognose minus
observation, og procentfejl bruges ikke som genvej ved negative/nære nulværdier.

Ved `paused` (exitkode 2) gentages præcis samme kommando med `--resume`.
Ctrl+C eller en `STOP`-fil stopper efter gemt arbejde; fjern `STOP` før resume.
En halv time kan kræve genberegning. Workerantal og tidsloft må ændres, mens
model, numeriske valg, periode, input og kildeidentitet skal matche. Numerisk
fejl giver `FAILED.json`; ingen automatisk genstart. Færdig kompatibel resume
returnerer `complete` uden at omskrive resultatet; manglende eksport afvises.

Gamle checkpoints binder deres oprindelige kilder. Brug det [bevarede originaltræ](../../archive/bm/ORIGINALS.md)
frem for at ommærke manifests. En 600-sekunders pause er ikke et bestået
uge-runtime-mål. Udvendig procesmåling omfatter også imports, input og eksport;
`max_seconds` og interne fasetider har mindre scope.

## G: gemt model og ny prognose

```powershell
python -m src.exe.run_model.bm_1_g --help
python -m src.exe.run_model.bm_1_g bands --help
```

Konvertér først et gyldigt `weekday_export_snapshot.json` til en selvstændig
G-model. Det kontrollerer de oprindelige kildebindingshashes og gemmer
`model.json`, originalsnapshot og `conversion_freeze.json` i en **ny mappe**.
Konverteringen laver hverken fit eller forecast:

```powershell
python -m src.exe.run_model.bm_1_g export-model --snapshot files/output/bm_1/examples/g_saved_week_20220228/snapshot.json --output files/output/bm_1/models/my_g_model
```

Brug den frosne model ved samme eller en senere mandagsorigin. CSV'en har
`Datetime,Price,Temp` for **hele ugen**, også når `--hours` kun vælger et kort
udsnit fra origin. Labels skal være entydige lokale København-timebegyndelser;
en uge med gentaget efterårstime kræver en anden godkendt datakontrakt.
Priser skal ligge i modellens deklarerede domæne (her −500 til 6000 i rå
prisenheder). Det er en fast skala, ikke et estimeret min/max; der klippes ikke.

```powershell
python -m src.exe.run_model.bm_1_g forecast --model files/output/bm_1/models/my_g_model/model.json --origin 2022-02-28 --inputs-csv files/output/bm_1/examples/g_saved_week_20220228/inputs.csv --group MED --hours 6 --check
python -m src.exe.run_model.bm_1_g forecast --model files/output/bm_1/models/my_g_model/model.json --origin 2022-02-28 --inputs-csv files/output/bm_1/examples/g_saved_week_20220228/inputs.csv --group MED --hours 6 --output files/output/bm_1/runs/my_g_new_week
python -m src.exe.run_model.bm_1_g show --result files/output/bm_1/runs/my_g_new_week
```

Første eksempel bruger de eksisterende lokale inputs for 28. februar. Ved
et senere periodevalg, f.eks. `--origin 2022-03-07`, skal `--inputs-csv`
pege på din egen matchende fulde uge; filen skal opfylde samme kontrakt.

Der findes også en lokal prøve af en senere uge med genbrugt model og
verificerede inputs fra det afsluttede studie:

```powershell
python -m src.exe.run_model.bm_1_g forecast --model files/output/bm_1/examples/g_deployment_20261009/model_current/model.json --origin 2022-03-07 --inputs-csv files/output/bm_1/examples/g_deployment_20261009/inputs_20220307.csv --group MED --check
```

Denne CSV er faktisk realiseret Price/Temp fra 7.–14. marts, så prøven er en
betinget historisk integrationstest, ikke et bevis på information ved origin.
Artefakterne er lokale og følger ikke med Git; kontrollen genestimerer intet.

Dette er en ny G-prognose med frosne parametre, syv daglige medianstarts,
baselines, skala og residualer; den estimerer ikke nye parametre eller WM.
`--origin` ændrer prognosetidspunktet, ikke fitcutoff. Output skelner cutoff,
modelalder, origin, fysiske UTC-intervaller og horisont. `forecast.csv` og
`forecast_normalized.csv` har rå henholdsvis normaliserede værdier;
`forecast.json`, freeze og summary dokumenterer proveniens og beregning.
`show` verificerer gemte kvitteringer og læser resultatet uden modelberegning.

De gamle snapshotmodellers gruppe er erklæret MED, men den oprindelige
målerliste er ikke verificeret (`legacy-unverified`). `--group` er en
matchkontrol; UDEN kræver sin egen model og kan ikke fremstilles ved ommærkning.
Eksemplet kræver lokale verificerede artefakter; private scenarieinputs følger
ikke med koden. Fremtidige Price/Temp er et erklæret scenarie, hvis tilgængelighed
ved origin ikke bevises af CSV'en. Tilføj eventuelt `--score observations.csv`
med `Datetime,Demand`; targets åbnes først efter forecastfreeze.

`replay --snapshot ... --inputs-csv ...` bevarer den tidligere G/WM-timeadapters
saved-uge. Her vælger `--start`/`--hours` kun et udsnit i den oprindelige uge;
hele ugen beregnes fortsat til numerikkontrol. Den gamle form uden `replay`
virker også. En replay er hverken nyt fit eller ny ugeprognose.

G's empiriske residualbånd er ukalibrerede og giver ingen dækningsgaranti.
De er forskellige fra F's observationsstøj og WM's proceskovarians.

`bm_1_g bands calibrate/evaluate` anvender erklærede kronologiske datoer,
matchende informationskontrakt og mindst 300 målte kalibreringsscores. Alle
evalforecasts fryses før evaltargets åbnes. Se [båndmetoden](G_WEEKDAY_STUDY.md).

## Kort brugerprøve og dokumentation

1. Kør F `--help`/`--check`, derefter det korte eksempel i ny mappe. Læs CSV,
   figur og status. Ændr periode eller modelvalg med de almindelige argumenter.
2. Prøv origin før 21. februar 2022; det gemte fits cutoff skal afvises.
   Et populationsskift må ikke genbruge den samme fitidentitet.
3. Brug G's eksplicitte model-/inputvalg. Skeln mellem saved-ugens udsnit,
   ny tilladt prognose og et nyt fit; læs outputkontrakten før sammenligning.
4. Genlæs det færdige resultat uden nyt fit. Ved reel F-pause fortsættes med
   samme identitet og `--resume`.

Johans faktiske brugerprøve **afventer**; de automatiske kontroller erstatter
ikke hans vurdering af læsbarhed og brugervenlighed. Målte runtimes og præcise
kilde-/inputversioner står kun i [sammenligningen](COMPARISON.md). F's korte
eksempel og åbne uge bruger én positiv filtertime; de løser ikke det
oprindelige 24-timers filters negative målinger. [Status](STATUS.txt) angiver
næste arbejde og de resterende faglige spørgsmål.

[Matematik](MATHEMATICS.md) er autoritativ for model- og datakontrakter,
[kalendervejledningen](WEEKDAY_BACKTEST.md) for nye ugedagsfits og
[arkivet](../../archive/bm/INDEX.md) for historiske beslutninger og resultater.
WM4-timeadapter og original WM4-CLI er forskellige protokoller. Fælles
rangering kræver samme gruppe, enhed, observation, inputs og informationscutoff.
