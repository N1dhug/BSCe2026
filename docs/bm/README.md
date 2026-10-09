# Kør BM-F og BM-G

Kør fra repositoryets rod med Python 3.11/3.12 og
[requirements-f4.txt](../../requirements-f4.txt). Workbooks og gemte modeller
er lokale; de downloades ikke automatisk. MED/UDEN betegner elvarmegrupper.
Dataantagelser, negative observationer og usikkerhed står i [matematikken](MATHEMATICS.md).

| Opgave | Kommando/vejledning |
|---|---|
| F: filtrering og prognose med gemt fit | `python -m src.exe.run_model.bm_1_f4` |
| G: eksport, ny prognose eller gammel replay | `python -m src.exe.run_model.bm_1_g --help` |
| Nyt F/G-fit | [Weekday-fit, CSV-kontrakt og budget](WEEKDAY_BACKTEST.md) |

## F: gemt fit

```powershell
python -m src.exe.run_model.bm_1_f4 --check
python -m src.exe.run_model.bm_1_f4 --output files/output/bm_1/runs/my_f_example
python -m src.exe.run_model.bm_1_f4 --read-result files/output/bm_1/runs/my_f_example
```

Standardeksemplet genbruger tostykkefittet: én filtertime og tre forecasttimer.
`--check` validerer input uden propagation. `--read-result` kontrollerer gemte
outputhashes uden workbooklæsning eller modelarbejde. `run_f4.cmd` videresender
samme argumenter. Terminalen viser fase og færdig/pauset/fejlet status.

Ændr normale valg direkte, og brug en ny outputmappe til en ny opsætning:

```powershell
python -m src.exe.run_model.bm_1_f4 --group MED --filter-start 2022-02-21T00:00:00+01:00 --forecast-origin 2022-02-21T02:00:00+01:00 --forecast-hours 6 --model fitted --grid base --output files/output/bm_1/runs/my_f_six_hours --check
python -m src.exe.run_model.bm_1_f4 --config configs/bm/f4_open_week.json --output files/output/bm_1/runs/my_f_open_week --check
```

Fjern `--check` for beregning. `--model fitted linear` tilføjer den lineære
reference. Gruppen skal matche det gemte fit; MED-fittet afviser UDEN.
Hele timer kræves. Naive datoer fortolkes i Europe/Copenhagen; tvetydige eller
manglende sommertidstimer afvises. Brug eksplicit offset ved behov.
Kurvens informationscutoff er 21. februar 2022 kl. 00, baseline/skala 16. februar.
Origin og filterstart kontrolleres mod hver sin informationsgrænse.

Kopiér [f4_quick.json](../../configs/bm/f4_quick.json) til avancerede ændringer;
CLI har forrang. [f4_open_week.json](../../configs/bm/f4_open_week.json) vælger
168 åbne timer og syv kvantilhorisonter uden fremtidige observationer/scoring.
Det oprindelige negative filterdøgn er en fejlprøve, ikke en ekstra config:

```powershell
python -m src.exe.run_model.bm_1_f4 --forecast-origin 2022-02-22T00:00:00+01:00 --check
```

| Configfelt | Betydning |
|---|---|
| `filter_start`, `forecast_origin` | Frisk Beta-prior ved start; kun historiske målinger assimileres. |
| `forecast_hours`, `quantile_hours` | 1–168 timer; kvantiler ved valgte timer inden for horisonten. |
| `models`, `grid` | `fitted`/`linear`, `base`/`fine`. |
| `workers`, `max_seconds` | 1–8 arbejdere; tidsloft for modeldelen, ikke hele processen. |
| `evaluate_forecast` | Scoringtargets åbnes først efter alle forecasts er gemt. |
| `price_policy` | `previous_week` gentager historiske priser; `supplied` læser erklæret scenarie. |

F4's relative data/model/outputstier regnes fra repoets rod; configstien fra
terminalens mappe. Leverede priser bruger `future_prices_csv` med præcis én række
pr. forecasttime og kolonnerne `timestamp,spot_price_DKK_MWh`. Priser uden for
frossen skala afvises. Realiserede priser er ikke automatisk kendt ved origin.

`predictions.csv` indeholder intervaller, prognose/baseline/observation, state X
og usikkerhed; `comparison.png` viser kurverne. `manifest.json` binder modellen,
data og kilder; `summary.json` indeholder status, fejlmål og tider.
`origin.npz` bevarer filtreret origin; `checkpoint.npz` seneste hele time.
Kvantiler ved øvrige horisonter er tomme, ikke nul usikkerhed.

Ved `paused` (exit 2) genbruges præcis samme kommando med `--resume`.
Ctrl+C eller en `STOP`-fil stopper efter gemt arbejde; fjern STOP før resume.
En delvist beregnet time kan gentages. Workers/tidsloft må ændres; model,
periode, input, numeriske valg og kildeidentitet skal matche. Færdig resume
genberegner intet. Numerisk fejl giver `FAILED.json`, uden automatisk retry.
Gamle checkpoints kræver deres [oprindelige kilder](../../archive/bm/ORIGINALS.md).

## G: modelpakke, ny prognose og genlæsning

Eksport af et nyt weekday-fit finder automatisk det tilhørende
`fit_population_binding.json`. Bindingen skal matche fit, træningsdata,
skalaer og baselines. En flyttet originalbinding kan angives med
`--population-binding STI`. Modelpakken bevarer deklarationen og dens status
`declared-unverified`; CSV og hashes beviser ikke måleraggregation.

```powershell
python -m src.exe.run_model.bm_1_g export-model --snapshot files/output/bm_1/runs/my_g_fit/20220411/weekday_export_snapshot.json --output files/output/bm_1/models/my_g_model
```

Eksport kontrollerer de oprindelige kildebindingshashes og udfører intet fit
eller forecast. Outputmappen skal være ny. Et gammelt snapshot uden
populationsbinding kræver et eksplicit legacyvalg:

```powershell
python -m src.exe.run_model.bm_1_g export-model --snapshot files/output/bm_1/examples/g_saved_week_20220228/snapshot.json --legacy-unverified --output files/output/bm_1/models/my_legacy_g_model
python -m src.exe.run_model.bm_1_g forecast --model files/output/bm_1/models/my_legacy_g_model/model.json --origin 2022-02-28 --inputs-csv files/output/bm_1/examples/g_saved_week_20220228/inputs.csv --group MED --hours 6 --check
python -m src.exe.run_model.bm_1_g forecast --model files/output/bm_1/models/my_legacy_g_model/model.json --origin 2022-02-28 --inputs-csv files/output/bm_1/examples/g_saved_week_20220228/inputs.csv --group MED --hours 6 --output files/output/bm_1/runs/my_g_forecast
python -m src.exe.run_model.bm_1_g show --result files/output/bm_1/runs/my_g_forecast
```

Legacyvalget kan ikke omgå en manglende binding på et markeret nyt fit.
Eksisterende modeller ommærkes aldrig til en anden gruppe eller ny kildeversion.
Før-markør-fit med deklaration kræver deres originale freeze/kvitteringer.

`--origin` skal være samme eller en senere mandag end fitcutoff. Ændring til
f.eks. 7. marts kræver en matchende CSV for hele 7.–14. marts. Inputkolonner er
`Datetime,Price,Temp` for **hele ugen**, også når `--hours` vælger et kort udsnit.
Labels er entydige lokale København-timebegyndelser; gentaget efterårstime
understøttes ikke af G-kontrakten. Prisskalaen er her fast −500 til 6000 i rå
enheder. Fremtidige inputs er et erklæret scenarie med separat tilgængelighedsstatus.

G bruger syv frosne dagsfits og medianstarts. En senere origin opdaterer
hverken fitcutoff, parametre, baselines eller starttilstande. `forecast.csv`
og `forecast_normalized.csv` har rå/normaliserede værdier; JSON/freeze/summary
binder population, tidsintervaller, modelalder, input og integritet.
`show` verificerer kvitteringer uden beregning. Valgfri `--score observations.csv`
kræver `Datetime,Demand`; targets åbnes først efter forecastfreeze.

`replay --snapshot ... --inputs-csv ... --start ... --hours ...` bevarer den
gamle G/WM-timeadapters faste uge. Det er forskelligt fra en ny forecast.
`bands calibrate/evaluate` er en særskilt kronologisk båndprocedure;
se [båndmetoden](G_WEEKDAY_STUDY.md). Ingen fælles modelrangering startes her.

## Kort brugerprøve og dokumentation

1. Kør F `--check`, det korte eksempel og `--read-result`. Læs CSV/figur/status.
2. Ændr periode/horizont/modelvalg. Prøv et cutoff før 21. februar, UDEN og
   det negative filterdøgn; de skal afvises uden fit eller propagation.
3. Eksportér G-modellen, kør `--check` og seks timers forecast, og brug `show`.
   En senere uge kræver matchende scenarieinput; et nyt fit er en anden handling.
4. Ved reel F-pause: brug samme identitet og `--resume`.

Din brugerprøve afventer. [Status](STATUS.txt) viser afsluttede kontroller og
resterende arbejde; [sammenligningen](COMPARISON.md) ejer runtime og før/efter-tal.
[Matematik](MATHEMATICS.md) ejer data-/modelkontrakten, [weekday-guiden](WEEKDAY_BACKTEST.md)
nye fits og [arkivet](../../archive/bm/INDEX.md) de historiske forsøg.
