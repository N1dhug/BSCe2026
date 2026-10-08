# G-S3: fungerende selvstændig grundversion

8. oktober 2026. Den første praktiske milepæl er implementeret og kørt:
**fit → gem → genindlæs → prognose → valgfri scoring**.
Kildecommit til det historiske fit: `b3ad159`. Ingen nye solvere eller
responsfamilier; de eksisterende S3-kerner genbruges. F er ikke ændret.

## Brug modellen

Kør fra repositoryets rod med Python, NumPy, SciPy, pandas og matplotlib.
Eksemplet ligger i den lokalt bevarede pakke `G-S3-example-20261008` i arbejdsområdet.
De følgende outputmapper skal være nye; færdige resultater overskrives ikke.

```powershell
python -B -m src.exe.run_model.bm_1.s3_forecast fit --raw-csv ../G-S3-example-20261008/raw_inputs.csv --origin 2022-02-07 --calibration-start 2022-01-03 --calibration-end 2022-02-07 --output s3_new_fit
python -B -m src.exe.run_model.bm_1.s3_forecast predict --snapshot s3_new_fit/s3_snapshot.json --inputs-csv ../G-S3-example-20261008/future_inputs.csv --observations-csv ../G-S3-example-20261008/observations.csv --output s3_new_prediction
```

For kun at genbruge det allerede fittede eksempel:

```powershell
python -B -m src.exe.run_model.bm_1.s3_forecast predict --snapshot ../G-S3-example-20261008/fit/s3_snapshot.json --inputs-csv ../G-S3-example-20261008/future_inputs.csv --output s3_reloaded_prediction
```

Det sidste kald åbner ingen demandfil og udfører intet fit. Tilføj
`--observations-csv` for scoring bagefter. Den selvstændige `forecast.csv`
og dens kvittering gemmes, før observationsfilen læses. En fejl ved scoring
bevarer derfor den færdige input-only-prognose. CLI returnerer 2 ved
unfinished, numerisk fejl eller ugyldigt input; 0 ved færdig kørsel.

Fit-input er `Datetime,Demand,Price,Temp` i rå enheder. Forecast-input er
`Datetime,Price,Temp`. Datetime er unikke lokale, naive timetidsstempler i
Europe/Copenhagen. V1 håndterer ordinære og forårs-DST-uger; efterårs-DST
med gentagne lokale timer afvises eksplicit. Prognosetabellen gemmer også
entydig UTC-tid og fysisk elapsed_hours. Manglende forecastinputs og priser
uden for frosne grænser repareres ikke ved klipning eller refit.

API: `fit_s3`, `save_snapshot`, `load_snapshot`, `predict_s3`, `score_s3`
i [modeladapteren](../src/method/bm_1_s3_forecast.py).
Kørsels-/eksportlaget er [den lille CLI](../src/exe/run_model/bm_1/s3_forecast.py).
Disse enkle grænser giver en praktisk base for senere BM2/flere parametre.

## Det faktisk kørte historiske eksempel

Origin 7. februar 2022, prognose 7.–14. februar. MED er gennemsnittet af
tilgængelige observerede målere per time. De 35 historiske dagsforløb har
**720 målte targettimer af 840**; de 120 manglende timer indgår ikke i fit.
Warm-up er 24 fysiske timer per bane fra x=.5. Baseline fittes på målte
data før cutoff. Prisgrænserne [-500,6000] er de allerede erklærede eksogene
grænser; demand-normalisering bruger kun 3. januar–7. februar.

| Resultat | Faktisk værdi |
|---|---:|
| Fit | Konvergeret; 144 residualkald; 18,400 s |
| Grid | 12 steps/h, kontrolleret mod 24 |
| Historisk demandforfining | 7,245e-8, tolerance 1e-5 |
| Største forecastforfiningsfejl | 7,042e-8, tolerance 1e-5 |
| Scoring | 168 af 168 målte prognosetimer |
| G-S3 RMSE | 0,377435 i workbookens rå enheder |
| Baseline RMSE | 0,378180 i de samme enheder |
| G-S3 MAE / bias | 0,283876 / +0,066020 |

C=1,517686 h, k=0,143507, Xi=0,286350, g(.5)=−0,999983.
Der er syv egne coarse/fine medianstarts i snapshot; ingen kontinuerlig
168h-statebane eller fremtidig observationsassimilation.

RMSE-forskellen fra baseline er lille. Prisresponsen er tæt på en rand,
og små lokale singularværdier samt rå træningsdiagnostik gemmes i fit.
Denne ene kørsel dokumenterer en fungerende pipeline, ikke at G slår WM,
robust parameteridentifikation eller historisk gevinst af prisstyring.
Data har realiseret fremtidig pris og temperatur: et betinget
udviklingseksempel, ikke en blind prognosetest eller historisk vejrudsigt.
Workbooks' fysiske demand-enhed og timeintervalfortolkning er uverificeret;
den eksisterende punktapproksimation er mærket i både snapshot og output.

Input/modelresultater og tids-/kontrolkvittering er bevaret lokalt i
`G-S3-example-20261008.zip` i arbejdsområdet. Figuren `predict/week.png`
viser G, baseline, observationer og priser. Den komplette rå input-/
resultatpakke sendes ikke til GitHub; kode og denne resultatsammenfatning
er den remote leverance. Automatisk godkendelsesreview afviste den første
fulde datapakke som potentielt følsom; kodeleverancen er derfor afgrænset. Fit-registreringen er
[bevaret separat](../archive/bm/G/s3-transition-history.md#g-s3-standalone-run-20261008).

## Kontroller og holdested

Otte fokuserede kontrolmetoder består: roundtrip uden fit, egen start per
dag, forecast- og fit-uafhængighed af fremtidig demand, scoring uden mutation,
ugyldigt snapshot/scenario, spring-DST/autumn-afvisning, forecast bevaret
før scoringfejl og budgetstop uden snapshot. Seks kørt i 1,098 s og de to
yderligere fitkontroller i 1,471 s. Alle tre nye Python-filer AST-parser.
Det historiske fit og CLI-forecast er reelt udført, ikke testfixtures.

Efterfølgende er [det afgrænsede proceslag](../docs/bm/G_UNCERTAINTY.md#g-s3-uncertainty-guide-20261008)
implementeret og numerisk kontrolleret. Historisk kalibrering er afvist;
G6a's tidligere failure står ved magt. Den deterministiske prisstyring er stadig den
eksisterende selvstændige vej; den er ikke koblet til historisk closed loop.
Grundimplementeringen er afleveret. Timeintervaloperator/enhed og gyldig
usikkerhedskalibrering er åbne faglige udvidelser; se
[den aktive arbejdsramme](G_COMPLETION_PLAN_8H_20261008.md).
