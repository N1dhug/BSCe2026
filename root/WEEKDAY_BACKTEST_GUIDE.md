# Rullende ugedagstest for F og G

Den fælles forsøgsramme bruger fem kalenderuger til tilpasning og den sjette
til test. Mandagens baseline og parametre får de fem foregående mandage,
tirsdagens de fem tirsdage osv. Derefter flyttes vinduet én uge frem.
Datoerne er frie; den tidligere G-kørsel med seks faste datoer er bevaret.

G og WM er tilsluttet gennem deres eksisterende ugedagsfit. F kan bruge samme
kalender, datadeling og forsøgsstyring gennem egne funktioner. F4's hurtigkørsel
bruger et fast tidligere fit og udfører ikke denne genkalibrering. Der er
derfor ingen automatisk `--adapter F`, der fejlagtigt genbruger det gamle fit.

## Planlæg perioder uden at starte modellerne

Kør fra repoets rod med Python 3.11 og de eksisterende NumPy, SciPy og Pandas.
Alle outputmapper skal være nye. Dette eksempel skriver kun kalenderen:

```powershell
python -B -m src.exe.run_model.weekday_backtest `
  --first-origin 2022-02-28 --last-origin 2022-04-04 `
  --output ../weekday-calendar-plan --plan-only
```

Begge datoer er mandage kl. 00 lokal tid og medtages. Seks origins giver seks
testuger og kræver elleve uger af data, samt modelafhængig opvarmning. Første
vindue er 24. januar til 7. marts, næste 31. januar til 14. marts osv.

## Kør G og WM på nye perioder

`history.csv` indeholder `Datetime,Demand,Price,Temp` i rå enheder. Historien
skal dække fem uger før hver origin og G's ekstra 24 fysiske inputtimer.
`inputs.csv` indeholder `Datetime,Price,Temp` for testugerne. En eventuel
Demand-kolonne i inputfilen bliver ikke givet til modellen. De fremtidige
priser og temperaturer er et erklæret scenarie; realiseret vejr giver en
betinget historisk test, ikke en operationel vejrprognose.

```powershell
python -B -m src.exe.run_model.weekday_backtest `
  --adapter G --first-origin 2022-02-28 --last-origin 2022-04-04 `
  --history-csv ../history.csv --inputs-csv ../inputs.csv `
  --observations-csv ../observations.csv --output ../weekday-G-new
```

`observations.csv` har `Datetime,Demand` og åbnes først, når alle bestilte
prognoser er gemt og bundet til deres modeller med kontrolsummer. Udelad
argumentet for kun at tilpasse og lave prognoser. Senere rullende fits må
bruge en tidligere testuge, når den ligger før den nye origin. Der foretages
ingen samlet tilpasning på tværs af de bestilte origins.

G bruger fortsat separat baseline pr. ugedag, egne G- og WM-parametre,
fælles femugersnormalisering, eksisterende numeriske kontroller og adskilte
residualer. Loftet pr. ugedag er G 30 sekunder og 750 kald, WM 10 sekunder og
400 kald; numerisk export har 15 sekunder pr. uge. En ufærdig tilpasning
stopper forsøget og bevarer checkpoints. Der er ingen automatisk genkørsel,
udvidelse af budgettet eller overgang til fælles parametre. Den gamle
eksplicitte fortsættelsesvej kan vurderes separat.

## Tilslut F med egne tilpasningsfunktioner

`src.tools.weekday_backtest.WeekdayCallbacks` giver fire funktioner til en
model. De kaldes separat for hver ugedag:

| Funktion | Data og ansvar |
| --- | --- |
| `fit_baseline(day)` | Fem adskilte dagsrammer i `day.records`, ældste først. Tilpas egen baseline. |
| `fit_parameters(day, baseline)` | Egen parameter- eller likelihoodtilpasning. Overhold egne budgetter og stop ved ufærdige fits. |
| `save_fit(day, baseline, parameters, directory)` | Gem egne kildereferencer, parametre, tilstande, konvergens og kontrolresultater før prognosen. |
| `predict(day, baseline, parameters)` | Returnér en DataFrame på præcis `day.inputs.index` i erklærede rå måleenheder. Ingen targetmålinger er tilgængelige. |

Under fit indeholder `day.context` den fysiske historie med alle erklærede
inputtimer, men med Demand maskeret uden for den valgte ugedag og under
opvarmningen. F kan bruge en sammenhængende likelihood med manglende
observationer gennem mellemperioderne. Fem dage må ikke sættes sammen til
en kunstig femdages kontinuerlig tidsserie. Under prediction indeholder
context historiske inputs og testugens inputs fra mandag til slutningen af
den valgte dag; alle Demand-værdier og træningsrecords er fjernet. F's egen
frosne posterior kan dermed føres gennem tirsdag før en onsdagsprognose.
Rammen bestemmer ikke prior, nulstiller ikke en tilstand og deler ingen
likelihood, diffusion eller prisrespons mellem F og G.

Et adaptermodul, for eksempel `src.tools.my_f_weekday_adapter`, kan have:

```python
from src.tools.weekday_backtest import WeekdayCallbacks
from my_f_implementation import (
    fit_baseline, fit_parameters, save_fit, predict, source_hashes,
)

def make_adapter():
    return WeekdayCallbacks(
        name='F',
        metadata={
            'fit_budget': {'seconds': 30, 'evaluations': 100},
            'initialization_policy': 'own explicitly frozen continuous posterior',
            'population': 'own explicitly selected meter population',
            'target_unit': 'own verified raw unit',
            'source_sha256': source_hashes(),
        },
        prediction_columns={'F': 'f_output_mean'},
        fit_baseline=fit_baseline, fit_parameters=fit_parameters,
        save_fit=save_fit, predict=predict,
        warmup_hours=24,
    )
```

Dette er tilslutningsformen, ikke en ny F-estimator. Funktionerne, budgettet,
populationen og initialiseringen skal komme fra det konkrete F-forsøg.
De illustrerede budgetter godkender ikke en udvidelse af et gammelt F/H-budget.
Kør derefter samme indgang med
`--adapter src.tools.my_f_weekday_adapter:make_adapter`.

En model, der skal tilpasse alle syv baselines før sin tilstandberegning, kan
også implementere `forecast_week(data, directory)` direkte som G-adapteren.
Den skal selv udvælge de fem matchingdage pr. baseline/parameterfit og
overholde samme regler om cutoff og egne numeriske kontroller. Kalenderen
og `split_weekday(data, weekday)` kan genbruges. Der er ingen implicit
flytning af gamle F3/H-punktfits til et nyt intervalforsøg.

## Tid, manglende data og usikkerhed

Data skal allerede være tilpasset til timeintervallernes begyndelse. En
end-label skal omplaceres efter den valgte datakontrakt før indgangen bruges.
Ugegrænser følger lokal kalender, mens timeskridt følger fysisk tid. En uge
omkring dansk sommertid har 167 timer; omkring vintertid 169. Den fælles
Python-ramme understøtter tidszonebevidste data med begge efterårstimer.
Naive gentagne efterårslabels afvises; den eksisterende G-adapter accepterer
kun entydige naive København-labels. CSV'er med blandede UTC-offsets skal
omformes til entydig UTC eller indlæses til et tidszonebevidst DatetimeIndex
før Python-API'et kaldes.

Manglende Demand bevares og udelades kun fra scoringens fælles maske. Hver af
de fem træningsdage skal have målte targets; adapteren stiller sine egne
strengere samplekrav. Manglende eller uendelige inputs stopper forsøget.
Skala, population og datametadata er modellens ansvar og fremgår af dens
metadata. G's fysiske Demand-enhed og labelbetydning er fortsat
arbejdsantagelser; en fælles ramme verificerer dem ikke.

G's eksisterende residualbånd gemmes sammen med prognoserne. Den nye ramme
tilpasser ikke nye bånd på testmålinger. Den tidligere båndkalibrering er
fortsat et separat, fast datodefineret forsøg. Nye kalibrerings- og testuger
skal erklæres kronologisk, før yderligere usikkerhedskalibrering foretages.

## Gemte filer og kontrol

Hver uge får eget datovindue med kontrolsummer, modellens fitfiler,
`predictions.csv` og `forecast_freeze.json`. `all_forecasts_frozen.json`
bekræfter, at alle forecasts eksisterer før scoring. Kontrolsummerne
kontrolleres også efter åbning af scoringtargets. Scoring giver RMSE, MAE
og bias pr. model og uge. Bias betyder prognose minus observation.
Samlet RMSE beregnes ud fra alle kvadrerede fejl og antal målte timer,
ikke som et gennemsnit af ugers RMSE. Forsøget vælger ingen vinder eller
optimale parametre ud fra disse testresultater.

En failure giver `failure.json` og bevarer allerede gemte ugeartefakter.
Eksisterende outputmapper overskrives ikke. Data, resultater, snapshots og
kontrolsummer af private målerdata forbliver lokale efter den gældende
publiceringsaftale. Generisk kode, vejledning og syntetiske tests kan merges.

Fokuserede kontroller:

```powershell
python -B -m unittest tests.test_weekday_backtest tests.test_g_weekday_bands
```

Tests af F-tilslutningen bruger en lille syntetisk model og ændrer hverken
F4's fysiske model eller dens videnskabelige accept. Den nye G-indgang
delegerer til den allerede testede G/WM-matematik.
