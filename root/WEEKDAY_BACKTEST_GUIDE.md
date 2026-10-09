# Rullende ugedagstest for G og WM

Den fælles forsøgsramme bruger fem kalenderuger til tilpasning og den sjette
til test. Mandagens baseline og parametre får de fem foregående mandage,
tirsdagens de fem tirsdage osv. Derefter flyttes vinduet én uge frem.
Datoerne er frie; den tidligere G-kørsel med seks faste datoer er bevaret.

G og WM-timeadapteren bruger deres eksisterende ugedagsfit.

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

G-indgangen delegerer til den eksisterende G/WM-matematik.
