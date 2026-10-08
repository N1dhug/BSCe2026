# G/WM: fem tilsvarende ugedage tilpasning, sjette uge test

Johans bestilling ændrer det gamle samlede35-dagesfit til **syv separate
ugedagsfit**. Dette er en egen indgang og egen snapshotprotokol; tidligere
G/WM-resultater og F-kode er bevaret. Numeriske resultater, rådata og pakker
fra den nye serie er lokale. Denne vejledning indeholder kun metode/brug.
Læs også `G_MULTIPERIOD_BANDS_PROTOCOL_20261008.md` for forsøgsregistreringen.

## Hvad et vindue bruger

Ved mandagsorigin t er træningsvinduet [t−35kalenderdage,t), testen [t,t+7dage).
Mandagens baseline og G-/WM-parametre bruger mandagene t−7,t−14,…,t−35dage;
tirsdag bruger tilsvarende fem tirsdage osv. Alle syv parameter-/baselinefit
fryses mandag før egen testuges observationsfil åbnes. Derefter flyttes
trænings-/testvinduet syv kalenderdage. Fem historiske targetdøgn kræves;
mindst to målte timer pr. døgn, ingen targetimputation eller duplikerede dage.

Fælles normalisering bruger hele de fem træningsuger: i=(Y−Ymin)/(Ymax−Ymin),
G-pris u=(Price+500)/6500 og WM-pris U=2u−1. Demandmin/max har ingen ældre
eller fremtidige målinger; ex-ante-prisdomænet er[-500,6000], uden clipping.
Således er fitted losses ugedagsspecifikke, mens den deklarerede skala er
fælles for modellerne og ugedagene. Forårets manglende02:00 opfindes ikke:
træning/test bruger den fysiske Europe/Copenhagen-kalender. Gentagne lokale
efterårstimer kan ikke repræsenteres entydigt i denne naive-label indgang.

Standard `--baseline-policy weekday` tilpasser også baseline separat til
ugedagen. `shared` er en eksplicit anden protokolvariant, hvor femugers-
baseline er fælles; den er ikke valgt/undersøgt i denne kampagne. G/WM deler
samme baseline uanset valgt policy. Ingen varianten vælges ud fra testfejl.

## Baseline og G-matematik

Baselinefamilien er den eksisterende WM-regression: timeharmoniske,
ugedags-/timeinteraktioner, sæsonharmoniske og temperatur/chill-led. Hver
designrække er timegennemsnittet af features på et60-sample/hour-net med
trapezreglen. Temperaturinterpolation bruger den sammenhængende historiske
femugersserie, ikke en kunstig sammenkædning af fem mandage. Ridge/Huber-IRLS,
standardisering og bounded continuation er bevaret. Regressionen bruger
uklippede lineære features; den anvendte baseline B klippes som hidtil til[0,1].

Den gamle generiske baseline kræver168målinger, som fem mandage ikke har.
Den separate adapter bruger derfor den samme regressionsmatematik med et
udtrykkeligt femdøgnskrav. Snapshot gemmer sampletal, featuretal, designrang
og singularværdier. Ridge håndterer de konstante/lineært afhængige weekday-
kolonner; fem dage identificerer ikke nødvendigvis sæson-/temperatureffekter
godt. Parametre skal fortolkes som udviklingsfit, ikke sikker fysisk identifikation.

G beholder sin deterministiske S3-drift:

    f(x)=1−2x,  g(u)=piecewise-linear through (0,1),(.5,a),(1,−1)
    δ=tanh(k*(f(x)+g(u))/2)
    D=B+Xi*δ*((1−B) if δ>=0 else B)
    dx/dh=(D−B)/C

Hvert ugedagsfit optimerer C,k,Xi,a, med samme bounds og to faste starts,
lambda0. Loss er measured-only MSE af **timegennemsnit** på fem targetdage;
disse er fem uafhængige tilstandsforløb. Historisk x starter neutral.5 og
har24fysiske timers input-only warmup. Ekstra søndag før første træningsmandag
leverer kun Price/Temp; Demand sættes eksplicit utilgængelig. Loss udelukker
alle warmuptimer. Den relevante ugedagsbaseline anvendes også under warmup;
det er en deklareret initialiseringsapproksimation uden target-state smoothing.

Ved deploy er starttilstanden medianen af fem egne warmupsluttilstande.
Hver af sjette uges dage starter separat fra sin frosne median. Timeoperatoren
er Dbar=Bbar+C*(x_end−x_start)/1hour, som følger af energibalancen. RK4-grid
12/24/48/60samples/hour og dobbelt finegrid skal afvige højst1e−5;60-griddet
aligner alle baseline-minuteknæk. Forfining/refit deler samme cap, ingen
grid-/toleranceforhøjelse efter resultater. G30s/750rawcalls pr. ugedag.

Alle perioders endelige numericalexport bruger60/120grid, også når selve
fitttet bestod på et lavere grid. Denne separate15s-replay bruger samme
parametre/baselines og egen historik, genberegner medianstarts/empirical-
fejlbank, og gemmer originalparent uændret. Egen exportkilde/hash/protokol
og nuloptimizerkald er deklareret; gamle fit/failedexports overskrives ikke.
Valget følger første input-onlyforecasts numeriske afvisning, før targetY.
Det er ikke et nyt fit eller en tolerancelempelse. Et komplet tidligere
fit kan eksporteres uden optimizer via `src.exe.run_model.bm_1.g_weekday_export`.

## WM-matematik og sammenligning

WM4's egne affine ligninger beholdes, tid i døgn:

    F'=R,  R'=rF*F+rU*U+rR*R,  I=B+R
    rF=−a0, rU=−a0*rho, rR=−a0*(1+rho)−extra
    a0>0, extra>=0, rho in[0,1]

Eksponentiel parametrisering og tidligere bounds/toseeds er bevaret. WM
har nu også et separat parameterfit pr. ugedag, på de samme fem targets og
samme målte maske/baseline. Exact4x4matrixexponential med Z'=R integrerer
hver times konstantU og gennemsnitR. Operatoren kontrollerer Ibar−B=24ΔF.
Variable projection tilpasser hver historisk(F0,R0): F0∈[−1,1], R0 fri,
eksakt mindst-kvadraters minimisering inkl. randløsninger. Medianen af fem
egne starttilstande anvendes i forecast, ingen WM-warmup. WM10s/400calls pr.
ugedag. F's efterfølgende domæneoverskridelser rapporteres; ingen clipping.

Sammenligning deler skala/baseline/inputscenario/timeoperator/maske/vindue.
G/WM har egne ligninger, parametre og initialiseringer; den forskel er åben.
MED er mean over tilgængelige observerede målere. Demand's fysiske enhed og
hour_start(10→10–11) er endnu uverificerede arbejdsantagelser. Fremtidig
realiseret DK1pris/DMIcountrytemperature er kendt scenarieinput i dette
forsøg, ikke en operationel weatherforecast eller et kausalt priseksperiment.

## Residualer og forbedrede bånd

For hver ugedag/model replayes de fem targets med **deployed medianinitial**,
ikke states tilpasset via targetY. Fejlbanken E=Y−prediction er measured-only,
sorteret og in-sample. Empiricaloutput mean er m=prediction+mean(E); egen
historisk variance er var(E,ddof0). De gamle95%-bånd er egne inverse empirical-
CDF2.5%/97.5%kvantiler tilføjet prediction. Ingen Gaussian-/procesσ-erstatning.

Tre tidligere **sjette-uge-forecasts** bruges derefter til båndkalibrering:
Feb28,Mar7,Mar14. Deres egne fit så aldrig egen ugeY. Standardiseret score
z=abs(Y−m)/s, hvor s er denne forecastugedags allerede kendte residualSD.
G/WM får egne pooledstandardiserede scorebanker på fælles measuredmaske,
alle tre uger og mindst300scores/model. Positiv s kræves, ingen skjultfloor.
For95% er k=ceil((n+1)*19/20), q=k'te sorteret score. Hvis k>n, er båndet
ikke finit deployable; en mindre rang indsættes ikke. Decimal/rationel
rangberegning undgår floating-point-rangfejl på heltalsgrænser.

Kalibreringsrecord fryses vedMar21 medkilder/datoer/scorebank/referencehashes.
Testugerne Mar21,Mar28,Apr4 får **samme q pr. model** og egne kausalt opdaterede
weekdayfit/skala. Nye bånd er m±q*s; m, gammel variance og gamle bånd ændres
ikke. q er et intervalmål, ikke en fuld fordeling eller nyt fysisk diffusionstal.
Ingen ændring af q efter slut-Y, clipping, residual/procesdoublecount eller
statistisk lag inde i prisstyringen. Negative rånedregrænser tælles.

Dette er kronologisk/prequential **udviklingskalibrering**. Afhængighed,
refits og tidsvariation betyder, at finite-rank-formlen ikke giver en bevist
exchangeability/split-conformal95%-garanti her. Lagkorrelation og daglig/
ugentlig dækning rapporteres beskrivende. Ingen samtidig95%-helugegaranti,
iidconfidenceinterval eller blindholdoutaccept. Første tre testuger er
bandkalibreringsdata; de tre sidste vurderer båndreglen.

Rapporten viser raw-unitRMSE/MAE/bias, coverage,width og95%intervalscore
W+40*max(L−Y,0)+40*max(Y−H,0), samme measuredmask. PooledRMSE beregnes af alle
squared errors, aldrig gennemsnit af ugeRMSE. Bedre coverage alene kan være
et dyrt bredere bånd; width/intervalscore skal derfor læses sammen med den.

## Kørsel og artefakter

Fra repoets rod, med Datetime,Demand,Price,Temp i historisk raw-unitCSV,
og separat fremtidig Datetime,Price,Temp. Brug nyt outputdirectory hver gang:

```powershell
python -B -m src.exe.run_model.bm_1.g_weekday_period --raw-csv RAW.csv --inputs-csv FUTURE.csv --origin 2022-02-28 --output OUT/20220228
```

Gentag sekventielt for alle seks registrerede origins. Ingen automatisk
retry, capforhøjelse eller pooledfallback. Fit/checkpoints/failure og elapsed
forbrug bevares. Forecastfiler og SHA256freeze skrives før scoring; genbrug
kræver uændrede kilder/artefakter, ikke et nyt fit.

Hvis en finer-grid-refit brugte sin rawcallcap efter et tidligere converged
fit, kan `src.exe.run_model.bm_1.g_weekday_continue` fortsætte med
`--previous-output OLD --raw-csv RAW.csv --inputs-csv FUTURE.csv --output NEW`.
Doneugedage genbruges; den afbrudte dags **savedconverged** parametre kontrolleres
numerisk60/120 uden optimizer inden resten af samme30s-budget. Kun endnu
ukørte dage har nye, første optimizerinvocations. Faktisk coarseobjective-
grid og manglende convergenceclaim på exportgriddet markeres eksplicit.
No-convergedfit eller failedprecision/budget giver afvisning; ingen provisional
accept, større cap eller fallbackmodel. Originalfile/receipts bevares.

```powershell
python -B -m src.exe.run_model.bm_1.g_band_evaluation calibrate --forecasts OUT/20220228 OUT/20220307 OUT/20220314 --observations Y_20220228.csv Y_20220307.csv Y_20220314.csv --output CAL
python -B -m src.exe.run_model.bm_1.g_band_evaluation evaluate --calibration CAL/calibration_snapshot.json --forecasts OUT/20220321 OUT/20220328 OUT/20220404 --observations Y_20220321.csv Y_20220328.csv Y_20220404.csv --output EVAL
```

Evalindgangen verificerer alle forecastfiler mod receipts, gemmer **alle tre**
input-only calibratedbandforecasts og fælles freeze, før den åbner nogen
evalobservationsfil. Derefter skrives scoredCSV/JSON, daglige/ugentlige checks
og korrekt pooledrapport. ArtifactCSV læses med round-trip floatprecision.
Snapshot/sourcehashes forhindrer tavs relabelling af ældre pooledfits.

De seks nye resultater og pakket råmateriale forbliver lokale. Kode, denne
generiske guide/protokol og syntetiske tests kan merges under Johans tilladelse.
Den tidligere konkrete aggregate-uploadtilladelse udvides ikke automatisk til
denne nye serie. G/F, bank/BM2, metadataverifikation og praktisk styringsgevinst
er stadig særskilte opgaver. Den oprindelige8h-ramme og tidligere tidsforbrug
bevares; der påstås ikke et præcist restbudget, når tidlig reviewtid er umålt.
