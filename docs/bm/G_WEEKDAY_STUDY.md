# G: ugedagsfit og det historiske seksugers båndstudie

G og den særskilte WM4-timeadapter tilpasses på fem tilsvarende historiske
ugedage og prognosticerer sjette uge. Syv egne ugedagsmodeller fryses ved hver
mandagsorigin. Det følgende beskriver matematik og det daterede seksugersstudie;
det indeholder ingen genaktiverede forsøgsordrer eller budgetundtagelser.

Aktuel modelbrug står i [BM-guiden](../../root/BM_GUIDE.md). Den generiske
kalender beskrives i [WEEKDAY_BACKTEST_GUIDE](../../root/WEEKDAY_BACKTEST_GUIDE.md),
og samlet evidens i [sammenligningen](COMPARISON.md). De fulde oprindelige
protokoller kan genskabes fra [originalarkivet](../../archive/bm/ORIGINALS.md#original-tree).

<a id="g-weekday-rolling-guide-20261008"></a>
<a id="g-multiperiod-bands-protocol-20261008--kalender-og-information"></a>
<a id="g-multiperiod-bands-protocol-20261008--fit-og-initialisering"></a>
## Kalender og information

Ved mandagsorigin t er TRAIN [t−35 kalenderdage,t) og testen [t,t+7 dage).
Mandag bruger t−7,t−14,…,t−35; tirsdag tilsvarende fem tirsdage osv.
Alle syv baseline-/parameterfit fryses før egen testuges observationsfil åbnes.
Næste origin flytter vinduet syv kalenderdage og kan kausalt træne på tidligere
testuger. Hver af fem targetdage kræver mindst to målte timer; ingen imputation,
padding eller duplikerede dage.

Normalisering bruger kun de fem hele træningsuger:
i=(Y−Ymin)/(Ymax−Ymin), G-pris u=(Price+500)/6500 og WM-pris U=2u−1.
Ex-ante-prisdomænet er [−500,6000], uden clipping. Skalaen er fælles, mens
loss er ugedagsspecifik. Fysisk Europe/Copenhagen-kalender opfinder ikke
forårets 02:00; gentagne naive efterårslabels afvises i denne studieindgang.
Historie-/ugechunks af weather holdes adskilt, så senere uger ikke udfylder
tidligere temperaturegaps.

Studiet brugte `weekday`-baselinepolicy. `shared` er en separat, her ikke
undersøgt variant med fælles femugersbaseline; G/WM deler baseline under begge.
MED er mean over tilgængelige observerede målere, ikke F's faste komplette
population. Råenhed og hour_start(10→10–11) er arbejdsantagelser. Realiseret
DK1-pris/DMI-countrytemperature er et betinget scenarie, ikke en operationel
vejrudsigtsprognose eller kausal pristest.

<a id="g-weekday-rolling-guide-20261008--baseline-og-g-matematik"></a>
## Baseline og G

Baseline er WM-regressionens timeharmoniske, ugedags-/timeinteraktioner,
sæsonharmoniske og temperatur/chill-led. Features gennemsnittes over timen
med trapezregel på 60 samples/hour. Temperatur interpoleres gennem den
sammenhængende femugershistorik, ikke sammenkædede mandage. Ridge/Huber-IRLS,
standardisering og bounded continuation bevares; regression bruger uklippede
features, mens B klippes til [0,1].

Den særskilte adapter erstatter den generiske baselines 168-målekrav med et
udtrykkeligt femdøgnskrav uden at ændre regressionsmatematikken. Snapshot
gemmer samples, featuretal, rang og singularværdier. Ridge håndterer afhængige
weekdaykolonner; fem dage identificerer ikke nødvendigvis sæson/temperatur godt.

G's deterministiske S3-lov, med tid i timer, er:

    f(x)=1−2x,  g(u)=piecewise-linear through (0,1),(.5,a),(1,−1)
    δ=tanh(k*(f(x)+g(u))/2)
    D=B+Xi*δ*((1−B) if δ>=0 else B)
    dx/dh=(D−B)/C

C,k,Xi,a tilpasses med samme bounds, to faste starts og lambda=0 til
measured-only MSE af timegennemsnit på fem uafhængige targetdage.
Historisk state starter .5 med 24 fysiske timers input-only warm-up;
ekstra søndag før første træningsmandag leverer kun Price/Temp, aldrig Demand
til skala, baseline eller loss. Ugedagsbaseline bruges også under warm-up:
det er en initialiseringsapproksimation uden target-state smoothing.

Hver forecastdag starter separat fra medianen af fem egne warm-up-slutstates.
Timeoperatoren er \(\bar D=\bar B+C(x_{end}-x_{start})/1\text{ hour}\).
RK4 bruger baselineknæk og egne coarse/fine-gates ≤1e−5. Studiets endelige
exports brugte 60/120 samples/hour, også når optimizer var konvergeret på et
grovere grid. Dette er en numerisk replay af samme parametre, ikke påstået
optimizerkonvergens på exportgriddet.

<a id="g-weekday-rolling-guide-20261008--wm-matematik-og-sammenligning"></a>
## WM og sammenligningsgrænsen

WM4 bruger tid i døgn og sine affine ligninger:

    F'=R,  R'=rF*F+rU*U+rR*R,  I=B+R
    rF=−a0, rU=−a0*rho, rR=−a0*(1+rho)−extra
    a0>0, extra>=0, rho in[0,1]

Eksponentiel parametrisering, bounds og to seeds bevares. Hver ugedag bruger
samme fem targets, maske og baseline som G. Eksakt 4×4-matrixeksponential
med Z'=R integrerer held U og timegennemsnit af R; energikontrollen er
\(\bar I-\bar B=24\Delta F\) for én time. Variable projection tilpasser
hver historisk (F0,R0), F0∈[−1,1], R0 fri, inklusive randløsninger.
Medianen af fem egne starts bruges i forecast; WM har ingen warm-up.
Senere F-domæneoverskridelser rapporteres uden clipping.

Skala, baseline, inputscenarie, timeoperator, måltemaske og vindue er fælles.
G/WM beholder egne ligninger og initialiseringer. Ingen af de statistiske
outputbånd føres ind i fysisk prisstyring eller ændrer WM's modelkode.

<a id="g-weekday-rolling-guide-20261008--residualer-og-forbedrede-bånd"></a>
<a id="g-multiperiod-bands-protocol-20261008--ét-afgrænset-båndforsøg"></a>
## Residualbank og frosne bånd

Hver model/ugedag replayes med sin deployed medianinitial, ikke states fitted
til targetY. Measured-only fejl E=Y−prediction danner en sorteret in-sample-bank.
Middel m=prediction+mean(E), s²=var(E,ddof=0). De oprindelige 95%-bånd tilføjer
egne inverse empiriske 2.5%/97.5%-kvantiler til prediction; ingen Gaussian-
eller proces-sigma-erstatning.

Tre tidligere sjette-uge-forecasts giver standardiserede scores
z=abs(Y−m)/s. Skala s var kendt ved den enkelte origin og skal være finit
positiv uden skjult floor. G/WM har hver sin pooled bank på fælles measured
maske, alle tre uger og mindst 300 scores/model. For 95% bruges
\(k=\lceil(n+1)19/20\rceil\) og q som den k'te sorterede score.
Hvis k>n findes intet finit deployable bånd; en lavere rang indsættes ikke.
Rationel rangberegning undgår floating-point-fejl ved heltalsgrænser.

Senere uger bruger m±q*s med samme frosne q pr. model og egen kausalt
opdaterede residualskala. Middel, gammel varians og gamle bånd bevares.
q er et intervalmål, ikke diffusion eller en fuld fordeling. Negative rå
nedregrænser tælles uden clipping; residual/procesvarians blandes ikke.

Dette er prequential udviklingskalibrering. Tidsafhængighed, refits og
tidsvariation giver ingen bevist exchangeability/split-conformal 95%-garanti,
samtidig helugegaranti eller iid-konfidensintervaller. Lagdiagnoser og
daglig/ugentlig dækning er beskrivende. Rapporten viser RMSE/MAE/bias,
dækning/bredde og intervalscore W+40·max(L−Y,0)+40·max(Y−H,0).
Pooled RMSE bygger på alle squared errors, ikke gennemsnittet af uge-RMSE.
Bredere bånd er ikke automatisk bedre.

<a id="g-weekday-rolling-guide-20261008--hvad-et-vindue-bruger"></a>
<a id="g-weekday-rolling-guide-20261008--kørsel-og-artefakter"></a>
<a id="g-multiperiod-bands-protocol-20261008--leverance-og-provenance"></a>
## Kode, artefakter og genbrug

Se [G-metoden](../../src/method/g_weekday_rolling.py),
[eksporten](../../src/method/g_weekday_export.py),
[båndmetoden](../../src/method/bm_1_bands.py) og
[kalenderadapteren](../../src/tools/g_weekday_backtest_adapter.py).
Aktuelle indgange er `python -m src.exe.run_model.weekday_backtest --adapter G`
og `python -m src.exe.run_model.bm_1_g bands`; deres `--help` beskriver input.
De erstatter ikke argument-for-argument de pensionerede studie-CLI'er.
Gamle sourcebundne studiekommandoer køres kun i et restaureret originaltræ.

Forecastfiler, snapshot/sourcehashes og receipts bindes før scoring. Alle
bandforecasts og en fælles freeze gemmes, før nogen evalobservationsfil åbnes.
CSV læses med round-trip floatprecision. Ældre pooledfits genmærkes ikke som
ugedagsfit. Ufuldstændige eller numerisk afviste forløb er ikke accepterede fits.
Studiets rå data, stateforløb, fejlbanker, tidsserier og fulde pakker er lokale.

<a id="g-weekday-rolling-guide-20261008--gwm-fem-tilsvarende-ugedage-tilpasning-sjette-uge-test"></a>
<a id="g-multiperiod-bands-protocol-20261008"></a>
<a id="g-multiperiod-bands-protocol-20261008--aktiv-g-protokol-rullende-seksugersvinduer-og-frosne-usikkerhedsbånd"></a>
<a id="g-multiperiod-bands-protocol-20261008--solverbudget-og-numerik"></a>
## Det registrerede studie, 8. oktober 2026

| Origin 2022 | Fem træningsuger fra | Testugen slutter | Rolle |
|---|---|---|---|
| Feb28 | Jan24 | Mar7 | Båndkalibrering |
| Mar7 | Jan31 | Mar14 | Båndkalibrering |
| Mar14 | Feb7 | Mar21 | Båndkalibrering |
| Mar21 | Feb14 | Mar28 | Senere båndtest |
| Mar28 | Feb21 | Apr4 | Senere båndtest |
| Apr4 | Feb28 | Apr11 | Senere båndtest |

Datoerne blev valgt ud fra datadækning før senere targetscore. Den tidligere
Feb14/Feb21-plan fejlede fem-mandagskravet på grund af Jan13–Jan17-hullet.
Tidligere pooledforecasts og missing-datafailure er bevaret som historik.
Januar og andre uger var allerede inspiceret; serien var ikke blind holdout.
Forårsugen havde 167 timer og senere træningsvinduer 839 timer.
Kalibreringsrecord med scorebank, datoer og referencehashes frøs ved Mar21;
de tre senere testuger anvendte samme q pr. model.

Første input-only export blev numerisk afvist før targetY. Den komplette
parameterfit blev derfor replayet på det minuteknæk-alignerede 60/120-net
uden optimizer, med 15s/origin og uden tolerancelempelse; medianstarts og fejlbank blev
genberegnet. Et tidligere konvergeret coarse-fit kunne verificeres sådan,
mens provisional-parametre ikke blev accepteret. Ét diagnosticeret WM-fit
fik en registreret callcap-revision og faktisk optimizerretry før senere
targetscore: 400 oprindelige + højst 1200 ekstra kald, højst 1600 samlet,
inden for de oprindelige 10s. Det var ingen no-retry-kørsel. Originale failure-/
budgetkvitteringer bevarer forbruget; disse gamle undtagelser er ikke aktuelle
arbejdsordrer eller nye budgetter.

Outputprotokollen var `G-WM-five-matching-weekdays-hourly-export60-v1`,
båndprotokollen `G-weekday-rolling-prequential-bands-v2`.
Tidlige protokoller: cb88c5b/34ae624/708b796; numerisk eksport: da73a1f;
bounded continuation: 8d9d770/edc70c9; WM-callrevision: 4d0bca2.
[Ugedagstests](../../tests/test_weekday_backtest.py) og
[båndtests](../../tests/test_g_weekday_bands.py) dokumenterer kalender-,
isolations-, rang- og artefaktkontrakter. Denne tekst tilføjer ingen nye
modelkørsler eller tal fra det lokale seksugersstudie.
