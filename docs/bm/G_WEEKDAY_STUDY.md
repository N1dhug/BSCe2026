# G: ugedagsfit og det historiske seksugers båndstudie

G og den særskilte WM4-timeadapter har syv ugedagsfit pr. origin. Det konkrete seksugersstudie og dets frosne bånd er udviklingsevidens. Den generiske kalender findes i [WEEKDAY_BACKTEST_GUIDE](../../root/WEEKDAY_BACKTEST_GUIDE.md); nye frie bånddatoer håndteres af `python -m src.exe.run_model.bm_1_g bands --help`. Ingen af disse muligheder autoriserer nye historiske fits.

Aktuel brugerindgang: [BM-guiden](../../root/BM_GUIDE.md). Afsnit mærket protokol bevarer det allerede udførte forsøgs numerik, stopregler og provenance; deres gamle kørselsordrer er historiske. Nye fits eller brede benchmarkkørsler kræver særskilt budget.

Statusord i de bevarede originalafsnit beskriver den daværende delundersøgelse. Den aktuelle samlede accept og senere udførte arbejde står i [sammenligningen](COMPARISON.md) og [status](STATUS.txt); ældre formuleringer som endnu ikke udført må ikke læses som global slutstatus.

Originale kilder og SHA256 findes i [arkivets stikort](../../archive/bm/old-paths.json). De oprindelige snapshots/protokoller er uændrede; dokumentkonsolideringen ændrer ingen matematisk model.

<a id="g-weekday-rolling-guide-20261008"></a>
<a id="g-weekday-rolling-guide-20261008--gwm-fem-tilsvarende-ugedage-tilpasning-sjette-uge-test"></a>
<a id="g-weekday-rolling-guide-20261008--hvad-et-vindue-bruger"></a>
<a id="g-weekday-rolling-guide-20261008--baseline-og-g-matematik"></a>
<a id="g-weekday-rolling-guide-20261008--wm-matematik-og-sammenligning"></a>
<a id="g-weekday-rolling-guide-20261008--residualer-og-forbedrede-bånd"></a>
<a id="g-weekday-rolling-guide-20261008--kørsel-og-artefakter"></a>

## Model, anvendelse og bevaret evidens: G_WEEKDAY_ROLLING_GUIDE_20261008.md

### G/WM: fem tilsvarende ugedage tilpasning, sjette uge test

Johans bestilling ændrer det gamle samlede35-dagesfit til **syv separate
ugedagsfit**. Dette er en egen indgang og egen snapshotprotokol; tidligere
G/WM-resultater og F-kode er bevaret. Numeriske resultater, rådata og pakker
fra den nye serie er lokale. Denne vejledning indeholder kun metode/brug.
Læs også `G_MULTIPERIOD_BANDS_PROTOCOL_20261008.md` for forsøgsregistreringen.

#### Hvad et vindue bruger

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

#### Baseline og G-matematik

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

#### WM-matematik og sammenligning

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

#### Residualer og forbedrede bånd

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

#### Kørsel og artefakter

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

Én eksplicit solver-undtagelse er registreret før senere testscore: hvis et
WMfit ramte400calls uden convergence men havde tid tilbage, kan den valgte
continuation bruge `--wm-extra-calls 1200`.
Samme model/objektiv/seeds/tolerances får én1200-callkørsel inden **resten af
den oprindelige10s**. Snapshot beholder oldfailure og kumulative calls≤1600,
tid≤10s; default0 udfører ingen retry. Ingen yderligere forlængelse eller
provisionalaccept. Undtagelsen er solverkonvergens, ikke valg efter test-Y.

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

<a id="g-multiperiod-bands-protocol-20261008"></a>
<a id="g-multiperiod-bands-protocol-20261008--aktiv-g-protokol-rullende-seksugersvinduer-og-frosne-usikkerhedsbånd"></a>
<a id="g-multiperiod-bands-protocol-20261008--kalender-og-information"></a>
<a id="g-multiperiod-bands-protocol-20261008--fit-og-initialisering"></a>
<a id="g-multiperiod-bands-protocol-20261008--solverbudget-og-numerik"></a>
<a id="g-multiperiod-bands-protocol-20261008--ét-afgrænset-båndforsøg"></a>
<a id="g-multiperiod-bands-protocol-20261008--leverance-og-provenance"></a>

## Historisk forsøgsprotokol: G_MULTIPERIOD_BANDS_PROTOCOL_20261008.md

### Aktiv G-protokol: rullende seksugersvinduer og frosne usikkerhedsbånd

Johan bestilte flere testperioder og usikkerhedsbånd og præciserede derefter:
fem foregående mandage tilpasser mandagsmodellen, fem tirsdage tirsdagsmodellen
osv.; sjette uge testes, og vinduet flyttes en uge. G/F holdes adskilt.
Bank/BM2, nye AR/OU-lag, fysisk metadataverifikation og prisstyring er særskilte
opgaver. Den oprindelige8h-ramme og tidligere forbrug nulstilles ikke.
Pas startet8.oktober2026 kl.20:32:13UTC. Tidlig reviewtid er fortsat umålt.

#### Kalender og information

| Origin2022 | Fem træningsuger fra | Testugen slutter | Rolle |
|---|---|---|---|
| Feb28 | Jan24 | Mar7 | Båndkalibrering |
| Mar7 | Jan31 | Mar14 | Båndkalibrering |
| Mar14 | Feb7 | Mar21 | Båndkalibrering |
| Mar21 | Feb14 | Mar28 | Senere båndtest |
| Mar28 | Feb21 | Apr4 | Senere båndtest |
| Apr4 | Feb28 | Apr11 | Senere båndtest |

Datoerne blev valgt ud fra datadækning før senere targetscore. Tidligere
Feb14/Feb21-plan kunne ikke give fem målte mandage pga. Jan13–Jan17-hullet.
De oprindelige pooled-forecasts, missing-datafailure og forbrug bevares som
historik; ingen genbruges/omdøbes til nye ugedagsfit. Januar/andre uger var
allerede inspiceret; denne serie er kronologisk udvikling, ikke blindholdout.

MEDmean over tilgængelige observerede målere, measured-only Demand, uverificeret
råeffektenhed og hour_start(10→10–11) som arbejdsantagelse. Ex-ante DK1-prisdomæne
[-500,6000], realiseret historisk pris/DMIcountrytemperature. Betinget inputscenario,
ikke en operationel vejrudsigtsprognose. Weather genbruges/hentes i separate
historie-/ugechunks, så en senere uge ikke udfylder tidligere temperaturegaps.
Workbooks/cache/datahashes bevares lokalt. Fysisk DST-kalender opfinder ikke02:00;
forårsugen har167timer, senere træningsvinduer839timer. Efterårsdobbeltlabels afvises.

#### Fit og initialisering

Pr. origin er der syv G-fit og syv WM-fit, egne fem matchingweekdaytargets.
Baseline tilpasses også separat til samme fem dage, med den eksisterende
WM-featurefamilie, timeintegrerede features, ridge/Huber-IRLS og bounded
continuation. Kræv alle fem dage og mindst to målte timer pr. dag, ingen
padding eller Demandimputation. Samples/features/rang/singularværdier gemmes;
fem dage giver ikke nødvendigvis sikker parameteridentifikation.

G/WM deler hver ugedags baseline samt fælles normalisering fra de fem hele
træningsuger. G beholder S3-ligning, bounds/toseeds/lambda0 og24h-inputwarmup;
neutralstate.5, median af fem egne warmupslutstates. Ekstra søndag før ældste
mandag leverer kun Price/Temp, aldrig Demand til baseline/skala/loss. WM beholder
sine affine ligninger, exacthourlyoperator, variable projection af targetinitials,
median af fem egne starts, ingen WM-warmup. Forecast er syv uafhængige dagsforløb,
alle frosset mandag. Den næste origin må kausalt træne på tidligere testuger;
egen testuge påvirker aldrig eget fit eller den frosne band-q.

#### Solverbudget og numerik

Standard pr. ugedag: G30s/750rawcalls, WM10s/400rawcalls. Seks origins har
42fit/model og højst1680s standard optimizer-walltid, eksklusive baseline/replay/
I/O/tests. Ingen completedfit gentages eller budgets nulstilles. Failure,
provisional, checkpoints, tidligere attempts og tidsforbrug bevares.

G-fitgrids12/24/48/60 deler egen cap. Alle endelige exports bruger det allerede
minutebaseline-alignerede60/120-net, samme<=1e−5 gate. Denne source-bound
numericalexport har15s/origin, nuloptimizerkald, originalparent uændret og
samme fittedparametre/baselines/scales; medianstarts/residualreplay genberegnes
numerisk. Første input-onlyexport blev afvist før targetY; dens komplette fit
blev derfor numerisk eksporteret separat uden nyt fit. Ingen tolerancelempelse.

Hvis en finer-grid-refit nåede callcap efter et tidligere optimizerconvergedfit,
kan det eksisterende convergedfit kontrolleres60/120 inden resten af samme
G30s-budget, uden optimizer. Aldrig bestprovisional. Faktisk coarseobjective-
grid deklareres; der hævdes ikke optimizerconvergence på exportgriddet.
Continuation genbruger doneugedage; kun endnu ukørte dage får første fitbudget.
Den ikkefærdige dags baseline genskabes deterministisk fra præcis samme input.
Driverhash/originalfitreport/kumulativekald/tid bindes i receipts.

Én diagnosticeret WMstop nåede400calls langt inden10s. Der er registreret
én eksplicit exception før senere targetscore: højst1200ekstracalls, højst1600
samlet og kun resten af de oprindelige10s. Samme model/loss/bounds/seeds/
tolerances, oldfailure/calls/tid bevaret. Det er en deklareret callcap-revision
og faktisk optimizerretry, aldrig nulstillet forbrug eller no-retryclaim.
Ingen yderligere extension/provisionalaccept; andre fits beholder standardcaps.

#### Ét afgrænset båndforsøg

Hver ugedag/model har sin egen empiricalbank E=Y−prediction fra fem historiske
replays med deployed medianinitial. In-sampleudviklingsfejl, ikke identificeret
målerstøj. m=prediction+mean(E); s=SD(E,ddof0) i råenheden, allerede kendt ved
origin. s skal være finitepositiv; ingen floor, Gaussian, processσ-combination
eller fysisk clipping. Gamle inverse-CDF95%-bånd bevares ved siden af nye.

På de tre tidlige sjette-uge-forecasts er z=abs(Y−m)/s. G/WM får separate
pooledstandardiserede banker, samme measuredmask, alle tre uger, mindst300scores.
For95% er k=ceil((n+1)*19/20), q=k'te sorteret score. Hvis k>n er der intet
finit deployablebånd; ingen lavere rang indsættes. Record med scorebank,
forecast-/observationhashes/kilder fryses vedMar21 før senere test-Y.

Alle tre senere uger bruger m±q*s, samme frosne q/model og egen kausalt opdaterede
historiske ugedagsskala. m, gammel variance og gamle bånd ændres ikke. q er et
intervalmål, ikke diffusion eller en fuld ny fordeling. Ingen retuning på de
senere testuger. Alle tre input-onlybandforecasts gemmes med hashes før nogen
senere observationsfil åbnes. Intet lag føres ind i fysisk prisstyring.

Fortolkningen er prequential finite-rank **udviklingskalibrering**, ikke bevist
split-conformalexchangeability/95%-garanti under tidsafhængighed og refits.
Rapportér lagdiagnose, dækning/bredde/intervalscore og daglig/ugentlig variation.
Ingen universal/simultaneousweek95%-accept eller iidconfidenceintervals.
PooledRMSE beregnes fra alle squared errors, ikke gennemsnit af ugeRMSE.
Bredere bånd regnes ikke automatisk som bedre.

#### Leverance og provenance

Modeloutputprotokol: G-WM-five-matching-weekdays-hourly-export60-v1.
Båndprotokol: G-weekday-rolling-prequential-bands-v2. Se
[G_WEEKDAY_ROLLING_GUIDE_20261008.md](G_WEEKDAY_STUDY.md#g-weekday-rolling-guide-20261008) for
matematik/brug. Tidlige protokolversioner findes i Git-historik cb88c5b,
34ae624 og708b796; numerisk eksport registreretda73a1f, boundedcontinuation
8d9d770/edc70c9 og eksplicitWMcall-undtagelse4d0bca2 før senere targetscore.
Ingen ny periodemodelvalg/scorebaseret ugeudskiftning.

Kode/protokol/generisk vejledning og syntetiske tests må pushes/merges under
Johans tilladelse. **Nye perioders numeriske resultater, rå inputs/states,
fejlbanker og komplette pakker forbliver lokale.** Tidligere aggregate-upload-
tilladelse dækkede det gamle Feb7-eksempel; den udvides ikke automatisk.
Synkroniserede sources ændres ikke. F-kode/studier og eksisterende G7-/F-budgetter
bevares særskilt.
