# Aktiv G-protokol: rullende seksugersvinduer og frosne usikkerhedsbånd

Johan bestilte flere testperioder og usikkerhedsbånd og præciserede derefter:
fem foregående mandage tilpasser mandagsmodellen, fem tirsdage tirsdagsmodellen
osv.; sjette uge testes, og vinduet flyttes en uge. G/F holdes adskilt.
Bank/BM2, nye AR/OU-lag, fysisk metadataverifikation og prisstyring er særskilte
opgaver. Den oprindelige8h-ramme og tidligere forbrug nulstilles ikke.
Pas startet8.oktober2026 kl.20:32:13UTC. Tidlig reviewtid er fortsat umålt.

## Kalender og information

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

## Fit og initialisering

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

## Solverbudget og numerik

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

## Ét afgrænset båndforsøg

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

## Leverance og provenance

Modeloutputprotokol: G-WM-five-matching-weekdays-hourly-export60-v1.
Båndprotokol: G-weekday-rolling-prequential-bands-v2. Se
[G_WEEKDAY_ROLLING_GUIDE_20261008.md](G_WEEKDAY_ROLLING_GUIDE_20261008.md) for
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
