# G: timeoperator, baseline og integrerede procesmomenter

Intervalmiddel og punktværdi er forskellige observationsoperatorer. Denne vej bevarer egen intervalfitidentitet. Procesmomenter giver middel og varians gennem timen; de giver ikke intervalkvantiler. Demand-enhed og hour-start er fortsat arbejdsantagelser.

Aktuel brugerindgang: [BM-guiden](../../root/BM_GUIDE.md). Afsnit mærket protokol bevarer det allerede udførte forsøgs numerik, stopregler og provenance; deres gamle kørselsordrer er historiske. Nye fits eller brede benchmarkkørsler kræver særskilt budget.

Statusord i de bevarede originalafsnit beskriver den daværende delundersøgelse. Den aktuelle samlede accept og senere udførte arbejde står i [sammenligningen](COMPARISON.md) og [status](STATUS.txt); ældre formuleringer som endnu ikke udført må ikke læses som global slutstatus.

Originale kilder og SHA256 findes i [arkivets stikort](../../archive/bm/old-paths.json). De oprindelige snapshots/protokoller er uændrede; dokumentkonsolideringen ændrer ingen matematisk model.

<a id="g-s3-interval-guide-20261008"></a>
<a id="g-s3-interval-guide-20261008--g-s3-timegennemsnit-med-integreret-proceskovarians"></a>
<a id="g-s3-interval-guide-20261008--observation-og-baseline"></a>
<a id="g-s3-interval-guide-20261008--procesmiddel-og-varians-gennem-timen"></a>
<a id="g-s3-interval-guide-20261008--den-særskilte-kalibreringsdiagnose"></a>
<a id="g-s3-interval-guide-20261008--brug-og-kontrakt"></a>
<a id="g-s3-interval-guide-20261008--faktiske-udviklingsresultater"></a>
<a id="g-s3-interval-guide-20261008--kontroller-og-næste-holdested"></a>

## Model, anvendelse og bevaret evidens: G_S3_INTERVAL_GUIDE_20261008.md

### G-S3: timegennemsnit med integreret proceskovarians

G har nu sin egen timeversion: fit, komplet snapshot, input-only forecast,
scoring og procesmiddel/varians for forbrug gennem en hel fysisk time.
Baselinefittets statistiske features integreres over samme time. Den
oprindelige punktversion og dens snapshots er bevaret; de genmærkes ikke.
S3 lambda=0 og den eksisterende deterministiske BM1/SDE-familie fortsætter.
F's kode, studier og budget ændres ikke.

Johan oplyste, at Demand er forbrug over en time og efter hans erindring
Power. Han antager kl.10→kl.10–11, uden fuld sikkerhed. Dette er derfor
**hour_start som arbejdsantagelse**, ikke verificeret fysisk metadata.
Effektens enhed er stadig uverificeret. Rå tal behandles som gennemsnitlig
effekt over fysiske en-timers intervaller; for en energi/time-konvention
skal den præcise enhed afklares før fysisk kapacitet eller styringsgevinst.

Se [præregistreret kørsel og korrektion](G_INTERVAL_MODEL.md#g-s3-interval-protocol-20261008),
[fit/snapshot/predict](../../src/method/bm_1_s3_interval.py),
[baselinefeatures](../../src/method/bm_1_s3_interval_baseline.py),
[integrerede procesmomenter og diagnose](../../src/method/bm_1_s3_interval_process.py)
og [indgang/eksport](../../src/exe/run_model/bm_1/s3_interval.py).

#### Observation og baseline

Den eksisterende BM1-lov er uændret:
\(dX=(D(X,U,B)-B)dt/C+\sigma_xX(1-X)dW\), timer, X i [0,1].
Observationen er nu \(Y_h=\Delta^{-1}\int_h^{h+\Delta}D_tdt\), Delta=1
fysisk time. U holdes gennem timen; B interpoleres på et fast 60-sample/h
grid. Ingen nye parametre, bank eller observationsstøj tilføjes.

Deterministisk følger \(\bar D_h=\bar B_h+C(X_{h+1}-X_h)/\Delta\).
G's eksisterende RK4-kernel har allerede denne operator; den bruges i
både fit og forecast. Den er ikke endpoint-D eller D i timens start.
G's 35 histories, warm-up, egne medianstarts og mandagsfrysning bevares.

WM's robuste featurefamilie/ridge/Huber/normalisering genbruges. Til den
kontinuerte baselinekurve tilpasses nu designrækker
\(\bar\phi_h=\Delta^{-1}\int_h^{h+\Delta}\phi(t,T_t)dt\), på det samme
minutegrid. Standardisering kommuterer med denne lineære gennemsnitsoperator.
Temperatur interpoleres i fysisk tid af kendte historiske inputs; ved cutoff
holdes sidste kendte temperatur. Ingen fremtidig temperatur bruges i fit.
Forecast bruger kun det erklærede ugescenarie, inklusive næste times
temperatur hvis den findes inden for scenariet; sidste værdi holdes ved slut.

Den eksisterende clipping af baselinefortsættelsen til [0,1] bevares som
deklareret modelvalg. Regression integrerer de **lineære, unclipped** features;
det er ikke en ny ikke-lineær bounded-baseline-estimator. Clipping kan derfor
introducere forskel fra det lineære observationsdesign. Baseline er en
statistisk inputmodel, ikke rekonstrueret fysisk sandhed.

Minute-tidsstempler bygges med heltalsminutter/nanosekunder. Float-hour
konvertering kunne ellers lande lidt før en minutgrænse, mens WM-features
bruger heltalsminutter/sekunder. En uafhængig harmonisk integralreference
består den analytiske trapezoidgrænse omega²/(12·60²), ikke en efterfølgende
tilpasset testtolerance. DST bruger fysisk tid og lokale kalenderfeatures.

#### Procesmiddel og varians gennem timen

I SDE gælder \(\int Ddt=\int Bdt+C\Delta X-C\int b(X)dW\). Den
deterministiske endpointidentitet må derfor ikke genbruges til tilfældige
timeforbrug. Vi følger det additive fleksibilitetsintegral
\(Z=\int(D-B)dt\) sammen med statefordelingen. For FV-generator Q og
reward c(x,t)=D−B er de tre fremadrettede felter:

\[
m'=Qm,\qquad r'=Qr+\operatorname{diag}(c)m,\qquad
s'=Qs+2\operatorname{diag}(c)r.
\]

Her er m state-masser, r=E[Z·1_state] og s=E[Z²·1_state]. Rewardfelterne
nulstilles ved hver ny time; state-masser bæres videre gennem hver egen dag.
Middel er \(\bar B+\sum r\), og varians er \(\sum s-(\sum r)^2\).
Afhængigheden mellem state og integreret output bevares, så tidskovarians
indgår. Man kan ikke erstatte dette med et gennemsnit af punktvarianser.

For **middelværdien** gælder dog energibalancen
\(E[\bar D]=\bar B+C(E[X_{h+1}]-E[X_h])/\Delta\): det bounded
diffusionsled har nul forventet Ito-integral. Dette bestemmer ikke variansen.
En separat kontrol af den endelige FV-prognose giver største normaliserede
balanceafvigelse6,141e−5, under den eksisterende mean-gate0,005. FV's
face-flux og cellecentrerede reward giver en endelig diskretiseringsafvigelse;
kontrollen er ikke et matematisk fejlbevis.

Numerikken genbruger eksisterende konservativ zero-flux-FV-generator og
LAPACK-tridiagonal implicit Euler. Samme transition anvendes til alle tre
felter; reward halvt før/efter transition. Koefficienter fryses ved midtpunkt.
Implicit Euler er stadig førsteordens i tid. Signed r er ikke en negativ
sandsynlighed. Masser og varians klippes/renormaliseres ikke. B's integral
tilføjes præcist fra fine baselinegrid; kun fleksibilitetsintegralet bæres i
momentfelterne for at mindske subtraktionsfejl. Dirac-startens cellefejl
rapporteres særskilt. Coarse/fine er en udviklingskontrol, ikke et fejlbevis.
Meget små ikke-nul sigma kan kræve finere grids, fordi upwind-drift også
giver numerisk spredning; sigma=0 genbruger præcis egen interval-RK4.

Der eksporteres procesmiddel og procesvarians. **Intervalkvantiler/bånd er
ikke beregnet**: to momenter bestemmer ikke en ikke-lineær intervalfordeling.
Punktbånd, Gauss-antagelse, bånddækning eller intervalscore genbruges ikke
skjult. Måle-/modelfejl, parameter-, baseline- og inputusikkerhed er udeladt.

#### Den særskilte kalibreringsdiagnose

Langs den deterministiske bane lineariseres stateafvigelse eta og aktuelt
timeintegral z: \(d\eta=a\eta dt+b_0dW\), \(dz=c\eta dt\), med
a=D_x/C, c=D_x, b0=x(1−x). Den eksakte frozen-coefficient 2x2-transition
bruges sammen med positiv tre-node Gauss-kvadratur for støjkovariansen.
Kun z og dets cross-covariance nulstilles hver time; statevarians fortsætter.
Warm-up-støj er udeladt og P0=0 efter24 timer.

Enhedsprofiler sammenlignes med samplevarianser af fem matching weekdays,
ddof=1, mindst to målinger/bin. Én q>=0/uge tilpasses via varians-LS.
Den første time indgår nu: integreret støj har positiv varians i første
hele time, selv om startens statevarians er nul. Profilen er en lokal
approksimation og raw interday dispersion er ikke isoleret fysisk støj.
P>1/4, støttebrud eller numerisk afvisning forhindrer kalibreringsaccept.
q bruges aldrig automatisk som prognosens sigma.

Med f lineær og held U er de generelle intervalgrænser netop
D(1,U,bar B) og D(0,U,bar B), fordi endpoint-D er affin i B. Enhver proces-
statebane i [0,1] har timeforbrug inden for disse grænser. Større sigma
kan ikke få en observation uden for dem til at passe til et støjfrit
observationsmodel under samme frosne parametre og baseline.

#### Brug og kontrakt

Kør fra repoets rod med det eksisterende videnskabelige Python-miljø:

```text
python -B -m src.exe.run_model.bm_1.s3_interval fit --raw-csv HISTORY.csv --origin 2022-02-07 --demand-label hour_start --seconds 120 --max-calls 1000 --output NEW_FIT_DIR
python -B -m src.exe.run_model.bm_1.s3_interval predict --snapshot NEW_FIT_DIR/interval_snapshot.json --inputs-csv FUTURE.csv --observations-csv OBSERVED.csv --output NEW_FORECAST_DIR
python -B -m src.exe.run_model.bm_1.s3_interval predict --snapshot NEW_FIT_DIR/interval_snapshot.json --inputs-csv FUTURE.csv --sigma-x .2 --observations-csv OBSERVED.csv --output NEW_PROCESS_DIR
python -B -m src.exe.run_model.bm_1.s3_interval diagnose --snapshot NEW_FIT_DIR/interval_snapshot.json --raw-csv HISTORY.csv --output NEW_AUDIT_DIR
```

Inputs er rå Datetime,Demand,Price,Temp ved fit og Datetime,Price,Temp ved
forecast. Pris/temperatur angiver intervalstart. Demand-label er eksplicit:
hour_start eller hour_end; endlabels skifter én fysisk time, ikke én naiv
kalendertime. En endlabel ved cutoff kan bruges til forrige komplette time;
den tilsvarende fremtidige pris/temperatur bruges ikke i fit. Naive lokale
tider er unikke/sorterede; efterårs-DST-ambiguity afvises. Forårsugen har167
fysiske timer med entydige UTC-start/end. Ved endlabels skal observationerne
have entydige fysiske labels; tvetydige efterårslabels er ikke understøttet.

Snapshotkind/protokol/kildehash er særskilt fra punktversionen. Den fælles
S3-strukturvalidering genbruges internt; det er ikke en punktprognose eller
snapshotmigration. Fitted baseline, skalaer, C/k/Xi/a, g, starts, maske og
own coarse/fine-grids gemmes. Ændrede numeriske kilder kræver nyt eget fit
eller eksplicit reviewet migration. Første mellemtrins snapshot skal
læses med dets5189844-kilder, ikke relabels som den korrigerede version.

CSV har `prediction` som hourly average, integreret `baseline`, særskilt
`point_prediction`, state start/end, UTC-intervalstart/end og fælles metadata.
Procesversionen tilføjer `process_mean`, `process_variance` og state moments
start/end. Råvarians er normaliseret varians gange demand_scale². Forecast
gemmes før observationer åbnes; scoring følger samme finite measured maske
og råenhedens MSE/RMSE/MAE/bias. CLI returnerer2 ved afvist diagnose/fejl.

#### Faktiske udviklingsresultater

OriginFeb7,168 fysiske målte forecasttimer,720/840 historiske targettimer.
Realiseret pris/temperatur, MED-gennemsnit over tilgængelige observerede
målere: betinget udvikling, ikke blind vejrtest eller kausal prisstyringstest.
Kode og korrigeret protokol frøs vedb2a6776 før slutkørslerne.

| Korrigeret aktiv version | Resultat |
|---|---:|
| Fit | 20,995493s;118 residualkald; konvergeret |
| Grid / historisk meanforfining | 12/24 steps/h;3,250e−6 <=1e−5 |
| Største forecastmean-forfining | 1,550e−6 <=1e−5 |
| Interval-ODE / integreret baseline RMSE | 0,378396 / 0,379144 |
| Interval-ODE MAE / bias | 0,284756 / +0,063882 |
| Procesuge, sigma=.2 | 4,315248s;128/256 celler; alle numeriske gates består |
| Procesmiddel RMSE | 0,378612 |
| Største demand/state meanfejl | 5,681e−5 / 0,001954 |
| Største demand/state variansfejl | 1,553e−7 / 0,000240 |
| Største masserror | 9,104e−15; ingen repair |
| Uafhængig stokastisk mean-energibalance | 6,141e−5 normaliseret |
| Historisk intervalstøjdiagnose | 2,556601s; numerik består, fysisk fortolkning afvist |
| Målinger uden for integreret støtte | 662/720,91,944 % |
| Fine q / sigma pr.sqrt(time) | 29,723263 / 5,451905; afvist |
| Største fitted LNA-statevarians | 33,152770; nødvendig grænse0,25 |

C=1,361809h,k=.101995,Xi=.374275,a=.5,g(.5)≈−1. Prisrespons nær rand og
svag lokal Jacobianretning betyder, at vi ikke har dokumenteret robust
parameteridentifikation eller kausal effekt. RMSE-forbedringen fra baseline
er lille; målevariationen kan ikke kalibreres med processtøj alene i dette
frosne model. Ingen parameter-/støjsøgning startes for at omdøbe et failure.

Første mellemtrin:12,460015s/108kald,point-feature baseline,meanRMSE.386778,
før korrekt integrated-feature design. Begge fits er bevaret. Det korrigerede
design er valgt på matematisk observationskonsistens, ikke ved efterfølgende
forecast-modelvalg. Samlet fitcharge33,455508s/226kald er inden for oprindelige
120s/1000kald; der skete ingen budgetnulstilling. Der er også to gemte proces-
og diagnoseinvocations: første mellemtrin og én nødvendig kildekorrektion,
ikke skjulte retries. Ekstra FV-grid blev ikke brugt.

#### Kontroller og næste holdested

Ni nye fokuserede metoder dækker integrerede baselineharmonics, snapshot-
roundtrip uden fit, punkt/mean-adskillelse, fremtidsuafhængighed, endlabel-
availability, fysisk DST, integreret Brownsk kovarians, uafhængig Van-Loan-
reference, analytisk telegraph-integral, konservativ FV og nulgrænser.
De oprindelige16 G-kontroller bevares; alle25 består samlet på10,356s.
Lokal slutkontrol verificerer også
gamle punkt-snapshots, det nye own snapshot, positive fuldtime-varianser,
råvariansskalering og state-bound. Rå inputs/tidsserier/pakker er lokale;
remote leverance er kode, syntetiske tests og godkendte aggregerede resultater.

<a id="g-s3-interval-protocol-20261008"></a>
<a id="g-s3-interval-protocol-20261008--g-s3-timegennemsnit-protokol-før-faktisk-fit-og-proceskørsel"></a>
<a id="g-s3-interval-protocol-20261008--fast-kørsel"></a>
<a id="g-s3-interval-protocol-20261008--nødvendig-baselinekorrektion-efter-første-kørsel"></a>

## Historisk forsøgsprotokol: G_S3_INTERVAL_PROTOCOL_20261008.md

### G-S3 timegennemsnit: protokol før faktisk fit og proceskørsel

Pas startet 18:41:05 UTC, 8. oktober 2026. Johan bad om fortsættelse efter
grundversionen og oplyste, at Demand dækker forbrug over en hel time og
efter hans erindring er Power. Kl.10 tolkes foreløbigt som kl.10–11 efter
hans antagelse; det er ikke verificerede kildemetadata. F holdes adskilt.

Den nye vej er et selvstændigt interval-snapshot. G's punktkode/snapshots
ændres ikke og migreres ikke skjult. S3 lambda=0, lineær f, to-stykkers g,
fælles C/k/Xi/a, 35 egne histories med 24 timers warm-up, egne syv medianstarts
og mandagsfrossen origin bevares. Eksogen WM-baseline genbruges som frosset
kontinuerlig inputmodel; det er ikke en ny baseline- eller F-undersøgelse.

#### Fast kørsel

1. Én ny `fit_interval` på det eksisterende lokale Jan1–Feb14 input med
   origin Feb7 og demand-label hour_start. Kun komplette intervaller med
   end<=origin indgår. Maksimum 120 sekunder og 1000 residualkald samlet,
   750 per start, eksisterende 12/24/48 RK4-grids og demandforfining<=1e-5.
   Ingen punkt-fitparametre genmærkes som intervalfit. Ingen modeludvælgelse.
2. Én input-only deterministisk intervaluge, gemt før scoring på 168 timer.
   CSV indeholder point-readout særskilt fra hourly-average forecast og
   integrated baseline. UTC-start/end deklarerer de fysiske intervaller.
3. Én intervalprocesuge med eksplicit illustrations-sigma=.2 per sqrt(time),
   128/256 celler, trin<=1/24 og 1/48 time og 120 sekunders loft. Samme
   konservative FV-generator og implicit tridiagonal solve som punktvejen.
   Middel/varians af timeintegralet beregnes med tre joint state/rewardfelter.
   Kontroller normaliseret demand/state meanfejl<=.005 og variansfejl<=.002,
   masserror<=1e-10, finitte felter, ikke-negativ varians. Ingen klipning,
   observationstøj, normalisering eller efterfølgende sigmajustering.
   Kun hvis første grid numerisk afvises: én yderligere 256/512-forfining
   med trin<=1/48 og 1/96 og 120-sekunders loft. Ingen flere retries.
4. Én historisk intervalstøjdiagnose, højst60s. Matching-weekday raw sample-
   varians ddof=1, mindst to målte værdier/bin. Unitprofiler bruger joint
   lineær (state,integreret-output)-kovarians med positiv 3-node Gauss-
   støjkvadratur, P0=0 efterwarmup; hver timers outputaccumulator nulstilles.
   Time0 indgår nu: den første HELE time har processpredning selv ved P0=0.
   q>=0 varians-LS bruger alle24 timer. Historisk meanforfining<=1e-5,
   unit- og fitted profil-stdforfining<=1e-4. P>1/4 og målte gennemsnit uden
   for den frosne integrerede demandstøtte er hver sin afvisningsgrund.
   En kandidat kræver særskilt ikke-lineær validering; q bruges aldrig
   automatisk i den illustrative procesprognose.

Output er E[hourly-average D] og Var[hourly-average D], ikke middel/varians
af endpoint-D og ikke gennemsnittet af marginalvarianser. Intervalkvantiler
og dækning beregnes ikke fra kun to momenter; punktbånd genbruges ikke som
timebånd. Score råenheds MSE/RMSE/MAE/bias på samme finite måltemaske efter
forecastexport. Intet bredt støjfit, parameterbank eller nyt F-studie.

Kontroller før historisk kørsel omfatter snapshotroundtrip, point/mean
adskillelse, future-demand-uafhængighed, endlabel-cutoff, forårs-DST,
integreret Brownsk kovarians, uafhængig Van-Loan-reference, analytisk
telegraph-integral, konservativ FV og nulstøj/nulfleksibilitetsgrænser.
Rå input/resultattidsserier bevares lokale. Johans tilladelse til kode,
syntetiske tests og aggregerede resultater/merge gælder samme eksempel.

#### Nødvendig baselinekorrektion efter første kørsel

Første kode/faktiske fit er bevaret ved5189844. Fit12.4600146s/108kald,
point-feature baseline, meanRMSE.386778 og bestået numerik. Det er et
mellemtrin: den punktfittede statistiske featurekurve blev efterfølgende
integreret, hvilket ikke matcher timebaseret baselinefit og giver fasefejl.
Ingen baselinefamilie/hyperparameter eller modelvalg ændres efter scoring.
Featuredesignet skal integreres over samme fysiske time, før det eksisterende
robuste WM-fit bruges. Der anvendes samme60-sample/h trapezoidgrid, kendt
historisk temperaturinterpolation og clamping af sidste temperatur ved cutoff.
Regression integrerer de lineære features; WM's begrænsede baselinefortsættelse
med clipping bevares eksplicit. Det er stadig en statistisk inputmodel.

Én nødvendig korrigeret fitinvocation tillades inden for OPRINDELIGE120s/1000
residualkald: restloft107.53s og892kald. Ingen udvælgelse mellem de to fits
efter score: den korrigerede observationsdesign-vej er den aktive. Første fit
og resultater slettes ikke. Én korrigeret forecast/procesuge og diagnose køres
på samme grids/gates/runtime. Dette er begrundet kildekorrektion, ikke et
numerisk retry eller en skjult nulstilling af budgettet. Ingen accept erklæres.
