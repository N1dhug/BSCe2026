# G: procesusikkerhed og empirisk outputresidual

De to lag holdes adskilt: den oprindelige SDE-procesvej giver punktvise fordelinger, mens det senere empiriske outputresidual beskriver samlet model-/baseline-/målefejl. Proces- og residualvarianser må ikke automatisk lægges sammen. Negative kalibreringsresultater og forskellen på punkt- og timebånd er bevaret nedenfor.

Aktuel brugerindgang: [BM-guiden](../../root/BM_GUIDE.md). Afsnit mærket protokol bevarer det allerede udførte forsøgs numerik, stopregler og provenance; deres gamle kørselsordrer er historiske. Nye fits eller brede benchmarkkørsler kræver særskilt budget.

Statusord i de bevarede originalafsnit beskriver den daværende delundersøgelse. Den aktuelle samlede accept og senere udførte arbejde står i [sammenligningen](COMPARISON.md) og [status](STATUS.txt); ældre formuleringer som endnu ikke udført må ikke læses som global slutstatus.

Originale kilder og SHA256 findes i [arkivets stikort](../../archive/bm/old-paths.json). De oprindelige snapshots/protokoller er uændrede; dokumentkonsolideringen ændrer ingen matematisk model.

<a id="g-s3-uncertainty-guide-20261008"></a>
<a id="g-s3-uncertainty-guide-20261008--g-s3-grundversion-med-afgrænset-procesusikkerhed"></a>
<a id="g-s3-uncertainty-guide-20261008--matematik-og-fortolkning"></a>
<a id="g-s3-uncertainty-guide-20261008--brug"></a>
<a id="g-s3-uncertainty-guide-20261008--faktisk-feb7-udviklingskørsel"></a>
<a id="g-s3-uncertainty-guide-20261008--slutstatus-og-senere-arbejde"></a>

## Model, anvendelse og bevaret evidens: G_S3_UNCERTAINTY_GUIDE_20261008.md

### G-S3 grundversion med afgrænset procesusikkerhed

Efterfølgende er [den selvstændige timeversion](G_INTERVAL_MODEL.md#g-s3-interval-guide-20261008)
implementeret med integrerede baselinefeatures og joint state/output-momenter.
Den er den aktive hourly-average vej. Denne punktversion bevares selvstændigt;
dens bånd er ikke timebånd. Begge historiske processtøjdiagnoser er afvist.
De senere åbne punkter beskrives i timevejens seneste holdested.

Den deterministiske fit/snapshot/predict/score-vej er færdig. G kan nu også
beregne en ikke-lineær procesfordeling fra samme fitted model og eksportere
middel, varians, punktvise kvantiler og fælles fejlmål. **Historisk kalibrering
af støjstyrken er afvist i udviklingseksemplet**; den fungerende procesvej
kræver en eksplicit scenariestyrke. Ingen totalusikkerhed eller fysisk
95%-dækningsaccept er erklæret. Dette er første brugbare grundversion.

Se [deterministisk brug](../../archive/bm/G/forecast-development.md#g-s3-standalone-guide-20261008),
[protokol før kørsel](G_UNCERTAINTY.md#g-s3-uncertainty-protocol-20261008),
[modelkode](../../src/method/bm_1_s3_uncertainty.py) og
[indgang/eksport](../../archive/bm/ORIGINALS.md#original-tree).

#### Matematik og fortolkning

Alle dynamiske tider er timer. U er normaliseret til [0,1], f(x)=1−2x,
g er snapshotets kontinuerte monotone to-stykkers respons. For baseline B(t):

\[
\delta(x,t)=\tanh\{k[1-2x+g(U(t))]/2\},\qquad
D(x,t)=B(t)+\Xi\delta(x,t)
\begin{cases}1-B(t),&\delta\ge0,\\B(t),&\delta<0.\end{cases}
\]

\[
dX_t=\frac{D(X_t,t)-B(t)}{C}\,dt+
\sigma_xX_t(1-X_t)\,dW_t,\quad \operatorname{Var}(dW_t)=dt.
\]

Hver dag starter fra sit eget frosne historiske medianstate med P0=0.
Der assimileres ingen fremtidige målinger. Den numeriske procesvej genbruger
den eksisterende konservative Fokker–Planck-FV på [0,1] med zero-flux-rand,
upwind-drift og implicit Euler. Formelt er ligningen
\(\partial_t p=-\partial_x(ap)+\tfrac12\partial_{xx}(b^2p)\).
Sandsynlighedsmasser er de numeriske ukendte, ikke densityværdier som vægte.
Ingen klipning eller renormalisering reparerer et forløb. Dirac-starten
repræsenteres i én celle; afvigelsen fra ønsket start rapporteres særskilt.
Numerisk forfining er en udviklingskontrol, ikke et bevis for en eksakt lov.

Output er E[D(X)] og Var[D(X)] beregnet direkte fra masserne. E[D(X)] er
generelt forskellig fra D(E[X]) og ODE-prognosen. Båndgrænserne er den
diskrete generaliserede inverse CDF's .025/.975-kvantiler af D(X), ikke
Gauss-bånd. Ved sigma=0 genbruges præcis S3's RK4-prognose med nul varians.
Den særlige nulgrænse undgår kunstig FV-advektionsspredning.

Den separate historiske diagnose bruger lokal lineær støjpropagation:
\(P'=2(D_x/C)P+[x(1-x)]^2\) ved enhedsamplitude, P0=0 efter warm-up.
For delta>=0 er \(D_x=-k\Xi(1-B)(1-\delta^2)\); for delta<0 erstattes
1−B af B. Ved knækket bruges den højre demandgren som i eksisterende kode.
Demand-enhedsvariansen er \(D_x^2P\). Med matching-weekday samplevarianser
v (ddof=1) og gennemsnitlige fem-historie enhedsprofiler w tilpasses
\(q=\max(0,w^Tv/(w^Tw))\). Nulsensitivitet er uidentificerbar.
Time 0 udelades. Dette fit identificerer ikke fysisk processtøj alene.
Varians over 1/4 strider mod enhver lov på [0,1]; understøttelsesbrud i
frosne D([0,1]) viser uforenelighed med et observationsfejlfrit processpor
under den erklærede udviklingsapproksimation. Afvist q bruges aldrig skjult.

#### Brug

Kør fra repository-roden med det eksisterende videnskabelige Python-miljø:

```text
python -B -m src.exe.run_model.bm_1.s3 point-process diagnose --snapshot S3_SNAPSHOT.json --raw-csv HISTORY.csv --output NEW_AUDIT_DIR --seconds 60
python -B -m src.exe.run_model.bm_1.s3 point-process predict --snapshot S3_SNAPSHOT.json --inputs-csv FUTURE.csv --sigma-x 0.2 --output NEW_PROCESS_DIR --observations-csv OBSERVED.csv
```

.2 er kun en illustrationsstyrke per sqrt(time), ikke et accepteret fit.
API: `diagnose_calibration`, `process_forecast`, `score_process_forecast`.
Inputs/snapshot/kalender følger S3-kontrakten. Forårs-DST bevarer 167 fysiske
timer; efterårets gentagne naive lokale timer afvises. Forecast åbner ingen
observationsfil før CSV/JSON er gemt. Outputmappen skal være ny.
CLI returnerer 0 for komplet processcenarie; 2 for numerisk fejl/ugyldigt
input eller afvist diagnose. En diagnose-kandidat er ikke kalibreringsaccept.

CSV har eksisterende baseline/ODE/tid/pris plus `process_mean`,
`process_variance`, `process_lower95`, `process_upper95`, `process_state_mean`
og `process_state_variance`. Rå demandvarians er normaliseret varians gange
skala²; middel/kvantiler transformeres affint. Observationer/måltemaske
tilføjes kun ved scoring. Udeladte usikkerhedskilder er deklareret i JSON:
parameter-, baseline-, input- og observations/modelfejl. Procesvarians er
derfor ikke en estimeret totalvarians.

Fælles scorer er MSE/RMSE/MAE/bias på samme finite måltemaske for procesmiddel,
ODE og baseline. Båndets intervalscore er bredde+40·underskridelse+
40·overskridelse for alpha=.05. Dækning er punktvis og ikke samtidig over
ugen. Ingen refit eller støjjustering ud fra scoring. Disse felter gør WM-
sammenligning praktisk, når population, information, enheder og origin
matches; der er endnu ikke udført et fælles WM-benchmark.

#### Faktisk Feb7-udviklingskørsel

Protokol/kode frøs ved commit 1bba8e7 før begge nye kørsler. Det oprindelige
S3-fit genbruges; nul nye optimizerkald. Første FV-grid bestod, så den
betingede ekstra forfining blev ikke brugt. Tal her er aggregerede:

| Kontrol/resultat | Værdi |
|---|---:|
| Historisk diagnose | 1,127 s; numerik bestået, fysisk fortolkning afvist |
| Understøttelsesbrud | 657/720 historiske målinger, 91,25 % |
| Fine fitted q / sigma pr. sqrt(time) | 25,388687 / 5,038719; afvist |
| Største LNA-statevarians | 28,744177; nødvendig grænse 0,25 |
| Historisk demand / enheds-std / fitted profil-std fejl | 7,245e−8 / 6,123e−8 / 9,271e−8 |
| FV processcenarie | 2,368 s; sigma=.2; 168 timer; alle gates bestået |
| Grids / maksimumtrin | 128/256 celler; 1/24 og 1/48 time |
| Største mean-fejl, demand / state | 6,607e−5 / 0,001955 |
| Største variansfejl, demand / state | 1,922e−7 / 0,000225 |
| Største demandkvantil / state-CDF fejl | 0,000201 / 0,016145 |
| Procesmiddel / ODE / baseline RMSE | 0,377679 / 0,377435 / 0,378180 |
| Dækning / middelbredde / intervalscore | 4,1667 % / 0,027203 / 10,872159 |

Johan præciserede efter kørsel, at hver værdi er forbrug over en hel time,
og at Demand efter hans erindring er Power. Det peger på timegennemsnit
af effekt; fysisk enhed og effekt/energi-konvention er endnu ikke verificeret.
Koden og snapshotet bruger fortsat den eksplicit erklærede punktapproksimation.
Støttebrud og dækningsafvisning ovenfor gælder dette point-output-kontrakt;
de er ikke en afvisning af enhver intervalbaseret G-model. Et fysisk timeoutput
kræver en intervaloperator: for effektsignal er \(\bar D_h=\int_h^{h+1}D_tdt\)
divideret med intervallets varighed. Deterministisk kan eksisterende G-kode
bruge \(\bar B_h+C(X_{h+1}-X_h)/\Delta t\). I SDE følger i stedet
\(\int Ddt=\int Bdt+C\Delta X-C\int b(X)dW\), så man må ikke genbruge
den deterministiske endpointidentitet som stokastisk intervalreadout.
Det kræver også tidskovarians, ikke blot punktvise FV-marginaler.

Scorer er i uverificerede rå workbook-enheder. Realiseret pris/temperatur
er betingede inputs, ikke en blind vejrprognose. Båndene er meget smallere
end målespredningen: numerisk gyldighed er tydeligt adskilt fra brugbar
observeret dækningskalibrering. En større sigma kan ikke flytte demand uden
for D([0,1]); vi starter derfor ingen bred støjsøgning for at skjule dette.

#### Slutstatus og senere arbejde

Otte oprindelige S3-tests og otte nye proceskontroller består. De første
15 blev kørt samlet i 2,474 s; efter tilføjet ikke-nul spring-DST-check
bestod de otte proceskontroller i 2,595 s. Kontroller dækker variansfit,
native demandformel, nulgrænse uden refit/demandleak, konservativ ren
diffusion, budgetstop, måltemaske/skalering, numerisk afvisning, kalender
og forecast bevaret før scoringfejl. Windows-sandboxens atomiske testfil-
blokering blev løst ved autoriseret lokal testkørsel uden at svække I/O.

<a id="g-s3-uncertainty-protocol-20261008"></a>
<a id="g-s3-uncertainty-protocol-20261008--afgrænset-s3-proceslag-protokol-før-historisk-kørsel"></a>
<a id="g-s3-uncertainty-protocol-20261008--kørsel-og-stopregler"></a>

## Historisk forsøgsprotokol: G_S3_UNCERTAINTY_PROTOCOL_20261008.md

### Afgrænset S3-proceslag: protokol før historisk kørsel

8. oktober 2026. Implementeringspas startet 18:15:43 UTC. Dette er G's
eksisterende Ito-SDE, ikke et nyt F-studie. S3 lambda=0, fitted baseline,
normalisering, to-stykkers g og syv egne mandagsfrosne starts bevares.
Ingen ny optimizer, parameterbank, observationsstøj eller modeludvælgelse.
Den foreløbige støtteanalyse er allerede set: 657/720 historiske målinger
ligger uden for det frosne D(X)-interval. Denne kørsel er udvikling, ikke blind.

#### Kørsel og stopregler

1. Én træningsdiagnose på det lokale Feb7-snapshot og præcis dets historik.
   Maksimum 60 sekunder. Fem historiske målte timer per ugedag/time, sample-
   varians ddof=1, mindst to målinger. Time 0 udelades fra amplitudefit.
   Én q>=0 per uge: q=max(0,w·v/(w·w)), w er gennemsnittet af fem
   enhedsvarianser. Coarse/fine bruger snapshotgrids. Demandfejl <=1e-5,
   enheds- og fitted profil-standardafvigelsesfejl <=1e-4. P>1/4 og målinger
   uden for D([0,1]) er hver sin afvisningsgrund. q anvendes aldrig skjult
   som en accepteret støjstyrke. Numerisk bestået er ikke fysisk accept.
2. Én input-only FV-uge med udtrykkelig **illustrationsstyrke sigma=.2 per
   sqrt(time)**. Den er ikke et historisk støjfit. P0=0 ved hver egen start.
   128/256 celler, maksimumstrin 1/24 og 1/48 time, budget 120 sekunder.
   Hver prisgrænse segmenteres. Eksisterende konservativ zero-flux FV med
   implicit Euler genbruges; sandsynligheder klippes/renormaliseres ikke.
   Kontroller normaliserede middelfejl (state/demand) <=.005, variansfejl
   <=.002, demandkvantilfejl <=.005 og state-CDF-fejl <=.05 ved .25/.5/.75
   efter time 0. Disse er udviklingstolerancer, ikke en rigorøs fejlgrænse.
   Dirac-starten ligger i én celle: centreringsfejlen rapporteres særskilt.
3. Kun ved numerisk FV-afvisning tillades én på forhånd afgrænset forfining:
   256/512 celler, trin 1/48 og 1/96 time, samme 120-sekunders loft. Ingen
   sigmaændring, gateændring eller automatisk yderligere søgning.
4. Prognosen gemmes før observationerne åbnes. Derefter score samme endelige
   måltemaske i råenheder: MSE/RMSE/MAE/bias, punktvis bånddækning, bredde og
   intervalscore. Kvantiler beregnes direkte fra D(X)-sandsynlighedsmasser,
   ikke som klippede Gauss-bånd. E[D(X)] holdes adskilt fra ODE-prognosen.

Diagnosen bruger P'=2(D_x/C)P+[x(1-x)]² ved enhedsamplitude, P0=0 efter
24 timers warm-up. Dette er en lokal approksimation; stor fitted q giver
ikke automatisk en gyldig ikke-lineær lov. Rå historisk spredning indeholder
også ændrede inputs/baseline og identificerer ikke alene fysisk processtøj.

Historisk kalibrering får særskilt status. En numerisk god proces-scenarie-
vej kan afleveres selv om kalibrering af observeret totalusikkerhed afvises.
Ingen garanti om 95% observed-coverage eller samtidig ugedækning.
Rå inputs, tidsserier, snapshots og fulde outputpakker forbliver lokale.
Kode, syntetiske tests og aggregeret dokumentation må sendes og merges efter
Johans allerede givne tilladelse. F's kode, budget og accept ændres ikke.

<a id="g-s3-residual-guide-20261008"></a>
<a id="g-s3-residual-guide-20261008--g-særskilt-empirisk-residuallag"></a>
<a id="g-s3-residual-guide-20261008--matematik-og-fortolkning"></a>
<a id="g-s3-residual-guide-20261008--brug"></a>
<a id="g-s3-residual-guide-20261008--faktisk-udviklingsresultat"></a>
<a id="g-s3-residual-guide-20261008--holdested"></a>

## Model, anvendelse og bevaret evidens: G_S3_RESIDUAL_GUIDE_20261008.md

### G: særskilt empirisk residuallag

Johan godkendte8.oktober et enkelt, tydeligt mærket statistisk udviklingslag.
G's deterministiske intervalmodel, proces-scenarievej og fysisk styring bevares.
F/bank/BM2 ændres ikke. Se [fast protokol](G_UNCERTAINTY.md#g-s3-residual-protocol-20261008),
[metode](../../src/method/bm_1_s3_residual.py) og
[kørbar indgang](../../archive/bm/ORIGINALS.md#original-tree).

#### Matematik og fortolkning

Modellen er \(Y_h=G_h+E_h\). G_h er den eksisterende deterministiske
timegennemsnitsprognose. På de fem egne historiske uger genafspilles hver
targetdag fra den gemte weekdaymedianstate, som også bruges ved forecast.
Det er ikke hver dags eget efter-warm-up-state eller en state tilpasset
targetdagens målinger. Baseline og C/k/Xi/g genbruges uden fit.
De finite historiske fejl er e_i=Y_i−G_i, i=1,...,n. Den fælles lov er
\(\widehat F_E(z)=n^{-1}\sum_i1\{e_i\le z\}\).

Dette giver eksakt for den **erklærede empiriske lov**:

\[
E[Y_h]=G_h+\bar e,\qquad
\operatorname{Var}(Y_h)=n^{-1}\sum_i(e_i-\bar e)^2.
\]

Derfor bruges ddof=0: de n punkter er en diskret lov med kendte masser1/n.
Det er ikke en påstand om unbiased estimering af ukendt fysisk varians.
Middelfejlen bevares som en særskilt kolonne; den oprindelige G-prognose
forsvinder ikke. Ved råskalering y=demand_min+demand_scale·y_norm bliver
fejlmiddel ganget med demand_scale og varians med demand_scale².

Inverse empiricalCDF bruges til q_(alpha/2),q_(1−alpha/2). Det centrale
bånd er [G_h+q_low,G_h+q_high]. Ingen Gauss-antagelse er nødvendig.
Empirisk diskretisering kan give forskellig faktisk masse ved atoms/ties;
nominel95% er ikke garanteret fremtidig95%-dækning. Båndscore er
width+2/alpha·(lower−y)_++2/alpha·(y−upper)_+, samme finite raw mask som
MSE/RMSE/MAE/bias. Ingen slutmålinger læses før prognosen er gemt.

Laget er et samlet outputfejl/residuallag. Det identificerer ikke målerstøj
adskilt fra baseline-/model-/parameterfejl. Iid er en simpel arbejdsmodel;
kun marginale bånd og scores beregnes, ikke dokumenteret fælles168h-dækning.
Lag1-korrelation auditeres på par af finite målinger i samme dag med præcis
én fysisk time mellem startlabels. Missing timer og daglige resets bridged
ikke. Lagkorrelationen er et beskrivende check, ikke et automatisk modelvalg.

Alle fem uger indgår allerede i parentfit. Fejlbanken er derfor in-sample,
ikke en separat calibration/holdout/conformal-bank. Små træningsfejl kan
give for snævre fremtidsbånd. Dette skal stå ved siden af resultatet.
Temperatur/price er historiske/frosne scenarieinputs som i G-timeversionen.

WM4 har dF=Rdt, dR=(r_F F+r_U U+r_R R)dt+sigma dW og I=O+R, med
additiv stateproces og dens tidsafhængige kovarians. Vores outputresidual
er ikke den samme SDE eller en erstatning for G's multiplikative stateproces.
Den er et afgrænset statistisk supplement, som gør mean/variance/bånd/score
praktiske at sammenligne, hvis informationscutoff, timeoperator, population
og scenarie også matcher. Ingen WM-kode eller F-studier ændres.

Denne version har sigma_x=0. Procesmomenter og residualbank kombineres ikke:
replayfejlene indeholder allerede samlet uobserveret variation; en ekstra
procesvarians ville risikere dobbelttælling. To integrerede procesmomenter
bestemmer heller ikke en fuld convolutionfordeling. En sådan kombination
kræver en senere eksplicit modelbeslutning og egen validering.

Residualoutput indgår ikke i G's fysiske energibalance eller priskontrol.
Additive fejl kan give negative rågrænser; der klippes ikke skjult. Antallet
rapporteres. Identificerede fysiske enheder/time-startmetadata er stadig åbne:
hour_start er Johans usikre arbejdsantagelse.

#### Brug

```text
python -B -m src.exe.run_model.bm_1.s3 residual fit --snapshot INTERVAL_FIT/interval_snapshot.json --raw-csv HISTORY.csv --seconds 30 --output NEW_RESIDUAL_FIT
python -B -m src.exe.run_model.bm_1.s3 residual predict --snapshot INTERVAL_FIT/interval_snapshot.json --residual-snapshot NEW_RESIDUAL_FIT/residual_snapshot.json --inputs-csv FUTURE.csv --observations-csv OBSERVED.csv --output NEW_RESIDUAL_FORECAST
```

Fitnavnet angiver estimering af fejlloven; der er **nul optimizerkald**.
Parentens præcise canonical-JSON-fingerprint og egne kildehashes kontrolleres
ved hver brug. Parent/snapshot eller numeriske kildeændringer afvises; ingen
skjult migration/refit. Det eksisterende point-/intervalsnapshot er uændret.
Nye outputmapper kræves. JSON skrives atomisk uden overwrite.
Historisk replay, fejlbank, snapshots, CSV og figurer forbliver lokale.

CSV bevarer prediction, baseline og point_prediction og tilføjer
residual_mean, residual_variance, observed_mean, observed_variance,
observed_lower og observed_upper. Her betyder observed_* den modellerede
observerede prognose; observed alene tilføjes først af scoring.

#### Faktisk udviklingsresultat

Kode/protokol frøs vedafa9af0 før én replay og én forecast/scoring.
Samme Feb7-eksempel som tidligere; originFeb7,720 historicaltargets,
168 forecasttimer. Ingen søgning/retry eller udvidelse af fitbudget.

| Kontrol/resultat | Værdi |
|---|---:|
| Replaytid / nye optimizerkald | 1,525249s / 0 |
| Forecasttid før scoring | 0,458736s |
| Coarse/fine replaymiddel | 3,120e−6 <=1e−5 |
| Historisk fejlmiddel / SD i råenhed | −0,003003 / 0,198620 |
| Historisk replay RMSE i råenhed | 0,198642 |
| Støttebrud / supportafstand RMSE lower bound | 662/720 / 0,180648 |
| Lag1-korrelation / gyldige par | 0,380915 / 690 |
| Forecast pureG / residualmean RMSE | 0,378396 / 0,377901 |
| Residualmean MAE / bias | 0,284091 / +0,060879 |
| Nominelt95%-bånd: faktisk udviklingsdækning | 120/168 =71,4286% |
| Middelbredde / intervalscore i råenhed | 0,736655 / 3,617927 |
| Negative rå lower bounds | 0; ingen clipping |

Supportafstand er afstanden fra hver Y til [D(1,U,barB),D(0,U,barB)].
Den giver en nedre grænse for squared error for ethvert forbrug inden for
den frosne models støtte, uanset statebane. Grænsen svarer til82,70% af
det aktuelle replays MSE. Dette er en modelbegrænsning under frosne inputs/
parametre, ikke bevis for målerstøj eller for at alle andre modeller fejler.

Trænings-SD er klart mindre end forecastugens RMSE. Lag1≈.38 og dækning71.4%
viser, at den simple iid-træningslov ikke giver veldækkende95%-bånd her.
Den lille biasjustering forbedrer RMSE marginalt; det valgte lag ændres ikke
på baggrund af denne uge. Ingen kalibreringsaccept, kausal priseffekt,
robust parameteridentifikation eller fysisk totalusikkerhed erklæres.

Fire nye tests dækker exact empiricalmoments/quantiles/scaling, finite
scores/intervalscore, ingen fremtid/fit og parentbinding samt missing-hour
lagpar/support-distance bound. Alle29 fokuserede G-tests består samlet
på13,518s. En indledende Series-vs-DatetimeIndex-fejl blev rettet på syntetiske
tests før første sciencekørsel; ingen historisk retry skjules.

<a id="g-s3-residual-protocol-20261008"></a>
<a id="g-s3-residual-protocol-20261008--g-afgrænset-empirisk-outputfejl-før-faktisk-kørsel"></a>

## Historisk forsøgsprotokol: G_S3_RESIDUAL_PROTOCOL_20261008.md

### G: afgrænset empirisk outputfejl, før faktisk kørsel

Pas startet8.oktober2026 kl.19:36:59 UTC (21:36:59 Copenhagen).
Johan har godkendt et enkelt residuallag, tydeligt mærket udviklingsmodel.
Det eksisterende G-fit, timeoperator, baseline og kontrol ændres ikke.

Modellen er Y_h=G_h+E_h. E_h har en fælles empirisk fordeling af de endelige
historiske replayfejl, inklusive deres middel. Det er model-/baseline-/måle-
residual samlet, ikke identificeret fysisk processtøj eller separat målefejl.
E_h antages iid alene som en enkel udviklingsmodel. Historisk lag1-korrelation
rapporteres; den kan vise, at iid er utilstrækkelig. Ingen AR/OU-model vælges
automatisk ud fra dette eller forecastugen. WM har additiv diffusion i R;
outputfejlen her er en særskilt statistisk approksimation, ikke WM's SDE.

1. Én historisk replay, max30s, med uændret interval-snapshot fra b2a6776
   og den samme lokale Jan1–Feb14 udviklingscase. Kun parentens fem uger
   førFeb7 og samme måltemaske bruges. Ingen optimizer eller baselinefit.
   Hver historisk dags targetbane starter fra den gemte mandagsfrosne
   weekdaymedian, så initialization svarer til den leverede forecastregel.
   Alle fem uger deltager allerede i parentfit; dette er **in-sample**.
2. Coarse/fine med parentens egne12/24 steps/h og starts; maxmeanerror<=1e−5.
   Støttebrud, afstand til fysisk støtte (RMSE lower bound for frossen model),
   baselineeffekt, residualmiddel/varians og fysisk tilstødende lag1-par gemmes.
   Missing timer/daglige resets må ikke tælles som tilstødende par.
3. Én input-only ugeforecast med empiriske95%-kvantiler (inverse empiricalCDF),
   gemt før observationsfilen åbnes. PureG, residualmiddel og observeret
   middel/varians/bånd er særskilte kolonner. Sigma_x=0; ingen sammenblanding
   med integrerede procesmomenter, ingen Gauss- eller fysisk clipping.
4. Derefter samme168 udviklingstimer scores med fælles rå MSE/RMSE/MAE/bias,
   bånddækning, bredde og intervalscore. Ingen modelvalg/biasfravalg/ændret
   kvantil på baggrund af slutugen. Det er en allerede set udviklingsuge,
   ikke en blind kalibreringsaccept eller kausal prisstyringstest.
5. Egne sourcehashes og exact parentfingerprint binder residualsnapshot.
   Rå fejl/replay/snapshots/forecast/figurer/pakke forbliver lokale. Kun kode,
   syntetiske tests og godkendte aggregerede resultater fra samme case merges.

Ingen ny model/baseline-/parameter-/sigma-søgning, bank/BM2/F-opgave eller
budgetnulstilling. Original otte-timers-ramme og tidligere charges bevares.
Næste accept kræver senere data, korrekt fysisk metadata og kontrol af
bias/stationaritet/seriel afhængighed; udviklingsbånd giver ikke denne accept.
