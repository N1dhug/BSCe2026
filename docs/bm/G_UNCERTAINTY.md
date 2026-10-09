# G: procesusikkerhed og empirisk outputresidual

G's procesfordeling og empiriske outputresidual beskriver forskellige ting.
Punktprocessen følger den multiplicative state-SDE; residualen samler
model-/baseline-/målefejl. Deres varianser lægges ikke automatisk sammen.
Begge historiske processtøjdiagnoser blev afvist i udviklingseksemplet.

Aktuel brug og samlet accept står i [BM-guiden](README.md) og
[sammenligningen](COMPARISON.md). Den separate [timeversion](G_INTERVAL_MODEL.md)
giver integrerede procesmomenter. Denne punktversions kvantiler er ikke
timebånd. Oprindelige forsøgsprotokoller findes i [originalarkivet](../../archive/bm/ORIGINALS.md#original-tree).

<a id="g-s3-uncertainty-guide-20261008"></a>
<a id="g-s3-uncertainty-guide-20261008--matematik-og-fortolkning"></a>
## Punktprocessens lov og numerik

Tid er timer, U∈[0,1], f(x)=1−2x og g er snapshotets monotone to-stykkers respons:

\[
\delta(x,t)=\tanh\{k[1-2x+g(U(t))]/2\},\qquad
D(x,t)=B(t)+\Xi\delta(x,t)
\begin{cases}1-B(t),&\delta\ge0,\\B(t),&\delta<0.\end{cases}
\]

\[
dX_t=\frac{D(X_t,t)-B(t)}{C}\,dt+
\sigma_xX_t(1-X_t)\,dW_t,\qquad \operatorname{Var}(dW_t)=dt.
\]

Hver dag starter fra sin egen frosne historiske medianstate med P0=0;
ingen fremtidige målinger assimileres. Konservativ zero-flux FV på [0,1]
løser \(\partial_t p=-\partial_x(ap)+\tfrac12\partial_{xx}(b^2p)\) med
upwind-drift og implicit Euler. Numeriske ukendte er sandsynlighedsmasser.
Der er ingen repair ved klipning/renormalisering. Dirac-startens cellefejl
rapporteres særskilt; coarse/fine er en udviklingskontrol, ikke et fejlbevis.

E[D(X)] og Var[D(X)] beregnes fra masserne. E[D(X)] er generelt forskellig
fra D(E[X]) og ODE-prognosen. Bånd bruger den diskrete inverse CDF's
.025/.975-kvantiler af D(X), ikke Gauss-bånd. Sigma=0 genbruger S3's RK4
med nul varians for at undgå kunstig FV-advektionsspredning.

Udviklingsgates var normaliseret state/demand-meanfejl≤.005, variansfejl≤.002,
demandkvantilfejl≤.005 og state-CDF-fejl≤.05 ved .25/.5/.75 efter time 0.
Hver prisgrænse segmenteres. De anvendte grids fremgår af resultattabellen.
Parameter-, baseline-, input- og observations/modelfejl er udeladt;
procesvarians er derfor ikke estimeret totalvarians.

## Lokal støjdiagnose

Ved enhedsamplitude følger diagnosen
\(P'=2(D_x/C)P+[x(1-x)]^2\), P0=0 efter 24 timers warm-up.
For delta≥0 er \(D_x=-k\Xi(1-B)(1-\delta^2)\); ellers erstattes 1−B af B.
Ved knækket bruges højre demandgren. Demand-enhedsvariansen er \(D_x^2P\).
Matching-weekday samplevarianser v bruger ddof=1 og mindst to målinger/bin;
w er gennemsnittet af de fem enhedsprofiler. Én ikke-negativ amplitude er
\(q=\max(0,w^Tv/(w^Tw))\). Nulsensitivitet er uidentificerbar, og time 0
udelades. Historisk demandforfining blev kontrolleret mod 1e−5 og unit-/fitted
profil-std mod 1e−4. Raw interday dispersion isolerer ikke fysisk processtøj.

P>1/4 strider mod enhver lov på [0,1]. Målinger udenfor det frosne D([0,1])
er desuden uforenelige med en observationsfejlfri punktproces. Numerisk
bestået diagnose er ikke fysisk accept, og afvist q bruges ikke som sigma.

<a id="g-s3-residual-guide-20261008"></a>
<a id="g-s3-residual-guide-20261008--g-særskilt-empirisk-residuallag"></a>
<a id="g-s3-residual-guide-20261008--matematik-og-fortolkning"></a>
## Empirisk residual til timeprognosen

Den separate model er \(Y_h=G_h+E_h\), hvor G_h er den deterministiske
timegennemsnitsprognose. De fem egne historiske ugers targetdage replayes
fra den gemte weekdaymedianstate, som ved forecast, uden target-state-fit.
Baseline og C/k/Xi/g genbruges. Finite fejl e_i=Y_i−G_i danner den fælles lov
\(\widehat F_E(z)=n^{-1}\sum_i1\{e_i\le z\}\), med eksakte empiriske momenter:

\[
E[Y_h]=G_h+\bar e,\qquad
\operatorname{Var}(Y_h)=n^{-1}\sum_i(e_i-\bar e)^2.
\]

ddof=0 beskriver den diskrete lov med masser 1/n; det er ikke et unbiased
estimat af ukendt fysisk varians. Ved y=demand_min+demand_scale·y_norm
skaleres fejlmiddel med demand_scale og varians med demand_scale².
Pure G og biasjusteret middel bevares i særskilte kolonner.

Det centrale bånd er [G_h+q_(alpha/2),G_h+q_(1−alpha/2)], med inverse
empiriske CDF-kvantiler. Atomer/ties kan ændre faktisk central masse.
Båndscore er width+2/alpha·(lower−y)_++2/alpha·(y−upper)_+, på samme finite
rå måltemaske som MSE/RMSE/MAE/bias. Punktvis nominel 95% giver ikke en
garanti om fremtidig eller samtidig 168-timers dækning.

Alle fem uger indgår i parentfit: banken er in-sample, ikke holdout- eller
conformal-kalibrering. Iid er en enkel arbejdsmodel. Lag1-korrelation bruger
kun finite par i samme dag med præcis én fysisk time mellem labels;
missing timer/dagsresets forbindes ikke. Diagnosen er beskrivende.

WM4 har dF=Rdt, dR=(r_F F+r_U U+r_R R)dt+sigma dW og I=B+R.
Dens additive stateproces og tidsafhængige kovarians er forskellig fra denne
outputresidual og G's multiplicative stateproces. Sammenligning kræver ens
informationscutoff, timeoperator, population og scenarie.

Residualversionen har sigma_x=0. Replayfejl indeholder allerede samlet
uobserveret variation; ekstra procesvarians risikerer dobbelttælling.
To procesmomenter bestemmer heller ikke en konvolutionfordeling. Residualen
indgår ikke i fysisk energibalance eller priskontrol. Negative rågrænser
klippes ikke; deres antal rapporteres. Hour-start og fysisk enhed er
fortsat arbejdsantagelser i dette udviklingseksempel.

<a id="g-s3-uncertainty-guide-20261008--brug"></a>
<a id="g-s3-residual-guide-20261008--brug"></a>
## Implementering, snapshots og scoring

[Punktprocessen](../../src/method/bm_1_s3_uncertainty.py) har API
`diagnose_calibration`, `process_forecast`, `score_process_forecast`.
[Residualmetoden](../../src/method/bm_1_s3_residual.py) bruger et eget snapshot
bundet til parentens canonical-JSON-fingerprint og egne kildehashes.
Kilde-/parentændringer afvises uden skjult migration eller refit.

Den [nuværende CLI](../../src/exe/run_model/bm_1/s3.py) har `s3 point-process
diagnose/predict` og `s3 residual fit/predict`; brug variantens `--help` for
input/budget. Residualfit estimerer fejlloven med nul optimizerkald.
Punktprocessens sigma er et eksplicit scenarie, ikke et accepteret støjfit.
Gamle sourcebundne snapshots læses i [deres originale træ](../../archive/bm/ORIGINALS.md#original-tree).

Inputs følger S3-kalenderen: fysisk forårsuge har 167 timer, og gentagne naive
efterårslabels afvises. Ny outputmappe kræves. CSV/JSON gemmes før observations-
filen åbnes. Procesoutput tilføjer mean/variance, pointwise lower95/upper95
og state-momenter; rå varians skaleres med demand_scale². Residualoutput
bevarer prediction/baseline/point_prediction og tilføjer residual_mean,
residual_variance samt observed_mean/variance/lower/upper. `observed_*` er
modelleret prognose; målte `observed` tilføjes først ved scoring.

Fælles finite maske bruges til MSE/RMSE/MAE/bias, dækning, bredde og intervalscore.
For alpha=.05 er straffen 40 gange under-/overskridelse. Rå replay, fejlbank,
snapshots, CSV og figurer forbliver lokale. Scoring ændrer ikke modellen.

<a id="g-s3-uncertainty-guide-20261008--g-s3-grundversion-med-afgrænset-procesusikkerhed"></a>
<a id="g-s3-uncertainty-guide-20261008--faktisk-feb7-udviklingskørsel"></a>
<a id="g-s3-uncertainty-guide-20261008--slutstatus-og-senere-arbejde"></a>
<a id="g-s3-uncertainty-protocol-20261008"></a>
<a id="g-s3-uncertainty-protocol-20261008--afgrænset-s3-proceslag-protokol-før-historisk-kørsel"></a>
<a id="g-s3-uncertainty-protocol-20261008--kørsel-og-stopregler"></a>
<a id="g-s3-residual-guide-20261008--faktisk-udviklingsresultat"></a>
<a id="g-s3-residual-guide-20261008--holdested"></a>
<a id="g-s3-residual-protocol-20261008"></a>
<a id="g-s3-residual-protocol-20261008--g-afgrænset-empirisk-outputfejl-før-faktisk-kørsel"></a>
## Udviklingsevidens fra 8. oktober 2026

Punktprocessens kode/protokol frøs ved 1bba8e7. Feb7-eksemplet genbrugte
det oprindelige S3-fit uden nye optimizerkald; første FV-grid bestod.

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

Efter forsøget blev Demand præciseret som forbrug over en hel time.
Enhed og effekt/energi-konvention var ikke verificeret. Støtte- og dæknings-
afvisningen vedrører derfor denne eksplicitte punktapproksimation, ikke alle
intervalmodeller. Stokastisk intervaloutput kræver tidskovarians, som forklaret
i [timeoperatoren](G_INTERVAL_MODEL.md). Båndene var langt smallere end målespredningen;
en større sigma kan ikke flytte output udenfor frossen D([0,1]).

Residualforsøget frøs ved afa9af0 med samme Feb7-case, 720 historiske targets
og 168 forecasttimer. Der var én replay og én forecast/scoring, uden nyt
parameter-/baselinefit. Numerisk replay blev kontrolleret mod egen parent.

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

Middelværdien af kvadrerede supportafstande til [D(1,U,barB),D(0,U,barB)]
er en nedre grænse for MSE under den frosne models støtte, uanset statebane.
Grænsens kvadratrod var 0,180648; MSE-grænsen udgjorde 82,70% af replayets MSE.
Det identificerer ikke målerstøj og afviser ikke alle
andre modeller. Lag1≈.38 og 71,4% dækning viser, at iid-træningsloven ikke
gav veldækkende 95%-bånd; den lille biasjustering ændrede RMSE marginalt.
Der blev ikke opnået kalibreringsaccept, robust fysisk identifikation eller
kausal priseffekt. Realiseret pris/temperatur var betingede udviklingsinputs.

[Procestests](../../tests/test_bm1_s3_uncertainty.py) dækker blandt andet
variansfit, nulgrænse, konservativ diffusion, budgetstop, maske/skalering og DST.
[Residualtests](../../tests/test_bm1_s3_residual.py) dækker empiriske momenter/
kvantiler, scores, parentbinding, future-isolation og lagpar/supportafstand.
Den daværende samlede G-suite bestod 29 tests på 13,518s; det er dateret
evidens og dokumenterer ikke en ny beregning eller historisk accept.
