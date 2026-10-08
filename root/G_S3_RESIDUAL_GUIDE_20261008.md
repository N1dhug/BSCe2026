# G: særskilt empirisk residuallag

Johan godkendte8.oktober et enkelt, tydeligt mærket statistisk udviklingslag.
G's deterministiske intervalmodel, proces-scenarievej og fysisk styring bevares.
F/bank/BM2 ændres ikke. Se [fast protokol](G_S3_RESIDUAL_PROTOCOL_20261008.md),
[metode](../src/method/bm_1_s3_residual.py) og
[kørbar indgang](../src/exe/run_model/bm_1/s3_residual.py).

## Matematik og fortolkning

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

## Brug

```text
python -B -m src.exe.run_model.bm_1.s3_residual fit --snapshot INTERVAL_FIT/interval_snapshot.json --raw-csv HISTORY.csv --seconds 30 --output NEW_RESIDUAL_FIT
python -B -m src.exe.run_model.bm_1.s3_residual predict --snapshot INTERVAL_FIT/interval_snapshot.json --residual-snapshot NEW_RESIDUAL_FIT/residual_snapshot.json --inputs-csv FUTURE.csv --observations-csv OBSERVED.csv --output NEW_RESIDUAL_FORECAST
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

## Faktisk udviklingsresultat

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

## Holdested

Den kørbare grundversion, timeoperator, processcenarie og det godkendte
residuallag er implementeret. Fysisk/observeret kalibreret usikkerhed er
fortsat åben. Stop modeludvidelser og sigma-/bank-/AR-søgning i dette pas.
Næste nyttige trin er én afgrænset G/WM-sammenligning med samme informations-
og observationsgrundlag, eller en senere tidsblok til uafhængig residual-
validering. WM's eksisterende punktvej må ikke omtales som en allerede
matchende timegennemsnitsbenchmark. Denne G-chat på Sol High er passende.
