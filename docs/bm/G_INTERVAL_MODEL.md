# G: timeoperator, baseline og integrerede procesmomenter

G's intervalmodel beskriver gennemsnitligt forbrug over en fysisk time.
Den har egen fit-/snapshotidentitet; punktprognosens parametre og bånd kan ikke
genmærkes som intervalresultater. Proceslaget giver intervalmiddel og varians,
men ingen intervalkvantiler. Demand-enhed og hour-start er arbejdsantagelser.

Den gemte aktuelle G-model bruges gennem [BM-guiden](../../root/BM_GUIDE.md).
Dette dokument forklarer den særskilte S3-intervalmetode og dens daterede
udviklingsevidens. Samlet accept står i [sammenligningen](COMPARISON.md).
De fulde oprindelige protokoller findes i [originalarkivet](../../archive/bm/ORIGINALS.md#original-tree).

<a id="g-s3-interval-guide-20261008"></a>
<a id="g-s3-interval-guide-20261008--observation-og-baseline"></a>
## Observation, baseline og diskretisering

G's eksisterende lov har tid i timer og state i [0,1]:

\[
dX_t=(D(X_t,U_t,B_t)-B_t)dt/C+\sigma_xX_t(1-X_t)dW_t,
\qquad Y_h=\Delta^{-1}\int_h^{h+\Delta}D_tdt,\quad\Delta=1\text{ time}.
\]

S3 bruger lineær f, monoton to-stykkers g og lambda=0. U holdes gennem timen;
B interpoleres på et fast 60-sample/h-net. Den deterministiske timeoperator er
\(\bar D_h=\bar B_h+C(X_{h+1}-X_h)/\Delta\), genbrugt i både fit og forecast.
Den er forskellig fra endpoint-D og værdien ved timens begyndelse. Det pooled
S3-fit har 35 egne historiske dagsforløb, 24 timers input-only warm-up, egne
syv medianstarts og mandagsfrossen origin. Det senere separate ugedagsfit er
beskrevet i [ugedagsstudiet](G_WEEKDAY_STUDY.md).

Baseline genbruger WM's featurefamilie, ridge/Huber og standardisering, men
designrækkerne følger samme observationsoperator:

\[
\bar\phi_h=\Delta^{-1}\int_h^{h+\Delta}\phi(t,T_t)dt.
\]

Lineær standardisering kommuterer med timegennemsnittet. Regressionen bruger
uklippede features; baselinefortsættelsens clipping til [0,1] er et erklæret
modelvalg og kan give afvigelse fra det lineære design. Baseline er en
statistisk inputmodel. Historisk temperatur interpoleres kun fra kendte
inputs; sidste temperatur holdes ved cutoff. Forecast bruger det erklærede
scenarie, inklusive næste times temperatur inden for scenariet og hold ved slut.

Minute-tidsstempler bygges med heltalsminutter/nanosekunder. Det undgår, at
float-hour-afrunding rammer den forkerte kalenderfeature. Den uafhængige
harmoniske integraltest bruger trapezoidgrænsen \(\omega^2/(12\cdot60^2)\).
DST følger fysisk tid og lokale kalenderfeatures.

<a id="g-s3-interval-guide-20261008--procesmiddel-og-varians-gennem-timen"></a>
## Integrerede procesmomenter

I SDE gælder
\(\int Ddt=\int Bdt+C\Delta X-C\int b(X)dW\).
Den deterministiske endpointidentitet kan derfor ikke bruges som tilfældigt
intervaloutput. Med \(Z=\int(D-B)dt\), FV-generator Q og reward c=D−B bæres:

\[
m'=Qm,\qquad r'=Qr+\operatorname{diag}(c)m,\qquad
s'=Qs+2\operatorname{diag}(c)r.
\]

Her er m state-masser, r=E[Z·1_state] og s=E[Z²·1_state]. Kun rewardfelterne
nulstilles ved ny time; state bæres videre i hver egen dag. For \(\Delta=1\)
er middel \(\bar B+\sum r\) og varians \(\sum s-(\sum r)^2\). Dermed bevares
tidskovariansen, som et gennemsnit af marginale punktvarianser ville miste.
Forventet energibalance giver stadig
\(E[\bar D]=\bar B+C(E[X_{h+1}]-E[X_h])/\Delta\), fordi det bounded
diffusionsleds Itô-integral har middel nul; identiteten bestemmer ikke variansen.

Numerikken bruger konservativ zero-flux FV og LAPACK-tridiagonal implicit
Euler, samme transition for alle tre felter, halvt reward før/efter og
midtpunktfrosne koefficienter. Den er førsteordens i tid. Signed r er ikke
negativ sandsynlighed. Masser og varians klippes eller renormaliseres ikke.
Baselineintegralet tilføjes fra det fine grid for at mindske subtraktionsfejl;
Dirac-startens cellefejl rapporteres særskilt. Små positive sigma kan kræve
finere grids på grund af upwind-spredning; sigma=0 bruger egen interval-RK4.

De historiske udviklingsgates var normaliseret demand/state-meanfejl≤0,005,
variansfejl≤0,002 og masserror≤1e−10, med finitte felter og ikke-negativ varians.
Coarse/fine og energibalancekontrollen er numeriske checks, ikke fejlbeviser.
To momenter bestemmer ikke intervalfordelingen: punktbånd, Gauss-bånd og
intervaldækning følger ikke heraf. Parameter-, baseline-, input- og
observations/modelfejl er udeladt.

<a id="g-s3-interval-guide-20261008--den-særskilte-kalibreringsdiagnose"></a>
## Processtøjdiagnose og fysisk støtte

Den lokale diagnose lineariserer stateafvigelse eta og timeintegral z langs
den deterministiske bane:
\(d\eta=a\eta dt+b_0dW,\ dz=c\eta dt\), hvor a=D_x/C, c=D_x og b0=x(1−x).
Den bruger eksakt frossen 2×2-transition og positiv tre-node Gauss-kvadratur
for støjkovariansen. Kun z og dets cross-covariance nulstilles hver time;
statevarians fortsætter. P0=0 efter warm-up; warm-up-støj er udeladt.

Matching-weekday samplevarianser bruger ddof=1 og mindst to målinger/bin.
Én q≥0 tilpasses via varians-LS til enhedsprofiler; første hele time indgår,
fordi integreret støj har positiv varians trods P0=0. Historisk meanforfining
blev kontrolleret mod 1e−5, unit- og fitted profil-std mod 1e−4. Raw interday
dispersion identificerer ikke fysisk processtøj. P>1/4, støttebrud eller
numerisk afvisning forhindrer kalibreringsaccept; q bliver ikke automatisk sigma.

Med lineær f og held U er intervalstøtten netop
[D(1,U,bar B),D(0,U,bar B)], fordi endpoint-D er affin i B. Enhver statebane
i [0,1] ligger inden for disse intervalgrænser. Større sigma kan ikke forklare
en observation udenfor under samme parametre, baseline og støjfri observation.

<a id="g-s3-interval-guide-20261008--brug-og-kontrakt"></a>
## Implementering og datakontrakt

Se [fit/snapshot/predict](../../src/method/bm_1_s3_interval.py),
[baselinefeatures](../../src/method/bm_1_s3_interval_baseline.py),
[procesmomenter/diagnose](../../src/method/bm_1_s3_interval_process.py) og
[den nuværende S3-indgang](../../src/exe/run_model/bm_1/s3.py).
CLI-formerne er `s3 interval fit`, `s3 interval predict` og `s3 interval diagnose`;
variantens `--help` beskriver eksplicit budget og inputvalg. Point-, interval-
og residual-snapshots har særskilte kontrakter.

Fitinput er Datetime,Demand,Price,Temp; forecastinput Datetime,Price,Temp.
Price/Temp er intervalstart. Demand-label vælges eksplicit som hour_start eller
hour_end; endlabels flyttes én fysisk time. Ved cutoff kan forrige komplette
time med endlabel bruges, uden at fremtidig Price/Temp føres ind i fit.
Naive lokale labels skal være unikke/sorterede; tvetydige efterårstimer afvises.
Forårsugen har 167 fysiske timer med entydige UTC-start/end.

Snapshot binder baseline, skala, C/k/Xi/x0, g, starts, maske, egne grids og
numeriske kildehashes. Ændrede kilder kræver eget fit eller eksplicit reviewet
migration. Gamle snapshots kan genskabes med [deres originale kildetræ](../../archive/bm/ORIGINALS.md#original-tree).
CSV skelner hourly-average `prediction`, integreret `baseline`,
`point_prediction`, state start/end og UTC-intervaller. Procesoutput tilføjer
`process_mean`, `process_variance` og state-momenter; rå varians skaleres med
demand_scale². Forecast gemmes før observationsfilen åbnes. Scoring bruger
fælles finite measured maske og rå MSE/RMSE/MAE/bias. Afvist diagnose/fejl giver
CLI-returkode 2. Rå inputs, snapshots og tidsserier er lokale.

<a id="g-s3-interval-guide-20261008--g-s3-timegennemsnit-med-integreret-proceskovarians"></a>
<a id="g-s3-interval-guide-20261008--faktiske-udviklingsresultater"></a>
<a id="g-s3-interval-guide-20261008--kontroller-og-næste-holdested"></a>
<a id="g-s3-interval-protocol-20261008"></a>
<a id="g-s3-interval-protocol-20261008--g-s3-timegennemsnit-protokol-før-faktisk-fit-og-proceskørsel"></a>
<a id="g-s3-interval-protocol-20261008--fast-kørsel"></a>
<a id="g-s3-interval-protocol-20261008--nødvendig-baselinekorrektion-efter-første-kørsel"></a>
## Udviklingsevidens fra 8. oktober 2026

Det korrigerede eksperiment havde origin 7. februar 2022, 168 målte forecasttimer
og 720/840 historiske targettimer. Populationen var MED-gennemsnit over de
tilgængelige observerede målere. Realiseret pris/temperatur var betingede inputs;
det var hverken blind vejrtest eller kausal prisstyringstest. Kode/protokol
frøs ved b2a6776 før slutkørslerne.

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

C=1,361809h, k=.101995, Xi=.374275, historisk x0=.5 og g(.5)≈−1. Prisrespons tæt ved
randen og en svag Jacobianretning dokumenterer ikke robust identifikation eller
kausal effekt. Baselineforbedringen var lille; processtøj alene kunne ikke
forklare den målte variation. Energibalancens diskretiseringsafvigelse skyldes
FV face-flux versus cellecentreret reward og er ikke et matematisk fejlbevis.

Det tidligere mellemtrin ved 5189844 brugte punktfittede baselinefeatures og
gav RMSE .386778. Integrering bagefter matchede ikke det tidsintegrerede
regressionsdesign og gav fasefejl. Det korrigerede design blev valgt af
observationskonsistens, ikke forecastscore. De to fits brugte samlet
33,455508s/226 residualkald inden for den oprindelige 120s/1000-kaldsramme;
første mellemtrin og begge proces-/diagnoseinvocations er bevaret. Dette er
historisk forbrug, ikke et nyt kørselsbudget. Afløste ordrer ligger i originalarkivet.

[Intervaltests](../../tests/test_bm1_s3_interval.py) dækker blandt andet
harmonisk baselineintegral, snapshotroundtrip, punkt/mean-adskillelse,
fremtidsuafhængighed, cutoff/DST, integreret Brownsk kovarians, Van Loan-reference,
telegraph-integral, konservativ FV og nulgrænser. Den daværende samlede G-kontrol
bestod 25 tests på 10,356s; det er dateret evidens, ikke en ny testkørsel.
