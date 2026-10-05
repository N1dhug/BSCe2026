# G4-afslutning: fast S3-sammenligning på nye uger 13–18

Gennemført 6. oktober 2026 efter Johans accept af den afgrænsede afslutning.
Den registrerede regel vælger **S3 uden straf (lambda=0)** på den nye blok.
Dette afslutter det afgrænsede G4-modelvalg for den aktuelle punktprotokol;
det er ikke historisk intervalaccept, fuld operationel ugeforecast-accept
eller afslutning af G6/G8/G7. G5's slutbenchmark er fortsat særskilt.

## Protokol og valg før scoring

Der køres kun S3: ét fælles theta=(C,k,Xi,a) over alle 35 dage i de foregående
fem kalenderuger. Seks mandagscutoffs: 28. marts, 4., 11., 18., 25. april og
2. maj 2022. Første vindue er uge 8–12→13, sidste er uge 13–17→18.
Alle syv dagsprognoser fryses mandag. Senere origins kan bruge tidligere
valideringsugers observerede demand, fordi den da er historie; der sker
ingen manuel ændring af modelopskriften mellem ugerne.

Før nye scores blev lambda låst til {0,0.001}, med pooled forecast-MSE som
primær metric, laveste MSE som valg og lambda=0 ved eksakt lighed. Ugevise
fejl, MAE/bias, identifikation og leave-one-week-out er diagnostik, ikke en
efterfølgende search efter nye lambdaer, datoer eller modeltyper. Den nye
blok bruges til G4-valget; den valgte models score er derfor ikke en
uafhængig sluttest. `registered_protocol.json` bevarer beslutningen før
scoring; `run/protocol.json` binder den faktiske kørsel til data og source.

S1 er ikke gentaget, fordi dens C,k,Xi fra hele 12-ugersblokken var
retrospektive. S2 er ikke gentaget, fordi separate ugedagsfits gav dårligere
samlet fejl og ofte svag identifikation. S3 bruger mere historisk information
per parametersæt og matcher ønsket om én fælles kurve på tværs af dagene.
Ingen fire-stykkers g eller fleksibel f er tilføjet: vi afslutter den valgte
familie, før flere frihedsgrader og en ny modeludvælgelse overvejes.

## Matematik og fit

Den eksisterende deterministiske BM-kernel genbruges:

\[
\dot x=\frac{D_{\theta,a}-B}{C},\quad
\delta=\tanh\left(\frac{k}{2}[1-2x+g_a(u)]\right),\quad
D_{\theta,a}=B+\Xi\delta\begin{cases}1-B,&\delta\ge0,\\B,&\delta<0.\end{cases}
\]

\[
g_a(u)=1-2u+a\,h(u),\qquad
h(u)=\begin{cases}2u,&u\le1/2,\\2(1-u),&u>1/2,\end{cases}
\quad -1\le a\le1.
\]

Dermed er g kontinuerlig og ikke-voksende med endpoints 1 og −1.
a=g(1/2), og a=0 giver den lineære g. f(x)=1−2x holdes fast.

For de målte træningstimer M_w efter warmup minimeres

\[
J_{\lambda,w}(C,k,\Xi,a)=\frac1{N_w}\sum_{(d,h)\in M_w}
\bigl[D_{C,k,\Xi,a,d}(h)-y_d(h)\bigr]^2+\lambda a^2.
\]

Alle modelresidualer divideres med sqrt(N_w), og sqrt(lambda)*a tilføjes
som sidste LS-residual. Normalisering med N gør strafstyrken sammenlignelig
ved forskellige antal målte træningstimer. Forecastfejlen indeholder ingen
straf; den sammenligner alene forudsagt og observeret demand.

\[
\int_0^1[g_a(u)-g_0(u)]^2\,du=\frac{a^2}{3}.
\]

Straffen er derfor 3*lambda gange kurvens kvadratiske afstand til lineær
respons. Det er en bias–varians-afvejning, ikke en fysisk antagelse om, at
knuden skal være nul. Den lineære reference har stadig prisrespons.
En lavere regulariseret TRAIN-score er ikke evidens for bedre predictions.

Bounds og optimizer er uændrede: C∈[1,744] timer, k∈[0.05,10], Xi∈[0,1],
a∈[−1,1]. C og k fittes i log-koordinater. SciPy `least_squares` med TRF
understøtter netop disse box-bounds; modellen er nonlineær i parametrene,
så almindelig lineær LS kan ikke erstatte dette fit. To faste starter og
numerisk Jacobian er en billig, afgrænset løsning; en stor global search
ville være en anden og dyrere undersøgelse. Konvergens fra to starter
garanterer ikke globalt optimum.

Starterne er det oprindelige før-første-origin januarfit med a=0 og den
faste seed C=12,k=2,Xi=0.5,a=0.25. Den oprindelige seeds træningsresidualer
tilføjes ikke. Ingen seed eller bound tilpasses til de nye target-scores.

## Historiske forløb og rettelsen omkring sommertid

Dagene er uafhængige ODE-forløb. Hver starter neutralt x=0.5 præcis 24
fysiske timer før den historiske dags lokale midnat; warmup scorer ikke.
En dags forecaststate er medianen af de fem matchende historiske x-værdier
ved mål-dagens start, med kandidatens egne parametre. S3's parametersæt
er fælles, men de syv dagsstarter er separate historiske approksimationer.
Det er ikke én sammenhængende 168-timers statebane eller target-oracle-fit.

Kodeaudit fandt, at start ved »foregående lokale midnat« giver kun 23
warmup-timer for 28. marts. Dette er rettet til lokal midnat minus 24
fysiske timer: start er 26. marts kl. 23, og state efter 24 timer er
28. marts kl. 00. For 27. marts er warmup stadig 24 timer og targetdagen
23 timer. Denne rettelse ændrer ingen af de tidligere ugers 6–12-trajektorier
(der sluttede 27. marts), og ingen af deres fits er genkørt eller ommærket.

Foregående fem ugers faktiske mål-dage indeholder sommertidsdagen i de
første fem origins: 839 målte træningstimer. Sidste origin har 840.
Alle nye forecastuger har 168 fysiske timer, i alt 1008 målte timer.

## Fælles baseline, skala og data

B er samme eksisterende WM4 `statistical_output`, estimeret på alle målte
historiske demand/temperaturtimer før mandag; baselinehistorikken er
expanding, mens BM-parametrene bruger fem uger. »Uden straf« betyder uden
lambda*a², ikke uden baselinens eksisterende ridge/Huber-procedure.
Baseline evalueres på et fast 60-punkter/time-grid, også når ODE-griddet
forfines. Pris holdes per time, temperatur interpoleres mellem timepunkter
med sidste temperatur holdt ved endpoint.

Demandminimum 0.4023076923 og spænd 2.44 fra den oprindelige femugersperiode
er frosne. Prisen bruger det allerede deklarerede domæne [−500,6000] DKK/MWh,
u=(price+500)/6500. De nye priser ligger i [−18.52,2431.25], så domænet er
ikke ændret. Alle nye u ligger under 0.5: den nye blok tester derfor kun
venstre del af g i forecast, og giver ikke evidens for højprisgrenen.

Git-blobs for `input.xlsx`/`prices.xlsx` er verificeret identiske med den
første undersøgelse. Jan.–27. marts råværdier og DMI-temperaturer genbruges
uændret. Kun 28. marts–8. maj tilføjes. Originalt DMI countryValue/mean_temp-
API-svar er gemt; 1008 faktiske temperaturtimer, ingen fremtidig imputation.
Demand, pris og temperatur er komplette for de samme 1008 forecasttimer.
Observationer uden for den frosne demand-skala klippes ikke.

Realiserede priser og temperaturer giver **betingede demandprognoser**.
En operationel ugeprognose kræver inputs kendt ved origin. Historisk
intervalmiddel/timeenergi og fast-populationskontrakt er stadig særskilte
opgaver; disse gamle punktdata får ikke en ny fysisk fortolkning.

## Resultater

Fejlmål i rå workbook-enheder; MSE i kvadrerede enheder:

| Model | MSE | RMSE | MAE | Bias |
|---|---:|---:|---:|---:|
| O | 0.1806157175 | 0.4249890793 | 0.3288992822 | 0.0770548938 |
| s3_without_penalty | 0.1798347601 | 0.4240692870 | 0.3287236282 | 0.0902463450 |
| s3_with_penalty | 0.1843188395 | 0.4293237001 | 0.3327553895 | 0.0854222383 |

| Testuge | Baseline-MSE | S3 uden straf | S3 med straf |
|---|---:|---:|---:|
| 13 | 0.20034363 | 0.19497083 | 0.20223000 |
| 14 | 0.18587670 | 0.18342649 | 0.18587670 |
| 15 | 0.20381300 | 0.21284063 | 0.21171538 |
| 16 | 0.18565873 | 0.18258364 | 0.19756918 |
| 17 | 0.16093575 | 0.15319399 | 0.16079334 |
| 18 | 0.14706648 | 0.15199299 | 0.14772844 |

Straffens samlede MSE-reduktion er -2.4934 %. Den vinder 2/6 uger
og 12/42 dage. Fortegn i følsomhedsanalysen: uden uge 13: -2.2222 %; uden uge 14: -2.7305 %; uden uge 15: -3.2361 %; uden uge 16: -1.3296 %; uden uge 17: -2.0852 %; uden uge 18: -3.3623 %.
Positive procenter betyder lavere MSE med straf. Følsomhedsanalysen bruger
de gemte forecasts uden nye fits; ingen uge fjernes fra hovedresultatet.
Overlappende træningsvinduer giver ikke seks uafhængige gentagelser, så
der påstås ingen iid-uge-signifikans eller generel helårsgevinst.

Den registrerede regel vælger lambda=0. Straf øger MSE med 2.4934 %,
mens uden straf forbedrer baseline med 0.4324 %. Ustraffet MAE og RMSE
er også lavere end straffet, men den positive bias er større end baseline.
Alle seks leave-one-week-out-sammenligninger beholder samme fortegn;
det er et mere stabilt resultat end den tidligere strafgevinst på uge 6–12.
Det beviser ikke, at straf aldrig hjælper. Det giver derimod ikke grundlag
for at beholde lambda=0.001 i den aktuelle forecastingopskrift.

207/1008 targets (20.54 %) ligger under det frosne demandminimum. Minimum
er −0.6715385 i rå enheder, normaliseret −0.44010; 107 observationer er
negative i rå enheder. BM/baseline med output i [0,1] kan derfor ikke fange
alle disse observationer. De negative og lave værdier er beholdt i scores;
skalaen er ikke ændret efter resultatet. Dette er en vigtig begrænsning for
målet om at ramme hele demandprofilen.

I de tre sidste ustraffede fits er a≈1 og k≈10 (bounds). Da alle nye
forecastpriser har u<0.5, er g_a(u)≈1 og g'_a(u)=2(a−1)≈0 i det område.
Disse forecasts er dermed næsten ufølsomme over for variation i den nye
prisprofil; en eventuel gevinst kan komme fra state-/baselinekorrektionen.
Lavere forecast-MSE er ikke bevis for identificeret kausal priselasticitet
eller en valideret historisk prisstyringsmodel. Fremtidig G7 må skelne en
syntetisk metodekontrol fra historiske claims om optimale priser/profit.

| Uge | lambda | C [timer] | k | Xi | a | lokal data-rang |
|---|---:|---:|---:|---:|---:|---:|
| 13 | 0.0 | 744 | 9.9885 | 0.035174 | -0.82326 | 4 |
| 13 | 0.001 | 744 | 0.063605 | 0.30471 | -0.067996 | 4 |
| 14 | 0.0 | 12.993 | 0.10841 | 0.11829 | -0.72981 | 4 |
| 14 | 0.001 | 1 | 0.05 | 5.8603e-15 | -1.6666e-08 | 1 |
| 15 | 0.0 | 18.909 | 0.41417 | 0.37702 | -0.75476 | 4 |
| 15 | 0.001 | 2.4692 | 0.28622 | 0.29553 | -0.15589 | 4 |
| 16 | 0.0 | 12.538 | 10 | 0.40705 | 1 | 4 |
| 16 | 0.001 | 3.7593 | 0.75002 | 0.32172 | -0.039971 | 4 |
| 17 | 0.0 | 27.305 | 10 | 1 | 1 | 4 |
| 17 | 0.001 | 1.0003 | 0.27576 | 0.24301 | -0.0062655 | 4 |
| 18 | 0.0 | 26.702 | 10 | 1 | 1 | 4 |
| 18 | 0.001 | 1 | 0.50024 | 0.18039 | -0.0076487 | 4 |

Data-rang beregnes fra Jacobian uden strafrækken; fuld lokal rang betyder
ikke nødvendigvis stærk praktisk identifikation. Bounds og store ændringer
i C/k/Xi/a begrænser fysisk fortolkning og ekstrapolation. Det bedste
forecast er ikke automatisk recovery af de sande fysiske parametre.

## Numerik, kontrol og faktisk runtime

Alle 12 valgte kandidater har konvergeret fit og består samme
max-demand-tidsforfiningskrav ≤1e-5. Værste normaliserede forskel er
8.9932885e-06. Final ODE-grids er [12, 24] trin/time, kontrolleret ved det
dobbelte. Uge 17 og 18 krævede genfit på 24 efter negativ 12/24-kontrol;
tolerancer, families og budgetter er uændrede. Alle optimizerforsøg fra
begge grids er bevaret. Den observerede negative 12-trinskontrol for uge
18 er også kopieret før latest-numerics-filen blev erstattet.

32/32 optimizerstarter konvergerede, inklusive starter
ved numerisk genfit. Højeste parrede opgave brugte 1377 residualkald
mod 3000; Jacobianprober tæller med. Samlet parallel kørselsinvocation tog
297.03 s; summeret optimizer-walltid over workers var 733.80 s.
Otte workers, BLAS=1. Disse tidsmål skal ikke blandes sammen.

To fokuserede testmetoder består på 0.163 s: eksisterende DST-test udvidet
til dagen efter sommertid og én ny kontrol af 35 fælles historiske forløb,
839 målte timer og fravær af fremtidig demand i jobs. AST-syntakskontrol af
runner, ændret scenariointerface og testfil består. Ingen fuld discovery,
ny WM-dynamikfit eller genfit af de første 12 uger er kørt.

Runneren genbruger `calendar_inputs`, `checked_task`, `BM1PooledInputs`,
`fit_strategy`, eksisterende responsegeometri og RK4. Ingen ny solver.
Der er ingen ændring i F-kode eller F4-H2-status. G6a/G6b/G8/G7 er endnu
ikke implementeret af denne leverance.

## Reproduktion og næste trin

Benchmark: `files/output/bm_1/benchmarks/g4_validation_weeks_13_18_20261006.json`.
Råarkiv: `G4_validering_uge13_18_raaresultater_20261006.zip`, med inputaudit,
registrering, alle fits, numerik, gemte forecasts, scores, analyse og figur.

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python -B -m src.exe.run_model.bm_1.deterministic_validation `
  --raw-csv PATH_TO_ARCHIVE/raw_inputs.csv `
  --reference-fit PATH_TO_ARCHIVE/reference/fit.json `
  --output files/output/bm_1/runs/deterministic/g4_validation_new `
  --workers 8
```

`--resume` kontrollerer numerisk sourcehash, inputfingerprint, seed og
protokol og genbruger komplette opgaver. Eksisterende negative stop er
evidens, ikke konvergens. Ingen lokal kørsel kræves for denne leverance.

Næste trin er en afgrænset G6a-variansreference på en frosset strukturel
model med egne støjparametre og diagnostik. Den efterfølgende G6b→G8→G7-
kæde består. G5's samlede benchmark/slutrapport kommer med klart skel
mellem disse udviklings-/valideringsdata og en eventuel uafhængig sluttest.
