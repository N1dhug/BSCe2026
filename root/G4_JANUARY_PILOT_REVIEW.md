# G4 — januar-pilot og frosset februarprognose, 5. oktober 2026

## Resultat og beslutning

**Piloten er gennemført. Lineær BM beholdes som G's aktuelle udviklingsmodel.**
Begge to-stykkers g-kandidater konvergerer og består deres egen numeriske
kontrol, men den bedste kandidat forbedrer februarugens MSE med kun
0.0150409 % mod lineær BM. Det er klart under den fastlagte 1 %-regel.
Baseline har stadig den laveste MSE.

Dette afslutter det lille G4-januarforsøg og dets udviklingsvalg, ikke
historisk/videnskabelig validering. Det fravælger ikke fleksible kurver i
et senere fælles fit på længere historik og ændrer ikke F's arbejde.
Ingen ny optimering er udført af original lineær BM, baseline eller WM.

Maskinlæsbar evidens:
[benchmark JSON](../files/output/bm_1/benchmarks/g4_january_pilot_20261005.json).
Den separate resultatpakke indeholder komplette kandidatarkiver, CSV'er,
figurer, inputaudit, kilderevisioner, originalsikring og metodekørslen.

## Data og præcis informationsgrænse

| Felt | Værdi |
|---|---|
| TRAIN | 1. januar 2022 kl. 00 til 1. februar kl. 00; 744 timer |
| Målte TRAIN-timer | 624; 120 manglende demand-timer |
| DEVELOPMENT | 1.–7. februar 2022; 168/168 målte timer |
| Population/kontrakt | Det originale MED-aggregat og legacy-hourly-point-approximation |
| Faste inputs | Original baseline, skalaer, held timepris og initial-state-policy |
| Forecast | Kendt historisk pris/temperatur; betinget ugeprognose |
| Teststatus | Allerede eksponeret udviklingsuge; ingen blind sluttest |

De manglende TRAIN-observationer bidrager ikke til LS eller scorer.
Statebanen propagerer over alle timer med de originale modelinputs.
Ingen demand udfyldes som en ny måling.

Februar-CSV'erne indeholdt ikke temperatur. Den blev hentet fra samme
DMI countryValue/mean_temp-hour-kilde og behandlet som i den eksisterende
importer. Den samlede pris-/temperaturfingerprint matcher præcist G3:

ceace32558eea579fea4edb1badd9ea6b89b4e9d7c1916aea3cf79c17b294055

Genskabt baseline, lineær BM, WM og deres states matcher den uploadede
G3-prognose med maksimal forskel 2.78e-16 i normaliserede værdier.
Der er derfor ikke indført et nyt baselineforløb eller andre februarinputs.
Forecasts blev skrevet før observationsfilen blev læst til scoring.

## Matematik og hvorfor den er enkel

BM's eksisterende deterministiske ligninger anvendes uændret:

\[
\delta(t)=\tanh\!\left(\frac{k}{2}[f(x(t))+g(u(t))]\right),
\]

\[
D(t)=B(t)+\Xi\delta(t)
\begin{cases}
1-B(t),&\delta(t)\ge0,\\
B(t),&\delta(t)<0,
\end{cases}
\qquad
\dot x(t)=\frac{D(t)-B(t)}{C_{\rm hours}}.
\]

Her er x, u, B og D normaliserede størrelser, og t måles i timer.
Enheder/aggregat og observationernes intervalbetydning er endnu ikke
endeligt afklaret; C er derfor ikke dokumenteret fysisk kapacitet.

Vi beholder f(x)=1-2x og én fælles prisrespons med én ekstra parameter:

\[
g_a(u)=
\begin{cases}
1+2(a-1)u,&0\le u\le 1/2,\\
a-2(1+a)(u-1/2),&1/2<u\le1,
\end{cases}
\qquad -1\le a\le1.
\]

Begge udtryk giver a i u=1/2, så kurven er kontinuert.
Hældningerne 2(a-1) og -2(1+a) er ikke-positive.
Endpoints er g(0)=1 og g(1)=-1; a=0 giver præcis lineær g.
Ved a=-1 bliver det øvre prisinterval fladt. Monotonicitet betyder her
ikke-voksende, ikke nødvendigvis strengt faldende.

Vi estimerer (C_hours,Xi,a) fælles over de målte januartimer.
k=0.22688550696235388 og x0=0.5 holdes faste, ligesom baseline og skalaer.
Kurven deles af alle dage; der er ingen daglige resets eller dagskurvefit.

For hver fast strafstyrke lambda løses

\[
\min_\theta
J_\lambda(\theta)
=\frac1{624}\sum_{i\in{\cal T}_{\rm obs}}
[\widehat D_i(\theta)-y_i]^2+\lambda a^2.
\]

Residualerne er datafejl/sqrt(624) plus én strafresidual sqrt(lambda)*a.
Dermed minimerer SciPy halvdelen af J_lambda, hvilket giver samme optimum.
Datascore og straf gemmes særskilt. Straf mod a=0 favoriserer den enklere,
lineære kurve; den er et deklareret udviklingsvalg, ikke en fysisk prior.

Det er nonlinear LS, selv om g har rette delstykker: både tanh og den
simulerede tilstand afhænger af parametrene.
Vi bruger scipy.optimize.least_squares/TRF med bounds, log-C,
two-point-Jacobian, diff_step=1e-4 og x_scale='jac'.
TRF giver et lokalt fit, ikke en garanti for global optimalitet.
C ligger i [1,744] timer, Xi i [0,1] og a i [-1,1].
Disse er de eksisterende udviklingsbounds.

To forudfastlagte lambdaer, 0 og 0.001, og to starter, a=0 og 0.25,
var tilstrækkelige. Ingen ekstra søgning, større grid eller udvidelse til f
blev startet efter resultaterne.
Data-Jacobianens singularværdier beregnes uden strafrækken, så
regularisering ikke tæller som observationsinformation.

## Kodevej og versionsforsvar

| Fil/funktion | Opgave |
|---|---|
| src/tools/bm_1_deterministic_snapshot.py | Validerer gemt fit, baseline og kildehashes uden refit |
| src/tools/bm_1_deterministic_response.py: read_training_archive | Kontrollerer TRAIN-grid, masker, original score og præcis cutoff-state |
| src/params/bm_1_response.py | Validerer fælles kurveknuder, endpoints og monotonicitet |
| src/method/bm_1_deterministic_response.py: fit_response | Afgrænset fælles LS med tælling af alle residual-/Jacobiankald |
| BM1ResponseInputs.solve | BM-RK4 med held pris, original fin baseline og uden state-clipping |
| fit_training_archive | Eksporterer kun konvergerede kandidater med bestået tidsforfining |
| forecast_response | Genbruger original lineær BM/O/WM; propagerer G4 fra sin egen cutoff-state |
| score_response_forecast | Samme målte maske og fejlmål for alle fire modeller |

G4-adapterne og øvrige dependencies er hentet fra main
14ece7c0d3131d90dcddc2708cca65b96bb3c5b8.
De otte G3-forwardfiler er versionslåst til 607dcd7 i en isoleret
kørselskopi. Alle otte SHA256-hashes matcher det uploadede fit_snapshot.
Det bevarer det originale fits numeriske identitet, mens nyere WM-støjkode
kan udvikles på main. Ingen source-check, tolerance eller state-guard er
omgået, og main/F/WM-kode er ikke overskrevet.

En G4-kandidat fra denne pakke kan derfor ikke blot indlæses på en nyere,
inkompatibel WM-kildeversion. Brug den dokumenterede source-map ved replay;
en fremtidig migration kræver konkret review, ikke omskrivning af hashes.

60 BM-trin/time og det originale 60-trins baselinegrid er bevaret.
120 BM-trin/time kontrollerer kandidater og forecasts.
Største observerede demandforskel er 2.22e-16 mod tolerance 1e-5.
Denne særlige høje overensstemmelse dokumenterer ikke generel maskinpræcision:
baselineknæk, få frie parametre og den svage respons gør netop dette case let.

Der er ikke kørt en ekstra generel testpakke. Selve arkivreplay,
fit, kandidatforfining, forecastforfining og referenceparitet dækker
den aktuelle forsøgsopgave. Ingen solver-/estimatorimplementation er ændret.

## Kandidater og kontrol

| Model | C_hours | Xi | a=g(0.5) | TRAIN-MSE, normaliseret |
|---|---:|---:|---:|---:|
| Original lineær BM | 1.6349238174 | 0.2312064792 | 0 | 0.006605684571 |
| G4, lambda=0 | 1.4579146989 | 0.1906205581 | -0.9999999997 | 0.006591583435 |
| G4, lambda=0.001 | 1.6479898840 | 0.2306417855 | -0.0135459969 | 0.006605311148 |

Alle fire starter konvergerede. Begge starter ved lambda=0 landede tæt på
nedre a-bound; de regulariserede starter landede tæt på lineær kurve.
Valgte data-Jacobianer har lokal rang 3 i koordinaterne (log-C,Xi,a).
Singularværdierne er henholdsvis
[0.0327333,0.00245179,0.00169356] og
[0.0223250,0.00359447,0.00166340].
Lokal rang er ikke global identifikation, parameterusikkerhed eller kausalitet.
Den ustraffede kandidats aktive a-bound kræver særlig forsigtighed.

Deres egne forecast-startstates er henholdsvis
x=0.43975622195 og x=0.68306455369; lineær BM har x=0.68593118411.
Forskellige states er forventelige ved forskellige g-kurver.
Vi har ikke genbrugt lineær BM's state i en fleksibel kandidat.

## Ugeprognose og modelvalg

Alle scorer herunder er pooled over præcis samme 168 målte timer og
vises i de originale workbook-enheder:

| Model | MSE | RMSE | MAE | Bias |
|---|---:|---:|---:|---:|
| Baseline O | 0.06934621735 | 0.26333669959 | 0.20998114942 | 0.09767430577 |
| Lineær BM | 0.06951591504 | 0.26365870939 | 0.20888953028 | 0.09952142665 |
| G4, lambda=0 | 0.06950545924 | 0.26363888037 | 0.20832059533 | 0.09970477391 |
| G4, lambda=0.001 | 0.06951595226 | 0.26365877997 | 0.20887847793 | 0.09952258218 |
| WM4 | 0.07099789830 | 0.26645430809 | 0.21142737270 | 0.09888989397 |

Den låste beslutningsregel er at foretrække lineær BM, hvis

\[
\frac{{\rm MSE}_{\rm linear}}{{\rm MSE}_{\rm best\ checked}}-1\le0.01.
\]

Her er venstresiden 0.0001504313, svarende til 0.0150431 %.
Lineær BM vælges derfor; dens fuldpræcisionsparametre og cutoff-state
genbruges fra originalarkivet. Der er ikke udført et nyt lineært fit.

Den ustraffede kandidat forbedrer TRAIN-MSE med 0.21347 %,
men uge-MSE med kun 0.01504 %. MAE falder lidt, mens positiv bias stiger.
Den regulariserede kandidat giver næsten samme prognose som lineær BM.
Den bedste fleksible kandidat er stadig 0.22963 % dårligere end baseline
på MSE. Dette viser hverken signifikant gevinst eller at prisrespons er
umulig at estimere på andre data.

BM kan heller ikke frit reparere en vedvarende baselinebias, fordi

\[
\int_{t_0}^{t_1}(D-B)\,dt=C_{\rm hours}[x(t_1)-x(t_0)].
\]

Den samlede korrektion er begrænset af x i [0,1].
Det begrunder at undersøge baseline/historik og datakontrakt før flere knæk.

## Runtime og næste handling

Fit inklusive arkivreplay og kandidatforfining: 80.386 s.
Begge forecasts inklusive eksport/figurer: 3.034 s.
206 residualkald; 219 registrerede BM-forward-solves og
38,517,120 BM-RHS-evalueringer. Desuden ét indledende arkivreplay og
to WM-forecast-solves. Budgettet på 3,000 residualkald/600 s er overholdt
med god margin. BLAS=1, én proces; ingen ny natkørsel.

G4's lille udviklingsvalg er frosset til lineær BM.
Den næste afgrænsede prognoseopgave er datadækning/split og nødvendig
adapter til et fælles længere fit, eventuelt helår, med efterfølgende
testuger. Samme lange træningsvindue bruges til matched BM/WM.
Kilde-/populations-/målekontrakten afklares før en ny historisk validering.
Fleksibel g kan vurderes igen på den nye TRAIN/DEVELOPMENT-kontrakt,
uden at den gamle februaruge ommærkes til blind test.
G6b→G8→G7 kan fortsætte fra den frosne lineære model;
et årsfit er ikke en ny gate for de obligatoriske metodeleverancer.
