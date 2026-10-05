# G4: tre rullende strategier og straf på de første 12 uger

Gennemført 6. oktober 2026 efter Johans forsøgsdefinition. Strategi 3 med
lambda=0.001 har lavest samlet forecast-MSE, 0.2052253468, mod baseline
0.2076682236: 1.1763 % forbedring. Straffen forbedrer strategi 3 med 0.6207 %
og strategi 2 med 9.0688 %. Begge forbedringer drives især af uge 10 og skifter
fortegn, hvis den uge udelades. Derfor er S3 med straf en foreløbig
udviklingskandidat; en stabil forbedring fra straf er ikke dokumenteret.

Dette er DEVELOPMENT med realiserede fremtidige priser og temperaturer,
legacy-punktobservationer og tilstandsapproksimation. Det er ikke blind
validering, en endelig observationskontrakt eller en origin-tilgængelig
pris-/vejrudsigtsprognose. Strategi 1 er eksplicit retrospektiv.

## Den præcise sammenligning

ISO-uger 1–12 i 2022 betyder 3. januar til 28. marts eksklusive. Syv origins:
7., 14., 21., 28. februar og 7., 14., 21. marts. Hvert mandagscutoff fryser
alle syv dagsprognoser, baseline og parametre for hele den kommende uge.
Første vindue er uge 1–5 → 6, sidste er uge 7–11 → 12. Intet tirsdagsfit
får observeret demand fra mandagen i den uge, det forudsiger.

| Strategi | Faste/fælles parametre | Rullende fit | Fortolkning |
|---|---|---|---|
| S1 | C,k,Xi estimeres én gang over alle 12 uger med lineær g | a separat for hver ugedag på fem matchende historiske dage | Retrospektiv diagnostik, ikke en fair prospektiv konkurrent |
| S2 | Lineær f, fast neutral historisk x0 | C,k,Xi,a separat for hver ugedag på fem matchende historiske dage | Syv fits per mandag og strafstyrke |
| S3 | Lineær f, fast neutral historisk x0 | Ét fælles C,k,Xi,a over alle 35 træningsdage | Ét fit per mandag og strafstyrke, samme kurve på alle ugedage |

For S1/S2 indgår fx de fem mandage i samme LS-fit; de er ikke fem separate
fits, som bagefter gennemsnittes. S3 summerer residualer fra alle 35 dage.
De historiske dage simuleres som uafhængige forløb: hver begynder det
foregående døgn med x=0.5 og 24 timers warmup. Warmup indgår ikke i loss.
Ingen adskilte mandage kædes til én kunstig sammenhængende ODE.

Prognosestarten for en ugedag er medianen af de fem tilsvarende historiske
x-værdier efter warmup, beregnet med kandidatens parametre. S3 bruger samme
ugedagsmedian, selv om parametrene er fælles for 35 dage. De syv prognosedage
har dermed egne historiske startapproksimationer; de er ikke én uafbrudt
168-timers statepropagation. Dette følger dagsfit-ideen og skal skelnes fra
G's senere rekursive cutoff-state-prognose.

Alle strategier bruger samme WM4 `statistical_output`-baseline B, estimeret
på alle tilgængelige målte historiske demand/temperaturtimer før mandag.
Baselinen er altså expanding-window, mens de fire BM-koordinater i S2/S3
bruger det erklærede femugersvindue. Baselinens egen faste ridge/Huber-
procedure er ens i alle modeller. Forsøgets »uden straf« betyder uden straf
på a; det fjerner ikke WM-baselinens eksisterende regularisering.
S1's indledende retrospektive fit bruger en separat baseline over alle 12
uger; dens testprognoser bruger samme ærlige mandagsbaseline som S2/S3.

## Matematik og hvorfor straffen kan hjælpe

Den eksisterende deterministiske BM-kerne anvendes uændret:

\[
\dot x(t)=\frac{D_\theta(t)-B(t)}{C},\qquad
\delta(t)=\tanh\!\left(\frac{k}{2}[1-2x(t)+g_a(u(t))]\right),
\]

\[
D_\theta(t)=B(t)+\Xi\,\delta(t)
\begin{cases}1-B(t),&\delta(t)\ge0,\\B(t),&\delta(t)<0.\end{cases}
\]

C er tidsparameteren i timer, k styrer responsens stejlhed og Xi dens
amplitude. Her er de empiriske parametre på den eksisterende normalisering,
ikke validerede fysiske kapaciteter. f(x)=1−2x holdes lineær. Prisen u er i
[0,1], og den kontinuerte, ikke-voksende to-stykkers g er

\[
g_a(u)=\begin{cases}
1+2(a-1)u,&0\le u\le\tfrac12,\\
a-2(1+a)(u-\tfrac12),&\tfrac12<u\le1,
\end{cases}\qquad -1\le a\le1.
\]

Dermed er g(0)=1, g(1)=−1 og g(1/2)=a. a=0 er præcis den lineære
reference g0(u)=1−2u. a er en kurveknude; det er ikke hele prisresponsens
styrke. Straf mod a=0 fjerner ikke prisresponsen.

For en samling historiske forløb H og målte timer M er kriteriet

\[
J_\lambda(\theta,a;H)=\frac1{N_H}
\sum_{(d,h)\in M_H}[D_{\theta,a,d}(h)-y_d(h)]^2+\lambda a^2,
\qquad N_H=|M_H|,\quad\lambda\in\{0,0.001\}.
\]

S1 estimerer først theta=(C,k,Xi) på H=alle 12 uger, med a=0. Dernæst
minimeres J over a alene på hver femdages ugedagssamling. S2 minimerer over
(C,k,Xi,a) på hver sådan samling. S3 minimerer over de samme fire koordinater
på alle 35 træningsdage. x0 under historisk warmup er altid fast 0.5.
Manglende y giver ingen residual; dagen udskiftes ikke med en sjette ugedag.

Residualvektoren i SciPy er de målte modelresidualer divideret med sqrt(N),
efterfulgt af sqrt(lambda)*a. Squared norm er netop J (SciPy's indre cost
har den irrelevante faktor 1/2). N-normaliseringen gør den deklarerede
straf sammenlignelig mellem 96/120 og 720/840 træningstimer.

Kurven kan skrives g_a=g0+a*h, hvor h(u)=2u på venstre halvdel og
2(1−u) på højre. Derfor

\[
\int_0^1[g_a(u)-g_0(u)]^2\,du=\frac{a^2}{3}.
\]

Straffen lambda*a² er altså 3*lambda gange kurvens kvadratiske afstand til
den lineære reference. Der er en klar geometrisk begrundelse for denne
straf; den er ikke i sig selv en garanti for bedre predictions.

I en lokal g-only-approksimation D(a)≈D(0)+a*q bliver løsningen

\[
\hat a_\lambda\approx
\frac{\sum_i q_i[y_i-D_i(0)]}{\sum_iq_i^2+N\lambda}.
\]

En svagt identificeret knude trækkes derfor mod den lineære reference.
Det kan reducere følsomhed over for træningsstøj, men introducerer bias.
Formlen er kun en lokal forklaring med faste øvrige parametre; joint-fit
kan også ændre C,k,Xi og lande i forskellige lokale minima. Ved præcist
løste globale optima kan straf ikke forbedre den ustraffede TRAIN-MSE;
den relevante gevinst er lavere fejl på nye dage. Vores LS er nonkonveks
med to deklarerede starter, så der påstås ingen garanti for globalt optimum.

## Forecastresultater på samme 1175 målte timer

Fejlmål er i rå workbook-enheder (MSE i kvadrerede enheder). Ingen
kWh-fortolkning tilføjes til den eksisterende legacy-kontrakt.

| Model | MSE uden straf | MSE med straf | MSE-reduktion fra straf | Uger hvor straf vinder |
|---|---:|---:|---:|---:|
| Fælles baseline O | 0.2076682236 | — | — | — |
| S1, retrospektiv | 0.2072895064 | 0.2071932666 | 0.0464 % | 5/7 |
| S2, separate ugedagsfits | 0.2358689857 | 0.2144786038 | 9.0688 % | 4/7 |
| S3, fælles 35-dagesfit | 0.2065071888 | 0.2052253468 | 0.6207 % | 3/7 |

S3 med straf har RMSE 0.4530180424, MAE 0.3520961399 og bias
0.0247906140. Uden straf er S3's RMSE 0.4544306204 og MAE 0.3532369035.
S2 med straf er stadig 3.2795 % dårligere end baseline. S1 må ikke bruges
til at erklære en prospektiv modelvinder, selv om dens score er tæt på O.

| Testuge | S2 uden | S2 med | S3 uden | S3 med |
|---|---:|---:|---:|---:|
| 6 | 0.14131231 | 0.14052288 | 0.14245707 | 0.14268866 |
| 7 | 0.15425502 | 0.15801835 | 0.15577895 | 0.15579431 |
| 8 | 0.40532409 | 0.41016730 | 0.40622925 | 0.41142362 |
| 9 | 0.21182751 | 0.21929192 | 0.21243031 | 0.21115576 |
| 10 | 0.42441981 | 0.27013636 | 0.21989361 | 0.20745576 |
| 11 | 0.15125117 | 0.14219885 | 0.15311372 | 0.15045634 |
| 12 | 0.16225480 | 0.16069442 | 0.15534286 | 0.15731781 |

Uden uge 10 øger straffen samlet MSE med 0.3818 % i S2 og 0.2836 % i S3.
Begge vinder kun 21 af 49 dage med straf. Dette er en følsomhedsanalyse på
de samme frosne forecasts, ingen nye fits og intet alternativt modelsplit.
Uge 10 skal fortsat tælle i hovedresultatet; dens store forbedring er reel
evidens i forsøget. De overlappende træningsvinduer og kun syv vinteruger
understøtter ikke en simpel iid-uge-test eller en bred konklusion om hele året.

I S2 er median |a| uden straf 0.999998, med straf 0.044281. I S3 er den
0.753758 mod 0.014333. Straffen holder altså g tættere ved den lineære
reference, og udfaldet er foreneligt med beskyttelse mod enkelte ekstreme
fits. Den beviser ikke, at den sande prisrespons er lineær eller kausal.
S2's valgte data-Jacobian har lokale rangtal 1–4 og mange bound-løsninger;
S3's har rang 4, men nogle parametre rammer stadig bounds. Lokal fuld rang
er ikke ensbetydende med praktisk stærk identifikation.

## Data, normalisering og kendte begrænsninger

Rå input er MED-gruppens eksisterende `input.xlsx` og DK1 `prices.xlsx`,
Git-blobs henholdsvis 373f924089b3c2f603a8e88bff1018e06616bfa1 og
36224ea1521a00d18614e221f41e8ee82bbc6f1e. DMI's timeopløste `countryValue`,
`mean_temp`, bruges efter UTC→Europe/Copenhagen-konvertering. Originalt
DMI-svar, rå inputs og audit er arkiveret sammen med resultaterne.

Demand normaliseres én gang på uger 1–5 med minimum 0.4023076923 og spænd
2.44. Senere observationer uden for dette interval klippes ikke. Prisen
normaliseres med det faste scenariedomæne [−500,6000] DKK/MWh:
u=(price+500)/6500. Det er erklæret efter inputdækningens audit, før fits;
det er ikke en skala estimeret af TRAIN eller af fremtidige min/max. Det er
heller ikke en påstand om officielt markedsdomæne. Observeret prisinterval
er [−141.699997,5207.439941]. Dette domæne er et udviklingsvalg, som skal
fastlåses før senere blind validering. Ingen fremtidig reskalering eller
prisklipning udføres.

Jan. 13–17 har 120 manglende demandtimer; de indgår ikke i fitloss eller
baselinefit. Warmup kræver kun pris/temperatur. Den ikke-eksisterende lokale
time 27. marts kl. 02 er fjernet: arbejdsbogen har en demandværdi, men hverken
pris eller temperatur, fordi dagen har 23 fysiske timer. Alle modeller
scorer de samme 1175/1175 faktiske timer. Negative demandobservationer,
herunder −0.333077 den 27. marts, beholdes. Model/baseline på [0,1]-skalaen
kan ikke ramme alle observationer uden for første træningsspænd; dette er
en konkret begrænsning i den eksisterende skalakontrakt.

Pris holdes konstant per fysisk time, temperaturen interpoleres mellem
timepunkter og sidste temperatur holdes ved døgnets endpoint. Baseline
evalueres på et fast 60-punkter/time-grid med lokale kalenderfeatures,
uafhængigt af ODE-forfiningen. Realiserede fremtidige inputs giver en
betinget forecasttest; næste reelle ugeprognose skal bruge scenarier eller
arkiverede pris-/vejrudsigter kendt ved origin. Resultatet dokumenterer
derfor demandmodellens betingede ydeevne, ikke hele prognosekædens fejl.

## Kode, numerik og faktisk kontrol

`BM1PooledInputs` generaliserer fem-forløbsadapteren til en ikke-tom samling
af uafhængige 47/48-timers forløb. `BM1WeekdayInputs` kræver stadig præcis
fem. Den eksisterende responsegeometri og RK4-kernel genbruges; ingen ny
ODE-kernel, WM-dynamik eller F-kode er implementeret eller ændret.

`bm_1_deterministic_strategy.fit_strategy` har tre eksplicitte koordinatsæt:
linear, g-only og joint. Positive C og k fittes i log-koordinater.
Bounds er C∈[1,744] timer, k∈[0.05,10], Xi∈[0,1], a∈[−1,1].
SciPy TRF bruger 2-point-Jacobian, diff_step=1e-4, x_scale=jac og de faste
ftol/xtol/gtol=1e-7. Data-Jacobianens singularværdier beregnes uden
strafresidualen; straf må ikke skabe et falsk identifikationsargument.

S2/S3 starter med det oprindelige januarfits C,k,Xi og a=0 samt den faste
anden seed C=12,k=2,Xi=0.5,a=0.25. Januarfit slutter før første testorigin;
det bruges kun som seed, ikke som fast fit eller ekstra TRAIN-residualer.
Ingen S2/S3 seed kommer fra S1's fremtidsinformerede fit. S1 bruger a=0/0.25.
Der udføres højst to deklarerede starter per styrke. 750 residualkald per
start, indledningsvis 1500 per styrke, højst 3000 samlet per parret opgave
over genoptagelser/forfininger og 600 beregningssekunder. Jacobianprober
tæller med. Budgetstop er ikke konvergens.

Alle 105 parrede opgaver er komplette (210 valgte kandidater), plus ét
retrospektivt globalfit. Alle valgte kandidater består sammenligning af
12 og 24 ODE-trin/time med max normaliseret demandforskel ≤1e-5, både
historiske forløb og forecasts med forfinede medianstates. Værste forskel:
8.4114933e-6 (S2); S3: 2.2329664e-6. Ingen kandidater krævede genfit på
de forhåndsdeklarerede finere grids. Dette er en tidsforfiningskontrol,
ikke bevis for kontinuumsfejl under enhver tolerance.

412/421 optimizerstarter i de parrede opgaver konvergerede; hvert valgt
fit har mindst én konvergeret start. S2/uge 7/fredag med straf havde først
to 750-kalds budgetstops. Kun den kandidat blev fortsat fra gemt bedste
provisoriske punkt og konvergerede efter 20 yderligere kald, i alt 1520.
De øvrige 104 komplette opgaver blev genbrugt. Oprindeligt negativt
resultat og logs bevares. Ingen tolerance, lambda, bounds eller objektiv
blev ændret for fortsættelsen. Den største parrede opgave brugte 2278
residualkald, altså under 3000.

Det retrospektive globalfit fandt C=743.9884943, k=0.0521046780,
Xi=0.4049450; værdierne er tæt ved C/k-bounds, og den mindste lokale
singularværdi er 2.51e-7 mod største 0.0101. Det er en advarsel om svag
praktisk identifikation, ikke en fysisk kapacitetsmåling.

Summen af optimizer-walltid over parallelle workers var S1 39.33 s,
S2 644.54 s, S3 508.44 s samt globalfit ca. 15.10 s; cirka 20.1 worker-
minutter, ikke brugerens samlede ventetid. Otte workers, BLAS=1 blev brugt.
`summary.parallel_seconds=0.881` beskriver kun den sidste resume-invocation
og må ikke rapporteres som hele forsøgets runtime. Numeriske checks og
filskrivning ligger ud over de summerede optimizer-tider.

Kørte fokuserede tests: de to eksisterende weekday-tests (uafhængige
forløb/masker og target-demand-invarians), de nye g-only/straf- og DST-tests,
samt senere kun g-only med checkpointcontinuation og inputfingerprint.
Tre nye testmetoder dækkes samlet; den sidste tokontrol tog 0.163 s.
AST-syntakskontrol af de fire ændrede/nye Python-filer består, herunder
runnerens endelige budgetstramning. Ingen fuld discovery eller unødvendig
WM-/januargenfit blev udført.

Efter komplette fits blev resume bundet til SHA256 af pandas' hash af
indekserede Demand/Price/Temp samt reference-seeden. Den parrede totalbudget-
guard blev samtidig strammet fra separate styrkeloft til 3000 samlet.
Alle faktiske opgaver lå allerede under dette loft. Metadataopdateringen
er deklareret i `provenance_hardening.json`; scores/forecasts er uændrede,
og modellerne blev ikke genkørt for denne hærdning. Sourcepin/review mod
nyeste main bekræftede uændrede fem fælles G4/WM-kernefiler. G/F-sporene
forbliver separate; F4-H2-status og samtidige WM-ændringer er bevaret.

## Reproduktion og næste beslutning

Råarkivet `G4_12_uger_strategier_raaresultater_20261006.zip` indeholder
inputs, DMI-svar, baselineaudit, alle fits/checkpoints, numerik, forecasts,
scoring, analyse og figur. Kode/versioner ligger i denne PR; benchmark-JSON
ligger i `files/output/bm_1/benchmarks/g4_strategies_12_weeks_20261006.json`.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -B -m src.exe.run_model.bm_1.deterministic_strategies `
  --raw-csv PATH_TO_ARCHIVE/raw_inputs.csv `
  --reference-fit PATH_TO_ARCHIVE/reference/fit.json `
  --output files/output/bm_1/runs/deterministic/g4_strategies_new `
  --workers 8
```

`--resume` genbruger kun med identisk inputfingerprint, seed og protokol;
komplette opgaver optimeres ikke igen. Brug en ny outputmappe til ændrede
studier. Johan behøver ikke køre noget lokalt for at gennemføre dette forsøg.

Vi beholder S3 med straf som bedst scorede kandidat og S3 uden straf som
obligatorisk sammenligningsreference. En senere, hidtil ubrugt periode skal
afgøre, om den lille strafgevinst holder; lambda skal være fast før scoring.
Den tidligere treugers rolling-day-konklusion (lambda=0) bevares for den
gamle protokol: daglige cutoffs, fast k og anden prisnormalisering gør den
til et andet forsøg. Ingen universel ændring af G's strafpolicy eller
implicit implementering af helårsfit foretages. G6b→G8→G7 består som
obligatorisk metodekæde; denne prognosediagnose erstatter den ikke.
