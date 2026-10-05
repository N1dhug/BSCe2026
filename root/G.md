# G-plan — deterministisk BM-kerne, stokastisk udvidelse og WM4-sammenligning

**Beslutning fra Johan: 5. oktober 2026. Status: G0–G3-kode implementeret og syntetisk udviklingskørsel gennemført; historisk januar-fit afventer lokale inputs. G4–G7 er planlagt.**

G er et selvstændigt spor ved siden af F-planen. Formålet er en enkel, hurtig
BM-arbejdsgang: historiske data → deterministisk parameterfit → fastlåst
ugeprognose → sammenligning med baseline og WM4. Grundlaget er den eksisterende
BM1, ikke en ny model med WM's ligninger. Processtøj, filtre, parameterbanker og
fordelingspropagation indgår ikke i første G-leverance. Derefter er et billigt
stokastisk usikkerhedslag en obligatorisk G-leverance, ligesom WM's trinvise
opbygning; G som helhed er altså ikke 100 % deterministisk.

**Præcisering fra Johan, 5. oktober kl. 12:56: WM4 er fælles standard dér,
hvor modellerne kan gøre det samme.** Genbrug konkret baselineimplementation,
fit/predict-protokoller, evalueringsdefinitioner og rapportstruktur. Afvigelser
skal skyldes modellernes matematik eller en eksplicit fælles datakorrektion og
registreres i manifestet; en særskilt BM-baseline er ikke default.

**Navne:** Skriv altid `G-plan/G0` ... `G-plan/G7` i status og henvisninger.
`F4-G`, `F4-G2a`, `F4-G2b` og `F4-G2c` er fortsat de gamle stokastiske studier.
G kræver ikke, at disse studier eller H-f/g afsluttes. G overtager heller ikke
deres evidens eller afslutter deres krav. F-sporets status står i [F4.md](F4.md).
Brugerens nye G-bestilling tillader dette deterministiske spor nu; den ophæver
ikke F-sporets aftalte rækkefølge eller dets senere prognosediagnose.

## Aktuel G0–G3-leverance, 5. oktober

Johan har bestilt **parameterfit over hele januar**. Første konkrete månedsvariant
bruger 1. januar 2022 kl. 00 til 1. februar kl. 00 (744 timer), ét fælles
(C_hours,k,Xi,x0), og derefter 168 timer til 8. februar. BM og WM bruger begge
den fulde fitmåned. Månedsvarianten har warm-up=0 for at score hele januar;
ugevarianten bevarer WM's 24 timer og én fituge. Ingen daglige state-resets.

Implementationen omfatter dataaudit, træningsfast fælles skalering, genbrugt
WM-baseline, lineær BM1-forwardmodel, bounded SciPy-TRF, checkpoints, cutoff,
tidsforfining og fælles WM/BM-rapporter. Se [G_JANUARY.md](G_JANUARY.md) for
kommandoer, bounds, numerik og begrænsninger. Måletype/enhed er endnu ikke
afklaret nok til et historisk videnskabeligt acceptclaim; kørslen mærkes
hourly-point-approximation og development.

En syntetisk 744+168-timers kæde med to starter er gennemført på cirka 22 s,
130 residualkald og bestået tidsforfining. Det er implementerings-/runtimeevidens,
ikke historisk januar-estimation eller en retfærdig modelrangering: data er
genereret med BM1. Den store historiske input-workbook kunne ikke materialiseres
gennem denne chats connector. Lokal januarkørsel mangler stadig.

## 1. Leverancer og rækkefølge

| Etape | Arbejde og konkret leverance | Afslutningspunkt |
|---|---|---|
| G-plan/G0: fælles forsøgsgrundlag | Fastlæg population, WM4-commit, datakontrakt, tidsenhed, fælles baseline, pris-/effektskala, historik, split og runtime-budget i et manifest. | Begge modeller kan få samme fysiske inputs, observationer og information ved samme forecast-origin. Uklar måletype giver kun en mærket udviklingsdemo. |
| G-plan/G1: deterministisk forwardmodel | Genbrug BM1's ODE og F1-reference. Klargør en lille evaluator med punkt-/intervaloutput, energibalance og en eksakt intervalvej, hvor antagelserne gælder. | Meningsfulde tests af energibalance, randtilfælde, inputskift og enheder; eksakt vej kontrolleret mod uafhængig ODE. |
| G-plan/G2: parameterfit | Fit fælles C, k og Xi samt højst én initialtilstand for et sammenhængende træningsforløb; først f(x)=1−2x og g(u)=1−2u. Bounded nonlinear least squares med få startpunkter. | Fit-artifact med parametre, score, stopårsag, samtlige solve-kald, runtime, initialisering og lokal identifikationsdiagnose. |
| G-plan/G3: første ugeprognose | Frys parametre ved origin, viderefør træningens sluttilstand og simulér syv døgn uden observationer fra testugen. Kør matched WM4 og baseline. | Fælles CSV, ugefigur og RMSE/MAE/bias samt baselinegevinst. Resultatet må gerne være negativt. G0–G3 er første brugbare leverance. |
| G-plan/G4: fælles f/g-kurver | Undersøg først en lille prisresponsudvidelse, derefter eventuelt f; brug eksisterende kontinuerte, monotone kurveobjekter. Ét kurvepar per population og træningsfit. | Afgrænset udviklingssammenligning mod lineære kurver; validering afgør om ekstra fleksibilitet beholdes. Ingen kurver per dag. |
| G-plan/G5: afsluttende benchmark og dokumentation | Fastlås opskrift og modelvalg før sluttest. Evaluer de på forhånd valgte ugeorigins i én måned med identisk information og rapportér alle uger. | Reproducerbar lille benchmarkpakke med kvalitet, runtime, begrænsninger og klart skel mellem udvikling og sluttest. |
| G-plan/G6: stokastisk usikkerhed og evaluering | Bevar det deterministiske fit som reference, tilføj BM-processtøj, fit få støjparametre og propagér approksimativ varians. Genbrug WM's usikkerhedsmål og figurer. | Obligatorisk afgrænset pilot og held-out evaluering af dækning, bredde og intervalscore; negative resultater rapporteres. Se afsnit 8. |
| G-plan/G7: separat senere prisoptimering | Når forecastdelen er vurderet: genbrug eksisterende deterministisk kontrol med et fælles, eksplicit mål og få prisbeslutninger. | Valgfri efterfølgende leverance. Ingen forudsætning for G0–G6 og ingen fælles estimation af parametre og priser. |

**Præcisering fra Johan, 5. oktober: G skal også indeholde stokastik.**
G0–G3 er fortsat første deterministiske milepæl; G6 følger det fastlåste
strukturelle modelvalg i G4–G5 og er del af den samlede G-leverance.
G5's punktprognosebenchmark afslutter derfor ikke hele G-planen. G6 får eget
udviklings-/testsplit fastlagt i G0, så allerede inspicerede G5-data ikke
senere præsenteres som blind sluttest for usikkerhedsmodellen.
Ingen stor arkitekturomskrivning eller komplet ny F1–F4-valideringsserie.

### 1a. Bindende genbrug fra WM4

| Område | Konkret WM4-reference | G-planens valg |
|---|---|---|
| Baseline | `output_model.py`: `statistical_output.fit/predict`, `output_callable` | Samme implementation, indstillinger, historik, cutoff og features. Fit én gang per origin; samme fitted baselineforløb gives til begge modeller efter eksplicit enhedsomregning: B=O. |
| Ugeprotokol | `predict_weekly.py`: én fituge, 24 timers warm-up, næste 168 timer | Første standardbenchmark for begge modeller. 28 træningsdage er kun en senere fælles variant. |
| Dagsprotokol | `predict_daily.py`: fem tidligere matchende ugedage, 24 timers warm-up | Samme observationsudvalg og cutoff ved et særskilt dagsbenchmark. Daglig opdatering og en uges åben prognose holdes adskilt. |
| Fejlmål | `deviation_metrics` | Samme MSE, RMSE, MAE og bias=prediction−observation på identiske observationer og i fælles enheder. |
| Performanceanalyse | `analyze_performance.py`: `metric_gain`, `analyze`, `summary_table`, `weekday_summary` | Samme baselinegevinst, procentvis MSE-forbedring, bedre/lige/dårligere dage, gennemsnit/median, ugedagsopdeling og bedste/værste dage. |
| Figurer | `save_plot` i dags-/ugekørslerne | Samme datoakse, observationsmarkører, baseline, prognose og pris. Sammenlignelige paneler med korrekte modelnavne/enheder. |
| Eksport | Daglige metrics-CSV og performance-CSV | Samme metriknavne/suffikser og baselinekolonner; BM får eget modelnavn. Fælles evaluator accepterer modelnavn og bevarer kompatibilitet med WM4-output. |
| Usikkerhed (G6) | `test_diffusion_hypotheses.py`, `process_uncertainty` | Samme nominelle niveau, dækning, intervalbredde, intervalscore og prognosehorisonter; modelspecifik varianspropagation. WM's oracle-initialisering må kun genbruges i særskilt oracle-diagnostik. |
| Diagnostik | Baselinegevinst, residualer og initial-state-følsomhed | Samme spørgsmål/informationsgrænse; modelspecifik beregning. WM's dæmpningsratio og acceleratorstate kopieres ikke til BM. |

Baselinen bruger aktuelt døgn-/sæsonharmoniske led, ugedag, temperatur og
interaktioner, med robust ridge/Huber-fit. Defaults er base_temperature=18,
hour_harmonics=4, season_harmonics=2, ridge=1e-3, huber_delta=1.5, max_iter=50.
De versionsfastlåses; samme fitted objekt eller serialiserede parametre bruges
for BM/WM. Et fem-ugers timegennemsnit erstatter ikke denne baseline: fem
matchende ugedage er her WM4's dynamiske træningsudvalg.

`statistical_output.predict` og `output_callable` begrænser aktuelt baseline
til [0,1]. Det er en eksplicit del af fælles baselinekontrakt: gem policy og
antal/rå overskridelser i en audit. Ingen ekstra BM-specifik klipning af
baseline, observationer eller state. Nødvendige datakorrektioner, fx en anden
normalisering, gælder begge modeller og versionsmærkes; original WM4 under en
anden kontrakt står særskilt. Dette præciserer planens regel mod skjult clipping.

WM's baseline interpoleres aktuelt lineært mellem dens fine samples.
Hovedsammenligningen genbruger dette tidsforløb. Den eksakte BM-intervalvej
bruges kun, når dens konstant-input-forudsætninger gælder. En held-baseline-
variant kræver samme ændring for WM og separat resultatlabel. Samme outputgrid
kræver ikke samme interne solver eller 60 ODE-substeps per time i BM.

Genbrug kode direkte, hvor API'et passer; ellers udtræk små modeluafhængige
helpers med bagudkompatible WM-wrappers og numerisk paritetstest. Ingen kopier
af baseline-/metricskode, der senere udvikler sig forskelligt. WM's QP-fitter
erstatter ikke BM's nonlinear fit. Predict-setup, der kræver fremtidigt observed
I, opdeles i inputforberedelse og efterfølgende scoring for begge modeller.

Performancekonvention: gain=baseline_error−model_error; for bias bruges
|baseline_bias|−|model_bias|. Procentgevinst er 100*gain/baseline_MSE og NaN
ved baseline_MSE≤1e-12 som i WM. Better/equal-tolerancen 1e-12 genbruges til
kompatibel rapportering, men er ikke statistisk signifikans eller praktisk
modelvalg; G's numeriske/praktiske tolerance rapporteres separat. Vis både
WM's gennemsnit af dagsmål og pooled fejl over observationer med tydelige
labels: gennemsnitlig dags-RMSE er ikke pooled RMSE. Bedste/værste dage er
visninger af den samlede rapport, ikke udvælgelse af testdata.

Figurerne skal vise, om korrektionen forbedrer baseline, hvornår den hjælper
eller skader, og hvor stor den er. Vis D−B for BM og I−O=R for WM i samme
forbrugsenhed. BM's x og eventuelt dx/dt vises separat fra WM's F/R;
WM's dR/dt er ikke BM's dx/dt. G0–G5 viser punktprognoser; G6 tilføjer tydeligt mærkede approksimative
stokastiske bånd og fælles usikkerhedsmål.
Billige dags-/ugedagsanalyser bruger gemte prognoser uden nye fits. Øvrige
WM-diagnoser tilføjes kun ved relevans og inden for eksisterende G-budget.

G0 leverer en kompatibilitetsmatrix med kildecommit, genbrugt helper, nødvendig
adapter og begrundelse for hver forskel. G3 kontrollerer identisk baseline på
fælles timestamps og identiske metrics for samme predictions. G5 leverer
fælles dag-/ugefigurer, metrics og WM-lignende performanceanalyse.

## 2. Første trin: samme BM1, uden processtøj

Med x i [0,1], normaliseret baseline B og pris u er

$$\dot x(t)=\frac{D(t)-B(t)}{C},\qquad x(t_0)=x_0,$$

$$\delta(x,u)=\tanh\!\left(\frac{k}{2}[f(x)+g(u)]\right),$$

$$D(t)=B(t)+\Xi\begin{cases}(1-B(t))\delta(x(t),u(t)),&\delta\geq0,\\B(t)\delta(x(t),u(t)),&\delta<0.\end{cases}$$

C og k er positive, Xi og x0 ligger i [0,1]. f og g er kontinuerte,
ikke-voksende og går fra 1 til −1. Det giver indadrettet randdrift for gyldige
inputs og dermed invarians af det kontinuerte tilstandsinterval. Numeriske
overskridelser skal stadig kontrolleres; ingen skjult clipping af state/data.
Fysisk tid skal være ens i C og t: C=C_phys/P_scale er en tidsstørrelse.
Projektets nuværende dagsskala kan genbruges med eksplicit omregning til timer.

Den deterministiske energibalance er

$$C[x(t_b)-x(t_a)]=\int_{t_a}^{t_b}(D(t)-B(t))\,dt.$$

Det betyder også, at en begrænset virtuel batteritilstand ikke kan kompensere
for vedvarende baselinebias ubegrænset. Store systematiske residualer skal
diagnosticeres, før man tilføjer flere kurveparametre. Nul processtøj og et
bestemt x0 giver én bane; det er generelt ikke middelbanen af den ikke-lineære
stokastiske BM. Historiske målefejl/residualer kan godt findes, selv om den
latente dynamik i G0–G5 er deterministisk.

### Eksakt og billig reference for lineære kurver

For f(x)=1−2x, g(u)=1−2u og konstante B_j,u_j over et interval af længde h_j,
sæt q_j=1−u_j og gamma_j=1−B_j, hvis x_j<q_j, ellers gamma_j=B_j. Da fås

$$x_{j+1}=q_j+\frac{1}{k}\operatorname{arsinh}\!\left(\sinh(k(x_j-q_j))\exp[-k\Xi\gamma_jh_j/C]\right).$$

Formlen er allerede dokumenteret i F1 og implementeret som et fast benchmark.
G1 generaliserer om nødvendigt den lille intervalrutine til parameterforslag
og inputarrays; den importerer ikke benchmarkets hårdkodede parametre til
produktionsfit. Kald med store/små argumenter kontrolleres for cancellation
og overflow. Xi=0 og x_j=q_j skal give konstant tilstand.

For intervalmiddeldata kan efterspørgslen beregnes direkte fra balancen:

$$\widehat y_j=B_j+\frac{C(x_{j+1}-x_j)}{h_j}.$$

Det kræver ingen fin kvadratur under ovenstående forudsætninger. For energi
er observationen i stedet integralet, med korrekt fysisk effektskala.
For punktdata bruges D ved det deklarerede måletidspunkt og den korrekte side
af et inputskift. Intervalgrænser opdeles ved alle B/u-skift.

En konstant-timebaseline må ikke indføres alene for at bruge formlen, hvis
den aftalte WM-sammenligning kræver en kontinuerlig baseline. G0 fastlåser
inputrepræsentationen for begge modeller. G0–G3 bruger ved varierende B en billig skalær RK4-vej med de samme BM1-
ligninger, verificeret mod eksisterende segmenteret `solve_ivp`. Første fit har
60 trin/time for at ramme knækkene i WM's fine, lineært interpolerede baseline;
120 trin/time er numerikkontrollen. Et grovere grid afviste en syntetisk kandidat
under uændret præcisionsgrænse. Dette er en kontrolleret numerisk specialisering,
ikke en konstant-baseline-ændring. Stykvise f/g er endnu ikke implementeret i
G-fitteren; den eksisterende ODE-vej er fortsat tilgængelig.

## 3. Parameterestimation

Første fit har theta=(C,k,Xi) og eventuelt x0 som en fjerde ukendt:

$$\min_{\theta,x_0}\frac{1}{2N}\sum_{i\in\mathcal T}\left(\mathcal H_i[D_{\theta,x_0}]-y_i\right)^2.$$

H_i er punktværdi, intervalmiddel eller energiintegral efter datakontrakten.
Observationernes tidsopløsning bestemmer residualernes antal. Et fint internt
grid eller interpolation af y skaber ikke nye uafhængige observationer.
Priser og baseline er faste inputs under fit; A/aggregatorkøb indgår ikke i
dette estimationsobjective. En enkelt samlet theta bruges på alle træningsdage.

Praktisk valg: `scipy.optimize.least_squares(method="trf", loss="linear")`
med bounds, skalerede variable og to-punkts-Jacobian i første implementation.
Brug log C/log k eller eksplicit parameterskalering; dokumentér endelige
fysiske bounds i G0 før optimering. Grænser bestemmes af enheder, model og
udviklingsdata, aldrig af sluttest. Funktionen har et knæk ved delta=0;
komplekse skridt er ikke en gyldig automatisk differentiation her. Ved senere
kurvefit kommer også knæk i f/g. Ingen egen global optimizer eller fuld
Hessian er nødvendig; sensitivitetssystemer tilføjes kun ved målt behov.

Begynd med højst fire deklarerede startpunkter. Inden for samme model vælges
den bedste gyldige, konvergerede TRAIN-kandidat; valg mellem modelvarianter
sker på DEVELOPMENT. Hvis et numerisk check ikke består, registreres fejlen
og forsøget afklares. Ingen skjult udskiftning af modellen ved fejlede kald.

BM-fit er generelt ikke-konvekst, fordi theta ændrer både tilstanden og
responsen. WM4's constrained drift-regression kan være en konveks QP, når
rekonstruerede states er fastholdt; det gør ikke BM-fit til en QP og garanterer
heller ikke, at WM4's regressionsscore er optimal for en hel ugeprognose.
Sammenlign begge modellers held-out forbrugsscore, ikke deres interne fitmål.

### Små identifikationskontroller

- Undersøg skaleret Jacobian/SVD, aktive bounds og forskelle mellem de højst
  fire startpunkter. Det er lokal diagnostik, ikke et globalt entydighedsbevis.
- Nær delta=0 er fleksibilitetsamplituden omtrent proportional med Xi*k;
  parametrene kan være svære at skille ad. Ved svag excitation kan C også
  være dårligt bestemt. Xi=0 gør dynamikparametre uidentificerbare fra demand.
- Brug højst én erklæret reduceret variant med k fastholdt, hvis fuldt fit er
  ustabilt. Et godt forecast må rapporteres med svag parameteridentifikation.
  Ingen store profilgrids, bootstraps eller recovery-populationer i G-kernen.
- På udviklingsdata kan kumuleret y−B og den krævede statebevægelse bruges som
  billig feasibility-diagnose. Observeret støj gør ikke dette til et eksakt
  fysisk certifikat eller en erstatning for simulation-fit.

## 4. Data, initialisering og retfærdig WM4-sammenligning

Standard er WM4's ugeprotokol: én population (først MED), én historisk fituge
med 24 timers warm-up og en prognose på næste 168 fysiske timer. Baselinefit
bruger samme tilgængelige historik før origin som WM4. Datoer og historik
fastlægges i G0. Et 28-dages fit er en senere fælles variant, ikke BM-default.
Et eventuelt dagsbenchmark bruger samme fem matchende ugedage som WM4 og
rapporteres særskilt. Kurver er fælles på tværs af de historiske træningsdage
og fastlåste under forecast. Første integration bruger en lille udviklingsuge;
fuld deklareret træning køres inden den tilhørende sluttest. Afvigelser fra
24 timer per lokalt døgn håndteres eksplicit ved DST.

Der bruges en tidsmæssig TRAIN/DEVELOPMENT/TEST-opdeling. Februar-eksempler,
som allerede er inspiceret, er udviklingsdata. Holdout er en særskilt endnu
uåbnet periode, hvis en sådan findes; ellers kaldes studiet udviklingsbenchmark.
Ved månedlig rolling-origin-test låses hele fitopskriften før første score.
Tidligere testugers data kan indgå ved senere origins efter den låste opskrift;
de må ikke bruges til manuel ændring af modelvalg eller budgetter under testen.

G0-manifestet fastlåser:

1. Hvilken population/måler, fysisk power-/energienhed, intervalbetydning,
   timestamp-side og timezone hver observation har. Gaps maskeres eksplicit;
   gap-perioder kræver kendte inputforløb for statepropagation eller separate
   segmenter med deklareret initialisering. Ingen fremtidig udfyldning.
2. Fælles WM4-baseline B=O fra `statistical_output`, estimeret uden data efter forecast-origin. Baselinens
   egen træningshistorik og in-sample/cross-fit-konvention gemmes. Udvid ikke
   til en ny baseline-modelsøgning i G. En baseline uden dynamisk pris som
   feature er ikke i sig selv bevis for en kausalt prisfri baseline.
3. Samme rå pris-/temperaturforløb og informationstidspunkt. WM4 bruger aktuelt
   signed U i [−1,1], BM u i [0,1]. Ved samme normalisering kan u=(U+1)/2
   bruges; det skal verificeres på det konkrete input. Sammenlign i fælles
   fysiske forbrugsenheder, ikke ukritisk på hver models egen normalisering.
4. Træningsfast effektskala uden skjult klipning eller held-out genestimation.
   Hvis gyldige BM-inputs ikke kan opnås, registreres domæneproblemet. Original
   WM-resultat fra en anden skala/kontrakt må kun stå som separat reference.
5. Finite parameterbounds, starter, numerik, modelvarianter, stopregler,
   kriterium for kompleksitet og både per-fit- og samlet beregningsbudget.

Kausal initialisering: fit højst ét x0 ved begyndelsen af et sammenhængende
træningsforløb og viderefør den simulerede sluttilstand til forecast-origin.
En deklareret warm-up før det scorede træningsvindue kan bruges. Ingen frie
daglige state-resets og ingen fit af x0 til fremtidige y. WM4 anvender samme
informationsgrænse med sin egen state-dimension. Hvis en state-only opdatering
fra et kort historisk prefix senere ønskes, valideres den som en separat variant.

WM4-noternes optimering af starttilstand mod hele testdagen er en ORACLE-diagnose.
Den må ikke stå som prognose over for kausal BM. En eventuel fælles oracle-
sammenligning rapporteres separat og fritager ikke for hovedbenchmarkets cutoff.
Kendt faktisk pris og temperatur i en historisk test giver en betinget prognose.
En operationel ugeprognose kræver pris-/vejrinputs, der faktisk var tilgængelige
ved origin, og behandles først, når sådanne inputs findes.

## 5. Fleksible f/g i et lille ekstra trin

Genbrug `BM1PiecewiseResponse` med endpoints 1,−1. Undersøg først to stykker
(indre knude 0.5), og kun ved relevant udviklingsgevinst fire stykker med
knuder 0,0.25,0.5,0.75,1. Første kurveforsøg udvider g med f lineær; næste
kan udvide f. Det er én sekventiel, på forhånd budgetteret vej, ikke alle
mulige kombinationer. En negativ to-stykkers pilot kan afslutte G4 med
lineær reference og en begrundet anbefaling om at stoppe.

Hold k på G2's udviklingsvalgte værdi i den første kurvepilot for at begrænse
kompensation mellem hældning og kurveform. Fit C, Xi og træningens x0 sammen
med kurveværdierne. To fire-stykkers kurver giver højst seks ekstra frihedsgrader.
Alle stykker indgår samtidigt i samme residualvektor over hele træningen.

Monotonicitet håndhæves ved parameterisering eller en optimizer, der faktisk
understøtter de nødvendige constraints. TRF understøtter box-bounds, ikke
arbitrære ordensuligheder. Genbrug en gyldig eksisterende parameterisering;
hvis positive normaliserede fald bruges, fastlås én logit for at fjerne
softmax-skiftinvarians. Eksakte flade stykker ligger da eventuelt kun i grænsen,
hvilket skal oplyses. Undgå sortering/clipping af kandidatknuder under fit.

En lille, dimensionsløs straf mod lineære kurver eller knæk kan tilføjes som
ekstra residualer, med styrke fastlagt på udviklingsdata og gemt i manifestet.
Test højst to på forhånd valgte styrker inklusive nul inden for G4's samlede
budget. Rapportér både datafejl og regulariseret mål. Den enklere model vælges,
hvis udviklingsgevinsten ligger inden for den på forhånd fastlagte praktiske
tolerance, med numerisk fejl medregnet. Pris-/stateintervaller uden tilstrækkelig
dækning får ingen claim om estimeret kurveform.

G4 genbruger H's kurvegeometri, men H's likelihood-fit og stokastiske accept
er selvstændige. Parametre/kurver overføres ikke ukritisk mellem støjfri og
stokastisk model. Den eksisterende obligatoriske F4-H-f/g-leverance bevares.

## 6. Beregningsbudget og stopregler

Tallene er **foreslåede loftværdier for første implementation**, ikke målte
køretider eller løfter om konvergens. Før en studiekørsel låses de i manifestet.
Det første nåede loft stopper ny beregning; en pause er ikke en bestået fit.

| Arbejde | Første loft | Hvad der tælles |
|---|---|---|
| G1/G2 runtimepilot | 30 residualkald og højst 60 sekunder | Repræsentativ fuld træningslængde; opvarmning/opsætning måles særskilt. |
| Ét G2-fit | 4 starter, højst 250 residualkald per start, højst 1.000 samlet, højst 5 minutter | Inklusive numerisk Jacobian; ikke kun optimizerens rapporterede nfev. |
| G4-kurveundersøgelse | Højst 3.000 residualkald og 10 minutter i alt | Alle familier, starter, strafstyrker og genstarter deler samme loft. |
| G6-usikkerhedspilot og benchmark | Pilot: højst 60 sekunder/30 varianspropagationer. Derefter højst 500 propagationer og 5 minutter samlet for BM/WM-usikkerhedsfit, numerik og scoring | Højst to støjfamilier, to starter per familie; finite differences tæller med. Genbrug gemte deterministiske baner. Ekstra nødvendige strukturelle fits skal budgetteres inden kørsel. |
| G5-månedsbenchmark | Højst 20 minutter samlet som første driftsmål/loft | Alle deklarerede ugeorigins, BM-fit, fælles baseline og WM4-fit/forecast; delresultater gemmes. |

Et residualkald kan kræve flere baner, hvis data er opdelt i segmenter. Gem
derfor både residualkald, faktisk antal forward-solves, RHS-evalueringer og
wall-time. Numerikchecks og final evaluering har egne eksplicitte små kvoter
og tælles i det samlede studiebudget; et tidsstop må ikke bruge den sidste
kandidat som skjult færdig model. Hvis ingen start konvergerer, markeres fit
som uafsluttet, og baseline kan stadig rapporteres.

Begynd enkeltprocesseret med BLAS=1. Vectorisér inputs/residualer og genbrug
baseline, prisarrays, splits og de dele af numerikken, som er parameteruafhængige.
Cache ikke tilstandsbaner på tværs af forskellige parametre. Kun hvis målt
overhead tillader det, bruges op til 8 workers på uafhængige starter/ugeorigins
med BLAS=1 og fælles samlet budget; ingen nested parallelisering.

Undgå Cartesian-grids, Monte Carlo, FV/mixtures, tusindvis af tilfældige starter,
hel sæson før pilot og brede gentagelser af F-studier. Brug den eksakte vej
hvor kontrakten tillader det. Ellers genbrug ODE og forfin kun et lille
deklareret numerikpanel plus den valgte kandidat. Numeriske præcisionstolerancer
tilpasses ikke efter et ønsket vinderresultat. Hvis budgettet ikke rækker,
rapporteres stop, profilering og en konkret afgrænsning; ingen automatisk ekstra natkørsel.

Gem afsluttede starter, konfiguration og bedste gyldige kandidat atomisk.
Løbende forbedringer kan checkpointes med begrænset frekvens. Et fortsat
SciPy-fit fra gemt theta er en ny warm-start, ikke eksakt genoptagelse af
internt optimizer-state; tidligere forbrug skal stadig tælles. Vis grov ETA
ved længere jobs og bevar samme source under aktive kørsler.

## 7. Kodens placering, kontrol og output

Genbrug eksisterende matematik i `src/model/base/bm.py`, ODE i
`src/model/bm_1.py`, kontrakter i `src/tools/bm_1_data.py`, kurveobjekter i
`src/params/bm_1_response.py` og I/O i `src/tools/bm_1_reports.py`.
`src/params/bm_1.py` er allerede F4's observations/Bayes-primitiver og skal
ikke overskrives med G-fitteren. Den eksisterende punkt-observationscontainer
må heller ikke mærkes som intervalmodel; et intervalinterface kræver egen kontrakt.

G0–G3-kode er oprettet; G6-koden nedenfor er fortsat planlagt:

- `src/params/bm_1_deterministic.py`: lille fitkonfiguration, fitresultat og
  public API til deterministisk estimation.
- `src/method/bm_1_deterministic_fit.py`: residualopbygning, observationsoperator,
  optimizerbudget og fit; en generaliseret eksakt overgang placeres som genbrugelig
  numerisk helper efter inspektion af eksisterende solverstruktur.
- `src/exe/run_model/bm_1/deterministic.py`: tynd indgang til audit, fit,
  predict-week og compare; måneds-/ugekørsel og audit-only er nu kørbare.
- `src/validation/bm_1/deterministic_fit.py` og fokuserede tests: små syntetiske
  regressioner og cutoff-kontroller, ingen lange forskningsstudier i CI.
- G6 tilføjer en lille særskilt usikkerhedsmodul/indgang efter inspektion af
  WM's helpers; deterministiske API'er bevares. Resultater gemmes under
  `files/output/bm_1/runs/uncertainty/<run_id>/` med reference til strukturelt fit.
- `files/output/bm_1/runs/deterministic/<run_id>/`: egne resultater med
  `manifest.json`, `fit.json`, `predictions.csv`, `metrics.json` og ugefigurer.

G1 kontrollerer en lille eksakt/ODE-case, inputskift, Xi=0, rande og
timer↔dage. G2 kontrollerer et støjfrit identificerbart eksempel og en bevidst
uidentificerbar case; kræv god prediction frem for recovery, hvor identifikation
ikke findes. G3 tester, at ændrede fremtidige observationer ikke ændrer fit,
baseline, initialtilstand eller forecast, og at punkt/interval ikke forveksles.
Numerisk præcision kontrolleres på outputs/residualer; optimizerens små skridt
må ikke domineres af ODE-fejl. Ændringer i eksisterende interfaces kræver deres
relevante regressioner. Ingen lange benchmarkkørsler gentages for dokumentation alene.

Rapportér per uge og samlet: samme observed/B/WM/BM på fælles tidsstempler,
RMSE, MAE, bias=prediction−observation, MSE-gevinst over baseline og forbrugt tid.
Gem fejl og uafsluttede uger; vælg ikke kun de gode. Vis x for BM og F/R for
WM i egne paneler med deres forskellige betydninger. State-værdier er ikke
direkte sammenlignelige. Statistik fra én måned er beskrivende og dokumenterer
ikke generel sæsonperformance eller kausal priselasticitet.

## 8. Obligatorisk stokastisk trin G6 og valgfri kontrol G7

### G6: samme trinvise idé som WM, med BM's egen diffusion

Frys først de udviklingsvalgte strukturelle parametre og f/g-kurver. Brug
derefter den eksisterende BM-diffusionsform:

$$dX_t=a(X_t,t)\,dt+b(X_t,t;\eta)\,dW_t,\qquad
a(x,t)=\frac{D(x,t)-B(t)}{C},\quad b(x,t;\eta)=s(t;\eta)x(1-x).$$

Første støjfamilie har kun én parameter, s(t)=sigma≥0. Kun hvis udviklingsdata
begrunder det, prøves én ekstra familie med et fast døgnmønster:
s(t)=sigma_0+sigma_t[1+cos(2*pi*(t-phi)/24)]/2 med ikke-negative koefficienter.
phi fastlægges på træningsdata, ikke med en stor phasesøgning; t er her timer.
Ved dagsskala skal både frekvens og diffusionsenhed omregnes
(sigma_hour=sigma_day/sqrt(24)). WM's H0/H1/H2 kopieres ikke direkte ind i
BM: faktoren x(1−x) og BM's randstruktur bevares. Xi og C skalerer ikke
automatisk den eksisterende BM-diffusion.

Den billige første beregning er en lineær støjapproksimation langs den
deterministiske bane x_bar:

$$\dot{\bar x}=a(\bar x,t),\qquad
\dot P=2a_x(\bar x,t)P+b(\bar x,t;\eta)^2,\qquad P(t_0)=0.$$

Det kræver én ekstra skalarligning, ikke en Monte Carlo-population eller
Fokker–Planck-løsning. P0=0 betyder betinget kendt initialtilstand; parameter-,
baseline-, fremtidig input- og initialtilstandsusikkerhed er ikke inkluderet.
Dette er en eksplicit approksimation, ikke den eksakte varians for den
multiplikative SDE. Den deterministiske bane er heller ikke automatisk dens
middelværdi. Behandl responsknæk med segmentering og numeriske kontroller.

For punktmålt demand bruges delta-metoden:

$$V_D(t)\approx D_x(\bar x,t)^2P(t),\qquad
\widehat D(t)\pm1.96\sqrt{V_D(t)}.$$

Båndene er nominelle, approksimative 95 %-procesbånd. De klippes ikke skjult
til et fysisk interval. Ved stor spredning, randkontakt eller utilstrækkelig
kalibrering rapporteres begrænsningen; en dyrere metode startes ikke automatisk.
For intervalmiddel/energi propagéres også kovarians med det integrerede demand
i et lille udvidet system; gennemsnit af punktvarianser er ikke intervalvariansen.
Den deterministiske energiformel fra afsnit 2 må ikke bruges til stokastisk
demand uden støjintegralet i statebalancen.

Fit støjparametrene separat efter strukturelt fit, med samme afgrænsede
least-squares-idé som WM: match modelberegnet standardafvigelse mod historisk
timevis spredning fra de fem tidligere matchende ugedage. Brug samme
observationsoperator, datoer og cutoff for BM/WM. Det er en
**WM-kompatibel spredningskalibrering**, ikke likelihood-estimation af isoleret
processtøj: historisk spredning indeholder også variation i baseline, inputs og
målinger. Derfor er god tilpasning af spredningen alene ikke validering af SDE'en.
Ingen ekstra målevariansparameter i første pilot; en senere sådan udvidelse
kræver særskilt identifikations- og observationskontrakt.

Genbrug WM's kode til mål/figurer, hvor muligt: sigma-RMSE/MAE/bias som
diagnostik samt empirisk dækning, gennemsnitlig intervalbredde og intervalscore
på tidsmæssigt adskilte prognoser. Rapportér efter horisont og samlet med
samme nominelle niveau. Vis både alle tidspunkter og den fælles opgørelse uden
origin med P0=0, som WM gør. Vis punktfejl ved siden af usikkerhedsmålene.
WM/BM vurderes begge med initialisering og fit fra historik før origin;
aktuel WM-oracle-kode må ikke levere hovedsammenligningens prognosecenter.

G6 bruger en lille pilot med sigma=0-regression, variansens ikke-negativitet,
timer/dage-omregning og tidsforfining. Intervaldata kræver kontrol af den
udvidede kovariansberegning. Gem støjparametre, split, scope for usikkerheden,
approksimation, stopårsag og dækningstabeller separat fra deterministiske fits.
Hvis støjlaget ikke giver brugbare bånd inden for budgettet, afsluttes den
afgrænsede undersøgelse med dette negative resultat; der hævdes ikke kalibrerede
prognoseintervaller. G6 erstatter ikke F-sporets fulde stokastiske validering.

### G7: separat senere prisoptimering

G7 holder de fittede parametre/kurver faste og vælger eventuelt få priser:

$$\min_{u_0,\ldots,u_{m-1}}\sum_i[\mathcal H_i(D_\theta(u))-a_i]^2,\qquad 0\leq u_j\leq1.$$

Start med 2–6 prisblokke og et fælles fast trackingmål a. Hvis WM bruger en
endogen købskurve A(u), skal begge modellers objective og constraints først
gøres sammenlignelige. Trackinggevinst er ikke automatisk økonomisk profit.
BM's nonlinearitet består også her, så WM's konveksitetsbevis kan ikke kopieres.
Første kontrol er deterministisk; stokastisk risikokontrol er en særskilt senere
beslutning og er ikke nødvendig for G6's usikkerhedsleverance.

## 9. Kilder og første konkrete opgave

Planen er udarbejdet mod main `fe53cce858979afabeaa7ab1e36b9e44874b0fc5`.
Ved implementation læses nyeste AGENTS/F4/G og aktuelle WM-kode igen; bevar
andres samtidige arbejde og brug branch i eksisterende checkout, ikke nye worktrees.

- [BM-reference](BM_REFERENCE.md): ligninger, eksakt F1-løsning, feasibility og enheder.
- [BM-guide](BM_GUIDE.md): genbrugelige interfaces og eksisterende outputs.
- `src/model/wm_4.py` og `src/exe/run_model/wm_4/predict_weekly.py` på ovenstående
  commit: signed WM4 og deterministisk strukturelt fit/forecast.
- Johan's `WM4_notes(1).zip`, `main.tex`, dateret 27. september: deterministisk
  fit og oracle-initialisering; ældre fire-koefficient/[0,1]-parameterisering.
  Den skal ikke blandes med den nyere tre-koefficient/signed kode.
- Nocedal & Wright, Numerical Optimization, 2. udgave, kapitel 10: least-squares-
  struktur og lokale metoder; ingen påstand om global konveksitet af BM-fit.
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html):
  bounds/TRF, numerisk Jacobian og separate tællinger af derivativevalueringer.
  Kontrollér installeret SciPy før eventuel brug af nyere callback/workers-API.

**Næste opgave:** Kør den historiske januarkommando i G_JANUARY.md og gennemgå
fit, numerik, baselinegevinst, domæneproblemer og runtime.
Første målsætning er G0–G3 med faste lineære kurver. Start ikke G4 eller en
fuld månedskørsel, før denne kæde fungerer og dens målinger passer til budgettet.
G0–G3 er implementeret og syntetisk kontrolleret. Historisk januar-fit, G4–G7
og videnskabelig accept er ikke gennemført.
