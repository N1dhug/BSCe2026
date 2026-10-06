# G8 — eget Bernstein M254-filter: parret 513-forfining

Dette er én særskilt bestilt opfølgning på [257-filteret](G8_M254_FILTER_REVIEW.md).
Vejlederens normaliserede Bernstein/Beta-familie forbliver primær.
Legacy-M38/M78, statisk M254-review og den lokaliserede 42-leds-reference
bevares med deres oprindelige resultater. De genkøres ikke.

## Afgrænsning og kildegrundlag

Samme gemte syntetiske døgn, 24 timer, 22 punktobservationer, manglende målinger
ved 6 og 17 timer. Fem rho-noder: 0.125, 0.1875, 0.25, 0.375, 0.5.
Samme model, Beta(20,20)-prior, uniform parameterprior, sigma_y=.03,
GL3-observationskvadratur, højre prisgrænse, nul randflux og CFL=.4.
Ingen historiske data, nye f/g-fits, parametersearch eller prisoptimering.

Direkte vejledergrundlag: tavle `07-wSnLZuio-1-.jpg` for den normaliserede
Bernstein-familie; `05-Mkh9zqUP-1-.jpg` for SDE/FP-sammenhængen.
Mixture/PDE/positiv LS er beskrevet i
`MixtureApproximationsForTheFokkerPlanckEquation-1.pdf`, slides6 (eq4),
8 (eq7) og12 (active-set-henvisning til Nocedal/Wright16.3).
Den særlige ortogonale basis på slide9 gælder ikke denne overlappende basis.
Gaussisk målemodel er et eksplicit valg; `MSC_AGP_FINAL-1.pdf`, kap6.1.1,
eq6.2, trykt side90/PDF-side102, giver relevant observationsstøjgrundlag.
Specialets EKF er ikke vores diskrete fem-node-bank.

Gradforhøjelse, celleintegraler og Bayes-normalisering er standardidentiteter.
M254-valget, de fem noder, GL3, grids/tidsgrænser, tolerancer, budget og
checkpointformat er vores deklarerede udviklings-/numerikvalg. Ingen kilde
påstås at kræve netop disse tal. Det detaljerede eksisterende kildekort og
udledningen af M254/prioren findes i [257-reviewet](G8_M254_FILTER_REVIEW.md).

## Matematik og hvorfor dette er den relevante kontrol

Den eksisterende model er

\[
dX_t=v_\rho(X_t,t)dt+b(X_t)dW_t,\quad
v_\rho=\rho(D-B),\quad b(x)=.2x(1-x),\quad C=1/\rho.
\]

Dermed

\[
\partial_t p=-\partial_x(v_\rho p)+\tfrac12\partial_{xx}(b^2p).
\]

Vi bruger fortsat

\[
\ell_{m,254}(x)=255\binom{254}{m}x^m(1-x)^{254-m},\quad
p(x)=\sum_{m=0}^{254}c_m\ell_{m,254}(x),\quad c\ge0,\quad\mathbf1^Tc=1.
\]

Der er 255 basisfunktioner uanset antallet af FV-celler. Cellerne er
intervaller af tilstanden x, ikke tider, observationer eller parameterhypoteser.
På N ens celler med kanter e_i=i/N og dx=1/N anvendes

\[
B^{(N)}_{mi}=\frac1{dx}\int_{e_i}^{e_{i+1}}\ell_{m,254}(x)dx,\quad
P_i=dx\sum_m c_mB^{(N)}_{mi}.
\]

Beta-CDF/complementær CDF beregner integralerne. Ingen midpoint-approximation,
efterfølgende rækkerenormalisering eller FV-posteriorinitialisering.
Beta(20,20)-prioren gradforhøjes eksakt fra M38; de samme koefficienter er
prior på begge grids. Der er ingen initial LS-fit eller nye priorantagelser.

Efter hvert eksisterende eksplicit FP-substep og hver observeret
Bayes-opdatering løses den uændrede simplex-LS:

\[
c^*=\arg\min_{c\ge0,\;\mathbf1^Tc=1}\tfrac12\|Ac-q\|_2^2,
\quad A=\sqrt{dx}B^T,\quad q=\sqrt{dx}\,p^{raw}.
\]

Dette er diskret kvadratur af integralet af den kvadrerede tæthedsfejl.
LS-objektivet erstatter ikke vores direkte L1-, moment-, CDF-, kvantil- og
likelihoodgates. Positivitet/masse alene er heller ikke præcisionsaccept.
Overlappende Bernstein-kolonner kan være stærkt illkonditionerede; active-set
anvender eksisterende SVD-baserede face-solves og kontrolleret optimality-gap.
Der tilføjes hverken ridge-straf, ny solver eller clipping. Individuelle c_m
kan være følsomme, selv når den repræsenterede fordeling er stabil.

Hver rho-node fører sine egne koefficienter og sin egen conditional posterior
videre. En node får aldrig en posterior fra FV-referencen eller en anden node.
Likelihood/opdatering er fortsat

\[
L_n(x)=\frac1{\sqrt{2\pi}\sigma_y}
\exp\{-\tfrac12[(y_n-D(x,t_n))/\sigma_y]^2\},\quad
Z_{jn}=\int L_np^-_{jn}dx,
\]

\[
p^{raw,+}_{jn}=L_np^-_{jn}/Z_{jn},\quad
w^+_{jn}=w^-_{jn}Z_{jn}/\sum_k w^-_{kn}Z_{kn}.
\]

Kun tilstands-Bayes-resultatet projiceres; evidence og nodevægte beregnes fra
den uprojicerede opdatering. Ved manglende måling er L=1, Z=1, ingen
Bayes-projektion og uændrede nodevægte. Procesfremskrivning udføres stadig.

Vi ændrer samlet (N,h_max) fra (257,1/96) til (513,1/192).
Dette undersøger rum-/tidsforfining samlet, ikke separate konvergensordener.
Det faktiske substep er minimum af resterende tid, h_max og eksisterende
CFL-grænse .4/outgoing_rate. Diffusion kan derfor allerede kræve væsentligt
mindre skridt; en halveret h_max er ikke et løfte om præcis dobbelt antal skridt.

At forøge M samtidig ville ændre både repræsentation og diskretisering og gøre
årsagen til en forskel vanskeligere at identificere. Derfor fastholdes M254.
At genstarte hvert trin fra FV ville kun kontrollere lokale projektioner og
skjule akkumuleret filterfejl; derfor køres hele det fine døgn fra egen prior.

## Sammenligning og faste acceptkrav

Det fine filter sammenlignes med både eget gemte M254257-filter og gemt FV513.
Alle 24 timer og fem conditional love indgår. Marginalen er sum_j w_j p_j.
Forskellige grids sammenholdes via den eksisterende conservative massetransfer,
der antager konstant tæthed i hver kildecelle. Det er en deklareret diskret
sammenligning, ikke en eksakt fejl mod den ukendte kontinuerte FP-løsning.

Samme gates som tidligere: conditional/marginal L1 .05, mean .005,
variance .002, CDF .03, kvantil .02, log-evidence .05, nodevægt .03,
projektion L1 .05, massefejl1e-9 og checkpoint1e-10. Conditional-checket
omfatter L1/mean/log-evidence; marginalchecket omfatter prediction/posterior
state/demand-momenter, CDF'er/kvantiler, masse-L1, evidence og vægte.

Faktisk JSON-resume fra det fine filters egen time22 til24 sammenholdes med
den uafbrudte fine kørsel: conditional masser, Bernstein-koefficienter,
log-nodevægte, marginal og conditional log-evidence. Et roundtrip af JSON
alene ville ikke afprøve det rekursive warm-state-handoff.

## Runtime, implementering og provenance

Ny runner: `src/exe/run_model/bm_1/g8_bernstein_refine.py`.
De eksisterende G-adaptere, shared FP/mixture/LS/Bayes-kerner og gamle runners
er uændrede. Importerede kildefiler kontrolleres mod branchhead
`a368090719ddf97f0ae1c18443191acd66961f5c` før numerik, og inputs/source
SHA256 kontrolleres igen efter. Script/source/inputs/settings registreres
før kørslen i et nyt outputdirectory, der afviser overwrite.

Særskilt autoriseret loft900s inklusive to-timers pilot (højst60s), paired
gates, faktisk resume og30s eksportreserve. Piloten er de første to timer
af den faktiske kørsel; ingen gentagelse eller nulstilling. Admission kræver
elapsed +1.25*max(last2 hour times)*(remaining hours+2 resume hours)+10s <=870s.
Dette er et konservativt praktisk estimat, ikke et runtimebevis. POSIX alarm
rejser TimeoutError for at afbryde en lang LS uden solverens RuntimeError-fallback.
Atomiske JSON-checkpoints og ETA skrives ved hver fuldført time.

Det gamle300s-M254-forsøg og dets ikke-admitted refinement bevares.
Det tidligere foreslåede10min-G8-loft udvides eksplicit for netop denne
autoriserede repræsentationskontrol; panelets1800s samlede ramme bevares.
Tidligere målte G6b/G8-kørsler er cirka345s. Der reserveres konservativt300s
for de tidlige umålte interfaceforsøg/arkivarbejde, afrundet debit650s.
650+900=1550s efterlader250s ved worst-case-budget, uden ekstra pilotpulje.
Dette er budgetreservation, ikke et fuldt rekonstrueret faktisk runtime-ledger.
G7 er ikke admitted af dette overslag; dets egen pilot/jobgraf skal prissættes.

Ingen gamle tests gentages: numerisk fint replay, parrede gates, actual resume
og kildehashes dækker denne adapterkørsel. Ingen ny offentlig kernekontrakt.
Ny Python-runner syntakskontrolleres; ingen bred regression eller historisk fit.

## Resultater

**Complete og numerisk udviklingsaccept på denne case.** Alle24 timer/fem
noder er gennemført fra egen eksakte prior; alle udførte conditional- og
marginalgates mod eget257-filter og FV513 består. Ingen tolerancer blev ændret.

| Maksimal forskel over hele døgnet | Eget M254257 | FV513 | Gate |
| --- | ---: | ---: | ---: |
| Conditional masse-L1 | .0326092160 | .00354065417 | .05 |
| Conditional state-mean | .00192045323 | .000204425387 | .005 |
| Conditional log-evidence | .0203211339 | .00314723869 | .05 |
| Marginal posterior masse-L1 | .0128255026 | .00108110760 | .05 |
| Marginal log-evidence | .00819834712 | .000853472196 | .05 |
| Nodevægt | .00165130952 | .000265417496 | .03 |

Alle marginale prediction-/posterior-state/demand-momenter, CDF'er og
kvantiler består også. Mod FV513 er prediction-demand-mean1.21030e-5,
posterior-demand-mean1.33274e-5, prediction-demand-CDF.000296492 og
prediction-demand-kvantil5.92986e-5. Den fine conditional-FV513-L1 er
cirka8.35 gange mindre end det tidligere257-filters.0295811; dette er en
forskel til en gemt diskret reference, ikke en bevist FP-konvergensorden.

| Bevarings-/projektionskontrol | Målt maksimum/minimum |
| --- | ---: |
| Marginal massefejl | 1.77636e-15 |
| Conditional massefejl | 6.21725e-15 |
| Mindste conditional cellemasse | 4.43723e-247 |
| Substep-/Bayes-projektion masse-L1 | .000971943317 |
| Skaleret optimality-gap | 9.99998e-12 |
| Koefficienternes massefejl | 6.88338e-15 |
| Manglende måling: posterior-vs-prediction | 0 |
| Manglende måling: log-nodevægtændring | 0 |

200940 completed/checked LS-solves, 168785 face-factorizations og200326
cachehits. Active-set-wall179.393s; alle255 funktioner forblev tilgængelige.
Den fine kørsel har cirka3.91 gange det grove filters51420 LS-solves,
foreneligt med diffusionens omtrentlige dx^2-begrænsning. Faktisk total
filterruntime er cirka2.63 gange større; dette er ikke en gridfri eller
dokumenteret hurtigere metode end FV. Shared solvere er uændrede.

To-timerspilot34.15s; admissionestimat638.52s for resten inklusive tail/reserve,
inden for870s. Filter309.742953s; actual JSON-resume22→24 tog17.326955s;
hele invocation inklusive registrering/output/comparisons327.828576s.
Resume gav eksakt0 forskel i conditional masser, koefficienter, log-nodevægte,
marginal log-evidence og conditional log-evidence. Begge sidste timer er
observerede, så evidencesammenligningen er finite.36 importerede kodefiler
matcher branchens Gitblobs; alle37 registrerede sourcehashes (inklusive
runner og importernes deduplikerede stier) og alle fire inputhashes er uændrede.

`full_filter_accept=true` og `mixture_to_G7_ready=true` betegner den
afgrænsede repræsentations-/rekursionsaccept. **G7-backend og prisstyring er
stadig ikke implementeret eller kørt.** Historisk accept og parameterrecovery
forbliver false. Basis/graden er udviklet på den eksponerede case; der er
hverken uafhængig validering, universel basisgaranti eller dokumenteret
168-timers-runtime. Den kendte dårlige Bernstein-konditionering forsvinder
ikke af et positivt replay; koefficienterne fortolkes ikke som identificerede
fysiske parametre. Det fine typed origin er reference til G7-interfacekontrol;
eget257-grid består også den deklarerede paired gate og kan indgå i en
særskilt runtimepilot. Prisforward kræver stadig sin egen numerikkontrol.

Rå24 timeorigins, den faktiske resumed origin, fixture, frozen references,
SHA256-/Git-audit og de præcise importerede sourcefiler bevares i
`G8_M254_513_forfining_raaresultater_20261006.zip`.
Den oprindelige summary/registration og gamle arkiver ændres ikke.
Reporesultat: `files/output/bm_1/runs/g8_m254_refinement_20261006.json`.
Ny runner består AST-syntakskontrol; ingen gamle numeriske tests gentaget.

## Næste interface: hvad dette review ikke implementerer

`weighted_control_evaluation` i G-adapteren fremskriver allerede nodernes
ejede cellelove med FV. Det er en eksplicit referenceevaluation, også når
origin kom fra mixture. Det må ikke ommærkes til mixture-prisstyring.
Den eksisterende `DynamicMixturePriceEvaluator` bruger legacy midpoint-basis
og initial LS-projektion. Den er derfor heller ikke automatisk et M254-handoff.

Et senere lille G7-interface skal bruge originens præcise celleintegrerede
basis og hver nodes egne c_j som start for den eksisterende
`evolve_projected_density`. Legacy-evaluatoren og de fælles kerner bevares.
For et fast kandidatprisforløb U fryses cutoff-vægtene w_j og de betingede
love fremskrives under deres respektive rho_j. Fremtidige observationer
indgår ikke i denne open-loop objective. Det eksisterende trackingmål er

\[
J(U)=\sum_j w_j\int_{t_0}^{t_0+H}
\int_0^1[D(x,s;U)-A(s)]^2p_j^U(x,s)\,dx\,ds.
\]

Det svarer til integreret kvadreret bias plus total demandvarians, der
indeholder både within-node og between-node varians. Et gennemsnitligt rho
eller alene marginalens gennemsnit bevarer ikke dette generelt, fordi
dynamikken og demandoperatoren er ikke-lineære. Ingen ny prisstraf er valgt
her; eventuelle eksisterende terminal-/smoothness-led skal deklareres separat.

Før nogen optimizer køres skal den konkrete G7-jobgraf budgetteres ud fra
en kort own-origin-prisforwardpilot. Filterets24h-runtime er hverken målt
objective-runtime eller budget for100 optimizerkald. Hele denne G7-opgave
og dens policy-/EM-krydskontrol er stadig uafsluttet.
