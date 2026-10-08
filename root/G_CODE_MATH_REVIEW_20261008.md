# G: kode, matematik og plan efter genoptagelsen

8. oktober 2026. Kodegrundlag: main `d19259c9292bd9bc6b0dd2ff037973d5613ef8fc`.
Dette er et kode- og designreview, ikke en ny historisk validering eller
godkendelse af det store G7-døgnforsøg. Ingen modelkerner er ændret her.

**Konklusion:** G's centrale matematiske metoder er implementeret. Den største
resterende opgave er at gøre den valgte historiske model og en lille
filter→styringskæde til selvstændige, reproducerbare leverancer. Den seneste
historiske model er **S3 med lambda=0**, ikke automatisk den ældre lineære
januarmodel. Den store G7-forsøgsplan er mere omfattende end den grundversion,
Johan nu har valgt at få færdig inden for otte timers arbejde.

Aktiv fortsættelse: [otte-timers-planen](G_COMPLETION_PLAN_8H_20261008.md).
Den komplette genfundne matematik med udledninger og gamle resultater ligger i
[matematiksamlingen](G_MATHEMATICS_RECOVERY_20261008.md); dens
[originalindeks](G_MATHEMATICS_SOURCE_INDEX_20261008.md) bevarer kilderne.

## 1. Hvad gennemgangen dækker

Der er lavet en statisk importoversigt fra 25 indgange: de historiske
G-kørsler, G6/G8/G7-kørsler, døgnintegration samt energi- og budgethelpers.
Den finder **100 filer og 18.478 kildelinjer**, inklusive delte BM/WM-helpers
og validatorer. [Inventaret](g_code_review_inventory_20261008.json) indeholder
importrelationer, funktionernes placeringer og SHA256 for checkout- og
LF-normaliserede bytes. Dette er ikke en måling af testdækning eller et
udsagn om, at alle importerede funktioner kaldes i G.

De aktive matematiske forløb er læst og sammenholdt med noterne: input og
normalisering; baseline; deterministisk fit/forecast og S3; SDE/FP/FV;
Beta-projektion; Bayes og parameterbank; posterioroverførsel, prisobjective,
deterministisk reference, receipts og den nye døgnorkestrering. Importerede
alternativer som hele WM3, alle WM-fitvarianter, alle legacy-solvers og alle
validatorgrene er **ikke** påstået udtømmende linjerevideret. Dynamiske imports
og tredjepartsbibliotekernes interne implementation er heller ikke dækket af
AST-oversigten. Det er derfor ikke et formelt bevis for fejlfrihed.

39 udvalgte eksisterende tests er faktisk kørt og består; unittest rapporterer
0,664 s for selve testkørslen, eksklusive opstart/import. De kontrollerer
objective-algebra, Bayes, konservativ masseoverførsel, drift/diffusion og
verificeret genindlæsning af en M254-basis med platformafrunding. De genkører
hverken historiske fits, hele filterforløb eller det store kontrolforsøg.
Præcis testliste og begrænsning står i [kontrolkvitteringen](g_code_review_checks_20261008.json).

## 2. Planen, der faktisk var nået frem til

| Kilde | Beslutning | Betydning nu |
|---|---|---|
| [G.md](G.md), reviderede G0–G8-krav | G6b, G8 og G7 er obligatoriske; G6a alene er utilstrækkelig. G5's brede benchmark må ikke blokere kernen. | Bevares. En punktprognose alene afslutter ikke G. |
| [G4-januarpilot](G4_JANUARY_PILOT_REVIEW.md) | Lineær model beholdes i det lille januar/Februar-forsøg. | Gyldig lokal beslutning; den er ældre end S3-valget. |
| [G4, uger 13–18](G4_VALIDATION_WEEKS_13_18_REVIEW.md) | S3 uden straf vælges efter den på forhånd fastlagte regel. | Aktiv historisk modelopskrift; ingen ny lambda-/kurvesøgning. |
| [G8-M254-review](G8_M254_REFINEMENT_REVIEW.md) | Vejlederens normaliserede Bernstein/Beta-familie med M=254 er primær. | Bevares i den stokastiske demo; lokal 42-basis er referencehistorik. |
| [G7-design, 7. oktober](G7_INTEGRATED_EXPERIMENT_DESIGN.md) | 24h, fire prisblokke, to prissøgninger, tre policies, fem kontrolkontekster, EM og feedback. | Omfattende forsøgsdesign, nu udskudt fra grundversionen. |
| [G7-driverreview, 8. oktober](G7_FULL_DAY_DRIVER_REVIEW.md) | Driver implementeret med MAP-node og dens egen betingede middeltilstand. | Dette er den implementerede deterministiske reference. |
| [G7-completion-review](G7_DAY_COMPLETION_REVIEW.md) | Sammensat fixturekontrol består; numerisk døgnforsøg er ikke udført/valideret. | Må ikke ommærkes til et gennemført videnskabeligt forsøg. |
| Johans svar i den genoptagne chat | Højst ca. otte timers arbejde; historisk WM-sammenlignelig prognose og afgrænset filter→prisstyringsdemo. | Ny prioritering og leveranceafgrænsning; dokumenteret i den aktive plan. |

Den rejusterede plan er altså fundet. Den behøver en mindre afslutningsscope,
ikke en genopfindelse af G. Ældre noter bevares som historik, også når deres
"næste trin" nu er overhalet.

## 3. Matematikken i den aktive kode

### BM er en begrænset dynamisk korrektion til den fælles baseline

Med timer som tidsenhed bruger G

\[
\delta(x,u)=\tanh\!\left(\frac{k}{2}[f(x)+g(u)]\right),\qquad
D=B+\Xi\delta\begin{cases}1-B,&\delta\geq0,\\B,&\delta<0,\end{cases}
\]
\[
dX_t=\frac{D(X_t,u_t,B_t)-B_t}{C}\,dt
       +\sigma_xX_t(1-X_t)\,dW_t.
\]

Se [base/bm.py](../src/model/base/bm.py), [bm_1.py](../src/model/bm_1.py)
og [den deterministiske fitkernel](../src/method/bm_1_deterministic_fit.py).
SDE-støjen tilføjes af solveren; det gamle `stochastic`-argument til drift
gør ikke i sig selv en ODE stokastisk. Diffusionen har ingen ekstra C- eller
Xi-faktor: Xi=0 slukker driftkorrektionen, men ikke en positiv sigma_x.

Ved 0≤B,Xi≤1 bliver D i [0,1]. Med responsendepunkterne f(0)=g(0)=1 og
f(1)=g(1)=-1 peger drift indad ved x=0 og x=1, mens diffusionen er nul der.
Dette begrunder den kontinuerte models interval. Det garanterer ikke, at
en vilkårlig diskret EM-step bliver indenfor: de kontrollerede forsøg bruger
`boundary='none'` og registrerer rå overskridelser. Standard-EM-API'ets
refleksion må ikke forveksles med denne numeriske kontrol.

For sigma_x=0 fås balancen
\[
\int_0^T(D-B)\,dt=C[x(T)-x(0)].
\]
Det er forklaringen på koblingen mellem forbrugsflytning og sluttilstand.
For SDE'en indeholder den stivise balance også stokastikintegralet;
en deterministisk endpointbetingelse er ikke automatisk en stokastisk
energiligning. En konstant pris eller samme middelpris sikrer heller ikke
bevaret samlet forbrug.

Timeskift skal ske i både drift og diffusion: ved dage i stedet for timer er
C_day=C_hour/24 og sigma_day=sqrt(24)*sigma_hour. Prøvestøjen sigma_y
beskriver en måling og får ingen sqrt(dt)-faktor.

### Baseline, fit og hvorfor S3 er en anden prognoseprotokol

[WM4's baseline](../src/exe/run_model/wm_4/output_model.py) genbruges direkte:
kalenderharmoniske led, ugedag, temperatur og interaktioner; standardisering
på træningshistorikken; vægtet ridge-LS med itererede robuste vægte. Det er
en konkret IRLS-algoritme, ikke et bevis for global løsning af én fast
Huber-målfunktion. Baselineklipningen til [0,1] er en eksisterende fælles
inputpolicy og skal fremgå eksplicit. G's state eller rå observationer må
ikke klippes for at få et resultat til at passe.

G2/G4 minimerer målte residualer gennem hele den deterministiske fremadmodel.
Med residualer divideret med sqrt(N) er residualnormens kvadrat MSE;
SciPy's rapporterede `cost` er halvdelen. Positive C og k parameteriseres
logaritmisk. Manglende målinger fjernes fra tabet, mens modellen stadig
propagerer gennem timerne. RK4 segmenteres ved timepriser; forfiningschecket
holder det eksterne baselineforløb fast. Forskellen mellem timepunkt og
timegennemsnit er reel: sidstnævnte kan deterministisk beregnes fra
baselineintegral og C gange stateændringen.

S3 bruger lineær f og
\[
g_a(u)=1-2u+a h(u),\quad
h(u)=\begin{cases}2u,&u\leq1/2,\\2(1-u),&u>1/2,\end{cases}
\quad -1\leq a\leq1.
\]
Denne parameter bevarer endepunkter og monotonitet. En straf lambda*a²
er 3*lambda gange kvadreret L2-afstand fra den lineære kurve, fordi
integralet af h² er 1/3. Det seneste valg er lambda=0.

[fit_strategy](../src/method/bm_1_deterministic_strategy.py),
[BM1PooledInputs](../src/method/bm_1_deterministic_weekday.py) og
[week_job](../src/exe/run_model/bm_1/deterministic_validation.py) /
[checked_task](../src/exe/run_model/bm_1/deterministic_strategies.py)
viser den udvidede idé: ét fælles (C,k,Xi,a) fra 35 historiske dagsforløb.
Hvert forløb starter neutralt og får 24 timers warm-up. Dagene føjes ikke
sammen til en fysisk umulig tidslinje. Hver fremtidig ugedags x0 er medianen
af de fem tilsvarende historiske states efter warm-up. Alle syv fremtidige
dage fryses ved mandagens cutoff uden fremtidig demand.

Det er derfor **syv åbne dagsbaner frosset samtidig**, ikke en kontinuerlig
168h-statebane, og heller ikke daglig opdatering med ugens nye målinger.
Den ældre G3-januarvej bruger derimod den eksakte fit-sluttilstand til én
sammenhængende uge. Begge skal have eget protokolnavn.

Et godt fit identificerer ikke nødvendigvis parametrene. Hvis a≈1 og kun
u<1/2 ses, bliver g næsten flad dér; k kan ramme en grænse uden at data
bestemmer priselasticitet. Et numerisk Jacobian-rankcheck er lokal
følsomhedsdiagnostik, ikke global identifikation eller et fysisk bevis.

### FP, FV og Beta-mixture repræsenterer samme stokastiske model forskelligt

Med drift mu og diffusion b er FP-ligningen
\[
\partial_t p=-\partial_x(\mu p)+\tfrac12\partial_{xx}(b^2p).
\]
[FV-kernen](../src/solver/finite_volume.py) udvikler cellemasser, ikke
punktværdier af en tæthed. Upwind-advektion og forskelle af b²p giver en
generator Q med ikke-negative offdiagonaler og nul kolonnesummer.
Implicit Euler løser (I-hQ)m_next=m. Nul randflux kommer fra udeladte
overgange udenfor intervallet. Der foretages ingen efterfølgende
masseklipning/renormalisering i denne udvikling.

[Den aktive mixturekernel](../src/method/beta_mixture.py) bruger derimod
eksplicit konservativ FP-udvikling med CFL-grænse og projektion tilbage på
en fast Beta-familie efter hvert tidsstep. Derfor kan en massebevarende,
positiv beregning stadig være en dårlig tæthedsapproksimation. Hvert
projektionsfit og coarse/fine/FV-afvigelsen skal kontrolleres særskilt.
Legacy-metoderne i [fokker_planc.py](../src/method/fokker_planc.py) indeholder
også ældre normaliseringsveje; udsagnet om ingen reparation gælder den
deklarerede aktive G-vej, ikke enhver funktion med FP i navnet.

For M254 er de 255 komponenter Beta(j+1,M-j+1), j=0,...,M, integreret over
cellerne ved Beta-CDF'er. Med komponentmiddel mu_j fås
\[
\operatorname{Var}(X)=\frac{\mu(1-\mu)}{M+3}
 +\frac{M+2}{M+3}\operatorname{Var}_c(\mu_j).
\]
Variansgulvet forklarer, hvorfor en lille basis kan fungere i åben
propagation og alligevel fejle efter en skarp Bayes-opdatering. M=254 er et
begrundet, afprøvet udviklingsvalg, ikke bevist mindste nødvendige grad.

Projektionen løser min 1/2||Ac-b||² under c≥0 og sum(c)=1, med
A=sqrt(dx)*basis.T. Den aktive løser anvender SVD på simpleksens frie
tangentretning fremfor normal-ligninger, som kvadrerer konditionstallet.
Den uafhængige kontrol
\[
g=A^T(Ac-b),\qquad c^Tg-\min_jg_j
\]
giver for et gennemførligt c en øvre grænse for suboptimaliteten i dette
konvekse projektionsproblem. Den siger ikke, at familien repræsenterer den
sande tæthed præcist. En lille optimalitetsrest og en stor L1-projektionsfejl
kan derfor forekomme samtidig. Det er også tæthed/momenter, ikke enkelte
ustabile koefficienter, der bør gives fysisk fortolkning.

G6a's [lineære støjreference](../src/method/bm_1_linear_noise.py) løser en
lokal variansligning langs den deterministiske bane. Den omfatter hverken
fuld tæthed, parameterusikkerhed eller hele den ikke-lineære middelbane.
Det tidligere dårlige resultat bevares; den må ikke erstatte G6b/G8.

### Parameterbanken bevarer afhængigheden mellem parameter og tilstand

[Bayes-primitiverne](../src/params/bm_1.py),
[filterkernen](../src/method/bm_1_estimation.py) og
[G8-origin](../src/method/bm_1_g8.py) repræsenterer
\[
p(\rho,x\mid y_{1:n})=\sum_j w_j\,\delta_{\rho_j}(\rho)p_j(x),
\qquad \rho=1/C.
\]
Hver hypotese udvikler sin egen tæthed. For observation y er
L_j(x)=Normal(y;D_j(x,u,B),sigma_y²), Z_j=sum_i(m^-_{ji}L_{ji}),
m^+_{ji}=m^-_{ji}L_{ji}/Z_j og w^+_j proportional med w^-_j Z_j.
Der indgår ingen ekstra dx i evidencen, når m allerede er cellemasse.
Ved kvadratur repræsenterer L en gennemsnitlig likelihood indenfor cellen.

Log-vægte og logsumexp beskytter meget små hypoteser mod almindeligt
underflow. Manglende observation betyder propagation uden opdatering.
Parameter-evidence beregnes **før** posteriorprojektionen: en begrænset
repræsentation må ikke få lov at ændre det statistiske observationsbevis.
Banken må heller ikke reduceres til én marginal state, som derefter kopieres
til alle parametre; det ville kassere den lærte afhængighed.

Den nuværende G8-kørselsadapter er en fast syntetisk timepunktcase med fem
rho-knuder og kendte øvrige parametre. Den generelle bankkernel kan mere,
men gør ikke G's wrapper til en færdig historisk estimator. Et sandt rho,
som ligger på det valgte grid, og kendte støjparametre er en algoritmisk
kontrol, ikke bevis for historisk parameteridentifikation.

### Prisvalg og observation er to forskellige operationer

[OwnedBankPriceEvaluator](../src/method/bm_1_g7.py) starter hvert prisforslag
fra samme frosne betingede posteriorer og holder vægtene faste under
den åbne prognose. Målet er
\[
J(u)=\int\!\left[(\mathbb E D-A)^2
 +\sum_jw_j\operatorname{Var}(D\mid j)
 +\operatorname{Var}_w(\mathbb E[D\mid j])\right]dt.
\]
At styre D(E[X]) eller kun (E[D]-A)² ville være et andet problem. Koden
krydskontrollerer J mod summen af de betingede nodeomkostninger og variansen
mod totalvariansformlen. Midtpunktskvadraturen krydser ikke prisspring.

Den deterministiske reference bruger nu j*=argmax(w), C=1/rho_j* og
x0=E[X|j*], med sigma_x=0. Det er en erklæret plug-in-reference; den er
hverken fuld bankmiddelbane eller posterior-middelparameter. Dens valgte
pris skal også vurderes med samme stokastiske J som de øvrige priser.

Først efter den valgte pris er anvendt og en observation foreligger, kaldes
`assimilate`. Den tilhørende pris, observationstid, prisens venstre/højre
grænse, egne betingede laws og rå evidences følger opdateringen. Den nye
posterior er udgangspunkt for næste prisvalg. Det er den minimale feedback,
grundversionen skal demonstrere numerisk.

For EM-reference med fast antal baner pr. hypotese gælder
J_hat=sum_j w_j mean(L_j) og SE²=sum_j w_j² s_j²/N_j. Ved sammenligning
af policies bruges samme starttræk og Brownian-støj, og SE beregnes på de
parrede omkostningsforskelle. Groft Brownian-step er summen af fine steps.
Støj til observationen skal have separat tilfældighedsstrøm. Færre timer
eller prisforslag kan spare arbejde; ukoblet støj eller et andet objective
kan ikke anvendes som genvej til samme evidens.

## 4. Sammenlignelighed med WM

WM4 bruger F'=R, R'=r_F F+r_U U+r_R R og I=O+R. Dens native ur er dage,
og den signerede pris er U=2u-1. G's x har ikke samme dynamik eller fysiske
betydning som F. Udtrykket 1-2x kan bruges som neutral koordinatvisning,
men er ikke en dynamisk identifikation af de to modeller.

WM's QP fitter lineære overgangsresidualer på rekonstruerede F/R-states.
G's ikke-lineære fit minimerer demand-residualer gennem en forwardmodel.
QP'ens minimum kan derfor ikke sammenlignes direkte med G's fit-MSE.
Sammenlign faktiske fremadprognoser på samme observationer og information.

| Fælles kontrakt | Krav |
|---|---|
| Baseline | Samme fitted `statistical_output`, historik, cutoff, interpolering og normalisering; B=O. |
| Information | Samme tilgængelige priser/temperatur og observationsmaske. Realiserede fremtidsinputs mærkes som en betinget historisk prognose. |
| Protokol | S3's syv mandagsfrosne dagsbaner er egen protokol; sammenlign ikke umærket med dagligt opdateret WM eller kontinuerlig G3. |
| Enheder | Fælles rå demand og pris; u∈[0,1], WM-U=2u-1; eksplicit timer/dage. |
| Prognoseoutput | Samme timestamps, baseline, prediction og rå/normaliserede enheder; egne statefelter som separat diagnostik. |
| Scoring | MSE/RMSE/MAE/bias på identisk målemaske; bias=prediction−observation. Pooled RMSE og gennemsnitlig dags-RMSE mærkes forskelligt. |
| Kontrol | Samme exogene A, prisblokke, bounds, objective og eventuelle energikrav. WM's særskilte A(U)-indkøbsmål er en anden opgave. |
| Usikkerhed | Skeln proces-, parameter- og observationsusikkerhed; samme dækningsniveau/horisonter ved senere sammenligning. |

Den eksisterende [matched control-adapter](../src/method/bm_1_g7_matched.py)
og [energivariant](../src/method/bm_1_g7_energy.py) har allerede nødvendige
omregninger. WM's fireprisproblem er affint og kan løses som en konveks
box-QP; BM's tilsvarende søgning er ikke-lineær og giver lokal evidens.
Det er en legitim modelforskel, som skal rapporteres, ikke skjules ved
at kalde begge resultater globale optimummer.

## 5. Konkrete fund og prioritet

| Prioritet | Fund og belæg | Handling |
|---|---|---|
| Første leverance | S3 er valgt, men `deterministic_validation.run` er en fast seks-ugers sammenligningskørsel, og `checked_task` beregner stadig begge lambdaer. Dens baselinefiler gemmer metadata, ikke et komplet selvstændigt fitted baselineobjekt. | Lav en lille S3 fit/snapshot/predict-indgang med den allerede valgte lambda=0 og syv frosne startstates. Genbrug kernels; undgå at genkøre hele modelvalget. |
| Første leverance | Det generelle G3-snapshot understøtter den lineære model; G4-januarsnapshot kræver fast k fra referencen. Ingen af dem kan uden videre ommærkes til fælles S3-fit. | Versionsmærk S3-pakken særskilt; afvis forkert type. Bevar gamle indgange. |
| Før demo | 7. oktober-designets middelrho/middelstate er forskelligt fra 8. oktober-driverens MAP/conditional-mean. | Brug den implementerede MAP-definition i den aktive plan. Ældre design forbliver historik. |
| Før demo | G8-context, checkpointadapter og døgnrunner er bevidst bundet til analytisk syntetik. Fuld døgnkørsel kræver fire egne origins og 16 kompatible rater. | Afgræns et nyt kort numerisk demo-design. Dets beståelse må ikke sætte den gamle døgnaccept til true. |
| Brugbarhed | Den ældre lineære `run_benchmark` forsøger også WM-fit. Uden WM-origin bliver status `wm-rejected`, selv hvis BM kan levere. Det er en sammensat benchmarkstatus. | Hold G-resultat og WM-sammenligningsstatus adskilt i den nye brugerindgang; en WM-fejl skal ikke skjule en gyldig G-prognose. |
| Før lovet resume | `from_verified_bytes` kan bevare en certificeret afrundet M254-basis, men G8 `bank_filter(origin=...)` sammenligner med en frisk basis ved `array_equal`. | Portabilitetskant fundet ved læsning. Undersøg/ret kun hvis demoen bruger denne vej; den nu kørte reload-test er ikke en test af filter-resume. G7 kan bruge den ejede basis direkte. |
| Rapportering | Lokal `metrics` i weekday-runs gentager WM's formler, men har en anden API; de historiske protokoller er forskellige. | Brug/paritetstest fælles scorer i færdigindgangen og gem protokolnavn, maske og enheder. |
| Videnskabelig grænse | Timepunktapproksimation, fysiske enheder/aggregat, svag prisidentifikation og allerede inspicerede testuger. | Synligt i rapporten. En fungerende implementation er ikke automatisk fysisk kalibrering, kausal prisrespons eller blind sluttest. |

Der er ikke i dette review påvist en ny fejl, der kræver omskrivning af BM's
drift, den konservative FV-kernel eller Bayes-formlen. Det er ikke en garanti
for, at alle kodegrene er fejlfrie. De konkrete integrations- og
kontraktmangler ovenfor bør løses før yderligere forskningsudvidelser.

## 6. Hvad er bevaret, og hvad mangler stadig?

Kode, planer og refererede resultater er i Git. Recovery har verificeret
de to egne M254-origins ved time 12. De oprindelige to egne FV-filterorigins
er endnu ikke gendannet her; den gamle reference, som starter fra M254-law,
er fortsat en sådan reference. De tre ældre manglende projektionsaudits
for quadnode 0/1/2 er heller ikke genskabt. Se
[recovery-noten](G7_CHAT_RECOVERY_20261008.md).

Rå input-workbooks findes i Git-træet (`files/input/data/input.xlsx` og
`prices.xlsx`), men var ikke med i dette sparse review-checkout. Det er ikke
datatab. Originale historiske fit-/forecastmapper og alle tilhørende
Library-pakker er ikke dermed verificeret tilgængelige lokalt. Før historisk
replay skal den nødvendige konkrete pakke findes eller en ny afgrænset
kørsel etableres fra dokumenterede inputs. Kildetekst eller en summary-JSON
erstatter ikke et manglende fitted snapshot.

Der kan ikke sættes et troværdigt timetal på tabte samtaleformuleringer.
Der er ikke fundet dokumentation for, at den slettede chat fjernede den
implementerede G-kode. De identificerede evidenshuller skal fortsat stå åbne;
den nye korte demo må skabe ny, klart adskilt evidens uden at reparere
historikken ved ommærkning.
