# G7: samlet døgnstyring implementeret, endnu ikke kørt

8. oktober 2026. Kodelevering oven på den mergede feedbackintegration
(PR119, main7586a9d2). Ingen ny model-, EM-, likelihood-, projektions-
eller stubkørsel. De originale numeriske kerner, M254-basis, F og WM bevares.

## Det konkrete resultat

Den nye driver forbinder nu alle trin i det eksisterende v2-design:

1. Egne G8-posteriorer ved time12 for M254257, M254513, FV257 og FV513
   indlæses og kontrolleres før søgning. Modeller/cutoff/rho-bank skal være
   ens; grids har egne statefordelinger og vægte, som skal bestå de
   eksisterende input-sammenligningsgates. Ingen posterior overtages fra
   en anden backend eller et senere tidspunkt.
2. Den native deterministiske ODE og den fulde M254257-bank optimeres
   separat. Hver får to bounded Powell-starts, højst24 objectivekald
   per start,48 per policy,96 samlet og fælles300s searchgrænse.
   Alle fem conditional forwards tæller i én bankobjective.
3. Neutralpris og de to fundne prisplaner fastfryses. Alle tre vurderes
   på alle fire backend/grids samt en M254513-costpartitionforfining.
   Der foretages ingen prisgenoptimering på de fine grids eller på FV.
4. Tre parrede EM-referencejobs køres fra den fine M254-prior med
   256 paths per hypotese. Coarse/fine EM deler startstates og Wienertræk,
   og alle prisplaner bruger præcis de samme draws. Raw boundary
   violations stopper forsøget; ingen clip/reflect accepteres.
5. Observationen ved36 dannes fra den på forhånd udpegede stochastic
   fine-EM-hypotese2/path0 plus en separat observationsstøjstrøm.
   Før dette punkt har søgningen ingen adgang til denne observation.
6. De allerede gemte auditerede døgnforwards leveres til den uændrede
   feedbackfase. Fire egne posteriors opdateres, tre coarse M254-prisvalg
   vurderes, og den valgte plan fremføres til38 på alle fire egne laws.
7. Saved update-evidences/laws og næste forwards sammenlignes. Næste
   forwardresultater genlæses og reduceres fra deres durable receipts;
   kontrolreviewet udfører ikke ekstra propagation.

Den fælles validation/EM/feedbackfase har300s grænse. En ekstra adapter
omkring store.node afviser et helt nyt job, hvis dets reservation ikke
kan være inden for den aktuelle fase; native driftguards og parenttimeout
bevares. Et stop gemmer day_stop.json. En ny invocation kræver eksplicit
recovery-/budgetbeslutning; ingen automatisk retry eller refund.

| Fil | Rolle |
|---|---|
| g7/day_contract.py | Stdlib-kontrol af implementeret design, numerik, feedback og tolerancer før native import |
| g7/day_control.py | Deterministisk/M254-search, durable EM, kontrolgraf og faktisk feedbackkobling |
| g7/day_review.py | Native forwardreview, posterior/evidencekontrol og stratificeret/parret MC-algebra |
| g7/launch_day_control.py | Separate registrerede inputs, helgrafadmission, én durable invocation og parenttimeout |

Det eksisterende integration_design.json er et historisk designproposal
med et tidligere debit-snapshot; det ændres ikke til en launchregistration.
Launcheren bruger altid en separat pinned previous_budget.json og
registration.json. Et tomt registration-template i denne levering er
ikke autoriseret og kan ikke åbne workeren.

## Styringsproblemet og dets to søgninger

Beslutningen er fire normaliserede priser
$u=(u_1,u_2,u_3,u_4)\in[0,1]^4$ på intervallerne
$[12,18),[18,24),[24,30),[30,36]$ med den eksisterende price-side-kontrakt.
Målprofilen er den faste $A(t)=0.5$; baseline er den eksisterende
$B(t)=0.5+0.1\sin(2\pi t/24)$. Fysiske/modelparametre og responses fittes
ikke i dette trin. Tid er timer, demand er normaliseret, og J har enhed
normaliseret demand i anden gange timer.

For hypotese $j$ betegner $\rho_j$ parameteren, $w_j$ dens posteriorvægt
ved cutoff12 og $p_j(t,x;u)$ dens betingede statefordeling. Native BM1
bestemmer drift $a_j$, diffusion $b_j$ og demand $D_j(t,x,u)$:

$$
dX_t=a_j(t,X_t,u)\,dt+b_j(X_t)\,dW_t.
$$

Den deterministiske plug-in-reference vælger
$j_*=\arg\max_jw_j$ (første hypotese ved ties), starter fra
$x_{12}=\sum_\ell m_{j_*,\ell}x_\ell$ på eget coarse-grid og løser den
uændrede ODE $\dot x=a_{j_*}(t,x,u)$. Den søger efter en lav værdi af

$$
J_{\mathrm{det}}(u)=
\int_{12}^{36}\bigl(D_{j_*}(t,x_t(u),u)-A(t)\bigr)^2\,dt.
$$

Dette er en eksplicit plug-in-approksimation. En nonlinear ODE startet
fra middelstate er generelt ikke lig den stokastiske middeltrajektorie.
Den tidligere bevaret-forbrug-variant ændres ikke; det aktuelle
stokastiske design tilføjer hverken energilighed, terminalstraf eller
prisglatningsstraf.

M254-søgningen bruger hele den faste posteriorbank:

$$
J(u)=\int_{12}^{36}\sum_j w_j
\int_0^1\bigl(D_j(t,x,u)-A(t)\bigr)^2p_j(t,x;u)\,dx\,dt.
$$

Sæt $\mu_j=E[D\mid j]$, $v_j=\operatorname{Var}(D\mid j)$ og
$\mu=\sum_jw_j\mu_j$. Ved at udvide kvadratet omkring $\mu$ fås

$$
E[(D-A)^2]=(\mu-A)^2+\operatorname{Var}(D),
$$

og totalvariansidentiteten giver

$$
\operatorname{Var}(D)=\sum_jw_jv_j+
\sum_jw_j(\mu_j-\mu)^2.
$$

Dermed rapporteres J som sum af bias, within-parameter-varians og
between-parameter-varians fra de samme forwards. Middelfejlen alene
bevares som beskrivende kontrolmål; den nye driver laver ikke en tredje
separat optimering af bankens bias-only-mål. En korrekt sammenligning
vurderer også den deterministisk fundne pris under det fulde J.

Vægtene er faste på det åbne døgn. De opdateres først ved observationen36.
Dette er posteriorbaseret open-loop single shooting efterfulgt af et
feedbacktrin; det er ikke en løsning af dual control eller en Bellman-PDE.

## M254, FV og quadrature

Vejlederens primære familie er

$$
b_{i,254}(x)=255{254\choose i}x^i(1-x)^{254-i}
=\operatorname{Beta}(x;i+1,255-i),\qquad i=0,\ldots,254.
$$

$p_j\simeq\sum_i c_{ji}b_{i,254}$ med $c_{ji}\ge0$ og
$\sum_i c_{ji}=1$. For cellernes integralmatrix Q udføres den eksisterende
projektion

$$
c_j^\star=\arg\min_{c\ge0,\;\mathbf1^\top c=1}
\tfrac12\|Qc-\widetilde m_j\|_2^2.
$$

$Q^\top Q$ er positiv semidefinit, så dette er et konvekst LS-problem
over simplekset. Det beviser ikke, at $u\mapsto J(u)$ er konveks.
De native transport-/LS-kernels og legacy-basis ændres ikke.
Per-hypoteseprojection-L1 gemmes inde i forwardreceipt før completion.
L1=.05 er en eksisterende numerisk gate, ikke et kontinuert fejlbevis.

FV er en reference med egen posterior, uden mixture-LS-projektion.
Coarse257/fine513 bruger transportsteps1/96 og1/192 time; begge bruger
costpartition1/96. Ekstra M254513-partition1/192 ændrer også transportens
output/pausepunkter. Derfor kaldes det partitionforfining og ikke et
bevis på isoleret tidskvadraturfejl.

Native compare_forward genbruges for terminalmomenter, CDF/kvantiler,
conditional/marginal L1 og J-gaten
$|J_a-J_b|\le\max(.001,.02|J_b|)$.
Der suppleres med eksplicit bankvægtkontrol. Ingen gates lempes.

## Hvorfor bounded Powell

Vi genbruger den eksisterende scipy.optimize/Powell-kode. Priserne har
kun fire komponenter, og hvert objectivekald kræver en dyr fremføring
med aktive LS-begrænsninger. Vi har ikke indført eller valideret
kontrolgradienter. To faste starts og et lille callcap giver et eksplicit
afgrænset søgeforsøg uden et nyt stort grid eller en ny solverkerne.

Et maxfev-stop er ikke et konvergensbevis. Driveren gemmer alle complete
kandidater, startstatus og det bedste complete valg, også hvis en start
ikke konvergerer. Det bedste observerede valg kan komme fra en sådan
start; selected_start_converged rapporteres særskilt fra
any_start_converged. Kernel-, journal-, source- og deadlinefejl propagates;
kun native Powells specifikke all-starts-nonconvergence registreres som
searchstatus. Ingen global optimalitet eller garanteret forbedring
påstås. Neutralplanen kan fortsat være bedst.

## Parret EM og hvad standardfejlen dækker

Fine M254-prior sampled kontinuert: vælg komponent
$I\sim\operatorname{Categorical}(c_{j,0},\ldots,c_{j,254})$,
derefter $X_{12}\sim\operatorname{Beta}(I+1,255-I)$.
Alle policies får samme draws for samme j. Role1 er startstate,
role2 er Brownian og role3 er observationsstøj; hypotese-strømme adskilles
med SeedSequence(20261008, spawn_key=(role,j)). Observationen bruger role3
med sin særskilte key. Numpyversion og regeneration recipe gemmes.

Euler–Maruyama genbruges uændret:

$$
X_{n+1}=X_n+a_j(t_n,X_n,u)\Delta t+
b_j(X_n)\Delta W_n,\qquad \Delta W_n\sim N(0,\Delta t).
$$

Fine $\Delta t=1/384$, coarse1/192 time. Coarseincrements er summen af to
fineincrements, så deres varians er2/384=1/192 og coupling bevares.
Begge niveauer vurderer pathcost på samme1/96-midtpunkter og vægte:
$Q_{jr}(u)=\sum_n h_n(D_{jr}(t_n,u)-A(t_n))^2$.
Der gemmes coarse/fine pathcosts, startstates, inputhash, tidsgrid,
seeds/runtime og det forudbestemte fine truth-demand.
Inputhash bruger eksplicit float64 little-endian arraybytes frem for
millioner af midlertidige JSON-skalarelementer.

For R=256 fås

$$
\widehat J=\sum_jw_j\overline Q_j,\qquad
\widehat{\operatorname{SE}}(\widehat J)=
\sqrt{\sum_j\frac{w_j^2s_j^2}{R}}.
$$

Formlen følger af uafhængige conditional strata. Vægtene er faste; paths
må ikke ommærkes til R gange fem iid draws fra marginalfordelingen.

Parret forbedring over neutral $d_{jr}=Q_{jr}(u_0)-Q_{jr}(u)$ giver

$$
\widehat\Delta=\sum_jw_j\overline d_j,\qquad
\widehat{\operatorname{SE}}(\widehat\Delta)=
\sqrt{\sum_j\frac{w_j^2s_{d,j}^2}{R}}.
$$

Positiv Delta betyder lavere candidatecost. Inputhashes kontrolleres før
parring. Covariance er med i $s_{d,j}^2$, hvilket kan reducere støjen i
prisforskellen. Det garanterer ikke, at SE altid bliver mindre.

Fine EM/FV-costs skal bestå max(.001,4SE), coupled coarse/fine-J-difference
.001 og nul rawboundaryviolations. Det er de eksisterende numeriske
heuristikker, ikke et formelt dækningsbevis. SE beskriver MC-sampling,
ikke EM-tidsbias, FP-/LS-fejl, bankdiskretisering, optimeringsfejl eller
model-/parameterusikkerheden selv. EM starter fra kontinuert M254,
FV fra sin egen celleposterior; forskellen kombinerer derfor også
prior-/backendapproksimation. Own-origin-gates gør ikke priors eksakt ens.

## Feedback

Den uændrede feedbackadapter beregner rå evidences før stateprojektion:

$$
Z_j=\int L_j(y\mid x)p_j^-(x)dx,\qquad
w_j^+=\frac{w_j^-Z_j}{\sum_kw_k^-Z_k},\qquad
p_j^{raw,+}=\frac{L_j(y\mid x)p_j^-}{Z_j}.
$$

Hver backend beholder sin egen projected/raw posterior. Samme y36
anvendes til alle fire. State-laws overføres konservativt alene til
sammenligning, aldrig som en ny egen initialtilstand. Conditional og
samlet logevidence sammenlignes sammen med vægte og laws.
Derefter bruges egne $p_j^+,w_j^+$ fra36 til38.
Der foretages ikke en ny EM-undersøgelse af denne2h-continuation;
den registrerede v2-graf kræver her M254/FV-krydsreview.

## Launcher, budget og kontrolstatus

Default CLI udfører kun stdlib-preflight; --execute kræver
g7-full-day-registration-v1, execution_authorized=true og separate
hashpinned design/rates/priorbudget/origins/source/runtime. Der findes
ingen udfyldt autoriseret registration i denne levering.

Alle16 ratefamilier skal være whole-job measured/rawreviewed under den
samme problem/source/origin/runtime-context. Ukendte rates er null.
Ingen2h→24h-skalering, ingen implicit budgetforøgelse. Original41-source
manifest bevares; alle loaded nye helpers indgår særskilt i nye hashes.

Før workerstart fsynces en eksklusiv one-invocation-claim med hele scopecap.
Workers kan kun startes under denne claim; parenttimeout og BLAS1 anvendes.
Parent debit er prior+max(scopecap,actualwall) på alle outcomes. Native
ledger står for intern jobprovenance og må ikke lægges til parentcharge
én gang mere. Hele scope beholdes ved loss/stop; manglende receipts må
ikke stille replayes. Recovery kræver et nyt konkret registreret scope.

Denne levering er statisk læst og AST-parset uden imports/modelkørsel.
GitHub PR-CI skal syntakskontrollere de fire changed Python-filer.
Ingen runtime-/stubtest, timeout-integrationstest eller numerisk
fuldgrafvalidering er udført. Det er en konkret begrænsning: før et
dyrt forsøgsrun bør én fokuseret stubkontrol dække hele orchestrationens
receipt-/callbackforløb under et separat bounded checks-scope.
Tidligere feedbackadapter-stubresultater gælder deres egen frosne source;
de ommærkes ikke til en test af denne nye driver.

G-debit er uændret1851.3372718589885/1910s, rest58.662728141011485s.
Ingen nye science/stubcharges. Statisk redigering/syntax er ikke debiteret
som et videnskabeligt beregningsforsøg. Hele døgnforsøget er ikke kørt,
ikke budgetadmitteret og ikke scientific/historically accepted.
Gamle quad0/1/2-audits forbliver manglende.

Næste trin: én målrettet fuldgraf-stubkontrol, derefter konkrete whole-job
rates eller særskilt bounded måleregistrering og samlet budgetbeslutning;
så immutabel numerisk præregistrering og det faktiske døgnforsøg.
Der kræves ingen lokal kørsel fra Johan i denne levering.
Direkte BM/WM-sammenligning venter fortsat til BM er færdig.
