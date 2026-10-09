# G: prisstyring, bevaret energi og stokastisk evidens

<a id="control-checkpoints"></a>
<a id="g7-seed-repair-review"></a>
<a id="g7-receipt-recovery-review"></a>
<a id="g7-per-hypothesis-audit-review"></a>
<a id="g7-owned-handoff-review"></a>
<a id="g7-node-runtime-review"></a>
<a id="g7-node-job-review"></a>
<a id="g7-matched-control-review"></a>
<a id="g7-m254-runtime-job-review"></a>
<a id="g7-integrated-experiment-design"></a>
<a id="g7-full-day-driver-review"></a>
<a id="g7-feedback-integration-review"></a>
<a id="g7-energy-neutral-review"></a>
<a id="g7-deterministic-control-review"></a>
<a id="g7-day-completion-review"></a>
<a id="g7-day-check-and-rate-review"></a>
<a id="g7-complete-quad-recovery-review"></a>
<a id="g7-bounded-price-quadrature-design"></a>
<a id="g7-bounded-launcher-review"></a>
<a id="g7-auxiliary-runtime-review"></a>
<a id="g7-auxiliary-measurement-review"></a>
<a id="g7-auxiliary-integration-design"></a>
Dette er en samlet faglig fremstilling af bevaret udviklingsevidens. Tidligere arbejdsordrer og budgetter er historik, ikke nye kørselsordrer. Aktuel brug, accept og opgaver står i [BM-guiden](../../../root/BM_GUIDE.md) og [status](../../../docs/bm/STATUS.txt). Originaler, hashes og genskabelse: [arkivindeks](../INDEX.md).

Historiske statusord som “åben”, “endnu ikke” og “Remaining scope” gælder den pågældende delundersøgelse 5.–8. oktober 2026. De beskriver ikke automatisk den aktuelle F4-kommando eller senere G-timeleverance.

## 10. G7: deterministisk prisstyring

Fire holdte priser vælges på seks-timersblokke:

$$\min_{u\in[0,1]^4}J_{\mathrm{det}}(u),\qquad
J_{\mathrm{det}}=\int_0^{24}[D(x^u(t),u(t))-A(t)]^2dt.$$

Single shooting betyder, at hver prisplan integreres fra samme frosne startstate; en tidligere kandidats slutstate bliver aldrig den næstes start. Fysiske parametre og responskurver genfittes ikke til kontrolmålet.

Midpoint-cost og residualer er

$$J_h=\sum_qh_q(D_q-A_q)^2=r^Tr,\qquad r_q=\sqrt{h_q}(D_q-A_q).$$

Pris-/inputbrud splitter quadraturen. $\sqrt{h_q}$ er nødvendigt for de rigtige tidsvægte. TRF minimerer $F=J_h/2$; rapporteret J er fortsat $J_h$. Med residual-Jacobian $R$ er

$$\nabla F=R^Tr,\qquad
\nabla^2F=R^TR+\sum_qr_q\nabla^2r_q.$$

Gauss–Newton-leddet er positivt semidefinit, men den ekstra term og dynamikken gør ikke automatisk BM-opgaven konveks. To starter med næsten ens J er lokal numerisk evidens, ikke global optimalitet.

Den lille syntetiske deterministiske reference gav neutral J 0.12 og bedste fundne J cirka 0.0209268019845 med priser cirka $(0.686689795,0.797805159,0.503310238,0.392193631)$, svarende til 82.56 % lavere J i netop denne opgave. Det er tracking, ikke elregning, profit eller kausal effekt.

### Historisk matched kontrakt og bevaret forbrug

Et særskilt tidligere januar-origin-forsøg bruger frosne fits og fladt mål

$$A=\bar B=\frac1T\int_0^TB(t)dt=0.39900697422105735.$$

Baselineintegralet beregnes på dens egne lineære knuder. Valget bevarer målprofilens total, men **tvinger ikke kandidatens demand-total til at være den samme**. Derfor blev en særskilt lighed indført:

$$E(u)=\int_0^{24}[D(t;u)-B(t)]dt=0.$$

BM-balancen giver $E=C[x(24)-x(0)]$, så kontinuert energineutralitet svarer til uændret slut-SoC. Et kvadratisk strafled $\lambda E^2$ ville ændre tradeoff og kræve en ny vilkårlig vægt; ligheden udtrykker det bestilte krav direkte. SLSQP blev brugt til BM's fire variable og ene ikke-lineære lighed. De bedste gennemførte feasible kandidater er lokale resultater.

For den historiske WM-reference er time-dynamikken $\dot F=R/24$, $I=B+R$, så $E=24[F(24)-F(0)]$. Original Euler bruger venstre R, mens cost bruger midpoint-R. På uniformt timegrid h teleskoperer det til

$$E_{h,\mathrm{WM}}=24(F_N-F_0)+\frac h2(R_N-R_0).$$

Derfor håndhæves samme diskrete $E_h$ som i rapporteret cost; $F_N=F_0$ alene er ikke den samme diskrete betingelse. Der kræves ikke også $R_N=R_0$.

Frosset WM er affint i u: residual $r=c+Hu$, $E=e_0+a^Tu$. Det giver en konveks QP med $Q=2H^TH$. Alle $3^4=81$ frie/nedre/øvre faces blev undersøgt algebraisk; KKT-kontroller og redundans af pathbounds gav en global løsning af **den diskrete WM-opgave**. Dette argument kan ikke overføres til nonlinear BM eller den kontinuerte limit uden yderligere kontrol.

| Historisk frosset fit | J uden lighed | J med bevaret forbrug |
|---|---:|---:|
| Lineær BM | 0.268818599455 | 0.275408887552 |
| G4 BM | 0.273316332715 | 0.276532279602 |
| WM4 | 0.299704871421 | 0.303629020154 |

BM's fine neutrale energirester var cirka $-4.60\cdot10^{-10}$ og $-3.88\cdot10^{-10}$; WM's $1.47\cdot10^{-6}$, under $10^{-5}$-gaten. De nye constraints var dyrere i J, som forventeligt ved indsnævring af feasible muligheder. Dette er bevaret historisk kontrol, ikke den aktuelle endelige BM/WM-sammenligning.

Kilder: [deterministisk kontrol](control-and-evidence.md#g7-deterministic-control-review), [matched kontrakt](control-and-evidence.md#g7-matched-control-review), [energineutral udledning og KKT](control-and-evidence.md#g7-energy-neutral-review). Implementation: [deterministisk evaluator](../../../src/method/bm_1_g7_deterministic.py), [energy](../../../src/method/bm_1_g7_energy.py).

## 11. G7: stokastisk mål, korte resultater og feedback

For egne betingede posteriorer og faste cutoff-vægte bruges

$$J(u)=\int\sum_jw_j\int_0^1[D_j(t,x,u)-A(t)]^2p_j(t,x;u)dx\,dt.$$

Udvid kvadratet med $\mu_j=E_j[D]$, $v_j=\operatorname{Var}_j(D)$, $\mu=\sum_jw_j\mu_j$:

$$J=\int(\mu-A)^2dt+
\int\sum_jw_jv_jdt+
\int\sum_jw_j(\mu_j-\mu)^2dt.$$

Krydsleddet forsvinder, fordi $\sum_jw_j(\mu_j-\mu)=0$. Middelfejl, within- og between-varians er derfor en eksakt dekomposition af ét objective, ikke tre valgte straffe. En separat deterministisk ODE ved middelstate/MAP-node er generelt ikke bankens mean; den fundne deterministiske pris skal også scores under det fulde stokastiske J ved en fair kontrolsammenligning.

### Kort tre-plan-forsøg, time 12→14

De tre på forhånd valgte planer er $u(\delta)=(0.5+\delta,0.5-\delta)$ med $\delta=0,-0.2,+0.2$. Lige middelpris betyder ikke bevaret forbrug. Coarse-grid valgte laveste komplette bank-J, med tie-rækkefølge neutral først. Fine/FV/ny partition validerede valget og blev ikke brugt til efterfølgende reselektion.

Neutral vandt: coarse J 0.003741963573462555 mod 0.017303375697677884 for $(0.3,0.7)$ og 0.010655853331904826 for $(0.7,0.3)$. Neutral fine M254-J var 0.0037155174638558177, FV-J 0.0037153098107086916 og den finere partition-J 0.003715523422208866. De tilsvarende forward-pargates bestod. Tilts blev ikke fine-valideret; resultatet er en coarse-rangering af tre planer, ikke et kontinuerligt optimum.

Ny partition halverede objective-step fra 1/96 til 1/192 time, med 192→384 forskellige midtpunkter. Den ændrede samtidig transportens output-/pausepunkter. Derfor er forsøget partitionfølsomhed, ikke et isoleret quadraturfejl- eller Richardsoncertifikat. Den fine partitions J blev rekonstrueret som

$$0.0022613178834201752+0.0013782185162290082
+0.00007598702255962573=0.003715523422208866.$$

De gamle per-node-projektionsmaxima for quadnode 0/1/2 var gået tabt før den aktuelle chatsletning. Komplette fordelingsbaner kan ikke i sig selv genskabe samtlige mellemprojektioners maxima. Derfor blev den samlede numerical/scientific accept ikke erklæret, selv om bank- og terminalpargates bestod.

### EM-sampling, uafhængighed og parret SE

Fra den repræsenterede M254-law trækkes først $I\sim\mathrm{Categorical}(c_j)$ og dernæst $X_{12}\mid I=i\sim\mathrm{Beta}(i+1,255-i)$. Dette sampler den kontinuerte mixture, ikke FV-cellemasser som koefficienter. Hver node har 256 baner. Initialdraws, Brownsk støj og observationsstøj har separate role/node-RNG-identiteter. Den konkrete tidligere streamgenbrug blev rettet; forskellige seeds er ikke et fysisk bevis for universel uafhængighed.

Med pathcost $Q_{jr}=\sum_qh_q(D_{jr,q}-A_q)^2$ er

$$\widehat J=\sum_jw_j\overline Q_j,\qquad
\widehat{\mathrm{SE}}(\widehat J)=
\sqrt{\sum_j\frac{w_j^2s_j^2}{256}}.$$

Formlen følger af uafhængige betingede strata; alle fem noder er simuleret med faste vægte. De 1280 paths er ikke 1280 iid draws fra marginalbanken. SE dækker MC-sampling betinget på bank og numerik, ikke EM-bias, FP-/LS-fejl, bankestimation, modelusikkerhed eller optimizerfejl.

Coarse Brownian increments er summen af to fine increments. Samme initialdraws og Brownian input bruges på tværs af policy, så forbedringen $d_{jr}=Q_{jr}(u_0)-Q_{jr}(u)$ har

$$\widehat\Delta=\sum_jw_j\overline d_j,\qquad
\widehat{\mathrm{SE}}(\widehat\Delta)=
\sqrt{\sum_j\frac{w_j^2s_{d,j}^2}{256}}.$$

Covariance er dermed inkluderet i forskellens varians. Inputhashes kontrolleres før parring. Kobling kan reducere sammenligningsstøj, men lover ikke altid mindre SE. I den korte seed-rettede prøve var EM fine J 0.0036682534675161536 og SE $8.870328259665031\cdot10^{-5}$, med nul rå randoverskridelser. Fine M254/EM-afvigelsen $4.7264\cdot10^{-5}$ bestod den praktiske gate $\max(0.001,4SE)$.

Den fælles syntetiske endpointmåling gav fire egne backend/grid-Bayes-opdateringer, med største conditional-L1 0.0145280 og nodevægtforskel 0.00200737. Dette var et kort forward-/EM-/endpoint-update-forsøg, ikke hele det efterfølgende feedbackstyringsdøgn.

### Implementeret døgnstyring og seneste faktiske status

Den nye driver søger fire priser på $[12,18),[18,24),[24,30),[30,36]$ med fast $A=0.5$ og den analytiske baseline. Den genbruger bounded Powell med to faste starts/callcaps; fulde kandidater og konvergensstatus bevares særskilt. Maxfev-stop er ikke konvergens, og bedste observerede kandidat er ikke et bevist optimum.

Designet har M254257/513 og FV257/513 fra **hver backends egen filtrerede origin**, neutral/deterministisk/stokastisk policyreview, partitionkontrol, parret EM og samme måling ved 36. Rå evidence opdaterer egne laws/vægte; de videreføres til 38. Det er posteriorbaseret open-loop-styring efterfulgt af et feedbacktrin, ikke dual control eller en Bellman-PDE. Der er ikke tilføjet stokastisk energilighed, terminal- eller smoothingstraf.

Den fulde **stubgraf** blev gennemført; et separat saved-fixture-review autentificerede 132 receipts og fire gemte Bayes-opdateringer uden compute. Det kontrollerer orchestration og algebra. Stubtransport, stub-likelihood og stubprojektion er ikke et numerisk døgnresultat. Den historisk fejlede originalinvocation bevares som fejlet.

Én ægte standalone deterministisk 24h-neutral måling blev kørt: 2.3313609979995817 s, J 0.12120382930681177, MAP-node 3 og egen middelstate ved 12 på 0.5062519458461637. Den er én-origin-timing, ikke et kompatibelt fire-origin-ratepacket eller en fuld prissøgning.

Kilder: [seed-rettelse og SE](control-and-evidence.md#g7-seed-repair-review), [quad-recovery](control-and-evidence.md#g7-complete-quad-recovery-review), [feedback](control-and-evidence.md#g7-feedback-integration-review), [døgndriver](control-and-evidence.md#g7-full-day-driver-review), [seneste fixturekontrol](control-and-evidence.md#g7-day-completion-review). Implementation: [døgndriver](../ORIGINALS.md#original-tree), [råreview](../ORIGINALS.md#original-tree), [auditeret feedback](../ORIGINALS.md#original-tree).

## 12. Hvorfor ejerskab, versionsidentitet og budget er fagligt relevante

Et genbrugt forward er kun samme eksperiment, når model, observation/cutoff, egen prior, node, policy, tids-/cellegrid, basis, numerik og sourceidentitet matcher. Et cache-hit med forkert state eller grid kan ændre J, evidence og kandidatvalg. SHA256, receipts og fulde kontrakter beskytter dermed den matematiske sammenligning; de beviser ikke, at modellen er fysisk sand.

M254 og FV skal beholde hver deres egen law. En state-law kan overføres konservativt til sammenligning, men må ikke derefter kaldes den anden backends egen filterposterior. Den gamle auxiliary-FV-forward fra en M254-origin er en dokumenteret reference; den er ikke en uafhængigt FV-filtreret origin til det nye døgnforsøg.

Den planlagte beregningsramme er

$$B_{\mathrm{plan}}=B_{\mathrm{reserve}}+1.25\sum_kn_kr_k,$$

hvor $r_k$ skal være kompatible målte whole-job-rater. En ukendt rate forbliver ukendt, og budgetestimat/admission forbliver null/false. Faktor 1.25 er en planlægningsreserve, ikke et matematisk worst-case-bevis. Korte 2h-rater og stubtider giver ikke automatisk en 24h-rate. Parent-/node-deadlines og varig registrering er separate værn.

Ved seneste afsluttede G-review var den bevarede debit 1927.3372718589885 s, registreret loft 1930 s og rest 2.6627281410114847 s. Alle 16 kompatible døgnrater var stadig ukendte. Denne dokumentindsamling genkører ingen numeriske kerner og giver ingen nye modelberegningstider eller nulstilling af tidligere charges.

Kilder: [day completion](control-and-evidence.md#g7-day-completion-review), [day check/rate](control-and-evidence.md#g7-day-check-and-rate-review), [recovery](control-and-evidence.md#control-checkpoints), [budgetintegration](../ORIGINALS.md#original-tree).


## Parret EM og hvad standardfejlen dækker
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


## Konkrete implementeringsfejl og uafsluttede delmål
Den første auxiliary-kørsel genbrugte RNG-streams mellem roller; dens påstand
om uafhængighed blev derfor afvist og senere rettet med role/node-identiteter.
Den reparerede korte256paths/node-kontrol er ikke en accept af det planlagte
fulde døgn. Parent-/node-receipts, write-aheadreservation og kildeidentitet
beskytter netop hvilke forwards der faktisk tilhører bank, prior og policy.

Tre gamle quadnode0/1/2-projektionsmaxima mangler. Færdige endpoint-/lawarrays
rekonstruerer ikke maxima over alle tidligere mellemprojektioner. De manglende
tal udfyldes ikke, selv om terminal- ogbankpargates består.

Den deterministiske nattepakke havde6policies,17units/19råfiler og374forwards
med3.775397722s proces. To node-fixturekontroller var stubinvocations og
gav ingen numeriske forwards; den bevaredecharge20s nulstilles ikke.
Den senere komplette døgngrafs fixturecheck er stadig ikke en science-run.

G7's langsigtede design havde fire6hprisblokke, to søgninger, trepolicies,
femkontekster og egen Bayes-feedback ved næste observation. Det er bevaret
som forsøgsdesign, ikke aktuel implicit autorisation. Alle originale
controllers/ledgers kan genskabes i deres nøjagtige træ via ORIGINALS.

## Attribution
Trackingbalancer, totalvarians, Bayes og stratificeret SE er direkte
udledninger. Nocedal–Wright kap10/16/18 understøtter LS/Gauss–Newton,
QP/KKT og SQP; konkrete TRF/SLSQP/Powell-starts, step ogcaps er egne valg.
Higham2001 understøtter Itô/EM; den konkrete Brownian-coupling, seeds,
RNG-roller ogtolerancer er forsøgets deklarationer, ingen fysisk garanti.

## Den diskrete WM-reference: konkret KKT-system

Dette er BM-studiets historiske sammenligningsargument; WM-koden ændres ikke.

### WM: affin respons og fire-dimensional konveks QP

Euler-dynamikken er affin i u med faste r, B og origin. Derfor kan residualen
skrives r(u)=c+Hu. Fem native forwards (u=0 og de fire enhedsvektorer) giver
c og hver søjle H_j=r(e_j)−r(0), uden parameterfit eller approksimativ surrogate.
E(u)=e0+a^Tu er ligeledes affin. Dermed

\[
 J=\|c+Hu\|^2,\quad Q=2H^TH\succeq0,\quad
 E=0\iff a^Tu=-e_0.
\]

Alle state-/demandkoordinater er affine. Hele prisboksens koordinatvise
minimum/maximum beregnes eksakt som v0+sum_j min(0,v_j) og
v0+sum_j max(0,v_j). Her er demand i [.2229618,.6098922] og
F i [.3212726,.3429841] for ALLE u∈[0,1]^4. Native pathconstraints er derfor
redundante i netop denne pakke; koden afviser at udelade dem, hvis certifikatet fejler.

Vi gennemgår alle 3^4=81 faces: hver pris er fri, på0 eller på1. På hvert
face løses de frie stationaritetsligninger, eventuelt KKT-systemet

\[
 \begin{bmatrix}Q_{ff}&a_f\\a_f^T&0\end{bmatrix}
 \begin{bmatrix}u_f\\\lambda\end{bmatrix}
 =\begin{bmatrix}-q_f-Q_{fb}u_b\\-e_0-a_b^Tu_b\end{bmatrix},\quad q=2H^Tc.
\]

Feasible kandidater sammenlignes i J. Dette er et lille udtømmende aktivt-
face-søg, ikke 81 native modelreplays. Q's egenværdier er
[.000170468,.0003529998,.0011353557,.0085948708], condition≈50.42:
normal equations er her kontrollerbart velkonditionerede. Dette begrunder
ikke normal equations i den ill-konditionerede M254-basis, hvor SVD bevares.

Endelig kontrolleres gradienten Q u+q+λa: nul på frie priser, ≥0 ved0,
≤0 ved1. KKT-residual er numerisk0 i begge WM-opgaver. Native forward på
valgt pris stemmer med den affine residual til højst3.47e−17. Med positiv
definit Q og konveks feasible mængde er WM-prisen entydigt global for den
DISKRETE opgave, inden for de rapporterede floating-point checks.
Dette er ikke et globalt optimum for den kontinuerte ODE uden discretiseringsfejl.


## Numerisk forbedringsgate i tre-plan-forsøget


En finere repræsentation kan ændre rækkefølgen. Derfor rapporteres fine-
forbedringen mod den samme originale costpartition separat:
\[
 G=J_{513,\Delta}(u_0)-J_{513,\Delta}(u_*),\quad
 e_0=|J_{257,\Delta}(u_0)-J_{513,\Delta}(u_0)|,\quad
 e_*=|J_{257,\Delta}(u_*)-J_{513,\Delta}(u_*)|.
\]
En numerisk forbedringspåstand kræver G>e_0+e_* og de uændrede forward-
gates. Det er en konservativ screening mod observerede gridforskelle,
ikke et rigorøst kontinuumsfejlestimat. Et failed fine-valg afvises;
neutralreference kan bevares uden at blive ommærket som optimeret policy.

Rapportér også E_gap=sum_ell Delta t_ell(mu_ell−B_ell) og endpoint-
middeltilstanden relativt til neutral. De viser, om lavere J ledsages af
ændret mængde/slutlager. De er ikke nye optimeringsbetingelser, pathwise
energibevarelse eller fysisk/kausal effektvalidering.



## Hvorfor den oprindelige RNG-fejl invaliderede SE
Uden de deklarerede uafhængige strata er
$$\operatorname{Var}(\widehat J)=\sum_jw_j^2\operatorname{Var}(\overline Q_j)
+2\sum_{j<k}w_jw_k\operatorname{Cov}(\overline Q_j,\overline Q_k).$$
Den tidligere diagonale SE udelod kovariansleddet. Kun efter den særskilte
role/node-reparation kan den uafhængige stratificerede formel anvendes under
forsøgets RNG-antagelser. Fine Brownian increments summeres til coarse;
denne tilsigtede parring er forskellig fra utilsigtet genbrug mellem roller.

## Hvorfor en endpointfil ikke erstatter projektionsaudit
Hvert nodeforward skal bevare
$$e_j=\max_{n\text{ i forward }j}\|Qc^*_{jn}-\widetilde m_{jn}\|_1.$$
Payload består af både råresultat og audit, $P_j=(R_j,a_j)$, og bindes samlet
til SHA256(encode(P_j)). En score eller terminaltilstand alene rekonstruerer
ikke dette maksimum. Cacheidentiteten omfatter origin, bank, policy, horizon,
grid, modelsource, costpartition, environment og eventuelle tilfældige inputs.
Den historiske receipt-API og alle mellemcharges findes i originaltræet;
de er ikke en ny aktiv launcher eller en nulstilling af tidligere tid.

## Konkrete kode- og kontrolindgange

[deterministisk evaluator](../../../src/method/bm_1_g7_deterministic.py), [energiligning](../../../src/method/bm_1_g7_energy.py), [job/receipt](../../../src/method/bm_1_g7_jobs.py), [energi-tests](../../../tests/test_bm1_g7_energy.py), [job-tests](../../../tests/test_bm1_g7_jobs.py). Historiske CLI- og controllerkommandoer findes kun i det restaurerede originaltræ; disse links peger på de bevarede matematiske komponenter. Tests er links til tidligere eller løbende kontroller, ikke påstået genkørt ved denne dokumentændring.
