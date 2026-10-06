# G7 — vejlederens M254-posterior til prisstyring og ny observation

7. oktober2026. Dette er G-sporet, selvstændigt fra F4-H2 og WM.
Udgangspunkt: main `1bf048940483a4707035e84f707ce4ef407e9fe0`.
Genbrug de accepterede egne time12-origins fra M254257/M254513-filteret.
Ingen gamle G8-døgn genkøres, ingen historiske demanddata læses eller fits ændres.

**Aktuelt resultat: runtimepilot stoppet efter60.001s.** Det grove neutrale
to-timersforward gennemføres efter17.61s elapsed; den fine mixture-forward
blev afbrudt under sin første evaluation. Ingen parret præcisionsaccept,
prisoptimering, EM eller ny observationsopdatering er udført. De følgende
afsnit beskriver den implementerede og registrerede protokol, ikke en
gennemført kontrolundersøgelse. G8's tidligere accept og rådata er uændrede.

## Afgrænset leverance og krav, der stadig står åbne

Denne leverance implementerer typed own-posterior price-forward og en
**to-timers interface-/kontrolprototype**12→14. Det planlagte G7-døgn er
stadig24 timer med2–6 prisblokke og sine egne policy-/FP/mixture-/EM-gates.
Prototypen erstatter ikke dette krav. En målt pilot prissætter hele døgnets
jobgraf før admission; budgetstop/afvist admission er ikke fuld G7-accept.

Samme fem rho-noder og hver nodes egne c/masser fra G8. Vejlederens M254
er primær mixture-basis. FV er uafhængig numerisk reference og den allerede
tilladte FP-backend til selve prissøgningen. Legacy DynamicMixturePriceEvaluator,
G8's FV-control-helper samt alle fælles FP/LS/Bayes/EM-kerner bevares.
Ingen af de gamle evaluators ommærkes til typed M254.

Prototypen har to timeholdte prisblokke[12,13),[13,14], u_j i[0,1],
A(t)=.5 valgt ex ante, samme analytiske G8-baseline B(t)=.5+.1sin(2pi t/24).
Der indgår ikke fremtidige observerede priser eller demand i objective.
Terminal- og prisglathedsstraffe er begge0. Tracking af A undersøges først
med det oprindelige objective; nye strafvægte ville kræve et særskilt valg.
Dette er målprofilstyring, ikke forecast-fit til observeret demand eller profit.

## Udledning af kontrolmålet

Ved cutoff t0 har vi den fælles parameter-/tilstandsposterior

\[
p(x,\rho_j\mid\mathcal F_{t_0})=w_jp_j(x),\qquad
\sum_jw_j=1.
\]

For et kandidatprisforløb U fremskrives hver conditional law fra sit eget
startpunkt med samme frosne fysiske parametre og rho_j:

\[
\partial_s p_j^U=-\partial_x\{\rho_j(D(x,s;U)-B(s))p_j^U\}
+\tfrac12\partial_{xx}\{[.2x(1-x)]^2p_j^U\}.
\]

Vægtene w_j fryses under open-loop evaluering, fordi der ikke kommer en ny
observation i objective. Hver prisvektor prøves fra samme cutoff-posterior;
de sidste koefficienter fra en tidligere kandidat må aldrig blive næste
kandidats starttilstand. Kun geometri/face-factorizations kan caches.
Dette er single shooting: kun prisvektoren optimeres, mens hele state-/
fordelingsbanen beregnes fremad for hver kandidat.

Det eksisterende kvadratiske trackingmål er

\[
J(U)=\sum_j w_j\int_{t_0}^{T}\int_0^1
[D(x,s;U)-A(s)]^2p_j^U(x,s)\,dx\,ds.
\]

Sæt mu_j=E_j[D], v_j=Var_j(D), mu=sum_j w_j mu_j. Ved ekspansion af
kvadratet og loven om total varians fås

\[
J(U)=\int_{t_0}^{T}\left[(\mu-A)^2+
\underbrace{\sum_jw_jv_j}_{\text{within-node varians}}+
\underbrace{\sum_jw_j(\mu_j-\mu)^2}_{\text{between-node varians}}\right]ds.
\]

Det sidste led bevarer parameterusikkerhed. At styre alene middelbanens
kvadrerede fejl er et andet objective. En gennemsnitlig rho kombineret med
en marginal statefordeling bevarer heller ikke generelt det fulde mål,
fordi state og parameter er afhængige og dynamik/demand er ikke-lineære.
I koden kontrolleres både J=sum_j w_j J_j og totalvariansidentiteten1e-10.

## Basis, quadratur og konkret kodegenbrug

\[
\ell_{m,254}=255\binom{254}{m}x^m(1-x)^{254-m},\quad
p_j=\sum_m c_{jm}\ell_{m,254},\quad c_j\ge0,\quad\sum_m c_{jm}=1.
\]

Den præcise checkpoint-basis har B_mi=dx^{-1}int_cell ell_m dx.
`OwnedBankPriceEvaluator` i `src/method/bm_1_g7.py` kræver samme grid og
typed basis, bygger eksisterende BM1ControlProblem/FV-support og genbruger
`evolve_projected_density` direkte med originens ejede c_j og B.
Der er ingen initial LS-reprojektion, ingen priorreset eller FV-reset.
Den eksisterende simplex-SVD-LS bruges ved de interne transportskridt.

Time-midpoint-quadratur har h_quad=1/96 for begge grids, med knæk ved13.
Solver-max_step er1/96 på257 og1/192 på513; eksisterende CFL=.4 er uændret.
Parret forward-forfining undersøger rum/tid, ikke selvstændig objektiv-
kvadraturkonvergens. Den uændrede quadratur skal stadig indgå i fuldt G7's
numerikkontrol. U_j er præcis timeholdt; der indføres ikke lineær prisinterpolation.

Tracking-/demandmomenter og kvadratur kommer fra `src/control/objective.py`.
FV-forward, deterministic-evaluator og bounded Powell kommer fra eksisterende
`src/control/bm_1.py`. Kun G-adapter-/forsøgsfiler tilføjes; shared kode ændres ikke.

## Prissøgning, referencer og hvad optimum betyder

Der bruges én neutral start[.5,.5] per policyfit. Deterministisk reference
bruger rho-MAP-nodens egen conditional mean som en eksplicit punktapproximation,
ikke en marginal state kombineret med en fremmed parameter. Stokastisk søgning
vægter de fulde fem conditional FV-laws fra det ejede grove M254-origin.
FP-søgning er tilladt i G-planen; M254-prisforward krydsevaluerer de få
valgte kandidater for at undgå en dyr mixture-forward ved alle objectivekald.
Det er en beregningsbeslutning, ikke en udskiftning af vejlederens mixture-basis.

Existing multistart_powell genbruges med én start, maxfev48 per fit,
maxiter6, xtol1e-3, ftol1e-5. Begge fits deler højst96 nye objectivekald
og300s. Ingen gridsearch eller ekstra starts. Powell er derivative-free,
passer til eksisterende price-evaluator og undgår nye adjoint-/gradientkerner.
Det er ikke et bevis for global optimalitet. Ved manglende optimizerkonvergens
bevares bedste fuldførte kandidat og stopbeskeden; kandidaten ommærkes ikke
til et optimum eller et fuldt accepteret fit.

Neutral, deterministisk kandidat og stokastisk kandidat sammenholdes under
samme stokastiske objective. Hver policy kontrolleres med eget M254257,
eget M254513 og independent FV513. Terminale conditional-L1/marginal-
momenter/CDF/kvantiler bruger de uændrede G8-gates. J-gaten er særskilt
registreret som max(.001, .02*abs(J_reference)); den erstatter ikke terminalgates.
Det er en praktisk prototypegate, ikke en garanti for rangordning af næsten
ens politikker. Ingen gates lempes efter et resultat.

## EM og estimation→kontrol→ny observation

256 paths per node, stratificeret over parameterbanken. Samme conditional
M254-Beta-draw og Brownian increments bruges for alle politikker.
En start samples ved m~Categorical(c_j), X~Beta(m+1,255-m): dette følger
checkpointets kontinuerte Bernstein-mixture, ikke en ny Beta(20,20)-prior.
Fine EM har h1/384, coarse h1/192; coarse dW er summerede fine increments.
Ingen clipping/reflection. Rå randoverskridelser stopper forsøget.

Stratificeret joint cost-estimat er sum_j w_j mean(cost_j), med
sampling-SE sqrt(sum_j w_j^2 s_j^2/256). Det er ikke SE for iid samples
af tilfældigt valgte parameternoder. FP-sammenligning kræver error <=
max(.001,4SE); coupled timeerror <=.001. Dette er praktiske samplingchecks,
ikke simultane konfidensgarantier. Rå parrede pathcosts gemmes til politikforskelle.

Efter prissøgningen genereres én ny syntetisk punktmåling ved14 under den
valgte stokastiske policy: den på forhånd bestemte første fine EM-path for
rho=.25 plus uafhængig N(0,.03^2)-støj. Dette er en ny posterior-predictive
kontroldemo, ikke original-fixturens fremtidige Y under dens gamle priser.
Ingen path vælges efter, hvor godt den passer til modellen eller target.

Den separate assimilate-metode bruger den faktiske candidate price-plan i
`bm1_observation_update`, GL3 og højre side; ingen ny FP-propagation:

\[
Z_j=\int L(y\mid x,U_T)p_j^-(x)dx,\quad
p_j^{raw,+}=Lp_j^-/Z_j,\quad w_j^+=w_jZ_j/\sum_kw_kZ_k.
\]

Evidence beregnes før LS-projektion af state-Bayes-resultatet. Logvægte
opdateres med eksisterende parameter_bank_weights. Det opdaterede typed
origin bevarer hver nodes egne koefficienter og kan bruges ved næste cutoff.
Ved missing measurement er der ingen Bayes-projektion og ingen vægtændring.
Assimilation afviser evaluation fra en anden evaluator/handoff.

## Kilder og egne valg

Vejlederens basis: `07-wSnLZuio-1-.jpg`; SDE/FP: `05-Mkh9zqUP-1-.jpg`.
Mixture/positiv LS: `MixtureApproximationsForTheFokkerPlanckEquation-1.pdf`,
slides6/8/12; active-set-genbrug: Nocedal/Wright16.3.
Gaussisk observation: `MSC_AGP_FINAL-1.pdf`, kap6.1.1eq6.2, trykt90/PDF102.
EM/metodens sampling og Brownian kobling: Higham2001, plus eksisterende
projektkerner. Trackingdekomposition, totalvarians og Bayes-normalisering
er standardidentiteter; M254-forfiningsreviewet dokumenterer basiskilderne.
Specialets EKF er ikke vores bank, og en kilde foreskriver ikke vores
to-timers horizon, rho-grid, Powell-tolerancer eller budgettal. Disse er
deklarerede egne prototype-/numerikvalg. F/WM-modeller ændres ikke.

## Budget, kontroller og resultater

Nyt scoped loft600s inklusive pilot<=60s, search<=300s/96kald og eksportreserve30s.
Én fokuseret kontraktcheck tog.410s, konservativt charged2s uden gentagelse.
Paneldebit er983.2665s (650s tidligere reservation +333.2665s M254fine/archive),
så selv loft600s holder sig under1800s. Budgettet nulstilles ikke ved stop/resume.
Alle importerede eksisterende kilder kontrolleres mod latest-main Gitblobs før
kørsel; source/inputs hashes kontrolleres igen efter. Registrering og atomiske
checkpoints bevarer partials. POSIX TimeoutError kan afbryde LS uden fallback.

### Faktisk udført og bevaret

Én målrettet kontrakttest består på.410s. Den bruger en stub for transporten
for konkret at kontrollere kandidaat-reset, forskellige ejede nodekoefficienter,
totalvarians, uændret missing-update og afvisning af fremmed evaluation/
tidsforskudt horizon. Den er ikke en numerisk FP- eller observationsaccept.
Testen genkøres ikke efter eksport-/gate-guard-fixet i forsøgsrunneren.

Den faktiske grove M254-price-forward under U=[.5,.5] når slutpunkt14
uden exception og giver et typed predicted origin fra sine egne koefficienter.
Dette konstateres fra den completed-kørselslog; elapsed17.61s er afrundet
og inkluderer preparation. Det er ikke en præcis solver-only-rate.
Fine-forward nåede ikke at returnere inden pilotloftet; timeout kom60.001028s
efter registrering. POSIX-alarmen afslutter invocation, og intet job fortsætter
i baggrunden. Alle originale39 sourcehashes og alle3 inputhashes er uændrede
under kørslen;36 importerede eksisterende Gitblobs matcher latest main.

Ingen complete fine-rate, objective, parrede precisiongates eller faktisk
kontrol-feedback er opnået. `prototype_accept=false`, `full_day_G7_accept=false`,
`historical_accept=false`. Ingen optimering eller simuleret Y er udført.
Registrering, partial progress, failure, stdout, sourceaudit og exact executed
source bevares. Pilotloftet forlænges ikke, og der køres ikke en skjult retry.

### Eksportmangel fundet ved budgetstoppet og rettet

V1 gemte pilotens raw trajectories først efter alle fire forwards. Derfor
blev det grove completed-forward ikke råeksporteret, før den fine evaluation
blev afbrudt. Dets trajectory kan ikke genskabes uden replay, som ikke er
kørt. Den bevarede coarse timing/log er derfor runtimeevidens, ikke et
råreviewet positivt numerisk sammenligningsresultat.

Den publicerede runner gemmer nu NPZ og typed JSON straks efter hvert
completed pilot-forward og logger completed tællere/rates før næste solve.
Projector auditcounters eksporteres også i finally ved en afbrydelse.
Eksportfixet supplerer også de allerede erklærede masser/evidence/vægt-gates
eksplicit for marginal-L1 og den endnu ukørte feedback-sammenligning.
Det ændrer ingen tærskler eller tidligere udførte checks/acceptresultater.
Model, FP/LS, prisforløb, tolerancer og
60s-pilotloft er uændrede. Original executed V1-source er frozen før fixet
og ligger separat i råarkivet. V2 består AST-syntaks; ingen ny pilot er kørt.
Det gamle source-unchanged-resultat vedrører korrekt den originalt udførteV1,
ikke den senere corrected runner. Se correction_audit.json med nySHA256.

### Budget og næste faglige beslutning

Ny conservative paneldebit1045.267573s =983.266544 tidligere +60.001028
invocation +2s testcharge, før dette råarkivs faktisk målte eksporttid.
Det reserverede600s er et loft, ikke faktisk brugte600s og ikke et nyt budget
efter timeout. Resterende ramme er cirka754.73s før arkiv, og piloten er
allerede brugt. Ingen fuld jobgraf er admitted, selv om compute-loftet havde
ubrugt tid: en runtimepilot er en admissiongate, ikke kun et ur.

En illustration med3 policies *12 (24h/2h) *cirka60s giver cirka2160s
alene til to-grid mixture-krydskontrol, før prissøgning/FV/EM. Piloten er
ufuldstændig, model-/policy-rates kan variere, og geometrisk cacheopvarmning
kan ændre omkostninger. Dette er **ikke** en fuldt målt rate eller et formelt
runtime-lower-bound; det åbner under ingen omstændigheder et fuldt døgnjob.

Næste er et smalt runtime-/forsøgsdesignreview af G7: vælg konkrete
policy-/forwardopgaver og en eksplicit budgetteret representative pilot,
genbrug G8-evidens og de ejede origins. En billigere lovlig objective/backend
eller et kortere deklareret udviklingscase kræver sin egen begrundelse og
præcisionskontrol. M254 og det planlagte døgn må ikke stiltiende udskiftes
eller markeres accepteret. Ingen ny stor bank, gridsearch, større støj,
løsere gate eller månedspanel. Lokal PC-runtime kan eventuelt undersøges
senere med frozen source; ingen lokal modelkommando kræves af denne leverance.

Råarkiv: `G7_owned_handoff_pilot_20261007.zip` med original executed source,
corrected runner, original coarse/fine inputorigins, metadata og matematik.
Reporesultat: `files/output/bm_1/runs/g7_owned_handoff_20261007.json`.
Fuld G7-døgnaccept, global optimalitet, parameterrecovery, historisk forecast
og økonomisk profit forbliver uafsluttet.
