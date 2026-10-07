# G7: durable Bayes-opdatering, næste forward og v2-budgetgate

8. oktober 2026. Kodeintegration og én fokuseret stubkontrol; intet nyt
videnskabeligt BM-forsøg. PR117 var allerede merged og verificeret.
Denne levering bygger på main306c7af, som også indeholder F-PR106.
F's nyeste ændringer og budget bevares.

## Resultat

FeedbackBankControl udvider den allerede kontrollerede G-auditadapter.
Den anvender den uændrede native OwnedBankPriceEvaluator.assimilate én gang
og gemmer rå stateposteriorer, nodevise evidences, projected posterior,
update-projektionernes audit og observationens identitet i én native
hashed/fsynced receipt før journalens completion. Ved reuse læses den
gemte posterior; der udføres ikke likelihood eller projektion igen.

Præregistrering79c761975844af495a74ce4754ff1bc9dd4e188b er committed og
læst byte-identisk tilbage før én check-invocation. Tyve checks består;
parentwall .908693493 s, body .222163066 s. Native transportforwards,
likelihood-evalueringer og projektionsløsninger: alle nul.

Konservativ checkcharge5 s: G1851.3372718589885/1910 s;
rest58.662728141011485 s. Alle tidligere charges og tab bevares.
Gamle quad0/1/2-audits er stadig tabte; gamle scientific/numerical
statusser er ikke opgraderet.

## Konkrete nye dele

| Fil | Funktion | Afgrænsning |
|---|---|---|
| g7/audited_feedback.py | Durable native bankupdate og authenticated continuation | Punktobservation; egne conditional states; samme likelihood/projection |
| g7/integrated_budget.py | V2-jobgraf og konservativ whole-job-pricing | Ukendte/incompatible rates er null; ingen implicit 2h→24h-skalering |
| g7/feedback_phase.py | Fire backend/grid-updates, tre nye prisvalg og næste2h-forwards | Kræver komplette auditerede24h-inputforwards fra tidligere fase |
| g7/launch_integrated_feedback.py | Stdlib preflight, én invocation, parent timeout og journalreserve | Feedbackfase; ingen samlet døgnoptimizer/EM-driver |
| g7/check_integrated_feedback.py | Én targeted crash/reuse/continuation/budgetkontrol | Forward, likelihood og solver er stubs |

Den tidligere audited_node_control og receipt_recovery genbruges uændret.
Native src/tests/.github ændres ikke i denne levering. De 41 oprindeligt
frosne numeriske sources ligger fortsat på872e777 og er hashbundet;
nye G-helpers indgår særskilt i den registrerede beregningskontekst.
Ingen original receipt eller journal er repareret eller ommærket.

Update-receiptens identitet binder:

- den fulde native forwardkontrakt, inklusive own origin, model, source,
  runtime, backend/grid, prisplan og transport/costpartition;
- alle fem forwardjobidentiteter og payloadhashes;
- det samlede videnskabelige forwardresultats hash;
- endpoint, observation, observed-status, point-contract og price-side;
- et eventuelt eksternt receiptpackets hash.

Eksterne completed forwards læses read-only fra et launchregistreret
packet. De skal have præcis samme kontrakt, gyldige payloadhashes og
gemt per-hypoteseaudit; de gamle unauditerede forwards kan ikke adopteres
som nye auditerede jobs. Den forrige budgetfil er særskilt pinned, så
opstart ikke nulstiller kampagnens debit.

## Matematik: adskil evidence og stateprojektion

Lad den allerede fremførte lov ved observationstid36 være
$P(X\in dx,\rho=\rho_j)=w_j^-p_j^-(x)dx$. Den syntetiske observation
er punkt-demand med den native normalfordelte observationsstøj:

$$
L_j(y\mid x)=\frac{1}{\sqrt{2\pi}\sigma_y}
\exp\left[-\frac{(y-D_j(36,x,u))^2}{2\sigma_y^2}\right].
$$

Bayes' regel giver

$$
Z_j=\int L_j(y\mid x)p_j^-(x)dx,\quad
Z=\sum_jw_j^-Z_j,
$$

$$
p_j^{raw,+}(x)=\frac{L_j(y\mid x)p_j^-(x)}{Z_j},\qquad
w_j^+=\frac{w_j^-Z_j}{Z}.
$$

Normalisering følger direkte: integralet af den rå stateposterior er1,
og summen af de nye parametervægte er1. $Z_j$ er en observationstæthed,
ikke en diskret sandsynlighed; dens log kan godt være positiv.

Den eksisterende numeriske operator bruger cellemasser $m_{j\ell}^-$
og trepunkts Gauss–Legendre-gennemsnit af likelihood i hver celle:

$$
\overline L_{j\ell}\approx\frac1{|C_\ell|}
\int_{C_\ell}L_j(y\mid x)dx,\quad
Z_j^{grid}=\sum_\ell m_{j\ell}^-\overline L_{j\ell},\quad
m_{j\ell}^{raw,+}=
\frac{m_{j\ell}^-\overline L_{j\ell}}{Z_j^{grid}}.
$$

Dette er den native celletilnærmelse; det er ikke eksakt integration
af den kontinuerte M254-lov. Registrerede egne grids/conditional priors
må ikke udskiftes med et andet grids mere gunstige tal.
Parameteropdateringen sker stabilt i logaritmer:

$$
\ell_Z=\operatorname{logsumexp}_j(\log w_j^-+\log Z_j),\quad
\log w_j^+=\log w_j^-+\log Z_j-\ell_Z.
$$

Først derefter repræsenteres hver rå stateposterior i vejlederens M254-basis:

$$
c_j^+=\arg\min_{c\geq0,\;\mathbf1^\top c=1}
\frac12\|Qc-m_j^{raw,+}\|_2^2,\qquad
m_j^{repr,+}=Qc_j^+.
$$

Projektionen ændrer stateapproksimationen, ikke de allerede beregnede
parameter-evidences. Derfor gemmes både $m_j^{raw,+}$ og
$m_j^{repr,+}$, de rå $Z_j$ samt
$e_j=\|m_j^{repr,+}-m_j^{raw,+}\|_1$.
Receiptvalideringen reproducerer logsumexp-vægtene og kontrollerer,
at den gemte projection-L1 svarer til forskellen mellem de to arrays.
SolverAudit opsamler alle udførte updateprojektioner, også eventuelle
fallbacks; raw nodecoverage skal være komplet.

Masse/ownership/algebra er særskilt fra numerisk accept. Helperen
returnerer ikke scientific_accept=true alene på grund af en komplet
receipt. Feedbackfasen afviser input-/posterior-projektion over den
registreredes .05-gate. Samlet M254/FV-/EM-/distributionreview er
fortsat nødvendigt; en .05-gate er ikke en kontinuerlig fejlgaranti.

Ved missing observation foretages ingen evidence eller projektion.
Den predicted lov genbruges uændret, evidence gemmes som JSON-null
i stedet for NaN, og audit får eksplicit not-applicable-status.
FV gemmer den rå posterior direkte og angiver, at mixture-projektion
ikke anvendes.

## Hvorfor næste forward faktisk er feedback

En opdateret posterior er kun et handoff, indtil den bruges.
continuation genlæser den hashverificerede update-receipt og bygger
den uændrede native evaluator med dens egen cutoff, conditional masses,
conditional coefficients, basis, parametre og logvægte.
En continuation fra det gamle cutoff eller med lånte parametre afvises.

Den implementerede feedbackfase tager én fælles observation ved36
til M254257/M254513/FV257/FV513's egne24h-forwards og opdaterer hver.
Fra den nye coarse M254-posterior fremføres tre erklærede planer
på[36,38]: [.5,.5], [.3,.7], [.7,.3]. Lavest coarse-J vælges; eksakte
ties følger kandidatrækkefølgen. De andre tre backend/grids bruger den
samme valgte pris fra deres egne posteriorer. Coarse valgt forward
genbruges som identisk receipt; ingen efterfølgende fine-reselection.

Målet bevares:

$$
J(u)=\int\left[(\mu-A)^2+
\sum_jw_jv_j+\sum_jw_j(\mu_j-\mu)^2\right]dt.
$$

Frem til næste observation holdes bankvægtene faste. Det er en kort
feedbackdemo med finite prisfamilie, ikke global optimering eller dual
control. Den samlede24h-søgning efter deterministisk/stokastisk policy
og dens paired-EM/crossreview skal levere input til denne fase.

## Budget og launcher

V2 har16 eksplicitte worktyper; counts og horizons følger det gemte
integrationsdesign. Den oprindelige native v1-budgethelper er uændret.
Én M254-objective-rate er et fem-hypotese bankjob, ikke én node.
Feedbackupdate-raten er én backend/grid-opdatering; fire sådanne tælles.

For worktype k med count $N_k$, målt whole-job-rate $r_k$ og deklareret
sikkerhedsfaktor $s=1.25$:

$$
T_{plan}=T_{reserve}+s\sum_kN_kr_k.
$$

Hvis en relevant rate mangler, passer til et andet horizon/own-origin/
source/runtime/problem eller ikke har deklareret rawreview og receipt-
provenance, bliver dens bidrag ukendt og admission=false. Det erstattes
aldrig med0. Gyldige rates forudsætter et særskilt reelt rawreview;
positive ratefixtures i testen er ikke empiriske målinger.

Admission kræver desuden plads i kampagneloftet, samlet optimizerestimat
<=300s og samlet validation-/feedbackestimat<=300s. Reserver er planlægning,
ikke worst-case runtimebeviser. Den konkrete feedbackfase skal passe
i sin workerreserve og i sin separat godkendte phase-hardcap.

Preflight læser alene standardbiblioteksfiler og validerer pinned
registrering/design/rates/budget/source/inputbytes. Native imports sker
først i worker efter admission. Parent bruger en persistent exklusiv
one-invocation-claim, fuld reserveret charge før opstart, BLAS=1 og
subprocess-timeout. Ingen automatisk retry. Den fulde phasecap beholdes
på både completed/failed/hard-stop; en større faktisk parentwall øger
chargen og markeres som overskridelse. Dette koster konservativt, men
undgår en ny ubegrundet refusion ved proces-tab.

Før numerisk launch mangler følgende:

1. Den samlede døgnsearch/EM/crossreview-driver, ikke blot feedbackworker.
2. Frosne og rawreviewede whole-job-rates eller et særskilt konkret
   begrænset måleforsøg samt en passende samlet budgetbeslutning.
3. Færdige auditerede24h-inputreceipts under samme frosne v2-source.
4. Særskilt launchregistrering med source/origin/runtime, hele grafen,
   phasecap og stop/eksport/rawreview-reserver, committed før invocation.

Der findes ingen udfyldt autoriseret numerisk launchregistrering i denne
levering. launch_ready/budget_admitted/full_day/scientific_accept=false.
Planen er ikke startet, og2h-evidens er ikke ommærket til24h-admission.

## Faktisk kontrol og begrænsning

Den ene check bruger native store/guard, bankreduktion, SolverAudit,
originvalidering og Bayes-algebra. Forward, likelihood og solver er små
stubs; den opdaterer ikke historiske eller syntetiske forsøgsresultater.
Den simulerer process-tab efter durable update-receipt før ledgercompletion,
kontrollerer genbrug uden replay/refusion, anvendt opdateret cutoff/lov,
missing/FV-status, read-only ekstern adoption, afvisning af ændrede
evidences/parametre/hashes/horizons samt hele grafens budgetafvisninger.

Den kontrollerer ikke et helt fire-grid-feedbackfaseforsøg eller
subprocess-timeout med en virkelig native beregning. Der er ingen
ny numerical/scientific/historisk accept eller BM/WM-sammenligning.
Tyve delchecks er én samlet targeted invocation, ikke et ekstra panel.
Den tidligere beståede auditerede-forward-test blev ikke genkørt.

Reproduktion, fra den registrerede native checkout med de nye helpers:

    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -B root/g7/check_integrated_feedback.py

Gemte registrering, manifest, design, checkresultat og budgetstatus findes
i g7/integration_*_20261008.json samt integration_design.json og
integration_native_manifest.json. Reproduktionskommandoen er ikke
en opfordring til at gentage en allerede bestået kontrol.
