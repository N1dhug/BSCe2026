# G7 neutral auxiliary: afsluttet kørsel og crash-recovery, 7. oktober 2026

## Resultat og afgrænsning

Den aktive G-opgave, et neutralt to-timers BM-forsøg fra egne gemte hour12-
posteriorer, er beregnet færdig. Afbrydelsen ramte efter resultatcheckpointet
bbe29b63e47f6164a2fbd1c32b764cad86840a2b og før publicering af det samlede
råreview og den opdaterede overdragelse. Genoptagelsen gentager ingen
færdige modeljobs.

Den oprindelige vejleder-M254-basis, numeriske tolerancer, native kerner,
deterministiske fits og F/WM-spor bevares. Direkte BM/WM-sammenligning er
udsat efter Johans præcisering; disse er interne BM-kontroller.

| Størrelse | Gennemført / resultat |
|---|---|
| Policy og horizon | u=(0.5,0.5), hour12→14; syntetisk fixture |
| Tidligere egne M254-forwards | Alle ti genbrugt; fem noder på hvert grid |
| FV-forwards | Ti nye i første invocation, alle genbrugt i continuation |
| EM | Fem nodepar × coarse/fine × 256 paths; ti gemte level-forwards |
| Endpoint-feedback | Fire komplette backend/grid-banker |
| M254 fine J | 0.0037155174638558177 |
| FV fine J | 0.0037153098107086916 |
| EM fine J | 0.003673709879183713 |
| M254 fine vs FV fine J | Absolut forskel 2.0765314712612587e-7 |
| Coupled EM tidsdifference | 9.378549894068638e-8 |
| Nye optimizer-/historiske-/WM-jobs | Ingen |
| Videnskabelig/full-day/kvadraturaccept | False |

Det samme diskrete trackingmål er
\[
J=\sum_\ell \Delta t_\ell E[(D(t_\ell)-A(t_\ell))^2].
\]
For parameterbanken bliver det
\[
J=\sum_j w_j J_j,\qquad
E[(D-A)^2]=(\mu_D-A)^2+\operatorname{Var}(D).
\]
Den totale demandvarians skal inkludere både conditional varians og
variansen mellem node-midler:
\[
\operatorname{Var}(D)
=\sum_jw_j\{v_j+(\mu_j-\sum_kw_k\mu_k)^2\}.
\]
Der er ikke brugt et gennemsnitligt parameter-plug-in eller fælles posterior
som erstatning for nodernes egne initialfordelinger.

## Færdige numeriske checks og deres grænse

Workerens fire forwardpar og fire feedbackpar består de uændrede gates.
M254/FV fine terminal conditional/marginal L1 er henholdsvis
0.0002562256386506235 og 0.00019348697010475703.
M254 coarse/fine conditional L1 er 0.011187751618265681.
Feedbacks største conditional L1 er 0.006378186315973684;
største vægtdifference er 0.0012942177487487716.
Begge M254-feedbackprojektioner har cell-mass L1 under 0.001626.
Ingen raw EM-boundary violations; ingen clipping eller reflection.

Det gemte råreview kontrollerer 14 completed receipts fra første forsøg
og 24 fra continuation, eksakt payload/job-identitet, EM-pathcosts og
posteriorbankens log-evidence/vægte. EM-pathcosts rekonstrueres uden
afvigelse i reviewet. Alle fire missing-observation checks bevarer egne
masser og vægte. Hvert grids Bayes-opdatering bruger sin egen prior:
\[
a_j=\log w_j+\log L_j(y),\quad
\log Z=\operatorname{logsumexp}(a),\quad
\log w'_j=a_j-\log Z.
\]
Bayes-regningen er et algebraisk/numerisk check ved det faste gemte y,
ikke en accept af generativ observationsuafhængighed.

Recovery har kun verificeret den allerede gemte reviews 47 filhashes og
begge komplette ZIP-hashes. Den gentager ikke råreviewets solverfrie
beregninger eller de tidligere beståede tests. Det nye hashcheck tog
0.207883419s og ligger inden for den bevarede konservative reviewcharge;
nul nye forwards. Råreviewets oprindelige tid er 8.384171951s.

## Konkret RNG-fejl: uafhængighed kan ikke påstås

Den frosne launcher bruger initial_seed+j og brownian_seed+j.
De fem initial-seeds er 20261021–20261025; Brownian-seeds er
20261022–20261026. Derfor er fire streams genbrugt på tværs af noder,
om end de trækkes til forskellige operationer. noise_seed=20261023
er samtidig initial-seed for truth-node2.

Inden for hvert coarse/fine-par er fælles initialtilstand og coupled
Brownianincrements tilsigtet:
\[
\Delta W^{c}_r=\Delta W^{f}_{2r}+\Delta W^{f}_{2r+1}.
\]
Det utilsigtede genbrug på tværs af node-initialisering og andre noders
Brownianforløb er ikke denne kontrollerede coupling. Samme seed for
truth-initialisering og observationstøj understøtter heller ikke modellen
y=D+\epsilon med uafhængig \epsilon.

Den gemte diagonalformel giver
\[
SE_{\rm diag}^2=\sum_jw_j^2s_j^2/N,\quad N=256,
\]
SE_diag=0.00009404807057972644, men den forudsætter nul covarians mellem
nodernes estimater. Generelt gælder
\[
\operatorname{Var}(\sum_jw_j\widehat J_j)
=\sum_{j,k}w_jw_k\operatorname{Cov}(\widehat J_j,\widehat J_k).
\]
Den diagonale SE er derfor ikke verificeret som den korrekte MC-standardfejl.
Den ekstra plugin-værdi sum_j w_j s_j/sqrt(N)=0.00020679082144193418
er kun en covariance-envelope-illustration; den er ikke et certifieret
interval og erstatter ingen acceptgate.

De rå FP/EM-differencer er omkring 4.18e-5 og under forsøgets faste .001-
numerikgrænse. Dette er et punktvist numerisk resultat. Workerens saved
auxiliary_gates_passed=true bevares som råregistrering, men giver ikke
independent_mc_standard_error_verified eller independent_synthetic_noise_verified.
Begge er false i råreviewet, og scientific_accept er false.
Der er ikke efterfølgende ændret seeds, observationer eller raw passed-flags.

## Fejlforsøg, continuation og fuld charge

Første invocation fejlede ved JSON-skrivning af diagnostic metadata
requested_max_step=Infinity efter EM-node0. De to beregnede, men ikke
gemte EM-level-forwards tælles fortsat som udført arbejde. Alle ti
færdige FV-forwards overlevede. Metadatarettelsen ændrer kun Infinity
til JSON-strengen "Infinity"; native max_step og science arrays bevares.

Den særskilt registrerede continuation har frossen kilde
164efc28e1f208b14950e3aeffbdcfa667f1cd7e og genbrugte de ti complete
FV-resultater efter eksakt kontrakt- og payloadkontrol. Ingen nye FV
eller M254-forwards. Worker returncode=0, hard_stop=false.

To senere read-only auditorforsøg fejlede ved henholdsvis floating-roundoff
i midpoint-indeksering og brug af fine prior ved coarse feedbackreview.
Begge kilder/fejl bevares. Den korrekte auditor bruger rounded uniform-grid
indeks med eksplicit tidscheck og hvert grids egen log-prior. Der er ikke
genkørt et modeljob for disse rettelser.

| Charge | Sekunder |
|---|---:|
| Tidligere G-debit | 1234.712123334 |
| Første invocation inkl. eksport/review | 37.469749930 |
| Continuation inkl. eksport/review | 55.472371850 |
| Ekstra konservativt auditorreview | 20.000000000 |
| Samlet auxiliary-task | 112.942121780 / 200 |
| Ny samlet G-debit | 1347.654245114 / 1800 |
| Resterende G-budget | 452.345754886 |

De to invocation-charges beregnes som max(worker-wall, ledger-increment)
+ faktisk eksport +20s review. Første charge er 12.993654967+4.476094963+20;
continuation er 22.903794014+12.568577836+20.
Ekstra20s dækker de ikke individuelt instrumenterede failed auditorforsøg
og recovery-hashcheck sammen med den tidligere reviewreserve. Ingen refund,
nulstilling eller ekstra allowance. Råreview-JSON angiver debit før ekstra
reviewcharge; auxiliary_final_budget_20261007.json er den samlede slutstatus.

## Durable råfiler og præcist fortsættelsespunkt

Begge komplette råarkiver er allerede gemt, og SHA256 er verificeret:
- g7_auxiliary_run_data.zip: 22905252 bytes,
  SHA256 4229bae5267a8869ab83aa2a73f3509b8d865d90a22dc6515eff06dddd1a4651,
  Library ID libfile_b835ffbd0bf88191b03b7c474db1149f.
- g7_auxiliary_resume_data.zip: 60945256 bytes,
  SHA256 9727ca5e6edccc957c5ad07f8b392776ec0a4be26ea5ddc82348b6aefcdedf03,
  Library ID libfile_5676976bf6648191afbe9b541d1fde8d.

Gamle origins, native source og de ti egne M254-forwards hører til det
tidligere node-runtimearkiv og dets 41 hashverificerede sourcefiler.
Ingen numerisk G-proces kører ved recovery. Den samtidige F-proces røres ikke.

Næste konkrete arbejde er en NY, lille registrering:
1. Giv hver node separate streamidentiteter for initialtilstand og Brownian,
   og observationstøj sin egen identitet. Bevar kun tilsigtet coarse/fine-coupling.
2. Hashfrys nyt launcher-/streamdesign og verificer ejerskab mod alle gemte
   M254/FV-data. Den gamle launcher/design/observation er historisk evidens.
3. Genbrug alle ti M254 og alle ti FV-forwards; beregn kun fem nye EM-par og
   den ændrede endpoint-generation samt fire berørte feedback-opdateringer.
   Frys ny observation før backend-feedback; behold gamle resultater separat.
4. Registrér samlet worker/stop/export/reviewcharge inden start inden for
   de resterende452.345754886s; ingen automatisk retry eller budgetudvidelse.
5. Vurder derefter faktisk policyoptimering, selvstændig quadraturpartition
   og et konkret budgetmæssigt muligt døgnscope.

Dette dokument starter ikke den nye registrering. Den oprindelige
one_invocation-only auxiliary-opgave er lukket med begrænset numerisk
evidens. M254/FV/J replays kræves kun hvis policy, origin eller partition
ændres. En ny prisplan kan ikke evalueres med de neutrale saved forwards.

Døgnplanens 14 rækker har stadig ukendte faktiske24h-rates. Gemte2h-rates
må ikke ommærkes til24h-admission. Kvadraturforfining kræver nye midtpunkter
og den tilhørende transportpartition; averaging af gamle midpoint-costs er
algebraisk samme sum. Prisoptimizer, optimeret-policy-feedback og full-day
accept er stadig åbne. G7/G som helhed markeres ikke færdig.
