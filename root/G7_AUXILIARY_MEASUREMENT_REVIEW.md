Historisk review fra commit1394a7a før seed-rettelsen. Bevar de oprindelige
tal og forbehold nedenfor; de er ikke den aktuelle budget-/næste-trin-status.
Senere RNG-rettelse: G7_SEED_REPAIR_REVIEW.md. Nyeste recovery:
[G7_COMPLETE_QUAD_RECOVERY_REVIEW.md](G7_COMPLETE_QUAD_RECOVERY_REVIEW.md).

# G7: eget neutralt FV/EM/endpoint-forsøg, råreview og begrænsninger

7. oktober 2026. G-sporet, BM alene. Vejlederens normaliserede M254-basis
er primær og uændret; legacy og lokal referencebasis er bevaret. F- og
WM-kilder, deres kørsler og budgetter er uændrede. Direkte BM/WM-sammenligning
venter, til BM er færdig. Her sammenlignes kun numeriske BM-metoder.

Det registrerede **12→14h-neutralforsøg er gennemført**: ti tidligere egne
M254-forwards genbruges, ti egne FV-forwards er målt, fem paired EM-noder
og fire endpoint-opdateringer er gemt. Alle registrerede numeriske gates
består. Råreview finder dog overlappende RNG-seeds: standardfejlens
uafhængighedsantagelse og generativ uafhængig observationsstøj er ikke
dokumenteret. Resultatet understøtter derfor intern numerisk konsistens
på den fælles fixture, **ikke fuld stokastisk eller døgnaccept**.

## 1. Ejerskab, scope og faktisk udført

Numerisk source forbliver `872e77714d1bc5848d8dce5b1fac058417b796cf`.
Restore kontrollerer de 41 frosne sourcehashes, begge originale inputbytes,
tolv gamle ledger-attempts, hele evaluatorgeometrien/versionerne og de ti
gamle completed nodepayloads. Originalledger og M254-forwards er read-only.
Bankerne rekonstrueres gennem native `combine_nodes`, som kontrollerer egne
baner, demandmomenter, J og totalvarians; ejerskab til `assimilate` genoprettes
derigennem. Der indsættes ikke et frit `_owner` på et vilkårligt JSON.

Fixture: horizon [12,14], pris [.5,.5] med skift ved13, analytisk baseline
B(t)=.5+.1sin(2πt/24), fast A(t)=.5. Grid257/513, transport-maxstep1/96h
henholdsvis1/192h, fælles objective-midpointstep1/96h, CFL.4. Rho-noderne
er [.125,.1875,.25,.375,.5] med hver sin ejede posterior og cutoff-vægt.
Normaliseret demand og timer fastholdes; J er i normaliseret demand²×timer,
ikke et valideret fysisk kW²h-tal. Januarens deterministiske24h-case er en
anden fixture, og dens rå J kan ikke rangeres mod denne2h-case.

FV starter fra det tilsvarende grids egne integrerede M254-cellemasser.
Dermed isoleres transportmetoden. Det er ikke en ny historisk FV-filterbank.
EM starter fra den fine kontinuerte M254-law; celle- versus kontinuerlig
startsandsynlighed er en kendt del af sammenligningen.

Første execution `9e56b1cdd3448f3f76f6945e4e33e43589c3a139` gennemførte
fire setups og alle ti FV-forwards. EM-node0 beregnede begge tidsniveauer,
men receiptens metadata `requested_max_step=inf` blev afvist af strict JSON.
Der var ingen gemte EM-arrays at genbruge. Fejl, delresultater og charge
bevares i første ZIP og checkpoint `c05381801f8f84a9f57b5509660cf9498589b2cb`.

En **særskilt registreret fortsættelse**, execution
`164efc28e1f208b14950e3aeffbdcfa667f1cd7e`, gentager eksplicit kun den
ugemte EM-node0 og udfører resten. De ti FV-nodepayloads importeres efter
hash-, ledger-, payload- og fuld kontraktkontrol; ingen native FV-genkørsel.
Levende setups rekonstrueres med ny charge. Kun metadatafeltets positive
uendelighed skrives som JSON-strengen `Infinity`. Native `max_step=np.inf`,
SDE, tidsgrids og startsamples bevares; ikke-endelige science-arrays afvises.

Faktiske native antal på tværs af de to invocations: **10 FV-forwards,
12 EM-level-forwards** (2 ugemte i fejlforsøget og10 gemte i fortsættelsen),
4 observerede bankopdateringer og4 missing-observation-identitetskontroller.
De10 originale M254-forwards genbruges; ingen nye M254-forwards, optimizer,
refits, historiske observationer eller G8-filtre. En opdateret origin er
gemt; næste kontrolforward fra den er endnu ikke udført.

## 2. Matematikken og hvorfor den bruges

BM's deterministiske mål og opgaven med bevaret samlet forbrug bevares:
\[
 J_d(u)=\int_0^{24}[q(t;u)-A(t)]^2dt,\qquad
 E_d(u)=\int_0^{24}[q(t;u)-B(t)]dt=0.
\]
Samme diskrete demand og tidsvægte bruges i begge. En lighed sikrer, at
tracking ikke forbedres ved at ændre nettomængden. En straf λE² ville
tillade afvigelser og indføre et ekstra valg af λ. De allerede gemte BM-
policies og reference uden ligheden bevares; der søges ikke igen her.
Denne2h-auxiliarycase indfører ikke en ny stokastisk energibegrænsning.

Det eksisterende stokastiske originalmål er
\[
 J_s(u)=\sum_jw_j\int\mathbb E_j[(D_t(u)-A_t)^2]dt
       =\int[(\mu-A)^2+v_{\rm within}+v_{\rm between}]dt,
\]
\[
 \mu=\sum_jw_j\mu_j,\quad
 v_{\rm within}=\sum_jw_jv_j,\quad
 v_{\rm between}=\sum_jw_j(\mu_j-\mu)^2.
\]
Udledning: E[(D−A)²]=(E[D]−A)²+Var(D), efterfulgt af totalvariansloven.
Variansleddene er en del af det forventede kvadratiske tab, ikke en ny
regulariseringsstraf. Kun middeltracking ville løse en anden opgave.
En enkelt gennemsnitlig rho ville miste parameter/state-afhængigheden.

Vejlederbasis er den normaliserede Bernstein/Beta-familie
\[
 p_j(x)=\sum_{i=0}^{254}c_{ji}b_{i,254}(x),\quad
 b_{i,m}(x)=(m+1){m\choose i}x^i(1-x)^{m-i},\quad
 c_{ji}\ge0,\quad\sum_ic_{ji}=1.
\]
Integralet af hvert b er1. Beta-identiteten giver derfor eksakt sampling:
I~Categorical(c_j), X₀~Beta(I+1,255−I). Det bevarer kontinuerlig M254-law
og kræver hverken ny prior eller erstatning af vejlederbasis. Eksisterende
positiv LS-projektion genbruges med uændrede tolerancer og egne coefficients;
ingen ny basisfamilie eller skjult straf indføres.

EM genbruger Euler–Maruyama:
\[
 X_{n+1}=X_n+b(t_n,X_n,u_n)h+\sigma(X_n)\Delta W_n,
 \qquad \Delta W_n\sim N(0,h).
\]
Fine1/384h og coarse1/192h bruger samme startsamples og summerede fine
Brownian-increments. Summen har korrekt coarse varians; koblingen reducerer
støj i tidsforskellen. 256 paths per node bevares. Raw boundary violations
er0; clipping/reflection bruges ikke til at skjule tidsfejl.

For Q_jℓ=Σ_n Δt_n(D_jℓ(t_n)−A_n)² er estimatoren
\[
 \widehat J=\sum_jw_j\overline Q_j.
\]
Hvis node-middelværdier er uafhængige, er variansen Σ_j w_j² Var(Q_j)/256,
og diagonal sample-SE er sqrt(Σ_j w_j²s_j²/256). Denne antagelse blev
ikke opfyldt af det registrerede RNG-design; begrænsningen nedenfor er bindende.

Endpoint-opdateringen er Bayes med egne forudsagte laws og faktisk pris:
\[
 Z_j=\int L(y\mid x,u_{14})p_j^-(x)dx,\qquad
 p_j^{raw,+}=L p_j^-/Z_j,\qquad
 w_j^+=\frac{w_jZ_j}{\sum_kw_kZ_k}.
\]
Evidence beregnes før state-LS-projektion. Det bevarer parameter-likelihoodens
definition og undgår, at projektionens approximation bliver en ny model.
Ingen ekstra forward behøves til selve endpoint-opdateringen. Missing y
bevarer masses og log_weights eksakt. Timeenergi/historiske intervaller
kræver fortsat egen observationsoperator; punktdemoen validerer dem ikke.

## 3. Faktiske numeriske resultater

| Metode | Tracking J |
|---|---:|
| M254513, gemt egen fine forward | .0037155174638558177 |
| FV513, egen fine forward | .0037153098107086916 |
| M254257, gemt egen coarse forward | .003741963573462555 |
| FV257, egen coarse forward | .0037415842631927966 |
| EM fine, deskriptiv vægtet samplemiddel | .003673709879183713 |
| EM coarse, deskriptiv vægtet samplemiddel | .0036738036646826538 |

Fine M254/FV-J-forskel **2.07653147e−7**, mod gate.001. Deres terminal
conditional-L1 er.000256225639 og marginal-L1.000193486970.
M254 coarse/fine conditional-L1.0111877516 og FV.0110940852, begge under.05.
Alle fire registrerede forward-pairs består deres J-, moment-, CDF-, kvantil-
og massgates. Det er kontroller på den frosne fixture og horizon, ikke bevis
for alle priser eller24h. Coarse/fine J bruger samme kvadraturpartition.

EM-fine minus M254-fine er −4.18075847e−5; coarse/fine-EM-forskel
9.37854989e−8. Registreret diagonal-SE.00009404807058; max(.001,4SE)=.001,
så det faktiske check er her den faste.001-gate. Seedforbeholdet kan ikke
fjernes ved, at en numerisk tolerance består.

Fælles y ved14 er **.508113473411657**: valgt neutral fine-path
rho=.25/path0, truth.45750241979370576, noise.05061105361795124.
Den er kun en fælles numerisk testmåling; uafhængig generativ støj er ikke bevist.

| Posterior-pair | Max conditional-L1 | Max vægtforskel | Max conditional log-evidenceforskel |
|---|---:|---:|---:|
| M254 coarse/fine | .00637818632 | .00129421775 | .00769300305 |
| FV coarse/fine | .00634616838 | .00128913488 | .00773000997 |
| M254/FV fine | .00165428517 | .00002037548 | .00018420554 |
| M254/FV coarse | .00173304910 | .00003839540 | .00036763045 |

Uændrede gates: posteriorconditional/marginal-L1≤.05, vægtfejl≤.03 og
conditional/aggregate log-evidencefejl≤.05. Alle fire pairs består.
M254 endpoint-projektioners maxcellmass-L1 er.00162554802 fine og.00161073741
coarse mod.05; maxcoefficient-massfejl4.663e−15. Fem egne projektioner
per M254-bank; FV bruger ingen M254-LS. Posteriorvægtene rekonstrueres fra
hver egen grids prior og conditional evidence til1e−12 uden modelkald.

## 4. RNG-begrænsning og næste korrekte konstruktion

Præregistreret initialseed=20261021+j og Brownianseed=20261022+j giver
overlap **20261022–20261025**. Brownsk RNG på node j bliver samme generator-
start som initial-RNG på node j+1. Noise20261023 genstarter også den
valgte truth-nodes initial-RNG (og node1's Brownian-RNG). Forskellige sampling-
funktioner gør ikke sådanne generatorer uafhængige.

Derfor mangler diagonal-SE covariance-leddene:
\[
 \operatorname{Var}(\widehat J)
 =\sum_jw_j^2\operatorname{Var}(\overline Q_j)
 +2\sum_{j<k}w_jw_k\operatorname{Cov}(\overline Q_j,\overline Q_k).
\]
Cauchy–Schwarz giver for ikke-negative vægte den algebraiske envelope
SD(Σw_jQ̄_j)≤Σw_jSD(Q̄_j). Indsættelse af marginale sample-SE giver her
**.00020679082144**. Dette er en konservativ *plug-in illustration*, ikke
en certificeret SE, et konfidensinterval eller en ny post-hoc acceptgate.
4 gange denne er også under.001, men den konstatering validerer ikke RNG.

Rå tal/registreringsgates bevares uændret. Autoritativt rawreview markerer
`independent_mc_standard_error_verified=false`,
`independent_synthetic_noise_verified=false`, `scientific_accept=false`.
Fast fælles y er stadig egnet til at sammenligne den numeriske Bayes-operator
mellem backends/grids, betinget på dette y; det er ikke filterkalibrering.

Næste RNG-konstruktion skal være særskilt præregistreret med disjunkte
roller, fx `default_rng(SeedSequence([20261021, role, node]))`, role0 initial,
role1 Brownian, role2 observation. Genbrug præcis samme initial/dW inden for
et coarse/fine-par; roller/noder genstarter ikke samme stream. Ingen nye
paths eller observationer er kørt som en automatisk erstatning her.

## 5. Kode, råreview, crashrecovery og provenance

Ny kode er kun `root/g7/restore_auxiliary.py`,
`run_auxiliary_measurement.py` og det fokuserede readonly auditværktøj.
Frosne `src`-kerner ændres ikke. SIGALRM giver lokal POSIX-jobdeadline;
parent-workerstop gælder på alle OS. Windows-native-adfærd er ikke målt.
NodeJobStore har write-ahead reservation, OS-lock, fsync og atomisk receipt.
Deadline dækker compute; receiptserialisering/reduktion indgår i målt wall-
charge og parentcap. Per-jobreservation er ikke et bevis for strengt writecap.

Første focused helpercheck:6 ownership/deadlinechecks,1.283373s process,
ingen native solverimports. Rapportens oprindeligt fejlmærkede "checked"
hashes er bevaret under `originally_reported_source_sha256`; actual checked
bytes korrigeres til den faktisk testede version. Registreringsautentifikation
og parentpredicate var efterfølgende statiske additions, ikke retestet numerik.
Fortsættelsescheck: ten full FV receipt-identities, strict Infinity-metadata
og syntaks,1.454997s body, ingen forwards. Konkrete sourcehashes er gemt.

Råreview:38 completed receipts (14 første,24 fortsættelse) og den ene
failed EM-attempt verificeres, alle5×2 EM-state/demand/pathcosts er finite,
grids og raw [0,1]-bounds kontrolleres. Gemte pathcosts rekonstrueres med
**maxfejl0**; begge vægtede EM-J og SE reproduceres inden1e−12.
Posteriornormalisering, own-prior-Bayes og budgetaritmetik består.
Succesfuldt rawreview8.384172s, ingen native kernel/forwardimports.

To tidligere readonly-auditorfejl bevares med deres sourceversioner:
`searchsorted` valgte næste gridpunkt ved floatroundoff; rettet til den
oprindelige uniform-grid-rounded index med explicit tolerance. Dernæst
brugte evidence-audit fine prior på coarse posterior; rettet til hver
egen setupcontract. Ingen af disse rettelser ændrer data, solver eller gates.
Individuelle failuretider er ukendte; ekstra20s konservativ reviewcharge
bevares frem for at kalde dem gratis. Se auditor-corrections receipt.

Begge rå-ZIP'er er varigt gemt, med whole-file SHA256 og præcis identitet i
`g7/auxiliary_archive_manifest_20261007.json`. Første22905252bytes,
fortsættelse60945256bytes. De indeholder rawarrays, tilfældige inputbytes,
ledgers, registrations, progress og logs. ZIP'ens invocation er før målt
eksporttid; **final invocation og ZIP-index er særskilt bevaret i Git**.
Source er Git-backed og findes i de frosne executioncommits. ZIP skal ikke
fejlagtigt beskrives som indeholdende den senere final-budgetreceipt.

## 6. Budget og præcis resterende opgave

| Charge | Sekunder |
|---|---:|
| Tidligere G-debit | 1234.712123334 |
| Første workerwall / ledgerincrement | 12.993654967 /9.665917149 |
| Første export /review | 4.476094963 /20 |
| Første scopecharge | 37.469749930 |
| Fortsættelse workerwall /ledgerincrement | 22.903794014 /14.337568887 |
| Fortsættelse export /review | 12.568577836 /20 |
| Fortsættelse scopecharge | 55.472371850 |
| Ekstra readonly-reviewreserve | 20 |
| Samlet nye charges | **112.942121780** |
| Ny G-debit /rest af1800 | **1347.654245114 /452.345754886** |

Hvert modelscope charges max(workerwall,ledgerincrement)+målt eksport+review,
så serialisering, imports og mislykkede forsøg ikke bortfalder. Fortsættelsen
var max160s=110worker+30export+20review,107s jobreservationer. Administrative
reservationer er stoprammer, ikke målte native rates. Ekstra review er en
separat konservativ administrative charge; samlet udført112.942s holder sig
under den oprindelige200s-ramme. Ingen hardstop eller budgetnulstilling.

Næste trin er **RNG-/admissiondesign for den resterende BM-G7-policyopgave**:
adskil streams i en ny immutable registration, afklar hvilken kort kontrol
der reelt behøves, og pris alle nye jobs før nogen modelrun. Genbrug denne
fixtures valide M254/FV-baner og de eksisterende deterministiske BM-policies.
Ingen ny WM-benchmarkrunde.

Åbent: uafhængigt RNG-/generativt støjcheck; en reel separat objective-
kvadraturforfining; budgetadmission for hele14-rækkers24h-jobgrafen; valgt
stokastisk prispolicy og dens egen fine/FV/EM-kontrol; faktisk næste forward
fra opdateret posterior. Gemte midpoints giver ikke gratis finere kvadratur.
Den gamle lineære M254-only24h-illustration41.1min er ikke en målt døgnrate
eller worst-case bound og overstiger allerede det resterende panelbudget.
Et større run er derfor ikke automatisk frigivet.

Arbejdsskøn: **2–4 timers implementering/review for de resterende G7-led**,
forudsat at eksisterende solver kan bruges og nødvendig runtime kan admitted.
Compute-tid kommer oveni og kan ikke gives præcist før komplet jobprissætning.
BM's samlede afslutning afhænger også af endeligt historisk scope/operator;
denne2h-konsistensdemo er ikke en færdig historisk BM-model. Sol High er
passende til næste implementation/design; ingen lokal kommando nødvendig nu.

## Kilder til udledning og valg

Originalmodel/basis og legacy-politik følger repoets bevarede G-plan og
vejledermateriale; dette review påstår ikke, at de manglende tavlebilledfiler
er genlæst. Totalvarians, Bayes og Beta-sampling er eksplicit udledt ovenfor.
Native implementation: `src/method/bm_1_g7.py`, `src/control/objective.py`,
`src/method/bm_1_g8.py`, `src/method/euler_maruyama.py`,
`src/method/bm_1_g8_bernstein.py`, `src/method/bm_1_g7_jobs.py` i frozen source.
Scope/RNG/grids/gates stammer fra den bevarede
`g7/auxiliary_integration_design_20261007.json`; dens RNG-fejl dokumenteres
her uden at omskrive den gamle registrering. For deterministisk energilighed
genbruges `G7_ENERGY_NEUTRAL_REVIEW.md`. Ingen nye source-attributioner til
ikke-læste PDF'er eller tavler opfindes.
