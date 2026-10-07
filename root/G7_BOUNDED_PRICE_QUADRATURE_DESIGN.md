# G7: begrænset prissøgning og selvstændig costkvadratur, 7. oktober 2026

## Beslutning og faktisk fremdrift

PR112 er merged først og PR113 derefter, med PR113 retargetet til main.
Mergecommits er ce75db9e8c92104d1092784d00abd215fd88f36f og
db44080fb2cd2f0833c203632ec86bfbd8bf1960. Der opstod ingen mergekonflikter.
Alle 29 ændrede filblobs matcher det reviewede PR113-head; src, tests og
workflows er byte-identiske med main før merges. Syntakschecks på begge
PR-heads bestod. Tidligere numeriske checks er ikke genkørt.
Den oprindelige seed-journal er fortsat ugyldig til direkte resume; merge
ændrer ikke dette. Verificeret read-only adoption og guards er nu på main.

Næste trin er konkret budgetteret som en lille prissøgning blandt tre
planer og efterfølgende kontrol af den coarse-valgte plan på fine M254,
en reel ny midpointpartition og fine FV. Designet er færdigt og kontrolleret,
men dette er ikke en ny modelkørsel: launcher/sourcefrysning/præregistrering
mangler stadig. launch_ready=false; scientific/full-day-accept=false.
G's deterministiske BM-arbejde og den tidligere forbrugsbevarende kontrol
bevares. Denne syntetiske stokastiske kontrol er en intern numerisk BM-opgave;
den er ikke januarbenchmark eller en direkte BM/WM-sammenligning.

## Hvad næste forsøg konkret skal beregne

| Trin | Beregning og genbrug |
|---|---|
| Neutral reference | Genbrug egne M254257/M254513- og FV513-nodeforwards under fulde oprindelige kontrakter |
| Begrænset søgning | Beregn M254257 på [.3,.7] og [.7,.3]; sammenlign med den gemte [.5,.5]-reference |
| Valg | Vælg laveste komplette coarse-J; præcis tie-rækkefølge er neutral, [.3,.7], [.7,.3] |
| Fine kontrol | Beregn den valgte plan på egne M254513-origins, eller genbrug neutral fine hvis neutral vælges |
| Ny costpartition | Beregn valgt M254513-plan igen med objective_max_step halveret fra 1/96h til 1/192h |
| FV-kontrol | Beregn valgt plan på FV513 med den oprindelige costpartition, eller genbrug neutral FV hvis neutral vælges |
| Rapport | Sammenlign samme-policy-resultater; rapportér J, varians, energi-/sluttilstandsforskelle, integritet og faktisk charge |

Der er højst 25 nye nodeforwards: ti coarse, fem fine, fem kvadratur og fem
FV. Hvis neutral vinder, bortfalder nye fine/FV-forwards; den nye kvadratur
kræver stadig propagation. Alle fem noder kræves for enhver bankreduktion.
Det er en begrænset søgning, ikke en kontinuerlig SLSQP/Powell-optimering
eller et certifikat for optimum på hele prisboksen. Ingen ny EM eller
optimeret-policy-feedback indgår i denne scope.

## Matematik og begrundelse for valg

De to timesblokke er [12,13] og [13,14]. For delta i {0,−.2,+.2} defineres
\[
 u(\delta)=(.5+\delta,\ .5-\delta),\qquad 0\le u_k\le1.
\]
Planerne undersøger begge retninger af en tidsmæssig pristilt omkring den
samme neutrale middelpris. Det er et eget, budgetbegrundet forsøgsvalg.
At (u_1+u_2)/2=.5 betyder ikke, at samlet forbrug er bevaret. Tilstands-
udviklingen og demand afhænger af hele prisforløbet. Ingen stokastisk
energiligning eller ny straf indføres for at få dette lille forsøg til at passe.

Med egne cutoff-vægte w_j, egne nodevise tilstandslove p_j(t;u), analytisk
B(t)=.5+.1sin(2pi t/24) og kendt A(t)=.5 bruges det eksisterende mål
\[
 J(u)=\sum_jw_j\int_{12}^{14}E_j[(D_t(u)-A(t))^2]dt.
\]
Vægtene er faste under hvert open-loop-forward, og hver kandidat nulstilles
til den samme egen startlov på sit grid. Ingen observation ved t=14 eller
senere bruges til valg af u. Målprofilen indgår direkte; der foretages ikke
et nyt parameter-/f/g-fit i denne G-opgave.

Med mu=sum_j w_j E_j[D_t] er integranden
\[
 (\mu-A)^2+\sum_jw_j\operatorname{Var}_j(D_t)
             +\sum_jw_j(E_j[D_t]-\mu)^2.
\]
Dette er kvadratudvidelse og totalvarians, ikke nye regulariseringsled.
Se [udledningen](G7_SEED_REPAIR_REVIEW.md). Selv om kvadratet er konvekst
som funktion af demand, er demand/tilstandslov en prisafhængig dynamisk
funktion. Det giver ikke automatisk konveksitet i u. Den begrænsede
kandidatregel undgår et dyrt, ubestemt antal Jacobian-/optimizerevalueringer.

På coarse-grid vælges
\[
 \delta_*\in\arg\min_{\delta\in\{0,-.2,+.2\}}J_{257,\Delta}(u(\delta)),
 \qquad\Delta=1/96h.
\]
Præcis tie-rækkefølge fastholdes før kørsel. Fine/FV/kvadratur anvendes
til validering af dette valg, ikke til efterfølgende re-selektion.
Reglen finder minimum blandt de tre komplette coarse-kandidater. Den
finder ikke minimum over alle delta eller alle u i [0,1]^2. Hvis en bank
er ufuldstændig, stoppes forsøget i stedet for at ændre kandidatlisten.

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

## Hvorfor kvadraturjobbet er reelt og særskilt

For costintegranden r(t)=E[(D_t−A_t)^2] er midpointtilnærmelsen
\[
 J_\Delta=\sum_\ell\Delta t_\ell\,r(t_\ell).
\]
Den gamle partition har 192 midtpunkter over to timer; halveret partition
har 384. De nye punkter er forskellige fra samtlige gamle midtpunkter.
Et gennemsnit af gemte midpointcosts reproducerer den gamle sum og giver
ingen nye værdier af r. Derfor kræves en ny fuld forwardkontrakt, nye
tilstandsværdier og egne receipts; interpolation tæller ikke som denne kontrol.

M254's evolve_projected_density opdeler transporten ved t_eval og pris-
breakpoints og projicerer under propagation. En ændret outputpartition
kan derfor også ændre faktiske transport/projektionsskridt. Solver_max_step
fastholdes på 1/192h; sammenligningen måler følsomhed over for cost/output-
partition ved samme deklarerede maksimumstrin. Den isolerer ikke en ren
kvadraturfejl med matematisk garanti. For en eksakt, tilstrækkeligt glat
integrand ville midpointfejlen være af orden Delta²; den antagelse og
kontinuerte fejlkonstant er ikke verificeret her. Ingen automatisk
Richardson-division af forskellen med 3 eller certificeret error-bound bruges.

Kvadratur-gaten anvender den eksisterende compare_forward: costdifference
≤max(.001,.02|reference-J|) samt bevarede mass/fordelings-/moment-/projektion-
gates. Resultatet bruges som selvstændig følsomhedskontrol af valgt policy.
Den halverede J må ikke erstatte kandidatens originale-partition-J i
neutralforbedringen, da neutral ikke nødvendigvis har samme nye partition.
EM og endpoint-feedback på den valgte policy er senere særskilte krav.

## Konkret budget, dependencies og usikkerhed

Gemte neutrale målinger: M254257-bank16.828349175s og M254513-bank37.961316554s;
første FV513-bank har3.906186455s i summerede job-stages. De siger noget om
den gamle policy/partition, ikke sikre rates for ændrede priser eller ny
kvadratur. Nedenstående reservationer er stoprammer, ikke målte bounds.

| Jobgruppe | Antal | Reservation pr.job | Sum |
|---|---:|---:|---:|
| Source-/receiptadoption | 1 | 5s | 5s |
| Fire evaluator-setups | 4 | 8s | 32s |
| To nye coarse-kandidater, fem noder hver | 10 | 6s | 60s |
| Coarse-valg | 1 | 1s | 1s |
| Valgt fine M254 | 5 | 12s | 60s |
| Valgt fine M254 med ny costpartition | 5 | 18s | 90s |
| Valgt fine FV | 5 | 2s | 10s |
| Reduktion og rapport | 1 | 2s | 2s |
| Worker-headroom | — | — | 10s |
| Stop/eksport | — | — | 30s |
| Råreview | — | — | 20s |

Jobreservationer260s; worker270s; foreslået scope320s. Rækkefølge og
dependencies er eksplicitte i [JSON-designet](g7/bounded_price_quadrature_design_20261007.json).
Quad-jobbet kan være langsommere end sin reservation; det vil stoppe scope
med bevaret deldata. Tolerancer, basis eller nodeantal må ikke lempes for
at få en ufuldstændig bank til at fremstå bestået.

Dette design/review charges konservativt10s, inklusive merge-/strukturkontrol:
G1431.082740428s → **1441.082740428s; rest358.917259572s** af1800s.
320s er foreslået, ikke udført eller allerede charged. Hvis hele scope
senere charges, bliver38.917259572s tilbage. Yderligere launcher-/checkcharge
skal først trækkes fra, og scope skal re-admitteres mod den aktuelle rest.
Ingen automatisk ny allowance, reset eller refund af gammel historik.

## Integration før numerisk start

En ny parent/worker skal implementeres. Brug den nye GuardedNodeJobStore i
et nyt scope med fuld aktuel debit, bevaret OS-lock og reservation før hvert
job. Gamle seed-/native-journaler læses kun. Verificeret FV-adoption og
read_owned/restore til egne M254 origins bruges under deres fulde kontrakter.
Ny origin/price/partition må aldrig låne et lignende gammelt job.

De 41 native sourcehashes og egne originale hour12-inputs bevares. Aktuelle
guard-/restore-/launcher-sources har et særskilt provenance-manifest; de
må ikke ændre den historiske numeriske kontrakt og derved skjule et mismatch.
Frys inputs, runtimeversioner, kandidat-/tie-regel, jobs og launcher i en
særskilt pre-run-registrering. Genskabt live setup har en ny charge.

Parent hard timeout270s og per-job stop skal bevares. Også partials eksporteres;
stop/eksportreserven må ikke bruges til flere modeljobs. Faktisk charge er
max(worker-wall,konservativ ledgerincrement)+målt eksport+råreview.
Overshoot bevares som faktisk charge og blokerer nye jobs. Afbrudte jobs
kræver en eksplicit ny retrybeslutning; ingen automatisk genberegning.
Dette design frigiver ikke den gamle one-invocation runner til resume.

## Faktisk kontrol, kilder og næste handling

Én stdlibkontrol består104 struktur-/aritmetik-/hashkontroller på .007533s:
budget, DAG, maksimum25 forwards, kandidatgrænser, forskellige kvadratur-
midtpunkter,41 sourcehashes, guard/adoption-identiteter, uændrede native
tolerancer og false-acceptflags. Ingen numerical import/forward eller gamle
numeriske tests. [Kontrol](g7/price_design_check_20261007.json),
[mergeverifikation](g7/pr112_113_merge_verification_20261007.json) og
[budget](g7/price_design_budget_20261007.json) er gemt.

Matematisk basis og totalvarians følger G7_SEED_REPAIR_REVIEW.md;
OwnedBankPriceEvaluator, combine_nodes og compare_forward er bevaret.
Kvadraturforklaringen følger de frosne midpoint_quadrature og
evolve_projected_density-interfaces samt G7_AUXILIARY_INTEGRATION_DESIGN.md.
Tilts, tie-rækkefølge, screeningmargin og320s reservation er egne designvalg.
Der er ikke foretaget nyt PDF-/tavlereview eller indført nye vejlederformler.

Næste handling er implementering/review af den konkrete launcher og separat
præregistrering; derefter kan det afgrænsede forsøg udføres. En vellykket
trepunktskontrol lukker ikke kandidat-EM, optimeret-policy-feedback,
historiske timeinterval-likelihoods, global prisoptimalitet eller24h-admission.
Det fulde døgnbudget mangler stadig sine matchende rates. G7/G er åbent;
F/WM bevares, og direkte modelsammenligning venter til BM er færdig.
