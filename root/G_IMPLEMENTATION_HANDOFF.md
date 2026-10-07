## G7: durable feedback og v2-budgetgate kontrolleret, 8. oktober

PR117 er merged c8b76c2d; denne levering bygger på main306c7af og bevarer
F-PR106's senere merge. Ingen native src/tests/.github- eller F-budgetændring.
Ny FeedbackBankControl gemmer rå posterior/evidences, projected posterior,
updateaudit og observation i én native receipt før ledgercompletion.
Reuse udfører ikke likelihood/projektion; continuation genlæser den egen
posterior med korrekt cutoff, state, koefficienter, basis og bankvægte.
Fire backend/grids og tre coarse-prisvalg er komponeret i feedback_phase;
next[36,38]-forwards bruger de egne opdaterede posteriorer. Input kræver
komplette auditerede24h-forwards under præcis samme frosne sourcekontrakt.
Ny stdlib launcher afviser unadmitted helgraf før native workerimport;
persistent one-invocation-claim, parent timeout, hele phasecap bevares.
V2-budget kræver16 whole-job-rates med samme problem/source/origin/runtime;
ukendte rates er null. Native v1-budgethelper og gamle gates bevares.
Præregistrering79c76197 er læst byte-identisk før én stub-invocation:
20 checks består, .908693493s parentwall. Native transport/likelihood/
projection solves alle0; ingen gamle videnskabelige kørsler genkørt.
Konservativ checkcharge5s: G1851.337271859/1910s; rest58.662728141s.
Gamle quad0/1/2-audits forbliver tabt. Ingen numerical/scientific/full-day-
eller historisk accept. F's historik/aktive arbejde bevares.
Feedbackfasekode er færdig; samlet døgnsearch/EM/crossreview-driver og
reelt rate-/budgetgrundlag mangler. Ingen udfyldt autoriseret numerisk
launchregistration eller24h-inputpacket: launch_ready/budget_admitted=false.
Næste kodeopgave: samlet døgnorchestration med native deterministisk/M254-
søgning og fælles reference-/EM-review. Før numerisk opstart kræves konkret
samlet budget eller et særskilt afgrænset målescope. Ingen automatisk retry.
BM færdig før direkte WM-sammenligning. Eget færdigt arbejde merges efter
relevante kontroller som besluttet af Johan; ingen branchsletning her.
Læs [matematik, integration, kontrol og begrænsninger](G7_FEEDBACK_INTEGRATION_REVIEW.md).
De ældre G7-statusafsnit nedenfor er historik.

## G7: PR115/116 merged; atomisk audit kontrolleret, 7. oktober

PR115 merge d5b06119 og historiske PR116 merge 02812cab er verificeret
mod reviewede træer. Januar-konflikten er løst med aktuel status bevaret;
gamle auxiliary-noter er historik. Fjorten andre gamle G-branches var
allerede merged. F-PR106 er draft; ingen branches slettet.
Ny opt-in AuditedBankControl gemmer forward + per-hypotese projectionaudit
inde i samme hashed/fsynced receipt før ledgercommit. Source/kontrakt binder
auditpolicy; reuse kræver gemt audit. FV har eksplicit not-applicable.
Ingen native src/tests/workflow- eller kernelændring. Separat checkpoint
b5843438 og byte-identisk registration før én stub-invocation: 6 checks
består, .782377885s process, nul native forwards. Coverage er ikke en
numerisk fitgate; update/Bayes-projektion kræver særskilt receiptintegration.
Gamle quad0/1/2-audits er stadig tabt; scientific/full-day=false.
Konservativ checkcharge5s: G1846.337271859/1910s; rest63.662728141s.
Nyt v2-forslag: 24h/4 blokke, deterministisk og primær M254-stokastisk
søgning, tre fælles policies, M254/FV/quad/pairedEM samt én observation→
update→nyt bounded prisvalg→2h-forward. 16 synlige jobtyper; rates ukendte.
Gamle 14-job-budgethelper/source bevares og accepterer ikke v2 automatisk.
Design er ikke launchregistration: launch_ready/budget_admitted=false.
Næste kode: konkret v2-integration inkl. update-receipts og samlet budget;
ingen automatisk ny numerisk kørsel. BM færdig først; WM-sammenligning senere.
Læs [adapter, matematik, merges og faktisk kontrol](G7_PER_HYPOTHESIS_AUDIT_REVIEW.md)
og [samlet forsøgsforslag](G7_INTEGRATED_EXPERIMENT_DESIGN.md).
De ældre G7-statusafsnit nedenfor er historik.

## G7: recovery afsluttet; fuld quadbank, åben projektionsaudit, 7. oktober

Nyeste status; de ældre partial-statusser nedenfor er bevaret som historik.
Johan godkendte højst110 ekstra sekunder. Separat præregistreringf208bbc3
blev committed og læst tilbage før invocation. Kun quadnode3/4 er nye;
alle13 completed forwards genbruges read-only. Hele femknudebanken er
reduceret. Coarse/fine, fine/384-midpoint og fine/FV består alle uændrede
terminal-/J-gates; J-forskelle2.64461e-5,5.95835e-9,2.07653e-7.
Neutral[.5,.5] er bedst af tre coarse-planer; ingen kontrolforbedring eller
kontinuert global optimalitet. Middelfejl og within/between-varians kommer
fra samme forwards. Tidligere quadnode0/1/2 mangler deres tabte projection-
fit auditmaxima: quad_bank_complete=true, forward_gates_passed=true,
men numerical_validation/scientific/full-day=false. Massechecks erstatter
ikke den manglende projection-L1. Ingen completed forward genkørt.
Ekstra charge42.195763999s inkl10 forberedelse/10 rawreview-reserve;
G1841.337271859s af godkendt1910, rest68.662728141s. Alle gamle charges
bevares. Fuld ZIP er gemt og CRC/SHA-verificeret efter rettet eksport.
PR114 er merged; PR115 er åben og ikke mergegodkendt. Næste kodearbejde er
atomisk projectionaudit per node før nye forsøg; resterende tid er ingen
automatisk admission. Direkte WM-sammenligning venter til BM er færdig.
Læs [matematik, resultat, begrænsning og budget](G7_COMPLETE_QUAD_RECOVERY_REVIEW.md).

## G7: PR114 merged, coarse-prissøgning færdig; quad partial, 7. oktober

PR114 merge66991a0 er verificeret:11 reviewede blobs, ingen src/tests/
workflowændringer og nyere WM4 bevaret. PR115 har launcher, separate
præregistreringer og rettelser til verifikation/inputpreflight.
To kontrollerede setupstop er bevaret med charges; nul forwards i dem.
Tredje scope gemmer10 nye coarse-forwards: neutral [.5,.5] J=.003741964
vinder mod [.3,.7] .017303376 og [.7,.3] .010655853. Ingen global optimalitet
eller forbedring påstås. Middelfejl/varians rapporteres fra samme forwards.
Tre af fem nye quad-forwards completed; node3runningreserve18s, node4unstarted.
Parent-session tabt; årsag ukendt og ingen final wallmåling.19complete
receipts/13nye forwards er rawreviewverificeret3.807s uden propagation.
Journal/arrays/source/inputs er bevaret; gammel scope må ikke genoptages.
Fuldt tabsscope273s bevares: G1799.141507860s af1800; rest.858492140s.
scientific/quad/full-day=false. PR115 og fuldt råarkiv dokumenterer partials.
Næste er særskilt recovery af kun quadnode3/4 og fuld reduktion, hvis et
nyt konkret budgetforslag110s ekstra godkendes. Forslaget er gemt, ikke
admitteret/udført. Bevar13complete forwards og alle gamle charges.
F/WM og direkte sammenligning forbliver uden for den aktive scope.
Læs [matematik, resultater, fejlrettelser og budget](G7_BOUNDED_LAUNCHER_REVIEW.md).

## G7: PR114 merged; launcher og særskilt præregistrering færdige, 7. oktober

PR114 merge66991a0 bevarer alle11 reviewede blobs og nyere main/WM4; ingen
src/tests/workflows ændres. Gamle rådata og41 native sources er gendannet
og hashverificeret. Ny parent/worker bruger opt-in GuardedNodeJobStore,
fulde kontrakter, read-only adoption, én invocation og270s hardstop.
Én stdlibcheck:27 checks,.042779s, nul numerical imports/forwards.
Gamle kontroller gentages ikke. Matematik/totalvarians/selection/quad er
dokumenteret sammen med antagelser og begrænsninger.
Recovery/launchercharge35s: G1476.082740428s; rest323.917259572s.
320s scope re-admitteret, endnu ikke charged; maxrest3.917259572s.
Separate registration/launch_ready=true; numerical/scientific/full-day=false.
Registreringen commits før én faktisk kørsel. Ingen automatisk retry,
energiligning, kandidat-EM/feedback, F/WM eller døgnadmission i scope.
Læs [launcher, matematik, kontrol og budget](G7_BOUNDED_LAUNCHER_REVIEW.md).

## Aktiv afgrænsning: BM færdig først, WM-sammenligning bagefter

Johan, 7. oktober 2026 kl.16:25: G-arbejdet skal udvikle og færdiggøre BM.
Direkte BM/WM-sammenligning, fælles benchmarkkørsler og rangering udskydes,
til BM er færdig. WM-resultater er ikke en adgangsgate for BM's næste trin.

BM skal under udviklingen være meningsfuldt sammenlignelig med WM:
dokumentér tidsakse/horizon, informationscutoff og dataopdeling, input,
baseline/målprofil, normalisering/enheder, prisdefinition og tilladte
kontroller, forbrugsbegrænsninger, metrics og outputformat. Genbrug relevante
WM-konventioner; begrund nødvendige modelforskelle. Det kræver ikke løbende
WM-kørsler eller tilpasning af BM til WM's resultater.

BM's deterministiske kontrolmål og varianten med bevaret samlet forbrug
bevares som BM-opgaver. Stokastiske M254/FV/EM-kontroller er interne
numeriske BM-kontroller og fortsætter. Vejlederbasis/legacy og adskillelsen
mellem G og F bevares.

Tidligere gennemførte BM/WM-forsøg og råresultater bevares som historisk
evidens, men er ikke den aktive udviklingsprioritet eller en endelig
modelsammenligning. Denne afgrænsning overstyrer ældre formuleringer om
"primær BM/WM-sammenligning", også i det gemte auxiliary-designsnapshot.
Det ændrer ikke snapshotbytes, numeriske gates eller beregningsbudget.
Næste G7-opgave fremgår af den nyeste status nedenfor. Selve BM/WM-
sammenligningen planlægges efter BM's færdiggørelse.

## G7: merges verified; bounded price/quad design ready, 7. oktober

PR112 og PR113 er merged i rækkefølge på main uden konflikter. Alle29 ændrede
blobs matcher reviewet G-head; src/tests/workflows er bevaret byte-identiske
med main før merge. Begge PR-heads har bestået syntax; gamle numeriske checks
er ikke gentaget. Originaljournalens resume-afvisning bevares.

Næste afgrænsede scope er designet: neutral [.5,.5] genbruges; to nye
M254257-kandidater [.3,.7]/[.7,.3] undersøges. Coarse vælger én plan med
fast tie-regel; valgt plan kontrolleres på M254513, reel ny costpartition
1/192h og FV513. Højst25 nye nodeforwards, ingen EM/feedback i dette scope.
Lige middelpris er ikke energibevarelse. Ingen kontinuert/global optimum-
claim; fine-J-gain og energi-/sluttilstandsforskelle rapporteres separat.
Kvadratur ændrer output/transportpartition og er ikke et rent fejlcertifikat.

Jobreservationer260s +10s worker-headroom; foreslået scope320s =
270 worker +30 stop/eksport +20 råreview. Nye policy/quad-rates er ukendte;
reservationer er stoprammer, ikke runtimebounds. Én stdlibdesignkontrol består
104 checks på.007533s med41 sourcehashes, DAG, budget, gates og distinkte
midpoints; nul native imports/forwards. Source/basis/inputs og F/WM bevares.
Design/reviewcharge10s: G1441.082740428s af1800; rest358.917259572s.
320s er endnu ikke charged; worst-case-rest38.917259572s før nye setup/check-
charges. Budget skal re-admitteres efter yderligere implementation-charge.

launch_ready/pre_run_registration/actual_optimizer/scientific/full-day=false.
Næste er konkret parent/worker med GuardedNodeJobStore, eksplicit read-only
adoption, sourcefrysning og særskilt præregistrering, derefter én kørsel.
Ingen gammel one-invocation-resume eller automatisk retry. Valgt-policy EM,
feedback, historiske intervalobservationer og døgnadmission er senere krav.
Direkte BM/WM-sammenligning venter til BM er færdig.
Læs [matematik, design, budget og næste trin](G7_BOUNDED_PRICE_QUADRATURE_DESIGN.md).

## G7: receipt-genbrug og writer-guards færdige, 7. oktober

24 seed-run-receipts er verificeret i et eksternt hash-låst read-only indeks;
alle ti egne FV-resultater kan genbruges ved eksakt original jobidentitet.
Et G-lokalt GuardedNodeJobStore-interface bevarer native lock/budget/retry og
kontrollerer samlet journal før jobs, reservation før compute, tidligere
historik og completion-write ved readback. Brud stopper scope med separat
markør. Adapteren skal vælges eksplicit og sourcefryses i næste launcher;
den gamle one-invocation runner er ikke ændret eller kørt igen.

Én stdlibkontrol består:12 cases,24 faktiske receipt-/payloadhashes,4.571s,
seks stub-computes, nul model-forwards/numeriske imports. Orphan/missing/
tampered files, lost reservation/completion, stale overwrite og crash/retry
kontrolleres. Eksakte native writer-definitioner bruges via AST; ingen fuld
solver-, Windows-, concurrency- eller power-loss-integrationstest påstås.
Originaljournal/raw-ZIP er bevaret byte-identisk. Årsagen til journaltabet er
stadig ukendt; source_journal_integrity_passed=false og original resume=false.
Eksplicit read-only adoption er verificeret; ingen scientific/24h-accept.

Præregistreret konservativ scopecharge30s inkluderer verifikation/review.
G1431.082740428s af1800; rest368.917259572s. Alle gamle charges bevares;
gamle receipts genberegnes/gencharges ikke. Næste trin er konkret budgetteret
prisoptimerings-/selvstændig costkvadraturdesign med den nye guardintegration
før nye forwards. F/WM/native/shared kilder bevares; BM/WM-sammenligning venter.
Læs [interface, begrundelse, kontrol og næste trin](G7_RECEIPT_RECOVERY_REVIEW.md).

## G7: seed-rettelse og kort EM/feedback-kontrol færdig, 7. oktober

Én ny præregistreret120s-opgave er gennemført på egne gemte hour12→14-data.
Alle ti M254 og alle ti FV-forwards genbruges; fem nye coupled EM-par
(256 paths/node) og fire nye endpoint-updates. Separate SeedSequence-role/node
identiteter giver11 forskellige integer-streams; tilsigtet coarse/fine-coupling
bevares. Hele worker-AST bortset fra to seed-lookups/collisionguard er bevaret.
Ét fokuseret stdlibcheck består(.022s); gamle tests/modelreplays ikke genkørt.
Råreview(8.916s) reproducerer præcist egne initialdraws, Brownian-inputs,
observationsstøj, pathcosts, standardfejl og egne Bayes-vægte. Kort numerik
består; fineEM J=.003668253468, SE=.000088703283, nul boundaryviolations.
RNG-uafhængighed fortolkes under den almindelige PRNG-antagelse; ingen
scientific/full-day/kvadratur-/optimizer-accept.

Ny journalafvigelse: coarseFV-reuse0/1 mangler entries, fineFV-reuse4 står
running, men alle24 complete raw payloads er hashverificeret. Originaljournal
og failed auditor bevares; sidecar-reconciliation charges mindst fulde caps
for manglende/running jobs.19.336s dækkes af parentens20.996s; ingen refund.
journal_integrity_passed=false, receipt_reconciliation_verified=true.
Næste er G-afgrænset journal-/receipt-recovery før nye optimizer/resumejobs;
årsagen er ikke fastlagt, og sidecaret giver ingen automatisk resume-adgang.
Ingen nye numeriske replays kræves for dette read-only recoveryarbejde.

Scopecharge53.428495314s inkl.ekstra eksport/review; G1401.082740428s af1800,
rest398.917259572s. Gamle charges bevares. Fuldt raw reviewarkiv er gemt.
Seed-opgaven er afsluttet; G7/G som helhed er stadig åbent. Derefter budgetteret
prisoptimering og selvstændig quadratur; ingen24h-admission fra2h-rates.
F/WM bevares, og direkte modelsammenligning venter til BM er færdig.
Læs [matematik, resultat, journalbegrænsning og næste trin](G7_SEED_REPAIR_REVIEW.md).

## G7: neutral auxiliary-kørsel færdig; råreview begrænser accept, 7. oktober

Den afbrudte chat genoptages fra completed resultater, ikke fra et nyt modelrun.
Ti FV-forwards og ti coupled EM-level-forwards samt fire endpoint-opdateringer
er gemt. De ti egne tidligere M254-nodeforwards er genbrugt; ingen ny M254,
prisoptimizer, historisk observation eller WM-kørsel. Første forsøg fejlede
kun JSON-eksport af max_step=Infinity efter EM-node0; dets data og charge
bevares. Eksplicit registreret continuation genbrugte alle ti FV-resultater.

Alle registrerede korte forward-/feedback-/projektionstal består. Råreview
verificerer 14+24 completed receipts og rekonstruerer EM-pathcosts og Bayes-
vægte uden native forwards. **Overlappende initial-/Brownian-seeds på tværs
af noder og noise-seed lig truth-nodens initial-seed betyder, at uafhængig
MC-standardfejl og uafhængig syntetisk observationsstøj ikke er verificeret.**
Numerisk backend-sammenligning ved det gemte faste endpoint kan bruges;
scientific_accept, independent_quadrature_accept, price-search og
full_day_budget_admitted forbliver false. Workerens rå passed-flag ændres
ikke; denne fortolkning og raw-review-JSON er autoritative for accept.

Hele budgethistorikken er bevaret: task112.942121780s af oprindelig200s,
G1347.654245114s af1800s; rest452.345754886s. To failed auditorversioner
og konservativ ekstra20s reviewcharge er med. Crash-recovery læser gemt
evidens og laver ingen ny modelkørsel eller gentagelse af beståede tests.
Begge fulde rå-ZIP'er er gemt og hashverificeret.

Næste G7-opgave: særskilt registreret minimal rettelse af RNG-streams og
endpoint-generation, der genbruger M254/FV og kun genberegner de berørte
EM-par og feedback. Frys kilde/streamidentiteter og totalcharge før kørsel;
ingen automatisk retry eller nyt budget. Derefter selvstændig kvadratur,
faktisk prisoptimering og konkret policy-/døgnbudgetbeslutning.
Direkte BM/WM-sammenligning venter stadig til BM er færdig; F/WM bevares.
Læs [resultater, matematik, begrænsninger og recovery](G7_AUXILIARY_RUNTIME_REVIEW.md).

## G7: auxiliary-budget og integrationsdesign færdigt, 7. oktober

Deterministisk BM/WM med bevaret døgnforbrug forbliver primær; færdige
fits/policies og råevidens genbruges. Supervisor M254/legacy og F/WM bevares.
Næste måleforsøg er konkret designet som neutral12→14h: genbrug de ti
gemte M254-noder, nye egne FV257/FV513-forwards, coupled EM256/node og
samme nye syntetiske endpoint-observation på fire backend/grids.
Ingen M254-forward, optimizer eller nyt G8-filter i det foreslåede scope.
Maks200s =150 worker +30 stop/eksport +20 review; jobreservationer130s,
workerheadroom20s. Reservationer er stoprammer, ikke målte rates/bounds.
Read-only restore med fuld gammel kontrakt/ny setupcharge og ny launcher-
provenance skal implementeres og sourcefryses før separat præregistrering.
launch_ready=false; scope er endnu IKKE udført/charged. Ingen auto-retry.

Vigtigt: gemte outputmidtpunkter giver ikke gratis kvadraturforfining;
gennemsnit af to midpoint-costs reproducerer blot samme sum. En ny
quadraturpartition kræver eget forwardjob/kontrakt og forbliver ukendt.
14-rækkers døgnplan og gates bevares; 2h-demo kan ikke give24h-admission.
Ingen sammenligning af J fra den analytiske2h-fixture med januarens24h-par.

Én stdlib struktur-/aritmetikkontrol består44 checks på.010431s; ingen
native import, modelrun eller gamle tests. Design/reviewcharge10s:
G1234.712123334s, rest565.287876666s. F/WM-budgetter er uændrede.
Næste: implementér/review restore+auxiliary-launcher, registrér og mål
det konkrete neutrale scope; derefter beslut policy-/døgnbudget på faktisk evidens.
Sol High; ingen lokal kommando nu.
Læs [design, udledning, budget og resterende arbejde](G7_AUXILIARY_INTEGRATION_DESIGN.md).
Følgende G-statusafsnit er historik.

## G7: eget M254-nodeforsøg gennemført, 7. oktober

10 native nodeforwards på gemte EGNE hour12-posteriorer, neutral12→14h,
M254513 først37.961317s, M254257 derefter16.828349s inkl.nodefiler/reduktion.
Jfine=.003715517464, Jcoarse=.003741963573; Jfejl2.644611e-5<.001.
Conditional/marginal slut-L1 .011187752/.005225045<.05; alle korte
moment/CDF/kvantil- og projektiongates består. Ingen FV/EM/optimizer/feedback.
Første invocation stoppede før forward ved bit-exact basisreload; original
fejl/source/input/ledger er bevaret. Optional verified-byte loader bevarer
gamle basisværdier/c og default strict guard; machine-roundoff-familiecheck
med eksakt owned-bytehash er begrundet. Ingen model-/acceptgate lempet.
Ét loadercheck består1.609s; gamle node-/modeltests ikke genkørt.
12 receipts og41 sourcehashes råreviewet; gemte demandmomenter/J rekonstrueres
identisk uden nye forwards. Fuld data-ZIP er gemt separat og hashverificeret.
Native source/kode og originalinputbytes er gemt i Git; vejleder M254 bevares.
G-debit1224.712123s, rest575.287877s; begge forsøg medregnet, F/WM uændrede.
2h-rates giver ingen24h-admission. En ren lineær M254-cross-illustration er
41.1min før andre jobs, ikke målt døgnrate/bound. Ingen automatisk større run.
Næste er budgetteret FV/EM/feedback-/policyintegrationdesign med recovery,
derefter eksplicit døgnadmission. Sol High; ingen lokal kommando nu.
Læs [forsøg, matematik, råreview og resterende arbejde](G7_NODE_RUNTIME_REVIEW.md).
Følgende G-statusafsnit er historik.

## G7: egne nodecheckpoints implementeret, 7. oktober

Durable nodejobs bevarer hver rho-nodes egne M254-c/posterior, eksakt
prisplan og original transportpartition. Atomiske råbaner og write-ahead
reservationer genbruges efter crash; afbrudte noder kræver eksplicit retry
med bevaret charge. Persistent kampagne- og studycap; én OS-locked ejer.
36 kildehashes/full origin/numerik/versioner indgår i kontrakten.
Native mixturekald, bankreduktion og assimilate er AST-identiske med før.
Én fokuseret stubtest består; anden invocation efter konkret caprettelse
består også. Begge receipts bevares;4.129374s process i alt. Ingen faktisk
M254/FV/EM-forward, nye fits eller gamle numeriske checks genkørt.
Komplet14-jobgraf inkl.FVfine/quad/coupledEM/feedback/setup/export er gemt.
Alle rates er ukendte; budget_admitted/scientific_accept/optimizer=false.
Kooperativ deadline er ikke hard kill; næste launcher kræver stopreserve.
Vejlederbasis og legacy bevares; F/WM-kilder og budgetter er uændrede.
G-debit1118.580078s, rest681.419922s efter konservativ20s scopecharge.
Næste er særskilt kort registreret faktisk måling fra hashverificerede
egne origins, ingen automatisk gentagelse af den gamle60s-pilot.
Sol High; ingen lokal modelkommando nu.
Læs [matematik, crashmekanik, budget og faktisk kontrol](G7_NODE_JOB_REVIEW.md).
Følgende G-statusafsnit er historik.

## G7: deterministisk døgnkontrol med bevaret forbrug, 7. oktober

Fælles frossent januarpar og lineær BM-reference er optimeret over fire6h-
prisblokke, både original fixed-A og separat lighed integral(demand-B)=0.
Alle seks valgte policies består coarse/fine;374 faktiske forwards inkl.
Jacobianprober,3.7754s process. Neutral fine-J: lineær BM .275408888,
G4 .276532280, WM .303629020; baselineJ .308609546. Forbedringer10.758%/
10.394%/1.614% er modelbetinget tracking i ét døgn, ikke fysisk/kausal ranking.
WM løses som certifieret affin/konveks4D-QP; BM er bedste fundne lokale
SLSQP-kandidat fra to starter, uden global optimalitetsclaim. Ingen refits.
Samme diskrete forbrugsbalance bruges i J og lighed; WM's Euler/midpoint-
korrektion er udledt. Fine energierror højst1.466e-6 i neutrale policies.
Én fokuseret analytisk kontrol og read-only byte/J/E-audit består.17 units/
19 råfiler, fulde baner og hashes er gemt med atomiske checkpoints.
S3-fladhed genbruges som identifikationsforbehold; ingen ny straf indføres.
M254's gamle døgnestimat mangler eksplicitte FV/EM/feedback/quad-led:
ingen ny stochastic admission eller pilotretry. Næste er durable egne
rho-nodejobs og komplet målt jobgraf før særskilt registreret måling.
Vejlederbasis, legacy, F/WM-kilder og deres budgetter er uændrede.
Konservativ nattecharge30s af max300s; G1098.580078s, rest701.419922s.
Læs [matematik, faktisk udført og resultater](G7_ENERGY_NEUTRAL_REVIEW.md)
og [M254-runtime/jobdesign](G7_M254_RUNTIME_JOB_REVIEW.md).
Følgende statusafsnit er historik.

## G7: fælles BM/WM-kontrakt og frozen-fit-prisrespons, 7. oktober

24h/fire blokke/known fixed-A med fælles januar-baseline og scales er låst.
A er baseline-døgnmiddel, uden target-Y; WM's native A(U)-opgave bevares.
Time/dage og u_BM=u/U_WM=2u-1 omregnes eksplicit; hver kandidat bruger
sin gemte origin. Januar-G4/lambda0 og matched januar-WM4 genbruges sammen
med original lineær BM-reference; nyere S3 har ikke matched WM-origins.
Én kort analytisk kontraktkontrol består.18 forwards (3 priser/3 modeller/
2 grids) består alle gates; .212s body/.505s process. Ingen optimizer/refit.
S3-audit:1008/1008 gemte forecastpriser u<.5; uge16–18 har næsten flad
g netop dér. Stærk respons over knækket er ikke kausalt dokumenteret.
J alene kan falde ved ændret slutlager/energi; rapportér terminal og
energy-gap separat, ingen skjult ny straf. Raw demandsemantik er endnu
legacy-uafklaret; ingen kW²h- eller fysisk ranking-claim.
18 gamle importerede Git-blobs matcher executionbase; WM deterministic
AST er scoped-reviewet, ny validation-only guard består. F/WM/legacy
kilder, basisfamilier, budgets og negative G7/M254-resultater bevares.
Næste: lille fireblokoptimering af samme januarpar, native feasibility,
J/energy/terminal rapportering. S3's senere historiske kontrol er ikke
frigivet. Stokastisk G7 obligatorisk; runtime-/jobdesignreview før retry.
G-debit1068.580078s, rest731.419922s. Sol High; ingen lokal kørsel nu.
Læs [kontrakt, udledning, kilder og faktiske resultater](G7_MATCHED_CONTROL_REVIEW.md).

Følgende daterede statusafsnit er historik.

**G7 deterministisk kontrol gennemført, 7. oktober 2026:** Begge kontrolmål beholdes med deterministisk BM/WM-reference først. Original BM i FP2025 bruger kendt exogent A(t); en ny WM-afledt A(U)-funktion er ikke nødvendig. Én syntetisk24h-case med fire prisblokke, fixed A=.5 og lineær reference giver fine J=.0209268019845 mod neutral .12 (82.561% mindre). Begge bounded least-squares/TRF-starter konvergerer via ftol, uden global/KKT-optimalitetsclaim. Alle coarse/fine-gates består: J-fejl6.78073e-7, statefejl1.32738e-7, demandfejl5.55150e-8. 68 faktiske forward-solves inklusive Jacobian-prober, invocation14.227s. Én fokuseret analytisk kontraktkontrol består efter rettet assertion ved aktiv bound; failed check bevares, produkttolerancer uændrede. 27 gamle importerede Git-blobs og execution-hashes matcher. F/WM/shared-kerner er uændrede. **Syntetisk deterministisk udviklingsaccept; faktisk BM/WM-rangering, historisk kontrol og stokastisk G7 er åbne.** Næste: fælles fixed-A benchmarkkontrakt, ens inputs/prisblokke/enheder og eksplicit initialisering; bevar WM's native A(U)-mål separat. G4-fits genbruges og svag prisrespons diagnosticeres før fysisk kontrolfortolkning. M254-prisoptimering afventer stadig runtime-/jobdesignreview; ingen automatisk genkørsel. Conservative paneldebit1062.580078s, rest737.419922s. Sol High er tilstrækkelig; ingen lokal kørsel nu. Se [udledning, kilder, matematik og råresultater](G7_DETERMINISTIC_CONTROL_REVIEW.md). Følgende daterede statusafsnit er historik.

## Aktiv kontrolprioritet — Johan, 7. oktober 2026

Det deterministiske kontrolmål er primær reference til BM/WM-sammenligningen. Det stokastiske mål beholdes som obligatorisk senere G7-leverance. Original BM bruger fast A(t); en prisafhængig indkøbsfunktion er en særskilt udvidelse og ingen adgangsgate. Fælles kontrolmål, input, prisconstraints, skala og informationsgrundlag skal låses før modelrangering. Tidligere P6-tekst om stokastisk førsteprioritet læses med denne rækkefølgeændring; den afsluttede syntetiske pilot genkøres ikke.

**G7 own-posterior-handoff implementeret, 7. oktober:** Den nye G-adapter bruger vejlederens typed M254-basis og hver rho-nodes egne koefficienter; priser evalueres fra samme frosne posterior med fuld within-/between-parameter-varians. Én fokuseret reset/missing/ownership-kontraktkontrol består (.410 s). **Den faktiske runtimepilot stoppede ved 60.001 s under første fine forward; grove to timer var færdige efter afrundet 17.61 s elapsed. Ingen parret precisiongate, prisoptimering, EM eller faktisk observations-feedback er gennemført.** G7 er ikke accepteret; G8's tidligere accept bevares. Original executed source og negative partials er arkiveret. Runneren gemmer nu completed forwards straks og håndhæver marginal-/feedback-gates eksplicit; dette eksport-/guard-fix er syntakskontrolleret uden retry. Næste er et afgrænset runtime-/jobdesignreview før en særskilt registreret pilot; ingen automatisk større run eller løsere gate. Conservative paneldebit inkl. arkiv er 1045.580 s, rest 754.420 s; gamle umålte probes dækkes fortsat af den tidligere reservation. Ingen lokal modelkommando kræves nu. Se [matematik, kilder, faktisk udført og begrænsninger](G7_OWNED_HANDOFF_REVIEW.md). De følgende statusafsnit er dateret historik.

**M254-forfining gennemført, 6. oktober:** Eget Bernstein-filter med 513 celler og max_step=1/192 gennemfører samme 24 timer fra egen eksakte prior. Alle parrede conditional-/marginalgates mod gemt M254257 og FV513 består; faktisk JSON-resume time22→24 er identisk, også i conditional evidence. Mod FV513: conditional L1 0.00354065, marginal L1 0.00108111, nodevægtforskel 0.000265417; største projektion L1 0.000971943. Filter 309.743 s; resume 17.327 s; invocation 327.829 s. Vejlederens 255 Bernstein-funktioner, model, likelihood og tolerancer er uændrede. **M254-repræsentationen er klar til næste G7-interfacearbejde på denne syntetiske case; G7-mixture-prisforward og prisstyring er endnu ikke implementeret eller accepteret.** Næste er et typed own-origin-handoff og kort prisforward-runtimepilot før et konkret optimizerbudget. Ingen lokal kørsel kræves nu. Se [matematik, valg, resultater og begrænsninger](G8_M254_REFINEMENT_REVIEW.md). Statusafsnittene nedenfor er dateret historik.

**M254-filterforsøg gennemført, 6. oktober:** Vejlederens oprindelige Bernstein-familie er implementeret som eksplicit celleintegreret M254 med eksakt gradforhøjet Beta(20,20)-prior og typed checkpoints. Eget fem-node-filter gennemfører24 timer fra prioren; alle conditional- og marginalgates mod gemte FV257/FV513 består. Maksimal conditional L1 modFV513 er0.0295811, marginal L1 er0.0126685, Bayes/substep-projektion er0.000963871. Tre målrettede checks består; faktisk JSON-resume fra time22 til24 er identisk i masser, koefficienter, vægte og evidence. Filter117.873 s; hovedinvocation119.901 s; resume6.204 s; samlet126.105 s før råarkiv. 33 øvrige importerede kodefiler matcher main c64198b. **Eget513-replay blev ikke admitted inden for300s-budgettet; egne mixture-forfiningsgate og mixture→G7-accept er stadig åbne.** Dette er et positivt257-udviklingsresultat, ikke et negativt basisresultat eller historisk accept. Næste er én særskilt budgetteret513-forfining med samme basis/model og checkpoint; Sol High er tilstrækkelig. Ingen lokal kørsel kræves nu. Se [matematik, valg, resultater og begrænsninger](G8_M254_FILTER_REVIEW.md).

**Vejledermateriale/legacy, Johan 6. oktober:** Tavlebillederne er vejlederens
grundlag. Originalmodeller og resultater bevares; udvidelser kræver udledning,
kilder, begrundelse og eksplicit sammenligning. Den nye basis er kun reference,
og dens allerede gennemførte G8-kørsler genbruges. Se
[detaljeret Beta-udledning, kildekort og designargument](G8_POSTERIOR_BASIS_REVIEW.md#vejledergrundlag-og-referenceudvidelse--dokumentationsregel).
Den oprindelige noteændring var dokumentation alene; det efterfølgende
M254-review er registreret særskilt nedenfor.

## Aktiv basisbeslutning — Johan, 6. oktober 2026

Vejlederens normaliserede Bernstein/Beta-basis er G's primære mixture-basis:

\[
\ell_m(x)=\frac{(M+1)!}{m!(M-m)!}x^m(1-x)^{M-m},
\qquad m=0,\ldots,M.
\]

Den nye faste 42-leds Beta-basis beholdes alene som sammenligningsreference.
FV er fortsat den numeriske reference; metodeprioritet og numerisk accept er
forskellige beslutninger. De gemte negative M38/M78-resultater og positive
42-leds-resultater bevares uændret. M38's åbne G6b-propagation består, men
M38/M78 er ikke accepteret til G8's posterior→G7-kæde.

**G8-review gennemført, 6. oktober:** Én statisk kandidat, M254 med 255 af
vejlederens oprindelige Bernstein/Beta-funktioner og eksplicit celleintegration,
består syv afgrænsede LS-kontroller på gemte love. Maksimal posterior-L1 er
1.27144e-6; invocation9.596 s. Gradforhøjelse bevarer Beta(20,20)-prioren
med L1-fejl3.18e-14. De eksisterende produktionskerner og legacy-defaults er
uændrede. Dårlig konditionering består; dette er repræsentationsaccept på
udvalgte eksponerede love, ikke accept af et rekursivt filter eller G7.

**Det planlagte M254-filterforsøg er nu gennemført; aktuel status står øverst.**


G7's primære mixture-handoff skal komme fra en kontrolleret kørsel med den
oprindelige basisfamilie og hver nodes egne koefficienter/posterior.
Det gemte bank_local257-origin må kun bruges som eksplicit 42-leds-reference,
ikke som primær origin eller som skjult restart for Bernstein-filteret.
Bevar modellen, likelihooden, simplex-constraints og accepttolerancer;
ingen større målefejl, signed koefficienter eller FV-fallback forklædt som
mixture for at få den primære metode til at bestå. Hvis den primære basis
ikke består inden for praktisk budget, dokumenteres begrænsningen.

Denne beslutning erstatter tidligere anbefalinger om 42-leds-basis som
G7's primære startkonfiguration. De tidligere resultater bevares uændret.
Det efterfølgende M254-review ovenfor tilføjer kun statisk evidens.

**G8 punktfilter/bank gennemført som afgrænset integration, 6. oktober 2026:** FV kendt-parameter-filter og fem-node-rho-bank består forfining, missing og faktisk JSON-resume. 24 syntetiske timer/22 målinger; predictive demand-RMSE .0208264 mod åben .0286866 (27.40 % mindre på denne ene bane). Banken identificerer ikke sand rho sikkert. M38/M78 gennemfører, men begge afvises efter Bayes-opdateringer: kendt posterior-L1 .498924/.139169 og bank .500165/.138551 mod .05. M78's gode mean er ikke fordelingsaccept. Fuld-bank→FV fast kontrolforslag bevarer egne nodeposteriors og parameterusikkerhed; ingen prisoptimering. V2 invocation49.112 s, tre fokuserede tests består, fælles kerner/F/WM uændret. Se [matematik, råreview og begrænsninger](G8_POINT_BANK_REVIEW.md). **G8 mixture er ikke accepteret: næste lille opgave er posterior-repræsentation før mixture→G7; FV-handoff er klar til separat G7-referencearbejde.** Historisk timeenergi og ugeforecast er ikke accepteret. Ingen lokal kørsel kræves nu. Ældre næste-trin-afsnit nedenfor er historiske.

**G8 posterior-basis afklaret, 6. oktober 2026:** En fast celleintegreret ordbog med 41 smalle Beta-led (kappa256) plus Beta(20,20)-prioren sænker Bayes-projektionsfejlen fra .128495 til .001494 allerede ved samme129-grid. 42 koefficienter bruges i stedet for M78's79. En ny conditional-node-kontrol afviste129-cellers grid for både den gemte FV-bank og ny mixture; det negative første studie bevares. Én målrettet257→513-forfining består nu for hver node og marginalen. **Historisk anbefaling, nu erstattet af Johans basisbeslutning ovenfor: 42 komponenter/257 celler er kun reference for G7.** Alle31 eksisterende importerede shared-moduler er uændrede; kun G-adapterens basisvalg udvides. Faktisk JSON-resume er identisk; to fokuserede tests består. Nye numeriske invocations150.086 s i alt. Se [udledning, kildegrundlag, valg og råreview](G8_POSTERIOR_BASIS_REVIEW.md). Næste G7-evaluator skal bruge originens faktiske basis og egne nodekoefficienter, ikke rekonstruere M38. Ingen prisoptimering, historisk forecastaccept eller parameterrecovery i denne leverance. G4/F/WM uændret; ingen lokal kørsel kræves nu. Tidligere blokering nedenfor er løst for denne udviklingscase, ikke universelt.

# G3 og frem — grundlag og overdragelse til Sol

**G6b-panel gennemført, 6. oktober 2026:** Én låst syntetisk 24-timerscase med samme Beta(20,20)-prior, lineær f/g og fast sigma_x=0.2. FV består rum-/tidskontroller. M20 afvises (initial L1=0.291667, statevarians/kvantiler og demand-CDF); M38 matcher FV (maks. demandmiddelforskel 0.0001061, J-forskel 0.0221 %). EM2048 stemmer med FV inden for registrerede praktiske tolerancer og punktvise sampling-SE; nul rå randoverskridelser. FV er primær reference og hurtigere end M38 her. V3-hovedpanel 7.101 s, invocation 8.840 s plus separat genbrugt M20-pilot 1.120 s; fejlede interfaceforsøg er bevaret uden komplet samlet timing. To fokuserede tests består; ingen nye fits eller shared-kernelændringer. Dette er numerisk udviklingsevidens, ikke historisk forecast-/intervalaccept. Se [matematik, valg og råreview](G6B_DISTRIBUTION_PANEL_REVIEW.md). **Næste: G8 kendt-parameter-statefilter, derefter lille rho-bank og G7.** Ingen lokal kørsel kræves nu. Ældre næste-trin-afsnit nedenfor er historiske.


**G6a-reference færdig, 6. oktober 2026:** Konstant støj kalibreret med analytisk nonnegative LS på fem tidligere matchende weekdays, genbrug af S3/lambda=0 uge13+18. 336 målte timer; procesbånd dækker 45.83 % mod 88.99 % for historisk spredning omkring samme G4-center. Uge13 lineær statevarians overskrider den fysiske bound1/4; båndene er ikke accepteret som kalibrerede 95 %-intervaller. 168 propagationer/2.283 s, begge tidsforfininger og tre fokuserede tests består. Ingen nye strukturelle fits eller støjfamilier; F/WM uændret. Se [matematik og negativt G6a-resultat](G6A_LINEAR_NOISE_REVIEW.md). **Næste: G6b med FP/FV, Beta-mixture og EM på det faste lille syntetiske panel.**


**G4 afsluttet for den afgrænsede punktprotokol, 6. oktober 2026:** Den fastlåste S3-sammenligning på nye ISO-uger 13–18 er complete: seks fælles mandagscutoffs, 1008/1008 målte timer. Uden straf har MSE 0.1798347601 mod 0.1843188395 med lambda=0.001 og baseline 0.1806157175. Straf øger MSE 2.4934 %, vinder kun 2/6 uger og taber alle leave-one-week-out-sammenligninger. Det registrerede modelvalg er derfor **S3 med lambda=0**, lineær f og fælles to-stykkers g over 35 historiske dage; ingen nye kurver eller lambdaer. Baselinegevinsten er kun 0.4324 %. Bounds, næsten flad prisrespons i tre senere fits og 207 observationer under den frosne demand-skala begrænser fysisk/prisstyringsfortolkning. Dette er fortsat betingede prognoser med realiseret pris/temperatur på legacy-punktkontrakten, ikke en uafhængig sluttest eller operationel forecast-accept. Alle 12 valgte kandidater består tidsforfining; to ugefits krævede 24 trin/time. Runnerinvocation 297.03 s, to fokuserede tests og syntakskontrol består. Præcis 24 fysiske warmup-timer efter sommertid er rettet; tidligere fits er bevaret uden genkørsel. Se [matematik, valg og råreview](G4_VALIDATION_WEEKS_13_18_REVIEW.md). **Næste: G6a-variansreference, derefter G6b→G8→G7; G5 dokumenteres løbende og afsluttes særskilt.** Ingen lokal kørsel kræves nu.


**Nyeste G4-forsøg, 6. oktober 2026 — 12 uger og tre strategier:** Syv fælles mandagscutoffs (uge 1–5→6, …, 7–11→12), 49 dagsprognoser og 1175/1175 fysiske målte timer er komplette. S1 er retrospektiv (C,k,Xi fra alle 12 uger); S2 fitter C,k,Xi,a per ugedag; S3 fitter dem samlet over 35 dage. S3 med lambda=0.001 har lavest MSE 0.2052253468 mod 0.2065071888 uden straf og baseline 0.2076682236. Straf forbedrer S3 med 0.6207 %, men vinder kun 3/7 uger; uden uge 10 vender gevinsten til en forværring på 0.2836 %. S2's samlede strafgevinst 9.0688 % drives også af uge 10. S3 med straf er derfor kun en foreløbig udviklingskandidat, med lambda=0 bevaret som reference. Realiseret pris/temperatur gør forsøget betinget; ingen blind eller origin-tilgængelig ugeforecast-accept. Se [matematik, kode og resultater](G4_STRATEGIES_12_WEEKS_REVIEW.md). Alle 210 valgte kandidater består tidsforfining; fem fokuserede testmetoder er dækket, ingen fuld testpakke. Én kandidat blev fortsat fra checkpoint, øvrige 104 opgaver genbrugt. Tidligere protokolresultater nedenfor er bevaret og gælder deres egne forsøg. Ingen lokal kørsel kræves nu; G6b→G8→G7 består som obligatorisk kæde.


**Nyeste prognoseudvikling, 6. oktober 2026:** Johans fem-matchende-ugedage-protokol er gennemført for 7.–27. februar: 21 rolling-origin døgnprognoser og 504/504 målte timer. G4 uden straf har MSE 0.1639224439 mod 0.1666013393 med lambda=0.001; straffen øger samlet MSE med 1.6342 %. Uden straf slår samme origins' WM-baseline med 3.8626 %. Valget for denne protokol er lambda=0, baseret på DEVELOPMENT-forecastfejl. Det historiske januarvalg bevares og gælder den gamle protokol. Dette er daglige refits, ikke tre frosne 168-timers forecasts. Se [matematik, kodereview og resultater](G4_WEEKDAY_DEVELOPMENT_REVIEW.md). To fokuserede tests samt alle 42 valgte kandidaters 60/120-kontrol består. Ingen lokal kørsel kræves nu; G6b→G8→G7 består som næste obligatoriske kæde.


**G4-resultat, 5. oktober:** Januar-pilot og begge frosne februarprognoser er nu
kørt og rå-reviewet. 206 residualkald, ca. 83 s inklusive eksport/figurer.
Bedste fleksible g forbedrer DEVELOPMENT-MSE med 0.01504 %; lineær BM
beholdes efter den låste 1 %-regel. Baseline har stadig lavest MSE.
Se [G4-reviewet](G4_JANUARY_PILOT_REVIEW.md) og dets rå benchmark.
Dette er udviklingsvalg på legacy-punktkontrakten, ikke blind validering.
Ældre beskrivelser af manglende G4-pilot nedenfor er historisk status;
den aktuelle næste handling står i reviewet.

Forberedt 5. oktober 2026 mod `main` ved
`d9f95a0513605d58f0fe01464067e8f0a2a0eeeb` efter PR #69/#70.
**Implementeringsstatus 5. oktober:** P1/P2 er nu implementeret og syntetisk
kontrolleret: `deterministic --from-fit`, versioneret JSON-snapshot og
kontrolleret 07fad9f-import. Fælles BM/WM/origin/baseline og varig eksport før
optional scoring er verificeret. 20 G-tests plus tre relevante layout/import-
kontroller består på ca. 6.7 s. En CI-fundet pandas copy-on-write-fejl i
scoringen er reproduceret og rettet med en eksplicit observationskopi; de
samme 23 kontroller består også med copy-on-write aktiveret (ca. 5.3 s).
G3 er siden rapporteret complete på main 607dcd7 med 168/168 scored timer.
G4's adapter/LS er implementeret med ni små syntetiske kontroller; sammen med
de eksisterende 20 G-tests består 29 målrettede kontroller på ca. 13.6 s.
Historisk råreview, lokalt G4-modelvalg og G6–G8-integration mangler.
Beskrivelsen nedenfor bevarer designet; denne status erstatter ældre formuleringer
om manglende P1/P2. Kommando: [G_JANUARY.md](G_JANUARY.md).
Læs nyeste `AGENTS.md`, [G.md](G.md) og [G_JANUARY.md](G_JANUARY.md)
ved genoptagelse af G. Læs kun F-dokumenter, når en konkret fælles grænseflade
skal genbruges eller ændres. F4-H2 tilhører F-chatten; bevar samtidige F/WM-ændringer.

## Ny beslutning: langt fit og genbrug af WM's ugeprognose

Johan har 5. oktober valgt WM's fit→frys→viderefør cutoff-state→168 timers
forecast som G's princip, med fælles helårsfit hvis data tillader det.
G3/G4 har allerede forward/snapshot-grundlaget. Nødvendige udvidelser
implementeres ved den relevante etape; ingen ny forecast-solver nu.
Se beslutningsafsnittet i [G.md](G.md) for verificeret WM-kode,
origin-tilgængelige inputs, årsfit/testsplit og skel til oracle-dagsstate.
Dette ændrer den planlagte lange variants træningsvindue, ikke januararkivet
eller G4-pilotens budget. Langt fit er ikke en gate for G6b→G8→G7.

## Opgaven og det allerede kendte

Bachelorens projekt 111866 handler om at optimere prissignaler, så en gruppes
forbrug følger en ønsket profil, med en stokastisk demand-response-model.
FP-approksimation med mixtures, stokastisk single shooting og rekursiv
tilstands- **og** parameterestimation er centrale leverancer. EM betyder her
Euler–Maruyama og bruges til simulation/krydskontrol. Et godt deterministisk
fit eller forecast afslutter ikke bachelorens metodekrav.

Johan har nu bestilt fortsættelsen og autoriseret merge efter checks. G3's
genbrug af gemt fit er gennemført lokalt; næste trin er den lille G4-pilot
fra original TRAIN-eksport, efterfulgt af G6b→G8→G7 på frosset model.
Den aktive historiske variant er **hele januar 2022 → næste 168 timer**, MED,
neutral start, nul warm-up; den ældre uge/24-timers-warm-up-protokol er en
anden deklareret variant. G4's lille forsøg ændrer ikke F4-H's obligatoriske
fælles f/g-leverance eller dens acceptkrav.

| Kendt evidens | Status og begrænsning |
|---|---|
| Januarfit på `07fad9f` | Johan har rapporteret `status=complete`, 624 målte timer af 744; 120 demandhuller. |
| BM-parametre | C=1.63492382 timer, k=0.22688551, Xi=0.23120648; x0=0.5 fast. Tallene er afrundet terminaloutput, ikke input til genstart. |
| Normaliseret MSE | BM 0.0066056846, WM 0.0072743310, baseline O 0.0066347027. In-sample; BM's gevinst over O er ca. 0.44 %. |
| Råartefakter | Run `g_january_fit_20261005_154824` er ikke modtaget/råreviewet her. `fit.png` alene er utilstrækkelig. |
| G3 | Rapporteret complete på 607dcd7, run g_january_forecast_20261005_204059; 168/168 scored timer. Råreview mangler. |
| G4 | Lille to-stykkers g/LS/arkivadapter implementeret og syntetisk kontrolleret. Lokalt historisk udviklingsvalg mangler. |
| G6/G8/G7 | Ny G-integration er planlagt. Eksisterende F2/F3/F4-kode/evidens genbruges med konkrete kildehenvisninger. |

Arbejd i branch i eksisterende checkout, ikke nyt worktree. Skift/pull ikke
under aktive kørsler. BLAS=1; ingen ny historisk fitrunde alene for dokumentation.
En nødvendig ændring af den videnskabelige datakontrakt kan kræve et nyt,
særskilt fit; det gamle resultat må hverken overskrives eller ommærkes.

## Fund der skal håndteres før historisk G3-accept

1. Den oprindelige fit-mode kalder `run_benchmark`. Ny `--from-fit` indlæser
   fitmappen uden baseline/BM/WM-fit og foretager kun forward propagation.
   `checkpoint.json` er kandidat/status, ikke en genoptagelig SciPy-session.
2. Forecast-inputs bruger kun pris/temperatur. G3 gemmer nu prognoser før
   observationslæsning/scoring; manglende demand giver særskilt scoringstatus.
3. `consumption_series` i `wm_3/fit_weekly.py` bruger `mean(axis=1)` med
   pandas' skip-NaN-adfærd. En delvist observeret time kan få en anden
   population. 624 aggregerede måletimer beviser ikke 624 komplette MED-timer.
4. F4-H2's metadataaudit har verificeret MED/UDEN-medlemskab. G kan citere
   disse oplysninger i sin egen datakontrakt. Johan har deklareret timeforbrug,
   foreløbigt kWh, hour-start og foreløbig Copenhagen-tid; arbejdsantagelserne
   skal fortsat fremgå. F4-H2's intervalsolver og numerikstatus er F-arbejde,
   ikke en opgave eller acceptgate for G's syntetiske P4–P6.
5. Prisnormalisering er frosset på TRAIN. Februarpriser uden for domænet
   giver korrekt afvisning; de må ikke klippes eller bruges til efterfølgende
   ændring af januarfittets skala. Faktiske DK1-spotpriser er en deklareret
   proxy, ikke dokumentation for husstandenes faktiske tariff/kausale respons.
6. Nye snapshots gemmer alle baselineindstillinger og BM/WM-cutoff-states.
   Legacy-import genskaber dette fra verificeret source og finegrid-CSV.

Historisk G3 kan først mærkes valideret, når den valgte datakontrakt er
dokumenteret. En videreført legacy-punktapproksimation kan fortsat vises som
udviklingsdemo med sin oprindelige populations-/observationskontrakt. Den
afklarer ikke F4-H2. Syntetisk adapterarbejde behøver ikke vente på råmappen.

## P0 — datakontrakt, fitarkiv og inputaudit

**Filer:** `src/tools/bm_1_deterministic_data.py`, den nye G-snapshot-adapter
beskrevet under P1, G's manifest. F4-H2's metadata kan bruges som kilde;
G fastlægger selv population, eksponering og observationsoperator.
Ret ikke WM's selvstændige entrypoints som bivirkning.

- Arkivér original `manifest.json`, `fit.json`, `result.json`, audit/checkpoint
  og `fit_signals_normalized.csv` samt hourly-CSV'er uændret, med SHA256.
  Læs fuld præcision fra filerne; brug ikke terminalens afrundede parametre.
- Auditér de valgte MED-kolonner og antal tilgængelige medlemmer per time i
  den deklarerede TRAIN. Hold sluttest afskærmet. For ny fast-populations-
  kontrakt er aggregatet NaN, hvis et medlem mangler. Gem medlemsliste,
  antal, råmaske og kontraktversion. Legacy-skip-NaN forbliver særskilt.
- Dokumentér måleenhed, observationstype, timelabel og tidszone med kilde.
  Hourly middel/energi kræver korrekt operator for både BM og matched WM,
  ikke blot en ny etiket. Deterministiske intervalintegraler og stokastisk
  interval-likelihood er forskellige implementeringsopgaver; sidstnævnte
  kræver særskilt design/validering før historisk G8.
- Frys træningshistorik, skala, observation/operator, baseline og prisproxy.
  Registrér eksponering: januar og allerede sete februarperioder er udvikling.
  Uåbnet sluttest skal udpeges og protokollen låses før åbning.
- Auditér G3's 168 timers pris-/temperaturinputs og nødvendige endepunkter
  uden fit eller brug af fremtidig demand. Hold sidste input ved slutkanten
  som den eksisterende adapter, medmindre en ny version deklarerer andet.
  Pris uden for frossent domæne stopper denne prognose; ingen skjult refit.

**Leverance:** kort data-/arkivrapport med konkrete åbne felter. Ingen
opdigtede metadata, ingen ubegrundet `historical_fit_ready=True`.

## P1 — gemt fit → frosset forecast-origin

**Implementeret fil:** `src/tools/bm_1_deterministic_snapshot.py`.
**Berørte eksisterende filer:** `src/method/bm_1_deterministic_benchmark.py`,
`src/exe/run_model/bm_1/deterministic.py`, G's validering og små tests.
Designets schema er implementeret som validerede JSON-data med almindelige
helpers; der er ikke indført et ekstra solver-/klasseframework.

Det versionerede snapshot har følgende kontrakt:

| Feltgruppe | Skal gemmes og valideres |
|---|---|
| Identitet | Schema-version, original run-ID/commit, hashes for afhængige numeriske kilder og input/artefakter, softwareversioner, migrationsproveniens. |
| Data/information | Population, observationstype/enhed, timezone, TRAIN-start/slut, baselinehistorik, cutoff, prisproxy, alle frosne skalaer og manglende-data-politik. |
| Baseline | coef, feature_mean/scale, metadata samt base_temperature, hour_harmonics, season_harmonics, ridge, huber_delta, max_iter; features/version, clipping og interpolation. |
| BM | Uafrundede C_hours,k,Xi, kurver, initial-state-mode, valgt trænings-x0, eksakt simuleret x ved cutoff. |
| WM | Uafrundet r, fitkonvention, modelversion og simuleret F,R ved cutoff. Ingen reset til INIT_F eller I−O ved forecast-start. |
| Numerik | Grid/trin, tolerance og faktisk kontrolresultat, tids-/priskonvention og endepunktsregel. |

Implementerede operationer: `export_fit_snapshot(result)`,
`load_fit_snapshot(path)` og eksplicit `migrate_legacy_fit(directory)`.
Snapshot skal kunne indlæses uden optimizer, workbook eller netværkskald.
Genskab WM-baselineobjektet med de gemte indstillinger og koefficienter uden
`.fit()`. JSON må ikke indeholde pickle/eksekverbar kode.

Legacy ved `07fad9f`: `result.json` har `x_forecast_origin`, `wm_fit.r` og
numerik; `manifest.json` har baselinevektorer og skalaer; finegrid-CSV'en har
F/R på cutoff. Kræv nøjagtigt det forventede sluttimestamp, finitte states,
overensstemmelse mellem gemt BM-origin og finegrid-endpoint og bestået numerik.
Sidste hourly-fitpunkt er **ikke** forecast-origin. Manglende baselineindstillinger
må kun rekonstrueres fra den verificerede originale implementation på denne
kendte sourceversion og kontrolleres mod gemt baseline. Ingen aktuelle defaults
som tavs erstatning. Migreringen skriver ny fil i ny mappe og bevarer originalen.

Afvis ukendt/incompatible schema, manglende states eller uverificerbar migration
med præcis årsag. Et nyt docs-commit gør ikke i sig selv numerikken inkompatibel;
kontrollér relevante afhængigheder og registrér kompatibilitetsbeslutningen.
Et manglende felt er ikke tilladelse til automatisk re-fit.

**Accept:** lille syntetisk fit eksport/import giver samme baseline, BM/WM-origin
og forecast inden for fastlagt round-trip-tolerance; ingen fitfunktion kaldes
under import/forecast. Korrupte skalaer, parameterrækkefølge, endpoint og
inkompatibel baseline afvises. Lås kildedata mod utilsigtet overskrivning.

## P2 — G3-prognose, varig eksport og særskilt scoring

**Implementeret:** `forecast_from_snapshot(snapshot, raw_inputs)` i
benchmarkmodulet får kun frosset model og inputs. `deterministic --from-fit`
vælger denne vej og afviser eksplicitte fitindstillinger. Den faktiske
kommando og output står i januarvejledningen.

- BM starter med gemt x ved cutoff. WM starter med gemt F/R. Samme fælles
  O/B-samples, held-price-forløb og enhedsomregning som den oprindelige kæde.
- Skriv predictions, states, inputaudit, provenance og numerisk status atomisk
  før optional scoring. Et scoringproblem må ikke slette en gyldig prognose.
- `score_forecast(result, observations)` bruger fælles målte timer for
  BM/WM/O. Ingen udfyldt demand tæller som måling. Gem expected/observed/scored
  samples, coverage, maske og årsag til eksklusion. Døgn uden målinger får
  samples=0 og NaN-metrics; nul observationer giver `not-scored`, ikke fiasko
  i selve simulationen. Ufuldstændig coverage mærkes tydeligt og er ikke en
  fuldt observeret uge. Ingen ny arbitrær coverage-gate skjult i kode.
- Genbrug WM's MSE/RMSE/MAE/bias og performanceopgørelser, og hold pooled score
  adskilt fra gennemsnitlige dagsscorer. Kræv samme maske for sammenligning;
  hvis WM mangler, rapporteres BM særskilt uden matched-accept.
- Resultat og statusfiler adskiller `fit_status`, `forecast_status`,
  `numerics.passed` og `scoring.status`; manifestet angiver udviklingskontrakt
  og evidens. En færdig prognose er ikke automatisk
  historisk/videnskabelig accept. `status=complete` fra et gammelt fit bevares
  i originalen og fortolkes i sin oprindelige betydning.

**Små, men meningsfulde kontroller:** frosset replay mod eksisterende syntetisk
fit+forecast; ændrede/fjernede fremtidige y ændrer kun scoring; delvis/ingen
demand bevarer forecast; startstate kommer fra eksakt cutoff; inputdomain-fejl
giver tydelig afvisning uden optimizerkald. Bevar G's 1e-5 demand-refinement-
tolerance. Ingen ny hel-måneds-smoke alene for denne refaktorering.

Historisk næste handling efter P0–P2: importer den originale januarmappe,
auditér de frosne inputs og kør kun 168-timers-forwarddelen. G3 kan godt give
negativ baselinegevinst; det udløser ikke automatisk nye fits.

## P3 — lille G4 og ét fælles modelinterface

Den implementerede pilots ligninger, monotonicitetsbevis, residualer og
metodeforsvar står nu i [G-planens afsnit 5](G.md#5-fleksible-fg-i-et-lille-ekstra-trin).
Det historiske dokumentationsreview mod main `0da7e2a9` ændrede ikke solver
eller acceptstatus. Den efterfølgende G4-pilot er gennemført og rå-reviewet
som beskrevet øverst; originalarkivet kræves fortsat ved reproduktion.

**Genbrug:** `src/params/bm_1_response.py`, `src/method/bm_1_response.py` og
`BM1Parameters` i `src/control/bm_1.py`. H's likelihood-fit er ikke G's LS-fit.
**Adapterarbejde:** `src/method/bm_1_deterministic_fit.py` har en lineær,
specialiseret kerne. Den er bevaret uændret, så G3/januararkiver fortsat kan
indlæses. Den nye afgrænsede kurvevej er implementeret separat i
`src/method/bm_1_deterministic_response.py`; lineær geometri delegerer præcist
til den oprindelige kerne.

**Implementeret 5. oktober:** `src/params/bm_1_deterministic_response.py`
gemmer valideret BM1PiecewiseResponse-geometri og leverer eksisterende
BM1Parameters med eksplicit hours/days-capacityomregning. `src/tools/
bm_1_deterministic_response.py` læser/verificerer TRAIN-arkiv, fitter kandidater,
gemmer nested originalt reference-snapshot og egen cutoff-state, og udfører
G4-forecast med frosne lineære BM/O/WM-referencer. Runner:
`python -B -m src.exe.run_model.bm_1.deterministic_response --reference-fit FITDIR
--output NEWDIR`, derefter `--from-fit NEWDIR/variants/lambda_00`/`lambda_01`.
Ingen af dem fitter baseline/WM/G2 igen; G4-optimeringen bruger kun TRAIN.
Ni korte kontroller i `tests.test_bm1_deterministic_response` dækker kurve-
paritet/boundaries/prisspring/enheder, fælles LS og budgetprober, arkivreplay,
snapshot, egen origin, kausal scoring og varig forecasteksport ved scoringfejl.
Kun warmup=0-arkiver understøttes i den første pilot. Denne kodekontrol er
ikke et historisk G4-valg eller mandatory H-f/g-accept.

1. Et frosset G-modelspec gemmer C_hours,k,Xi, f/g-knuder/værdier, skala,
   observationstype og tidsenhed. Konstant sigma_x tilføjes i G6; sigma_y er
   separat. Callables dannes fra validerede data, ikke fra serialiseret kode.
2. Første pilot: to-stykkers g, lineær f, k fast på G2's værdi, C/Xi og g's
   indre knude estimeres samlet over hele TRAIN. Bevar samme x0-policy som
   G2: wm-neutral holder x0=0.5; kun `initial-state fit` estimerer x0.
3. Endpoints/monotonicitet valideres, ingen sorting/clipping i objective.
   Udvidelse til f/fire stykker kræver den deklarerede udviklingsbeslutning.
   Lineær løsning er et legitimt negativt pilotresultat, ikke H-f/g-accept.
4. Krydstjek drift og demand på indre/boundary-state-, pris- og baselinepunkter
   mellem G-kerne og `bm_1` med samme kurver. Kontrollér retning, signs og
   tidsenheder før nye stokastiske studier. Undgå et ekstra solverframework.

Budget fra G-planen: samlet højst 3.000 residualkald/10 minutter for alle
varianter, strafstyrker og starter. Kodeadapters syntetiske paritet kommer først.
Gem datascore og regulariseret objective separat; DEVELOPMENT vælger model,
TRAIN alene gør det ikke. Frosset kurvevalg overføres til G6/G8/G7.

## P4 — G6b: lille FP/mixture/EM-panel

**Genbrug:** `src/model/bm_1.py`, `src/method/fokker_planc.py`,
`src/method/beta_mixture.py`, `src/method/euler_maruyama.py` og F3's
fælles Monte Carlo-evaluering i `src/control/bm_1.py`.
Foreslået G-studierunner er en tynd adapter, ikke en ny FP- eller EM-solver.
G6a's variansreference er separat; WM4's varianskernel er ikke fuld FP/EM.

Følgende er et **registreringsudkast for en syntetisk udviklingscase**, ikke
en kørt undersøgelse eller fysiske estimater. Lås et faktisk manifest før
generering/evaluering; alle ændringer efterfølgende får et nyt forsøgs-ID.

| Felt | Første afgrænsede forslag |
|---|---|
| Tid | Timer, horizon [0,24], hourly rapportering, alle prisbreakpoints eksplicit. |
| Model | Lineær f/g-reference; C=4 timer, k=2, Xi=0.4; konstant sigma_x=0.2 per sqrt(time). Den G4-valgte kurve får senere sin egen paritetscase. |
| Inputs | B(t)=0.5+0.1 sin(2 pi t/24); u=[0.3,0.7,0.3,0.7] på [0,6,12,18,24]. Ingen historiske targets bruges. |
| Initiallov | Beta(20,20) for FV og EM; mixture-init fra samme lov med særskilt projektionstab. Ikke punktstart forklædt som samme prior. |
| EM | Seedede initialstates/increments; start 256, loft 2048 baner; fin/grov koblet via summerede increments. Fast seedliste registreres før kørsel. |
| Numerik | To FV-grids og to mixturegrader; genbrug F2/F3's definitioner af celler/grader. Vælg eksplicitte små niveauer i 60-sekunders pilot og lås dem før sammenligningen. To tidsniveauer deles af de relevante veje. |
| Output | Masse/positivitet, initial-/løbende projektionstab, E[X], Var(X), E[D], Var(D), udvalgte CDF/kvantiler, trackingtab, MC-standardfejl, randoverskridelser, runtime. |

Bevar eksisterende sandsynligheds-/randguards. Fastlæg moment-, CDF- og
objective-tolerancer i samme manifest før hovedpanelet og i forhold til den
praktiske størrelse, ikke efter den observerede metodeforskel. Runtimepiloten
er ikke videnskabelig accept. Hvis to niveauer eller MC-budgettet ikke kan
opløse forskellen, rapportér `inconclusive`; udvid ikke automatisk studiet.
Ingen claim om generel intervalkalibrering fra ét døgn/en sand bane.

Timer er G's kanoniske tid. F-kode omregner ikke automatisk: alle inputtider,
breakpoints, drift/diffusion og objective-integraler skal bruge samme enhed.
Ved dag-adapter: C_days=C_hours/24 og sigma_days=sqrt(24)*sigma_hours;
integreret trackingtab omregnes også (J_hours=24*J_days for uændret integrand).
Likelihood-sigma_y omregnes ikke som procesdiffusion. Dokumentér enhedstesten.

EM bruger samme SDE og initiallov; rå randoverskridelser registreres før
korrektion. Brug `boundary='none'` og stop ved ugyldige baner frem for at
skjule dem med clipping. Mix-metoden er FV-transport + Beta-projektion, ikke
gridfri koefficient-ODE. Genbrug ikke FV-posterior som skjult restart ved
hvert mixture-tidsskridt. Ikke-lineær demand skal integreres over loven.

Budget: 60-sekunders pilot, derefter højst 10 minutter inkl. valgt forfining,
højst to cases. Syntetisk først; historisk case kun med passende datakontrakt.
G6a's separate 5-minuttersloft ændres ikke.

## P5 — G8: rekursiv state/rho-bank og ærligt posterior-handoff

**Genbrug:** `run_fv_filter`, `run_mixture_filter`, `run_parameter_bank` i
`src/method/bm_1_estimation.py` samt `src/control/bm_1_handoff.py`.
Start med G6b's syntetiske model, kendte parametre og eksplicit punktobservation
D(X,t)+målefejl. Hold sigma_y adskilt fra sigma_x; f.eks. 0.03 i normaliserede
demandenheder i det låste syntetiske manifest, ikke estimeret fra samme testbane.

- Først kendt-parameter-statefilter, dernæst rho=1/C. Forslag til fem noder:
  [0.125,0.1875,0.25,0.375,0.5] per time med uniform prior; sand rho=0.25
  er en kendt syntetisk indre node, ikke et historisk fund. Øvrige parametre
  holdes faste. Simuler observationer med fin EM og separat registreret seed.
- Samme observationssekvens og manglende-data-maske gives til FV/mixture.
  Gem prædiktiv score **før** opdatering, posterior **efter**, log-evidence,
  bankvægte og hver nodes egen conditional state. NaN giver propagation
  uden likelihoodopdatering. Ingen interpoleret observation eller dagligt reset.
- Vis en lille split/genoptagelse med identisk slutposterior; gem mixture-
  koefficienter, ikke kun marginal masse. Rapporter svag identifikation og
  randvægte; én case dokumenterer integration, ikke generel parameterrecovery.
- Det generiske `run_parameter_bank` returnerer en dict. Den eksisterende
  `BM1ForecastOrigin.from_rho_bank_step` kræver `BM1FVRhoBankStep` og vælger
  **MAP**, ikke hele parameterposterioren. Direkte indpassing er ikke nok.
  Skriv en lille valideret G-adapter til frozen origin med noder, vægte,
  conditional masses/coefficients, kurver, parametre, grid og cutoff.
- G7's fulde bankobjective er summen af vægtede nodeobjectives fra disse
  egne states. MAP er en særskilt deklareret billig approksimation med den
  valgte nodes state, ikke gennemsnitsparametre + marginal state. Implementér
  og test 1-node-reduktionen og et 2-node-eksempel før femnodedemonstrationen.

Undgå dobbeltbrug af data: syntetisk sekventiel kontrol starter fra en
deklareret prior og assimilerer hver måling én gang. Hvis historisk G2-fit
bruges til faste parametre/prior, skal dette være en markeret empirisk-Bayes-
approksimation eller bygge på et tidligere adskilt vindue; kald ikke replay
af samme januar en uafhængig posteriorvalidering.

Budget højst 24 timer/5–9 hypoteser/10 minutters samlet beregning inklusive
FV/mixture/handoff-kontroller. Bankens k-akse findes ikke i standard-API'et og
er ikke del af denne første opgave. Historisk intervalfilter er ikke dækket
af den syntetiske punktobservationsdemo.

## P6 — G7: single shooting med samme stokastiske objective

**Genbrug:** `BM1ControlProblem`, deterministisk/FV/dynamisk-mixture-evaluator,
`multistart_powell`, `evaluate_mc_candidates_common_random_numbers` og
`paired_mc_improvement` i `src/control/bm_1.py`.

Første controller bruger 4 prisblokke over 24 timer, bounds u i [0,1], et
fast ex-ante mål A(t) og nul terminal-/glathedsstraffe. Et lille syntetisk
forslag er A(t)=B(t)+0.03 sin(2 pi t/24); frys det før politikoptimering og
rapportér eventuel utilstrækkelig fleksibilitet uden efterfølgende at tilpasse A.
Prisvektoren er optimizerens eneste beslutning; hver evaluering starter fra
samme frosne origin og propagérer fordelingen. Ingen test-demand i objective.

Den deterministiske reference bruger deklareret state fra samme information
(f.eks. posteriorens middel, udtrykkeligt mærket plug-in). Den stokastiske
mixture-politik optimerer E[(D−A)^2], inklusive Var(D). Ved bankhandoff bruges
nodevægte og nodeconditional states; vægtene holdes faste gennem det åbne
kontrolhorizon. Dette er en åben single-shooting-policy, ikke dual control.

Neutral pris, deterministisk policy og stokastisk policy evalueres alle på
samme stokastiske problem med finere FV/mixture og parret EM. Ved bankposterior
samples også parameter-node og dens conditional initialstate konsistent;
samme node/state/increments deles mellem politikkerne. F3's standard-Beta-
initialsampler er ikke automatisk gyldig for en filtreret bankposterior.

Efter G8 vises et kort faktisk loop: observation → posterior → prisvalg →
simuleret anvendt pris → ny observation → ny opdatering. Handoff alene er
ikke et lukket loop. Afgræns antal kontroloptimeringer på forhånd, og tæl dem
i samme budget; en lille demo må ikke snige et helt MPC-studie ind.

**Accept:** fælles objective/enheder/cutoff, ikke-negativ varians, korrekte
prisconstraints, mean/variance-dekomposition, 1-node-bankparitet, ingen
posterior-mutation mellem objective-kald, numeriske/MC-fejl rapporteret.
En stokastisk policy behøver ikke vinde statistisk i én case. Lokalt
optimizerstop er ikke bevis for global optimalitet eller kausal priselasticitet.

Budget fra G-planen: højst to starter per politikfit; **samlet** 100 objective-
kald/5 minutter til de to fits og højst 5 minutter til krydskontrol. Ved
bankobjective tæller ét optimizerkald én kandidat, men alle node-forward-solves
registreres særskilt og er omfattet af væguret. Genbrug cached identiske kald.
G6b/G8/G7 har samlet 30 minutter plus tre 60-sekunders piloter; stop er
uafsluttet/negativt resultat, ikke mulighed for at løsne acceptkrav.

## Leverancer, testscope og næste sessions start

| Pakke | Færdig når | Afhænger af |
|---|---|---|
| P0 | Råarkiv/databetydning og mangler er eksplicit auditeret. | Januarfiler til historisk del; kodeaudit kan laves nu. |
| P1 | Versioneret snapshot og kontrolleret legacy-import uden fit. | Eksisterende fitformat; historisk import afventer råfiler. |
| P2 / G3 | Input-only-forecast, gemning før scoring, matched output og små kontroller. | P1; historisk accept også P0. |
| P3 / G4 | Fælles kurveinterface, paritet og afgrænset udviklingsvalg. | G3-kæden fungerer; budget er målt. |
| P4 / G6b | FP/mixture/EM-panel på låst model med numerisk/MC-status. | Frosset modelinterface; syntetisk først. |
| P5 / G8 | Rekursiv state/rho, conditional posterior, verificeret origin. | P4-paritet. |
| P6 / G7 | Politikoptimering, fælles krydsevaluering og kort estimation/kontrol-loop. | P4/P5; kendt-prior-adapter kan forberedes tidligere. |

G5's bredere forecastbenchmark og G6a's variansreference må ikke blokere
bachelorens centrale FP/estimation/kontrolkæde. Stærk forecastgevinst er
heller ikke adgangskrav. Bevar skellet implementeret / syntetisk demonstreret /
numerisk kontrolleret / historisk valideret i alle statusfiler.

P1/P2's syntetiske round-trip og scoringkontroller er gennemført. Johan har
rapporteret G3 complete på main 607dcd7: 168/168 scored timer, BM-MSE
0.06951591504 mod O 0.06934621735 og WM 0.07099789830. Ingen baselinegevinst;
ugen er nu eksponeret DEVELOPMENT. Originale CSV/plots er ikke modtaget,
og P0/råreview er stadig åbent. P3's adapter/LS er nu implementeret og
syntetisk kontrolleret; næste skridt er lokal lille G4-pilot/udviklingsvalg
med kommandoerne i G_JANUARY. Derefter P4→P5→P6 på frosset model. G har ingen fast standardpakke: vælg kun den mindste eksisterende test, der
dækker ændringen. F er et selvstændigt spor; kun en faktisk ændret fælles
kontrakt begrunder checks i begge spor. Se den bindende regel i AGENTS.md. Ingen fuld test-discovery, historisk månedssmoke eller F1–F4-
genkørsel alene for dokumentation. Lange studier er ikke unit-tests.

Ved hver kodeleverance: opdatér G-status, januarvejledning og den korte
F4/AGENTS-indgang efter faktisk status, med kildecommit, kommando, budget og
råartefakter. Genlæs seneste main før publicering. Denne forberedelse har
oprindeligt ændret dokumentation alene. P1/P2 har nu nye syntetiske tests;
ingen historisk kørsel i chatten eller G4–G8-accept følger af dem.

## Reserveret til senere Astra-review

Johan har bedt om, at opgaver som har fordel af dybere matematisk behandling
gemmes til senere. Følgende er anbefalinger om reviewniveau, ikke dokumenterede
modelbenchmarks eller krav om at skifte model for rutineimplementation.

| Opgave | Anbefalet niveau | Konkret afgrænsning |
|---|---|---|
| G-review: posterior → stokastisk prisstyring | **Astra High** | Review parameter–state-afhængighed, mixture/FV-paritet, enheder, forventet kvadratisk tab inklusive varians, posterior-initialiseret EM og cutoff i P4–P6 før samlet metodeaccept. Rutineadaptere kan fortsat implementeres af Sol. |

G-reviewet foretages når adaptere/resultater er konkrete; et nyt stort studie
kræves ikke alene for review. G4 og den syntetiske G6b→G8→G7-kæde kan
fortsætte uafhængigt af F4-H2.

F4-H2's interval-likelihood-review (Astra XHigh) hører alene til F-sporet:
se [F4-H2-overdragelsen](H2_INTERVAL_ASTRA_HANDOFF.md) som ekstern reference.
Hvis G senere skal assimilere historisk timeenergi, kræves en særskilt valgt
og kontrolleret intervaloperator. Det kan være genbrug af en egnet F-løsning;
F4-H2 som helhed er ikke en forudsætning. Syntetiske punktmålinger må ikke
omdøbes til historisk timeenergi for at springe dette krav over.


