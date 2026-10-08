## F4-H2:45 kald og checkpoint08 sikret; aktiv20h-fortsættelse

PR122 er merged52006d77. Alle originaldata og kildefiler er gendannet;
input.xlsx er hash-identisk og tilgængelig til export efter forecastfreeze.
Ingen slut-Y er læst. Checkpoint01–08 sikrer1300 nye råjobs og450 adoptioner.
45 firestykkekald er komplette, kandidat46 har28/72 timer. Første supplement-
blok ramte hård watchdog; fuld600s-charge bevares. Næste blok lukkede rent
med tidligere STOP-pause ved480s; dens originale marker er bevaret under
supplement_block02_preserved_STOP.txt før autoriseret fortsættelse. Alle
numeriske kilder/caps/guards er uændrede. Nye blokke får samme ekstra
shutdownmargin. Aktuelle output ligger på PR123 mens arbejdet kører.
H/I/J/F4 er fortsat åbne; ingen convergence eller modelvinder hævdes.
20h total/native1080s og23:55-frist gælder. G/WM-status bevares.
Følgende ældre F-statusafsnit er historik.

## G7: døgn-fixture, timing og budgetstop, 8. oktober 2026

Døgn-driverens stub-forløb gennemførte begge prissøgninger, tre frosne
styringer, fem forward-kontekster, fælles EM-krydskontrol, en observation,
fire egne posterioropdateringer og fortsættelsen. Dette er kontrol af
orchestration med stub-kernels; ingen numerisk validering af døgnforsøget.

Den samlede kontrolpakke er IKKE bestået. Første scope stoppede på en for
kort node-reservation; næste ramte parent-loftet; tredje gennemførte
døgnforløbet men fejlede ved den ekstra genbrugstest, som konstruerede et
andet numerisk kontrakt-id. Den korrekte kontraktforskel blev afvist.
Genbrugstesten er rettet til driverens make_control, men rettelsen er ikke
runtime-verificeret. En separat read-only audit af gemte resultater nåede
også sit parent-loft. Ingen fejl eller fulde reservationscharges slettes.

En særskilt preregistreret native deterministisk 24h-målfunktion ved
u=(.5,.5,.5,.5) blev færdig:2.3313609979995817s, J=.12120382930681177,
MAP-node3 (nulbaseret), x12=.5062519458461637. Dette er én timing, ingen
prissøgning, ingen optimumclaim. Native kilder/kernels er uændrede.
Raten er standalone med én hash-verificeret egen M254-origin; den er endnu
ikke et kompatibelt ratepacket for hele grafens fire origins/source-context.

Konservative scopecharges:15+8+20+5+8=56s. G debit1907.3372718589885,
ceiling1910, rest2.6627281410114847s. Alle16 kompatible whole-job rater
står fortsat unknown; ingen samlet budgetadmission, fuld døgnpræregistrering,
numerisk døgnkørsel eller scientific_accept. Ingen bruger-PC-kørsel kræves.

Læs root/G7_DAY_CHECK_AND_RATE_REVIEW.md og
root/g7/day_check_and_rate_status_20261008.json.
Næste konkrete forslag er +20s til én sidste read-only completion-audit,
worker18s/reserve2s. Forslaget er IKKE godkendt og starter ingen proces.
Først derefter egne fire origins, kompatible rater, samlet admission,
særskilt immutable præregistrering og numerisk døgnforsøg.
Følgende G-afsnit/statusser er historik; samtidige F-afsnit bevares.

## F4-H2: godkendt 20h-fortsættelse, 8. oktober kl.15.17

Johan: "det er godkendt bare skynd dig". Det pending +7200s-forslag er
registreret i separat v4-campaign: H72000s, native1080s, calls25/49,
paircap73, invocation600s og stop/export60s. Brugerens aktuelle resume
superseder den passerede08:00-frist; nyt bounded stop er23:55 Copenhagen.
V3's43780.468989510904s og alle gamle reservationscharges bevares byte-identisk.
Forberedelse5s + lokal recovery/scoped kontrol35s giver43820.468989510904s;
native1015.3882277010393s. Alle69+4 numeriske kildehashes, frozen input,
environment og3655 predecessor-råjobs er verificeret;8 scoped guards består
med0 propagation/reservationer. Checkpoint01–06's1166 nye jobs er gendannet.
Original input.xlsx er nu genfundet/hashverificeret; ingen slut-Y er læst.
Recoverybytes er holdbart sikret. Midlertidig GitHub-byteeksport kørte ingen
model/test og er fjernet. Controllerb53e337b47613b1070037223b79befd96b0900d8
fortsætter fra kandidat44, ikke en ny optimizerstart.43 færdige kald er
historisk boundary; aktuelle supplementresults kommer i separat receipt.
H-fit/selection/forecast/finale og I/J/F4 er fortsat åbne. Ingen ny numerical
retry eller capudvidelse. G/WM-status og droppede F4-G2b/G2c bevares.
Læs root/f4/h2_budget_supplement_approval_20261008.json og recoveryreceipt.
Følgende ældre status-/approvalafsnit er historik.

## F4-H2: normalt budgetstop;43 kald sikret, 8. oktober

PR118 er merged. Blok6 stoppede rent for at beskytte restvalidation, ikke pga
numerisk fejl.43 firestykkekald og1166 nye råjobs er sikret i checkpoint01–06;
kandidat44 har38/72 timer gemt.450 adoptioner og alle frosne kilder består.
Ingen proces kører; gammel18h-approval tillader ikke mere fit nu. De20955.86s
restvalidation er fortsat beskyttet; alle crashcharges bevares fuldt.
Forslag +7200s (H18→20h), samme08:00-frist/native1080s og25/49-caps, er
IKKE godkendt. Registered remaining49-rate-scenarie:18.49min ved jobmidler,
27.74min med faktor1.5,82.42min ved beskyttede rater/hjælpere. Convergence er
ikke garanteret. Separat v4-runner/checkpointhelper er klargjort;7 scoped
kontroller består med0 propagationer/reservationer. Den afviser uden approval.
Læs root/H2_BUDGET_BOUNDARY_20261008.md og den konkrete proposal-JSON.
H-fit/selection/forecast/finale og I/J/F4 er åbne. Original input.xlsx mangler
stadig efter forecastfreeze. G/WM og droppede F4-G2b/G2c er uberørte.
Følgende status-/deadlineforslag er historik.

## F4-H2: checkpoint05 består; resultatfortsættelse, 8. oktober

PR106 er merged. Firestykkefit har40 komplette kald, næste kandidat delvis.
971 nye råjobs er sikret i checkpoint01–05; seneste191 jobs og450 adoptioner
består read-only audit.0 gemte numeriske failures, alle frosne kilder/miljø
uændrede. Blok4's fulde600s-crashreservation bevares; blok5 sluttede normalt.
Blok6 fortsætter under samme08:00-frist,64800/1080s-lofter og beskyttet
fuld restvalidering. Fit/selection/forecast/finale og I/J/F4 er fortsat åbne.
I/J-resultatreviewet beskriver nu det konvergerede/validerede tostykkefit.
Nyere main's G/WM-status og alle andre paths bevares. Følgende afsnit er historik.

## Arbejdsregel: merge færdige egne branches, Johan 8. oktober 2026

Johan: "merge altid naar du er faerdig med en branch". Når eget autoriseret
arbejde er færdigt og de relevante kontroller består, merge branchen/PR'en
og kontrollér merge-resultatet. Lad ikke en færdig egen PR vente på endnu
en mergegodkendelse. Reglen gælder fremover; uafsluttede drafts og andres
arbejde merges ikke alene på grund af denne regel. Branchsletning er en
særskilt handling og følger de eksisterende ejerskabs-/bevaringsregler.

## G7: samlet døgnstyring implementeret, ikke kørt, 8. oktober

Denne levering bygger på den mergede feedbackintegration PR119/main7586a9d2.
Ny day_control forbinder deterministic MAP/conditional-mean ODE og M254257-
Powell-search (to starts,24 calls/start,48/policy,96 total) med tre fælles
prisplaners M254/FV257/513-, partition-, parrede raw-EM- og feedbackreview.
Alle policies vurderes på samme stochastic J; fine/FV vælger ikke nye priser.
EM deler startstates/Brownian mellem policies og mellem coupled coarse/fine;
observation36 kommer fra forudbestemt fine node2/path0 plus separat noise.
Fire egne posteriorer bruges videre til38; kvitteringer genbruges eksakt.
Ny stdlib day_contract/launcher kræver fuld16-job-admission før native import,
source/origin/runtime-pins, eksklusiv invocation og parenttimeout. Ingen
autoriseret numerisk registration er oprettet; template er execution=false.
Native kerner/M254/legacy/F/WM/workflow bevares. Fire nye Python-filer er
statisk AST-parset; ingen runtime-, stub-, model- eller numerisk fuldgraftest
er kørt. PR-CI kontrollerer syntaks; statisk kontrol er ikke runtimeaccept.
G-debit uændret1851.3372718589885/1910s; rest58.662728141011485s.
Full-day/budget/scientific/historical=false; gamle quad0/1/2-audits stadig tabt.
Næste: én fokuseret fuldgraf-stubkontrol, derefter reelle whole-job-rates,
samlet budget og særskilt immutable numerisk præregistrering før døgnforsøget.
Johan skal ikke køre noget lokalt nu. Egen færdig branch merges efter kontrol.
Læs [implementering, matematik, forsvar og begrænsninger](root/G7_FULL_DAY_DRIVER_REVIEW.md).
De ældre G7-statusser nedenfor er historik.

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
Læs [matematik, integration, kontrol og begrænsninger](root/G7_FEEDBACK_INTEGRATION_REVIEW.md).
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
Læs [adapter, matematik, merges og faktisk kontrol](root/G7_PER_HYPOTHESIS_AUDIT_REVIEW.md)
og [samlet forsøgsforslag](root/G7_INTEGRATED_EXPERIMENT_DESIGN.md).
De ældre G7-statusafsnit nedenfor er historik.

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

## Aktiv dokumentationsregel for G

Johan, 7. oktober 2026: dokumentér og begrund matematikken, og forklar lidt
mere i chatten, hvad der konkret er indført og lavet. Ved G-leverancer skal
notation, mål, antagelser, relevante udledninger og metodevalg fremgå sammen
med koblingen til implementeringen. Skeln mellem nye beregninger og genbrugt
evidens, faktisk kontrol og dens begrænsninger. Begrundelser skal kunne
efterprøves; en bestået numerisk gate er ikke i sig selv et matematisk bevis.
Denne regel kræver ikke genkørsel af allerede beståede numeriske kontroller.

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
Læs [matematik, resultat, begrænsning og budget](root/G7_COMPLETE_QUAD_RECOVERY_REVIEW.md).

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
Læs [matematik, resultater, fejlrettelser og budget](root/G7_BOUNDED_LAUNCHER_REVIEW.md).

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
Læs [launcher, matematik, kontrol og budget](root/G7_BOUNDED_LAUNCHER_REVIEW.md).

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
Læs [matematik, design, budget og næste trin](root/G7_BOUNDED_PRICE_QUADRATURE_DESIGN.md).

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
Læs [interface, begrundelse, kontrol og næste trin](root/G7_RECEIPT_RECOVERY_REVIEW.md).

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
Læs [matematik, resultat, journalbegrænsning og næste trin](root/G7_SEED_REPAIR_REVIEW.md).

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
Læs [resultater, matematik, begrænsninger og recovery](root/G7_AUXILIARY_RUNTIME_REVIEW.md).

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
Læs [design, udledning, budget og resterende arbejde](root/G7_AUXILIARY_INTEGRATION_DESIGN.md).
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
Læs [forsøg, matematik, råreview og resterende arbejde](root/G7_NODE_RUNTIME_REVIEW.md).
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
Læs [matematik, crashmekanik, budget og faktisk kontrol](root/G7_NODE_JOB_REVIEW.md).
Følgende G-statusafsnit er historik.

## F4-H2: ny frist godkendt; separat fortsættelse startet

Johan godkendte8.oktober kl.00:28 ny stopfrist08:00 Copenhagen under samme
H64800s/native1080s-lofter. Den separate approval-JSON bevarer proposal og
alle gamle charges. Første600s-blok er startet med controllerb9983ee;
gamle lineære/to-stykkers valideringer adopteres, og kandidat28 fortsætter
under egen namespace med valideret ni-konturbank. Ingen lokale kommandoer
kræves før slutdataeksport efter forecastfreeze. F4/H/I/J er stadig åbne.
Johan har nu bedt om merge af færdige branches; tidligere "ingen merge"
er historik. Merge af implementation/evidens er ikke videnskabelig F4-accept.
Alle nedenstående tidligere statusafsnit bevares som historik.

## F4: næste trin samlet; kontrolleret H-fortsættelse klar

Aktuel samlet restplan: root/F4_REMAINING_PLAN_20261008.md. De to tidligere
formuleringer beskriver samme næste trin: H-adapter + runtime/admission.
A–F og afsluttet F4-G-evidens genbruges; G2b/G2c forbliver droppet. I/J er åbne.
Separat prepared run f4_h2_contour_continuation_20261008:3655 gamle jobs og69
kilder verificeret; gamle tokens/checksums/prior bytes+dtypes bevares. Nye jobs
har egen identitet; original terminalfejl og alle charges er uændrede.
Scoped offline identity/prior/failure/optimizer/budgetkontroller består;
0 propagationer og0 produktionsjobs. Ingen H-kørsel efter gammel23:55-frist.
H40481.86150/64800s inkl. separat preparation-ledger; native951.26557/1080s.
Ny runtimeplan beskytter20955.85668s≈5h49m; ca55m til yderligere fit efter
stopreserve. Fulde49-kaldsscenarie er IKKE admitted, convergence ikke lovet.
Konkret forslag: ny frist8.oktober08:00 Copenhagen, samme18h/1080s-lofter.
Deadline-proposal er IKKE godkendt; launcher afviser før reservation.
F4/forecast/population/I/J er ikke slutaccepteret. PR106 forbliver draft.
G-plan/G7 og WM er separate og uberørte. Følgende statusafsnit er historik.

## F4-H2: egen fine-kontrol består; 180 ekstra sekunder godkendt

Johan godkendte 180 ekstra native-s 7. oktober kl.23:47:41. Native-loft1080s,
H-loft64800s og stop23:55 Copenhagen er uændret. Separat kandidat28-kontrol
med egen2561-celle-prior24h + observeret time24 består på41.78s. Logscore-
forskel1.06323e-5; state-L1 .000834817; alle oprindelige guards består.
Alle24 priorcheckpoints og9 fulde konturer består råreview med0 propagationer;
posterior rekonstrueres præcist.69 gamle numeriske kilder er uændrede.
Native942.31036/1080s; H40472.90630/64800s efter kontrol/review/ZIP-eksport.
Gamle3655 jobs/27 firestykkekald og terminalfejl bevares. Ingen fitgenstart.
Næste trin er særskilt proveniensverificeret adoption og ny runtime-admission
for restfit/fullvalidation; dagens frist må ikke rulles automatisk videre.
Læs root/H2_TERMINAL_BANK_FINE_RESULT_20261007.md. PR106 forbliver draft.
Råresultat: f4_h2_terminal_bank_fine_20261007.zip, libfile_c10efdaac29881919c10ffae7600a37d.
G/WM er uberørt. Følgende statusafsnit er historik.

## F4-H2: terminal fejl afgrænset; udvidet konturbank består basistimen

Kandidat28 stoppede cap-extension efter3655 gemte jobs/27 komplette firestykkekald.
Det gamle run er terminalt; bevar EXTENSION_FAILED og alle checkpoints.
En separat9-konturdiagnose fra gemt prior løser alle1921 celler på10.23s med
uændrede guards. Gammel5-konturprefix giver samme25 uløste celler;0.875 løser dem.
Rårekonstruktion består uden modelreplay;69 gamle numeriske kilder er uændrede.
Ny opt-in adapter er ikke installeret i produktionsrunneren. Fine-check og
provenienssikker adoption mangler; ingen refit/final-Y/modelvalg/forecast.
Native898.08637/900s; H40428.29713/64800s før eksport. Forslag:180 nye native-s,
900→1080, samme18h-loft/23:55-frist; IKKE godkendt eller kørt. Egen fine-prior24h
plus den ene observerede time; base-resultatet genbruges. PR106 forbliver draft.
Læs root/H2_TERMINAL_BANK_EXTENSION_REVIEW.md og den tilhørende proposal-JSON.
Nyeste rå-checkpoints11/12 er sikret i f4_h2_takeover_checkpoint12_20261007.zip;
Library-id libfile_e7115c96991881918e421cd20264bd16. G/WM er uberørt.
Følgende statusafsnit er historik.

# Fælles projektkontekst

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
Læs [matematik, faktisk udført og resultater](root/G7_ENERGY_NEUTRAL_REVIEW.md)
og [M254-runtime/jobdesign](root/G7_M254_RUNTIME_JOB_REVIEW.md).
Følgende statusafsnit er historik.


## F4-H2: pauset med 3.324 råjob; to-stykkers validering består

To-stykkers COBYQA konvergerede efter 20 kald og består fuld egen base/fine-
TRAIN/selection. Fire-stykkers fit har 22 komplette kald af højst 49 samt
53 gemte timer i næste kandidat; endnu ingen konvergens eller modelvalg.
Alle 3.324 jobs, 69 numeriske kilder og begge kaldforløb består råaudit.
Final-Y er ulæst; forecast, finale og I/J-handoff mangler. PR106 er draft.

Ingen modelproces kører efter værktøjsafbrydelsen kl. ca. 18:29 Copenhagen.
Blok 009's hard-stop og blok 010's afbrydelse bevares med eksterne rapporter;
begge 600s reservationer og gammel 30000s-reservation forbliver fuldt debiteret.
Original checkpoint01–08 + extension01–10 bevarer hele kæden. V2-ledgeren
er autoritativ, ca. H39576.36s af64800s, native877.03384/900s; stop23:55.

Controller-source er fortsat 4f4e35ae64cbbf6e1f75f8e10462baacbadf2561.
Et tidligere parent-only SIGINT-stop er kontroltestet uden modeller;
praktisk modelnedlukning afventer første uafbrudte blok. Ingen numeriske
kilder, caps eller præcisionsgates er ændret. G/WM er uberørt.
Læs [aktuel status og fortsættelse](root/F4_H2_PROGRESS_20261007.md) og
[nedlukningsprotokol](root/H2_SOFT_STOP_RECOVERY_20261007.md).
Følgende F4-statusafsnit er historik.


## F4-H2: to-stykkers fit konvergeret og valideret; fire-stykkers fit aktivt

Den særskilte fortsættelse på Johans instruktion 7. oktober kl. 16:28 bruger
25/49 maxfev under uændret 18h-loft og 23:55-frist. Controllercommit:
4f4e35ae64cbbf6e1f75f8e10462baacbadf2561. Aktiv metode er
[root/H2_CAP_EXTENSION_20261007.md](root/H2_CAP_EXTENSION_20261007.md).
Run: files/output/bm_1/runs/estimation/f4_h2_cap_extension_20261007.
Samme f4_h2_same_day_campaign_20261007.json er autoritativ; alle gamle
charges og den åbne 30000s-reservation bevares. 600s-blokke, ingen auto-retry
ved numerisk fejl eller nyt kandidatloft; fuld restvalidering beskyttes.

To-stykkers COBYQA konvergerede med status0 efter20 kald; de17 gamle kald
blev genbrugt præcist. Råreview bekræfter20 objectives uden model-replay.
Dens egne base/fine TRAIN48+selection24 består: total logscore-difference
.0030801524/.0007200562; max state-L1 .002927696. Resultat ligger i
family_2_validation.json. Fire-stykkers optimizer er startet; første27 timer
af den gamle PROBE genbruges. Checkpoint extension01–04 bevarer1965 jobs
oven på den oprindelige01–08-kæde. Nyeste extension_report og campaign,
ikke dette løbende statustal, bestemmer senere genoptagelse.

Modelvalg,168h forecast,finaldiagnose og faktisk I-handoff er endnu åbne.
Final-Y er ulæst. PC-databroen til v2-budget er export_h2_cap_extension_final.py;
kør den først efter den hashverificerede forecastfreeze. De69 oprindelige
numeriske/sourcefiler er uændrede. F/G/WM holdes adskilt. H/I/J/F4 er ikke
slutaccepteret; PR106 forbliver draft. Gammel17-cap-kørsel bevares som negativ
evidens. Følgende statusafsnit er historik.

## F4-H2: terminalt kandidatstop; 1402 jobs gemt, 7. oktober

Dagscontroller7171875 nåede to-stykkers maxfev17 med COBYQAstatus5,
converged=false.17 komplette TRAIN-objectives, alle1402 rawjobs og69 frosne
numeriske/runner-kilder består rawaudit; præcis COBYQA-termination er
rekonstrueret med0 nativepropagationer. Ingen gemte numeriske jobfailures.
326 nyejobs over to lukkede dagsblokke; hele checkpoint01–08-kæden verificeret.
To-stykkers fit er ikke færdigt; fire-stykkers fit, nonlinear selection,
modelvalg, forecast, finaldiagnose og I-handoff mangler. Final-Y er ulæst.

Aktiv run er terminal: SAME_DAY_FAILED.json og optimizerresultat bevares;
ingen retry, markørsletning eller skjult capforøgelse. V2 H34030.249462s,
rest30769.750538s under18h; protectedvalidation22005.997020s;
native854.498417/900s. Begge auditcharges og gammel30000s-reservation bevares.
Stop23:55 Copenhagen. PR106 forbliver draft; H/I/J/F4 ikke afsluttet.
Læs [terminalresultat og konkret metodeforslag](root/F4_H2_TERMINAL_AND_NEXT_DECISION_20261007.md).
25/49 maxfev er kun et særskilt forslag og er IKKE godkendt eller udført.
G/WM og droppede F4-G2b/G2c berøres ikke. Følgende statusafsnit er historik.

## F4-H2: første dagsblok gemt; 1241 jobs, 7. oktober

Dagscontroller7171875 kørte første600s-blok og lukkede rent efter540.23s.
165 nye jobs er gemt; samlet1241. To-stykkers optimizer har14 komplette kald;
næste kandidat er delvist gemt. Ingen numerisk failure, fit er stadig uafsluttet.
Checkpoint07 fortsætter den verificerede01–06-kæde; hele1241-job-kæden er
verificeret uden modelreplay. Snapshothelperen er rettet til at udelade en
gammel campaign_at_checkpoint.json i metadata; første lokale arkiv havde
dobbelt member og blev bevaret som ugyldigt. Kun korrigeret arkiv publiceres.
Numeriske69 kilder og frosne controller7171875 er uændrede. V2-loft18h,
stop23:55 Copenhagen; H33496.780253s efter første blok. Slut-Y ulæst.
[I/J-resultatreview](root/F4_H2_IJ_RESULT_REVIEW_20261007.md) er en arbejdsversion;
H/I/J/F4 er ikke afsluttet. G/WM og droppedeG2b/G2c berøres ikke.
Følgende statusafsnit er historik.

## F4-H2: 18h-budget godkendt; korte beregningsblokke, 7. oktober

Johan godkendte kl15:48:55 et samlet18h-loft og stop23:55 Copenhagen.
Den nye [dagscontroller](root/H2_SAME_DAY_EXECUTION_20261007.md) bevarer alle
69 numeriske kilder, præcis cached COBYQA-trace og gamle jobs/identiteter.
Særskilt v2-ledger bevarer alle gamle entries, inklusive30000s-crashcharge;
ingen refundering eller reset. H32956.549305s før første nye blok;
native848.307761/900s. Restvalidation22005.997020s beskyttes med aktuelle
maksimumrater/faktor1.5. Fuld49-ramme eller konvergens loves ikke.
Én scoped budget-/delegationskontrol består med0 modeljobs. Miljøet matcher.
600s invocations, lukket ledger og publiceret delta efter hver blok.
Aktuelle resultater følger i same_day_report_NNN.json; cap/failure er terminal,
invocation-/budgetstop er pause. Final-Y afventer faktisk frosset forecast.
G/WM og droppede G2b/G2c berøres ikke. H/I/J/F4 er endnu ikke afsluttet.
Følgende statusafsnit er historik.

## F4-H2: dagsplan og præcis genoptagelse kontrolleret, 7. oktober

Johan ønsker at afslutte F4 i dag, hvis muligt. [Restplan og budgetforslag](root/F4_SAME_DAY_REMAINING_PLAN_20261007.md)
bygger på alle1076 hashverificerede jobs: fuld resterende kandidatramme
4.85h ved målte gennemsnit,7.19h med faktor1.5,14.01h ved maksimumrater.
Det er scenarier, ikke målt kode-speedup eller konvergensløfter; protected
validation er fortsat6.11278h. Alle12 cached COBYQA-kald og næste afbrudte
kandidat matcher; den gemte fire-stykkers PROBE genbruges også. Ingen nye
modeljobs. Runtime numba0.62.1/llvmlite0.45.1 er gendannet fra lokale wheels;
hele miljøet og alle69 numeriske kildehashes matcher. Anbefalet fortsættelse
bruger600s invocations med lukket ledger/durable delta efter hver blok.

H32941.549305s; native833.307761/900s; oprindeligt10h-loft er uændret.
Forslag18h samlet og dagsstop23:55 Copenhagen er IKKE godkendt/implementeret.
Ingen STOP-fjernelse eller ny modelkørsel. Ny controller-/budgetrevision
skal bevare gammel reservation og alle kilde-/jobidentiteter. F/G/WM holdes
adskilt, G2b/G2c er droppet. H/I/J er fortsat uafsluttet; betinget historisk
48/24/168h-studie er ikke blind H3 eller generel forecast-accept.
Følgende statusafsnit er historik.

## F4-H2: crash gendannet; videre kørsel budgetblokeret, 7. oktober

Alle seks checkpoints fra `e79aa10` er faktisk gendannet og hashkontrolleret:
1076 jobs, 69 frosne sourcefiler, 12 komplette to-stykkers objective-kald
(inkl. lineær start), næste kandidat34/72 timer. Fire-stykkers fit er ikke
startet. Ingen optimizer-resultatfil, terminalrapport, selection eller
frosset forecast. Ingen gemte failed-jobfiler; ingen accept eller ny modelkørsel.
Alle536 lokalt genfundne jobs indgår byte-identisk i gendannelsen.

Den autoritative ledger matcher checkpoint06 og har en åben30000s-reservation.
Crashreglen bevarer den fuldt. Efter særskilt60s recoverycharge er
H32911.549305s, rest3088.450695s; native803.307761/900s.
Sidste resterende validation19960.314316s overstiger restbudgettet.
Ingen automatisk refundering, ny reservation eller fitstart; ikke brug
en ældre seed/ledger. Aktuel afstemning og ledger-efter er gemt i
`f4_h2_crash_budget_reconciliation_20261007.json` og
`f4_h2_crash_campaign_after_20261007.json` under estimation-output.
Basejobs er nu målt op til9.62828s: nyt budget kræver genprisning.
Numba0.62.1/llvmlite0.45.1 skal også gendannes før numerisk resume.

Læs [gendannelse, evidens og præcist næste trin](root/F4_CRASH_RECOVERY_20261007.md).
Videre beregning kræver en senere autoritativ lukket ledger eller Johans
udtrykkelige nye budgetbeslutning. Det eksisterende36000s-loft er uændret.
PR106 forbliver draft; F4/H/I/J er ikke afsluttet. G/WM ændres ikke.
Følgende statusafsnit er historik.


## F4-H2: hel lineær reference består; tidsbegrænset natforsøg, 7. oktober

Aktiv metode: [budgetteret søgeforsøg](root/H2_COMPACT_BUDGETED_SEARCH.md).
96 kronologiske timer (24 prior, TRAIN48, selection24) består egne base/fine
streams. Total-scorefejl .00555411/.00045416, max state-L1 .00104931.
Den gamle lineære PC-prefix og nonlinear-prøvens 64 jobs genbruges.
Råreviewet har ingen FV-replays. Alle 69 numeriske/sourcefiler fra
`e115242de79756a4a7163da28593f2452f17bcc6` bevares uændret.

De faktiske senere rates er højere: base8.30102/fine18.13960 s observeret time.
Komplet konservativ 49-par-graf er nu14.61 h og er AFVIST. Ny eksplicit
admission-revision tillader alene et tidsbegrænset konvergensforsøg:
hele resterende validation19960.3143 s beskyttes først; ved seneste H2851.55 s
er ca13128.14 s tilgængelig for søgning plus særskilt60 s stopmargin.
Det giver ingen garanti for alle49 kald eller konvergens. Samme1/2/4 familier,
17/33 maxfev,48/24/168 split, precisiongates og samlet36000 s loft.
Cap/timeout/failure betyder ufuldstændigt fit; ingen manglende familie ommærkes.

Ny ydre controller `root/f4/run_h2_budgeted_search.py` har eget source-/execution-
manifest og genbruger original Engine/COBYQA/selection/forecast. Det gamle
`run_h2_compact.py run` må ikke genoptages efter runtime-reviewet.
Ingen optimizer var startet ved helvinduesreviewet. Slut-Y er stadig ulæst.
Fysik/støj er nominelt betinget, tidligere slutugeeksponering ukendt;
historisk fit/videnskabelig/blind-H3 accept er ikke afsluttet.

Nine scoped orchestration controls, prior-API control, nonlinear raw review
og ny reserve-/sourcekontrol består. To preparation-fejl og alle charges er
bevaret. H2851.5493 s; native743.30776/900 s, rest156.69224 s,48 entries.
PR106 er et løbende checkpoint; læs `root/F4_NIGHT_CHECKPOINT_20261007.md`
for aktiv run/ledger. G/WM og legacy/native kerner ændres ikke.
Følgende statusafsnit er historik.

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
Læs [matematik, faktisk udført og resultater](root/G7_ENERGY_NEUTRAL_REVIEW.md)
og [M254-runtime/jobdesign](root/G7_M254_RUNTIME_JOB_REVIEW.md).
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
Læs [kontrakt, udledning, kilder og faktiske resultater](root/G7_MATCHED_CONTROL_REVIEW.md).


## F4-H2: faktisk lineær PC-pilot råreviewet, 7. oktober

Aktiv overdragelse: [historiske rates og næste admission-design](root/H2_COMPACT_RATE_REVIEW.md).
PC-invocation120.03943 s består27 kronologiske timer (24 prior+3 observerede
TRAIN), på egne base/fine states.64 sourcehashes matcher præcis Git-source;
64 raw jobs/checkpoints, CF→CDF/kvantiler, moments/score og budgetgraf er
verificeret uden FV/JIT-replay. Præcision gælder den lineære prefix alene.
Maks state L1 .000493025, total log-scorefejl3.52e-6; alle uændrede gates består.

Prissat49-kandidat/hele-valideringsgraf med faktor1.5 og alle auxiliaryjobs
estimerer7.8513 h efter reviewcharges: under10 h, over4–6 h-målet. Linear
prefix er ikke et nonlinear worst-case eller full-window precisionbevis;
fit/runtime/scientific/blind-H3 admission forbliver false. Ingen optimizer.
Original rå-ZIP, PC-ledger, Git-evidence, auditor/rapport og alle fejlcharges
bevares; to read-only auditorrettelser er dokumenteret, ingen model-genkørsel.
Ny seed: H1671.04086 s, native646.40974/900 s, rest253.59026 s,38 entries.
Beskyttet validation-reserve hæves til11840.47191 s, uden dobbelttælling.

Næste er et konkret nonlinear/full-window admission-design under den
resterende allowance, ikke en gentagelse af prefix eller automatisk fit.
Sol High til råreview/implementering; G/WM/frosne data/native kerner uændrede.
Følgende statusafsnit er historik.

## F4-H2: fejlforsøgets ledger samlet, ny PC-prøve klar, 7. oktober

Den komplette PC-ledger er modtaget som tekst og matcher alle 30 tidligere
entries nøjagtigt; fejlforsøgets 6.964884199900553 s er bevaret som entry31.
Den afsluttede eksportkontrol på 0.3752458700000716 s tilføjes uden genkørsel,
og ledgerens råreview får 2 s konservativ charge. Ny seed er et præcist
prefix-superset med 33 entries: H **1543.03445744 s**, native
**518.40333680/900 s**, rest **381.59666320 s**. Original fejlforsøgsledger,
ny samlet ledger og reconciliation-rapport er bevaret. Windows ordered-sum
og den hidtidige historik er verificeret. Fejl-ZIP er ikke modtaget; ledger-
verifikation er ikke et raw source/checkpoint-review eller numerisk accept.

PR100's afgrænsede -text Git-attributter bevarer frosne audit/packet-bytes
under Windows-eksport; hashgate og numerik er uændrede. Én ny invocation må
måle den oprindeligt planlagte historiske lineære pilot, højst 300 s inkl.
60 s stopreserve, med source-commit der også indeholder denne nye seed.
Launcheren forlænger automatisk PC-ledgerens identiske prefix; ingen lokal
manuel ledgeroverskrivning, ny datapreparation eller gamle modelchecks.
Sol High; 8 frekvensworkers/BLAS1; review-ZIP også ved fejl/stop, ingen auto-
retry. Fit/admission/scientific/blind-H3 flags er fortsat false. G/WM er
uændrede. Aktiv overdragelse: [historisk pilot](root/H2_COMPACT_RATE_PILOT.md).
Følgende F-statusafsnit er historik.

## G7: deterministisk primærreference gennemført, 7. oktober

Johan prioriterer deterministisk BM/WM-kontrol; stokastisk G7 beholdes som
obligatorisk leverance. Original FP2025-BM bruger kendt fixed A(t); der
opfindes ikke en ny WM-indkøbsfunktion. Én syntetisk24h/fire blokke består
bounded residual-LS/TRF og coarse/fine: J=.0209268019845 mod neutral .12,
82.561% mindre,68 actual forwards inkl.Jacobian probes,14.227s invocation.
Begge starter konvergerer via ftol; ikke global/KKT-optimalitetsaccept.
Én analytisk kontraktkontrol består efter rettet assertion ved aktiv bound;
failed attempt/source/raw er bevaret.27 gamle importerede Git-blobs matcher;
F/WM/shared-kerner og legacy-optimizere ændres ikke.
Næste er fælles fixed-A BM/WM-benchmarkkontrakt og frozen-fit/prisrespons-
audit før historisk kontrol. Faktisk WM-rangering og stokastisk policy/
mixture/EM/feedback er åbne; den gamle G7-runtimepilot genkøres ikke.
G-ledger1062.580078s,rest737.419922s; F/WM-budgetter er uændrede.
Sol High er tilstrækkelig; ingen lokal modelkommando nu.
Læs [G7-matematik, kilder og faktisk udført](root/G7_DETERMINISTIC_CONTROL_REVIEW.md).


## F4-H2: Windows-exportfejl rettet, PC-ledger afventer råreview, 7. oktober

PC-forsøget `f4_h2_compact_rates_pc_20261007_014819` stoppede ved
`Raw data audit changed.` i frozen_packet, før observationslæsning/modeljobs.
Rapport og terminal oplyser 6.964884199900553 s invocation og
1540.6592115679418 s H; review-ZIP/fuld PC-ledger er endnu ikke modtaget.
Ingen ny pilot eller seed-reset før den komplette ledger er reconcileret.

Årsagen er reproduceret: Git archive med core.autocrlf=true ændrer audit-LF
til CRLF. Afgrænset `.gitattributes` -text bevarer de frosne packet-/audit-
bytes; strict SHA256 og alle numeriske gates er uændrede. Launcheren og
source-manifestet inkluderer nu attributfilen. Ét Git-only check reproducerer
fejlen og bevarer alle fire frosne filer i seks autocrlf/eol-konfigurationer.
Check 0.3752458700000716 s, ingen model/JIT/workbook. Råreceipt:
`files/output/bm_1/runs/estimation/f4_h2_frozen_export_check_20261007.json`.
Dette check-charge afventer sammenfletning efter PC-entry; budgetseed er ikke
ændret. Sol High fortsætter. G/WM/frosne data/native kerner er uændrede.
Aktiv overdragelse: [historisk pilot](root/H2_COMPACT_RATE_PILOT.md).
Følgende F-afsnit er historik.

## F4-H2: faktisk rate-/præcisionsprøve klar til PC, 7. oktober

Aktiv overdragelse: [afgrænset historisk pilot](root/H2_COMPACT_RATE_PILOT.md).
Sol High: ny immutable PC-launcher måler den lineære F-reference på faktisk
MED-data med uafhængigt base/fine Beta(20,20)-priorforløb 17.–18. februar
og de første tre komplette TRAIN-timer i kronologisk rækkefølge. Manglende
timer propagateres, state nulstilles ikke. Base1921/h=1/32768 og
fine2561/h=1/65536, 8 frekvensworkers/BLAS1, uændrede precisiongates.
Pre-update intervalmomenter og to CDF/kvantil-prober per grid indgår.
Ingen optimizer, ingen prior-/final-Y; TRAIN+selection læses kun til
frosne observationer/counts. MED's 13 huse er separat og uden imputation.

Runner: `root/f4/measure_h2_compact_rates.py`; launcher
`root/f4/run_h2_compact_rates.ps1`. Højst 300 s inklusive 60 s stopreserve
under den fælles native/H-ledger. Checkpoints og review-ZIP også ved rent
budgetstop/fejl; ingen automatisk gentagelse eller løsere tolerancer.
Varm maksimumrate for observeret/prediction/moment/CDF på begge grids
priser alle godkendte 49-kandidat/validerings-/ugejobs med faktor1.5.
Lineær prefix-ekstrapolation er ikke nonlinear worst-case eller fuld
TRAIN/selection/final præcisionsaccept: alle fit/admission/accept-flags=false.

Én ny fokuseret data-boundary/jobgraf-kontrol består på 1.39395 s; ingen
FV/JIT-propagation eller gamle checks genkørt. Kilde/rapport bevaret i
`f4_h2_compact_rate_control_20261007_review.zip`. Senere startup/shutdown/ZIP-ændringer
er statisk kontrolleret; boundary-/jobgraf-funktionernes AST er identisk.
H-ledger **1533.69433 s**; native **509.06321/900 s**,
rest **390.93679 s** samlet. Den faktiske PC-modelprøve er endnu ikke
kørt: den private demand-workbook findes kun på PC. Brug ny seed, ikke
datapakkens historiske budgettal. G/WM og native kerner er uændrede.
Følgende F-statusafsnit er historik.

## F4-H2: betinget datagrundlag frosset og råreviewet, 7. oktober

Aktiv overdragelse: [F's betingede datagrundlag](root/H2_COMPACT_DATA_FOUNDATION.md).
PC-pakken fra `d95fd26` består råaudit uden genkørsel eller fit. Historik
1. januar→16. februar: 1104 MED-gruppetimer, 984 komplette og 120 manglende.
Alle 168 weekday/hour-OLS-bins har 5–7 komplette samples. Effektskala
3.1265384615384613 kW; DK1 spot-proxy-bounds −259.66200100000003 og
2865.162011 DKK/MWh. De 13 MED-huse holdes separat; ingen imputation.
60 sourcefiler matcher Git-blobs; rå baseline/NPZ er konsistente, priser
er uafhængigt rekonstrueret fra hash-verificeret workbook. Demand-workbook
medfølger ikke: rå hushistorik/OLS/effektmax er ikke uafhængigt genberegnet.
Provenance følger verificeret runner og inputhashes. Target-Y er fraværende.

Aktiv indeksfil: `root/f4/h2_compact_data_ready_20261007.json`; original ZIP,
packet, baseline, NPZ, audit og komplet ledger er bevaret. Conditional data
ready=true; historisk fit, numerisk likelihood, runtime, blind H3 og
videnskabelig accept=false. Nominale fysiske/støj/prior-betingelser og
udviklings-holdout med ukendt tidligere eksponering fastholdes.

PC-preparation 30.45019 s; råreview 1.39197 s plus 2 s konservativ
receipt-charge. H-ledger **1532.30038 s**; native **507.66926/900 s**,
rest **392.33074 s** samlet på PC/chat/resumes. Ny seed udvider hele
historikken; gentag ikke preparation eller afsluttede native cases. Næste er
faktiske historiske base/fine/moment/CDF-rates og prissat samlet jobgraf med
base h=1/32768/fine h=1/65536 før fit. Gammelt 8.76 h-proxyoverslag åbner
ikke fit. Samlet 10 h loft, 4–6 h mål og alle precisiongates bevares.
Sol High er tilstrækkelig. Følgende F-statusafsnit er historik.

## F4-H2: data-policy implementeret, lokal frysning er næste trin, 6. oktober

Aktiv overdragelse: [F's betingede datagrundlag](root/H2_COMPACT_DATA_FOUNDATION.md).
Johan valgte kurver først: C=1/k=2/Xi=1/sigma_x=.5/Beta(20,20) er faste
F-referenceantagelser; sigma_y=.04 er særskilt nominal likelihood-støj.
Ingen historisk fysisk calibration eller G/WM-parametre. De 13 MED-huse
behandles separat. Historik 1. januar→16. februar (1104 h) giver en frosset
168-bin OLS-baseline (mindst 3 komplette timer/bin), effektskala 1.1×max
og DK1 spot-proxy-min/max med 10% historikpadding. Ingen imputation/klipning.
Kun pre-origin forbrug og pre-final prisinput læses; target-Y er ikke i pakken.
Slutugens tidligere eksponering kan ikke verificeres; udviklings-holdout,
ikke blind H3-accept. En exposure-intervalguard afviser blindhedsclaim.

Ny policy: `root/f4/h2_compact_data_policy_20261006.json`; gammelt partial og
metodevalg bevares. Native kerne/celler/precisiongates uændrede; fysisk
begrundet revision til base h=1/32768 og fine h=1/65536 for positiv CN.
Det gamle 8.76 h-proxyoverslag må ikke åbne fit. Faktiske rates og samlet
jobgraf mangler stadig; 10 h samlet loft, 4–6 h mål. Ingen fit startet.

Én kort syntetisk datakontrol består på 4.06 s; OLS-fejl 1.78e-15,
cutoff/poison/NaN/hash/overwrite/positivity/blindhedsafvisning kontrolleret.
H-ledger 1498.46 s; native 473.83/900 s, rest 426.17 s samlet.
Stor privat input.xlsx kan ikke hentes her: faktiske baseline/skala-tal
skal fryses med `root/f4/run_h2_compact_data.ps1` lokalt og ZIP råreviewes.
180 s maksimum inklusive 60 s stopreserve, 1 proces/BLAS=1, ingen model/fit.
Launcheren eksporterer immutable source uden aktiv branch/worktree-skift.
Sol High er tilstrækkelig. Historisk/numerisk/runtime/videnskabelig accept
forbliver false. Følgende F-statusafsnit er historik.

## Vejledermateriale og legacy — Johan, 6. oktober 2026

Alle tavlebilleder Johan har givet er som tommelfingerregel materiale fra
vejlederen og det oprindelige faglige grundlag. Bevar de oprindelige
legacy-modeller, formler, kodeveje og råresultater. Udvidelser er mulige,
men kræver grundig matematisk udledning, konkret begrundelse, kildeangivelse,
sammenligning og tydelig dokumentation af forskellen fra originalen.
Skeln mellem direkte vejledermateriale, standardidentiteter og egne
metode-/numerikvalg. Erstat eller ommærk ikke originalen stiltiende.

For G-mixture er tavlens normaliserede Bernstein/Beta-familie primær;
den nye lokale 42-leds Beta-basis er kun sammenligningsreference.
FV er numerisk reference. Metodeprioritet ændrer ikke acceptkrav eller
negative testresultater. Genbrug den allerede gennemførte referencekørsel.
Detaljeret kildekort/udledning: [G8-basisnoterne](root/G8_POSTERIOR_BASIS_REVIEW.md).
Denne dokumentationsregel ændrer ikke F/WM's aktive opgaver eller resultater.

## G7: own M254-handoff implementeret, runtimepilot stoppet, 7. oktober

Ny G-adapter bevarer typed supervisor M254, hver nodes egne c/posterior og
frosne vægte ved candidate-forward. Kvadratisk tracking inkluderer within/
between-varians; candidate-state nulstilles til samme origin. Shared kerner,
legacy-evaluatorer og F/WM ændres ikke. Én kontrakttest består .410s.
Pilot12→14: coarse neutralforward returnerer ved17.61s elapsed; fineforward
afbrydes ved60.001s-loftet. Ingen paired precisionaccept, optimizer, EM eller
actual feedback; G7-accept er false. G8's positive egen-filterevidens bevares.
Original39 kildehashes/inputs og V1-source er frozen; coarse trajectory blev
ikke råeksporteret før timeout. Corrected runner eksporterer completed forwards
straks/audit ved stop og håndhæver oprindelige marginal-/feedback-gates eksplicit.
Kun syntaks efter fix; ingen retry eller ny numerik. Se det konkrete review.
Conservative paneldebit1045.580s inkl.2s testcharge/arkiv, rest754.420s;
oprindelig umålt-probe-reservation bevares, budget nulstilles ikke.600s scoped
G7-loft er ikke600s forbrugt og åbner ikke nyt job efter failed60s-pilot.
Næste er afgrænset runtime-/jobdesignreview før et særskilt registreret run.
Ingen større bank, løsere gate, noiseændring eller automatisk fuldt døgnjob.
Ingen lokal modelkommando nu. F4-H2/WM-budgetter og status er uændrede.
Læs [G7-matematik og faktisk udført](root/G7_OWNED_HANDOFF_REVIEW.md).

## G8: eget Bernstein M254-filter forfinet, 6. oktober

Eget 513/1-192-filter med vejlederens 255 celleintegrerede Bernstein-funktioner
består hele døgnet og alle uændrede paired gates mod eget257-filter/FV513.
Conditional L1 mod FV513 .00354065; marginal .00108111; projektion .000971943.
Faktisk typed JSON-resume22→24 er identisk i masser/c/logvægte og begge
likelihoodfelter. Filter309.743s; resume17.327s; invocation327.829s.36 importerede
Gitblobs/37 sourcehashes og alle inputs er uændrede. Gamle resultater bevares.
Dette frigiver M254's afgrænsede syntetiske repræsentation til G7-interfacearbejde;
G7-mixture-prisforward/prisoptimering, historisk forecast og parameterrecovery
mangler stadig. FV-control-helperen må ikke ommærkes til mixture-control.
Næste er et typed own-origin-handoff og kort prisforwardpilot, derefter en
konkret samlet jobgraf før optimizer. Sol High er tilstrækkelig til adapteren;
Astra High kan bruges til den nye objective/handoff-matematiks review.
Nyt scoped loft900s efter eksplicit fortsættelse; gamle300s-forsøg ændres ikke.
Panel1800s bevares; tidligere målt ca345s+konservativ300s probe/arkivreserve
afrundes til debit650s, nye faktiske tider lægges til uden budgetnulstilling.
Budgetregnskabet er en konservativ reservation, ikke komplet timing af gamle
umålte probes; G7 er ikke admitted. Ingen automatisk ekstra panel eller toleranceløsning.
Læs [matematik og resultater](root/G8_M254_REFINEMENT_REVIEW.md).
F/WM's status, kerner og budgetter er uændrede.

## F4-H2: intervalprognose implementeret, 6. oktober

Aktiv overdragelse: [intervalprognose og næste datatrin](root/H2_INTERVAL_FORECAST_IMPLEMENTATION.md).
Sol-adapteren fører intervalmiddelværdi/varians og fuld log-state frem med den
samme positive native operator; CDF/kvantiler på syv horisonter, egne fine gates
og checkpoint/resume ved timegrænser er implementeret og afgrænset kontrolleret.
Uafhængige matrix-/Gauss-kontroller består; ingen gamle 40-timers cases genkørt.
De 13 præcise MED-kolonnenavne er nu fastlagt fra Johans eksisterende header-rapport.
Ny delvis F-datapakke: `root/f4/h2_compact_data_packet_20261006.json`.
Pris/skala, historisk MED-baseline, fysisk/støj-betingelse, Beta-prior og
slutugens eksponeringsaudit mangler stadig reel provenance før historisk rate/fit.
H-ledger 1494.40 s; native 469.77/900 s, rest 430.23 s på tværs af PC/chat.
Ingen historiske observationer læst, ingen fit eller faktisk uge-runtime godkendt.
Sol High er tilstrækkelig; ingen ny lokal modelkommando endnu. Følgende afsnit er historik.

## F4-H2: kompakt metodevalg efter Astra-review, 6. oktober

Aktiv overdragelse: [kompakt H2-design](root/H2_COMPACT_ESTIMATION_REVIEW.md).
Det tidligere syvdages TRAIN-design erstattes eksplicit af et mindre studie:
TRAIN 18.–19. februar 2022 (48 h), modelvalg 20. februar (24 h), uændret
slutuge 21.–27. februar (168 h), prediction-only prior fra 17. februar.
MED's 13 huse forbliver separat; samme præcise interval-likelihood, native v2,
lineær/to-/fire-stykkers kurver og COBYQA-loft på 49 par. Ingen daglige fits.
Ugeprognosen får alle 168 intervalmiddelværdier/varianser via positive
momentrekurrenser; fordelinger på de syv eksisterende horisonter 1/6/12/24/48/72/168 h.
Der er udledt en konkret implementering med bevaret fuld log-state og egne
fine streams. Det samlede betingede budgeteksempel er 8.76 h, inkl. tidligere
tid; valideringsreserven er ca. 4.01 h. Fine/moment/CDF-rates er endnu umålte:
intet historisk fit er admitted, og 4–6 h er ikke dokumenteret.
Næste trin kan udføres på Sol High: implementér adapteren, færdiggør F-kontrakten
og pris hele grafen før fit. Brug højst resterende 464.10/900 native-sekunder;
ingen gentagelse af de afsluttede 40-timers checks. Reviewet kørte ingen modeller
og ændrede ikke ledgeren på 1460.53 s. G/WM og legacy-kerner bevares.
Det mindre studie er ikke accept af det gamle syvdages fit, måneds-generalitet,
kalibreret populationsdækning eller hele F4. Følgende statusafsnit er historik.

## F4-H2: native v2 PC råkontrolleret, metodevalg er næste trin

Aktiv næste opgave: [review efter native v2](root/H2_NATIVE_V2_NEXT_REVIEW.md).
PC v2 består 40 units/36 observerede opdateringer på 104.31 s, cirka 12.1%
kortere invocation end første PC. Alle 54 sourcehashes/checkpoints/referencer/
guards/ledger består uden numerisk replay; største finite-log-fejl 9.19e-10.
Varm maksimumstime er 4.06821 s, så 49-par-planens basearbejde alene illustrerer
15.38 h med den uændrede 1.5-faktor. Historisk fit er fortsat ikke admitted.
Ingen ny timingprøve: næste er et konkret matematisk søge-/valideringsreview,
anbefalet Astra XHigh, derefter Sol High-implementering af det valgte design.
MED/valgte uger og acceptkrav bevares indtil en eksplicit begrundet designrevision.
Fuld F-datakontrakt og faktisk prissat heljob-gate er stadig åbne; G/WM uændret.
H-ledger 1460.53 s; native 435.90/900 s, rest 464.10 s samlet.
Original ZIP/source/ledger og råaudit er bevaret; ny seed udvider historikken.
Ingen nye modelkørsler, historiske værdier eller bred suite i råreviewet.
Følgende afsnit er historik og må ikke starte de afsluttede prøver igen.

## F4-H2: PC-native rå-reviewet, hurtigere v2 kontrolleret

Aktiv indgang: `root/H2_NATIVE_PC_REVIEW.md`. Første PC-native består alle
40 units/36 observerede opdateringer på 118.67 s; 54 sourcehashes/checkpoints/
referencer/guards/ledger er råvalideret uden FV-replay. Python 3.11-scoren
rekonstrueres eksakt i read-only-auditor; execution/resume/checksums ændres ikke.
Ved den målte varme maksimumstakt 3.54858 s giver den fulde 49-par-plan 13.41 h
alene for basearbejde. Historisk fit er derfor ikke admitted; fine-/forecast-
arbejde og faktisk historisk rate mangler stadig. Ingen automatisk kortere TRAIN,
manglende kurvefamilie eller gammel 17-kandidatpanel som løsning.
Samme native metode v2 fjerner gentagne exp/log/divisioner og består igen de
40 matchende timer på 80.72 s her; største log-state-fejl 9.19e-10 mod 1e-7.
Næste PC-kommando bruger nyt v2-run-dir og højst 300 s, inklusive 60 s stopreserve.
H-ledger 1352.86 s; native 328.23/900 s, rest 571.77 s over alle maskiner/resumes.
Nyeste seed udvider PC-historikken; originale rådata/source/seeds er bevaret.
MED/tre februaruger og 10-timersloftet bevares; G/WM og de 49 gamle F-kilder
er uændrede. Fuld datakontrakt og prissat heljob-gate blokerer fortsat optimizer.

## F4-H2: native kerne kontrolleret, 6. oktober 2026

Aktiv overdragelse: `root/H2_NATIVE_DEVELOPMENT.md`. Isoleret log-Thomas/Numba
med 8 frekvensworkers og BLAS=1 består uafhængig algebra og 24+8+8 gemte
syntetiske referencetimer på 104.43 s. Maksimal log-state-forskel <1e-9 mod
1e-7. Gamle referencer/49 F-kildefiler er uændrede; ingen G/WM-kobling.
STOP/resume og fælles ledger er afgrænset kontrolleret. H-ledger 1147.38 s;
native-udvikling 122.75/900 s, højst 777.25 s til PC/alle resumes inklusive
kompilering/imports og 60 s stopreserve. Brug ny seed, aldrig den gamle
468.09 s-seed. `root/f4/run_h2_native.ps1` måler samme case på PC og skriver
review-ZIP; historisk fit er ikke en runnerkommando. PC-hastighed, fuld
F-datakontrakt og prissat heljob-gate mangler. Ingen ekstra panel eller automatisk
optimizer. MED/perioder forbliver valgt; Astra-reviewet nedenfor er nu historik.

## F4-H2: Astra-metodevalg færdigt, 6. oktober 2026

Aktiv overdragelse er `root/H2_METHOD_REDESIGN.md`. MED's 13 elvarmehuse
vælges separat, med fast gruppemiddel. TRAIN er 7.–13. februar 2022,
MODEL_SELECTION 14.–20. februar og planlagt slutevaluering 21.–27. februar;
sidste uges blindhed kræver stadig eksponeringsaudit. Datoer er hour-start-
vinduer; målinger bliver tilgængelige ved intervalslut. Population/perioder
skal ikke vælges igen. Øvrig F-datakontrakt er stadig ufuldstændig.
Præcis interval-likelihood bevares; næste Sol High-opgave er en isoleret
kompileret positiv Thomas-rekurrens med 8 frekvensworkers/BLAS=1.
Matematisk ækvivalens er udledt; implementering/hastighed er ikke kontrolleret.
Højst 900 nye sekunder til den afgrænsede implementeringskontrol, inklusive
kompilering/resumes og 60 s stopreserve. Ingen nye forsøg er kørt i reviewet.
Senere adaptiv COBYQA er betinget af fuld data-/numerik-/budgetadmission;
49 TRAIN-par er et loft, ikke en bestilling eller konvergensgaranti.
Budgettér sekventielle kald, alle tre uger, finere replays og forecastarbejde;
den gamle ceil(kandidater/8)-formel gælder ikke frekvensworkers. Forøg den
mindst 2-timers valideringsreserve til faktisk prissat arbejde om nødvendigt.
10 timer er stadig samlet loft, mål 4–6 timer; ledger står på 1024.63 s.
Ingen automatisk 17-/73-kandidat-kørsel eller ny grov solverrunde. G/WM uændret.

## F4-H2 Windows-runtimepilot rå-reviewet, 6. oktober 2026

Johans syntetiske pilot er nu rå-reviewet: otte valide kandidat-checkpoints,
64 timer inklusive 56 Bayes-opdateringer og otte missing-prædiktioner.
Manifest/kandidat/checksums, log-normalisering, guards og tidsprojektion består
uden FV-replay. Alle 49 kildefiler og fixture matcher præcist pilotens Git-
indhold med Windows LF→CRLF-checkout; originalhashes/resumelåse er uændrede.
8 workers/BLAS=1, 556.53 s. Samlet H-ledger er stadig **1024.63 s**.
Rapportens 5.50 timers H-estimat gælder kun **24 syntetiske TRAIN-timer**;
det åbner ikke historisk search. Samme forsigtige regneregel på syv dage
giver 20.65 timer inklusive foreløbig 2-timers reserve og allerede brugt tid.
Den gamle 17-kandidater/syv-dage-budgetillustration er derfor overhalet.
Original ZIP, reproducerbar råauditor og raw_audit.json er arkiveret.
F-konfigurationen er stadig ufuldstændig: population, TRAIN/split/eksponering,
pris/skala, tidligere baseline/prior og fysisk/støj-betingelse mangler.
Næste er disse deklarationer og et revideret søgedesign, ikke endnu en pilot.
Se root/H2_PRECISE_BATCH.md. Ingen automatisk resume, ny pilot, kortere TRAIN
eller færre kurvefamilier; 10-timersloftet og alle H-acceptkrav bevares.
Ved flytning mellem PC/chat bruges den seneste fælles ledger, ikke seedens
gamle 468.09 s. Ingen numeriske kildefiler er ændret ved denne statusopdatering.

## F4-H2 præcis batch, 6. oktober 2026

Det begrænsede søgegrundlag er implementeret: `bm_1_h2_precise`, præcis
terminal-tilt, 17 faste koordinatkandidater og kandidat-parallelitet med egen
log-state hele TRAIN. Brug `root/H2_PRECISE_BATCH.md`; legacy `curves history-fit`
er ikke den nye intervalbackend. Den syntetiske 8-worker/8-time-runtimepilot
er nu rapporteret færdig; se status ovenfor. Den fastlægger ikke historisk
throughput. F-datakontrakten skal stadig udfyldes
før historisk prepare/pilot/search; G's kontrakt må ikke overtages automatisk.
Alle negative guards/budgetstops bevares. 17-kandidatprøven er ikke konvergeret
fit eller TRAIN-modelvalg. Numerisk/historisk/H2/H3-readiness er fortsat falsk.
Ingen automatisk fuld suite, ekstra panel eller lang kørsel i chatten.

## F4-H beregningsbudget, Johan 6. oktober 2026

Samlet resterende H-beregning skal planlægges mod 4–6 timer og holde sig under
10 timer inklusive piloter, fits og nødvendige kontroller. Budgettet følger
forløbet på tværs af resumes; de tidligere cirka 90 timer er afvist.
Den afgrænsede interval-søgeprøve er nu kørt på 415.89 s i alt.
De første timer kunne bestå, men positiv joint-CN med 321 celler og
da=1/1024 fejler den genbrugte døgnreference: score 3.58901, maksimal
endpoint-mean .00822776 og L1 .26784242. Ingen billig backend er godkendt.
Næste prioritet er et lavere, eksplicit kandidatbudget og parallelle
HELE TRAIN-replays med den præcise metode, efter datakontrakt og en kort
repræsentativ runtimepilot. 17 kandidater/7 dage/8 workers er kun en
budgetillustration, ikke et godkendt run eller et løfte om fitkonvergens.
Se root/H2_SEARCH_FEASIBILITY.md og root/H2_SEARCH_NEXT_HANDOFF.md.
Ingen automatisk ny surrogate-/gridpanelrunde eller Fourier-runtimekontrol.
De gamle grove grids fejlede numerik; de må ikke genindføres som accepteret
likelihood. Kontroller kandidatforskelle og valgte løsninger, og stop ved
utilstrækkelig præcision eller budget. Genbrug fitter/data/checkpoints og
timeholdte B/U-inputs. Brede profiler/bootstrap/robusthedspaneler er udsat;
fælles f/g, datakontrakt, modelvalg og frosset slutvalidering bevares.
Se den nye budgetbeslutning øverst i root/F4.md. Budgetmål er ikke målt runtime.

## Arbejdsregel fra Johan, 5. oktober: adskil G/F og minimér kontrolarbejdet

G er den deterministiske BM-kæde med et senere sandsynlighedslag; F er den
stokastiske BM-kæde. De har egne planer, kørsler, modelvalg og acceptkrav.
G må ikke vente på F/H-gates, og F må ikke få G's kontroller som standardkrav.
Genbrug stabile matematiske kerner gennem små eksplicitte interfaces; undgå
at koble fitforløb, konfigurationer og testpakker sammen eller kopiere kerner.
G's senere FP/EM-lag arver kun relevante interfacekrav, ikke hele F's forsøgsprogram.

- Vælg den mindste eksisterende test/metode, der dækker den konkrete ændring.
  Ingen fast 20-/29-testpakke for G og ingen automatisk fuld discovery for F.
- Rene tekst-/planændringer kræver ingen numeriske tests eller modelkørsler.
  Kør ikke hele pakken igen efter en bestået relevant kontrol uden ny grund.
- Tilføj kun en test for en konkret væsentlig fejlrisiko, som ikke allerede
  dækkes. Ingen ekstra cases, prøvekørsler eller benchmarkpaneler af vane.
- Ændres fælles kode, kontrolleres kun de faktisk berørte G/F/WM-kontrakter.
  Windows, layout og LaTeX kontrolleres kun, når ændringen kræver det.
- Bevar numeriske tolerancer, datavalidering og nødvendige acceptkrav i selve
  forskningskørslerne. Genbrug gemt evidens; færre gentagelser er ikke ny accept.
- CI foretager kun let syntakskontrol af ændrede Python-filer på PR'er.
  Relevante numeriske checks køres lokalt eller vælges med præcise unittest-
  navne via workflow_dispatch. Fuld regression og F4-PDF er eksplicitte tilvalg.
  Ingen ekstra push-kørsel efter merge. Notér hvilke kontroller der faktisk er kørt.

Denne regel erstatter tidligere standard-testpakker i projektplanerne.
Ved G-arbejde er G.md/G_IMPLEMENTATION_HANDOFF.md/G_JANUARY.md indgangen;
F4-H2 læses kun ved konkret grænsefladegenbrug. Skriv altid F4-H2 for F-fasen
og WM-H2 for vandmodellens støjhypotese; de er ikke G-faser. En chat skal
holde sit erklærede spor, selv om begge spor findes i samme repository/projekt.

**Nyt selvstændigt spor, Johan 5. oktober: [G-planen](root/G.md).**
Deterministisk BM1-kerne, afgrænset parameterfit, matched WM4-prognoser og
et efterfølgende billigt stokastisk usikkerhedslag.
Læs G-planen ved dette arbejde; `G-plan/G0`–`G-plan/G8` er ikke `F4-G2a/b/c`.
G0–G3 er første deterministiske leverance. G6 er en obligatorisk senere
stokastisk pilot med FP/mixture og EM; G7 omfatter en deterministisk
primærreference og obligatorisk stokastisk single-shooting-prisstyring;
G8 er rekursiv tilstands-/parameterestimation.
Bachelorens projekt 111866 kræver disse sammenhænge. Genbrug F2/F3/F4-kode
og evidens; G6a's varianslag og G2's batch-LS er ikke erstatninger.
Se kravmatrix, WM-kodereview og små forsøgsbudgetter i G-planens afsnit 0/8.
Grundarbejdet fra G3 ligger i [Sol-overdragelsen](root/G_IMPLEMENTATION_HANDOFF.md):
P1/P2 er nu implementeret: `deterministic --from-fit FITDIR` genbruger et gemt
fit eller kontrolleret legacy-format uden ny optimering og gemmer forecast før
optional scoring. Nye fits skriver `fit_snapshot.json`. 20 G-tests og tre
relevante layout/importkontroller består (23 i alt, ca. 6.7 s). G3 er siden
rapporteret complete lokalt; råreview mangler. G4-adapter/LS er nu implementeret
med ni korte tests; næste trin er lokalt G4-udviklingsvalg, derefter G6b→G8→G7.
G's legacy-aggregering bruger tilgængelige målere per time; fast population og
punkt/interval/enheder skal afklares før historisk validering. Ommærk ikke
januarfittet til en ny datakontrakt. Syntetisk adapterarbejde kan begynde nu.
G som helhed er ikke rent deterministisk. F4-status og dets
obligatoriske H-f/g-krav bevares. G0–G3 er implementeret og syntetisk kontrolleret; Johan har rapporteret historisk januarfit som complete på 07fad9f;
G3 er siden rapporteret complete, mens råreview fortsat mangler.
Johan har bestilt hele januar 2022 som fælles BM/WM-fitmåned; næste 168 timer
forudsiges med faste parametre. Læs [januarvejledningen](root/G_JANUARY.md).
Ny januarvisning: `--fit-only --initial-state wm-neutral` bruger de virkelige
DK1-priser og kræver ingen februardata. Standard-x0 er fast (1-INIT_F)/2=0.5
fra WM's neutrale INIT_F=0; kun C,k,Xi fittes. `--initial-state fit` bevarer
den tidligere x0-estimation. Fit.png/ugesnit, CSV og WM-metrics viser in-sample
januarfit, ikke forecast-accept. Ingen historisk kørsel er udført i chatten.
G-januar håndterer nu manglende demand/temperatur med den eksisterende import-
interpolation afgrænset til TRAIN. Rå demand beholdes som NaN; baseline/BM-fit
og fælles BM/WM-fejlmål bruger kun målte timer. WM-tilstandsrekonstruktion bruger
udfyldt demand; QP-overgange over manglende endpoints maskeres. G kræver stadig
mindst 168 målte demand-timer i TRAIN og komplette importerede priser. 10 tests består;
én syntetisk januar med 27 demandhuller og 3 temperaturhuller gennemført
på ca. 19 s (717 målte timer for begge modeller). Historisk fit er nu rapporteret gennemført lokalt; råreview mangler.
G3 er rapporteret complete på main 607dcd7, run g_january_forecast_20261005_204059:
168/168 scored timer. MSE BM=0.06951591504, O=0.06934621735, WM=0.07099789830;
ingen baselinegevinst på den eksponerede udviklingsuge. Råreview mangler.
G4's lille to-stykkers g-adapter/LS er implementeret og syntetisk kontrolleret;
historisk pilot/udviklingsvalg mangler. Brug `deterministic_response --reference-fit`
på original TRAIN-eksport, ingen G2/WM/baseline-refit. K er fast; samme x0-policy.
Lineær G3-kernel er bevaret af hensyn til arkivkompatibilitet. G4 har ni korte
tests i `tests.test_bm1_deterministic_response`; vælg kun de berørte metoder ved G4-ændringer.
Maks. to strafstyrker inkl. nul, 3.000 residualkald/600 s samlet inklusive
replay/forfining. TRAIN vælger ikke strafstyrke. Derefter G6b→G8→G7;
G5's brede benchmark og en stor baselinegevinst må ikke blokere kernekravene.
Nye G-adaptere får egne korte kontroller; F4/H-status og tidligere gates bevares.
For G0–G4 vælges kun den mindste relevante kontrol efter reglen ovenfor.
Ingen standardpakke, historisk genfit eller månedssmoke ved almindelig vedligeholdelse.
Præcisering kl. 12:56: genbrug WM4's `statistical_output`-baseline, samme
fit/predict-protokol, metrics, performanceanalyser og rapportlayout dér, hvor
muligt. Første benchmark følger WM's én-fituges/næste-uges-protokol. Forskelle
skal begrundes; se den bindende kompatibilitetsmatrix i G-planens afsnit 1a.
G-overdragelsen reserverer et senere matematisk review af G's posterior→
FP/mixture-kontrol. F4-H2's Astra XHigh-review tilhører F-sporet og står i
`root/H2_INTERVAL_ASTRA_HANDOFF.md`; det er ikke en G-opgave eller generel
G-gate. Historisk intervalbaseret G-estimation skal dog have sin egen korrekte
observationsoperator; den syntetiske punktobservationsdemo dækker ikke dette.

**H-recovery, 5. oktober:** H1a-v2 er reproduceret på `90211e4` og råarkiveret.
Alle seks betingede H1b-fits er nu færdige på `e7d9089`; to responsive
fire-stykkers fits nåede kandidatloftet. De valgte kurvers 12 units / 48 replays
består egen numerik på `e398a5a`, og 2.304 Bayes-opdateringer er rå-reviewet.
PR #64 supplerer PR #62 med checkpointvalidering og genfundne I/O-/TRAIN-fixes.
Dette er udviklingsevidens, ikke modelvalg, joint recovery eller H2/H3-accept.
H2-metadataaudit har nu verificeret alle 13 MED-ID'er som huse med elvarme
og alle 140 UDEN-ID'er som huse uden elvarme i samme workbook som Johans
header-rapport. Se `root/H2_DATA_CONTRACT.md`; nul observationsværdier er læst.
Johan har senere deklareret timeforbrug, foreløbigt kWh, og hour-start
(03 betyder [03:00,04:00)); Europe/Copenhagen er en arbejdsantagelse.
Separat joint state/akkumulator-likelihood og timeenergiadapter er implementeret
med afgrænset shared f/g-replay. Se `root/H2_HOURLY_INTERVALS.md`.
De første grove BM-grids fejler udviklingens numeriktolerancer trods bevaret
masse; anbefal ikke fuldt historisk fit på disse settings. Ingen faktiske
måleværdier er læst eller historisk/H2/H3-accept opnået. Dokumenteret blindt
split, populationvalg samt pris/skala/baseline/fysiske parametre mangler stadig.
Særskilt H2-characteristic/CN-udviklingskontrol er nu gennemført på samme
syntetiske døgn: log-amplituder løser de små tails; tre positive CN-rum-/tidschecks
består de eksisterende tolerancer. Se `root/H2_INTERVAL_CHARACTERISTIC_CONTROL.md`.
Kun dette faste lineære case er kontrolleret; historisk fitter/reference er uændret.
Ny intervalkontrol: fire-stykkers f/g med sigma_y=0.16 består fem tids-/rum-/
frekvens-/periodechecks; to-stykkers f/g med sigma_y=0.04 fejler alle fem på
første times betingede Fourier-tætheder. Log-amplituder løser ikke cancellation
i selve inversionen; eksakte tre-state-kontroller bekræfter dette uden FV-fejl.
Genbrug de gemte positive og negative råresultater; ingen ny stor panelkørsel.
Dette Astra XHigh-review er nu udført; se `root/H2_TERMINAL_TILT_REVIEW.md`.
Terminalvis eksponentiel tilting består de eksakte og positive springreferencer
og gennemfører det smalle cases to døgn-replays. Tre fokuserede nye tests består.
Hele døgnet fejler fortsat den uændrede scoregate: .08754758 > .05;
slut-mean/L1 består. Runtime 225.70/630.87 s; faktisk log-state-pause/resume
er kontrolleret. Dette er kun en pilot, med heuristisk roundoffguard og et
analytisk modeksempel på universel tilt-dækning. Sols krydsreplay er nu færdigt
på 433.43 s: tids-scoreforskel .00216009 består; rum-scoreforskel .08538749
fejler .05. Mean/L1 består begge; gamle døgn er genbrugt uden replay. Én ny
fokuseret genbrugstest består; den uændrede terminal-kernel og 35 relevante
BM-kildehashes er kontrolleret uden at koble uvedkommende G/WM-kode til F.
Læs `root/H2_TERMINAL_TILT_CROSS_CONTROL.md`. Den ene finere reference er nu
færdig: 2561 celler, h=1/49152, samme fixture og genbrugt 1921-cellers døgn.
Scoreforskel .03055462, slut-mean .00001396 og masse-L1 .00117516 består
de uændrede .05/.002/.05-grænser. Runtime 1503.30 s (25.1 min); 15 gemte
timer er genbrugt efter budgetpause med identisk log-state-fingerprint.
Dette er fælles tids-/rumforfining, ingen kontinuumsfejlgaranti eller historisk
fit-readiness. Se `root/H2_TERMINAL_TILT_FINE_REFERENCE.md` for rådata og scope.
Næste er én parret runtimekontrol af den gemte 1921-cellers time med svagest
conditional guard, 48 frekvenser/periode 1.5 versus 64/2; behold alle guards.
Ingen ny døgn-/gridpanelkørsel eller løsere gate. Sol er tilstrækkelig til
denne deklarerede repræsentations-/runtimekontrol. Det kontrollerede
1921-cellers døgn tager stadig 10.5 min; praktisk runtime-/adapterløsning,
tidsvariation og frosne historiske data-/split-/baselinevalg mangler.
Historisk backend og fælles kerner er uændrede.
Eksakt operatorgenbrug er testet; en parret én-time-median viser 1.23x, ikke et
historisk fit-speedup. Faktisk tidsvariation og historisk log-state-checkpointadapter er
stadig åbne. H2/H3-readiness forbliver falsk; historisk fitter er uændret.
Recovery, råartefakter og næste trin: `root/F4_H_CHECKPOINT_AUDIT_STATE.md`.

Ved BM/F4-arbejde: læs først `root/F4.md`, især **Aktiv prioritering — tidsreduktion,
3. oktober 2026**. Det er den autoritative fælles status og to-do-liste; ældre
chatplaner og historiske "næste opgave"-afsnit kan være overhalet.

- Aftalt rækkefølge, Johan 3. oktober kl. 23:08: **F4 færdig → grundig
  prognosediagnose og eventuelle begrundede forbedringer → F5 → F6**. Se
  **Opfølgning efter F4 — demand-prognose og baseline** i `root/F4.md`.
  Nye model-/baselineændringer for de oversete demand-hop starter først efter
  F4; mellemtrinnet er ikke en ny F4-gate. Det tidligere samlede tidsbudget
  skal revurderes med dette mellemtrin.
- Brugerens regel, 3. oktober 2026: brug **branches i eksisterende checkout**.
  Opret ikke flere worktrees, medmindre det er strengt nødvendigt; en ny fase,
  preview eller parallel chat er ikke i sig selv en grund. Bevar source under
  aktive låste kørsler. Før et eksisterende worktree fjernes: bevar branchen,
  egne ændringer og alle nødvendige ignorerede rådata/logs; kontrollér at ingen
  kørsel bruger mappen. Reglen gælder også WM-arbejde.
- I-baselinens origin/forecast/F3-integration er implementeret på det afsluttede
  F-spor; pooled H-f/g-fitprototype og gammel-record budgetpilot er implementeret.
  H1-cases/numerik, H1b-fit og H's data-/splitkontrakt er implementeret.
  H1a/H1b-udviklingsrådata er sikret; næste arbejde er faktiske metadata,
  eksponeringsaudit og H2/H3-design før historisk modelvalg.
  Obligatorisk H-f/g
  skal gennemføres før den endelige samlede I/J-leverance. G2b-v2 er valgfrit
  nat-sidearbejde; G2c er kun en mulig opfølgning efter G2b-accept. Ingen af dem
  må blokere H/I/J eller markeres gennemført uden komplette kontroller.
- G2a-v2 er komplet, rå-reviewet og numerisk bestået på ren `01c7c153`:
  8 units, 42.312 replays, 14 kontroller; nul nye FV-replays i råreviewet.
  Bevar original source og arkiv; løs ikke resume-kontrollen. Det frigiver
  hverken G2b/G2c eller recovery/forecast. Næste aktive arbejde er H1/H's datakontrakt.
- Beslutning fra Johan, 3. oktober kl. 22:26: natkørsler budgetteres til cirka
  otte timer per invocation, med hyppige atomiske blok-checkpoints og genoptagelse.
  Brug den særskilt registrerede `joint-bank-rectangular-robustness` (G2b-v2),
  som råvaliderer alle otte beståede G2a-v2-units før nyt arbejde. Tidsstop er
  en pause, ikke accept; bevarede hypoteser genkøres ikke. Gamle kvadratiske
  G2b/G2c-prerequisites løsnes ikke. Det rektangulære G2c-v2-design er nu
  registreret, men kræver komplet råvalideret G2b-accept. Fordelt PC/chat-kørsel
  bruger én fælles plan og én ejer per shard; se BM_GUIDE. Importeret G2b-arbejde
  kræver identisk numerisk kilde og bevarer originalfiler/source.
  Fastlås nat-source i en branch i samme checkout; skift/pull aldrig
  under en aktiv kørsel. H1/H2/H3 og endelig I/J er fortsat hovedsporet.
- Metode og forsøgsdefinitioner: `root/f4/notes.tex`; kommandoer: `root/BM_GUIDE.md`.
  Opdatér fælles status i `root/F4.md` sammen med en leverance.
- Genbrug F1–F3-interfaces og gennemført evidens. Bevar grids, seeds, observationers
  betydning og numeriske/statistiske gates ved rene performanceændringer.
- Kontrollér nyeste `main` før publicering og bevar andres samtidige ændringer,
  herunder WM-arbejde. Følg brugerens konkrete instruktion om branch/publicering.
- Vælg konkret berørte regressioner; fuld discovery er kun et begrundet tilvalg.
  Byg kun F4-noter ved relevante noteændringer. Lange forskningsstudier er ikke CI-tests.
- Nye lange kørsler skal vise et løbende groft estimat for både resterende og
  samlet tid ved eksisterende statusopdateringer. Genbrug
  `src/exe/run_model/progress.py`; medregn forskellig opgavepris, hvor den er
  kendt, og adskil nyt arbejde fra gemt arbejde ved resume. Ingen ekstra replays
  eller hyppigere checkpoint-skrivning for at måle ETA.
- Følg også F4-planens afsnit **Yderligere effektivisering**: fælles I/F5/F6-
  interfaces, tidlig H-f/g-pilot og forberedelse under beregninger. Brug
  `run_fv_filter(..., keep_history=False)` til rene likelihood-kandidater og
  `evaluate_targets` til flere kvadratiske A-mål ved uændret U/process. Fuld
  historik/validering bevares for valgte modeller; de nye helpers afslutter
  hverken H-f/g, F5 eller F6. Ingen ny spekulativ solveromskrivning på kritisk vej.
- Genbrug `BM1ForecastOrigin` i `src/control/bm_1_handoff.py` til I/F5/F6:
  eksplicit cutoff/tidsenhed, frosne parametre/kurver og konservativ posterior.
  Rho-MAP bruger nodens egen conditional posterior. Et gennemsnitligt parameter-
  plug-in kræver nyt replay eller erklæret approksimation. `i-baseline` er kun en
  gammel-record integrationsdemo; ingen forecast- eller H-f/g-accept.
  `BM1PiecewiseResponse` og `src/method/bm_1_response.py` giver kurve-API og
  pooled replay-fit. Gammel-record prototype er ikke H's endelige fit/accept.
  Bevar deklarerede kandidatbudgetter og negative budgetstops; ingen automatisk
  ekstra fitrunde eller modelvinder efter TRAIN-score. H1's udviklingscases og
  valgte-kurve-numerik er gennemført; faktisk dataaudit, hele den historiske
  træning og blind H3 mangler stadig.

Disse kurveregler gælder BM og ændrer ikke WM-modellernes egne fitkontrakter.

## Beslutning fra Johan, 3. oktober 2026

- Estimation og validering af fleksible `f(x)`/`g(u)` er en **obligatorisk H-f/g-leverance**,
  endnu ikke afsluttet; udviklingsfit er gennemført, mens historisk fit og
  videnskabelig validering mangler. Den må ikke igen klassificeres som en valgfri udvidelse.
- Ét fælles kurvepar skal bestemmes samlet på tværs af træningsdage for en erklæret
  population/gruppe og holdes fast ved validering/forecast. Ingen f/g-fit per dag.
  Daglig baseline og state-estimation kan fortsat variere.
- Undersøg kontinuerte, ikke-voksende stykvise kurver med knuder
  `0, 0.25, 0.5, 0.75, 1`, faste endpoints `1,-1` og fælles estimerede knudeværdier.
  Intervaller vælges efter state/pris, ikke tidspunkt. En enklere pilot og lineær
  reference bevares; alle delstykker indgår i samme likelihood-fit.
- H-f/g skal gennemføres før endelig H→I-handoff og J. Split, normalisering,
  observationskontrakt, numerik og modelvalg dokumenteres; slut-holdout holdes blind.
  Nye kurver kræver egne kontroller. Det faste lineære A--G-design og gemt evidens bevares.
- Budgettér kurveestimationen med en lille pilot og afgrænset optimizer; opret ikke
  et stort Cartesian-grid over kurve- og fysiske parametre. Følg H-f/g-planen.
- Ved ændret status skal `root/F4.md`, nødvendig matematik og vejledning stemme overens.
  Beskriv ikke en planlagt eller blot implementeret estimator som valideret.

Dette er den fælles repoindgang for projektchats. En allerede åben chats indlæste
kontekst opdateres ikke automatisk; ved genoptagelse skal den læse den aktuelle plan.

