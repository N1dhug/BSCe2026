## Nyeste status: recovery er afsluttet, 7. oktober

Det tidligere110s-forslag nedenfor er nu godkendt og gennemført under
separat præregistreringf208bbc3. Kun node3/4 er beregnet; fuld quadbank og
alle tre forward-pargates består. Tabte projectionauditmaxima for node0/1/2
begrænser fortsat numerical/scientific/full-day-accept. Ekstra charge42.195764s;
G1841.337272/1910s, rest68.662728s. De ældre stop/charges bevares som historik.
Læs den fulde [matematik, recovery og råreview](G7_COMPLETE_QUAD_RECOVERY_REVIEW.md).

# G7: launcher og særskilt præregistrering, 7. oktober 2026

## Faktisk ændring og mergekontrol

PR114 er merged i 66991a099aac2961083d7efa452be616a2f052d2, oven på
f42b229626530d3852875be03fc6610b073fe4ce. Alle11 ændrede filblobs matcher
det reviewede PR-head9571242. Ingen src-, tests- eller workflowfiler ændres
af mergen; de nyere WM4-ændringer på main bevares. PR-head syntax bestod;
selected-tests/notes var skipped. Tidligere modeltests/forwards er ikke genkørt.

Den gamle transient kørselsmappe var væk. Det eksisterende originale
G7_M254_node_runtime_20261007_data.zip er gendannet og matcher den registrerede
SHA256 ad730a148e770fffeeccdfcfff13af30fc33510d0bbcba9b988c15767af1e80d.
Alle41 numeriske filer er gendannet fra872e777 og byteverificeret mod den
oprindelige source_manifest. Kun tomme __init__.py-filer til packageimport
er tilføjet lokalt; ingen numerisk kode ændres. NumPy2.3.5/SciPy1.17.0 matcher
de gemte fulde kontrakter. Gamle data/journaler forbliver read-only.

Den nye parent/worker bruger eksplicit GuardedNodeJobStore, binder hver fuld
kontrakt under den allerede ejede native OS-lock og fjerner derfor ikke
writer-guards. Native CheckpointedBankControl.describe/check_sources genbruges
uden at kalde dens initializer under samme ikke-reentrante lock. Setupmetadata
kan ikke erstatte live evaluatorer. Der er én invocation og ingen automatisk
retry/resume; en fsync'et started-markør skrives før workerstart.

De 41 gamle numerical hashes holdes adskilt fra nye launcher/guard/restore-
hashes. Ny registrering fryser begge, de gamle inputs, versionsnumre,
designbytes, budget, candidates og tie-rækkefølge. FV genbruges kun gennem
det eksternt pinned24-receipt-indeks og den fulde oprindelige FV-kontrakt.
Den afvigende gamle journal må stadig ikke genoptages som aktiv writer.

## Matematik og hvorfor implementeringen svarer til den

På[12,13] og[13,14] bruges u(delta)=(.5+delta,.5-delta), delta i{0,-.2,+.2}.
Hver node j starter fra sine egne gemte hour12-masser og M254-koefficienter,
med faste cutoff-vægte w_j. Ingen fremtidig observation indgår i valget.

Målet er
\[
J(u)=\sum_jw_j\int_{12}^{14}E_j[(D_t(u)-A_t)^2]dt.
\]
Sæt mu_j=E_j[D], v_j=Var_j(D), mu=sum_jw_jmu_j. Med sum_jw_j=1 er
\[
\sum_jw_jE_j[(D-A)^2]
=\sum_jw_j(v_j+(mu_j-A)^2)
=(mu-A)^2+\sum_jw_jv_j+\sum_jw_j(mu_j-mu)^2.
\]
Krydsleddet er nul, fordi sum_jw_j(mu_j-mu)=0. Derfor er bias, within-
parameter-variance og between-parameter-variance en eksakt dekomposition af
samme mål, ikke tre vilkårlige strafled. combine_nodes kontrollerer både
totalvarians og J=sum_jw_jJ_j. Vi rapporterer alle led fra samme forward.
Middelkurvens kvadratfejl er ikke generelt identisk med en separat
deterministisk trajectory ved middelparametre; ikke-lineær dynamik bryder
typisk denne identifikation. Ingen anden objective optimeres særskilt her.

Valg foretages kun på tre komplette coarse-banker: argmin(J_0,J_1,J_2),
med eksakt tie-rækkefølge0,1,2. Alle fem noder kræves; en ufuldstændig bank
ændrer ikke kandidatlisten eller vægtene. Fine/FV/quad kontrollerer det
allerede fastlagte valg og bruges ikke til re-selektion. Denne beslutning
finder et minimum blandt tre prøvede planer på det grove grid. Ingen
konveksitet i u, kontinuert/global optimalitet eller historisk kausal effekt
følger af kvadratet i demand eller af numerisk agreement.

Midpointmålet er J_Delta=sum_l Delta_t_l r(t_l), hvor r=E[(D-A)^2].
Delta halveres1/96h→1/192h og giver192→384 nye midtpunkter. Maksimalt
solvertrin holdes1/192h, men ændrede outputpunkter opdeler også M254's
transport/projektionsforløb. Det er derfor en cost/output-følsomhedskontrol,
ikke et isoleret kvadraturfejlcertifikat eller Richardsonestimat.

Kun samme originale-partition-fine-J sammenlignes med neutral. Forbedring
kræver G=J_neutral,fine-J_valgt,fine>e_neutral+e_valgt samt forwardgates,
hvor e er den observerede coarse/fine-J-forskel. Det er screening mod
observeret numerisk følsomhed, ikke et rigorøst kontinuumsfejlbound.
Den nye partition må ikke bruges som kandidat-J mod en gammel neutralpartition.

Vi rapporterer også sum_l Delta_t_l(mu_l-B_l) og ændret endpoint-middeltilstand.
Lige middelpris er ikke bevaret samlet forbrug. Der indføres ingen ny
energiligning, terminal-/smoothingstraf eller fysisk enhedspåstand.
Den tidligere deterministiske kontrol med/uden energikrav bevares særskilt.

## Kontrol, budget og startstatus

Én fokuseret stdlibkontrol består27 checks på0.042779s: fulde finite
candidatecosts/ties, budgetadmission, ændrede designbytes, one-invocation-
stop før spawn, POSIX-jobtimer og de fire faktiske nested setup-funktioner
med stand-ins. Stand-in-init fejler ved lock-reentry; setup bruger den
native describe/bind-vej uden reentry. Der udføres nul numerical imports
eller forwards. Dette er ikke en fuld solver-/Windows-/concurrency-/power-
loss-integrationstest. Eksisterende guardcases og numeriske tests gentages ikke.

Gendannelse, forberedelse og launchercheck charges konservativt35s med
bevarede gamle charges: G1441.082740428→1476.082740428s; rest323.917259572s.
Det foreslåede320s scope er re-admitteret og endnu ikke charged. Ved dets
fulde loft ville3.917259572s være tilbage. Ingen ny allowance eller reset.
Worker270s, stop/eksport30s, råreview20s og alle jobcaps fra PR114 fastholdes.
Jobreservationer er stoprammer, ikke kendte runtimebounds for de nye priser.

Parent stopper worker ved270s og eksporterer også partials. Faktisk charge
er max(workerwall, ledgerincrement)+målt eksport+20s råreview; overshoot
bevares og blokerer senere jobs. Ved parenttab uden måling kræves eksplicit
reconciliation med fuld workerreserve og ikke-målte reserver bevaret.

Kilder, templates og særskilt registrering skal være committed før forsøget.
launch_ready/pre_run_registration=true; numerical_run/actual_optimizer/
scientific/full-day=false indtil den faktiske invocation. Denne registrering
ændrer ikke det oprindelige PR114-designsnapshot eller de historiske receipts.
Kandidat-EM/feedback, historiske intervalobservationer og24h-admission er
stadig uden for scope. G/F holdes adskilt; direkte BM/WM-sammenligning venter.

## Kørsel og output

Kopiér filerne i root/g7/bounded_price_20261007 til en ny, tom outputscope.
Den samme mappe indeholder launch_bounded_price.py og de låste helperbytes.
Giv --previous den gendannede originale råmappe med frozen_source, og --seed
den read-only seed-repair-råmappe. Kør først efter denne registreringscommit:
python -B launch_bounded_price.py --output OUTPUT --previous OLD --seed SEED.
Parent og worker bruger samme registrering; PYTHONPATH peger på OLD/frozen_source,
og BLAS/OpenMP/MKL/NumExpr sættes til én thread. Ingen lokal kørsel kræves af Johan.

Nodereceipts gemmer fulde arrays; kontrakter og selection gemmes separat.
worker_summary/progress viser fuldførte og partiale jobs. invocation indeholder
faktisk debit og eksportmåling; archive_index fastholder ZIP-hash og den
efterfølgende final-invocation-hash uden at omskrive ZIP'ens tidligere bytes.

## Første stop og særskilt, reviewet tillæg

Registreringdf877b91e772a5a4a78d29f6c5b6c83b283a4863 blev committed før
invocation. Read-only verifikationen ramte5s-cappen. Ingen evaluator-,
candidate- eller forwardjob blev startet. Journalen er konsistent med én
failed attempt og nul completed receipts; den gamle scope kan ikke resumes.
Workerwall5.780612s, ledger5.009154s, eksport.003839s, bevaret review20s:
scope25.784450717s; G1501.867191145s, rest298.132808855s.
Returncode0 betyder her kun håndteret stop, ikke et completed forsøg.
Workerens initiale reused10/5-counters var planlagte counts; ingen completed
adoption påstås for dette stop. Tillægget tæller dem først efter succes.
Stopråarkiv cc231d3406480565103d34e51b444d558cedc787e5b04e03afd46d6f5becc4fa
er bevaret. Råreview bestod en tom receipt-journal på.000585s; ingen costs
eller numerik blev valideret.

En særskilt eksplicit reviewbeslutning opretter en ny scope; ingen restart
af den gamle. Native equations, inputs, forwardcaps, candidates, tie-rule
og sciencegates ændres ikke. To rent read-only ineffektiviteter fjernes:
M254-receipts blev JSON-parset to gange; nu bruges samme pinned parse med
uændret canonical payloadhashkontrol. For adoption autentificeres alle24
receipts mod de eksternt reviewede fulde bytehashes, hvorefter kun de fem
FV-payloads faktisk brugt i scope parses og semantisk verificeres.

Dette er hashtransitivitet: hvis den fuldt reviewede fil havde hash h og
den aktuelle fil også har h, regnes dens bytes for identiske under SHA256's
alminelige collision-resistance-antagelse. Job-/payloadchecks ved faktisk
genbrug bevares. Eager payloadvalidation er fortsat default; byte-only
prevalidation er et eksplicit sourcefrosset valg i den nye G-snapshothelper.
Den fælles gamle guardhelper og gamle receipts ændres ikke.

Én ny fokuseret stdlibkontrol(1.312102s) verificerer41 sources, originale
metadata og10 M254-job/payloadhashes, alle24 adoption-filehashes, alle fem
fulde FV-job/payloadhashes, ændrede receiptbytes-afvisning, tie-rule og
budget. Nul numerical imports/forwards. Gamle guard-/solverchecks gentages
ikke. Det er ikke fysisk eller modelnumerisk validering.

Tillægget charges konservativt6s: G1507.867191145s; rest292.132808855s.
Ny scope292s =264 worker+20 stop/eksport+8 råreview. Verifycap12s, fire
setupcaps7s; øvrige nodecaps bevares6/12/18/2s. Jobreservationer263s plus
1s deklareret worker-headroom. De er stoprammer, ikke målte worst-case
bounds. Eksport/review reserverne er mindre; overshoot charges stadig
fuldt og blokerer videre jobs. Ved fuldt scope-loft ville.132808855s blive
tilbage. Ingen tidligere charge refunderes.

Den nye registrering i root/g7/bounded_price_addendum_20261007 commits
før den nye invocation. Det tidligere320s forsøg forbliver failed/zero-forward
og historisk. Det oprindelige PR114-designsnapshot bevares uændret.

## Manglende inputs rettet og faktisk setupkontrol

Registrering80c5b16 blev committed før næste invocation. Verifikation
bestod1.482263s; første setup fejlede straks med FileNotFoundError.
Det gendannede G7-råarkiv indeholdt kontrakter og receipts, men ikke de
separate inputs/coarse.json/fine.json. read_owned validerede historiske
kontrakter, ikke eksistensen af disse launcher-inputfiler. Det blev ikke
fanget af stand-in-checket; den manglende preflight var en launcherfejl.
Denne stopinvocation har nul nye forwards, én complete verify-receipt og
én failed setup. Journalen består read-only review; ingen numerisk accept.
Scopecharge10.274316715s bevares: G1518.141507860s, rest281.858492140s.

Begge originale origin_12.json er nu hentet fra de eksisterende G8 M254
filter-/513-forfiningsarkiver. Coarse1938113bytes hash1cd343248607cc27baa61d4e52303fbfbf5c08ca7674fac0b9b3dce43649b3bc;
fine3846405bytes hash746e45c4603e724824687e535c2e7235f824e19ecbb6adcdd1c6ee0c0a7593e0.
Ingen tilstand er rekonstrueret, refittet eller regriddet. Et forsøg på
bytegendannelse fra kontraktens JSON gav ikke match og blev ikke brugt.
Parent kontrollerer nu begge hash-pinnede inputfiler før started-markør
og spawn; worker bevarer den fulde native kontraktkontrol. Fejlrapporten
gemmer også str(exc) og traceback, så den konkrete sti ikke skjules.

Én nødvendig live setup-preflight bestod3.790773s med real native imports,
men evaluate_node var eksplicit deaktiveret. Faktiske fire nested setups,
neutral receipt-reduktion, coarse/fine/FV-contractidentitet og192/384
midpoints blev kontrolleret. Manglende inputs afvises før workerstart.
Ingen modelpropagation/replay blev kørt. Dette er stærkere setupevidens
end stand-ins, men ikke Windows-, concurrency- eller power-loss-test.

Konservativ input-/setupcheckcharge8s inkluderer den afviste2.176s
bytegendannelsescheck og live preflight: G1526.141507860s; rest273.858492140s.
Ny, særskilt registrering i root/g7/bounded_price_inputs_fixed_20261007:
scope273=245 worker+20 stop/eksport+8 review. Verifycap4s baseres på faktisk
1.48s læsning; setupcaps4s på live preflight. Øvrige forwardcaps bevares.
Jobreservationer243s, worker-headroom2s. Caps er stadig stoprammer, ikke
runtimebounds; intet matematisk resultat eller tolerance lempes. Ved fuldt
loft rest.858492140s. Alle tidligere charges og failed scopes bevares;
ingen automatisk genstart af dem. Ny registrering commits før ny invocation.

## Faktisk prissøgning og afbrudt kvadraturkontrol

Inputrettelsesregistreringee3dae66bb759b855bc8f44a37216796c31c21dd blev
committed før invocation. Verifikation1.358s og alle fire rigtige setups
.522/.961/.699/.644s bestod. Ti nye coarse-forwards blev gemt; derefter
valgte det præregistrerede coarse-argmin neutral. Der er ikke sket fine
re-selektion. Neutral fine/FV genbruges under de fulde oprindelige kontrakter.

| Prisplan | Coarse J | Fejl i middelkurve | Variansbidrag |
|---|---:|---:|---:|
| [.5,.5] | .003741963573 | .002278966740 | .001462996834 |
| [.3,.7] | .017303375698 | .016255560315 | .001047815382 |
| [.7,.3] | .010655853332 | .009386734230 | .001269119102 |

Neutralrække er genbrugt og hash-pinned; tiltrækker er nye og reproduceret
fra receipts. Begge tilts sænker variansbidraget, men øger middelkurvens
trackingfejl mere, så samlet J bliver højere. Det er en forklaring gennem
den præcise kvadratudvidelse, ikke efterfølgende ændring af målet.
De samme to nye tilts er også dårligere efter middelkurvens mål på coarse.
Ingen separat optimering af middelmålet eller deterministisk gammelt mål
er gennemført. Neutral er bedst blandt tre komplette coarse-planer, ikke
bevist globalt optimal og ikke ny dokumenteret kontrolforbedring.

På den nye384-midpointpartition er node0,1,2 completed. Node3 står running
med sin fulde18s reservation; node4 er ikke startet. Tool-sessionen blev
utilgængelig; worker_summary, final invocation og parentarkiv blev ikke
skrevet. Sidste completed log var45.33s. Det er ikke en measured runtime
for hele scope. Årsagen til parent/worker-tab er ukendt; ingen numerical
exception eller kontrolleret jobtimerfejl er gemt for node3.
Ved snapshot var writer-lock fri, scopealder308.359s og originaljournal
byte-identisk før/efter eksport; ingen aktiv writer blev genoptaget.

Én read-only råreview bestod3.807205s:19 complete receipts,13 nye forwards
(10 coarse,3 quad), konsistent20-attempt-journal inklusive running reserve,
alle job/file/payloadhashes og nye node-J fra demand-momenter. Begge
komplette coarse-banker reproducerer bias/within/between/totalvarians;
partial quad reduceres ikke. Ingen modelpropagation eller numerical imports
i review. Dette er validitet af gemt evidens, ikke en bestået quadgate.
Neutral fine J=.003715517464 er tidligere evidens. Coarse/fine/FV-/quad-
parrapporten blev ikke udført i denne scope, og quad accept er false.

Hele parent-loss-reserven bevares uden refund:245s ukendt workercharge
+20s eksportreserve+8s review=273s. G1526.141507860→1799.141507860s;
rest.858492140s af1800. Det målte stabile snapshoteksport tog4.556245s og
ligger i den bevarede20s reserve. Der bruges ikke45.33s logtid som falsk
walltime. Gamle failed scopes og charges er bevaret; ingen automatisk retry.

Fuldt partial-reviewarkiv28266794bytes, SHA256
e0d7dafa0c3c48cb7efa9477e472c8f8bbc671de1c8786be9fa3a9791bae121c,
indeholder alle gemte arrays, fulde kontrakter, separate registreringer,
originale own-inputbytes, log, ledger, selection, raw review og konservativt
budget. OriginaljournalSHA2568e05dd6dc4eece02679b99177bb05e9b97aeb7a6e7bda79822d16c190b2c0291.
Det er bevaret som en ufuldstændig kørsel, ikke en færdig scientific/G7-accept.

## Konkret næste beslutning

Nuværende allowance kan ikke dække de to manglende quad-forwards og
slutreduktionen. Et ikke-admitteret, reviewbart recoveryforslag er gemt:
110s ekstra maksimum =10s forberedelse/check +100s særskilt execution
(70 worker,20 eksport,10 rawreview). Jobcaps:verify8,quad-setup4,to
missing nodes18 hver,reduce2;50s jobs+20s workerheadroom. Bevar alle gamle
charges; foreslået nyt samlet loft1910s. Dette loft er ikke taget i brug.
Kun node3/4 ville få nye forwards; alle13 complete forwards genbruges
read-only under pinned identities. Ingen ny prisoptimering, fine/FV/EM,
refit eller WM/F-opgave. Ny launcher og separat registration skal source-
fryses før eventuel kørsel, med parent-sessionen overvåget til final export.
Numerical/scientific/full-day-accept forbliver false indtil relevante
fulde gates faktisk er afsluttet. Direkte BM/WM-sammenligning venter stadig.
