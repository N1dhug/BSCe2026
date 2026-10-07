# G7: verificeret receipt-genbrug og journalguards, 7. oktober 2026

## Resultat og afgrænsning

Næste G-trin efter seed-rettelsen er udført. Et separat read-only indeks
verificerer alle 24 færdige receipts fra seed-kørslen, herunder de ti egne
FV-resultater. Et G-lokalt adapterinterface til nye NodeJobStore-scopes
kontrollerer journalen før jobs, write-ahead-reservationen før compute og
den komplette journal efter completion-write. Et brud stopper scope med
en separat recovery_blocked.json-markør; der foretages ingen gættet reparation.

Originaljournalen og det eksisterende råarkiv bevares byte-identiske. De er
stadig inkonsistente. De to manglende coarse-FV-poster og den running fine-FV-
post er ikke omskrevet til complete. Løsningen tillader eksplicit verificeret
genbrug i en ny registreret opgave; den tillader ikke genoptagelse af den
oprindelige aktive store eller implicit retry af beregninger.

Ingen EM/M254/FV-forward, optimizer, ny posteriorberegning eller gammel
numerisk test er kørt. Native/shared source, modelmatematik, basis, priser,
tolerancer og F/WM er uændrede. BM færdiggøres før direkte WM-sammenligning.

## Hvad undersøgelsen faktisk kan forklare

Den lokale originaljournal og campaign/ledger.json i det komplette rå-ZIP
har begge SHA256 b90e8b90c87d6224e039b8ffc7076c5e94043289d163b16bbee392a2be47b2c1.
Journalen har 22 entries: 21 complete og én running. Der er 24 gyldige receipts
og tilsvarende worker-summary-stages. Workerens tekstlog slutter efter fine-FV-
node3; den komplette summary og senere receipts findes alligevel.

Den frosne native writer reserverer ved atomisk journalwrite før compute,
skriver receipt før completion og bruger OS-lock. Dens orphan-check gælder
det efterspurgte job; den laver ikke en samlet audit før et andet job, og
den læser ikke hvert write tilbage for at kontrollere den forventede journal.
Det forklarer, hvilke manglende kontroller adapteren lukker. Det viser ikke,
hvilken hændelse der tabte de konkrete poster/loglinjer. Den oprindelige
udløser er ikke reproduceret og forbliver ukendt. En atomisk filwrite alene
er ikke en garanti for, at to filer eller en hel jobhistorik er konsistente.

## Indført kode og brug

| Fil/interface | Hvad det gør |
|---|---|
| [receipt_recovery.py](g7/receipt_recovery.py): VerifiedReceiptIndex | Verificerer et eksternt hash-låst manifest, originale metadata og alle receipt-/payloadhashes før brug; payload kræver præcis original jobidentitet |
| receipt_recovery.py: audit_store | Finder orphan receipts, manglende complete-filer, forkert status/attempt-historik og ændret budgetseed; full=True hasher også samtlige payloads |
| receipt_recovery.py: guarded_store_class | G-lokal subclass af en sourcehash-låst native NodeJobStore; bevarer dens lock, reservationer og eksplicitte retry-regler |
| [receipt_adoption_manifest_20261007.json](g7/receipt_adoption_manifest_20261007.json) | 24 jobidentiteter med receipt-/payloadhashes, de oprindelige metadatahashes og bevaret charge |
| [check_receipt_recovery.py](g7/check_receipt_recovery.py) | Én fokuseret stdlibkontrol af faktiske råreceipts og injicerede journal-/crash-fejl med små stubpayloads |

Ny kode ligger kun under root/g7. Den gamle one-invocation seed-runner ændres
ikke og er ikke kørt igen. Adapteren er en eksplicit integration til næste
G-launcher, ikke en global udskiftning af NodeJobStore. Fremtidig registrering
skal inkludere adapterens sourcehash og den native store-hash, og bruge:

```python
from src.method.bm_1_g7_jobs import NodeJobStore
from receipt_recovery import VerifiedReceiptIndex, guarded_store_class

old = VerifiedReceiptIndex(raw_root, manifest_path,
                          manifest_sha256=registered_manifest_sha256)
payload = old.payload(exact_original_job)  # læser; propagérer ikke

GuardedStore = guarded_store_class(
    NodeJobStore, native_source_path=native_path,
    native_source_sha256=registered_native_sha256)
store = GuardedStore(new_scope_root, prior_debit=current_full_debit,
                     ceiling=1800, max_new_seconds=registered_worker_allowance)
with store.lock():
    result = store.node(new_exact_job, reservation, compute)
```

Et nyt job, der genbruger data, skal registrere manifest-/sourceidentiteten
og sin nye policy/origin/kontrakt korrekt. Payload-indekset tager ingen
genvej fra et ændret job til et lignende gammelt job. Priser ændret fra
[.5,.5] til [.6,.5] afvises i kontrollen. De egne M254-resultater genbruges
fortsat gennem det eksisterende read_owned/restore-interface, under deres
bevarede originale kontrakter. Dette indeks vedrører seed-kørslens 24 receipts.

## Matematik og argument for genbrug

For et frosset numerisk job er resultatet en funktion af hele kontrakten K:
\[
 R=F(K),\quad
 K=(\text{origin, parameterbank, policy, horizon, grid, modelsource,
 costpartition, versions, eventuelle tilfældige inputs}).
\]
Et gemt R kan genbruges, når identiteten K og råresultatet er bevaret.
Det giver samme numeriske problem; det gør ikke R til en eksakt kontinuumsløsning.
For EM skal de registrerede streamidentiteter og inputs også bevares. For
M254/FV har rettelsen af EM-/observationsseeds ikke ændret forwardkontrakten.

Job- og payloadhashes er integritetskontroller under den sædvanlige antagelse
om SHA256-kollisionsresistens. De beviser ikke, at model, diskretisering eller
cost er matematisk korrekt. Den numeriske kontrol og dens samplingantagelser
forbliver dem i [seed-reviewet](G7_SEED_REPAIR_REVIEW.md). Den nye kontrol
gentager ikke disse beregninger; den verifierer identiteten af deres evidens.

Journalstatus og numerisk indhold er forskellige udsagn: en korrekt receipt
kan eksistere uden en komplet journalpost, fordi write-rækkefølgen er
reservation → compute → receipt → completion. En receipt kan derfor genbruges
under eksplicit verificeret adoption, mens den oprindelige journal stadig
afvises som aktiv writer. En running reservation med receipt i en ellers
konsistent store kan afsluttes af den native recovery-regel med fuld charge.

## Budgetargument og bevaret historik

Lad c_i være den konservative receipt-charge, C_wall den tidligere målte
worker-wall og C_old den allerede bevarede kampagnedebit. For hver receipt
blev mindst stage-wall og tidligere charge medregnet; manglende/running
FV-poster fik mindst deres fulde .5s reservation. Her gælder
\[
 \sum_i c_i=19.336431619s < C_{wall}=20.996288735s.
\]
Den tidligere workercharge var max(C_wall, oprindelig ledgercharge), altså
C_wall. Alle afstemte receipts er derfor dækket; adoption kræver ingen
ny charge for deres gamle beregninger. Den nye verifikation/review skal
derimod medregnes. Denne præregistrerede scopecharge er konservativt 30s:
20s til forberedelse, fokuseret verifikation og metadataeksport, plus 10s
review/dokumentation. Det er en budgetcharge, ikke påstået målt modeltid.
\[
 C_{new}=1401.082740428s+30s=1431.082740428s,\qquad
 C_{rest}=1800s-C_{new}=368.917259572s.
\]
Alle gamle fejl/reviews/eksporter er bevaret. Ingen refund, reset eller
dobbelte charges for gamle forwards. Nye jobs skal stadig reserveres mod
den fulde aktuelle debit, og parenten skal medregne hele process-wall,
eksport og review. Writer-guards erstatter ikke en parent-timeout eller en
fuld opgave-/døgnbudgetbeslutning.

## Faktisk kontrol og begrænsninger

Source/registrering blev checkpointet før kontrol i commit
bfff708ac8f5da641d9a185b89f2f89b9b573b60.
Én fokuseret kontrol består med 12 cases og 24 faktiske hashverificerede
receipts. Checkens interne måling er 4.571205494s. Den bruger seks små
stub-computes og nul faktiske model-forwards eller numpy/scipy-imports.
Writer-testen udfører de eksakte native NodeJobStore-/lock-/atomic-write-
definitioner via AST med kun scalar-isfinite erstattet af stdlib math.

Den kontrollerer genbrug uden compute/recharge, receipt før completion-
crash med fuld reservation, orphan/manglende receipt før nyt arbejde,
payloadtampering, eksplicit retry med bevaret afbrudt charge, injiceret
tab af reservation/completion-write og afvisning af stale journaloverwrite.
Originaljournalen er uændret efter kontrol. Dette er afgrænset filesystem-
kontrol, ikke en fuld native solver-/launcher-, power-loss- eller Windows-
integrationstest. De eksisterende OS-lockregler er bevaret; konkurrerende
processer og årsagen til den historiske afvigelse er ikke nytestet her.
Filer og resultat: [check](g7/receipt_recovery_check_20261007.json),
[registrering](g7/receipt_recovery_registration_20261007.json),
[budget](g7/receipt_recovery_budget_20261007.json).

source_journal_integrity_passed=false; explicit_read_only_adoption_verified=true;
original_store_resume_permitted=false. Scientific/full-day-accept er false.
Guard skal vælges eksplicit i den næste sourcefrosne launcher. Stopmarkøren
skal reviewes særskilt efter et brud; den må ikke fjernes som automatisk retry.

## Næste G-trin

Receipt-genbrug og writer-guard-interface er færdige og afgrænset kontrolleret.
Planlæg nu en konkret prisoptimerings-/selvstændig costkvadraturopgave under
368.917s restbudget med dependencies, guards, stop/eksportreserve og acceptkrav
før numerisk start. Genbrug de verificerede neutrale referenceforwards.
Nye priser/origins kræver nye relevante forwards. Adgang til gamle receipts
er ikke optimizer- eller 24h-admission. Historiske intervalobservationer og
slutbenchmark mangler stadig; G7/G som helhed er åbent.
