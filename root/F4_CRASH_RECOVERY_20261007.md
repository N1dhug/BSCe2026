# F4-H2: gendannelse efter crash, 7. oktober 2026

## Resultat
Checkpoint 01–06 fra GitHub-commit `e79aa1005028137177a8321ac30b2988435f04ff`
er hentet, hashkontrolleret i rækkefølge og faktisk gendannet til en ny,
pauset mappe. Alle **1.076 modeljobs** er bevaret. De 536 jobs i det genfundne
lokale kørselsområde er byte-identiske med en delmængde af gendannelsen.
Ingen ekstra lokal jobfil blev fundet i det oprindelige F-run.

Alle 69 frosne numeriske/runner-kildefiler matcher manifestet og det indlejrede
sourcearkiv; den ydre controllers tre kilder matcher også. Råreviewet kontrollerer
hver jobidentitet, payload-checksum, normaliserede fulde logstates,
observationsidentitet, gemte guards, kædesammenhæng og alle 12 afsluttede
TRAIN-scorer. Det genberegner ingen modeltimer og rekonstruerer ikke
optimizerens interne tilstand. Den særskilte deterministiske COBYQA-genoptagelseskontrol
skal stadig bruges ved en senere finansieret genoptagelse.

## Præcist stopsted
- Lineær helvinduesvalidering er bevaret: 24 prior + 48 TRAIN + 24 selection
  på egne base/fine streams. Tidligere råreview: scoreforskelle
  0.0055541053 / 0.0004541615 mod grænsen 0.05.
- To-stykkers fit: **12 afsluttede objective-kald**, inklusive den fælles
  lineære start; 11 afsluttede ikke-lineære TRAIN-replays.
- Næste to-stykkers kandidat: **34/72 timer gemt** (24 prior + 10 TRAIN);
  de resterende 38 TRAIN-timer er ikke bevaret for denne kandidat.
  Nøgle: `67d94e70596f393e163ae47fa5c4d858d82ebc751e5f090b6b5921d7587308b0`.
- Fire-stykkers optimizer: **0 afsluttede kald**. Den tidligere korte
  fire-stykkers numerikprøve er bevaret, men er ikke et fit.
- Ingen gemt optimizer-resultatfil, terminal `budgeted_report.json`,
  modelvalg, frosset ugeprognose eller slutobservationsfil.
- Ingen `*.failed.json` modeljobfiler i checkpointkæden.
  Fravær af en failure-fil er ikke dokumentation for, hvad der skete efter
  sidste publicerede checkpoint.

Bedste gemte TRAIN-kandidat har penaliseret objective -10.0778622581
mod den lineære starts 9.2667177296. Dette er kun udviklingsoptimering;
kandidaten er hverken konvergeret, udvalgt eller valideret på fine/selection.
Der vælges ingen model på dette grundlag.

## Budget og hvorfor modellen ikke blev startet igen
Den genfundne autoritative campaign-ledger er byte-identisk med
`campaign_at_checkpoint.json` fra checkpoint06. Den indeholder:
- **2851.549305 sekunder** før den afbrudte invocation.
- En fortsat **åben reservation på 30.000 sekunder**.
- I alt **32851.549305 sekunder** før recovery.

Den eksisterende crashregel i `bm_1_h2_compact.reservation` og nattenoten
bevarer hele reservationen ved crash. Vi har ingen afsluttende ledger,
autoritative proces-sluttid eller terminal rapport, der kan dokumentere
en tilbageførsel. Jobtider summeret fra checkpoints udelader imports,
I/O, optimereroverhead og eventuelt arbejde efter checkpointet; de er
derfor ikke et grundlag for at refundere reservationen.

Recovery har en særskilt konservativ **60 s native-development-charge**
for transfer, checksum-gendannelse, råreview og pakning. De målte centrale
operationer tog 0.4393 s (gendannelse) og 3.1474 s (råreview). Alle 49 gamle
entries er bevaret uændret, inklusive den åbne 30.000 s-reservation.
Aktuel total: **32911.54930543392 s**; rest under 36000 s:
**3088.4506945660833 s = 51.47 minutter**.
Native-development: **803.307761054043 / 900 s**.

Den sidst finansierede resterende validering kræver allerede
**19960.31431629205 s = 5.54 timer**. Den kan ikke rummes i restbudgettet,
selv uden flere optimizer-kald. Ny søgning er derfor ikke admitted.
Den aktive ledger og den bevarede efterkopi indeholder nu også denne
højere validation-reserve; den gamle disk-ledger havde kun den tidligere
reserve, fordi controlleren opdaterede den i hukommelsen under invocation.

Der er desuden nu observeret basejob på op til 9.62828 s mod
admissionens 8.30102 s; et nyt budget skal genprises fra gemte aktuelle
jobtider. Det gamle 14.61 h-estimat eller de tidligere 7.85/7.88 h kan ikke
bruges som nyt løfte om resttid.

## Miljø
NumPy 2.3.5, SciPy 1.17.0 og pandas 2.2.3 findes i det genfundne miljø.
De frosne runtimeafhængigheder numba 0.62.1 og llvmlite 0.45.1 mangler.
Råreviewet kræver dem ikke og foretager ingen native propagation.
Numerisk genoptagelse er ikke godkendt ved at springe miljøkontrollen over.

## Næste arbejde
1. Fremskaf en eventuel senere autoritativ, lukket campaign-ledger og
   checkpointkæde fra den oprindelige invocation. Bevar alle entries,
   kildeidentiteter og rådata. Ellers kræver videre modelberegning en
   udtrykkelig ny budgetbeslutning fra Johan; intet nulstilles automatisk.
2. Pris det resterende arbejde med de nye målte maksimumrater og fuld
   valideringsreserve. Bevar 1/2/4-familierne, 17/33 maxfev, alle gates og
   frosne data. Et større loft er ikke en konvergensgaranti.
3. Gendan det præcise runtime, kontrollér optimizerens gemte kald og afklar
   genoptagelse via `run_h2_budgeted_search.py`. Den originale controller
   og alle 69 numeriske kilder er uændrede.
4. Færdiggør to-/fire-stykkers fit, egne fine/selection-streams og modelvalg.
   Frys derefter 168 timers åben prognose og I/J-handoff, før slutdata læses.
5. Slutdata findes fortsat kun i den private PC-workbook. Den eksisterende
   dataeksportbro bruges først efter et faktisk frosset forecast.
6. Afslut PR-review og I/J-dokumentation. PR106 forbliver draft under denne blokering.

Fysik og støj er fortsat nominelle betingelser; ukendt tidligere eksponering
betyder udviklings-holdout, ikke blind H3-accept. F og G/WM holdes adskilt.
F4-G2b/G2c genstartes ikke.

## Gemte filer
- `f4_h2_crash_recovery_audit_20261007.json`: råreview, alle jobhashes og præcist stopsted.
- `f4_h2_crash_campaign_before_20261007.json` og `...after_20261007.json`: fuldt regnskab.
- `f4_h2_crash_budget_reconciliation_20261007.json`: entry-/budgetafstemning.
- `f4_h2_crash_restoration_20261007.json`: udført gendannelse og arkivhashes.
- `f4_h2_crash_recovery_20261007.zip`: lille overdragelsespakke med review og regnskab;
  de seks originale ZIP-checkpoints er fortsat nødvendige.

Alle ovenstående råartefakter ligger under `files/output/bm_1/runs/estimation`.
Den gendannede lokale mappe hedder `f4_h2_staged_recovered_20261007` og
indeholder en STOP-markør. Det er en pauset gendannelse, ikke en ny kørsel.
