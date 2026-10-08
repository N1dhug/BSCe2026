# F: tidlige H2-afbrydelser og recovery

Disse hændelser er afløst af senere recovery. De gamle reservationer, fejl og checkpointafhængigheder er fortsat evidens; ingen gammel næste-kommando er en ny kørselstilladelse.

Status: historisk evidens. Aktuel brug og scope står i [BM-guiden](../../../root/BM_GUIDE.md).

Afsnittene nedenfor samler resultater og begrænsninger uden de gentagne arbejdsordrer. Oprindelige matematisk væsentlige afsnit er bevaret særskilt. Fuld originaltekst, kildehash og erstatningssti findes via [stikortet](../old-paths.json) og den verificerede kildepakke.

<a id="f4-crash-recovery-20261007"></a>
## F4_CRASH_RECOVERY_20261007.md

Original: `root/F4_CRASH_RECOVERY_20261007.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-crash-recovery-20261007--f4-h2-gendannelse-efter-crash-7-oktober-2026"></a>
<a id="f4-crash-recovery-20261007--resultat"></a>
<a id="f4-crash-recovery-20261007--præcist-stopsted"></a>
<a id="f4-crash-recovery-20261007--budget-og-hvorfor-modellen-ikke-blev-startet-igen"></a>
<a id="f4-crash-recovery-20261007--miljø"></a>
<a id="f4-crash-recovery-20261007--næste-arbejde"></a>
<a id="f4-crash-recovery-20261007--gemte-filer"></a>

Checkpoint01–06 genoprettede 1076 råjobs og 69 kildefiler; 12 tostykkekald var komplette. Den åbne 30000 s crashreservation blev bevaret. H32911.549305 s efter recovery var utilstrækkeligt til den beskyttede restvalidering; ingen automatisk fitgenstart eller budgetrefundering.

<a id="f4-h2-progress-20261007"></a>
## F4_H2_PROGRESS_20261007.md

Original: `root/F4_H2_PROGRESS_20261007.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-h2-progress-20261007--f4-h2-gemt-fremdrift-efter-afbrydelse-7-oktober"></a>
<a id="f4-h2-progress-20261007--resultater-som-faktisk-er-opnået"></a>
<a id="f4-h2-progress-20261007--afbrydelser-og-budget"></a>
<a id="f4-h2-progress-20261007--præcis-fortsættelse"></a>

3324 råjobs og alle 69 kilder bestod råaudit. Tostykkefit konvergerede efter 20 kald og bestod egen base/fine TRAIN/selection; firestykkefit havde 22 komplette kald. To afbrudte 600 s blokke og den gamle crashcharge blev bevaret. Original checkpoint01–08 plus extension01–10 er en afhængig kæde, ikke uafhængige snapshots.

<a id="f4-night-checkpoint-20261007"></a>
## F4_NIGHT_CHECKPOINT_20261007.md

Original: `root/F4_NIGHT_CHECKPOINT_20261007.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-night-checkpoint-20261007--f4-nattearbejde-7-oktober-2026"></a>
<a id="f4-night-checkpoint-20261007--f4-h2-crash-gendannet-videre-kørsel-budgetblokeret-7-oktober"></a>
<a id="f4-night-checkpoint-20261007--genoptagelse"></a>
<a id="f4-night-checkpoint-20261007--bindende-udgangspunkt"></a>
<a id="f4-night-checkpoint-20261007--status"></a>
<a id="f4-night-checkpoint-20261007--aktiv-kørsel-og-sidste-holdbare-kilde"></a>

Nattens checkpoint beskrev det budgetbegrænsede H2-forsøg og dets senere delresultater. Hver afsluttet blok skal læses sammen med controllerkilden, den autoritative ledger og predecessorarkiverne. Den gamle natfrist og instruktion om fortsættelse er afløst; udførelse er ikke H/I/J-slutaccept.

<a id="h2-soft-stop-recovery-20261007"></a>
## H2_SOFT_STOP_RECOVERY_20261007.md

Original: `root/H2_SOFT_STOP_RECOVERY_20261007.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="h2-soft-stop-recovery-20261007--f4-h2-kontrolleret-tidligt-blokstop"></a>

STOP bruges til kooperativ afslutning ved en holdbar jobgrænse, hvorefter ledger og checkpoint eksporteres. Parent/child-signaltesten er særskilt evidens; den må ikke ommærkes til fuldt verificeret modelnedlukning. Afbrudte reservationer og fejl beholdes.

<a id="f4-h-checkpoint-audit-state"></a>
## F4_H_CHECKPOINT_AUDIT_STATE.md

Original: `root/F4_H_CHECKPOINT_AUDIT_STATE.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-h-checkpoint-audit-state--f4-h-recovery-and-checkpoint-audit--session-state"></a>
<a id="f4-h-checkpoint-audit-state--repository-and-ownership"></a>
<a id="f4-h-checkpoint-audit-state--status"></a>
<a id="f4-h-checkpoint-audit-state--recovery-events-and-exact-commands"></a>
<a id="f4-h-checkpoint-audit-state--scientific-findings-and-limits"></a>
<a id="f4-h-checkpoint-audit-state--tests-actually-run"></a>
<a id="f4-h-checkpoint-audit-state--review-ledger"></a>
<a id="f4-h-checkpoint-audit-state--durable-artifacts"></a>
<a id="f4-h-checkpoint-audit-state--next-concrete-actions"></a>

Samlet historisk recoveryindgang for F4-H: originale råartefakter, sourceidentiteter og både beståede og fejlede intervalkontroller bevares. Grove grids og narrow-noise Fourier-inversion blev ikke accepteret alene ud fra massenormalisering. Den senere terminal-tilt/native-metode har særskilte kontroller; tidligere negative resultater og budgetstops er fortsat gyldige for de gamle settings. Aktuel matematik står i H2_TERMINAL_TILT_REVIEW/CROSS_CONTROL/FINE_REFERENCE og H2_COMPACT_ESTIMATION_REVIEW.
