# F4-H2: første faktiske historiske pilot råreviewet, 7. oktober 2026

Den lineære MED-reference består den afgrænsede prøve på Johans PC.
Invocation **120.03942690 s**, inklusive 2.8747403 s launcher-export og
source/checkpoint/review-ZIP. Ingen optimizer. 24 prediction-only prior-timer
17.–18. februar og tre komplette TRAIN-timer fra18. februar gennemløbes
kronologisk på hver sit base/fine grid, uden state-reset. TRAIN48/selection24
har komplette gruppemålinger. MED's faste13 huse er separat og uden imputation.

Dette er **præcisionsaccept af den lineære 27-timers prefix**, ikke af hele
TRAIN/selection/final eller nonlinear søgning. Nominelle fysiske/støj/prior-
betingelser og udviklings-holdout med ukendt tidligere eksponering bevares.
Historisk fit/runtime/scientific/blind-H3 acceptance er fortsat false.

## Råresultater og provenance

Original PC-source: `2846a5474027101a45b43aa66e9ec111a8727628`.
Råreceipt: `files/output/bm_1/runs/estimation/f4_h2_compact_rates_pc_review_20261007.zip`.
SHA256 `cb7a30a2712116082337248a35ec2644850585caf8b2abc97396a421eb91cffb`.
Aktivt indeks: `f4/h2_compact_rate_reviewed_20261007.json`.

Read-only auditor `f4/audit_h2_compact_rates.py` verificerer alle64 sourcehashes
mod den præcise Git-commit, med kun eksakt Windows LF/CRLF-checkout tilladt.
Manifest/packet/observation/checkpoint-hashes matcher. Frosne audit-bytes
bevares nu korrekt efter PR100. Alle64 raw jobs og fulde egne1921/2561-
logstates har korrekt kronologi, manifestidentitet, videreført prior og
normalisering. Total-score/momenter rekonstrueres; gemte CF'er rekonstruerer
CDF/kvantiler uden native/FV-replay eller ny JIT-propagation. Den samlede
12-jobgraf er prissat ved uafhængig aritmetik, uden at dividere candidates
med8 frekvensworkers. PC-Python3.11 ordered-ledger-sum og prefix matcher.

| Råmål | Største fejl | Uændret gate |
|---|---:|---:|
| Neutral masse | 1.670e-13 | 1e-10 |
| Conditional guard | 1.633e-9 | 1e-8 |
| Contour overlap log | 4.209e-9 | 1e-7 |
| Base/fine state mean | 2.358e-7 | .002 |
| Konservativ state L1 | .000493025 | .05 |
| Total prefix log-score | 3.520e-6 | .05 for hele vinduet |
| Pre-update interval mean | 2.419e-7 | .002 |
| Pre-update interval variance | 3.683e-8 | .002 |
| CDF | 3.766e-7 | .002 |
| Kvantil | .0001220703125 | .002 |

Workbooken med de13 rå husholdningsserier følger ikke med. Dens hash matcher
den auditerede preparation, og decoded72-hour group/mask/normalization er
konsistente med den verificerede runner. Rå household means kan ikke
uafhængigt genberegnes her; provenance-/databegrænsningen er eksplicit bevaret.

## Faktiske rates og samlet budget

| Varm maksimumrate, s/job | Base | Fine |
|---|---:|---:|
| Observeret interval/Bayes-opdatering | 3.865074 | 8.988610 |
| Prediction-only interval | .463259 | 1.185170 |
| Intervalmomenter | .509354 | .796035 |
| CDF/kvantiler | 1.322576 | 2.852026 |

49-kandidatplanens søgearbejde, alle deklarerede base/fine replays, winner+
linear reference, prequential/open-week momenter og CDF-prober, faktor1.5,
600 s imports/JIT/wrappers/export samt60 s stopreserve giver **7.84909 h**
med PC-ledgeren. Efter råreviewets egne charges er overslaget **7.85130 h**.
Det er under10 h, over4–6 h-målet. Beskyttet validation er **11840.47191 s**
(3.289 h); den fælles ledger hæver reserven til dette niveau uden dobbelttælling.
Cold jobs23.98924 s ligger inden for det særskilte300 s auxiliary-cap.
Denne pris er en lineær prefix-ekstrapolation, ikke et nonlinear worst-case
bound eller et løfte om optimizerkonvergens. Ingen pilotarbejde fratrækkes.

Alle PC120.03943 s, forberedende råreview4 s, to afbrudte read-only auditor-
forsøg1.22532/1.29270 s og bestået audit1.44895 s er bevaret. Første auditor
krævede normalisering af Windows ZIP-checkpoint-separatorer. Den anden
brugte for stram platformsammenligning af Beta-CDF-logtal (9e-12 forskel);
initiale Beta-cellemasser kontrolleres nu mod1e-13, derefter bevares de
originale fulde logstates og bitvis state-continuation. Ingen modelgate
eller PC-numerik ændres. Ingen gamle modelchecks eller pilot genkøres.

Ny H-ledger **1671.04085909 s**, native **646.40973845/900 s**, rest
**253.59026155 s** samlet på PC/chat/resumes. Seed er en præcis forlængelse
med38 entries, aldrig et reset. Original PC-ledger og alle reviewcharges er
bevarede. Reviewets source/environment og resultater må ikke ommærkes som
fuldt fit, blind evaluering eller validering af andre response-familier.

## Næste trin

Sol High er tilstrækkelig til dette råreview og den efterfølgende afgrænsede
implementering. Næste er et konkret admission-design, som afklarer både
nonlinear ratevariation og full-window base/fine præcision inden for den
resterende allowance og den prissatte validation-reserve. Det korte lineære
forløb og 7.85 h alene åbner ikke optimizer. Ingen ny PC-kørsel er bestilt
af dette review, og den afsluttede prefix må ikke blot køres igen.

Et design skal bruge de gemte states/data/source, bevare de deklarerede
1/2/4-piece familier, 49-kandidatloftet, fælles f/g og alle gates, og eksplicit
afklare om nødvendige hele-vindue-checks kan finansieres og gennemføres.
Det må ikke stille skjult længere arbejde uden for den900 s udviklingscap
eller starte fit før de faktiske forudsætninger er opfyldt. Hvis den cap og
den krævede validering ikke kan forenes, skal der vælges en konkret
begrundet revision; ingen automatisk flere sekunder, kortere TRAIN, færre
familier eller løsere tolerancer. G/WM er uændrede.
