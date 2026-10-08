# F4-H2: resultatfortsættelse og terminalt mappetab

Den godkendte +2h-fortsættelse blev registreret og kørt med samme model,
kurvefamilier, ni-konturbank, frosne data, SciPy/Numba-versioner og 25/49-caps.
Implementerings-/recovery-PR122 er merged. PR123 samler checkpoint07–12,
lukket autoritativ v4-ledger og terminal evidens. Det er en leverance af
resultater og historik; H/I/J og hele F4 er endnu ikke afsluttet.

## Faktisk resultat

| Evidens | Resultat |
|---|---:|
| Firestykkekald i gemt optimizertrace | 48 |
| Fuldt durable kald før sidste blok | 47 |
| Råjobs i checkpoint01–11 | 1461 |
| Nye råjobs bevaret/auditeret i checkpoint12 | 16 |
| Samlet bevarede fortsættelsesjobs | 1477 |
| Verificerede predecessor-råjobs | 3655 |
| Verificerede adoptioner | 450 |
| Lukket H-charge / loft | 46710.38319202991 / 72000 s |
| Lukket native-charge / loft | 1060.6012936430393 / 1080 s |
| Resterende H / native | 25289.616807970087 / 19.3987063569607 s |

Kald48 har gemt score46.98529721329487, penalty3.658694783147961 og
Q=-43.32660243014691. Det er det laveste gemte TRAIN-objective i denne trace.
Denne forbedring er ingen convergence-, selection- eller forecastaccept.
Der findes ingen firestykkers termination eller frosset forecast.

To-stykkers tidligere konvergerede20-kaldsresultat og egen base/fine-
TRAIN/selection-validering bevares. Disse jobs er ikke genkørt. Endpoints,
intervalgennemsnitsobservation, nominelle fysiske parametre, informationscutoff
og udviklings-holdout-afgrænsning er uændrede.

## Hvad der stoppede, og hvad der er ukendt

Blok6 rapporterede27 nye jobs og afsluttede kald48. Derefter fejlede næste
kandidats første predictionjob (nulbaseret time0) med FileNotFoundError.
Den gamle scratch-root `/workspace/scratch/8502b3d6ceb9/f4_recovery` var
forsvundet under den aktive proces: kildefiler og tidligere lokale råjobs
manglede. Numbas source-location-formattering ramte `os.getcwd` via relpath.
Årsagen til den eksterne mappefjernelse kan ikke udledes af denne traceback.
Det dokumenterer en miljøfejl; det beviser ingen matematisk model-/guardfejl.

Kun16 af blok6's27 rapporterede nye NPZ-filer overlevede. De øvrige11 timer
for kandidat48 (nulbaseret45–55, menneskeligt46–56) findes ikke i de tidligere
checkpoints. Efter byte-restoration har kandidat48 gemte timer0–44 og56–71.
Den gemte48-kaldstrace bevares som historik, men dens sidste score har dette
råevidenshul. Det må ikke dækkes af en antagelse om komplette checkpoints.

SUPPLEMENT_FAILED.json, supplement_report_005.json og job-failurekvitteringen
er bevaret urørt. Original controller afviser terminalt run; intet nyt modeljob
blev startet efter fejlen. Blokkens målte320.8691462249999s er charged;
alle tidligere fulde crashreservationer og negative resultater bevares.

## Recovery og kontrol

Et separat nødarkiv sikrede alle overlevende filer før restoration.
Alle20 predecessor-arkiver og checkpoint01–11 blev hashverificeret og
gendannet til den aktuelle arbejdsmappe. Original input.xlsx har korrekt
SHA256; slut-Y er stadig ikke decoded. De73 frosne numeriske/sourcefiler og
de to supplementkontroller er bevaret. Ancillary package-importfiler er
gendannet fra den immutable controllercommit, uden numerisk ændring.

Den separate preregistrerede15s-checkpointhelper ændrer kun audit/eksport-
reservation og crashsikker ledgerlukning. Original30s-helper og numeriske
kontroller forbliver uændrede. Den aktuelle read-only kontrol verificerede
16 nye raw tokens, filhashes, content-checksums, prior/result-state og450
adoptioners originale provenance. Den brugte2.3642036950004695s med nul
modelpropagationer. Det er råintegritetskontrol, ikke scientific acceptance.
Failed audit-startups og konservative recoverycharges11+5+4s er bevaret.

Checkpoint12 ligger i et verificeret ZIP med SHA256
`d4acb65b55840d467b97e63d299d9c78b6b2b1f4c14d3256124733ec23464b0c`.
Den nyeste committed v4-ledger er autoritativ; ZIP'ets ledger er et snapshot,
som konservativt kan indeholde checkpointreservationen før endelig lukning.
Ingen fuld numerisk testsuite eller tidligere scientific gate er genkørt.

## Næste konkrete beslutning

`root/f4/h2_filesystem_recovery_proposal_20261008.json` foreslår en særskilt
execution/manifest med vellykkede råjobs adopteret under deres originale
identitet og præcis replay af48-kaldstracen. Det gamle terminale run og alle
failurefiler ændres ikke. Den sidste kandidat49 og eventuel egen validering
skal bestå de samme guards; de11 manglende råtimer må genskabes som nye jobs
ved behov og må aldrig påstås genfundet. Ingen start fra bedste kandidat.

Forslaget ændrer ikke20h-total, deadline23:55 Copenhagen, 25/49-caps,
paircap73, 600s-invocation eller60s-stopreserve. Det foreslår +60s til native-
allowance1080→1140s inden for den samme20h-total:10s recoverypreparation og
15s checkpoint-reservation skal kunne admitted sammen med fremtidig eksport.
Nye source-/origin-pins og én fokuseret provenance/budgetkontrol kræves før
launch. Forslaget er IKKE godkendt eller kørt. Der loves ingen convergence.

AGENTS.md's aktuelle F-status siger: "Ingen ny numerical retry eller
capudvidelse." Den aktive plan siger også "Ingen selvstartende baggrundsloop
eller automatisk retry efter fejl." Derfor kræver denne nye execution
en særskilt brugerbeslutning. En almindelig filesystem-reparation ophæver
ikke den gemte terminalmarkør. H-fit/selection/forecast/finale og I/J/F4
står fortsat åbne. G/G7/WM og droppede F4-G2b/G2c bevares.
