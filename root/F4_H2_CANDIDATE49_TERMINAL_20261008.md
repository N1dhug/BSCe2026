# F4-H2: teknisk recovery afsluttet; firestykkefit rammer kandidatloftet

Det autoriserede V2-recoveryscope har nu leveret den sidste kandidat under
uændrede model-, data-, kernel-, solver- og optimizerindstillinger. Alle72
timer af kandidat49 er nye, fuldt gemte råresultater. De eksisterende lineære
og tostykkers egne base/fine TRAIN/selection-resultater blev genbrugt.

## Faktisk afgørelse

COBYQA afslutter med status5: maximum function evaluations exceeded,
nfev49 og converged=false. De registrerede trust-radii .5→.02,
logit-bounds[-4,4], én oprindelig lineær start og 25/49-caps er uændrede.
48 tidligere objectiveværdier blev replayet byte-/rækkefølgeverificeret,
derefter blev kun den oprindeligt foreslåede kandidat49 beregnet.
Der er ingen genstart fra bedste kandidat og ingen ændret acceptancegate.

| Størrelse | Resultat |
|---|---:|
| Nye native råjobs |72|
| Objectivekald i samlet firestykketrace |49|
| Kandidat49 logscore |47.04005545171794|
| Kandidat49 penalty |3.691774198270319|
| Kandidat49 objective = -score + penalty |-43.348281253447624|
| Kandidat48 objective |-43.32660243014691|
| Objectiveforbedring fra48 til49 |0.021678823300714|
| COBYQA convergence |false|
| Nye job-failurekvitteringer |0|
| Modelvalg / forecastfreeze / final-Y-decode |ikke udført|

Kandidat49 er bedste registrerede TRAIN-punkt, men maxfev-resultatet er ikke
et lokalt konvergeret optimum. Det er heller ikke bevis for at firestykkemodellen
er dårligere: modelvalg på selection og egen firestykkers fine-validering
blev slet ikke nået. En lav TRAIN-objective kan ikke erstatte disse gates.

Kandidat48's tidligere11 tabte rawtimer er fortsat eksplicit tabt.
Kandidat49 er fuldt durable og dækker ikke det tidligere evidenshul.
Lineær/tostykke var allerede validerede udviklingsresultater; deres genbrug
er ikke en ny videnskabelig kontrol. H/I/J/F4 er åbne.

## Opstart, kontrol og bevaringsregler

Userapproval: "du gaar bare igang"; den konkrete pending udvidelse er
native1080→1140s inden for samme20h/23:55/caps25/49/pair73.
V2-registrede-startup ramte FileExistsError, fordi predecessorens eksisterende
file-lock blev behandlet som directory. Det skete før reservation/import/
manifest. Separat V2a retter kun låsestien; V2 og fejl bevares.
Første produktionsadmission havde ved en fejl et placeholder-kontrolargument.
Det blev afvist før Engine/recovery_execution/numeriske jobs; fuld faktisk
2.837591051s-charge bevares i admission-rejection-JSON.

Den ene samlede V2a-kontrol består imports af alle nødvendige entrypoints,
65 ancillary kildehashes,1477+3655 rawfilhashes, exact token/checksum/prior
bytes+dtype-adoption og replay48 frem til samme kandidat49.0 propagationer.
Kontrollen/manifest blev committed og læst tilbage før korrekt production.
Kontrolcharge2.333800878s; production465.795034438s.

Checkpointauditen består72 nye NPZ tokens/checksums/prior/result-state og
427 adoption receipts inkl. den særskilte control-adoption. Produktions-
rapporten har426 adoptioner; den ekstra control-receipt forklarer forskellen.
Gamle successful jobs ommærkes ikke. ZIP-CRC og unikke medlemsnavne består.
Checkpointcharge3.806773610s. Gammel terminalfejl og reserved5s import-
watchdog bevares uændrede; ingen refund eller genoptagelse af failed execution.

## Budget og durable resultat

Autoritativ v6-ledger efter terminalreview:
H47211.81961026791/72000s, native1090.0755659060392/1140s.
Rest24788.18038973209s H og49.9244340939608s native.
Tidligere22005.997020189698s restvalidationreserve bevares i ledger.
Alle historiske reservationscharges er præcis bevaret prefix. ZIP-ledger
er et konservativt snapshot under checkpointreservationen; committed ledger
er den lukkede autoritative version, inklusive terminalreviewcharge2s.

Arkiv: f4_h2_candidate49_terminal_20261008.zip,
3279893bytes, SHA26b7120ae163abb21cc60da0337f61b85a1716f3c7cc13363f776aac76294263,
LibraryID libfile_f5c64d9cada08191829336f7ba67f3a1.
Arkivet afhænger stadig af den originale predecessor/checkpointkæde og det
terminale checkpoint12; det indeholder alle nye råjobs og source/decision.
Alle kilder/input og tidligere workbookprovenance bevares. Slut-Y er ulæst.

## Næste metodebeslutning

Den oprindelige plan kræver konvergens for alle tre retained familier og
stopper ved maxfev uden konvergens. Det krav er ikke ændret af recoveryapproval.
Der er derfor ingen automatisk ny invocation, capøgning eller selection.

Det konkrete pending forslag er at afslutte et tydeligt afgrænset historisk
H-studie for de allerede konvergerede lineære/tostykke-familier med deres
egne valideringer, selection,168h forecastfreeze og efterfølgende finale.
Firestykketracen bliver dokumenteret som ufuldstændig; ingen rangering mod
en accepteret firestykkemodel eller påstand om færdigt oprindeligt trefamilie-H.
Denne scopeændring foreslås efter kendt TRAIN-failure og allerede beregnede
development-selection-data; den må ikke beskrives som blind oprindelig protocol.
Eksisterende ukendt tidligere final-ugeeksponering og nominelle fysiske
parametre bevares. I kræver stadig selvstændig faktisk handoff/admission.

Alternativet er et særskilt større firestykkeforsøg med nye kandidat-/budget-
beregninger og præregistrering.49→flere kald er en metodeændring; dette
review giver ingen ny numerisk tilladelse og ingen garanti for convergence.
G/G7/WM, droppede G2b/G2c og fremtidige F5/F6 bevares.
