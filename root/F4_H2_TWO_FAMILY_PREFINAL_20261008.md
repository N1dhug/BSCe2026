# F4-H2: godkendt afgrænsning, modelvalg og frosset ugeprognose

Johan godkendte17:54 den konkrete indsnævring til lineær/tostykke. Det
firestykkersfit, der ramte49-call maxfev, ommærkes ikke til converged.
Den samme nominelle fysiske betingelse, frosne B/U, gruppens13 målinger,
intervalobservation, tid/cutoff, egne base/fine priors, kerner og gates bevares.
Ingen ny parameter- eller kurvefit er udført; den eksisterende konvergerede
tostykkekurve deles over hele studieperioden.

## Modelvalg og betydning

| Model | Base selection-logscore | Fine selection-logscore | Base normaliseret RMSE |
|---|---:|---:|---:|
| Lineær |13.789399793783542|13.78894563229434|.08607039646002732|
| To stykker |23.267887918742304|23.26716786258129|.07219447419119981|

Begge modeller har24 komplette selection-timer med egne pre-update momenter.
Den godkendte regel er uændret indenfor retained1/2: rangér ved summen af
predictive interval-logdensiteter; ved base/fine rank reversal eller gap≤.1
vælges enklere model blandt de tvetydige. Her er begge rankings[2,1],
gap≈9.478 og ambiguity=false. Resultatet gælder de to retained modeller.
Firestykkemodellen er ikke en accepteret konkurrent i denne sammenligning.

Ved cutoff21.februar2022 eksporteres hver models egen filtered posterior
på begge grids. Det åbne forløb har168 timer til28.februar og assimilates0
final-observationer. Alle intervalmomenter og CDF'er ved1/6/12/24/48/72/168h
er beregnet og sammenholdt mellem de egne grids før freeze. Slut-Y kunne
først læses, da forecast/selection identity, checksum og dependency-hashes
bestod. Alle168 slutobservationer er nu eksporteret; ingen finale endnu.

Matematisk er modelvalgs-L_m=Σ log p_m(Y_h|Y_<h,B,U), og RMSE vurderer
samme models pre-update predictive middel mod den normaliserede observation.
Open-grenen vurderer loven fra cutoff-posterioren med frosne fremtidsinputs,
uden opdateringer. Den kommende prequential gren begynder i den samme egen
cutoff-state, men bruger hver foregående slutobservation i filteret.
De to grenes fejl er forskellige diagnostics og må ikke blandes eller bruges
til efterfølgende at ændre modelvalget. Fysisk/input-parameterusikkerhed er
ikke inkluderet.7 horisonters coverage er beskrivende, ikke kalibreringsbevis.

Afgrænsningen blev valgt efter den kendte TRAIN-maxfev og allerede kendte
development-selection-resultater. Holdout-blindness er ikke verificeret,
og tidligere exposure til slutugen er fortsat ukendt. Dette er et godkendt
betinget development-studie, ikke originalt trefamilie-H eller populationsaccept.

## Faktisk implementering og verifikation

Separate immutable controller-revisioner ændrer kun narrowed selection,
provenance, admission/export og bounded process shutdown.73 oprindelige
numeriske kilder og65 gendannede ancillary moduler er hashverificerede.
Own full-window1/2-cacheadoption er verificeret,0 tidligere successjobs genkørt.

Første0-job-startup blev afvist pga en forkert controlargument og derefter
JSON's integer/string table keys. Fejl/charges og første controller bevares;
separate V2 canonicaliserer kun JSON-sammenligning.
V2 producerede500 vellykkede nye prognosejobs. Beregningen nåede deadline,
men multiprocessing pool shutdown hang; watchdog124 beholdt hele600s.
Separat V3 begrænser shutdown uden at ændre modelberegningen og har en
fokuseret actual-close-AST2-worker cancellationkontrol:2.001707805s,
0 numeriske modelpropagationer. V3 genbrugte500 jobs præcist og producerede
de resterende200. Den afsluttede normalt med prefinal_completed.
Ingen failed rawjobs, optimizerkørsel eller fiktiv infinity-score.

Alle700 jobs,427 adoptionreceipts og ZIP-integritet er auditeret.
Deltaeksport SHA-verificerer tidligere rawfiler, receipts og arkivbytes.
Hash-identisk allerede sikret metadata pakkes ikke gentagne gange;
eksakte hash-/arkivafhængigheder står i durable index. Hele kæden kræves.

## Holdbare artefakter og regnskab

| Arkiv | Nye jobs | LibraryID | SHA256 |
|---|---:|---|---|
| checkpoint01 |500|libfile_55acd074012081919605b90e2c10536b|021e2b5c79638310608b572729f5df88c961bcefb3b354b530dc59f58e8391a6|
| checkpoint02 |200|libfile_303558252e4c81919f6806763fcfc771|96e1e5a788e8a25df7711b415971252c2da18fe0ed76153c0c76dae27010e094|
| checkpoint03 |0|libfile_10356f71693881918bd567180f23b02d|66e74ffb229e5ce5b88dd0b1d8c580d2e34104fdacc7c02e1b6157bd177c7d8a|

Forecast_frozen SHA4f78a4f6abc1f5b4b555d892d317c9b8425faa4bfd3b5dcb62ece67a0423c169;
conditional_handoff SHAca358bc3ea3c12086490149073c8e520b7673efda63be49c048c410e8158bf16;
final_observations SHA393733a53c33147901081a1df88974004c28b0ba34e306adcd182e0c3259f1ab.
Tidligere1477+3655 predecessorjobs og originale failure/checkpointkæder
bevares; kandidat48's11 tabte rawtimer er stadig eksplicit tabt.

Seneste lukkede v8-ledger: H48259.3884600449/72000s,
native1124.6457150090405/1140s. Crash600 og gammel reserved5 bevares
som præcis prefix; intet nulstilles/refunderes. Checkpointets campaign er
konservativt reservationssnapshot; committed lukkede ledger er autoritativ.

Den fulde prissatte narrowed restgraf havde18343.252636245586s med factor1.5
og alle jobtyper plus300 helper/60 stop. Remaining modelrun er admitted
i20h-total, men native eksportreservation og23:55 deadline kontrolleres
før hver blok. Der loves ingen numerical finale-gate før faktisk resultat.

## Resterende

1. Finalprequential replay på begge models egne base/fine-cutoffstates;
   sammenlign åbne og prequential mean-errors/logscores/descriptive coverage.
2. Read-only kontrol af det faktiske H→I origin/interface og konservativ
   transfer; selvstændig I forecast/F3 evidence og admission er stadig åbne.
3. J's samlede resultater/reproduktions-/acceptmatrix og nødvendige figurer.
   H/I/J/F4 er ikke afsluttet af denne prefinalleverance. Direkte WM-
   sammenligning og F5/F6 er fortsat separate fremtidige trin.
