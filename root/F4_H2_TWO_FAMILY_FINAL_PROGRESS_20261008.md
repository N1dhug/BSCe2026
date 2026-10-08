# F4-H2: faktisk slutvurdering og næste admissiongrænse

Den godkendte indsnævring til lineær og eksisterende konvergeret tostykkemodel
er gennemført gennem modelvalg, åben ugeprognose og tre afgrænsede blokke af
slutvurderingen. Tostykkemodellen er valgt på begge egne grids uden ambiguity.
Der er ingen ny parameterfit. Firestykkers49-call maxfev forbliver ufuldstændigt.

## Slutvurderingens aktuelle resultat

Alle700 open-forecastjobs og168h forecast blev færdige og frosset før de168
slutobservationer fra21.–28.februar2022 blev eksporteret fra originalt workbook.
De åbne prognoser assimilerer ingen slutobservationer. Finalprequential bruger
egen cutoff-state og de foregående slutobservationer; den er en separat gren.

| Bounded blok | Nye finalrawjobs | Faktisk charge | Afslutning |
|---|---:|---:|---|
| 1 |123|510.2321375499996s|Normal paused; checkpoint04|
| 2 |124|512.1064546430007s|Normal paused; checkpoint05|
| 3 |98|512.3471486899998s|Normal paused; checkpoint06|

345 af1344 finalrawjobs er gemt. Lineær/base168h moments/observed er komplet;
lineær/fine er påbegyndt. Tostykke base/fine og samlede finalmetrics er endnu
ikke færdige. Ingen failed rawjobs, numerisk retry eller ændring af modelvalget.

| Checkpoint | Nye jobs | SHA256 | LibraryID |
|---|---:|---|---|
|04|123|1fd85c3ee82b5c492dac5398288fbd7be4f64c3f5b10c3a4b210e3f1e9ef393d|libfile_ba814a2c37d081919ed8657eed92684f|
|05|124|30e013ce97daafd344689a0700d8cf463980a3faa7308c7e920c9aa655fdba50|libfile_cc2270e29548819197319dd74cd57903|
|06|98|a7a99f1c8ce788e806424352c3aa3500c38c97d2aba6e14aa53b70e62ea6cb7c|libfile_77c38a406b1c81919d51bacb85d69013|

Checkpoint06 har4954274 bytes og råjob-/adoption-/ZIP-audit på4.199236299999029s.
Arkiverne er deltaer. Den fulde tidligere arkivkæde, inklusive forecastfreeze
i02 og predecessor1477/3655jobs, kræves. Kandidat48's11 manglende råtimer er
fortsat manglende. SHA-afhængigheder står i arkivindeks; intet tab ommærkes.

## Regnskab og konkret pending forslag

Autoritativ lukket V8-ledger: H49805.8944273059/72000s;
native1136.465941387039/1140s. Rest H22194.105572694098s, native3.534058612961s.
SHA26a5cf6d08d1fb6d135892b425a7336f4592a83a6e1a0393d192039603796087.
Gamle crash-reservationer5s og600s beholdes fuldt og alle gamle entries urørt.
Den sidste .5s charge dækker konservativ terminal/proposal/registration preparation.
Ingen proces kører. Næste numeriske blok kræver8s reserveret checkpoint og
kan derfor ikke admitted under den nu godkendte native-grænse.

Konkret forslag: +60 native/control/export sekunder,1140→1200, inden for
samme72000s og uændret23:55 Copenhagen-frist. Dette er approved=false.
V4-controller og deltaexporter er klargjort, statisk parset; numeriske funktioners
AST og73 frosne kernekilder er uændrede. V4 er ikke invoked. En separat approval,
V9-ledger med hele V8-prefix, kilderegistrering/readback og bounded provenance/
budgetkontrol skal foreligge før næste blok. Alle guards og tidligere failures
bevares. Ingen nye optimizerkald er tilladt. Fremtidig blokantal/runtime er usikker.

Forslag: root/f4/h2_two_family_v4_native_supplement_proposal_20261008.json.
Kildereview: root/f4/h2_two_family_v4_pending_source_review_20261008.json.
Frozen identities og samme offlinecontrol397d2ed5 er fastlåst i forslaget.

## Fortsat faglig status

H/I/J/F4 er ikke afsluttet. Afgrænsningen er et betinget development-studie
med nominelle fysiske parametre og ukendt tidligere slutugeeksponering.
Det er ikke originalt trefamilie-H, blind holdout eller populationskalibrering.
Efter faktisk finalgate kommer H→I origin/transferreview, særskilt I-forecast/
F3-admission og evidens, samt J's resultater/figurer/reproduktionsacceptmatrix.
Direkte WM-sammenligning/F5/F6 ligger fortsat udenfor, og G's parallelle arbejde
og droppede F4-G2b/G2c bevares.
