# F4-H2: terminalstatus og konkret fortsættelsesforslag, 7. oktober 2026

## Faktisk resultat

Den godkendte dagskørsel fra controllercommit `717187599f35cde455df20a5189a026204498dd5`
stoppede med optimizerstatus **5**, efter **17 evalueringer** af to-stykkers
familien. Meddelelsen er: maksimum for objective-evalueringer overskredet.
`converged=false`; den erklærede trust-radius-termination er ikke opnået.
Det er et kandidatloft, ikke et crash eller en numerisk jobfejl.

De to lukkede blokke brugte 540,230948 s og 527,278553 s og gemte i alt
326 nye jobs. Samlet **1402 immutable jobs**, alle **17 komplette TRAIN-objectives**
og alle **69 frosne numeriske/runner-kilder** består råreviewet. Den oprindelige
pinned COBYQA reproducerer præcis både kald og terminalstatus uden nye modeljobs.
Der er ingen gemte failed-jobfiler. Den separate fire-stykkers PROBE har27 timer;
fire-stykkers optimizer er ikke startet. Lineær egen base/fine96-timers reference
består fra tidligere. Checkpoint01–08-kæden er hashverificeret med1402 jobs.

Bedste gemte to-stykkers kandidat har logits
`[-1.0697627598794845, -1.4940067801109866]`, objective `-10.155866238911589`,
TRAIN-logscore `11.436595129459494` og regularisering `1.280728890547905`.
Lineær TRAIN-logscore er `-9.266717729604274`. Dette er TRAIN-resultater;
de dokumenterer ikke held-out forbedring, valgt model eller et konvergeret fit.

Maksimal gemt mass-error er `1.8829382497642655e-13`; conditional guard
`9.998935887328654e-09` og overlap-logdifference `2.689807843125891e-08`.
Alle oprindelige guards består; de er ikke løsnet. Conditional guard ligger tæt
på sin deklarerede grænse1e-8; yderligere kandidater kan stadig fejle numerisk.

## Budget og bevaret historik

V2-kontoen har charged_seconds **34030.249461839914**, rest
**30769.750538160086 s** =
**8.547153 h** under18h-loftet.
Beskyttet restvalidering er stadig **22005,997020 s** = **6,112777 h**.
Efter denne reserve og60s stopmargin er højst
**2.417709 h**
til rådighed for en eventuel ny søgning, før valideringskreditter frigives.
Native-development er **854.498416807043/900 s**; rest **45.501583192957 s**.
To lukkede raw-audit entries bevares, inklusive den overlappende gentagelse;
ingen skjult refundering. Den gamle åbne30000s-crashreservation er også bevaret.
Fristen er fortsat7. oktober23:55 Europe/Copenhagen.

## Hvad der mangler

1. Algoritmisk konvergens for to- og fire-stykkers familier.
2. Egne nonlinear fine-TRAIN/selection-streams og base/fine-gates.
3. Frosset sammenligning af1/2/4 familier og valgt model.
4.168 timers åbent ugeforecast med egne grids, moments og syv distributionshorisonter.
5. Slutdataeksport fra privat PC-workbook efter forecastfreeze, separat prequential
   finale, og faktisk conditional I-handoff/resultatkonklusion til J.

Slut-Y er ikke åbnet. H/I/J/F4 er ikke afsluttet; PR106 forbliver draft.
G/WM og de droppede F4-G2b/G2c berøres ikke.

## Konkret metodeændring til godkendelse — ikke udført

Dagsprotokollen fastholdt17/33 maxfev og erklærede cap/failure terminal.
Budgetgodkendelsen ophævede ikke dette stopkrav. Derfor køres ikke flere
kandidater i den fejlede run, og failure/resultatfiler slettes aldrig.

Forslaget er **25/49 maxfev** for henholdsvis to- og fire-stykkers familier:
højst24/48 nye nonlinear par og73 forskellige par inklusive lineær reference.
Det større kandidatloft er en eksplicit metodeændring, ikke konvergensaccept.
Samme COBYQA, startpunkt, logit-bounds[-4,4], regularisering, trust-radius.02,
1/2/4-familier, dataskel og numeriske tolerancer bevares. Ingen multistart.

En ny særskilt execution/run bevarer den negative17-cap-kørsel fuldt. Verificerede
immutable states og de17 præcise objective-kald skal genbruges med uændret
numerisk identitet; ingen startfra-nul eller genberegning af afsluttede jobs.
Det kræver en frosset controller-/registry-/auditoradapter, hvis egen sourceidentitet
og metodebeslutning registreres før nye modeljobs. Den gamle failed run forbliver
autoritativ evidens for den oprindelige metode. Gammel source/ledger omskrives ikke.

18h totalbudget,900s development, stop23:55,600s blokke og hele protected
validation-reserven bevares. Ingen fuld73-par-graf er admitted; nye jobs tillades
kun med plads efter fuld restvalidering og stopmargin. Ved nyt cap-/numerikstop
bevares ny negative evidens uden automatisk retry eller ekstra capforøgelse.

De otte ekstra tilladte to-stykkers evalueringer svarer isoleret til omtrent
27–36 minutters modelarbejde ved de hidtil målte komplette kandidatrater.
Det er ingen konvergensgaranti. Hele restarbejdet har ikke et sikkert timeestimat:
fire-stykkers fulde kandidater og finalgrene er endnu ikke målt, og højere caps
udvider kun den mulige søgning. Dagens afslutning afhænger af konvergens,
præcisionsgates og rettidig slutdataeksport. Et uændret17-cap-forsøg kan ikke
fortsætte lovligt alene ved at vente eller skifte chat/model.
