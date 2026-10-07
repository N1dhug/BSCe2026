# F4-H2: etapeopdelt natplan og kørselskontrakt

7. oktober 2026. Dette er en eksplicit revision af **rækkefølgen** i
`H2_COMPACT_RATE_REVIEW.md`, ikke accept af dets korte pilot som fuldt fit.
Johans natinstruks autoriserer selvstændigt metode-/implementeringsarbejde;
10-timersloftet, 900 s udviklingsloftet, kurvefamilier og tolerancer bevares.

## Hvorfor rækkefølgen ændres

En fuld ekstra 72-timers base/fine prøve koster omkring
`72*(3.8650736+8.9886096) = 925.47 s` før prior/startup. Kun 253.59 native-
udviklingssekunder er tilbage. Prefixfejlen må ikke ekstrapoleres lineært:
Bayes-normalisering afhænger af både kandidat, state og observation.

Den allerede prissatte produktionsgraf indeholder imidlertid én komplet
lineær base-TRAIN, dens selection og dens egen fine prior/TRAIN/selection.
Disse **samme obligatoriske resultater** produceres før COBYQA og genbruges
ved modelvalget. Der tilføjes ikke en separat hel-vindue udviklingsprøve,
og dens omkostning skjules ikke som en ny budgetetiket. Den hidtidige
valideringsreserve afvikles kun med det faktisk færdige valideringsarbejde.

## Fast rækkefølge

1. Læs den SHA-verificerede, råreviewede PC-receipt. Genbrug dens 72
   observationer, frosne B/U og lineære base/fine states time 0–26.
   Ingen workbook er nødvendig før final evaluering; ingen final-Y findes
   i denne pakke. Rå observationsprovenance har samme begrænsning som før.
2. Én **ny afgrænset nonlinear rate/præcisionsprøve**, ikke optimizer:
   fire stykker, logits `[.5,0,0,0,0,0]`, 24 prior + 3 TRAIN-timer,
   egne base/fine states. Den bruger den eksisterende native-rest, med
   60 s stopreserve. Ingen automatisk gentagelse ved failure/timeout.
   Den er repræsentativ lokal perturbation, ikke et worst-case-bevis.
3. Beregn den fulde jobgraf med maksimum af de faktisk målte rates og
   faktor 1.5. Afvis hvis H + hele grafen overskrider 36000 s. Gennemfør
   den obligatoriske lineære baseline som beskrevet ovenfor. Kræv alle
   guards, hver times state-gates samt total-scoregate for TRAIN og selection
   separat. Gem pre-update selection-momenter til senere modelvalg.
4. Først nu må betinget COBYQA starte. Én lineær start per familie,
   17/33 maxfev inklusive cached linear, 16/32 nye par, 49 i alt.
   Ved pause rekonstrueres kun optimizerens kontrolforløb med cached
   objective-kald i verificeret rækkefølge; færdige modeltimer genkøres ikke.
   Ingen fiktive uendelige objective-værdier. Konvergens kræver success og
   trust-radius status 0; cap/timeout er ufuldstændigt fit.
5. Alle tre retained familier får deres egne fine prior/TRAIN/selection
   og base selection. Alle uændrede gates gælder hele vinduer. Modelvalg
   bruger selection alene. Gap <= .1 eller base/fine rank reversal giver
   den på forhånd valgte enklere-modelregel uden forbedringsclaim.
6. Frys valget og begge grids' egne cutoff-states. Udarbejd den åbne uge
   for valgt model og lineær reference: 168 intervalmomenter, CDF/kvantiler
   ved 1/6/12/24/48/72/168 h, ingen assimilation. Gem fine-sammenligning.
7. Først efter frossen prognose må en særskilt PC-kommando læse final-Y.
   Evaluer de gemte forecasts og en separat prequential gren fra hver
   models cutoff-state. Final-data kan aldrig ændre valget eller open-grenen.

## Hvad prøverne kan og ikke kan dokumentere

For alle monotone kurver er værdierne i [-1,1]; det eksisterende positive
CN-bound for de faste fysiske parametre gælder hele familien. Antal celler,
substeps og højst fem konturer er fastlagt. Antal faktisk nødvendige
konturer og flydende-komma-arbejdet kan variere. Derfor er den målte
nonlinear prøve og faktor 1.5 et **planlægningsestimat**, ikke en garanti.
Runnerens fælles vægtede jobgraf og faktiske wall-clock-stop er bindende.
Intet argument her giver et uniformt præcisionsbevis for alle optimizerkald.
De endelige kurver kræver deres egne fulde fine replays før accept.

Den ændrede rækkefølge giver betinget søgeadmission efter trin 2–3;
`historical_fit_ready` er ikke lig med videnskabelig accept. Mislykket
konvergens eller finere kontrol stopper forløbet og bevarer evidensen.
Budgettet nulstilles aldrig. Et crash beholder hele invocation-reservationen;
automatisk refund er kun tilladt ved normal afslutning.

Slutugen er fortsat en udviklings-holdout med ukendt tidligere eksponering.
Ingen blind-H3-, fysisk kalibrerings-, global-optimum- eller parameter-
identifikationsclaim. F/G/WM er adskilte; de numeriske kerner ændres ikke.

## Crash og aflevering

Hvert native job gemmes atomisk med identitet, fuld log-state og checksum.
Et afsluttet job genbruges efter crash selv hvis et efterfølgende status-
snapshot mangler. Halvskrevne `.partial` filer er aldrig autoritative.
Kandidatregister, optimizerkald, selection og forecast-freeze er atomiske.
En mismatch i source, miljø, data, kurver eller log-continuation stopper.
En numerisk failure er terminal for det run; STOP/tidsstop kan genoptages
eksplicit under samme kilde og ledger. Der er ingen automatisk retry.

Implementeringsstatus, aktuelle checks og sidste publicerede checkpoint:
`F4_NIGHT_CHECKPOINT_20261007.md`. Ingen fuld historisk kørsel er dokumenteret
af selve denne designnote.
