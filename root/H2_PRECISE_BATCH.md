# F4-H2: afgrænset præcis søgning og lokal runtimepilot

**Aktivt næste trin, efter Astra-review 6. oktober:**
[H2_METHOD_REDESIGN.md](H2_METHOD_REDESIGN.md) vælger MED separat og de tre
februaruger samt en kompileret, algebraisk ækvivalent positiv state-solve.
Den senere adaptive søgning bruger frekvensparallelitet og kræver egen samlet
budgetadmission. Det eksisterende 17-kandidatdesign og kommandoerne her er
arkiveret pilotgrundlag; de må ikke startes automatisk på den valgte uge.
Population/perioder er valgt; øvrig datakontrakt, blindhed, implementering,
hastighed og H2/H3-accept er stadig åbne. Ingen nye modelkørsler i reviewet.

Grundlaget er implementeret 6. oktober 2026. Dette er F's stokastiske
intervalmodel; G's fit/data/modelvalg og WM ændres ikke. De negative
surrogater i H2_SEARCH_FEASIBILITY.md er fortsat afvist. Den eksisterende
terminalvise tilt-kernel genbruges uændret. Numerisk fit-readiness, H2/H3 og
videnskabelig accept er stadig falske; en bestået runtimepilot ændrer ikke dette.

## Fuld PC-råreview færdig, 6. oktober 2026

ZIP'en er nu modtaget og gennemgået uden solver-replay. Manifestets identitet,
alle otte kandidat-identiteter/checksums, kronologi/missing-masker, likelihood-
summer, afsluttende log-state og det oprindelige budgetestimat er kontrolleret.
64 timer er gemt: 56 Bayes-opdateringer og otte missing-prædiktioner uden
likelihood. Alle otte checkpoints har status `prefix_complete`; hele døgnet
og de øvrige ni kandidater er ikke beregnet af denne pilot.

| Gemt diagnostik | Maksimum | Uændret grænse |
|---|---:|---:|
| Conditional guard | 2.3256e-9 | 1e-8 |
| Overlap-logforskel | 1.7092e-8 | 1e-7 |
| Massefejl | 2.1760e-14 | 1e-10 |
| Afsluttende log-normaliseringsfejl | 1.3323e-13 | 1e-10 |

55 observerede opdateringer brugte tre contours, én brugte to. Alle sluttede
med nul uafklarede celler. Negative tætheder i en enkelt contours ubrugelige
Fourier-kandidater er ikke klippet ind i posterioren; den eksisterende kernel
vælger kun resolved/positive celler og fortsætter banken efter behov. Alle
otte sluttilstande har 1921 endelige log-masser, ned til **-1684.14**, og deres
state-mean rekonstrueres præcist fra den gemte log-state. Guards er fortsat
heuristiske roundoffkontroller; dette er ingen ny kontinuumsfejlgaranti.

De 49 rå kildehashes afviger fra Linux-checkoutet, men hver eneste matcher
præcist Git-indholdet med LF→CRLF, ligesom den gemte input-fixture. De lokale
Git-blobs er desuden kontrolleret mod den deklarerede `d6e805e`-commits tree.
Dette dokumenterer Windows-proveniens ved råreview; det ændrer ikke en
originalhash, kildefil, environment-guard eller execution-/resumelås.

Originale bytes ligger i `f4_h2_precise_pc_pilot_20261006/windows_review_20261006.zip`;
det aktuelle resultat er `raw_audit.json` i samme mappe. Den tidligere
`summary_audit.json` beskriver tidspunktet før ZIP-modtagelsen, ikke den
aktuelle råreviewstatus. Den udførte læse-/auditkode er
`root/f4/audit_h2_precise_pc.py`; den starter ingen likelihood-replay og
bogfører ingen ny beregning i H-ledgeren.

Kontrakten i ZIP'en er stadig den ufuldstændige template: 32 null-/tomme
deklarationsfelter, blandt andet population/målerkolonner, alle splitdatoer,
eksponeringsaudit, prisområde/skala, baselinefil/origin, fysiske parametre/
støj og Beta-prior. De udfyldte timeenheder/timezone er arbejdsantagelser.
Næste beslutning er F's faktiske population og TRAIN-/valideringsperioder,
med kendt tidligere dataeksponering; derefter fastlægges de øvrige inputs
og et søgedesign inden for loftet. G's januarvalg overtages ikke automatisk.
Der er ingen ny lokal numerisk kørsel til Johan på baggrund af denne ZIP.

## Windows-resultat og næste trin, 6. oktober 2026

Johans `pilot_report.json` har status `completed`, med otte færdige timer
for hver af otte kandidater, 8 workers/BLAS=1 og wall-time **556.53 s**.
Den uploadede fælles ledger er **1024.63 s**; summen af den oprindelige
425.89-sekunders baseline og alle fire bogførte poster stemmer præcist.
Der resterer højst **27775.37 s (7 timer 43 min)** til piloter/søgning, når
7200 s holdes fri og det samlede loft er 36000 s.

Rapportens `total_train_units=24` er afgørende: dens grønne budgetflag og
**5.50 timers** prognose gælder kun det syntetiske døgn. Det er ikke en
historisk pilot og kan ikke åbne runnerens `search`. Den dyreste målte
observations-time er 84.11975 s. Samme planformel for 17 kandidater med
8 workers er
\(T_{search}=1.5\cdot3\cdot(N\cdot84.11975+556.53255)\) sekunder.
Her er N antallet af replay-units, inklusive en eventuel prior-lead.
Samlet H-planbudget er 1024.63112 + T_search + 7200 sekunder.

| Regneeksempel, ingen prior-lead | Søgeestimat | H inkl. brugt tid og foreløbig 2-timers reserve |
|---|---:|---:|
| 24 timer | 3.22 timer | 5.50 timer |
| 48 timer | 5.74 timer | 8.03 timer |
| 72 timer | 8.27 timer | 10.55 timer |
| 168 timer | 18.36 timer | 20.65 timer |

Dette er forsigtig ekstrapolation fra syntetiske PC-tider, ikke målt
historisk throughput eller et løfte om at afslutte H. Ukendte kandidater,
senere tails, prior-lead og det faktiske kontrol-/valideringsarbejde kan
ændre omkostningen. Den gamle 17-kandidater/syv-dage-illustration er overhalet;
den giver ikke grundlag for admission. TRAIN forkortes ikke automatisk for
at passe til tabellen, og obligatoriske kurvefamilier/acceptkrav bevares.

De to originale uploadfiler og ren budgetaritmetik er arkiveret i
`files/output/bm_1/runs/estimation/f4_h2_precise_pc_pilot_20261006/`.
Det oprindelige næste trin var at samle fuld PC-manifest, checkpoints,
per-time-guards/contours og posterior samt F-konfigurationen. ZIP'en er nu
modtaget og rå-reviewet, se ovenfor. Den følgende indsamlingskommando er
allerede udført; den kræver ingen ny likelihood eller historisk workbook:

```powershell
$ErrorActionPreference = "Stop"
$base = "files/output/bm_1/runs/estimation"
$items = @(
    "$base/h2_precise_development_pc_v1",
    "$base/f4_h2_remaining_budget.json",
    "$base/h2_precise_contract_pc_v1.json"
)
foreach ($item in $items) {
    if (-not (Test-Path -LiteralPath $item)) { throw "Mangler: $item" }
}
$zip = "$base/h2_precise_review_$(Get-Date -Format yyyyMMdd_HHmmss).zip"
Compress-Archive -LiteralPath $items -DestinationPath $zip
Write-Host "Send denne fil: $zip"
```

Ingen pull/branchskift eller ny pilot er nødvendig for at samle filerne.
Kontrakten er stadig den ufuldstændige template; åbne felter forbliver
åbne. Efter det nu færdige råreview fastlægges faktisk population, TRAIN/split/eksponering,
baseline/prior-lead, skala og fysisk/støj-betingelse, før søgedesignet revideres.
Ved flytning PC/chat videreføres den seneste fælles ledger. Den uændrede
468.09-sekunders seed er kun det oprindelige startpunkt, ikke nyt restbudget.
Ved denne statusopdatering er ingen numeriske kilder ændret og ingen tests,
replays eller historiske fits kørt.

## Matematik og afgrænsning

Hver måling er det normaliserede timegennemsnit
\(Y_i=\Delta_i^{-1}\int_{s_i}^{t_i}D(v,X_v)\,dv+\epsilon_i\), med
\(\epsilon_i\sim N(0,\sigma_y^2)\). Den eksisterende terminal-tilt-metode
beregner fælles endpoint/observation-masser og normaliserer kun ved Bayes.
Hele log-posterioren, også endelige værdier under almindelig floats rækkevidde,
gemmes og bruges i næste interval. Kun ægte nulmasse bliver JSON-null/-infinity.
Ingen floors, clipping eller daglig genstart indføres. Manglende målinger og
hele perioden fra den erklærede prior-origin til TRAIN får prædiktion uden
observationslikelihood; dage inden for samme stream paralleliseres aldrig.

For hver kurve bestemmes faldstørrelserne som
\(d_j=2\exp(\theta_j)/\sum_l\exp(\theta_l)\), med sidste logit nul, og
knudeværdierne som \(v_j=1-\sum_{l\le j}d_l\). Dette genbruger H's
eksisterende kurvegeometri: kontinuitet, faldende kurver og endpoints 1/-1.
To stykker giver to uafhængige parametre for f/g tilsammen; fire stykker giver seks.

Den første batch er **17 på forhånd fastlagte kandidater**: én fælles lineær
reference, +/-0.5 i hver af to logit-koordinater og +/-0.5 i hver af seks.
Det er en lokal koordinatprøve fra det lineære udgangspunkt, ikke en færdig
optimering eller et stort parametergrid. Alle kandidater bruger samme fysiske
parametre/støj/prior og én fælles f/g over hele TRAIN. En eventuel straf er
den eksisterende deklarerede kvadratiske knudeafstand til lineære kurver.
Rapporten kan gemme bedst evaluerede TRAIN-kandidat per familie, men vælger
ikke modelfamilie, erklærer ikke konvergens og eksporterer ingen accepteret model.
Valgte kurvers numerik, MODEL_SELECTION og frosset H3 mangler stadig.

Den foreløbige præcise indstilling er 1921 celler, h=1/24576 dag, 64 frekvenser,
periode 2 og uændrede contour-guards. Den er lokalt kontrolleret på den gemte
fixture mod 2561/h=1/49152, ikke universelt valideret på alle kandidater/historik.
Ved en fejlet guard stoppes hele batchen; kandidaten droppes ikke og omkøres
ikke automatisk med andre settings. Frys B/U fra de eksisterende timeholdte
udviklings-/historiske adaptere. Vilkårlige kontinuerlige inputs er ikke omfattet.

## Budget, parallelitet og genoptagelse

Brug `src.exe.run_model.bm_1_h2_precise`, ikke den gamle `curves history-fit`.
8 workers og BLAS=1 er standard. Hver proces ejer én kandidats hele stream,
skriver en atomisk gzip-checkpoint efter hver færdig time og gemmer rå evidens,
posterior-mean, contour-brug, probability guards og faktisk log-state.
Kandidat/input-identitet og checkpoint-checksum kontrolleres inden genbrug.
Relevante F-kilder, originalinput-digests og Python/NumPy/SciPy/pandas/platform
skal matche. Uvedkommende G-/WM-filer er ikke med i F's kildehashliste.

Alle runs bruger samme `f4_h2_remaining_budget.json`. Udgangspunktet medregner
den tidligere prøve og kontrolreserve, 425.89 s; en gemt seed medregner også
denne leverances udførte kontroller. En invocation reserveres og bogføres før
processtart. Kun ordentlig afslutning refunderer ubrugt reservation; et crash
beholder reservationen. Parent og workers har tidswatchdogs, så en mistet
parent ikke lader en worker fortsætte ubundet. Normal timeout/Ctrl+C beholder
sidste færdige time og giver en pause, ikke accept. En `STOP`-fil i run-mappen
pauser; fjern den kun før ønsket resume. Slet aldrig ledger eller checkpoints
for at få et nyt budget. Nye mapper bruger samme campaign, også mellem PC/chat.

Mål 4–6 timer, samlet loft 10 timer. Søgningen får ikke lov at bruge den
reserverede 2-timers kontrol-/valideringspulje. Denne pulje er foreløbig og
skal måles imod den faktiske valideringsplan; den er ikke dokumentation for,
at H kan afsluttes inden loftet. Efterfølgende H-arbejde skal fortsætte samme
regnskab, ikke starte et nyt. Ingen 73-kandidat-kørsel eller automatisk udvidelse.

## Den allerede gennemførte udviklingspilot

Dette afsnit dokumenterer den oprindelige kommando. Piloten er siden
rapporteret komplet; næste handling er filindsamlingen ovenfor, ikke en
ny invocation eller automatisk resume.

Kør fra en ren repo-rod, uden en anden modelkørsel i dette checkout:

```powershell
& ./root/f4/run_h2_precise_pilot.ps1
```

Det opretter en ufuldstændig F-datakontrakt og måler de første otte kandidaters
første otte timer fra det gemte syntetiske døgn, sigma_y=.04. Forløbet indeholder
en prisændring og en manglende måling. Højst 15 minutter **per invocation**;
alle invocations tæller i samme samlede H-ledger. Råresultater gemmes også ved
pause/fejl. Den syntetiske pilot måler Windows-processer, contention og
contour-adfærd; den fastlægger ingen historisk population, TRAIN-periode,
historisk throughput, præcision eller H-accept.

De oprindeligt bestilte filer var `h2_precise_development_pc_v1/pilot_report.json` og
`f4_h2_remaining_budget.json`. Hvis status er fejlet, send også den relevante
log fra run-mappens `logs/`. Ved timeout er checkpointet bevaret; genoptagelse
er mulig med samme source/inputs og `-Resume`, men kræver ikke automatisk en ny
15-minutters prøve, hvis det allerede er tydeligt, at planen er for dyr.

```powershell
& ./root/f4/run_h2_precise_pilot.ps1 -Resume
```

## Historisk vej, først efter F-datakontrakten er udfyldt

Population, enheder, timezone, price proxy, skala, prior-origin, fysisk/støj-
betingelse, tidligere baseline og eksponerings-/splitkontrakt bliver ikke
gættet. G's januarfiler/-valg er ikke automatisk F's kontrakt. Den nye template
bevarer disse åbne felter og den eksisterende loader, som afviser dem før
observationsværdier læses. Baseline-CSV kræver `timestamp,baseline` og komplet
timehistorik fra origin til TRAIN-end. FINAL_HOLDOUT-værdier læses ikke.

Når kontrakten er dokumenteret og frosset:

```powershell
python -B -m src.exe.run_model.bm_1_h2_precise prepare --config files/output/bm_1/runs/estimation/h2_precise_contract_pc_v1.json --run-dir files/output/bm_1/runs/estimation/h2_precise_history_pc_v1
python -B -m src.exe.run_model.bm_1_h2_precise pilot --run-dir files/output/bm_1/runs/estimation/h2_precise_history_pc_v1 --workers 8 --seconds 900 --resume
```

Den historiske pilot begynder ved den deklarerede prior og omfatter de første
otte TRAIN-intervaller. En lang prior-lead skjules eller forkortes ikke. Kun
en komplet historisk pilot på samme workerantal kan åbne search-kommandoen.
Planestimatet bruger den dyreste målte observations-time, hele TRAIN-længden,
tre batchbølger og faktor 1.5 samt målt startup/storage; det er stadig et
estimat, ikke en garanti. Ukendte senere tails/kandidater kan blive dyrere.
Hele H-regnskabet plus planestimat/reserve skal være under loftet ved admission.

```powershell
python -B -m src.exe.run_model.bm_1_h2_precise search --run-dir files/output/bm_1/runs/estimation/h2_precise_history_pc_v1 --workers 8 --seconds 14400 --resume
```

Dette er en senere kommando, ikke bestillingen nu. Den genbruger pilotens
færdige timer for de samme kandidater og fortsætter dem; de resterende ni
starter fra deres egen prior. Fire timer er invocation-loft, ikke målt fit-tid.
Ufuldstændige budgetstops forbliver ufuldstændige. Ingen modelvalg efter TRAIN.

## Udførte kontroller

To fokuserede nye tests består (0.166 s): eksakt state-uafhængig Gaussian-
likelihood med sub-float log-state, midnight/gaps/missing og identisk resume;
samt cross-candidate/checksum-afvisning og bevaret crash-reservation med låst
valideringsreserve. Ingen fuld suite, nye numeriske paneler eller historisk fit.
Procesintegration og faktisk subprocess-resume er udført: 35.04 s og 5.17 s,
med to færdige kandidat-timer og to genbrugte timer. Timeout bevarede den
sidste færdige time. Lineær første-time-score, mean og hele log-posterioren
er bitvis identiske med den allerede gemte præcise reference; ingen gammel
reference blev genberegnet. Den daværende Linux-procespilot var ufuldstændig,
og ingen historiske værdier blev læst. Den var ikke Windows-runtime eller
fuld pilotaccept; det senere Windows-resultat står øverst.
Rådata: `f4_h2_precise_adapter_check_20261006.json.gz`; den indeholder også
den oprindelige kontrol-CLI før en afsluttende ændring af ETA-vægte og
afvisning af fejlede pilotrapporter. Numerisk adapter/kernel er uændret.
Den publicerede `f4_h2_budget_seed_20261006.json` starter PC'ens H-ledger ved
**468.09 s** inklusive den tidligere prøve og en konservativ 2 s kontrolpost.
Alle senere PC-piloter tilføjes dette samme budget, ikke et nyt nulpunkt.
