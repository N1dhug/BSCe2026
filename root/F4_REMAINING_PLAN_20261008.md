# Aktuel restplan for hele F4 — 8. oktober 2026

Dette er det samlede aktuelle overblik over resterende F4-arbejde. root/F4.md
bevarer den faglige plan og historikken; tidligere status-/timeafsnit er ikke
nye kørselsinstruktioner. G-plan/G7/WM er separate spor. Droppede F4-G2b/G2c
kommer ikke tilbage i denne plan. F5/F6 og direkte BM/WM-sammenligning ligger
uden for F4-afslutningen.

## Én rækkefølge

"Kontrolleret fortsættelse" og "fortsættelsesadapteren til H" er samme leverance:
en ny execution med kontrolleret adoption af immutable gamle checkpoints.
Planlægning af H→I→J er overblikket omkring denne leverance, ikke et alternativ.

| Del | Allerede færdigt | Resterende konkret leverance |
|---|---|---|
| A–F | Implementeret, undersøgt og råreviewet; F afsluttet/merged | Genbrug evidens. Ingen genkørsel for at skrive status. |
| F4-G | Afsluttet kerneevidens og G2a-v2 | Bevar omfang og negative resultater; ingen G2b/G2c. |
| H1 | Cases, betingede fits, valgte kurvers numerik og råreview | Brug som udviklingsevidens; ikke historisk accept. |
| H2 | Frossen datakontrakt; lineær og to-stykkers egen validering; 3655 gamle jobs; 27 komplette firestykkekald; bankrettelse består egen base/fine | Fortsæt firestykkefit med 25/49-caps; fuld egen TRAIN48/selection24; sammenlign 1/2/4 og frys kurver. |
| H3 / historisk slutstudie | Protokol og final-data-gate implementeret | Frys 168h åben prognose før slut-Y, eksportér slutdata og gennemfør prequential diagnose. Rapportér betingede nominelle parametre og ukendt tidligere slutugeeksponering. |
| I | MAP-posterior/F3-interface, alle horisonter i ét forløb og gammel lille demo | Tilslut faktisk H-handoff, verificér egen origin og konservativ transfer; selvstændig forecast-evidens og F3-kontrolsammenligning. |
| J | Bærende forsøgsnoter og mange råresultater | Saml settings, kurver, resultater, figurer, runtime og begrænsninger; afsluttende reproduktions-/acceptreview. |

## Næste udførbare trin er klart

Nyt prepared run:
files/output/bm_1/runs/estimation/f4_h2_contour_continuation_20261008.
Adapter: src/exe/run_model/bm_1_h2_continuation.py.
Launcher: root/f4/run_h2_continuation.py.

Alle 3655 gamle råjobs er checksum-/tokenverificeret og indekseret; 69 gamle
numeriske kilder er uændrede. Gamle jobs bliver kun adopteret ved deres præcise
originale token, filhash, content-checksum og prior bytes/dtype. De beholder
original identitet; ny anmodning får kun en separat adoption receipt. Nye jobs
får den nye manifestidentitet. Gamle fejl og alle budgetreservationer bevares.
Den oprindelige fem-konturprefix er uændret; de fire ekstra konturer ændrer
repræsentationens konditionering, ikke model, støj eller likelihoodmål.

Den gamle optimizertrace ligger fortsat hos forgængeren. Nye objectivekald
skrives separat; pinned SciPy/COBYQA replayer 27 gamle firestykkekald præcist,
aldrig en genstart fra bedste kandidat. Det gamle konvergerede to-stykkers
resultat genbruges med sin oprindelige identitet. Alle parrede guards bevares.

Prepared run har endnu ingen produktionsjobs. To preparation-manifest/source-
revisioner er bevaret før endelig frysning; ingen modelkørsel fandt sted under
revisionerne. En gentagen kontrolrapport blev afvist af no-overwrite-vagten;
gammelt bestået resultat og den ekstra charge bevares. Dette var en I/O-
afvisning, ikke en numerical model failure. Kun kontroller for konkrete
identity/prior/namespace/optimizer/budgetrisici er kørt, med nul propagationer.

## Beregningstid og konkret beslutning

Alle oprindelige charges ligger stadig urørt i gammel campaign. Nye read-only
preparation-/controlcharges står i separat preparation_ledger og skal indgå
som uændret prefix ved ny campaign-v3. Ingen crashreservation refunderes.

Efter preparation/kontroller/prisning: samlet H40481.86150/64800 s,
native951.26557/1080 s; ca6h45m rest. De konservative rater er genpriset med
alle kendte jobmaksima samt den nye ni-konturbanks faktiske basistime og
fine observed-phase. Faktor1.5 bruges fortsat én gang; ikke division med8.

| Budgetpost | Aktuelt scenarie |
|---|---:|
| Beskyttet resterende H-validering/forecast/finale inkl. priced hjælpe- og stopreserve | 20955.85668s ≈5h49m |
| Resterende fit efter yderligere60s stopmargin | 3302.28182s ≈55m |
| Alle22 mulige resterende kald ved gamle gennemsnit | 4319.77732s ≈72m |
| Samme gennemsnit med faktor1.5 | 6479.66598s ≈108m |
| Alle22 kald ved beskyttede rater med faktor1.5 | 17803.26861s ≈4h57m |

Derfor er et fuldt 49-kaldsscenarie ikke admitted. Forslaget er kun et bounded
konvergensforsøg under uændret18h-loft, med fuld restvalidering beskyttet.
Nær budgetgrænsen stopper søgningen før næste job. Timeout gemmer delarbejde;
numerical failure eller maxfev uden convergence er terminalt og afslutter
ikke H/F4. Ni-konturrettelsen garanterer ikke alle fremtidige kandidater.

Konkret nyt deadlineforslag: 8.oktober2026 kl.08:00 Europe/Copenhagen.
Dette øger ikke18h-campaign-loftet eller1080s-native-loftet. Forslaget er
root/f4/h2_continuation_deadline_proposal_20261008.json og er IKKE godkendt.
Den tidligere23:55-frist er ikke omgået. Launcher afviser før reservation,
når den separate nye deadlinebeslutning ikke er godkendt.

Efter en sådan godkendelse: én600s-blok ad gangen,60s stop/eksportmargin,
lukket ledger og durable checkpoint før næste blok. Ingen selvstartende
baggrundsloop eller automatisk retry efter fejl. Kildecommit fryses før launch.
Slut-Y må først eksporteres efter forecast_frozen.npz. Det originale demand-
workbook findes ikke i dette gendannede workspace; slutdataeksport kan derfor
kræve Johans oprindelige fil eller en separat PC-eksport, når freeze findes.
Det blokerer ikke fit/selection/open forecast.

## I og J efter H

1. Genbrug conditional_handoff.npz og as_forecast_origin; eksporter valgt f/g,
   fysisk betingelse, origin og egen posterior. Ingen dagligt genfit.
2. Verificér den faktiske H→I-grænseflade på én frossen origin. State/demand/Y
   og interval-/punktkontrakter skal mærkes korrekt; ingen nye målinger under
   hypotetisk forward eller policyoptimering.
3. Registrér et afgrænset selvstændigt I-forecastforsøg før kørsel. Horisonter
   1/6/12/24/48/72/168h kommer fra ét forløb. Aftal inputs/baselines/scores,
   numerical reference og faktisk runtimebudget. Den nuværende H-uge er én
   betinget udviklingsuge, ikke populationskalibrering. Ved populationsclaims
   gælder planens mindst200 uafhængige sekvenser fortsat; ingen nedskæring uden
   en eksplicit ny afgrænsning. Dette I-forsøg er endnu ikke runtime-admitted.
4. Genbrug eksisterende F3-evaluator med de faktiske frosne H-kurver. Kendte/
   estimerede parametre og policies sammenlignes ved samme informationsniveau,
   inputs og constraints. Lokal convergence er ikke global optimalitet.
5. J samler den bærende evidens og en kort kravmatrix. Model-fit, numerisk
   præcision, betinget forecast og generelle populationsclaims holdes adskilt.
   F4 afsluttes ved opfyldte relevante gates eller dokumenterede begrænsninger,
   aldrig ved ommærkning af en demonstration til fuld forecastaccept.

Det tidligere chatskøn12–24 aktive timer er et foreløbigt arbejdsbudget:
H4–8h, I5–10h, J3–6h, eksklusive modelventetid. Det er ikke målt fra H-ledgeren
og ikke en promise om at hele F4 er færdig næste morgen. I's konkrete workloads
og usikkerheden om firestykkefittets termination skal fastlægges for et bedre
samlet estimat. De gamle40–55h fra3.oktober er historik, ikke nyt restbudget.
