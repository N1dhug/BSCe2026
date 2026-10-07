# F4: restarbejde og mulighed for afslutning 7. oktober 2026

Johan ønsker at optimere det resterende arbejde og om muligt afslutte F4 i dag.
Dette review er udført på den gendannede, pausede 1.076-job-pakke. Der er
ikke startet nye modelberegninger eller ændret forsøgets budget/acceptkrav.

## Hvad der faktisk er gjort

- Alle 1.076 jobfiler er sammenholdt med råreviewets gemte SHA-256-hashes.
  De gemte køretider og konturdiagnostikker er samlet uden numerisk replay.
- Den frosne SciPy 1.17.0-optimizer genskaber præcis alle 12 gemte kald.
  Kontrollen stopper før første nye objective og giver netop den afbrudte
  kandidat `67d94e70596f393e163ae47fa5c4d858d82ebc751e5f090b6b5921d7587308b0`.
  Dens 34 gemte timer skal genbruges; kun de sidste 38 TRAIN-timer mangler.
- Initialdesignet for fire stykker er kontrolleret separat uden modelkald.
  Første positive koordinatkandidat er den allerede gemte PROBE. Dens
  24 prior- og 3 TRAIN-timer skal også trækkes fra restgrafen. Det sparer
  omkring et halvt minut ved hidtidige gennemsnit, ikke flere timer.
- De eksisterende lokale wheels for numba 0.62.1 og llvmlite 0.45.1 er
  genfundet og installeret. Første forsøg via standardindekset fandt ingen
  pakker og installerede intet. Det gendannede fulde runtime-manifest matcher
  nu det frosne miljø, og samtlige 69 numeriske kildehashes er uændrede.
- Én afgrænset cached-optimizerkontrol tog 0.74 s. Ingen fuld suite eller
  native propagation er kørt. Reviewet er ikke optimizerkonvergens.

## Tidsgrundlag

De 11 komplette ikke-lineære basekandidater tog 200–273 s per kandidat
(24 prior + 48 TRAIN), før ydre controller/eksport. Base-observed har
610 målte jobs med gennemsnit 4.813 s; fine-observed har 72 med gennemsnit
10.944 s. Fine-evidensen er hovedsageligt lineær; fire-stykkers hele TRAIN
og slutugens rates er endnu ukendte.

| Resterende modelarbejde ved fuld kandidatramme | Målte gennemsnit | Gennemsnit × 1,5 | Målte maksimumrater × 1,5 |
|---|---:|---:|---:|
| Søgning, inklusive 300 s ydre reserve | 2,52 h | 3,74 h | 7,90 h |
| Validering, forecast og finale, inklusive 360 s reserve | 2,33 h | 3,45 h | 6,11 h |
| I alt fremadrettet | **4,85 h** | **7,19 h** | **14,01 h** |

Færdige jobs er trukket fra præcis én gang. Der regnes med hele den
resterende 49-par-ramme, to forskellige forecastmodeller, komplet observeret
slutuge og alle deklarerede base/fine-kontroller. Tidlig konvergens eller
valg af lineær reference kan reducere arbejdet. Højere fremtidige rates,
fornyet JIT/startup og pauser kan øge det. Ingen af kolonnerne er et
konvergensbevis eller et statistisk tidsinterval.

**5–7½ timers modelkørsel er et brugbart, betinget planlægningsscenarie.**
Det er et bedre empirisk overslag, ikke dokumentation for en ny kode-speedup.
14,01 h er fortsat det konservative fuld-ramme-scenarie. De nye gennemsnit
sænker ikke automatisk den beskyttede valideringsreserve.

Opsætning, gennemgang, slutdataoverførsel og I/J-afslutning er særskilt
arbejde. Det tidligere 2–4 h-skøn var groft; meget dokumentationsarbejde kan
udføres under modelkørslen. Færdig modelkørsel er ikke i sig selv F4-afslutning.

## Korteste forsvarlige udførelse

1. Bevar numerisk kilde, data, 1/2/4-familier og de eksisterende gates.
   Genoptag den afbrudte to-stykkers kandidat og derefter samme COBYQA-forløb.
2. Brug de gemte lineære streams og den fire-stykkers PROBE. Ingen ekstra
   ratepilot, helvinduesprøve, gammelt panel eller genberegning af færdige jobs.
3. Brug højst **600 s per invocation**, med eksisterende stop-/eksportreserve.
   Luk ledgeren og publicér en ny checkpoint-delta efter hver blok. Før næste
   blok verificeres status, budget og kilde; numerisk fejl er ikke auto-retry.
   En ny uafklaret crashreservation begrænses dermed til højst 600 s i stedet
   for 30.000 s. Dette tilbageskriver ikke den gamle reservation.
4. Bevar rækkefølgen fit → egen fine/selection-validering → modelvalg →
   frossen åben uge. Under kørslen klargøres J's sammenfatning og begrænsninger
   fra allerede gemt evidens, og den eksisterende I-handoff genbruges.
5. Først efter den frosne uge eksporteres slutobservationerne fra Johans PC
   med den eksisterende databro. PC-opgaven er dataeksport, ikke nyt fit.
6. Gennemfør den deklarerede finale og vurder resultaterne. Afslut samlet
   I/J-review, reproduktionsvejledning og status. Drop ikke en manglende
   familie eller kald et cap-stop for konvergens for at nå dagsfristen.

## Hvorfor der ikke er omskrevet en solver

Kørslen bruger allerede 8 frekvensworkers og BLAS=1; miljøet har 8 CPU'ers
kvote. Kandidatjob kan derfor ikke endnu en gang divideres med 8. Kun
0,018/0,033 s per base/fine-observed job ligger uden for de gemte konturfaser.
Konturfaserne inkluderer workertransport, propagation og inversion; tallene
beviser ikke, at transporten isoleret set er gratis, men flytning af ydre
Python-kode alene forklarer ikke en stor besparelse.

Genbrug er allerede indbygget i den numeriske jobstore. Nye startpunkter,
konturrækkefølger, tolerancer eller solvere ville kræve ny metode-/numerik-
evidens. Der er ingen målt sikker flerfoldig speedup at love i dag.
F4-G2b/G2c er droppet og genstartes ikke; G/WM er egne spor.

## Konkret budgetvalg, endnu ikke godkendt

Den oprindelige grænse er fortsat 36.000 s. Efter denne gennemgangs
konservative 30 s charge er totalen **32.941,549305 s**, herunder den
gamle uafklarede 30.000 s reservation. Resten er **3.058,450695 s**.
Native-development er **833,307761/900 s**. Alle gamle entries bevares.

Forslag: hæv det samlede loft fra **10 til 18 timer**, med stop senest
**7. oktober kl. 23:55 Europe/Copenhagen**, og 600 s blokke. Det giver
31.858,45 s (8,85 h) bogførbar rest før nye opstarts-/kontrollercharges.
Den konservative valideringsreserve på ca. 6,11 h bevares; der er indledningsvis
ca. 2,72 h uden for reserven og en særskilt 60 s søgestopmargin.
Det finansierer et afgrænset konvergensforsøg, ikke hele 14,01 h-scenariet.
Clock-deadline og budgetloft gælder begge; det først ramte stopper arbejdet.

Et nyt loft kræver Johans beslutning efter den tidligere udtrykkelige
10-timersgrænse. Det skal derefter registreres i en særskilt, versioneret
budget-/controllerrevision med alle tidligere charges og kildeidentiteter
bevaret. Det gamle hardcodede 36.000 s-script må ikke blot køres med en
ændret ledger. Ingen budgetrevision, STOP-fjernelse eller modelstart er udført.

## Afslutningens faglige scope

Det aktive historiske studie er det vedtagne betingede 48/24/168 h-forløb.
Det afslutter ikke automatisk et tidligere syvdages-fit, blind H3,
kausal prisidentifikation eller generel månedskalibrering. J skal vise
den gennemførte sammenligning og disse begrænsninger. Historisk f/g-fit,
modelvalg, egen numerik og prognoseevaluering er fortsat obligatoriske.
En H→I-eksport er ikke i sig selv en ny historisk kontroloptimering.

Rågrundlag i `files/output/bm_1/runs/estimation`:
`f4_h2_remaining_work_review_20261007.json`,
`f4_h2_cached_optimizer_resume_review_20261007.json`,
`f4_h2_runtime_restore_review_20261007.json` og
`f4_h2_remaining_work_campaign_20261007.json`.
