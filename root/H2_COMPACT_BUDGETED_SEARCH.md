# F4-H2: tidsbegrænset søgeforsøg efter fuld lineær reference

7. oktober 2026. Dette er en eksplicit **admission-revision** af den første
etapeplan. Den oprindelige komplette 49-kandidatgraf åbnes ikke: dens nye
konservative pris er **52597.14 s = 14.61 h**, inklusive allerede brugt tid og
med de færdige lineære jobs trukket fra én gang. Johans 36000 s loft bevares.

## Ny evidens

Den fulde lineære reference (24 prior + 48 TRAIN + 24 selection) består
egne base/fine streams. TRAIN-total-scorefejl .0055541053; selection
.0004541615; maksimal state-meanfejl .0000297841 og L1 .0010493139.
Alle mass/conditional/overlap-gates består. De sidste 69 observerede timer
per grid og selection-momenter er nye; PC-prøvens første 27 timer er genbrugt.
Ingen optimizer er startet. Råauditor: `f4/audit_h2_linear_window.py`.

Senere observerede timer koster op til 8.30102 s på base og 18.13960 s på fine.
Prefixens rates er derfor utilstrækkelige til det gamle fulde worst-case-
overslag. Hele grafen skal dokumentere denne negative konklusion; ingen
udeladelse af langsomme timer, toleranceløsning eller skjult budgetforøgelse.

## Det finansierede forsøg

Den ændrede admission giver alene tilladelse til et **afgrænset forsøg på at
opnå konvergens**, ikke et løfte om at finansiere alle 49 mulige par.
49 har hele tiden været et maksimum, ikke et konvergensbevis. Bevar SciPy
1.17.0 COBYQA, én lineær start, 17/33 maxfev, 16/32 nye par, radius .5→.02,
regularisering 2, alle tre familier og hele det frosne 48/24/168 h studie.

Prissæt først AL resterende validering med de nye målte maksimale rates og
faktor 1.5, inklusive finale selected+linear, momenter, distributionspunkter,
eksport og stopreserve. Træk kun råverificeret allerede udført arbejde fra.
Denne reserve er **19960.314316 s** før yderligere validering afsluttes.
Ved H=2848.002755 s er højst **13131.682929 s** tilgængelig for ny søgning
ud over reserven og en særskilt 60 s søgestopmargin. Senere review/startup
debit trækkes også fra; dette tal er ikke en ny uafhængig tidskonto.

Før hvert nyt søgejob kontrolleres

`faktisk H-forbrug + resterende prissat validering + stopmargin + næste-job-allowance < 36000`.

Parent/worker-watchdogs og den eksisterende fælles reservation er bindende.
Reserve frigives kun efter færdig råkontrolleret validering af en fysisk
kurve; identiske kurver kan ikke frigive samme beløb flere gange. Imports,
optimizer, I/O, fejl og genoptagelser debiteres som før. 900 s native-
udviklingsrammen nulstilles eller udvides ikke.

Hvis begge nonlinear fits faktisk når deres erklærede trust-radius-stop,
fortsætter deres egne fine/selection-gates og det frosne modelvalg. Ved
tids-, kandidat- eller numerisk stop bevares alle råresultater, men fit er
ufuldstændigt. Ingen familiefjernelse, ekstra start, højere cap eller
automatisk genkørsel. En senere positiv konklusion kræver de samme gates
som før. Det oprindelige 14.61 h worst-case-estimat markeres fortsat afvist.

Dette er et ressourcebegrænset forsøgsdesign. Maksimumrater fra de observerede
kurver er heller ikke en uniform garanti for alle fremtidige kandidater;
derfor kan et budgetstop forekomme. Den hårde fælles 10-timersgrænse er
kontrollerbar, mens konvergens og videnskabelig accept aldrig kan loves.

## Kilde og genbrug

Den eksisterende numeriske manifestidentitet og dens 69 kildefiler fra
`e115242de79756a4a7163da28593f2452f17bcc6` ændres ikke. Alle gemte modeljobs
bevarer deres oprindelige identitet, bytes og egne fulde logstates.
Den nye ydre controller `f4/run_h2_budgeted_search.py` genbruger de samme
Engine-/COBYQA-/selection-/forecast-funktioner. Dens særskilte frosne
execution-manifest linker den oprindelige numeriske manifestidentitet,
controller/protokol/source-hashes og den nye budgetbeslutning. Der sker
ingen ommærkning eller rekonstruktion af gamle modelstates.

Det gamle `run_h2_compact.py run` må ikke genoptages efter denne revision.
Kun den nye controller anvender den højere valideringsreserve. Den tidligere
STOP-markør bevares som `STOP_before_budget_review.txt`; ny STOP eller
tidsstop giver stadig en pause. Numerisk failure er terminal for dette run.
Slut-Y er ikke tilgængelig i dette miljø; PC-broen kan først eksportere
slutdata efter et faktisk frosset valg og en fuldt kontrolleret åben prognose.

Ukendt tidligere eksponering, nominelle fysiske parametre og alle begrænsninger
bevares. Ingen blind-H3- eller global-optimum-claim. G/WM er uændrede.
