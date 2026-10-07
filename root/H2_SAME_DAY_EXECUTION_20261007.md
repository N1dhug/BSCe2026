# F4-H2: godkendt dagsfortsættelse, 7. oktober 2026

Johan godkendte kl. 15:48:55 Europe/Copenhagen det konkrete forslag:
18 timers samlet loft og stop senest 7. oktober kl. 23:55. Den oprindelige
10-timerskonto, dens fulde historik og den uafklarede 30.000 s reservation
bevares. Det er ikke en refundering eller et nyt løfte om konvergens.
900 s native-development-loftet, kandidatcaps og numeriske gates bevares.

## Budget og uændret model

`h2_same_day_budget_decision_20261007.json` fryser godkendelsen, den gamle
campaign-hash, maksimumraterne fra de 1.076 auditerede jobs og dagsfristen.
En særskilt v2-campaign har 64.800 s loft og samme gamle entries som prefix.
Hver invocation reserverer højst 600 s; normal afslutning lukker kun denne
nye entry med faktisk wall-clock-tid. Et nyt crash fastholder reservationen.
Den gamle `f4_h2_remaining_budget.json` ændres ikke af dagscontrolleren.

Alle 69 numeriske/runner-kilder fra `e115242` og den tidligere ydre controller
bevares byte for byte. `run_h2_same_day.py` ejer kun den nye budget-/deadline-
kontrol og kalder den oprindelige Engine/JobStore, native kernels, COBYQA,
egen base/fine-validering, modelvalg og åbne ugeforecast. Jobs har fortsat
samme numeriske manifestidentitet. Search-flag er ikke et modelinput.

Den oprindelige jobfunktion har en hardcodet 36.000 s search-watchdog.
Den nye wrapper deaktiverer netop denne gamle watchdog ved delegation og
erstatter den med samme beregning mod det godkendte 64.800 s loft. Wrapperen
bevarer search-context under poolarbejdet, beskytter hele restvalideringen
og har både invocation-watchdog og deadline i parent og workers. Dette
ændrer ikke operatorer, summationsrækkefølge, jobidentiteter eller output.

## Admission og rækkefølge

Genoptag den eksisterende to-stykkers trace; den præcise rækkefølge og næste
kandidat er allerede kontrolleret uden nye modelkald. Den gemte PROBE ligger
i fire-stykkers initialdesign og genbruges. Samme 17/33 maxfev, én lineær
start, 16/32 nye par og i alt højst 49 par. Cap/timeout er ikke konvergens.

Maksimumrater med faktor 1,5 priser hele grafen; den færdige lineære egen
base/fine-validering fratrækkes én gang. Restvalideringen er initialt
22.005,997020 s. Den reserveres før ethvert nyt search-job og krediteres
kun efter originale beståede valideringsstreams. Empiriske gennemsnit
sænker ingen reserve. Det fulde 14,01 h-restscenarie er fortsat ikke admitted.

Fortsæt fit → egne fine/selection-streams for alle tre familier → frosset
modelvalg → 168 h åben uge med syv distributionshorisonter → PC-eksport af
slutdata → separat prequential finale. Ingen final-Y læses før frossent
forecast. Revider ikke en model efter at have set slutdata. I/J-review
bruger eksisterende evidens under kørslen og dokumenterer nominelle
parametre, udviklings-holdout og ukendt tidligere eksponering.

Efter hver lukket blok publiceres en checkpoint-delta og den lukkede
v2-ledger. Numerisk failure eller manglende trust-radius-konvergens er
terminal for denne erklærede kørsel. En budget-/invocation-pause kan
genoptages eksplicit efter status/source/budgetkontrol. Ingen auto-retry
efter failure, nye starter, capforøgelse eller løsere tolerancer.

Nyeste `same_day_report_NNN.json` er controllerens status. Gamle natte-
rapporter og den gamle crashsnapshot-rapport bevares som historik.
Et controllerresultat er ikke automatisk fuld videnskabelig F4-accept.
F/G/WM holdes adskilt, og F4-G2b/G2c genstartes ikke.

## Reproduktion

Brug `root/f4/run_h2_same_day.py --run RUN --campaign V2_LEDGER
--decision root/f4/h2_same_day_budget_decision_20261007.json
--source-commit OUTER_COMMIT --max-seconds 600 --stage prefinal`.
Den første godkendte fortsættelse bevarer recovery-STOP under et nyt navn.
Ved en ny STOP kræves igen konkret pause-review. Kildecommit skal være den
immutable commit der indeholder denne controller, beslutningen og protokollen;
et senere checkpoint-commit erstatter den ikke. Final kræver en allerede
hashverificeret `final_observations.npz` fra den eksisterende dataeksportbro.
