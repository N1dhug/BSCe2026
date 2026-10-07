# F4 nattearbejde, 7. oktober 2026

## F4-H2: crash gendannet; videre kørsel budgetblokeret, 7. oktober

Alle seks checkpoints fra `e79aa10` er faktisk gendannet og hashkontrolleret:
1076 jobs, 69 frosne sourcefiler, 12 komplette to-stykkers objective-kald
(inkl. lineær start), næste kandidat34/72 timer. Fire-stykkers fit er ikke
startet. Ingen optimizer-resultatfil, terminalrapport, selection eller
frosset forecast. Ingen gemte failed-jobfiler; ingen accept eller ny modelkørsel.
Alle536 lokalt genfundne jobs indgår byte-identisk i gendannelsen.

Den autoritative ledger matcher checkpoint06 og har en åben30000s-reservation.
Crashreglen bevarer den fuldt. Efter særskilt60s recoverycharge er
H32911.549305s, rest3088.450695s; native803.307761/900s.
Sidste resterende validation19960.314316s overstiger restbudgettet.
Ingen automatisk refundering, ny reservation eller fitstart; ikke brug
en ældre seed/ledger. Aktuel afstemning og ledger-efter er gemt i
`f4_h2_crash_budget_reconciliation_20261007.json` og
`f4_h2_crash_campaign_after_20261007.json` under estimation-output.
Basejobs er nu målt op til9.62828s: nyt budget kræver genprisning.
Numba0.62.1/llvmlite0.45.1 skal også gendannes før numerisk resume.

Læs [gendannelse, evidens og præcist næste trin](F4_CRASH_RECOVERY_20261007.md).
Videre beregning kræver en senere autoritativ lukket ledger eller Johans
udtrykkelige nye budgetbeslutning. Det eksisterende36000s-loft er uændret.
PR106 forbliver draft; F4/H/I/J er ikke afsluttet. G/WM ændres ikke.
Følgende statusafsnit er historik.


Aktiv branch: `feature/f4-h2-night-runner-20261007`.
Udgangspunkt: lokal `b1a702a`, GitHub main `4f9ea30abee6a2673a33868ffad30e9ff82cc429`.
Kun eksplicitte F-filer publiceres oven på seneste main; samtidig G/WM bevares.

## Genoptagelse

1. Læs denne fil, `H2_COMPACT_RATE_REVIEW.md` og den aktuelle Git-status.
2. Genbrug den råreviewede PC-ZIP og dens 72 frosne observationer. Ingen ny
   workbook-preparation, ingen gentagelse af 27-timers pilot.
3. Færdiggør en afgrænset runner til fælles kurvefit, egne base/fine streams,
   selection, frosset open-week og separat final prequential evaluering.
4. Kontroller checkpoint/resume, budget, datagrænser og negative gates med
   små kontroller. Bevar fejl og charges; ingen automatisk ny modelprøve.
5. Publicér checkpoints, review egen ændring, merge når checks består.

## Bindende udgangspunkt

H: 1671.0408590929223 s; native: 646.4097384530473 / 900 s.
Native-rest: 253.59026154695266 s. Validation-reserve: 11840.471907224273 s.
Samlet H-loft: 36000 s. Samme 1/2/4-piece familier, 49 unikke TRAIN-par,
48/24/168 timers split, separate MED-huse, uændrede numeriske gates.

Den lineære pilot er kun 24 prior + 3 observerede timer. Et fuldt base/fine
TRAIN+selection-vindue kan ikke dokumenteres ved at gange dens fejl med 24;
filtreringen er observationafhængig og nonlinear. En sådan kontrol kan ikke
finansieres af 253.59 s ved de målte rates. Koden skal derfor skelne mellem
implementeret, kørselsadmitted og faktisk numerisk/videnskabeligt accepteret.
Ingen flytning af udviklingskontroller til en anden budgetetiket.

## Status

- [x] Aktuel kilde, råpilot, budget og seneste GitHub-main læst.
- [x] Konkret admission-/valideringsbeslutning dokumenteret i `H2_COMPACT_STAGED_RUN.md`.
- [x] Genbrug af frossen receipt og portable inputs.
- [x] Fit, egne valideringsstreams og atomiske checkpoints/resume implementeret.
- [x] Prognose og betinget I/J-handoff implementeret; syntetisk fuldt flow kontrolleret.
- [x] Ni afgrænsede kontroller består; separate prior-API-kontrol består.
- [ ] Publiceret og reviewet.

## Aktiv kørsel og sidste holdbare kilde

Checkpoint03 bevarer yderligere358 atomiske jobs (608 i alt med01+02),
SHA256 `8291b790365fe68854bcda139447c79eb3188148e851c7b022635c7301f487b9`.
Checkpoint04 tilføjer127 jobs (735 i alt), SHA256
`baef38f921964263b11fd41a9b39ceadbe013b34574d25651c8730cc9e6b216d`.
Checkpoint05 tilføjer159 jobs (894 i alt), SHA256
`e0964149339e6960f7bd9c7f9eb567f52eeb2c3ece6c79890985fc4c485550a8`.
Checkpoint06 tilføjer182 jobs (1076 i alt), SHA256
`f39cd9f7b9e706f284d62c6559e4d85bf2a83619682b771e1ed04bfd55aa1044`.
Det indeholder også den åbne produktionsreservation; denne må ikke nulstilles
ved crash. Færdige kald genbruges; et ufuldstændigt kald er ikke konvergens.
`root/f4/status_h2_budgeted.py` giver et separat vægtet statusestimat for det
aktuelle par uden at ændre de frosne numeriske kilder. Resultatauditoren
rekonstruerer gemte likelihoods og COBYQA-stop fra råfiler uden modelreplay.

`root/f4/restore_h2_staged_checkpoints.py` kontrollerer alle tidligere
arkivhashes, kilde-/datamanifest og hver jobfil før gendannelse til en ny
mappe. Angiv alle nummererede checkpoints i rækkefølge og `--destination`.
`--verify-only` verificerer uden udpakning. Gendannelse starter pauset med
en STOP-markør og overskriver aldrig den autoritative campaign-ledger.
Brug `campaign_at_checkpoint.json`, ikke en ældre `ledger.json`, ved
reconciliation; behold åbne reservationer. FAILED-runs kan ikke genoptages.
Ingen model starter fra dette værktøj. Endeligt råreview kontrollerer faktisk
gendannelse byte for byte samt afvisning af manglende prefix/overskrivning.

**Aktiv nu:** ydre controller-kilde `5e7875674447b13a9697cf47c358e2f7e3a98b45`,
`run_h2_budgeted_search.py`, samme V2-run-mappe og uændret numerisk manifest.
Startledger H2851.549305434 s. 30000 s maksimal invocation-reservation;
den fulde dynamiske valideringsreserve beskytter fælles36000 s loft før
hvert søgejob. `budgeted_execution.json` og `budgeted_admission_001.json`
indeholder den faktiske admission. Den første lineære objective er cached;
nonlinear to-stykkers søgning er startet. Se seneste fælles ledger ved crash.
Følgende afsnit beskriver afsluttede tidligere etaper.

**Nyeste status:** den første produktion er pauset efter fuld bestået lineær
reference og før optimizer. Invocation1083.610424 s. Hele referenceforløbet
er bevaret i checkpoint01+02; råaudit er `f4_h2_linear_window_audit_20261007.json`.
TRAIN-scorefejl .0055541053; selection .0004541615; L1 .0010493139.
Nye globale maksimumrater giver14.61 h for hele49-par-grafen, som afvises.
`H2_COMPACT_BUDGETED_SEARCH.md` registrerer i stedet et eksplicit tidsbegrænset
konvergensforsøg med hele resterende validering reserveret først. Ny controller
`root/f4/run_h2_budgeted_search.py`; gamle modeljobs og69 kilder ændres ikke.
Seneste lukkede ledger H2851.549305434 s, native743.307761054/900 s,48 entries.
Stopmarkøren flyttes til `STOP_before_budget_review.txt` før det nye forsøg.
Det gamle `run_h2_compact.py run` må ikke genoptages. Følgende er tidligere status.

Kilde på GitHub: `e115242de79756a4a7163da28593f2452f17bcc6`.
Run: `files/output/bm_1/runs/estimation/f4_h2_staged_night_v2_20261007`.
Manifest: `e4d14ffe2473bf44dbfe3d6514ee13fd98e572234cb854f46255dd16f5afbeec`.
Ændr ikke runner, protokol eller numeriske afhængigheder under denne kørsel.

Nonlinear prøve består på **83.738727247 s**. 69 kildefiler og 64 råjobs er
råkontrolleret uden native/FV-replay. Scoreforskel 2.0704491e-6; maksimal
state-L1 .0004062245; CF→CDF/kvantiler og moments består. Det er stadig en
27-timers prefix, ikke hele TRAIN/selection. Den eksisterende lineære
PC-prefix genbruges byte for byte, når produktionsreferencen fortsætter.

Før produktion: H **1764.392331237 s**, native **739.761210597 / 900 s**.
Komplet konservativ graf: **7.880713832 h**, validation **11851.641664530 s**.
En produktions-invocation har reserveret 29000 s inklusive stopreserve.
Den fælles lokale ledger er autoritativ; dens åbne reservation må ikke
erstattes af en ældre seed ved crash. En crash-reservation beholdes fuldt.
Først den fulde lineære reference; COBYQA starter kun ved beståede gates.
Slutugens workbook er ikke tilgængelig her og læses ikke i denne etape.

Bevarede implementeringsfejl: én IndentationError før reservation (2 s
konservativ charge), og ét forkert positional Beta-API-kald under preparation
(1.36381916 s, nul modeljobs). Begge kilder/fejl er bevaret. Keyword-kaldet
er rettet og særskilt kontrolleret; V2 er den første faktiske nonlinear prøve.

Råreview: `root/f4/audit_h2_staged_probe.py` og
`files/output/bm_1/runs/estimation/f4_h2_staged_probe_audit_20261007.json`.
Historisk fit og videnskabelig accept er endnu ikke afsluttet.
