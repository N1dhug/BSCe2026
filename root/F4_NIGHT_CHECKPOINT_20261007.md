# F4 nattearbejde, 7. oktober 2026

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
