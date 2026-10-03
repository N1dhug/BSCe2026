# Fælles projektkontekst

Ved BM/F4-arbejde: læs først `root/F4.md`, især **Aktiv prioritering — tidsreduktion,
3. oktober 2026**. Det er den autoritative fælles status og to-do-liste; ældre
chatplaner og historiske "næste opgave"-afsnit kan være overhalet.

- Brugerens regel, 3. oktober 2026: brug **branches i eksisterende checkout**.
  Opret ikke flere worktrees, medmindre det er strengt nødvendigt; en ny fase,
  preview eller parallel chat er ikke i sig selv en grund. Bevar source under
  aktive låste kørsler. Før et eksisterende worktree fjernes: bevar branchen,
  egne ændringer og alle nødvendige ignorerede rådata/logs; kontrollér at ingen
  kørsel bruger mappen. Reglen gælder også WM-arbejde.
- I-baselinens origin/forecast/F3-integration er implementeret på det afsluttede
  F-spor; pooled H-f/g-fitprototype og gammel-record budgetpilot er implementeret.
  Næste udvikling er H1's nye curve-cases/numerik og H's data-/splitkontrakt.
  Obligatorisk H-f/g
  skal gennemføres før den endelige samlede I/J-leverance. G2b-v2 er valgfrit
  nat-sidearbejde; G2c er kun en mulig opfølgning efter G2b-accept. Ingen af dem
  må blokere H/I/J eller markeres gennemført uden komplette kontroller.
- G2a-v2 er komplet, rå-reviewet og numerisk bestået på ren `01c7c153`:
  8 units, 42.312 replays, 14 kontroller; nul nye FV-replays i råreviewet.
  Bevar original source og arkiv; løs ikke resume-kontrollen. Det frigiver
  hverken G2b/G2c eller recovery/forecast. Næste aktive arbejde er H1/H's datakontrakt.
- Beslutning fra Johan, 3. oktober kl. 22:26: natkørsler budgetteres til cirka
  otte timer per invocation, med hyppige atomiske blok-checkpoints og genoptagelse.
  Brug den særskilt registrerede `joint-bank-rectangular-robustness` (G2b-v2),
  som råvaliderer alle otte beståede G2a-v2-units før nyt arbejde. Tidsstop er
  en pause, ikke accept; bevarede hypoteser genkøres ikke. Gamle kvadratiske
  G2b/G2c-prerequisites løsnes ikke. G2c kræver et nyt kompatibelt design og
  G2b-accept. Fastlås nat-source i en branch i samme checkout; skift/pull aldrig
  under en aktiv kørsel. H1/H2/H3 og endelig I/J er fortsat hovedsporet.
- Metode og forsøgsdefinitioner: `root/f4/notes.tex`; kommandoer: `root/BM_GUIDE.md`.
  Opdatér fælles status i `root/F4.md` sammen med en leverance.
- Genbrug F1–F3-interfaces og gennemført evidens. Bevar grids, seeds, observationers
  betydning og numeriske/statistiske gates ved rene performanceændringer.
- Kontrollér nyeste `main` før publicering og bevar andres samtidige ændringer,
  herunder WM-arbejde. Følg brugerens konkrete instruktion om branch/publicering.
- Relevant regression: `python -B -m unittest discover -s tests -q`.
  Noter: `python -B root/f4/build.py`. Lange forskningsstudier er ikke CI-tests.
- Nye lange kørsler skal vise et løbende groft estimat for både resterende og
  samlet tid ved eksisterende statusopdateringer. Genbrug
  `src/exe/run_model/progress.py`; medregn forskellig opgavepris, hvor den er
  kendt, og adskil nyt arbejde fra gemt arbejde ved resume. Ingen ekstra replays
  eller hyppigere checkpoint-skrivning for at måle ETA.
- Følg også F4-planens afsnit **Yderligere effektivisering**: fælles I/F5/F6-
  interfaces, tidlig H-f/g-pilot og forberedelse under beregninger. Brug
  `run_fv_filter(..., keep_history=False)` til rene likelihood-kandidater og
  `evaluate_targets` til flere kvadratiske A-mål ved uændret U/process. Fuld
  historik/validering bevares for valgte modeller; de nye helpers afslutter
  hverken H-f/g, F5 eller F6. Ingen ny spekulativ solveromskrivning på kritisk vej.
- Genbrug `BM1ForecastOrigin` i `src/control/bm_1_handoff.py` til I/F5/F6:
  eksplicit cutoff/tidsenhed, frosne parametre/kurver og konservativ posterior.
  Rho-MAP bruger nodens egen conditional posterior. Et gennemsnitligt parameter-
  plug-in kræver nyt replay eller erklæret approksimation. `i-baseline` er kun en
  gammel-record integrationsdemo; ingen forecast- eller H-f/g-accept.
  `BM1PiecewiseResponse` og `src/method/bm_1_response.py` giver kurve-API og
  pooled replay-fit. Gammel-record prototype er ikke H's endelige fit/accept.
  Bevar deklarerede kandidatbudgetter og negative budgetstops; ingen automatisk
  ekstra fitrunde eller modelvinder efter TRAIN-score. Nye kurvecases/numerik,
  dataaudit, hele deklarerede træning og blind H3 mangler stadig.

Disse kurveregler gælder BM og ændrer ikke WM-modellernes egne fitkontrakter.

## Beslutning fra Johan, 3. oktober 2026

- Estimation og validering af fleksible `f(x)`/`g(u)` er en **obligatorisk H-f/g-leverance**,
  endnu ikke afsluttet; udviklingsestimatoren er implementeret, fuldt fit og
  videnskabelig validering mangler. Den må ikke igen klassificeres som en valgfri udvidelse.
- Ét fælles kurvepar skal bestemmes samlet på tværs af træningsdage for en erklæret
  population/gruppe og holdes fast ved validering/forecast. Ingen f/g-fit per dag.
  Daglig baseline og state-estimation kan fortsat variere.
- Undersøg kontinuerte, ikke-voksende stykvise kurver med knuder
  `0, 0.25, 0.5, 0.75, 1`, faste endpoints `1,-1` og fælles estimerede knudeværdier.
  Intervaller vælges efter state/pris, ikke tidspunkt. En enklere pilot og lineær
  reference bevares; alle delstykker indgår i samme likelihood-fit.
- H-f/g skal gennemføres før endelig H→I-handoff og J. Split, normalisering,
  observationskontrakt, numerik og modelvalg dokumenteres; slut-holdout holdes blind.
  Nye kurver kræver egne kontroller. Det faste lineære A--G-design og gemt evidens bevares.
- Budgettér kurveestimationen med en lille pilot og afgrænset optimizer; opret ikke
  et stort Cartesian-grid over kurve- og fysiske parametre. Følg H-f/g-planen.
- Ved ændret status skal `root/F4.md`, nødvendig matematik og vejledning stemme overens.
  Beskriv ikke en planlagt eller blot implementeret estimator som valideret.

Dette er den fælles repoindgang for projektchats. En allerede åben chats indlæste
kontekst opdateres ikke automatisk; ved genoptagelse skal den læse den aktuelle plan.
