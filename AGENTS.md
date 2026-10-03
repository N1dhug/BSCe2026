# Fælles projektkontekst

Ved BM/F4-arbejde: læs først `root/F4.md`, især **Aktiv prioritering — tidsreduktion,
3. oktober 2026**. Det er den autoritative fælles status og to-do-liste; ældre
chatplaner og historiske "næste opgave"-afsnit kan være overhalet.

- Første korte leverance er I-baseline på det afsluttede F-spor; obligatorisk H-f/g
  skal gennemføres før den endelige samlede I/J-leverance. G2b/G2c er en udsat og fortsat
  blokeret udvidelse; markér dem ikke som gennemført.
- Bevar G2a-v2's kørende checkout og oprindelige kildecommit. Ny kode i en anden
  worktree må ikke ændre kilden under en kørsel eller løsne resume-kontrollen.
- Metode og forsøgsdefinitioner: `root/f4/notes.tex`; kommandoer: `root/BM_GUIDE.md`.
  Opdatér fælles status i `root/F4.md` sammen med en leverance.
- Genbrug F1–F3-interfaces og gennemført evidens. Bevar grids, seeds, observationers
  betydning og numeriske/statistiske gates ved rene performanceændringer.
- Kontrollér nyeste `main` før publicering og bevar andres samtidige ændringer,
  herunder WM-arbejde. Følg brugerens konkrete instruktion om branch/publicering.
- Relevant regression: `python -B -m unittest discover -s tests -q`.
  Noter: `python -B root/f4/build.py`. Lange forskningsstudier er ikke CI-tests.

Disse kurveregler gælder BM og ændrer ikke WM-modellernes egne fitkontrakter.

## Beslutning fra Johan, 3. oktober 2026

- Estimation og validering af fleksible `f(x)`/`g(u)` er en **obligatorisk H-f/g-leverance**,
  endnu ikke implementeret. Den må ikke igen klassificeres som en valgfri udvidelse.
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
