# Fælles projektkontekst

Ved BM/F4-arbejde: læs først `root/F4.md`, især **Aktiv prioritering — tidsreduktion,
3. oktober 2026**. Det er den autoritative fælles status og to-do-liste; ældre
chatplaner og historiske "næste opgave"-afsnit kan være overhalet.

- Næste leverance er I/J på det afsluttede F-spor. G2b/G2c er en udsat og fortsat
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
