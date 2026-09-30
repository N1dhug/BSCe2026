# BM — start her

Kør kommandoer fra repositoryroden (`BSCe2026`), ikke fra `root/`.
**Aktuel F4-status og næste opgave står kun i [F4.md](F4.md).**

## Tre dokumenter, tre formål

| Fil | Bruges til |
|---|---|
| [F4.md](F4.md) | Status, beslutninger og konkret plan for næste F4-leverance. |
| [BM_REFERENCE.md](BM_REFERENCE.md) | Samlet F1–F3-reference: matematik, afsluttede resultater og reproduktion. |
| [f4/main.tex](f4/main.tex) + [f4/notes.tex](f4/notes.tex) | Sammenhængende F4-matematik; ikke endnu en statuslog. |

`environment.yml` er bevaret. Data, WM-kode og eksisterende BM-metoder er ikke
ændret af dokumentoprydningen. Detaljeret tidligere udviklingshistorik findes i Git;
vi vedligeholder ikke parallelle audit-/performance-/closure-dokumenter.

## Daglige kommandoer

```powershell
python -B -m unittest tests.test_bm1_estimation -q
python -B -m src.exe.run_model.bm_1.estimate synthetic
python -B -m src.exe.run_model.bm_1.estimate filter
python -B -m src.exe.run_model.bm_1.estimate filter-refine
python -B -m src.exe.run_model.bm_1.estimate filter-study
python -B -m src.exe.run_model.bm_1.estimate filter-calibration
python -B -m src.exe.run_model.bm_1.estimate filter-truth-refine
python -B -m unittest discover -s tests -q
python -B root/f4/build.py
```

Den sidste kommando bygger PDF til `root/f4/build/main.pdf` og gentager selv
LaTeX-kørslen, indtil referencerne er stabile. Den ændrer ikke terminalens mappe
eller Windows PATH. Ingen `latexmk`/Perl kræves. En fundet MiKTeX-installation under
brugerens normale installationsmappe kan bruges, selv om PATH endnu ikke er opdateret.
Manglende LaTeX giver én klar fejl; Python-tests kræver ikke LaTeX.

GitHub-PR'en kører både regression og samme PDF-build og gemmer PDF/logs som
artifacts. Derfor er lokal LaTeX-installation **ikke en forudsætning for at arbejde
videre med modellen**. `Underfull`-linjer er layoutdiagnostik, ikke en grund til en
ny debugrunde. Undefined references, ustabile labels og fejl i selve buildet skal løses.

F4-0's uafhængige statiske kontrol er fortsat tilgængelig:

```powershell
python -B root/f4/verify_preparation.py
```

Kommandoen laver ikke filter-, parameter- eller forecast-forsøg. Se eksisterende BM-kørsler med `python -B -m src.exe.run_model.bm_1.verify --help` og
tilsvarende `optimize`, `refine`, `real_data`. F4-B tilføjer den første reelle estimation-entrypoint via `estimate synthetic`; F4-C
udvider samme entrypoint med det rekursive FV-filter og de dokumenterede validation-kørsler.

## Hvor kode hører hjemme

| Placering | Ansvar |
|---|---|
| `src/model/` | Fysiske ligninger og modelinterfaces. |
| `src/method/`, `src/solver/` | Numeriske metoder og lavniveau-solvers; `bm_1_estimation.py` indeholder den rekursive BM1-estimatorkerne. |
| `src/params/bm_1.py` | Generiske F4-observations-, Bayes-, likelihood- og remap-primitiver; ingen komplet estimator. |
| `src/control/` | F3-problem, fælles cost og prisoptimering. |
| `src/validation/estimation.py` | Offentlige inputkontroller; ingen eksperimenter. |
| `src/validation/bm_1/` | Referencecases, datagenerator og videnskabelige undersøgelser. |
| `src/diagnostics/`, `src/tools/` | Metrikker henholdsvis data/preprocessing. |
| `src/exe/run_model/bm_1/` | Argumenter, orkestrering, rapporter og valgfrie plots. |
| `tests/` | Hurtige regressionstests; ingen fulde videnskabelige benchmarks ved import. |

Udvid eksisterende ansvar før nye filer oprettes. Et F4-bogstav er ikke et
Python-modul. Ingen `_new`-/compatibility-kopier, omvendte imports fra core til CLI,
skjulte normaliseringer eller ændring af numeriske settings for at pynte på runtime.
FV-reference og fast FV er bevidst uafhængige implementeringer af samme scheme;
de skal ikke deduplikeres væk som led i oprydning. Shared WM-kode ændres kun i en
særskilt begrundet og testet ændring.

## Levering og filer

Én afgrænset feature-branch fra seneste `main`, én samlet PR med matematik, kode,
tests og nødvendig dokumentation. PR'en angiver kildecommit, faktiske testresultater,
begrænsninger og højst få lokale kommandoer. Små statusrettelser går i samme PR;
de kræver ikke deres egen closure-fase. Brugeren reviewer/merger; ingen automatisk merge.

Normale simulationskørsler skriver intet uden `--save`/eksplicit output. Lokale
resultater hører i ignoreret `files/output/bm_1/runs/`, PDF-build i `root/f4/build/`.
Bevar de oprindelige inputworkbooks og afsluttet evidens i
`files/output/bm_1/f3_b/` og `files/output/bm_1/real_data/`.
Slet ikke brugerens lokale arbejde, stashes eller miljøer for at opnå en ren status.
