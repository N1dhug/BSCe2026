# BM — arbejdsvejviser

**F4-status og næste opgave:** [F4.md](F4.md).
**Sammenhængende matematik:** [f4/main.tex](f4/main.tex) + [f4/notes.tex](f4/notes.tex).
**Afsluttet F1–F3-evidens:** [BM_REFERENCE.md](BM_REFERENCE.md).
Der vedligeholdes ikke parallelle audit-/closure-rapporter i `root`.

## Almindelig kontrol fra projektmappen

```powershell
python -B -m unittest discover -s tests -q
python -B root/f4/build.py
```

PDF'en skrives til `root/f4/build/main.pdf`. Buildscriptet gentager selv LaTeX,
indtil referencerne er stabile; ingen `latexmk`/Perl og ingen manuel PATH-ændring
kræves. MiKTeX i brugerens normale installationsmappe kan findes automatisk.
Manglende LaTeX giver en klar fejl; Python-tests kræver ikke LaTeX. CI bygger også
PDF og gemmer logs/artifacts. Underfull-linjer er layoutdiagnostik; uløste labels,
overflow og egentlige buildfejl skal undersøges.

## Kør kun det relevante forsøg

Fælles indgang: `python -B -m src.exe.run_model.bm_1.estimate <kommando>`.
Hver kommando har `--help`. De tunge studier nedenfor er **ikke** daglige checks.

| Formål | Kommando |
|---|---|
| F4-0 uafhængig kontinuert kvadratur/Bayes | `preparation` |
| F4-B syntetisk sekvens / koblet EM | `synthetic`, `synthetic --refine` |
| F4-C sanity / numerik | `filter`, `filter-refine` |
| F4-C population / diagnostik | `filter-study`, `filter-calibration`, `filter-truth-refine` |
| Historisk D1 / D1b / D1c basisdesign | `mixture-static`, `mixture-static-adaptive`, `mixture-static-shape` |
| D2 statisk holdout | `mixture-holdout` |
| D3a / D3b / D3c | `mixture-filter`, `mixture-filter-refine`, `mixture-filter-converge` |
| D3d population | `mixture-filter-population` |
| F4-E E1 likelihood-refinement | `likelihood-refine` |

`preparation` erstatter det tidligere `root/f4/verify_preparation.py`: samme
uafhængige regneregler, nu i validation og med fælles CLI. Ingen kontrol er slettet.
F3-kommandoer findes stadig via `verify`, `optimize`, `refine` og `real_data` i samme
BM1 entrypoint-mappe. F3-data, WM-kode og `environment.yml` er bevaret.

## Lange kørsler og reproduktion

Populationen viser nu løbende antal færdige sekvenser, elapsed og grov ETA.
`--quiet` skjuler kun progress, ikke slutresultatet. `--output` gemmer hele rapporten,
alle sekvensrækker, settings, source-commit og dirty-flag i en **ny** JSON-fil;
eksisterende output overskrives ikke. Uden eksplicit output skrives ingen resultatfiler.

```powershell
python -B -m src.exe.run_model.bm_1.estimate mixture-filter-population --workers 8 --blas-threads 1 --output files/output/bm_1/runs/f4_population.json
```

`--blas-threads 1` er opt-in: BLAS/OpenMP-miljøvariabler sættes for denne Python-proces
**før** NumPy/SciPy importeres og arves af worker-processer. Det ændrer ikke permanent
Windows-miljøet, tolerancer, seeds eller worker-fordelingen af sekvenser. Det undgår, at
mange processer hver starter mange BLAS-tråde. Flere workers er ikke automatisk hurtigere;
mål på egen maskine. Uden flaget beholdes bibliotekernes eksisterende thread-indstillinger.
I en allerede startet notebook kræver threadkontrol en runtime-mekanisme som threadpoolctl.

Batchgenerering bruger samme pr.-ID RNG-strømme og samme EM-grid som enkeltgenerering.
Caches indeholder kun target-uafhængig geometri eller eksplicitte, validerede startgæt;
FV-referenceprojektioner starter aldrig mixture-filteret. Den registrerede D3d-protokol
er uændret; rapporten markerer, om en kørsels settings faktisk matcher den.

## Hvor kode hører hjemme

| Placering | Ansvar |
|---|---|
| `src/model/` | Fysiske ligninger og modelinterfaces. |
| `src/method/`, `src/solver/` | Estimatorkerner, numeriske metoder og lavniveau-solvers. |
| `src/params/bm_1.py` | Observations-, Bayes-, likelihood- og remap-primitiver. |
| `src/control/` | F3-problem, fælles cost og prisoptimering. |
| `src/validation/estimation.py` | Offentlige inputkontroller, ingen eksperimenter. |
| `src/validation/bm_1/` | Referencecases, syntetisk data og videnskabelige studier. |
| `src/diagnostics/`, `src/tools/` | Metrikker henholdsvis data/preprocessing. |
| `src/exe/run_model/bm_1/` | CLI, progress, rapporter og eksplicit filoutput. |
| `tests/` | Regression og små integrationstests; ikke hele populationer ved import. |

Udvid eksisterende ansvar før nye moduler oprettes. Ingen `_new`-/compatibility-kopier,
omvendte core→CLI-imports eller skjulte normaliseringer. FV-reference og fast FV er
bevidst uafhængige implementeringer af samme scheme; de skal ikke deduplikeres væk.
Shared WM ændres kun i en særskilt begrundet og testet ændring.

## Levering og filer

Én afgrænset branch/PR med matematik, kode, tests og nødvendig dokumentation. Test og
sammenlign efter hver selvstændig rettelse; benchmark på samme settings/hardware og
rapportér også uændrede output eller relevante numeriske afvigelser. Brugeren merger;
derefter sync, branch-delete og prune. Ingen automatisk merge.

Lokale resultater hører i ignoreret `files/output/bm_1/runs/`, PDF-build i
`root/f4/build/`. Bevar inputworkbooks og afsluttet evidens i
`files/output/bm_1/f3_b/` og `files/output/bm_1/real_data/`. Slet ikke brugerens
lokale arbejde, stashes eller miljøer for at opnå en ren status.
