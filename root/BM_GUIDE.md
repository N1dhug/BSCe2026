# Kør BM-G

Brug Python 3.11/3.12 og [requirements-bm.txt](../requirements-bm.txt).
Input og gemte modeller vælges eksplicit.

## G: prognose fra gemt weekday-export

```powershell
python -m src.exe.run_model.bm_1_g --help
python -m src.exe.run_model.bm_1_g bands --help
```

G's nye indgang genbruger `weekday_export_snapshot.json` og et eksplicit
Price/Temp-scenarie. Den udfører hverken nyt fit eller historisk replay.
Snapshotets mandagsorigin og informationscutoff er faste; `--start` og
`--hours` vælger en del af den gemte uge. En anden uge kræver sit eget fit.
Det allerede gemte lokale eksempel er klargjort under `files/output/bm_1/examples/g_saved_week_20220228`:

```powershell
python -m src.exe.run_model.bm_1_g --snapshot files/output/bm_1/examples/g_saved_week_20220228/snapshot.json --inputs-csv files/output/bm_1/examples/g_saved_week_20220228/inputs.csv --check
python -m src.exe.run_model.bm_1_g --snapshot files/output/bm_1/examples/g_saved_week_20220228/snapshot.json --inputs-csv files/output/bm_1/examples/g_saved_week_20220228/inputs.csv --score files/output/bm_1/examples/g_saved_week_20220228/observations.csv --output files/output/bm_1/runs/my_g_week
```

Tilføj `--start "2022-02-28 06:00" --hours 24` og vælg en ny outputmappe
for at se et andet gyldigt udsnit. Hele den gemte uge beregnes stadig til
numerikkontrol; udsnittet er ingen ny fit-origin. Input-CSV har kolonnerne
`Datetime,Price,Temp`; observationsfilen har `Datetime,Demand` og åbnes først
efter frysning. Der foretages ingen automatisk vejrdownload.

Find `forecast.csv`, `scored.csv`, `comparison.png`, `saved_snapshot.json`
og `summary.json` i outputmappen. Tabellen viser rå enheder, UTC-intervaller,
horisont, pris/temperatur og modeltilstande. Gamle ukalibrerede empiriske bånd
vises ærligt; læs protokol og cutoff i `forecast.json`. Færdige mapper afvises
ved nyt kald. Ved afbrydelse bevares mappen; brug snapshot og en ny mappe.
Den korte G-propagation har ingen deltime-resume.

Eksemplet er lokalt og ignoreret af Git. En anden maskine skal have eget
verificeret snapshot/scenarieinput; rådata publiceres ikke med koden.

Generiske bånddatoer kan vælges med `bm_1_g bands calibrate/evaluate`.
Brug `--help` for begge trin. Kalibrering kræver hele tidligere forecastuger,
matchende informationskontrakt og mindst 300 målte scores; evaldatoer angives
på forhånd med `--evaluation-origins`. Evaluering kræver den uændrede
`calibration_snapshot.json` sammen med scores og freeze-receipt. Alle
prognoser fryses før evalueringens observationer åbnes. Det er et genbrugeligt
brugstrin, ikke en ny autorisation til historiske studier.

For ny fitplan bruges den eksisterende kalender:

```powershell
python -m src.exe.run_model.weekday_backtest --help
```

Kalenderen bruger G som standard. Se [kalendervejledningen](WEEKDAY_BACKTEST_GUIDE.md).

De ældre S3-værktøjer er samlet under ét modul med eksplicit modelvariant:

```powershell
python -m src.exe.run_model.bm_1.s3 interval --help
python -m src.exe.run_model.bm_1.s3 residual --help
python -m src.exe.run_model.bm_1.s3 point-process --help
```

Variantnavnet efterfølges af `fit`, `predict` eller `diagnose`, hvor det er
tilgængeligt. [Arkivet](../archive/bm/INDEX.md) forklarer deres betydning og
bevarer de gamle kommandoers præcise kildeversion. Gamle kildebundne
G7/G8/H2-kørsler hører til i det gendannede originaltræ.


## Videre læsning

[Matematik](../docs/bm/MATHEMATICS.md), [status](../docs/bm/STATUS.txt) og [arkiv](../archive/bm/INDEX.md).
