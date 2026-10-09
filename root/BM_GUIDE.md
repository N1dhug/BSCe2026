# Kør BM-F og BM-G

Åbn en terminal i repositoryets rod. Brug Python 3.11/3.12 med pakkerne i
[requirements-f4.txt](../requirements-f4.txt); der installeres eller downloades
intet automatisk. Input-workbooks ligger lokalt i `files/input/data/`.
Gemte G-modeller og scenarieinput er lokale artefakter, som vælges eksplicit.

## F: genbrug gemt fit, filtrér og prognosticér

```powershell
python -m src.exe.run_model.bm_1_f4 --help
python -m src.exe.run_model.bm_1_f4 --check
python -m src.exe.run_model.bm_1_f4 --output files/output/bm_1/runs/my_f_example
```

Inputkontrollen og det korte eksempel er nu gennemført i en almindelig
terminalkørsel. Den målte fitted/base-kørsel med én filtertime og tre
prognosetimer tog **35,88 sekunder inklusive opstart, input og eksport**.
Det gemte resultat matcher den tidligere fitted-reference eksakt. Terminalen
viser nu imports, input, filtrering, prognose og eksport med tidsmærker.
`--max-seconds` begrænser modelarbejdet; brug hele procesvarigheden til runtime.

Standardeksemplet har én filtertime og tre forecasttimer. Der fittes ingen
nye parametre. Skift periode uden at redigere kildekoden:

```powershell
python -m src.exe.run_model.bm_1_f4 --filter-start 2022-02-21T00:00:00+01:00 --forecast-origin 2022-02-21T02:00:00+01:00 --forecast-hours 2 --output files/output/bm_1/runs/my_f_other --check
```

Fjern `--check` for beregning. Brug en ny outputmappe ved ændret periode.
`--model fitted` eller `--model fitted linear` vælger gemt kurve/lineær reference;
`--grid base`/`fine` bevarer de eksisterende præcisionsvalg. Cutoff 21. februar
kan ikke omgås med et tidligere origin. Filtrering må ikke starte før
baselinecutoff 16. februar. En ugyldig periode afvises med en inputfejl.

`predictions.csv`, `comparison.png`, `manifest.json` og `summary.json` viser
intervaller/UTC, observed/predicted/baseline, tilstand X, parametre, cutoffs,
metrics, status og tidskvitteringer. 5/50/95-kvantiler findes kun ved de
valgte timer. Enhederne er antagelser, og båndene er ikke kalibreret.
Læs [den korte F4-detaljevejledning](F4_RUN.md) om priser og konfiguration.

En hel åben uge med samme gemte fitted/base-model, én filtertime og 168
prognosetimer gennemførte på **93,47 sekunder inklusive indlæsning og eksport**.
Den bruger syv kvantilhorisonter og ingen fremtidige målinger eller scoring:

```powershell
python -m src.exe.run_model.bm_1_f4 --config root/f4_open_week.json --output files/output/bm_1/runs/my_f_open_week
```

Dette dokumenterer runtime for det konkrete forløb. Den gamle 24-timers
filterkonfiguration og prognosekvaliteten har særskilte begrænsninger nedenfor.

Ved `paused` (exitkode 2) fortsættes med samme argumenter plus `--resume`.
Ctrl+C eller en `STOP`-fil i outputmappen stopper efter gemt arbejde; fjern
`STOP` før fortsættelse. Inkompatibel source/config/input afvises. Gamle
checkpoints kræver original kildepakke; de migreres ikke ved at slække guards.
En 600-sekunders pause er ikke et bestået uge-runtime-mål.

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

Kalenderen bruger G som standard. F vælges med `--adapter F --adapter-config`
og laver nye native kurvefits med tidligere frossen baseline/skala/prior.
CSV-kontrakt, plan-only-eksempel og de eksplicitte budgetter står i
[kalendervejledningen](WEEKDAY_BACKTEST_GUIDE.md). Historisk fit er ikke kørt;
den samlede weekday-CLI har ingen automatisk resume.
En plan med `--plan-only` er ingen modelberegning. Et nyt fit på fem historiske
ugedage er forskelligt fra F4's genbrug af et fast fit. WM4's `--week` betegner
fitugen, mens BM angiver forecast-origin eksplicit.

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

## Kort brugerprøve — Johans vurdering afventer

1. Kør F's `--help` og `--check` ovenfor, derefter det korte fitted-eksempel
   i en ny outputmappe. Find CSV, figur, parametre og fase-/tidskvitteringer.
2. Kør G's gemte eksempel. Vælg derefter et udsnit på 24 timer med `--start`
   og en anden mappe. En dato uden for den gemte uge skal afvises.
3. Prøv F med et origin før 21. februar 2022; cutoff skal afvises. Den gamle
   `root/f4_week.json` afvises også på de nuværende data: dens filterdøgn har
   negative råmålinger. Dette er en datagrænse, ikke en fastlåst indlæsning.
4. Ved en reel F-pause bruges præcis samme argumenter plus `--resume`.
   Den automatiske pause/resume-kontrol består; brugeroplevelsen er endnu
   ikke vurderet af Johan.

## Fortolkning og videre læsning

- [Matematik og konkrete kodefunktioner](../docs/bm/MATHEMATICS.md).
- [Sammenligning, målte resultater og åbne begrænsninger](../docs/bm/COMPARISON.md).
- [F4's aktuelle og historiske acceptstatus](F4.md).
- [Historisk arkiv med kilde-/resultatkæder](../archive/bm/INDEX.md).
- [Oprydningsstatus og næste præcise trin](../docs/bm/STATUS.txt).

WM4-timeadapteren og original WM4-CLI er forskellige evalueringsprotokoller.
F/G's input, målerpopulation, intervaldefinition, horisont og usikkerhed skal
afstemmes før en fælles rangliste. Gemte seksugersresultater er udviklingsevidens,
og Johans egen brugertest står afventer, indtil han har afprøvet vejledningen.
