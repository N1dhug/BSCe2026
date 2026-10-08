# Kør F4 selv

Denne forsøgsudgave bruger det gemte fælles tostykkefit for de 13 MED-huse.
De fysiske parametre er faste nominelle antagelser. Første korte kørsel bruger
én times filtrering og tre timers åben prognose på de eksisterende data.

## Første kørsel

Åbn en terminal i repositoryets rod (mappen med `src`, `root` og `files`):

```powershell
python -m src.exe.run_model.bm_1_f4 --check
python -m src.exe.run_model.bm_1_f4
```

I PowerShell kan du også bruge `./run_f4.cmd` fra projektmappen. Den vælger
automatisk projektets rod og videresender samme argumenter, f.eks.
`./run_f4.cmd --check`.

`--check` læser og kontrollerer de valgte input uden at starte modelberegninger.
Terminalen viser den effektive filter-/prognoseperiode og det gemte fits cutoff.
Der udføres intet nyt parameterfit.
Første beregning kan bruge ekstra tid på at kompilere den eksisterende Numba-kerne.
Der installeres ingen pakker automatisk. Python 3.11/3.12 med NumPy, SciPy,
Pandas, OpenPyXL, Matplotlib og Numba er nødvendigt for denne indgang.

Resultater ligger som standard i `files/output/bm_1/runs/f4_quick`:

- `predictions.csv`: lokale og UTC-tidsstempler for intervalstart/-slut,
  prognose, baseline, observationer, tilstand, varianser og 5/50/95%-kvantiler
  ved de valgte prognosetimer. Enhedsnavnene bygger på antagelsen nedenfor.
- `comparison.png`: demand og tilstand over tid. Den lodrette streg er forecast-origin.
- `summary.json`: status, målefejl og antal færdige timer.
- `manifest.json`: konfiguration, model, inputidentiteter og kildeidentiteter.
- `checkpoint.npz`: seneste færdige time for hver model; dette er resume-grundlaget.
- `origin.npz`: den faktisk filtrerede tilstand ved prognosestart, når den er beregnet.

## Vælg periode og model

Ret `root/f4_quick.json`, eller kopier filen og brug `--config din_config.json`.
Relative input-/outputstier regnes fra projektets rod; selve `--config` kan være
en absolut sti eller en sti fra terminalens aktuelle mappe.

Datoer, horisont, model og grid kan også ændres direkte. Eksempel på inputkontrol:

```powershell
python -m src.exe.run_model.bm_1_f4 --filter-start 2022-02-21T00:00:00+01:00 --forecast-origin 2022-02-21T02:00:00+01:00 --forecast-hours 6 --model fitted --grid base --output files/output/bm_1/runs/f4_six_hours --check
```

Fjern `--check` for at beregne. Argumenterne har forrang for konfigurationen.
Brug `--model fitted linear` for begge modeller. Der er ingen `--week`, fordi
F4 genbruger et fast fit; WM4's `--week` vælger derimod fit-ugen. En særskilt
F-weekday-estimator har egen kontrakt og nye fitbudgetter; se
[matematikoversigten](../docs/bm/MATHEMATICS.md). Det faste F4-fit må ikke
betegnes som et nyt fit på fem tidligere mandage osv.

| Felt | Betydning |
|---|---|
| `filter_start` | Start på tilstandsberegning fra den deklarerede Beta(20,20)-prior. |
| `forecast_origin` | Eksklusiv slutning på måleperioden; efter dette tidspunkt assimileres ingen målinger. |
| `forecast_hours` | 1 til 168 timer med faste kurver og parametre. |
| `models` | `["fitted"]`, `["linear"]` eller begge. |
| `grid` | `"base"` eller `"fine"`; de oprindelige præcisionsindstillinger bevares. |
| `quantile_hours` | Timer med observationskvantiler. Timer ud over prognoselængden udelades. |
| `workers` | 1 til 8 frekvensarbejdere; fire er standard. |
| `max_seconds` | Tidsramme for modeldelen i én invocation; 600 sekunder som standard. |
| `evaluate_forecast` | Om faktiske målinger skal læses til sammenligning efter alle prognoser er gemt. |
| `output_dir` | En ny resultatmappe for hver ændret konfiguration. |

Start og origin skal være hele timer. Naive datoer fortolkes i
Europe/Copenhagen; tvetydige/ikke-eksisterende sommertidstimer afvises.
Eksplicitte UTC-offsets kan bruges. Det gemte kurvefit har informationscutoff
21. februar 2022 kl. 00:00; tidligere prognoseorigins afvises. Baseline og skala
er frosset på historik frem til 16. februar. Filtrering må ikke starte tidligere.

Den gamle `root/f4_week.json` deklarerer 24 filtertimer og 168 prognosetimer.
Den er bevaret som oprindelig konfiguration, men **afvises på de lokale data**:
filterdøgnet indeholder 17 negative målerceller fordelt på seks timer.
Første fejl er 21. februar 2022 kl. 10, bl.a. −0,30 i `5080166_MED`.
F-inputfejlen viser nu tidspunkt, målerkolonne og værdi. Betydningen af de
negative værdier er ikke dokumenteret; de klippes eller udelades ikke.

En åben prognose behøver ingen fremtidige målinger. `evaluate_forecast: false`
giver derfor prognose/eksport uden efterfølgende scoring, men ændrer ikke
valideringen af filterhistorikken. Den særskilte runtimekontrol bruger det
allerede gyldige korte filter 21. februar kl. 00→01 og derefter 168 timer.
Den må ikke forveksles med det afviste filterdøgn eller en kvalitetsevaluering.
Denne case er nu gemt i `root/f4_open_week.json` og gennemførte på **93,472 s**
inklusive import, input og eksport: filter 12,854 s, forecast 69,138 s.
Alle 168 timer og syv kvantilhorisonter er verificeret. Kør den i en ny mappe:

```powershell
python -m src.exe.run_model.bm_1_f4 --config root/f4_open_week.json --output files/output/bm_1/runs/my_f_open_week
```

Den præcise målte kildeversion, konfiguration og output er bevaret. Den korte
case tog 35,876 s og matchede den gamle fitted-reference eksakt. Fuld
faseopdeling og begrænsninger står i [sammenligningen](../docs/bm/COMPARISON.md).

## Pause og fortsættelse

Ved tidsstop er status `paused`, og programmet returnerer exitkode 2. Fortsæt:

```powershell
python -m src.exe.run_model.bm_1_f4 --resume
```

Du kan også afbryde med Ctrl+C eller oprette en fil med navnet `STOP` i
resultatmappen. Fjern `STOP` før fortsættelse. Der gemmes efter hver færdig time;
en halvfærdig time må beregnes igen. Tidsrammen kontrolleres før en ny time og
under frekvensjobs; en igangværende skalarberegning kan afslutte efter rammen.
Indlæsning af Excel-data og eksporter er uden for modeldelens tidsramme.
Den ovenfor målte åbne uge er under ti minutter. Det er ingen garanti for
andre filterlængder, fine-grid eller to modeller; flere blokke eller et større
`--max-seconds` er ikke bevis på, at et andet forløb opfylder målet.
Denne oprydning giver ikke tilladelse til et nyt langt benchmark eller fit.

Konfiguration, input, model og kildekode skal være identiske ved resume.
Arbejderantal og tidsramme må ændres. Brug en ny outputmappe ved ændret periode
eller model. Numeriske fejl giver `FAILED.json`; ret årsagen og brug en ny mappe.
Der foretages ingen automatisk genstart efter en numerisk fejl.

Resume-identiteten indeholder alle Python-filer under `src`, også filer uden
for F. Ændret kildekode betyder derfor en ny kørsel. Gamle checkpoints bevares
sammen med den oprindelige kildeversion og skal genoptages dér med samme
konfiguration og input. Hashkontroller og gamle manifests må ikke omskrives
for at indlæse dem med ny kode. Ved CLI-overrides skal de samme argumenter
gentages ved `--resume`.
En allerede færdig, kompatibel kørsel returnerer `complete` uden at ændre
resultater eller tidskvitteringer. Identiteterne kontrolleres først. Hvis en
færdig pakke mangler eksportfiler eller har modstridende status, afvises den;
gendan originalpakken eller vælg en ny mappe frem for at overskrive den.

## Fremtidige priser

Standard `previous_week` gentager de samme timer fra de syv foregående døgn.
Fremtidige realiserede priser læses ikke til denne prognose. Baseline er den
gemte weekday/hour-profil fra den tidligere historik.

For et selvvalgt prisforløb: brug `price_policy: "supplied"` og tilføj
`future_prices_csv: "din_prisplan.csv"`. Filen skal indeholde præcis én række
for hver forecast-time og de to kolonner:

```csv
timestamp,spot_price_DKK_MWh
2022-02-21T01:00:00+01:00,1000
2022-02-21T02:00:00+01:00,1100
2022-02-21T03:00:00+01:00,900
```

Planen er et betinget input. Hvis du indsætter senere realiserede priser, er
resultatet en prognose betinget på disse priser, og deres tilgængelighed må
vurderes særskilt. Priser uden for den frosne normaliseringsskala afvises.

## Resultaternes betydning

**Enhed og tidsmærkning er arbejdsantagelser, ikke verificerede målerdata.**
Målingerne fortolkes foreløbigt som kWh i en time med hour-start-tidsstempel.
Én times kWh er numerisk lig den gennemsnitlige kW i samme interval. Model og
målinger sammenlignes på dette intervalgrundlag, ikke som punktmålinger.
CSV-felternes `kW`/`kWh`-navne er bevaret af hensyn til eksisterende læsere;
manifest, summary og figur angiver antagelsen. Et interval er altid én fysisk
time `[timestamp, interval_end)`, også hen over et sommertidsskift.
Gruppens time er manglende, hvis blot ét af de 13 MED-huse mangler. Tilstanden
forudsiges gennem hullet uden en Bayes-opdatering; målinger udfyldes ikke.

Filterrækker viser forudsigelsen før den pågældende måling; deres endpoint-state
er posterioren efter målingen. Forecast-rækker viser én åben propagation fra
den faktisk beregnede origin, uden fremtidige observationer. Forecast-targets
læses først, når samtlige modelprognoser er gemt, og påvirker kun fejlmål/figur.

Kvantilerne er marginaler for timegennemsnit med den deklarerede Gaussian
observationsstøj, ikke samtidige bånd over hele ugen. De øvrige timer har mean
og variance, men tomme kvantilfelter. Parameter- og inputusikkerhed er udeladt.
5–95%-intervallet er et nominelt 90%-bånd; det har ingen dokumenteret
dækningskalibrering og er ikke G's 2,5–97,5%-bånd.

Summary viser RMSE, MSE, MAE og bias for model og baseline på samme observerede
timer, særskilt for filter og forecast. Bias er prognose minus observation.
`summary.timing.last_invocation_phases` indeholder UTC-start, varighed og
status for kontrolsummer, demand, priser, hver filter-/prognosetime,
evalueringsinput, worker-lukning og eksport. Timerne omfatter deres
checkpointskrivning. Første brug af JIT og workers ligger i den pågældende
filter-/prognosefase; en selvstændig kompileringstid er ikke målt.

`import_phase.seconds` måler den udførende forældreproces efter de indledende
standardbiblioteksimports. `last_invocation_seconds` og
`cumulative_recorded_seconds` omfatter CLI-forberedelse gennem eksport, men
udelader imports og de sidste tidskvitteringers skrivning. Direkte kald til
`run` omfatter ikke en tidligere `prepare`. Den eksterne proceskvittering er
derfor den fulde ende-til-ende-måling. Caches blev ikke nulstillet; målingen
kaldes ikke en kontrolleret kold kompilering. En hårdt dræbt proces kan mangle
sin sidste tidskvittering. Afsluttet kompatibel resume omskriver ingen resultater.

Tilstanden starter på ny ved `filter_start`. Denne nye filtreringskørsel er
ikke en gendannelse af H-studiets oprindelige cutoff-posterior. Vælg en længere
filtreringsperiode, når du undersøger følsomhed over for startprioren.

`root/f4_fixed_model_20261008.json` indeholder de præcise kurvelogits, baseline,
skala, nominelle parametre og kildehenvisninger fra det eksisterende MED-fit.
Et andet populationvalg eller ændrede skaleringer kræver et særskilt modelgrundlag;
ændr ikke husetilslutningerne og beskriv resultatet som det samme validerede fit.
Denne indgang foretager ikke nyt fit. Det eksisterende H-estimationsspor er bevaret.

## Brug origin i den eksisterende F3-kode

```python
from src.exe.run_model.bm_1_f4 import load_origin

origin = load_origin('files/output/bm_1/runs/f4_quick', model='fitted')
# origin.time er relativ tid i dage fra filter_start.
# origin.control_problem(...) bruger den eksisterende F3-evaluator.
```

Overførslen bevarer den faktisk beregnede masse, kurver og nominelle parametre.
Origin-filen indeholder også den fulde log-state; den erstattes ikke af logaritmen
til en afrundet sandsynlighedsmasse. Dette er et interface, ikke et nyt F3-studie.

F4 er her leveret som kørbar udviklingskode. Den oprindelige H/I/J-slutaccept,
fulde final-evaluering og firestykkefittets manglende konvergens har fortsat
deres historiske status. Den nye indgang genoptager ikke den gamle kampagne.
