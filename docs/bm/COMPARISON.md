# BM-F/G og uændret WM4: evidens og sammenlignelighed

Tallene nedenfor genbruger afsluttede lokale resultater. Der er ikke kørt nyt
historisk fit eller ny seksugerskampagne. Perioderne er allerede eksponeret
udviklingsevidens. Ingen generel modelvinder kan udledes.

## Protokoller

| Punkt | F4 fast fit | G weekday | WM4 timeadapter | Oprindelig WM4 weekly |
|---|---|---|---|---|
| Population | Fast middel af alle 13 MED-huse; komplet time kræves | MED-middel over tilgængelige målere | Samme som G i benchmark | Loaderens tilgængelige målere |
| Observation | Fysisk timeintervalmiddel | Fysisk timeintervalmiddel | Integreret WM-output | Punktprøver i eksisterende CLI |
| Fit/baseline | Frosset fælles kurve; fast ugeprofil | Fem ens ugedage, syv fits | Egne syv free-run fits | En fituge, næste prognoseuge |
| Skala | Frossen historik til 16. februar | Demand fra femugershistorik; fast deklareret prisskala | Samme som G | Fuldperiodens loader-skala |
| Prognosestart | Filtreret state fra Beta-prior | Syv dagsvise medianstarts, frosset ved mandag | Egne medianstarts | WM's egen initialiseringspraksis |
| Fremtidigt input | Standard: sidste uges priser | Realiseret pris/temperatur som scenarie | Samme scenarie som G | Realiserede inputs efter CLI-praksis |
| Bånd | 5/50/95-kvantiler ved udvalgte horisonter; nominal støj | Empirisk outputfejl og frosset 95%-regel | Eget empirisk lag/95%-regel | Protokolspecifikt; ingen fælles båndclaim |

G's WM-reference ændrer ingen WM-fil, men bruger egen observationsoperator
og estimation. Den er altid mærket **WM4 timeadapter**. Oracle-tilstandsfit
med `--optimize-prediction-state` er udelukket fra almindelig prognosevurdering.
F's $X$ bevarer sit navn; det er ikke en ommærkning af WM's $F$ eller $R$.

De rå demand-enheder og hour-start er foreløbige metadataantagelser.
Derfor rapporteres fejl som **rå workbook-enheder**. F's historiske `kW`-felter
læses kun betinget på kWh-per-time-antagelsen. Ingen tal reskaleres her.

## Genbrugt G/WM-evidens

Kilde: den verificerede lokale `G-weekday-bands-20261008/results.json` og
`RESULTATER.md`; originalpakken er bevaret i recovery-kvitteringen.
G og adapteren er vurderet på samme rå observationsmaske, kalender og input.

| Periode / mål | Baseline | G | WM4 timeadapter |
|---|---:|---:|---:|
| Alle seks origins 28. feb.–4. apr., 1007 timer: RMSE, empirisk middel | 0.954721 | 0.984944 | 0.953195 |
| Senere tre origins 21. mar.–4. apr., 503 timer: RMSE, empirisk middel | 0.919460 | 0.904896 | 0.916514 |
| Samme 503 timer: RMSE, ren model | 0.919460 | 0.905762 | 0.927247 |
| Samme 503 timer: MAE, empirisk middel | — | 0.724782 | 0.734093 |
| Samme 503 timer: bias, empirisk middel | — | -0.074958 | -0.085794 |
| Samme 503 timer: gammel empirisk 95%-dækning | — | 40.56% | 40.95% |
| Samme 503 timer: frossen kalibreret dækning | — | 98.21% | 97.81% |
| Samme 503 timer: kalibreret middelbredde | — | 4.003992 | 3.955143 |
| Samme 503 timer: kalibreret intervalscore | — | 4.090001 | 4.064815 |

På de senere uger er G's fejl mindre; på alle seks er WM-adapterens mindre.
Båndbredden steg omtrent fire gange. G/WM har henholdsvis 420/414 negative
nedre rågrænser ud af 503; de klippes ikke efterfølgende for at forbedre rapporten.
Kalibreringen har 504 scores/model og rank 480. Lag-1-korrelation cirka .80/.83
og gentagne udviklingsvalg begrænser fortolkningen af dækningsprocenterne.

Alle 42 baselinefit har 49 features og designrank 20. Klipning ved 0/1 er
hyppig. Forfiningsfejl højst cirka $3.09\cdot10^{-10}$ i gemt G-mean er numerisk
evidens; den dokumenterer ikke prognosekvalitet, identificerbarhed eller calibration.

## F-evidens holdes adskilt

`F4_code/files/output/bm_1/runs/f4_delivery_final/summary.json` er komplet:
én filtertime og tre forecasttimer for fitted og linear. Forecast-RMSE er
0.358582 / 0.291181 mod baseline 0.285540 i historiske antagne rå kW-enheder.
Den gemte modeldel tog 46.465 s for begge modeller. Det er tre timer med et
andet cutoff, en anden population og et andet inputscenarie; tallene kan ikke
sættes ind som F-række i G's seksugersrangliste.

Et fælles F/G/WM-resultat er derfor **ikke dokumenteret** endnu. Det kræver
fælles fuldt observerede timer/population, eksplicit samme pris-/vejrinformation,
intervaloperator og horisont samt et autoriseret F-forløb. Manglende F-resultat
udfyldes ikke med interpolation, modelomdøbning eller en ny skjult fitkampagne.

## Aktuel strukturel gennemgang

Baseline er `c504806`, på samme lokale reviewbranch uden merge. WM149
udelades ens på begge sider af optællingen. Kommentarer og tomme linjer tælles;
originalZIP'er tælles udfoldet særskilt, ikke som én forenklet kildefil.

<!-- ROUND5_COUNTS -->
| Mappe | Filer før→efter | Mapper før→efter | Linjer før→efter |
|---|---:|---:|---:|
| `src/params` | 4→3 | 1→1 | 454→324 |
| `src/tools` | 10→8 | 1→1 | 3029→2247 |
| `src/validation/bm_1` | 30→16 | 1→1 | 14316→6904 |
| `src/method` | 16→15 | 1→1 | 5037→5018 |
| `src/exe/run_model` | 20→19 | 2→2 | 7902→5340 |
| `src/model` | 0→0 | 0→0 | 0→0 |
| `src/control` | 1→1 | 1→1 | 154→154 |
| `tests` | 79→66 | 1→1 | 15380→11361 |
| `configs/bm` | 5→5 | 1→1 | 335→335 |
| `docs/bm` | 11→11 | 1→1 | 2540→2484 |
| `root` | 12→12 | 2→2 | 3648→3648 |
| `files/output/bm_1` | 749→756 | 95→97 | — |

Kodeområder/tests viser Python-linjer; docs/config/root viser tekstlinjer.
Tests' filantal inkluderer én requirements-fil; Pythonfiler er78→65.
Mapper er logiske forældremapper i samme afgrænsning, inklusive den nævnte
mappe; fuld cache/diskhistorik bruges ikke som reduktionsmål. `src/model` har
ingen ubeskyttede filer i denne afgrænsning; fælles BM/WM-modeller består.
Output95→97 mapper skyldes én samlet checks-mappe og én ny regressionpakke.
Det er organisering og ny kontrol, ikke resultat- eller mappereduktion.
Én verificeret tom egen cachemappe er faktisk fjernet. Arkivet har fortsat
766+2 udfoldede medlemmer; komprimering er ikke kildeforenkling.
<!-- END_ROUND5_COUNTS -->

De vigtigste reelle reduktioner er afsluttede estimation/joint/kontrolkampagner
og overlappende modul-ejere. Bayes/remapping, intervalfilter, G7 og H1 har klare
ejere. Båndets CLI og real_data-rapportering flyttes til eksisterende indgange;
det er ansvarsflytning, ikke færre matematiske definitioner. Ingen facade,
researchmappe eller framework er indført.

## Nye korte regressioner

| Kontrol | Ekstern procestid | Verifikation |
|---|---:|---|
| F:1 filtertime+3 forecasttimer | 43.177s | Alle4×22 CSV-felter, origin/final1921-cellers NPY-bytes og PNG eksakte mod runde4. |
| G:legacykonvertering / ny168h-uge | 4.139/4.478s | Intet fit; alle15 numeriske forecastfelter eksakte mod runde4. |

Flyttet F-pakke og nye F/G-resultater er hashkontrolleret via de normale
læsekommandoer. Kilder og output er bevaret i `round5/integration`; tidsbudget
og alle gentagelser står i [status](STATUS.txt) og den lokale tests.json.
Dette er korrekthedsregression, ikke et kontrolleret koldt runtimebenchmark.

Runde3's fulde F-uge genbruges:169×22 felter, origin/final-logstates og PNG
var eksakte;67,888s. Runde4's færdige F-resume bevarede alle resultatbytes.
De originale receipts/sourceZIP'er under recovery/round3 og round4 bevares,
men disse tider påstås ikke at måle den aktuelle version.

Fem hele cleanup-pakker/35 filer er flyttet til `runs/checks`;749 BM-start-
outputfiler har uændrede bytes. Den nye F-regression tilføjer7 filer. Der er
ikke slettet unik evidens. Gamle kilder og forkortede testversioner bevares i
[originaltræet](../../archive/bm/ORIGINALS.md#round5-structure).

## Bevarede områder og begrænsninger

Beskyttede model/control/WM-afhængigheder er uændrede. Native interval,
karakteristisk reference, FV, mixture og EM har forskellige numeriske roller.
Workbookvinduer og signalskalaer har forskellige indlæsningskontrakter;
modelpakker bevarer deres source/populationsbinding og fælles beskyttede brugere.
Unikke eller uklart ejede outputpakker bliver liggende med deres proveniens.

Negative F-målinger, UDEN-understøttelse og fælles faglig accept er korte kendte
grænser i [matematikken](MATHEMATICS.md). Denne runde ændrer ingen modelantagelse
og starter ingen historisk fit- eller sammenligningskampagne.
