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
| Skala | Frossen historik til 16. februar | Femugershistorik | Samme som G | Fuldperiodens loader-skala |
| Prognosestart | Filtreret state fra Beta-prior | Syv historiske medianstarts ved mandag | Egne medianstarts | WM's egen initialiseringspraksis |
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

## Tid, genbrug og accept

G's tidligere log viser samlet optimizer/checktid 285.825 s for seks origins
og 24.322 s numerisk export. Det er ikke fuld kold/varm brugerkommando.
F's 600-sekunders pausegrænse beviser ikke, at en uge afsluttes under ti minutter.
En konkret saved-fit-regression for G/WM-timeadapteren, 28. februar 2022,
gennemførte 168 timer på **3.070 s** inklusive processtart/import, indlæsning,
forecast, scoring, figur og eksport. Forecastdelen tog 0.231 s. De 20 gemte
numeriske kolonner matcher før-versionen med højst $2.45\cdot10^{-15}$ rå
forskel. Det opfylder 600-sekundersmålet for denne gemte uge, ikke alle perioder
eller F. Ingen fit eller historisk replay blev genkørt. Sidste input-/receipt-QA
ændrede kun wrappervalidering/metadata; kilderne for numerikken er identiske.

Korte kontroller og den faktiske F-inputtid registreres i [status](STATUS.txt).
F's workbook-scan er en del af brugerens opstart, selv ved `--check`, og kan
ikke skjules bag den tidligere 46.465 s modelmåling.

Der er ikke bevilget et nyt historisk fitbudget eller samlet benchmarkbudget.
En eventuel næste måling skal afgrænses til én gemt model, navngiven periode,
ny outputmappe og et konkret tidsloft. En ny F-weekday-estimator er en særskilt
implementering med validering og fitbudget; [matematikoversigten](MATHEMATICS.md)
forklarer hvorfor en callback omkring det faste fit er utilstrækkelig.

## Fund og restarbejde

| Klasse | Fund / handling |
|---|---|
| Rettelse | F-enhedsmærkning er betinget; direkte datovalg og interval/UTC-output er tilføjet. |
| Nødvendig funktion | Gemte G-exports bruges direkte; generisk båndbrug er tilføjet med egne receipts. |
| Nødvendig senere funktion | Rigtig native F-weekday-estimator og fælles F/G/WM-resultat. |
| Begrænsning | Kildeenhed/hour-start mangler verificeret leverandørmetadata; ingen fysisk accept. |
| Begrænsning | F's fulde uge-runtime og brede prognosekvalitet er ikke fastslået. |
| Begrænsning | Gamle direkte G-input kan reindexere ugyldig DST-time væk; ny indgang validerer. |
| Valgfri forskning | Ny baseline, fysisk identifikation, andre støjlag og blind evalueringskampagne. |

<!-- FINAL_ACCEPTANCE -->
## Konkret før/efter og accept

Opgørelsen bruger udvidet kildebase `89e8337` og den aktuelle arbejdsgren;
117 beskyttede WM-stier er udeladt i begge kolonner. Tekstlinjer tælles i
ukomprimerede filer. En backup-ZIP regnes aldrig som kodeforenkling.

<!-- COUNT_TABLE -->
| Afgrænsning | Filer før -> efter | Tekstlinjer før -> efter |
|---|---:|---:|
| Aktiv BM-kode | 141 -> 143 | 40298 -> 40807 |
| Aktive noter/kontrolindgange | 8 -> 15 | 8447 -> 3220 |
| Historisk materiale/arkiv | 430 -> 405 | 56147 -> 50898 |
| Tests | 78 -> 80 | 14479 -> 14816 |
| Versionsstyrede genererede resultater | 382 -> 382 | 28000 -> 28000 |
| Øvrig konfiguration/reference | 13 -> 13 | 13418 -> 13418 |
| Hele ubeskyttede kildetræ | 1052 -> 1038 | 160789 -> 151159 |
| Heraf direkte filer i root/ | 109 -> 73 | 39815 -> 27187 |
| Lokalt resultatlager (separat) | 1411 -> 1428 | 1329827 -> 1330893 |
<!-- END_COUNT_TABLE -->

Aktive noter omfatter også startprompt, status, filplan og det nye maskinlæsbare
WM-manifest. Historisk materiale omfatter kildebundne scripts på deres gamle
stier. Kategorien aktiv BM-kode tæller konservativt alle ubeskyttede Pythonfiler
under src, også genbrugte forsøgs-/valideringsindgange. Resultatlageret tælles
separat og lægges ikke oven i kildetræets tal. Råfiler er ikke slettet.

De 36 udvalgte noter med 7009 linjer er samlet i 10 emnefiler med 1730 linjer;
indeks/stikort kommer oveni. Det er en afgrænset deduplikering, ikke den samlede
filreduktion. AGENTS er 1803 ->45 linjer, F4-indgangen 4135 ->42. Ingen fil er
blot flyttet for at tælle som global reduktion. Otte G-guider bliver tre større
fagnoter med bevaret matematik; her hævdes kun fem færre filer, ikke færre linjer.

De centrale statiske importlukninger er F4 **72 ->72** filer (47 beskyttede)
og G-weekday-export **88 ->88** (56 beskyttede). Ny G-CLI har90 inkl. indgangen;
den bruger den eksisterende kæde. Antallene er en konservativ AST-kortlægning,
ikke dynamisk runtime-tracing. [Matematikoversigten](MATHEMATICS.md) viser ansvar
og konkrete funktioner. De brede imports er fortsat en læsbarhedsbegrænsning.
WM-fællesmoduler, native produktionsnumerik under validation og sourcebundne
G-kilder kan ikke flyttes/slettes som almindelige døde hjælpefiler. Historiske
root/f4- og root/g7-scripts bevares pga. stier, ledgers og reproduktionskontrakter.

| Krav | Evidens | Status og restarbejde |
|---|---|---|
| Struktur/læsbarhed | Kort guide, 36 noter samlet, før/efter og importkæder | Delvist bestået; bred kode/importforenkling ikke opnået. |
| Brugervenlighed | Help, dato/cutoff/inputfejl, G-eksempel og alternativt udsnit kontrolleret | Delvist bestået; fuld F-inputkontrol og Johans brugertest afventer. |
| Matematisk/numerisk regression | 57 integrerede tests; derefter16 berørte tests; G-output afvigelse <=2.45e-15 | Bestået for kontrolleret scope; ingen fysisk enheds- eller generel modelvalidering. |
| WM-beskyttelse | 117 uændrede hashes, inkl. nye MPC/fælles afhængigheder | Bestået; WM urørt. |
| Reproducerbarhed | Verificeret3.28GB backup/bundles; F215 sourcehashes+origin-load, G316 packagehashes | Bestået bevaringsprøve; gamle F-runs bruger original source. |
| Pause/genindlæsning | F kort kernelcase matcher uafbrudt forløb; completed-resume bevarer bytes/mtime | Bestået F-test; G har snapshotgenbrug, ikke deltime-resume. |
| Sammenlignelighed | Samme G/WM-timeadaptermasker; F-protokolforskelle navngivet | Ikke bestået fælles F/G/WM-tabel/figur; mangler korrekt F-adapter og fælles dataaftale. |
| Runtime | G168 timer3.070 s; F fuld inputkontrol stoppet ved120 s | G bestået for én gemt uge; F-målet ikke dokumenteret. |
| Prognosekvalitet | Seks gemte udviklingsorigins; fejl/båndbredde/negative resultater bevaret | Ingen generel vinder eller blind test; F ikke fællesevalueret. |

Den seneste F-inputkontrol brugte det ene godkendte120-sekundersbudget og
blev stoppet automatisk uden modelberegning. Tomt output kan ikke lokalisere
stoppet; første fremdriftslinje ligger efter imports. Den afgrænsede100-rækkers
cacheprofil er forbedret, men dette beviser ikke fuld opstarts- eller uge-runtime.
Der gives derfor ingen samlet påstand om, at alle slutkrav er opfyldt.

Johans korte brugerprøve følger [BM-guiden](../../root/BM_GUIDE.md): læs help,
brug G-eksemplet, vælg24 timer og en ny mappe, find CSV/figur/parametre/metrics,
og prøv en dato uden for snapshotugen for den kontrollerede fejl. F's
pause/genoptagelse er testet syntetisk; en ny fuld lokal F-kørsel kræver først
afklaring af opstart/inputtid. Johans egen vurdering registreres som afventer.
