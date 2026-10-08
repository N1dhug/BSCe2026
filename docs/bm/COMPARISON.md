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

## Faktisk runtime og genbrug

Målingerne nedenfor er fulde lokale proceskørsler. Imports, input og eksport
er med i totaltiden. Caches blev ikke nulstillet; særskilt kold kompilering
er ikke målt. Første F-filtertime indeholder workerstart og eventuel JIT.

| Kontrol | Samlet proces | Resultat |
|---|---:|---|
| F opstartsprobe + faseopdelt fuld inputkontrol | 2,713 + 6,850 s | Bestået; ingen modelberegning. |
| F kort fitted/base, 1 filtertime + 3 forecasttimer | 35,876 s | Bestået; 18 fælles CSV-felter og begge 1921-cellers logstates matcher tidligere fitted-reference eksakt. |
| F oprindelig uge med 24 filtertimer | 7,992 s | Afvist før modelstart på negative råmålinger; ingen uge beregnet. |
| F særskilt åben uge, 1 filtertime + 168 forecasttimer | **93,472 s** | Bestået; 7 kvantilhorisonter, ingen fremtidige observationer/scoring. |
| G/WM-timeadapter, gemt 168-timersuge 28. februar | **3,070 s** | Første rundes regression genbrugt; højst 2,45e-15 rå forskel i 20 numeriske kolonner. |
| G gemt snapshot, runde 2 `--check` | 2,470 s | Bestået; ingen propagation eller nyt fit. |

Den åbne F-uge bruger `root/f4_open_week.json`: 21. februar 2022 kl. 00→01
filtreres, derefter prognosticeres 168 fysiske timer fra kl. 01. Priserne
gentager den foregående uge. Den er gennemført under 600 sekunder med
gemt fit, uændret base-grid og normal indlæsning/eksport. Dette beviser
hverken runtime for fine-grid/længere filtrering/to modeller eller ugefejlmål.

| F-fase i den åbne uge | Sekunder |
|---|---:|
| Imports efter indledende standardbibliotek | 1,133 |
| Demand / priser / kontrolsummer | 5,671 / 0,105 / 0,063 |
| Én filtertime inkl. første worker/JIT-arbejde | 12,854 |
| 168 forecasttimer inklusive syv kvantilberegninger | 69,138 |
| Worker-lukning / eksport | 2,004 / 0,960 |

Fasesummen udelader små orkestreringsmellemrum; totalen kommer fra den
eksterne proceskvittering. Maksimal masserest var 1,381e-12. De fulde
kildeversioner er bevaret i `short_source.zip` og `open_week_source.zip`
under recovery-roden i [status](STATUS.txt), sammen med logs og kvitteringer.
De kræves til præcis genindlæsning efter senere kildeændringer.

De tidligere tavse inputforsøg på 526 og 120 sekunder kunne ikke lokalisere
flaskehalsen. Normal terminalkørsel gennemførte nu fuld prepare på 4,328 s
inden for procesmålingen på 6,850 s. En forskel i proces-/adgangsmiljø er
sandsynlig, men er ikke bevist ved en kontrolleret årsagssammenligning.
Der er ikke foretaget en ny lempelse af inputvalideringen for at opnå tiden.

## Konkrete databegrænsninger

Én afgrænset read-only kontrol på 6,221 s lokaliserede den afviste gamle uge:
122 negative MED-celler i 44 af dens 192 filter-/prognosetimer, ingen uendelige
eller manglende værdier. Filterdøgnet har 17 negative celler fordelt på seks
timer; prognoseperioden har 105 i 38 timer. Kun tre gruppemidler er negative,
men F's eksisterende kontrakt kontrollerer hver valgt måler før aggregering.
Man må derfor ikke erstatte kontrollen med en kontrol af gennemsnittet alene.

Første fejl er 21. februar kl. 10; minimum er −4,02 i `5080166_MED` den
27. februar kl. 12. De præcise lokale celler er registreret i
`round2/f_runtime/week_data_findings.json`. Workbookmetadata og eksisterende
noter afgør ikke nettoeksport kontra målefejl. Hverken fortegn, manglende
værdier eller målerpopulation er ændret. F's fejlmeddelelse viser nu tid,
kolonne og rå værdi. Den åbne uge bruger en gyldig filtertime og åbner ingen
fremtidig demand; den er ikke en skjult reparation af de afviste data.

Enhed, intervalstart/slut og demandtidszone er fortsat arbejdsantagelser.
G's rankmangel, baselineklipning, ekstrapolation og empiriske bånd er bevaret
som begrænsninger. Der er ingen ny fysisk identifikation eller ny støjfamilie.

## Kode- og noteoprydning

Starten på denne runde er mergecommit `7249bf9` fra PR145. Første runde gav
1052→1038 ubeskyttede filer og 141→143 aktive kildefiler. Den brede anden
gennemgang bruger samme konservative kildeafgrænsning: alle ubeskyttede
Pythonfiler under `src`, også forsøgs- og valideringskode.

<!-- COUNT_TABLE -->
| Afgrænsning | Filer før → efter | Tekstlinjer før → efter |
|---|---:|---:|
| Aktiv BM-kode | 143 → 133 | 40.807 → 39.653 |
| Aktive noter og kontrolindgange | 15 → 15 | 3.220 → 2.920 |
| Løst historisk materiale og arkivmetadata | 405 → 32 | 50.898 → 33.761 |
| Tests | 80 → 81 | 14.816 → 15.090 |
| Versionsstyrede genererede resultater | 382 → 382 | 28.000 → 28.000 |
| Øvrig konfiguration/reference | 13 → 10 | 13.418 → 521 |
| Hele løse ubeskyttede træ, inkl. ZIP-beholder | 1038 → 653 | 151.159 → 119.945 |
| Heraf root med undermapper, ubeskyttet | 409 → 22 | 63.411 → 4.599 |
| Heraf direkte filer i root, ubeskyttet | 73 → 17 | 27.187 → 1.063 |
<!-- END_COUNT_TABLE -->

117 beskyttede WM-filer er udeladt i begge kolonner. Versionsstyrede
resultater holdes særskilt; ingen resultater eller rådata er slettet. Antallet
af løse filer beskriver det træ, man navigerer i. Det er **ikke** et globalt
deduplikeringstal: originalpakken tælles nedenfor også udfoldet.

Den konkrete faglige sammenlægning er 61 noter med 13.418 linjer til ni
fagtemaer med 2.434 linjer og otte kompatibilitetshenvisninger med 135 linjer.
Der blev gennemgået 364 lignings-/algoritmeblokke og 475 afsnitsoverskrifter.
Unikke KKT-, covariance-/SE-, LS-gap-, basis- og intervaludledninger er med;
negative numerik- og identifikationsresultater er bevaret. Tre tidligere
G-arkivtemaer er indarbejdet i de nye, frem for at ligge som dubletter.
De tre aktuelle G-noter er desuden strammet fra 1.051 til 583 linjer med
43 bevarede ankre. Gamle kørselsordrer er fjernet; unik matematik, faktiske
retry-/budgetforløb og negative resultater står som dateret evidens.
En arvet notation er præciseret fra a=.5 til historisk x0=.5; den fit-kurve
havde g(.5) tæt ved −1. Det er en dokumentrettelse, ingen ændring af modellen.

336 afsluttede root-forsøgsfiler er **arkiveret**, ikke kaldt deduplikeret:
113 Python, 216 JSON, 5 PowerShell, én SVG og én log. Originalpakken indeholder
766 filer i deres eksakte gamle stier, 7.732.773 bytes udfoldet / 2.560.128
bytes ZIP. Alle rå hashes, Git-korrespondance og faktisk sikker gendannelse
er kontrolleret. Input/output følger ikke med denne kildepakke; de er bevaret
i den tidligere verificerede lokale backup. [Stikort og indeks](../../archive/bm/INDEX.md)
fører fra gamle filer til emne, aktiv ejer og originalversion.

Den egentlige kodesammenlægning samler Bernstein-basis, G7-jobbudget,
response-job og sole-owner syntetisk fixture hos deres eksisterende ejere.
Tre S3-CLI'er bliver én med eksplicit `interval`, `residual` og `point-process`.
To afløste faste studiedrivere og tre gamle H2-kampagneindgange er udfaset
fra den aktive kode; genbrugelige numeriske metoder og alle tests beholdes.
Rene curve- og metrics-importer trækker først solver-/optimizerkoden ind ved
brug. Ved selve konsolideringen var 17 flyttede/berørte funktioner AST-identiske
bortset fra imports og den private fixtures navn/docstring. Den efterfølgende
F-integration tilføjer en eksplicit native-gren i responskoden; de seks gamle
native beregningsfunktioner er stadig uændrede. Begge trin har egne tests.

Fire F-historiktemaer bevares om konkrete ledgers, terminale fejl, admission og
H→I/J. De tre aktuelle G-noter beskriver senere interval/residual/weekday-
protokoller, forskelligt fra punktmodelarkivet. `root/f4` beholder TeX-kilder
og buildindgang med aktive CI-/testkontrakter. G7/G8, F's numeriske referencer
og testharnesses har forskellige matematiske ansvar og er ikke samlet i
én stor modelfil. WM og fælles afhængigheder er urørte.

<!-- IMPORT_COUNTS -->
Det samlede roottræ inklusive syv beskyttede filer er **416 → 29**; to nye
genbrugelige F-konfigurationer kommer oven i strukturdelens 27 filer.
Original-ZIP'ens 766 medlemmer omfatter 656 ubeskyttede filer med 123.159
tekstlinjer og 110 beskyttede kildefiler. De er bevaringskopier, ofte også af
beholdt kode, og må hverken regnes som én forenklet kilde eller lægges oven i
den aktive kode som selvstændige modeller. Arkivets løse linjetal inkluderer
maskinlæsbare hash-/stikort; det er ikke alene fagprosa. Øvrig-kategoriens
reduktion omfatter gamle G-kildebundter i originalpakken og to nye F-configs.

Samme konservative AST-kortlægning på begge versioner giver F4's statiske
importlukning **72 → 73** filer, gemt G-CLI **90 → 89** og G-exportkernen
**78 → 78**. Imports inde i funktioner tælles også med. Den nye F-integration
tilføjer ansvar; der påstås ingen dramatisk reduktion af den samlede modelkæde.
De rene curve-/metrics-kald indlæser først tunge solvere, når de bruges.
<!-- END_IMPORT_COUNTS -->

## Verifikation og resterende arbejde

- Kodekonso: 29 relevante tests; F-faser: seks runner-tests og målrettet
  fejlkonteksttest. Første rundes 57 integrations- og 16 inputtests genbruges.
- F's korte numeriske regression og fulde åbne uge er verificeret; G's
  eksisterende uge/snapshots genbruges. Ingen seksugerskampagne genkørt.
- WM's 117 rå hashes er uændrede. Arkivrestore, kildeidentiteter, stier og
  dokumentlinks kontrolleres efter ændringerne. Gamle hashguards omgås ikke.
- F4 pause/resume og completed-resume er testet; G genbruger snapshots og
  har ingen deltime-resume. Ny F-weekday-integration dokumenteres særskilt
  i [matematikoversigten](MATHEMATICS.md), med korte tests og uden historisk fit.
- Fælles F/G/WM-evaluering og rangering er **udskudt af Johan**. Et korrekt
  input-/populationsgrundlag og særskilt fitbudget kræves før en sådan kampagne.
- Afklaring af negative råmålinger og leverandørmetadata samt oprindelig
  H/I/J-accept og uafsluttet firestykkefit er fortsat åbne faglige spørgsmål.
- Johans faktiske brugerprøve **afventer**. Den korte prøve står i
  [BM-guiden](../../root/BM_GUIDE.md); den automatiske kontrol erstatter ikke
  hans vurdering af læsbarhed og brugervenlighed.
