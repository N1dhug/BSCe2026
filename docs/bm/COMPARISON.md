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

## Versionsbundne runtimekontroller fra tredje gennemgang

Følgende afsluttede målinger tilhører **runde 3** og deres gemte kildeversioner.
De er genbrugt evidens, ikke målinger af runde 4. Ekstern procestid omfatter
imports, input, beregning og eksport; OS-/JIT-caches blev ikke nulstillet.
Før/efter isolerer derfor ikke kodeeffekten fra cacheeffekten.

| Kontrol | Før | Runde 3 | Verifikation |
|---|---:|---:|---|
| F: 1 filtertime + 3 forecasttimer, fitted/base | 35,876 s | 23,525 s | 4 × 22 CSV-felter og origin/final-logstates eksakte. |
| F: 1 filtertime + 168 forecasttimer, fitted/base | 93,472 s | 67,888 s | 169 × 22 felter, begge 1921-cellers logstates og PNG eksakte; syv kvantilhorisonter. |
| G: samme gemte uge, direkte modelvej | Historisk replay | Paritet kontrolleret | 10 rå numeriske kolonner højst 2,45e-15 forskel; state/state_end/variance eksakte. |
| G: ny uge 7.–14. marts, fitcutoff 28. februar | Tidligere ikke understøttet | 2,674 s | 168 timer; alle 15 numeriske felter eksakte mod første senere-uge-kørsel. |

F's åbne uge bruger kun 21. februar kl. 00→01 som filtertime og gentager
foregående uges priser. Fremtidige targets læses ikke; der er ingen ugefejlmål.
Maksimal masserest er 1,381e-12. Runtimeaccept gælder kun denne opsætning.
Fine-grid, længere filtervindue, flere modeller og historisk fit er ikke
accepteret her. Den positive time løser ikke filterdøgnet med negative målere.

Begge F-kørsler binder den bevarede kildeZIP med SHA256
`971e9d3f6d6feb7d270c5aa0e7a1ebb2785f758b3a056177bd554df069acef67`.
Gamle checkpoints og modelmapper beholder deres kildeidentitet og ommærkes
ikke til ny kode. Genlæsning laver ingen ny prognose.

G-ugen bruger faktisk realiseret Price/Temp fra det afsluttede studie, uden
Demand. Tilgængelighed ved origin er ukendt: det er et betinget historisk
scenarie. Syv separate dagsbaner starter fra frosne historiske medianstates;
der er ingen ny filtrering, refit eller sammenhængende 168-timers statebane.
Kommandoer, logs og hashes findes i `round3/f` og `round3/g` under recovery-
roden i [status](STATUS.txt). Input-/adgangstimeouts tæller ikke som prognosetid.

## Fjerde gennemgang: ny kontrol

<!-- ROUND4_RUNTIME -->
| Ny kontrol på aktuelle kilder | Tid | Resultat |
|---|---:|---|
| F: 1 filtertime + 3 forecasttimer | 47,258 s | Alle 4×22 CSV-felter, 1921-cellers origin/final-states og PNG eksakte mod runde 3. |
| F: resume af færdigt resultat | 13,314 s | Ingen propagation; samtlige resultatbytes uændrede. |
| F: oprindeligt negativt filterdøgn | 14,353 s | Tydelig afvisning med måler, timestamp, råværdi og manglende kanaldefinition. |
| G: valideret legacykonvertering / ny 168h-uge | 4,013 / 4,919 s | Intet fit; alle 15 numeriske forecastfelter eksakte mod runde 3. |
| G / F: hashkontrolleret genlæsning | 2,755 / 1,858 s | Ingen modelberegning eller workbooklæsning. |

F's masserest er højst 5,351e-14. De nye tider er langsommere end runde 3;
der påstås ingen yderligere runtimegevinst. Begge bruger de samme numeriske
valg og inputs som referencerne; maskine/caches er ikke kontrollerede forsøgsfaktorer.
Aktuelle sources ligger i `round4/integration/source.zip`, SHA256
`e1c7f80f7d864db0885cbe9a5a6b42d79a5ff4c83cfbcdb923caafc9e64584ce`.
G's nye deklarerede fitkæde er særskilt testet med syntetisk/mocket fit og
faktisk forecast, inklusive ændret/forkert/manglende binding. Dette er funktionel
evidens; den faktiske historiske uge ovenfor bruger tydeligt legacy-medlemskab.
<!-- END_ROUND4_RUNTIME -->

Nye målinger bindes til kilde/model/input. Test, model og import/inputkontrol
opgøres særskilt; interne faser lægges ikke oven i ekstern procestid.
Ingen ny historisk fitkampagne eller fælles evaluering; budget står i STATUS.

## Aktiv forenkling og faste tællegrænser

Udgangspunktet er `26bdd15`. Runde 4 fjerner syv egne kildefiler og én config:

- F's precise replay/checkpoints samles i eksisterende `bm_1_interval_likelihood`;
  native og uafhængig reference deler kontrakt, mens referenceimports er lokale.
- G7's energy/matched-metoder samles i `bm_1_g7_deterministic`, G8's basis i
  `bm_1_g8`; matematik bevares hos færre ejere.
- De historiske CLI'er `bm_1_h2_precise`, `bm_1_interval_characteristic` og
  `bm_1_parameter_pilot` udfases. Numeriske referencer og faglige tests bevares.
- Den overlappende `f4_week.json` fjernes; periodevalg sker via den almindelige
  config/CLI. Fjernelsen ophæver ingen negativ-inputkontrol.

F's almindelige fitvej kræver ikke længere referenceforsøgenes importkæde.
G's normale vej samler inputvalidering/eksport og viderefører populationsbinding
fra nyt fit til model og output. Ubrugte videresendelser er fjernet, og fælles
kalenders testadapter er flyttet til testene. Ingen ny generisk hjælpearkitektur.

<!-- ROUND4_COUNTS -->
| Kategori | Filer før → efter | Linjer før → efter |
|---|---:|---:|
| Aktiv src-Python med vedligeholdte referencer | 93 → 86 | 31.784 → 31.264 |
| Normale indgange | 3 → 3 | 1.152 → 1.150 |
| Alle ubeskyttede BM-CLI’er | 23 → 20 | 8.390 → 7.902 |
| Tests | 78 → 78 | 15.158 → 15.380 |
| Aktuelle fagnoter | 4 → 4 | 799 → 801 |
| Historiske fagtemaer | 13 → 13 | 2.879 → 2.879 |
| Brugervejledninger | 2 → 2 | 411 → 336 |
| Status, sammenligning, plan og arbejdsordre | 4 → 4 | 647 → 502 |
| Root-henvisninger | 7 → 7 | 120 → 120 |
| Hash-/stioversigter | 5 → 5 | 28.317 → 28.403 |
| Configs og modelskabeloner | 6 → 5 | 363 → 335 |
| Versionsstyrede resultater | 380 → 380 | 27.999 → 27.999 |

Med den tidligere faste 117-afgrænsning: **122 → 115 filer og 38.299 → 37.779
linjer**. Netto syv kilder og 520 linjer færre; 149-scope giver samme reduktion.
Aktive kilders forældremapper: 12 → 12; faktisk fjernede mapper: 0.
Statiske importerede kilder inkl. indgang/packageinit: F4 73 → 72, G 90 → 90,
weekday 102 → 105. Observeret F4-initialimport med identisk kildescope: 59 → 59.
G-bindingen kræver flere kontroller; dens tre filer vokser netto 117 linjer.
Alle lokale outputs: 781 → 788, fordi den nye F-regression tilføjer syv filer.
Ingen startresultater er slettet. OriginalZIP'erne er uændrede: 766 + 2 medlemmer,
143.005 + 407 tekstlinjer, 7.732.773 + 28.729 udfoldede bytes og 2.569.654
pakkede bytes. Bevarede kildepakker tælles særskilt, ikke som slettet kode.
<!-- END_ROUND4_COUNTS -->

149 identiske beskyttede stier udelades på begge sider. Kategorier overlapper;
linjer inkluderer kommentarer. Den tidligere 117-afgrænsning bevares som anker,
så udvidet WM-beskyttelse ikke kan fremstå som kodefjernelse. Logiske mapper
og faktisk fjernede diskmapper er forskellige mål.

Samlet statisk importlukning omfatter også udsatte imports og packageinit;
F4 falder én fil, G er uændret, og fitkalenderen stiger tre. Direkte ejerskab og færre mellemled
er derfor den konkrete forbedring. Importmålinger sammenlignes kun med samme
afgrænsning; den gamle G-forecasts 57 kildefiler er historisk evidens og ikke
en ny runde-4-måling. Beskyttet packageinit og nødvendige referencekæder består.

## Bevarelse og faglige grænser

Originalarkivets 766 medlemmer og nødvendige kildepakker bevares; se
[restorevejledningen](../../archive/bm/ORIGINALS.md). Arkivering er ikke aktiv
kodeforenkling. Runde 4 ændrer ingen rådata, unikke resultater, WM, beskyttede
afhængigheder eller uklart ejet materiale.

G's nye bindingskontrol kan dokumentere bytes og en erklæret population,
ikke råt medlemskab i et aggregeret CSV: status forbliver `declared-unverified`.
Gamle modeller uden oplysninger forbliver `legacy-unverified`. UDEN kræver
eget fit og datakontrakt; et MED-fit kan ikke ommærkes.

Én nødvendig leverandørbeslutning består for F: betyder negative celler
**signeret nettoimport eller fejl/støj i bruttoforbrug**? Indtil kanaldefinitionen
foreligger, bevares afvisning før aggregering. Gaussian-støtte alene begrunder
ingen ændring af observation, baseline eller frossen skala; enhed og tidslabels
er stadig antagelser. Se [MATHEMATICS](MATHEMATICS.md) og den
[bevarede dataaudit](../../archive/bm/F/data-contract.md).

Historisk F-weekday-fit med konvergensaccept kræver særskilt budget. Johans
brugerprøve afventer, og fælles F/G/WM-protokol, kampagne og rangering er udskudt.
