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

## Kontrolleret runde efter PR146

Udgangspunkt: ren main `09c602f327a3497a763c47e734fac4a38b12227d`, fetchet
9. oktober 2026. Alle ændringer ligger på den lokale arbejdsbranch i
[status](STATUS.txt). Den eksisterende store backup genbruges; 76 nye/ændrede
lokale filer er sikret som verificeret delta, med kildeZIP og all-refs bundle.
WM-beskyttelsen er udvidet fra 117 til 149 stier ved en konservativ gennemgang
af eksisterende indirekte afhængigheder. Ingen af disse 149 filer er ændret.

## Faktisk prognose og runtime

Hele processens tid omfatter Python/imports, input, beregning og eksport.
OS-/JIT-/bibliotekscaches er ikke nulstillet. Før/efter er samme maskine og
samme input, men isolerer derfor ikke effekten af kodeændringer fra cacheeffekt.

| Kontrol | Før | Efter | Verifikation |
|---|---:|---:|---|
| F: 1 filtertime + 3 forecasttimer, fitted/base | 35,876 s | 23,525 s | Alle 4 rækker × 22 CSV-felter og origin/final 1921-cellers logstates eksakte. |
| F: 1 filtertime + 168 forecasttimer, fitted/base | 93,472 s | **67,888 s** | Alle 169 rækker × 22 felter og begge logstates eksakte; syv kvantilhorisonter. |
| G: samme gemte uge, ny direkte modelvej | Historisk replay | Verificeret | 10 rå numeriske kolonner højst 2,45e-15 forskel; state/state_end/variance eksakte. |
| G: senere uge 7.–14. marts, fitcutoff 28. februar | Ikke understøttet af tidligere saved-CLI | **2,674 s** | 168 faktisk beregnede timer med frosne parametre og nye kalenderinputs. |

F's korte og lange PNG'er er byteidentiske med de tidligere visuelt
kontrollerede figurer. Den åbne uge har maksimal masserest 1,381e-12.
Den bruger 21. februar kl. 00→01 som eneste filtertime og gentager den
foregående uges priser. Fremtidige observationer åbnes ikke, og ugefejlmål
beregnes ikke. Dette dokumenterer runtime under ti minutter for netop denne
opsætning. Fine-grid, længere filtervindue, flere modeller og historisk fit
har ikke fået en ny runtimeaccept.

| F-fase, den åbne uge | Sekunder |
|---|---:|
| Instrumenterede imports efter standardbibliotek/packageinit | 0,936 |
| Demand / priser / inputhashes | 3,636 / 0,097 / 0,028 |
| Én filtertime, inklusive workerstart | 7,891 |
| 168 forecasttimer med syv kvantilhorisonter | 51,301 |
| Worker-lukning / eksport | 2,007 / 0,596 |

Fasesummen har mindre scope end den eksterne procesmåling. Den uændrede
originalkildepakke til begge F-målinger har SHA256
`971e9d3f6d6feb7d270c5aa0e7a1ebb2785f758b3a056177bd554df069acef67`.
Gamle checkpoints kræver deres egne kildeversioner; de ommærkes ikke til
senere små statusrettelser. Resultatgenlæsning verificerer outputhashes uden
nyt modelarbejde. En direkte F-scriptgenlæsning tog 0,165 s med numeriske
biblioteksimports blokeret; `python -m` har stadig beskyttet packageinit.

G's nye uge bruger de faktisk realiserede Price/Temp fra det afsluttede
studies 7. marts-input, uden Demand. CSV'en er hverken forskudt eller syntetisk,
men tilgængelighed ved origin er ukendt: dette er et betinget historisk
scenarie. Syv selvstændige dagsbaner bruger frosne historiske medianstarts.
Der sker hverken refit, filtrering af nye observationer eller et kontinuerligt
168-timers stateforløb. Den gamle `replay` bevarer G/WM-timeadapterens faste uge.
Samme-uge-paritet og senere-uge-funktionalitet er særskilte kontroller.
Før den sidste statusrettelse tog samme senere uge 2,506 s. Derefter blev
originalsnapshot valideret og konverteret til en ny modelmappe; den endelige
2,674 s-kørsel har alle 15 numeriske CSV-felter eksakte mod første nye uge.
Gamle sourceidentiteter og modelmapper er bevaret.

Præcise kommandoer, logs, input-/sourcehashes og kvitteringer ligger i
`round3/f` og `round3/g` under den lokale recovery-rod i [status](STATUS.txt).
De tidligere inputtimeouts på 526/120 s og denne rundes sandboxprobe på
15,046 s er adgangs-/inputforsøg, aldrig målt prognosetid. Ingen validering
er lempet for at opnå tiderne.

## Data og populationsbinding

Metadata i den uændrede rå workbook er kontrolleret uden at genlæse dens
observationer: **MED er 13 installationer med elvarme; UDEN er 140 uden
elvarme**. F's præcise 13-målerliste matcher MED. Solceller er ikke
dokumenteret af gruppenavnene. Enhed, netto/brutto, label-start/slut og
demandtidszone er fortsat uafklarede leverandørforhold.

Den genbrugte diagnose af F's oprindelige 192 timer har 122 negative råceller
i 44 timer; filterdøgnet alene har 17 i seks timer. Kun tre gruppemidler er
negative. F modellerer ét fast gruppemiddel, ikke 13 separate likelihoods.
Latent intervalforbrug ligger i [0,1], mens additiv Gaussian-observation har
støtte på hele R. Hver-måler-guard er således en arvet forbrugsdatakontrakt,
ikke en sætning om likelihoodens støtte. Den bevares; ingen klipning,
udeladte målere, offset eller skjult middelkontrol er indført.

Ved dokumenteret nettoimport kræves en udtrykkelig signeret modelkontrakt,
f.eks. et særskilt generation/eksportled; ved dokumenteret målefejl kræves
kvalitetspolitik og genvalidering af træning/baseline. Disse valg kræver
faglig afklaring. Den åbne positive filtertime løser ikke det oprindelige
filterdøgn. G's affine normalisering kan give negative rå værdier, men
afklarer heller ikke nettoforbrug eller fysisk batteriidentifikation.

F binder gruppe, unik meterliste, modelhash, skala, baseline og cutoffs;
nyt weekday-fit arver kun tidligere baseline/skala/prior og får egne fits.
G's nye fitindgang kræver en hashbundet CSV- og populationserklæring;
råmedlemskab for et allerede aggregeret CSV kan kun være erklæret og
uverificeret. UDEN afvises i den gamle MED-specialiserede fitkerne.
Eksisterende G-snapshots mangler målerlisten og forbliver `legacy-unverified`.
Deres parametre ommærkes aldrig til en ny gruppe.

## Kode, dokumentation og bevarelse

<!-- ROUND3_COUNTS -->
Samme 149 beskyttede stier er udeladt i begge kolonner. Delmængder overlapper;
rækkerne må ikke lægges sammen. Fysiske tekstlinjer inkluderer kommentarer.

| Kategori | Filer før → efter | Linjer før → efter |
|---|---:|---:|
| Aktiv src-Python inkl. vedligeholdte referencer | 104 → 93 | 33.138 → 31.784 |
| Heraf normale indgangsmoduler | 3 → 3 | 871 → 1.152 |
| Alle ubeskyttede BM-CLI’er | 35 → 23 | 10.244 → 8.390 |
| Tests | 77 → 78 | 14.634 → 15.158 |
| Aktuelle fagnoter | 4 → 4 | 815 → 799 |
| Historiske fagtemaer | 13 → 13 | 2.879 → 2.879 |
| Brugervejledninger | 3 → 2 | 535 → 411 |
| Status, sammenligning, plan og arbejdsordre | 4 → 4 | 757 → 647 |
| AGENTS | 1 → 1 | 45 → 23 |
| Root-henvisninger | 9 → 7 | 177 → 120 |
| Maskinlæsbare hash-/stioversigter | 4 → 5 | 27.778 → 28.317 |
| BM-konfigurationer og modelskabeloner | 5 → 6 | 351 → 363 |
| Versionsstyrede genererede resultater | 382 → 380 | 28.000 → 27.999 |
| Root med undermapper (overlap) | 22 → 12 | 4.599 → 3.656 |

Med den tidligere uændrede 117-afgrænsning er kode **133 → 122 filer og
39.653 → 38.299 linjer**: 11 filer og 1.354 linjer færre. Udvidet WM-beskyttelse
regnes ikke som kodefjernelse. Inklusive beskyttede BM-CLI’er er antallet
42 → 30, tests 81 → 82 og hele root 29 → 19. De tre normale indgange bevares;
de vokser 281 linjer med ny forecast, gruppebinding og resultatgenlæsning.
Det forklarer en del af forskellen mellem 12 pensionerede drivere og nettoreduktion.

Arkivets oprindelige ZIP er uændret:766 medlemmer,143.005 tekstlinjer,
7.732.773 bytes udfoldet/2.560.128 bytes ZIP. Tillægget har 2 medlemmer,407 linjer,
28.729 bytes udfoldet/9.526 bytes ZIP. De tælles som bevaring, ikke fjernet kode.
Maskinoversigternes linjer er hashes/stikort, ikke fagprosa. De fem F-configs
er flyttet med samlet 351 linjer; den nye G-skabelon tilføjer 12 linjer.

Konservativ statisk importlukning, også udsatte/legacy-imports og packageinit:
F4 **73 → 73**, G-CLI **89 → 90**, kalender-CLI **102 → 102** filer.
S3 og gammel G-export er fortsat 84/78. Dette er ingen stor reduktion af hele
afhængighedsnettet. En normal F4-import indlæser 66 → 63 projektmoduler (heraf
62 → 60 faktiske kildefiler), målt mod det komplette 09c-træ. G’s gamle
runner+eksportejer mod aktuelle afhængigheder indlæser 81 kildefiler; ny faktisk
forecast 57 (66 → 43 uden packageinit). G-tallene sammenligner to forskellige
normalveje og er ikke en identisk importmikrobenchmark.

Yderligere samling begrænses af beskyttet src-packageinit/shared WM-kode,
forskellige numeriske referenceansvar, historiske kildebindingskontrakter og
aktive tests. F/G-fit og almindelig forecast deler ikke samme observations-
eller initialiseringsmodel; de er derfor ikke presset ind i én stor solver.

Lokale outputfiler stiger 755 → 774 med nye ignorerede regressionsresultater
og konverterede modelkvitteringer. Den faktiske dubletfjernelse er alene de
tre nedenfor; nettotællingen præsenteres ikke som deduplikering.
<!-- END_ROUND3_COUNTS -->

F's tre likelihoodprimitives har én kort numerisk ejer; den normale modelvej
importerer ikke længere karakteristik-/terminalpiloternes testapparat.
Alle ti native definitioner og tre flyttede primitives er AST-identiske.
`advance_hour` gør det normale intervaltrin synligt. Xi kontrolleres korrekt
i [0,1], gruppefejl afvises tidligt, modelbytes kontrolleres mod ændringer
under klargøring, og CSV/PNG publiceres atomisk med outputhashes.

G har en kort direkte prognoseejer med ligninger, frosne dagstilstande,
affin skala, cutoff og 60/120-forfiningskontrol. Legacy-konvertering validerer
de gamle 19 kildebindingsfiler; normal forecast importerer ikke historiske
S3-/fitdrivere. Bånd og legacy-output bevarer deres tidligere kontrakter.
F/G kaldes gennem tre normale `python -m`-moduler; handlingerne fit,
prognose, replay og genlæsning er forklaret i [guiden](README.md).

Tolv faste forsøgsdrivere er taget ud af aktiv kode: fire F/H2 og otte G7/G8.
Genbrugelige metoder, negative forskningsresultater og referenceberegninger
bevares. Tre CN-konstanter/funktioner er flyttet AST-identisk til deres
validationsejer, så en reference ikke afhænger af en udfaset CLI.
Den eksisterende originalpakke bevarer 766 råfiler. To senere ændrede G-
drivere kræver et lille tillæg med to filer; det er et delta, ikke et nyt
selvstændigt kørbart arkiv. [Restorevejledningen](../../archive/bm/ORIGINALS.md)
angiver præcis version og nødvendige lokale data.

Fem configs er samlet i `configs/bm`. Én startguide og én specialvejledning
erstatter tre overlappende rootguider. To overflødige H2-henvisninger er
fjernet; seks G-henvisninger og F4 bevares, hvor beskyttede dokumenter linker
til dem. Unik tematisk matematik fra den foregående oprydning er bevaret.

Resultatinventaret omfattede 755 filer/3,105 GB. Af 91 ekstra bytekopier i
58 grupper er kun tre fritstående aliaser slettet, i alt 400.194 bytes,
efter kontrol af keeper, fuld backup og manglende aktive referencer. To var
versionsstyrede. De øvrige 88 kopier i 55 grupper har manifestbinding,
uafhængig forsøgsproveniens eller uklart ejerskab og bevares. Unikke outputs,
rådata og WM-materiale er urørte. ZIP-arkivering regnes særskilt fra aktiv
kodeforenkling.

## Verifikation og nødvendigt restarbejde

Berørte numeriske tests, negative guards, identitets-/resume-tests og faktiske
F/G-prognoser er kontrolleret. Færdige F-weekday-fits og dagsprognoser genbruges
ved eksplicit resume med samme kilde/runtime/config/dataidentitet. Ufærdige
fitforsøg bevares, men optimizerens interne position genoptages ikke. Alle
forecasts fryses før scoringtargets. Der er ingen automatisk retry eller
skjult forøgelse af fitbudgettet. Præcis test- og budgetstatus står i STATUS.

Nødvendigt før fælles faglig accept:
- Leverandørafklaring af rå enhed, netto/brutto, negative værdier og tidslabels;
  derefter en bevidst F-observations-/datapolitik og verificeret G-målerliste.
- Et særskilt autoriseret historisk F-weekday-fitbudget og konvergensaccept.
- Johans [korte brugerprøve](README.md#kort-brugerprøve-og-dokumentation), som
  afventer; automatiske checks erstatter ikke hans vurdering.
- En senere fælles F/G/WM-protokol med samme population, periode, observation
  og information. Kampagnen og modelrangeringen er fortsat udskudt.

Valgfri forskning er videre identifikation, andre støj-/netloadmodeller,
nye parameterbanker og stærkere usikkerhedskalibrering. Ingen af disse er
startet. Historiske H/I/J- og firestykkefitnegative resultater er bevaret som
evidens; de er ikke gjort til en skjult fortsættelsesordre.
