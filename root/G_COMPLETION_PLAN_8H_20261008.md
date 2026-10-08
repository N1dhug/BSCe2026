# Aktiv G-plan: færdig grundversion inden for otte timers arbejde

Seneste præcisering fra Johan: Førsteprioriteten er fungerende G-kode,
som kan køres, undersøges og senere udvides med BM2 eller flere parametre.
G optimeres med metoder egnet til G inden for tidsrammen. WM er en konceptuel
reference, ikke et krav om 1:1-behandling. Solvere, fitkriterier, intern
skalering og modelspecifikke checks må variere med en kort faglig begrundelse.
En sammenligning skal have sammenlignelig opgave, information og horisont
og vurderes i fælles enheder på et tydeligt observationsgrundlag.
Første milepæl er en fungerende deterministisk grundversion. Støjlaget har
særskilt status og må ikke blokere denne milepæl alene af hensyn til WM.
Variansfit er en kandidat, ikke en binding til WM's metode. Ingen lange
metodesøgninger, fuld WM-paritet eller generel BM2-arkitektur nu; enkle
modulgrænser og et versionsmærket snapshot er tilstrækkeligt.

8. oktober 2026. Denne samlede plan erstatter de tidligere prioriteringer
af obligatorisk parameterbank/demo. Den bygger på Johans seneste instrukser,
[kode-/matematikreviewet](G_CODE_MATH_REVIEW_20261008.md) og
[gennemgangen af faktisk WM-kode](G_WM_ALIGNMENT_REVIEW_20261008.md).
Review og afgrænsning er udført. Den selvstændige deterministiske S3-vej
er nu implementeret og kørt; se [brug og resultat](G_S3_STANDALONE_GUIDE_20261008.md).
Et afgrænset, numerisk kontrolleret proces-scenarielag og slutgennemgang er
nu udført; [brug, matematik og afvisningsstatus](G_S3_UNCERTAINTY_GUIDE_20261008.md).
Historisk støjkalibrering er afvist i udviklingseksemplet. Fysisk kalibreret
observeret totalusikkerhed er fortsat åben. Ingen ny G7-accept er erklæret.
Johan har nu præciseret, at målingen dækker en hel time, og at Demand efter
hans erindring er Power. Punktoperatoren er derfor fortsat en mærket
udviklingsapproksimation; intervaloperator og præcis fysisk enhed er næste
faglige opgave, ikke en skjult ændring af den afleverede S3-model.

## 1. Mål og faste valg

G skal have et deterministisk grundlag, en brugbar historisk prognose og
et afgrænset stokastisk lag indført relativt som i WM. Input, output og
relevante checks skal gøre sammenligning let. F's undersøgelser er ikke
automatisk G-opgaver; F's kode, resultater og budgetter ændres ikke her.

- Bevar **S3, lambda=0**: lineær f, to-stykkers g, fælles C/k/Xi/a fra
  de fem foregående kalenderugers 35 dagsforløb. Ingen ny modeludvælgelse.
- Hver historisk bane starter x=.5 og har 24 fysiske warm-up-timer.
  De syv forecaststarts er historiske medianstates for hver ugedag.
  Alle syv dagsprognoser fryses mandag; de er ikke én kontinuerlig ugebane.
- Bevar G's SDE med diffusion σ_x x(1−x), timer og frosne deterministiske
  parametre. WM er metodisk reference; dens additive R-støj kopieres ikke
  ind i G som en ny model. Start med procesusikkerhed betinget på P0=0.
- Parameterbank/M254/filter→styring er et **bispor**, ikke et færdigkrav
  eller en automatisk anvendelse af tidsreserven. Detaljerne er bevaret i
  [udvidelsesplanen](G_PARAMETER_BANK_EXTENSION_20261008.md).
- Eksisterende deterministisk prisstyring og fixed-target-sammenligning
  bevares. Udsættelse af banken fjerner ikke dette udførte arbejde.

## 2. Først: selvstændig historisk S3-prognose

Byg et lille fit → komplet snapshot → predict → valgfri score-forløb,
der genbruger de eksisterende matematiske kerner. Den konkrete
[arbejdsinstruks](G_NEXT_STEP_20261008.md) er klar til næste trin.

Snapshot gemmer protokol/version, cutoff, population, træningsmaske,
råskalaer/enheder, prisdomæne, baselinekoefficienter og standardisering,
alle baselineindstillinger, C/k/Xi/a, syv egne starts, grids, forfiningsstatus
og input-/kildehashes. Den gamle januar/lineære snapshot-type erstatter
ikke et S3-snapshot. Metadata alene er ikke et komplet baselinefit.

Forecast kræver kun snapshot og fremtidige priser/temperaturer. Ingen refit
og ingen fremtidig demand. Observationer lægges til ved scoring. Kontroller
roundtrip, fravær af fitkald, uafhængighed af fremtidige observationer,
korrekte starts, kalender/DST, endelige tal og coarse/fine-demandfejl ≤1e-5.
Ikke-understøttede kalenderdata afvises tydeligt; timer må ikke tabes skjult.

Lever ét faktisk historisk udviklingseksempel med CSV/JSON, figur og
numerikstatus. Genbrug komplet verificeret gammelt fit, ellers ét
budgeteret S3-fit med lambda=0. Workbooks findes i Git, men temperaturcache
og fuld gammel fitpakke er endnu ikke verificeret lokalt. Implementeringens
små kontraktchecks kan udføres, mens dataadgangen afklares.
Januar og uger 6–18 er allerede inspiceret og er ikke nye blinde testdata.

## 3. Dernæst: et enkelt støjlag egnet til G

WM tilpasser q=σ² til empirisk varians efter deterministisk fit. G6a
tilpassede σ til standardafvigelse; det er et andet problem. Den gamle
G6a-underkendelse bevares og kan ikke omdøbes til ny accept.

En billig foreløbig kandidat er et særskilt variansfit med én q per mandagsfrossen uge,
de 35 historiske forløb og P0=0 ved hver forecaststart. Empirisk varians
beregnes per ugedag/time af målte data; mindst to observationer per bin.
G's kernelprofil er gennemsnittet af fem historiske enhedsvarianser.
Time 0 udelades fra fit og rapporteres særskilt. Denne en-q/uge-kontrakt
adskiller sig fra WM's gamle q/dag-wrapper og deklareres i output.
Ingen parameter-, baseline- eller særskilt observationsstøj tilføjes nu.

Genbrug først den billige lokale G-enhedspropagation som **kandidat**.
Bevar forfiningskrav for center 1e-5 og std/profil 1e-4; kontrollér q=0,
ikke-negativitet, stategrænser, varians og en repræsentativ ikke-lineær
FV-reference med faste parametre. Pilotens konkrete grids, tolerancer,
antal kald og runtime fastlægges før kørsel, efter S3-snapshot foreligger.
Varians over .25 på [0,1] er en afvisningsgrund; varians under .25 er ikke
i sig selv accept. Ingen klipning af bånd/states eller σ for at få et pass.

Ved afvisning gemmes resultatet ærligt. Fast-parameter FV kan vurderes
inden for samme tidsramme som propagation af den eksisterende SDE;
et sådant ikke-lineært støjfit er ikke automatisk det simple WM-QP.
Ingen bred solver-/kalibreringssøgning eller bank startes automatisk.
Hvis et gyldigt støjlag kræver mere tid, afsluttes den fungerende
deterministiske del med støjlaget eksplicit åbent og en konkret restopgave.
Den deterministiske grundversion kan afsluttes som brugbar milepæl; støjlaget
er fortsat åbent. Eventuel nødvendig overskridelse forklares før gennemførelse.

## 4. Fælles output og checks

Fælles resultat: tidsstempel/offset, cutoff, protokol, baseline, pris,
deterministisk prognose, stokastisk middel/approksimationsstatus, observation
når tilgængelig, måltemaske, proces-/initial-/totalvarians og bånd.
Rå og normaliserede værdier holdes sammen med deres skalaer. Kun relevante
felter udfyldes; fraværende eller underkendt støjlag er ikke nul usikkerhed.

Brug samme MSE/RMSE/MAE/bias, dækning, middelbredde og intervalscore.
Punktvise 95%-bånd er ikke 95% samtidig ugedækning. I ikke-lineær G er
deterministisk forecast ikke nødvendigvis den stokastiske middelprognose.
Empirisk spredningsfit identificerer ikke isoleret fysisk processtøj.

Direkte WM-sammenligning bruger sammenlignelig population, opgave, information
og pris/temperaturscenarie; der scores i fælles råenheder og på fælles
observationsmaske. Intern skalering må variere mellem modellerne. Signed WM-pris er U=2u−1;
WM bruger dage, G timer. Den gamle WM-indlæsnings sæson-/helårsskalaer og
dagligt opdaterede fits må ikke umærket bruges som fælles benchmark.
Grundversionen eksporterer forståelige prognoser og standardscorer. En WM-adapter
tilføjes kun, hvis den er en lille opgave; bred WM-rangering er udskudt. G-status og tilgængelig WM-sammenligningsstatus er særskilte.

Historiske timemålingers fysiske enhed og punkt/intervalfortolkning
verificeres; ellers mærkes den eksisterende udviklingsapproksimation,
og kun den konkret manglende metadata efterspørges. Realiserede fremtidige
temperaturer er betingede scenarieinputs, ikke en historisk vejrudsigt.
Svagt identificeret prisrespons betyder, at forecastfejl og model-cost
ikke dokumenterer en kausal eller historisk styringsgevinst.

## 5. Rækkefølge og arbejdsramme

| Del | Oprindeligt afsat | Slutpunkt/status |
|---|---:|---|
| WM/G-review og kontrakter | 0,5 h | Udført; findings og kontrolkvittering gemt. Faktisk samlet reviewtid er ikke målt. |
| S3 fit/snapshot/predict | 2,0 h | Udført: kode, komplet snapshot, 8 kontroller og faktisk historisk eksempel. |
| Enkelt G-egnet støjlag efter fungerende kerne | Højst 2,0 h | Udført procesvej med eksplicit sigma; historisk kalibrering afvist. |
| Historisk eksempel og fælles outputs | 1,0 h | Udført for S3 og procesvej: 168 timer, scorer, figur og numerik; råpakke lokal. |
| Samlet brugsvej, review og merge | 1,0 h | Dokumentation og færdige dele integreret; eksisterende styringsvej henvist. |
| Reserve | 1,5 h | Konkrete fejl og inputadgang; ingen udvidelser. |
| **Oprindelig ramme** | **8,0 h** | **Ingen automatisk nulstilling af arbejdsbudgettet.** |

Tallene er estimater, ikke målt forbrug. Vi kan ikke udlede præcis resterende
tid fra dem. Registrér faktisk tid fra næste trin og efter hver del;
fratræk kendt forbrug fra rammen. Omkring 6,5 h fryses nye funktioner.
Kørsler får eget konkret loft med tid til eksport; ingen skjulte retries.
Det gamle G-forsøgs 1927.3372718589885/1930 s ledger forbliver historik.
Nye udviklingskørsler registreres særskilt, ikke som omskrevet gammel accept.

## 6. Klar til fortsættelse

Næste trin køres i **denne G-chat, GPT-6.1 Sol, High** med
[den færdige S3-brugsvej](G_S3_STANDALONE_GUIDE_20261008.md) og
[procesvejens slutstatus](G_S3_UNCERTAINTY_GUIDE_20261008.md). Grundimplementeringen
er udført. Næste udvidelse er intervaloperator/enheder og derefter gyldig
usikkerhedskalibrering; dette er ikke allerede afsluttet. Færdige egne branches merges
efter relevante kontroller som autoriseret af Johan.

Udskudt: parameterbank/filterdemo, fuldt G7-døgnforsøg, historisk
interval-likelihood, nye kurvefamilier, global parametersøgning, nye
risikomål, bredt benchmark og F's undersøgelser. Deres eksisterende kode,
evidens og åbne gates bevares. Første grundversion er færdig, når fit,
gem/genindlæs, prognose, valgfri scoring og et dokumenteret faktisk eksempel
virker med relevante korrekthedschecks. Støjlaget har særskilt status;
kun et gyldigt lag mærkes færdigt. BM2/flere parametre er senere udvidelser.

Målt implementeringspas startet 17:09:27 UTC. Kontrolkvitteringen viser
ca. 0,92 h til den første komplette pakning, eksklusive efterfølgende
dokumentation/merge. Tidligere reviews er ikke målt. Aktuelt restskøn efter
S3-milepælen: ca. 2–4 h til støjlagets afgrænsede vurdering/implementering
og samlet aflevering, afhængigt af dets gyldighed. Dette er et estimat,
ikke en ny otte-timers-ramme eller en garanti for et gyldigt støjlag.
