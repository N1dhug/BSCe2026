# Aktiv G-plan: færdig grundversion inden for otte timers arbejde

8. oktober 2026. Denne samlede plan erstatter de tidligere prioriteringer
af obligatorisk parameterbank/demo. Den bygger på Johans seneste instrukser,
[kode-/matematikreviewet](G_CODE_MATH_REVIEW_20261008.md) og
[gennemgangen af faktisk WM-kode](G_WM_ALIGNMENT_REVIEW_20261008.md).
Review og afgrænsning er udført. De nye implementeringsleverancer nedenfor
er endnu ikke færdige. Ingen ny historisk eller numerisk G7-accept er erklæret.

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

## 3. Dernæst: ét WM-lignende støjlag med tydelig gyldighed

WM tilpasser q=σ² til empirisk varians efter deterministisk fit. G6a
tilpassede σ til standardafvigelse; det er et andet problem. Den gamle
G6a-underkendelse bevares og kan ikke omdøbes til ny accept.

Indfør et særskilt navngivet variansfit med én q per mandagsfrossen uge,
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
Det fulde mål er da ikke nået; overskridelse forklares før den gennemføres.

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

WM-sammenligningen bruger samme population, rådata, informationsgrænse,
skalaer, pris/temperaturscenarie og scoremaske. Signed WM-pris er U=2u−1;
WM bruger dage, G timer. Den gamle WM-indlæsnings sæson-/helårsskalaer og
dagligt opdaterede fits må ikke umærket bruges som fælles benchmark.
En tynd adapter og scorerparitet er med i grundversionen; bred WM-rangering
er udskudt. G-status og tilgængelig WM-sammenligningsstatus er særskilte.

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
| S3 fit/snapshot/predict | 2,0 h | Næste opgave; kode, snapshot, fokuserede kontraktchecks. |
| Variansfit og afgrænset støjpilot | 2,0 h | Implementeret metode, checks og gyldig/underkendt status. |
| Historisk eksempel og fælles outputs | 1,0 h | Rigtige data/resultater, scorer og figur. |
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
[den færdige arbejdsinstruks](G_NEXT_STEP_20261008.md). Ingen ny generel
planrunde eller godkendelse er nødvendig. Færdige egne branches merges
efter relevante kontroller som autoriseret af Johan.

Udskudt: parameterbank/filterdemo, fuldt G7-døgnforsøg, historisk
interval-likelihood, nye kurvefamilier, global parametersøgning, nye
risikomål, bredt benchmark og F's undersøgelser. Deres eksisterende kode,
evidens og åbne gates bevares. Grundversionen er færdig, når prognosevejen,
det gyldige afgrænsede støjlag og deres aftalte checks virker; en afvist
støjapproksimation alene opfylder ikke dette færdigkriterium.
