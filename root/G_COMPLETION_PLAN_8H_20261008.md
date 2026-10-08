# Aktiv G-afslutningsplan: grundversion inden for otte timers arbejde

8. oktober 2026. Denne plan er den aktuelle prioritering af det resterende
G-arbejde. Den bygger på [kodereviewet](G_CODE_MATH_REVIEW_20261008.md) og den
[genfundne matematik](G_MATHEMATICS_RECOVERY_20261008.md).
**Planstatus: fastlagt; leverancerne nedenfor er endnu ikke færdigimplementeret
eller kørt af dette review.** Der er ikke erklæret ny numerisk G7-accept.

## 1. Bestillingen og dens afgrænsning

Johan har valgt:

- Implementeringen skal helst være færdig inden for otte timers arbejde;
  kun strengt nødvendigt arbejde må begrunde en overskridelse.
- Leverancen er en **WM-sammenlignelig historisk prognose samt en fungerende,
  afgrænset filter→prisstyringsdemo**.
- G skal fortsat ligne WM i de opgaver, input og output, man ønsker at
  sammenligne. Metodiske udvidelser skal kunne begrundes ud fra disse opgaver.

Otte timer er arbejdsrammen for afslutningen, inklusive nødvendige checks,
kørsler og dokumentation. Det er ikke otte timers nye numeriske søgninger.
Registrér faktisk brugt tid ved delafleveringer; der er ikke grundlag her
for at oplyse et præcist hidtidigt tidsforbrug for den slettede chat.

Den første grundversion skal ikke omfatte en sammenhængende historisk kæde
fra fit gennem filter til løbende styring. Den skal heller ikke dokumentere,
at G slår WM. Negative, men korrekt beregnede resultater er gyldige.

## 2. Frosne valg — undgå at starte modeludviklingen forfra

1. **Historisk hovedvariant: S3, lambda=0.** Lineær f; fælles (C,k,Xi,a)
   fra de fem foregående kalenderugers 35 dagsforløb. To-stykkers g beholdes.
   De syv dagsstarter er de respektive historiske medianstates. Hele ugens
   prognoser fryses ved mandagens cutoff. Ingen ny kurve-/lambdaudvælgelse.
2. Den oprindelige lineære januar→168h-vej bevares som reference med eget
   protokolnavn. Den erstatter ikke stiltiende S3 og bruges ikke som bevis
   for, at S3's fremtidige selvstændige forecast-indgang allerede findes.
3. **Stokastisk hovedvej: den eksisterende M254-mixture og fem rho-hypoteser**
   med egne betingede laws. FV og rå EM er kontroller; de erstatter ikke den
   primære model for at gøre den hurtigere.
4. G7's deterministiske reference bruger MAP-rho og dens egen betingede
   middelstate, sigma_x=0. Dette følger den seneste implementation og
   afløser middelrho-formuleringen i det ældre døgnforslag.
5. De historiske og syntetiske observationer har forskellige erklærede
   kontrakter. Den historiske punktapproksimation mærkes tydeligt; demoen
   bruger egentlige syntetiske punktmålinger. Et nyt intervalfilter er ikke
   en skjult del af denne afslutning.
6. F-arbejdet holdes adskilt. G genbruger de allerede delte metoder; denne
   plan ændrer ikke F-forsøg, F-budgetter eller WM's modeldefinition.

## 3. Leverance A: selvstændig historisk S3-prognose

### Brugerforløb

Et lille fælles indgangslag skal kunne: (a) fitte én valgt historisk origin,
(b) gemme et komplet fitted snapshot, (c) indlæse det og forudsige den næste
uge fra priser/temperatur uden refit, (d) score bagefter, hvis demand findes.
Brugeren skal ikke manuelt samle en seksugers forskningskørsel for at få én
uge ud. Eksisterende målfunktion, RK4, warm-up, baseline og numeriske
forfiningskrav genbruges.

S3 skal have egen versionsmærket snapshot-type. Den skal gemme:

- cutoff, protokol, population, normaliseringsskalaer og eksplicitte enheder;
- fitted baselinekoefficienter, standardisering og alle indstillinger;
- C,k,Xi,a, responskurver og de syv **egne** historiske forecaststates;
- ODE-/baselinegrids, bestået forfiningscheck og kilde/inputfingerprints;
- træningsmaske og tidsvindue, prisdomæne, warm-up og status for identifikation;
- scenarieantagelsen om fremtidig pris/temperatur, uden fremtidig demand.

`fit_strategy`, `BM1PooledInputs`, `calendar_inputs` og den eksisterende
baseline bruges direkte gennem små adapters. `checked_task`'s gamle
to-lambda-forsøg gentages ikke for en valgt model. Genoptagelse af et afbrudt
optimizerfit må ikke forveksles med at genindlæse et færdigt snapshot.

### WM-kontrakten

Fælles resultatkolonner: tidspunkt, baseline, G-prognose, pris, enheder og
protokol. Ved scoring tilføjes observation, fælles scoremaske og
MSE/RMSE/MAE/bias. Tilstande og fitted parametre gemmes som egen diagnostik.
Bevar både rå og normaliserede tal. Skeln gennemsnitlige dagsmål fra pooled
fejl. Genbrug WM's scorer/rapportkonvention eller bevis paritet på faste tal.

En senere direkte WM-kørsel skal bruge samme informationsgrænse, baseline,
pris/temperaturscenarie og observationsudvalg. Modelspecificiteten i
initialisering og fit oplyses. Dagligt opdateret WM må ikke umærket
sammenlignes med S3, hvor alle syv dage er frosset mandag. Det er ikke et
krav at indføre identiske interne tilstande, fitalgoritmer eller parametertal.

G's gennemførelse og WM-sammenligningens tilgængelighed får hver sin status.
Det brede benchmark og en rangering af G mod WM udskydes. En fælles
output-/scoringskontrol er med i grundversionen.

### Færdigkriterier A

- Ét faktisk historisk, klart mærket udviklingseksempel med komplet pakke.
  Genbrug et verificerbart gammelt fit, hvis hele fitted pakken findes.
  Ellers udfør ét afgrænset nyt S3-fit fra dokumenterede inputs; ikke seks
  uger med to lambdaer. Git indeholder rå input-workbooks; adgang til den
  konkrete temperaturserie og historiske fitpakke kontrolleres først.
- Snapshot→forecast giver samme resultat uden kald til G-fit eller
  baselinefit. Forecast-API'et behøver ingen fremtidig demand.
- Fjernelse/ændring af fremtidige observationer ændrer kun scorerne.
- Syv historiske initialiseringer, 24 fysiske warm-up-timer og kalender/DST
  håndteres som i den valgte opskrift. Ikke-understøttede datovinduer
  afvises tydeligt fremfor at fremstille 168 fysiske timer på alle kalenderuger.
- Bestået bevaret coarse/fine-krav for demand (1e-5 normaliseret), finitte
  output og ingen skjult state-/prisklipning.
- CSV/JSON og en oversigtsfigur viser baseline, G, observationer når de
  findes, og pris. Rapporten viser runtime, numerik og databegrænsninger.

Januar/Februar og ugerne 6–18 er allerede inspiceret. De kan være
reproduktions-/udviklingseksempler, men bliver ikke nye blinde sluttestdata.

## 4. Leverance B: en kort, reel filter→prisstyringsdemo

### Mindste komplette forløb

Brug de verificerede egne M254257/513-origins ved t=12 fra recovery.
Bevar oprindelsesfilerne uændret. Det korte forløb er:

1. Vælg en prisplan for [12,14] fra en lille **på forhånd fastlagt** familie.
2. Propagér under den valgte pris. Generér én syntetisk punktobservation
   ved t=14 fra en særskilt deklareret sand bane og observationsstøj.
3. Udfør den rigtige Bayes-opdatering af hver nodes law og bankens vægte,
   og gem både rå evidence og posteriorprojektionens kontroller.
4. Vælg en ny prisplan fra den opdaterede posterior for [14,16], og
   gennemfør en faktisk continuation. Brug samme observation i referencesporene.

Den primære **basisafslutning** bruger to prisblokke per 2h-vindue og tre
registreret ordnede kandidater: neutral (.5,.5), (.3,.7), (.7,.3).
Prisvalg er argmin af de fuldt gennemførte bankomkostninger, med første
registreret kandidat ved eksakt lighed. Alle tre skal være beregnet før en
færdig kandidatbeslutning erklæres. Det giver et reelt afgrænset kontrolvalg,
men rapporteres som **bedst af tre**, ikke som kontinuerlig prisoptimering.

Beregn også den deterministiske MAP-references valg indenfor samme familie,
og sammenlign dens valgte pris og neutralprisen på samme stokastiske J.
Denne afgrænsning er ny: den reducerer 24h/96-kalds-forslaget, ikke det
matematiske objective. Den eksisterende kontinuerte optimizerkode bevares
og kan senere kobles til samme adapter; ingen ny optimizer udvikles nu.

Baseline B(t)=.5+.1sin(2*pi*t/24), A=.5, priser i [0,1], timer, sigma_y=.03,
de fem eksisterende rho-knuder og faste øvrige parametre bevares fra den
syntetiske case. Terminal-/glathedsstraffe er nul; ingen ny energilighed.
Observationens sande rho og tilfældighedsstrømme fastlægges før prisresultater
ses. Der må ikke vælges en observation, som får feedbacken til at se god ud.

### Hvad referencesporene betyder

Manglende gamle **egne FV-filterorigins** må ikke blokere denne anderledes,
afgrænsede demo. Til en kontrol af fremadudvikling og opdatering kan FV
starte fra samme eksplicit repræsenterede M254-prior på hvert grid. Det
mærkes `FV-reference fra fælles M254-startlaw`, ikke `egen G8-FV-posterior`.
Efter observationen fører hvert spor sin egen opdaterede law videre.

Denne reference undersøger den korte kæde fra det fælles informationspunkt.
Den validerer ikke hele det tidligere FV-filterforløb fra t=0 og opfylder
ikke automatisk den gamle døgnplans krav om fire egne t=12-filterorigins.
Der er derfor ingen grund til at svække eller omgå den gamle døgnlauncher;
den nye demo får sin egen korte run-kontrakt og status.

### Kontroller B

Genbrug `OwnedBankPriceEvaluator`, den eksisterende auditerede
receipt-/feedbackadapter, native FP/projektion/Bayes og review-algebra.
Tilføj kun den nødvendige korte orkestrering og eksport. En fixture med
erstattede numeriske funktioner er ikke tilstrækkelig slutkontrol.

- Alle kandidater begynder ved samme uændrede posterior; parameter/state-
  afhængigheden bevares. Input og output bindes til præcise kilde-/inputhashes.
- M254257/513-forfining, FV-reference og ændret costpartition kontrollerer
  de frosne valgte priser. Ingen fine/FV-udvælgelse af nye vindere.
- Gem projektions-L1 og optimalitetskontrol for hver hypotese **under**
  forward og Bayes. En manglende audit er ikke en nulfejl.
- Bevar de relevante eksisterende numeriske tolerancer: masse 1e-9,
  conditional/marginal/projektions-L1 .05, bankvægte .03, log-evidence .05,
  middelstate/demand .005, varians .002 og J-forskel max(.001,.02*abs(fine_J)).
  Eksisterende CDF-/kvantilchecks bevares, hvor de bruges. Definitioner og
  faktisk omfang fryses før kørsel; tolerancer tilpasses ikke til resultater.
- Én afgrænset rå EM-krydskontrol bruger koblede coarse/fine-steps og samme
  starttræk/Brownian-støj mellem de sammenlignede priser. Brug eksisterende
  fem gange 256 baner, stratificeret SE, ingen randkorrektion og de eksisterende
  EM/FV-kriterier. Observationens støj har separat stream.
- Gem en fuldstændig opdateret origin. Genindlæsning bevarer egne vægte,
  koefficienter, basis og cutoff. Den sidste 2h-forward skal faktisk bruge
  denne posterior, ikke den oprindelige t=12-law eller en kopieret FV-law.
- Rapportér også hvis samme pris vælges før/efter observationen, eller
  neutral er bedst. Det er ikke i sig selv en fejl eller en forbedringsclaim.

**Færdig B** kræver en reel gennemført kort kæde, beståede deklarerede
kontroller og en læsbar resultatsammenfatning. Hvis en gate fejler, gemmes
resultatet som fejlet/ufærdigt, og kun den konkrete fejl undersøges indenfor
reserven. Stor døgnaccept og historisk kontrolaccept står fortsat åbne.

## 5. Arbejdsbudget og rækkefølge

| Del | Planlagt tid | Konkret slutpunkt |
|---|---:|---|
| Afslut kode-/matematikreview og frys ovenstående kontrakter | 0,5 h | Én aktiv plan; S3, observationsbegreb og referencevalg er entydige. |
| Historisk S3-indgang, snapshot og prognoseeksport | 2,0 h | Leverance A's kode og fokuserede kontraktchecks. |
| Kort filter→kontrol-demo og holdbar resultateksport | 2,0 h | Leverance B's orkestrering over eksisterende kernels. |
| Faktisk historisk eksempel og native syntetisk integrationskontrol | 1,0 h | Gemte råresultater, numeriske checks og ærlig status. |
| Samlet gennemgang, fælles outputkontrol, dokumentation og merge | 1,0 h | Brugbar startvej, figurer og genoptagelsesnoter. |
| Reserve til konkrete fejl, inputadgang og uventet køretid | 1,5 h | Kun nødvendige rettelser; ingen nye modeludvidelser. |
| **I alt** | **8,0 h** | **Grundversion med to tydeligt adskilte anvendelser.** |

Tiderne er estimater, ikke dokumenteret runtime. Kodningen behøver ikke vente
på lange modelkørsler. Efter A og efter B føres en kort faktisk tidsstatus.
Ved ca. 6,5 timer fryses nye funktioner; resten bruges til at få de aftalte
forløb til at virke og til at dokumentere dem. En forventet overskridelse
forklares med konkret blokering og mindst nødvendige ekstra arbejde;
rammen udvides ikke stiltiende.

Numeriske budgetter fryses for hver konkret invocation med reserve til
stop og eksport. Ét repræsentativt kort bankjob måles, før resten af den
korte graf startes; rateenhed omfatter alle fem knuder, ikke kun den
billigste node. Mål og gem faktiske tider. Ingen gentagne automatiske
retryforsøg, skjulte budgetnulstillinger eller store globale søgninger.
Et afsluttet job genbruges kun under den samme verificerede kontrakt.

Den gamle forskningskampagnes debit 1927.3372718589885 af loft 1930 s
og alle dens hændelser bevares som historik. Johans nye arbejdsramme
autoriserer arbejdet med denne grundversion, men gør ikke det gamle
døgnforsøgs rate-/inputkontroller beståede. Nye udviklingschecks og den
afgrænsede demos konkrete compute-scope registreres særskilt, med reference
til denne bestilling; de bogføres ikke ved at omskrive gamle ledgerfiler.

## 6. Bevidst udskudt

- Det fulde 24h-forsøg med to kontinuerte søgninger og den store 16-jobgraf.
- Historisk interval-likelihood og en samlet historisk fit→filter→styringskæde.
- Flere parametre i banken, ny basis, nye kurvefamilier, lambda-søgning,
  globale optimizerstudier og gentagne MPC-cykler.
- Profit-/indkøbsmodeller og nye risikomål.
- Bredt måneds-/årsbenchmark, nye blinde testperioder og endelig WM-rangering.

Dette reducerer forsøgsomfanget, mens G6b, G8 og G7 indgår i grundversionen.
Den korte demo er en afsluttet **basisimplementation**, ikke en påstand om,
at alle krav i det tidligere fulde forskningsprogram er opfyldt.

## 7. Holdested til næste arbejdspas

Reviewet er færdigt på den angivne kilderevision; 39 afgrænsede tests består.
Modelkode er uændret i reviewleverancen. **Første implementering er den
selvstændige S3 fit/snapshot/predict-vej**, ikke et nyt G7-døgnforsøg.

Find/verificér det konkrete historiske input og eventuelle komplette
S3-arkiv, mens den lille indgang bygges. Hvis dataenes måleenhed eller
timeintervalfortolkning ikke kan genfindes, behold den eksisterende tydeligt
mærkede udviklingskontrakt og spørg Johan om den manglende metadata;
opfind ingen fysisk enhed. Der er ikke behov for at bede om en ny godkendelse
af den allerede valgte otte-timers-afgrænsning eller af rutinemæssige fixes.

Gem efter hver færdig del: hvad der virker, udførte checks, resterende
begrænsninger, faktisk tidsforbrug og præcist næste trin. Færdige egne
branches merges efter relevante kontroller som tidligere instrueret.
