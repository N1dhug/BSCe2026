# G: bevaret parameterbank- og filter→styringsudvidelse

8. oktober 2026. Udskilt fra den aktive plan for at holde hovedsporet kort.


Følgende design bevares til en senere udvidelse. Det er ikke længere en
obligatorisk leverance indenfor otte timer; se [den aktive plan](G_COMPLETION_PLAN_8H_20261008.md).

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
