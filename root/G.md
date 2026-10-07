## G7: auxiliary-budget og integrationsdesign færdigt, 7. oktober

Deterministisk BM/WM med bevaret døgnforbrug forbliver primær; færdige
fits/policies og råevidens genbruges. Supervisor M254/legacy og F/WM bevares.
Næste måleforsøg er konkret designet som neutral12→14h: genbrug de ti
gemte M254-noder, nye egne FV257/FV513-forwards, coupled EM256/node og
samme nye syntetiske endpoint-observation på fire backend/grids.
Ingen M254-forward, optimizer eller nyt G8-filter i det foreslåede scope.
Maks200s =150 worker +30 stop/eksport +20 review; jobreservationer130s,
workerheadroom20s. Reservationer er stoprammer, ikke målte rates/bounds.
Read-only restore med fuld gammel kontrakt/ny setupcharge og ny launcher-
provenance skal implementeres og sourcefryses før separat præregistrering.
launch_ready=false; scope er endnu IKKE udført/charged. Ingen auto-retry.

Vigtigt: gemte outputmidtpunkter giver ikke gratis kvadraturforfining;
gennemsnit af to midpoint-costs reproducerer blot samme sum. En ny
quadraturpartition kræver eget forwardjob/kontrakt og forbliver ukendt.
14-rækkers døgnplan og gates bevares; 2h-demo kan ikke give24h-admission.
Ingen sammenligning af J fra den analytiske2h-fixture med januarens24h-par.

Én stdlib struktur-/aritmetikkontrol består44 checks på.010431s; ingen
native import, modelrun eller gamle tests. Design/reviewcharge10s:
G1234.712123334s, rest565.287876666s. F/WM-budgetter er uændrede.
Næste: implementér/review restore+auxiliary-launcher, registrér og mål
det konkrete neutrale scope; derefter beslut policy-/døgnbudget på faktisk evidens.
Sol High; ingen lokal kommando nu.
Læs [design, udledning, budget og resterende arbejde](G7_AUXILIARY_INTEGRATION_DESIGN.md).
Følgende G-statusafsnit er historik.

## G7: eget M254-nodeforsøg gennemført, 7. oktober

10 native nodeforwards på gemte EGNE hour12-posteriorer, neutral12→14h,
M254513 først37.961317s, M254257 derefter16.828349s inkl.nodefiler/reduktion.
Jfine=.003715517464, Jcoarse=.003741963573; Jfejl2.644611e-5<.001.
Conditional/marginal slut-L1 .011187752/.005225045<.05; alle korte
moment/CDF/kvantil- og projektiongates består. Ingen FV/EM/optimizer/feedback.
Første invocation stoppede før forward ved bit-exact basisreload; original
fejl/source/input/ledger er bevaret. Optional verified-byte loader bevarer
gamle basisværdier/c og default strict guard; machine-roundoff-familiecheck
med eksakt owned-bytehash er begrundet. Ingen model-/acceptgate lempet.
Ét loadercheck består1.609s; gamle node-/modeltests ikke genkørt.
12 receipts og41 sourcehashes råreviewet; gemte demandmomenter/J rekonstrueres
identisk uden nye forwards. Fuld data-ZIP er gemt separat og hashverificeret.
Native source/kode og originalinputbytes er gemt i Git; vejleder M254 bevares.
G-debit1224.712123s, rest575.287877s; begge forsøg medregnet, F/WM uændrede.
2h-rates giver ingen24h-admission. En ren lineær M254-cross-illustration er
41.1min før andre jobs, ikke målt døgnrate/bound. Ingen automatisk større run.
Næste er budgetteret FV/EM/feedback-/policyintegrationdesign med recovery,
derefter eksplicit døgnadmission. Sol High; ingen lokal kommando nu.
Læs [forsøg, matematik, råreview og resterende arbejde](G7_NODE_RUNTIME_REVIEW.md).
Følgende G-statusafsnit er historik.

## G7: egne nodecheckpoints implementeret, 7. oktober

Durable nodejobs bevarer hver rho-nodes egne M254-c/posterior, eksakt
prisplan og original transportpartition. Atomiske råbaner og write-ahead
reservationer genbruges efter crash; afbrudte noder kræver eksplicit retry
med bevaret charge. Persistent kampagne- og studycap; én OS-locked ejer.
36 kildehashes/full origin/numerik/versioner indgår i kontrakten.
Native mixturekald, bankreduktion og assimilate er AST-identiske med før.
Én fokuseret stubtest består; anden invocation efter konkret caprettelse
består også. Begge receipts bevares;4.129374s process i alt. Ingen faktisk
M254/FV/EM-forward, nye fits eller gamle numeriske checks genkørt.
Komplet14-jobgraf inkl.FVfine/quad/coupledEM/feedback/setup/export er gemt.
Alle rates er ukendte; budget_admitted/scientific_accept/optimizer=false.
Kooperativ deadline er ikke hard kill; næste launcher kræver stopreserve.
Vejlederbasis og legacy bevares; F/WM-kilder og budgetter er uændrede.
G-debit1118.580078s, rest681.419922s efter konservativ20s scopecharge.
Næste er særskilt kort registreret faktisk måling fra hashverificerede
egne origins, ingen automatisk gentagelse af den gamle60s-pilot.
Sol High; ingen lokal modelkommando nu.
Læs [matematik, crashmekanik, budget og faktisk kontrol](G7_NODE_JOB_REVIEW.md).
Følgende G-statusafsnit er historik.

## G7: deterministisk døgnkontrol med bevaret forbrug, 7. oktober

Fælles frossent januarpar og lineær BM-reference er optimeret over fire6h-
prisblokke, både original fixed-A og separat lighed integral(demand-B)=0.
Alle seks valgte policies består coarse/fine;374 faktiske forwards inkl.
Jacobianprober,3.7754s process. Neutral fine-J: lineær BM .275408888,
G4 .276532280, WM .303629020; baselineJ .308609546. Forbedringer10.758%/
10.394%/1.614% er modelbetinget tracking i ét døgn, ikke fysisk/kausal ranking.
WM løses som certifieret affin/konveks4D-QP; BM er bedste fundne lokale
SLSQP-kandidat fra to starter, uden global optimalitetsclaim. Ingen refits.
Samme diskrete forbrugsbalance bruges i J og lighed; WM's Euler/midpoint-
korrektion er udledt. Fine energierror højst1.466e-6 i neutrale policies.
Én fokuseret analytisk kontrol og read-only byte/J/E-audit består.17 units/
19 råfiler, fulde baner og hashes er gemt med atomiske checkpoints.
S3-fladhed genbruges som identifikationsforbehold; ingen ny straf indføres.
M254's gamle døgnestimat mangler eksplicitte FV/EM/feedback/quad-led:
ingen ny stochastic admission eller pilotretry. Næste er durable egne
rho-nodejobs og komplet målt jobgraf før særskilt registreret måling.
Vejlederbasis, legacy, F/WM-kilder og deres budgetter er uændrede.
Konservativ nattecharge30s af max300s; G1098.580078s, rest701.419922s.
Læs [matematik, faktisk udført og resultater](G7_ENERGY_NEUTRAL_REVIEW.md)
og [M254-runtime/jobdesign](G7_M254_RUNTIME_JOB_REVIEW.md).
Følgende statusafsnit er historik.

## G7: fælles BM/WM-kontrakt og frozen-fit-prisrespons, 7. oktober

24h/fire blokke/known fixed-A med fælles januar-baseline og scales er låst.
A er baseline-døgnmiddel, uden target-Y; WM's native A(U)-opgave bevares.
Time/dage og u_BM=u/U_WM=2u-1 omregnes eksplicit; hver kandidat bruger
sin gemte origin. Januar-G4/lambda0 og matched januar-WM4 genbruges sammen
med original lineær BM-reference; nyere S3 har ikke matched WM-origins.
Én kort analytisk kontraktkontrol består.18 forwards (3 priser/3 modeller/
2 grids) består alle gates; .212s body/.505s process. Ingen optimizer/refit.
S3-audit:1008/1008 gemte forecastpriser u<.5; uge16–18 har næsten flad
g netop dér. Stærk respons over knækket er ikke kausalt dokumenteret.
J alene kan falde ved ændret slutlager/energi; rapportér terminal og
energy-gap separat, ingen skjult ny straf. Raw demandsemantik er endnu
legacy-uafklaret; ingen kW²h- eller fysisk ranking-claim.
18 gamle importerede Git-blobs matcher executionbase; WM deterministic
AST er scoped-reviewet, ny validation-only guard består. F/WM/legacy
kilder, basisfamilier, budgets og negative G7/M254-resultater bevares.
Næste: lille fireblokoptimering af samme januarpar, native feasibility,
J/energy/terminal rapportering. S3's senere historiske kontrol er ikke
frigivet. Stokastisk G7 obligatorisk; runtime-/jobdesignreview før retry.
G-debit1068.580078s, rest731.419922s. Sol High; ingen lokal kørsel nu.
Læs [kontrakt, udledning, kilder og faktiske resultater](G7_MATCHED_CONTROL_REVIEW.md).

Følgende daterede statusafsnit er historik.

**G7 deterministisk kontrol gennemført, 7. oktober 2026:** Begge kontrolmål beholdes med deterministisk BM/WM-reference først. Original BM i FP2025 bruger kendt exogent A(t); en ny WM-afledt A(U)-funktion er ikke nødvendig. Én syntetisk24h-case med fire prisblokke, fixed A=.5 og lineær reference giver fine J=.0209268019845 mod neutral .12 (82.561% mindre). Begge bounded least-squares/TRF-starter konvergerer via ftol, uden global/KKT-optimalitetsclaim. Alle coarse/fine-gates består: J-fejl6.78073e-7, statefejl1.32738e-7, demandfejl5.55150e-8. 68 faktiske forward-solves inklusive Jacobian-prober, invocation14.227s. Én fokuseret analytisk kontraktkontrol består efter rettet assertion ved aktiv bound; failed check bevares, produkttolerancer uændrede. 27 gamle importerede Git-blobs og execution-hashes matcher. F/WM/shared-kerner er uændrede. **Syntetisk deterministisk udviklingsaccept; faktisk BM/WM-rangering, historisk kontrol og stokastisk G7 er åbne.** Næste: fælles fixed-A benchmarkkontrakt, ens inputs/prisblokke/enheder og eksplicit initialisering; bevar WM's native A(U)-mål separat. G4-fits genbruges og svag prisrespons diagnosticeres før fysisk kontrolfortolkning. M254-prisoptimering afventer stadig runtime-/jobdesignreview; ingen automatisk genkørsel. Conservative paneldebit1062.580078s, rest737.419922s. Sol High er tilstrækkelig; ingen lokal kørsel nu. Se [udledning, kilder, matematik og råresultater](G7_DETERMINISTIC_CONTROL_REVIEW.md). Følgende daterede statusafsnit er historik.

## Aktiv kontrolprioritet — Johan, 7. oktober 2026

Det deterministiske kontrolmål er primær reference til BM/WM-sammenligningen. Det stokastiske mål beholdes som obligatorisk senere G7-leverance. Original BM bruger fast A(t); en prisafhængig indkøbsfunktion er en særskilt udvidelse og ingen adgangsgate. Fælles kontrolmål, input, prisconstraints, skala og informationsgrundlag skal låses før modelrangering. Tidligere P6-tekst om stokastisk førsteprioritet læses med denne rækkefølgeændring; den afsluttede syntetiske pilot genkøres ikke.

**G7 own-posterior-handoff implementeret, 7. oktober:** Den nye G-adapter bruger vejlederens typed M254-basis og hver rho-nodes egne koefficienter; priser evalueres fra samme frosne posterior med fuld within-/between-parameter-varians. Én fokuseret reset/missing/ownership-kontraktkontrol består (.410 s). **Den faktiske runtimepilot stoppede ved 60.001 s under første fine forward; grove to timer var færdige efter afrundet 17.61 s elapsed. Ingen parret precisiongate, prisoptimering, EM eller faktisk observations-feedback er gennemført.** G7 er ikke accepteret; G8's tidligere accept bevares. Original executed source og negative partials er arkiveret. Runneren gemmer nu completed forwards straks og håndhæver marginal-/feedback-gates eksplicit; dette eksport-/guard-fix er syntakskontrolleret uden retry. Næste er et afgrænset runtime-/jobdesignreview før en særskilt registreret pilot; ingen automatisk større run eller løsere gate. Conservative paneldebit inkl. arkiv er 1045.580 s, rest 754.420 s; gamle umålte probes dækkes fortsat af den tidligere reservation. Ingen lokal modelkommando kræves nu. Se [matematik, kilder, faktisk udført og begrænsninger](G7_OWNED_HANDOFF_REVIEW.md). De følgende statusafsnit er dateret historik.

**M254-forfining gennemført, 6. oktober:** Eget Bernstein-filter med 513 celler og max_step=1/192 gennemfører samme 24 timer fra egen eksakte prior. Alle parrede conditional-/marginalgates mod gemt M254257 og FV513 består; faktisk JSON-resume time22→24 er identisk, også i conditional evidence. Mod FV513: conditional L1 0.00354065, marginal L1 0.00108111, nodevægtforskel 0.000265417; største projektion L1 0.000971943. Filter 309.743 s; resume 17.327 s; invocation 327.829 s. Vejlederens 255 Bernstein-funktioner, model, likelihood og tolerancer er uændrede. **M254-repræsentationen er klar til næste G7-interfacearbejde på denne syntetiske case; G7-mixture-prisforward og prisstyring er endnu ikke implementeret eller accepteret.** Næste er et typed own-origin-handoff og kort prisforward-runtimepilot før et konkret optimizerbudget. Ingen lokal kørsel kræves nu. Se [matematik, valg, resultater og begrænsninger](G8_M254_REFINEMENT_REVIEW.md). Statusafsnittene nedenfor er dateret historik.

**M254-filterforsøg gennemført, 6. oktober:** Vejlederens oprindelige Bernstein-familie er implementeret som eksplicit celleintegreret M254 med eksakt gradforhøjet Beta(20,20)-prior og typed checkpoints. Eget fem-node-filter gennemfører24 timer fra prioren; alle conditional- og marginalgates mod gemte FV257/FV513 består. Maksimal conditional L1 modFV513 er0.0295811, marginal L1 er0.0126685, Bayes/substep-projektion er0.000963871. Tre målrettede checks består; faktisk JSON-resume fra time22 til24 er identisk i masser, koefficienter, vægte og evidence. Filter117.873 s; hovedinvocation119.901 s; resume6.204 s; samlet126.105 s før råarkiv. 33 øvrige importerede kodefiler matcher main c64198b. **Eget513-replay blev ikke admitted inden for300s-budgettet; egne mixture-forfiningsgate og mixture→G7-accept er stadig åbne.** Dette er et positivt257-udviklingsresultat, ikke et negativt basisresultat eller historisk accept. Næste er én særskilt budgetteret513-forfining med samme basis/model og checkpoint; Sol High er tilstrækkelig. Ingen lokal kørsel kræves nu. Se [matematik, valg, resultater og begrænsninger](G8_M254_FILTER_REVIEW.md).

**Vejledermateriale/legacy, Johan 6. oktober:** Tavlebillederne er vejlederens
grundlag. Originalmodeller og resultater bevares; udvidelser kræver udledning,
kilder, begrundelse og eksplicit sammenligning. Den nye basis er kun reference,
og dens allerede gennemførte G8-kørsler genbruges. Se
[detaljeret Beta-udledning, kildekort og designargument](G8_POSTERIOR_BASIS_REVIEW.md#vejledergrundlag-og-referenceudvidelse--dokumentationsregel).
Den oprindelige noteændring var dokumentation alene; det efterfølgende
M254-review er registreret særskilt nedenfor.

## Aktiv basisbeslutning — Johan, 6. oktober 2026

Vejlederens normaliserede Bernstein/Beta-basis er G's primære mixture-basis:

\[
\ell_m(x)=\frac{(M+1)!}{m!(M-m)!}x^m(1-x)^{M-m},
\qquad m=0,\ldots,M.
\]

Den nye faste 42-leds Beta-basis beholdes alene som sammenligningsreference.
FV er fortsat den numeriske reference; metodeprioritet og numerisk accept er
forskellige beslutninger. De gemte negative M38/M78-resultater og positive
42-leds-resultater bevares uændret. M38's åbne G6b-propagation består, men
M38/M78 er ikke accepteret til G8's posterior→G7-kæde.

**G8-review gennemført, 6. oktober:** Én statisk kandidat, M254 med 255 af
vejlederens oprindelige Bernstein/Beta-funktioner og eksplicit celleintegration,
består syv afgrænsede LS-kontroller på gemte love. Maksimal posterior-L1 er
1.27144e-6; invocation9.596 s. Gradforhøjelse bevarer Beta(20,20)-prioren
med L1-fejl3.18e-14. De eksisterende produktionskerner og legacy-defaults er
uændrede. Dårlig konditionering består; dette er repræsentationsaccept på
udvalgte eksponerede love, ikke accept af et rekursivt filter eller G7.

**Det planlagte M254-filterforsøg er nu gennemført; aktuel status står øverst.**


G7's primære mixture-handoff skal komme fra en kontrolleret kørsel med den
oprindelige basisfamilie og hver nodes egne koefficienter/posterior.
Det gemte bank_local257-origin må kun bruges som eksplicit 42-leds-reference,
ikke som primær origin eller som skjult restart for Bernstein-filteret.
Bevar modellen, likelihooden, simplex-constraints og accepttolerancer;
ingen større målefejl, signed koefficienter eller FV-fallback forklædt som
mixture for at få den primære metode til at bestå. Hvis den primære basis
ikke består inden for praktisk budget, dokumenteres begrænsningen.

Denne beslutning erstatter tidligere anbefalinger om 42-leds-basis som
G7's primære startkonfiguration. De tidligere resultater bevares uændret.
Det efterfølgende M254-review ovenfor tilføjer kun statisk evidens.

**G8 punktfilter/bank gennemført som afgrænset integration, 6. oktober 2026:** FV kendt-parameter-filter og fem-node-rho-bank består forfining, missing og faktisk JSON-resume. 24 syntetiske timer/22 målinger; predictive demand-RMSE .0208264 mod åben .0286866 (27.40 % mindre på denne ene bane). Banken identificerer ikke sand rho sikkert. M38/M78 gennemfører, men begge afvises efter Bayes-opdateringer: kendt posterior-L1 .498924/.139169 og bank .500165/.138551 mod .05. M78's gode mean er ikke fordelingsaccept. Fuld-bank→FV fast kontrolforslag bevarer egne nodeposteriors og parameterusikkerhed; ingen prisoptimering. V2 invocation49.112 s, tre fokuserede tests består, fælles kerner/F/WM uændret. Se [matematik, råreview og begrænsninger](G8_POINT_BANK_REVIEW.md). **G8 mixture er ikke accepteret: næste lille opgave er posterior-repræsentation før mixture→G7; FV-handoff er klar til separat G7-referencearbejde.** Historisk timeenergi og ugeforecast er ikke accepteret. Ingen lokal kørsel kræves nu. Ældre næste-trin-afsnit nedenfor er historiske.

**G8 posterior-basis afklaret, 6. oktober 2026:** En fast celleintegreret ordbog med 41 smalle Beta-led (kappa256) plus Beta(20,20)-prioren sænker Bayes-projektionsfejlen fra .128495 til .001494 allerede ved samme129-grid. 42 koefficienter bruges i stedet for M78's79. En ny conditional-node-kontrol afviste129-cellers grid for både den gemte FV-bank og ny mixture; det negative første studie bevares. Én målrettet257→513-forfining består nu for hver node og marginalen. **Historisk anbefaling, nu erstattet af Johans basisbeslutning ovenfor: 42 komponenter/257 celler er kun reference for G7.** Alle31 eksisterende importerede shared-moduler er uændrede; kun G-adapterens basisvalg udvides. Faktisk JSON-resume er identisk; to fokuserede tests består. Nye numeriske invocations150.086 s i alt. Se [udledning, kildegrundlag, valg og råreview](G8_POSTERIOR_BASIS_REVIEW.md). Næste G7-evaluator skal bruge originens faktiske basis og egne nodekoefficienter, ikke rekonstruere M38. Ingen prisoptimering, historisk forecastaccept eller parameterrecovery i denne leverance. G4/F/WM uændret; ingen lokal kørsel kræves nu. Tidligere blokering nedenfor er løst for denne udviklingscase, ikke universelt.

# G-plan — deterministisk BM-kerne, stokastisk udvidelse og WM4-sammenligning

**G6b-panel gennemført, 6. oktober 2026:** Én låst syntetisk 24-timerscase med samme Beta(20,20)-prior, lineær f/g og fast sigma_x=0.2. FV består rum-/tidskontroller. M20 afvises (initial L1=0.291667, statevarians/kvantiler og demand-CDF); M38 matcher FV (maks. demandmiddelforskel 0.0001061, J-forskel 0.0221 %). EM2048 stemmer med FV inden for registrerede praktiske tolerancer og punktvise sampling-SE; nul rå randoverskridelser. FV er primær reference og hurtigere end M38 her. V3-hovedpanel 7.101 s, invocation 8.840 s plus separat genbrugt M20-pilot 1.120 s; fejlede interfaceforsøg er bevaret uden komplet samlet timing. To fokuserede tests består; ingen nye fits eller shared-kernelændringer. Dette er numerisk udviklingsevidens, ikke historisk forecast-/intervalaccept. Se [matematik, valg og råreview](G6B_DISTRIBUTION_PANEL_REVIEW.md). **Næste: G8 kendt-parameter-statefilter, derefter lille rho-bank og G7.** Ingen lokal kørsel kræves nu. Ældre næste-trin-afsnit nedenfor er historiske.


**G6a-reference færdig, 6. oktober 2026:** Konstant støj kalibreret med analytisk nonnegative LS på fem tidligere matchende weekdays, genbrug af S3/lambda=0 uge13+18. 336 målte timer; procesbånd dækker 45.83 % mod 88.99 % for historisk spredning omkring samme G4-center. Uge13 lineær statevarians overskrider den fysiske bound1/4; båndene er ikke accepteret som kalibrerede 95 %-intervaller. 168 propagationer/2.283 s, begge tidsforfininger og tre fokuserede tests består. Ingen nye strukturelle fits eller støjfamilier; F/WM uændret. Se [matematik og negativt G6a-resultat](G6A_LINEAR_NOISE_REVIEW.md). **Næste: G6b med FP/FV, Beta-mixture og EM på det faste lille syntetiske panel.**


**G4 afsluttet for den afgrænsede punktprotokol, 6. oktober 2026:** Den fastlåste S3-sammenligning på nye ISO-uger 13–18 er complete: seks fælles mandagscutoffs, 1008/1008 målte timer. Uden straf har MSE 0.1798347601 mod 0.1843188395 med lambda=0.001 og baseline 0.1806157175. Straf øger MSE 2.4934 %, vinder kun 2/6 uger og taber alle leave-one-week-out-sammenligninger. Det registrerede modelvalg er derfor **S3 med lambda=0**, lineær f og fælles to-stykkers g over 35 historiske dage; ingen nye kurver eller lambdaer. Baselinegevinsten er kun 0.4324 %. Bounds, næsten flad prisrespons i tre senere fits og 207 observationer under den frosne demand-skala begrænser fysisk/prisstyringsfortolkning. Dette er fortsat betingede prognoser med realiseret pris/temperatur på legacy-punktkontrakten, ikke en uafhængig sluttest eller operationel forecast-accept. Alle 12 valgte kandidater består tidsforfining; to ugefits krævede 24 trin/time. Runnerinvocation 297.03 s, to fokuserede tests og syntakskontrol består. Præcis 24 fysiske warmup-timer efter sommertid er rettet; tidligere fits er bevaret uden genkørsel. Se [matematik, valg og råreview](G4_VALIDATION_WEEKS_13_18_REVIEW.md). **Næste: G6a-variansreference, derefter G6b→G8→G7; G5 dokumenteres løbende og afsluttes særskilt.** Ingen lokal kørsel kræves nu.


**Nyeste G4-forsøg, 6. oktober 2026 — 12 uger og tre strategier:** Syv fælles mandagscutoffs (uge 1–5→6, …, 7–11→12), 49 dagsprognoser og 1175/1175 fysiske målte timer er komplette. S1 er retrospektiv (C,k,Xi fra alle 12 uger); S2 fitter C,k,Xi,a per ugedag; S3 fitter dem samlet over 35 dage. S3 med lambda=0.001 har lavest MSE 0.2052253468 mod 0.2065071888 uden straf og baseline 0.2076682236. Straf forbedrer S3 med 0.6207 %, men vinder kun 3/7 uger; uden uge 10 vender gevinsten til en forværring på 0.2836 %. S2's samlede strafgevinst 9.0688 % drives også af uge 10. S3 med straf er derfor kun en foreløbig udviklingskandidat, med lambda=0 bevaret som reference. Realiseret pris/temperatur gør forsøget betinget; ingen blind eller origin-tilgængelig ugeforecast-accept. Se [matematik, kode og resultater](G4_STRATEGIES_12_WEEKS_REVIEW.md). Alle 210 valgte kandidater består tidsforfining; fem fokuserede testmetoder er dækket, ingen fuld testpakke. Én kandidat blev fortsat fra checkpoint, øvrige 104 opgaver genbrugt. Tidligere protokolresultater nedenfor er bevaret og gælder deres egne forsøg. Ingen lokal kørsel kræves nu; G6b→G8→G7 består som obligatorisk kæde.


**Nyeste prognoseudvikling, 6. oktober 2026:** Johans fem-matchende-ugedage-protokol er gennemført for 7.–27. februar: 21 rolling-origin døgnprognoser og 504/504 målte timer. G4 uden straf har MSE 0.1639224439 mod 0.1666013393 med lambda=0.001; straffen øger samlet MSE med 1.6342 %. Uden straf slår samme origins' WM-baseline med 3.8626 %. Valget for denne protokol er lambda=0, baseret på DEVELOPMENT-forecastfejl. Det historiske januarvalg bevares og gælder den gamle protokol. Dette er daglige refits, ikke tre frosne 168-timers forecasts. Se [matematik, kodereview og resultater](G4_WEEKDAY_DEVELOPMENT_REVIEW.md). To fokuserede tests samt alle 42 valgte kandidaters 60/120-kontrol består. Ingen lokal kørsel kræves nu; G6b→G8→G7 består som næste obligatoriske kæde.


**G4-resultat, 5. oktober:** Januar-pilot og begge frosne februarprognoser er nu
kørt og rå-reviewet. 206 residualkald, ca. 83 s inklusive eksport/figurer.
Bedste fleksible g forbedrer DEVELOPMENT-MSE med 0.01504 %; lineær BM
beholdes efter den låste 1 %-regel. Baseline har stadig lavest MSE.
Se [G4-reviewet](G4_JANUARY_PILOT_REVIEW.md) og dets rå benchmark.
Dette er udviklingsvalg på legacy-punktkontrakten, ikke blind validering.
Ældre beskrivelser af manglende G4-pilot nedenfor er historisk status;
den aktuelle næste handling står i reviewet.


## Arbejdsregel fra Johan, 5. oktober: adskil G/F og minimér kontrolarbejdet

G er den deterministiske BM-kæde med et senere sandsynlighedslag; F er den
stokastiske BM-kæde. De har egne planer, kørsler, modelvalg og acceptkrav.
G må ikke vente på F/H-gates, og F må ikke få G's kontroller som standardkrav.
Genbrug stabile matematiske kerner gennem små eksplicitte interfaces; undgå
at koble fitforløb, konfigurationer og testpakker sammen eller kopiere kerner.
G's senere FP/EM-lag arver kun relevante interfacekrav, ikke hele F's forsøgsprogram.

- Vælg den mindste eksisterende test/metode, der dækker den konkrete ændring.
  Ingen fast 20-/29-testpakke for G og ingen automatisk fuld discovery for F.
- Rene tekst-/planændringer kræver ingen numeriske tests eller modelkørsler.
  Kør ikke hele pakken igen efter en bestået relevant kontrol uden ny grund.
- Tilføj kun en test for en konkret væsentlig fejlrisiko, som ikke allerede
  dækkes. Ingen ekstra cases, prøvekørsler eller benchmarkpaneler af vane.
- Ændres fælles kode, kontrolleres kun de faktisk berørte G/F/WM-kontrakter.
  Windows, layout og LaTeX kontrolleres kun, når ændringen kræver det.
- Bevar numeriske tolerancer, datavalidering og nødvendige acceptkrav i selve
  forskningskørslerne. Genbrug gemt evidens; færre gentagelser er ikke ny accept.
- CI foretager kun let syntakskontrol af ændrede Python-filer på PR'er.
  Relevante numeriske checks køres lokalt eller vælges med præcise unittest-
  navne via workflow_dispatch. Fuld regression og F4-PDF er eksplicitte tilvalg.
  Ingen ekstra push-kørsel efter merge. Notér hvilke kontroller der faktisk er kørt.

Denne regel erstatter tidligere standard-testpakker i projektplanerne.

**Revideret 5. oktober mod bacheloropgaven, projekt 111866. G3 er rapporteret complete lokalt med 168/168 scored timer. G4's lille kurveadapter/LS-pilot er implementeret og syntetisk kontrolleret; historisk G4-valg og råreview mangler. G6–G8 er planlagt: FP/mixture/EM, obligatorisk stokastisk prisstyring og rekursiv estimation.**

G er et selvstændigt spor ved siden af F-planen. Formålet er en enkel, hurtig
BM-arbejdsgang: historiske data → deterministisk parameterfit → fastlåst
ugeprognose → sammenligning med baseline og WM4. Grundlaget er den eksisterende
BM1, ikke en ny model med WM's ligninger. Processtøj, filtre, parameterbanker og
fordelingspropagation indgår ikke i første G-leverance. Derefter er et billigt
stokastisk trin en obligatorisk G-leverance. Den hurtige deterministiske kerne
kobles til eksisterende FP/mixture, EM, rekursiv estimation og stokastisk
prisstyring; et varianslag alene opfylder ikke bachelorens formål.

**Præcisering fra Johan, 5. oktober kl. 12:56: WM4 er fælles standard dér,
hvor modellerne kan gøre det samme.** Genbrug konkret baselineimplementation,
fit/predict-protokoller, evalueringsdefinitioner og rapportstruktur. Afvigelser
skal skyldes modellernes matematik eller en eksplicit fælles datakorrektion og
registreres i manifestet; en særskilt BM-baseline er ikke default.

**Navne:** Skriv altid `G-plan/G0` ... `G-plan/G8` i status og henvisninger.
`F4-G`, `F4-G2a`, `F4-G2b` og `F4-G2c` er fortsat de gamle stokastiske studier.
G kræver ikke, at disse studier eller H-f/g afsluttes. G overtager heller ikke
deres evidens eller afslutter deres krav. F-sporets status står i [F4.md](F4.md).
Brugerens nye G-bestilling tillader dette deterministiske spor nu; den ophæver
ikke F-sporets aftalte rækkefølge eller dets senere prognosediagnose.

## Grundarbejde til næste implementation, 5. oktober

[Overdragelsen til Sol](G_IMPLEMENTATION_HANDOFF.md) kortlægger aktuelle
kodeinterfaces, mangler og små acceptkontroller fra G3 til G7/G8. P1/P2 er
nu implementeret: `deterministic --from-fit FITDIR` fortsætter fra et gemt fit
uden baseline/BM/WM-refit. Nye fits eksporterer `fit_snapshot.json`; den
oprindelige 07fad9f-mappe importeres med kilde-/endpointkontrol. Forecast gemmes
før optional scoring. Fælles målte timer, coverage og særskilt scoringstatus
bevares ved delvis/ingen future demand. 20 G-tests samt tre layout/import-
kontroller består på ca. 6.7 s; ingen historisk kørsel er udført her.

Datagennemgangen fandt også legacy-gennemsnit over tilgængelige målere frem
for en garanteret fast population. F4-H2 har verificeret MED/UDEN-medlemskab.
Johan har senere deklareret timeforbrug, foreløbigt kWh og hour-start som
arbejdsantagelser. F4-H2's interval-likelihood er en separat F-opgave.
G's syntetiske G6b/G8/G7 venter ikke på den. Bevar januarfittets
oprindelige kontrakt og auditér før historisk validering. Nye kontrakter
kræver egne resultater; syntetisk udvikling af adapterne kan begynde nu.

## Afgrænsningsreview, 5. oktober 2026

Review mod main `14767314`: G's ti indgangs-/metode-/parameter-/værktøjsmoduler
og deres 55 statisk fundne kildemoduler matcher denne commit. Ingen af disse
importveje omfatter F4-H2's intervalmoduler. G4 bruger den fælles kurvegeometri
`BM1PiecewiseResponse`, sit eget afgrænsede LS-fit og en eksplicit adapter
til `BM1Parameters` med nul diffusion. Importgenbrug er ikke F-accept.

Den gennemgåede G3/G4-dynamik er fortsat
$\dot x=(D-B)/C$, med
$\delta=\tanh(k[f(x)+g(u)]/2)$ og
$D=B+\Xi\delta[(1-B)\mathbf1_{\delta\geq0}+B\mathbf1_{\delta<0}]$.
Lineære f/g giver $\delta=\tanh(k(1-x-u))$.
Deterministisk intervalmiddel er
$\bar D=\bar B+C(x_{n+1}-x_n)/\Delta t$.
Ved senere SDE-udvidelse skal støjintegralet fratrækkes i denne baneidentitet;
endpoint-formlen alene giver ikke stokastisk intervaldemand.
Kapacitetsadapteren bruger $C_{\rm days}=C_{\rm hours}/24$;
diffusionsamplituden skal tilsvarende bruge
$\sigma_{\rm days}=\sqrt{24}\sigma_{\rm hours}$.
Konstant sigma betyder konstant amplitude i $b(x)=\sigma x(1-x)$.

Den fundne sammenblanding var F4-H2's Astra-opgave og readiness-status i
G-overdragelsen; dette er fjernet fra G's opgaveliste. Den historiske
punkt-/interval- og populationsafklaring består som en selvstændig G-opgave.
Reviewet er kilde-/matematikgennemgang, ikke ny numerisk eller historisk
validering. Ingen solverændringer, ekstra tests eller modelkørsler.

## Aktuel G0–G3-leverance, 5. oktober

Johan har bestilt **parameterfit over hele januar**. Første konkrete månedsvariant
bruger 1. januar 2022 kl. 00 til 1. februar kl. 00 (744 timer), ét fælles
(C_hours,k,Xi), fast WM-neutral x0=0.5, og derefter valgfrit 168 timer til
8. februar. `--initial-state fit` bevarer det tidligere fit af x0.
`--fit-only` viser januar uden februardata. BM og WM bruger begge
den fulde fitmåned. Månedsvarianten har warm-up=0 for at score hele januar;
ugevarianten bevarer WM's 24 timer og én fituge. Ingen daglige state-resets.

Implementationen omfatter dataaudit, træningsfast fælles skalering, genbrugt
WM-baseline, lineær BM1-forwardmodel, bounded SciPy-TRF, checkpoints, cutoff,
tidsforfining og fælles WM/BM-rapporter. Se [G_JANUARY.md](G_JANUARY.md) for
kommandoer, bounds, numerik og begrænsninger. Måletype/enhed er endnu ikke
afklaret nok til et historisk videnskabeligt acceptclaim; kørslen mærkes
hourly-point-approximation og development.

En syntetisk 744+168-timers kæde med to starter er gennemført på cirka 22 s,
130 residualkald og bestået tidsforfining. Det er implementerings-/runtimeevidens,
ikke historisk januar-estimation eller en retfærdig modelrangering: data er
genereret med BM1. Den store historiske input-workbook kunne ikke materialiseres
gennem denne chats connector. Johan har siden kørt januarfittet lokalt på
`07fad9f`, run `g_january_fit_20261005_154824`, med `status=complete`:
624 målte timer, 120 manglende demand-timer, komplette importerede pris-/temperaturinputs.
Normaliseret MSE: BM 0.0066056846, baseline 0.0066347027, WM 0.0072743310.
BM's in-sample MSE-gevinst er 0.4374 %. Kilden er terminaloutput; råfiler,
identifikation, præcis numerikfejl og samlet runtime er endnu ikke uafhængigt
reviewet. Genkør ikke januarfit for denne planrevision.

Johan har efterfølgende rapporteret G3 på main `607dcd7`, run
`g_january_forecast_20261005_204059`: forecast/scoring complete,
168/168 timer 1.–7. februar, samme målte maske for O/BM/WM, ingen optimering.
MSE i rå workbook-enheder: O=0.06934621735, BM=0.06951591504,
WM=0.07099789830. BM er ca. 0.245 % dårligere end O og ca. 2.087 % bedre
end WM målt på MSE. Det er et negativt baseline-resultat på en eksponeret
udviklingsuge, ikke et signal om automatisk genfit eller en blind sluttest.
CSV/plots/råarkiv er ikke modtaget i chatten; uafhængigt råreview mangler.

G4 implementerer nu to-stykkers g i en separat adapter, som genbruger
`BM1PiecewiseResponse` og kan levere eksisterende `BM1Parameters`.
Den gamle lineære kernel og G3-snapshot er bevaret. `deterministic_response
--reference-fit FITDIR` læser kun de oprindelige TRAIN-eksporter, verificerer
lineær replay, fryser k/baseline/WM/skala og bevarer x0-policy. Standard:
lambda=0 og 0.001, starter g_mid=0 og 0.25, samlet højst 3.000 residualkald
og 600 sekunder beregning inklusive replay/forfining. Kandidaterne vælges
ikke mellem strafstyrker på TRAIN. Kommandoer og status:
[G_JANUARY.md](G_JANUARY.md). Ni små G4-kontroller dækker paritet, enheder,
LS/budget, arkiv og kausalt gemt-fit-forecast. Ingen historisk G4-kørsel er
udført her; endeligt modelvalg afventer DEVELOPMENT-sammenligningen.


## 0. Bachelorens formål og krav

Autoritativ kilde: Johans screenshot af den registrerede projektbeskrivelse,
projekt-ID **111866**, **Stokastisk optimal regulering og estimering til
udnyttelse af forbrugerfleksibilitet** (Screenshot 2026-09-19 230718.png,
genlæst 5. oktober). Formålet er at finde prissignaler, som får en
forbrugergruppes samlede forbrug til at nærme sig en forudbestemt målprofil.
Responsen beskrives med SDE'er; mixture-approksimation af Fokker–Planck
indgår udtrykkeligt. Forecast-kvalitet er et hjælpemål, ikke hele formålet.

| Formelt arbejde | Eksisterende genbrug | G-leverance / kobling |
|---|---|---|
| Approksimere FP med mixture-metoder | F2/F3: BM FP/FV og dynamisk Beta-mixture | G6b: samme SDE og initiallov i FP, mixture og EM; fordelinger, momenter, runtime. |
| Formulere stokastisk optimalt kontrolproblem | F3's fælles trackingproblem | G7: fast målprofil, prisbounds, forventet trackingtab og deklarerede øvrige led. |
| Løse kontrolproblemet med single shooting og FP | F3's forward-evaluatorer og prisoptimering, inklusive mixture | G7: prisvektor → fordeling → objective → optimizer; krydsevaluér politikker. |
| Rekursiv tilstands- og parameterestimation baseret på FP | F4's FV/mixture-filtre og parameterbanker | G8: sekventielle Bayes-opdateringer og posterior-handoff til G7. Batch-LS alene er utilstrækkeligt. |
| Rapport og mundtligt forsvar | F1–F4-noter og gemt evidens | G5 og slutrapport: krav→kode→forsøg→resultat→begrænsning, også negative resultater. |

EM betyder **Euler–Maruyama**, som på vejledertavlen, ikke
expectation-maximization. EM simulerer SDE-baner og giver numerisk kontrol;
den erstatter hverken FP/mixture eller rekursiv estimation. EM er ikke et
selvstændigt nummereret STADS-krav, men indgår eksplicit i G6b/G7.

Kravene gælder det samlede projekt. Gennemført F-arbejde kan dække et krav
med præcis kilde og begrænsninger; gentag ikke hele F1–F4. Nye parametre,
kurver og initiallove kræver dog en lille kontrol af den genbrugte numerik.
G var før revisionen ufuldstændig som selvstændig bachelorplan: FP/mixture
og rekursiv estimation manglede, og prisstyring var valgfri.

### Hvad WM faktisk gør

Kildecommit for review: 9da2777fddefd32c870c1d66c0a07373796cf489.

- WM1 bruger en én-dimensional FP-vej via bm_1_fokker_planc med
  Beta-mixture-projektion. WM2/WM3/WM4 bruger wm_fokker_planc, hvis backend
  er finite_volume_2d; det er ikke automatisk en mixture-metode.
- WM4 har fokker_planc_solver for fordelingen af F og R. At metoden findes
  betyder ikke, at den kaldes af den deterministiske fit/prognose.
- WM4's process_uncertainty og process_variance_kernel propagérer
  kovarians for Euler-diskretiseret lineær dynamik. De beregner ikke en fuld
  FP-tæthed og simulerer ikke EM-baner. Stateafhængig diffusion evalueret langs
  den deterministiske bane er en approksimation til den fulde model.
- wm_4/test_diffusion_hypotheses.py bruger varianskernen. build_record
  optimerer initialtilstanden mod prognosedagen (oracle). Genbrug metrics
  og figurer, men ikke den initialisering som en kausal prognose.
- De gennemgåede WM1–WM4-modelklasser har ingen særskilt EM-banesolver.
  BM har allerede bm_1.euler_maruyama_solver og Brownian-hjælpere.

Genbrug WM's trinvise fit→stokastik og rapportering, men BM's egen SDE,
randstruktur og F-sporets FP/mixture/EM-/estimation-/kontrolkode.

## 1. Leverancer og rækkefølge

| Etape | Arbejde og konkret leverance | Afslutningspunkt |
|---|---|---|
| G-plan/G0: fælles forsøgsgrundlag | Fastlæg population, WM4-commit, datakontrakt, tidsenhed, fælles baseline, pris-/effektskala, historik, split og runtime-budget i et manifest. | Begge modeller kan få samme fysiske inputs, observationer og information ved samme forecast-origin. Uklar måletype giver kun en mærket udviklingsdemo. |
| G-plan/G1: deterministisk forwardmodel | Genbrug BM1's ODE og F1-reference. Klargør en lille evaluator med punkt-/intervaloutput, energibalance og en eksakt intervalvej, hvor antagelserne gælder. | Meningsfulde tests af energibalance, randtilfælde, inputskift og enheder; eksakt vej kontrolleret mod uafhængig ODE. |
| G-plan/G2: parameterfit | Fit fælles C, k og Xi samt højst én initialtilstand for et sammenhængende træningsforløb; først f(x)=1−2x og g(u)=1−2u. Bounded nonlinear least squares med få startpunkter. | Fit-artifact med parametre, score, stopårsag, samtlige solve-kald, runtime, initialisering og lokal identifikationsdiagnose. |
| G-plan/G3: første ugeprognose | Frys parametre/baseline og eksakte BM/WM-sluttilstande ved origin. Tilføj forecast fra gemt fit uden optimering; simulér syv døgn uden fremtidig demand. | Gem prognose før optional scoring. Fælles målemaske, coverage, CSV/ugefigur og RMSE/MAE/bias mod baseline. Resultatet må gerne være negativt. G0–G3 er første brugbare leverance. |
| G-plan/G4: fælles f/g-kurver | Undersøg først en lille prisresponsudvidelse, derefter eventuelt f; brug eksisterende kontinuerte, monotone kurveobjekter. Ét kurvepar per population og træningsfit. | Afgrænset udviklingssammenligning mod lineære kurver; validering afgør om ekstra fleksibilitet beholdes. Ingen kurver per dag. |
| G-plan/G5: afsluttende benchmark og dokumentation | Fastlås opskrift og modelvalg før sluttest. Evaluer de på forhånd valgte ugeorigins i én måned med identisk information og rapportér alle uger. | Reproducerbar lille benchmarkpakke med kvalitet, runtime, begrænsninger og klart skel mellem udvikling og sluttest. |
| G-plan/G6: stokastik, FP/mixture og EM | G6a: billig variansreference. G6b: FP/FV, dynamisk Beta-mixture og EM på samme lille forsøg. | Numerisk forfining, fordelinger/momenter, EM-usikkerhed og randdiagnoser; variansapproksimation mod FP. |
| G-plan/G7: obligatorisk stokastisk prisstyring | F3's single shooting med FP/mixture, få prisblokke og fast målprofil; deterministisk kontrol som reference. | Fælles stokastisk objective, finere FP/mixture og parret EM-evaluering; G8-posterior overføres til kontrol. |
| G-plan/G8: rekursiv tilstands- og parameterestimation | F4's FP/mixture-filtre og lille én-parameterbank med egne conditional states. | Sekventiel opdatering, syntetisk identifikationskontrol og posterior→G7-handoff; øvrige parametre eksplicit faste. |

**Præcisering fra Johan, 5. oktober: G skal også indeholde stokastik.**
G0–G3 er første deterministiske milepæl. Efter et afgrænset G4-valg låses
strukturen til G6b, derefter G8 og G7. G6a er billig reference. G7 kan
forberedes med kendt syntetisk initiallov før G8; samlet kræves også handoffet.
G5's brede månedsbenchmark må ikke blokere kernekravene; slutrapportering
følger G6/G8/G7. Stor baselinegevinst er ikke en adgangsgate.
G5's punktprognosebenchmark afslutter ikke G. G6–G8 er obligatoriske
integrationstrin. Økonomisk profit-/indkøbsoptimering er fortsat en mulig
senere udvidelse ud over den krævede målprofilstyring. G6 får eget
udviklings-/testsplit fastlagt i G0, så allerede inspicerede G5-data ikke
senere præsenteres som blind sluttest for usikkerhedsmodellen.
Ingen stor arkitekturomskrivning eller komplet ny F1–F4-valideringsserie.

### 1a. Bindende genbrug fra WM4

| Område | Konkret WM4-reference | G-planens valg |
|---|---|---|
| Baseline | `output_model.py`: `statistical_output.fit/predict`, `output_callable` | Samme implementation, indstillinger, historik, cutoff og features. Fit én gang per origin; samme fitted baselineforløb gives til begge modeller efter eksplicit enhedsomregning: B=O. |
| Ugeprotokol | `predict_weekly.py`: én fituge, 24 timers warm-up, næste 168 timer | Første standardbenchmark for begge modeller. 28 træningsdage er kun en senere fælles variant. |
| Dagsprotokol | `predict_daily.py`: fem tidligere matchende ugedage, 24 timers warm-up | Samme observationsudvalg og cutoff ved et særskilt dagsbenchmark. Daglig opdatering og en uges åben prognose holdes adskilt. |
| Fejlmål | `deviation_metrics` | Samme MSE, RMSE, MAE og bias=prediction−observation på identiske observationer og i fælles enheder. |
| Performanceanalyse | `analyze_performance.py`: `metric_gain`, `analyze`, `summary_table`, `weekday_summary` | Samme baselinegevinst, procentvis MSE-forbedring, bedre/lige/dårligere dage, gennemsnit/median, ugedagsopdeling og bedste/værste dage. |
| Figurer | `save_plot` i dags-/ugekørslerne | Samme datoakse, observationsmarkører, baseline, prognose og pris. Sammenlignelige paneler med korrekte modelnavne/enheder. |
| Eksport | Daglige metrics-CSV og performance-CSV | Samme metriknavne/suffikser og baselinekolonner; BM får eget modelnavn. Fælles evaluator accepterer modelnavn og bevarer kompatibilitet med WM4-output. |
| Usikkerhed (G6) | `test_diffusion_hypotheses.py`, `process_uncertainty` | Samme nominelle niveau, dækning, intervalbredde, intervalscore og prognosehorisonter; modelspecifik varianspropagation. WM's oracle-initialisering må kun genbruges i særskilt oracle-diagnostik. |
| Diagnostik | Baselinegevinst, residualer og initial-state-følsomhed | Samme spørgsmål/informationsgrænse; modelspecifik beregning. WM's dæmpningsratio og acceleratorstate kopieres ikke til BM. |

Baselinen bruger aktuelt døgn-/sæsonharmoniske led, ugedag, temperatur og
interaktioner, med robust ridge/Huber-fit. Defaults er base_temperature=18,
hour_harmonics=4, season_harmonics=2, ridge=1e-3, huber_delta=1.5, max_iter=50.
De versionsfastlåses; samme fitted objekt eller serialiserede parametre bruges
for BM/WM. Et fem-ugers timegennemsnit erstatter ikke denne baseline: fem
matchende ugedage er her WM4's dynamiske træningsudvalg.

`statistical_output.predict` og `output_callable` begrænser aktuelt baseline
til [0,1]. Det er en eksplicit del af fælles baselinekontrakt: gem policy og
antal/rå overskridelser i en audit. Ingen ekstra BM-specifik klipning af
baseline, observationer eller state. Nødvendige datakorrektioner, fx en anden
normalisering, gælder begge modeller og versionsmærkes; original WM4 under en
anden kontrakt står særskilt. Dette præciserer planens regel mod skjult clipping.

WM's baseline interpoleres aktuelt lineært mellem dens fine samples.
Hovedsammenligningen genbruger dette tidsforløb. Den eksakte BM-intervalvej
bruges kun, når dens konstant-input-forudsætninger gælder. En held-baseline-
variant kræver samme ændring for WM og separat resultatlabel. Samme outputgrid
kræver ikke samme interne solver eller 60 ODE-substeps per time i BM.

Genbrug kode direkte, hvor API'et passer; ellers udtræk små modeluafhængige
helpers med bagudkompatible WM-wrappers og numerisk paritetstest. Ingen kopier
af baseline-/metricskode, der senere udvikler sig forskelligt. WM's QP-fitter
erstatter ikke BM's nonlinear fit. Predict-setup, der kræver fremtidigt observed
I, opdeles i inputforberedelse og efterfølgende scoring for begge modeller.

Performancekonvention: gain=baseline_error−model_error; for bias bruges
|baseline_bias|−|model_bias|. Procentgevinst er 100*gain/baseline_MSE og NaN
ved baseline_MSE≤1e-12 som i WM. Better/equal-tolerancen 1e-12 genbruges til
kompatibel rapportering, men er ikke statistisk signifikans eller praktisk
modelvalg; G's numeriske/praktiske tolerance rapporteres separat. Vis både
WM's gennemsnit af dagsmål og pooled fejl over observationer med tydelige
labels: gennemsnitlig dags-RMSE er ikke pooled RMSE. Bedste/værste dage er
visninger af den samlede rapport, ikke udvælgelse af testdata.

Figurerne skal vise, om korrektionen forbedrer baseline, hvornår den hjælper
eller skader, og hvor stor den er. Vis D−B for BM og I−O=R for WM i samme
forbrugsenhed. BM's x og eventuelt dx/dt vises separat fra WM's F/R;
WM's dR/dt er ikke BM's dx/dt. G0–G5 viser punktprognoser; G6 tilføjer tydeligt mærkede approksimative
stokastiske bånd og fælles usikkerhedsmål.
Billige dags-/ugedagsanalyser bruger gemte prognoser uden nye fits. Øvrige
WM-diagnoser tilføjes kun ved relevans og inden for eksisterende G-budget.

G0 leverer en kompatibilitetsmatrix med kildecommit, genbrugt helper, nødvendig
adapter og begrundelse for hver forskel. G3 kontrollerer identisk baseline på
fælles timestamps og identiske metrics for samme predictions. G5 leverer
fælles dag-/ugefigurer, metrics og WM-lignende performanceanalyse.

## 2. Første trin: samme BM1, uden processtøj

Med x i [0,1], normaliseret baseline B og pris u er

$$\dot x(t)=\frac{D(t)-B(t)}{C},\qquad x(t_0)=x_0,$$

$$\delta(x,u)=\tanh\!\left(\frac{k}{2}[f(x)+g(u)]\right),$$

$$D(t)=B(t)+\Xi\begin{cases}(1-B(t))\delta(x(t),u(t)),&\delta\geq0,\\B(t)\delta(x(t),u(t)),&\delta<0.\end{cases}$$

C og k er positive, Xi og x0 ligger i [0,1]. f og g er kontinuerte,
ikke-voksende og går fra 1 til −1. Det giver indadrettet randdrift for gyldige
inputs og dermed invarians af det kontinuerte tilstandsinterval. Numeriske
overskridelser skal stadig kontrolleres; ingen skjult clipping af state/data.
Fysisk tid skal være ens i C og t: C=C_phys/P_scale er en tidsstørrelse.
Projektets nuværende dagsskala kan genbruges med eksplicit omregning til timer.

Den deterministiske energibalance er

$$C[x(t_b)-x(t_a)]=\int_{t_a}^{t_b}(D(t)-B(t))\,dt.$$

Det betyder også, at en begrænset virtuel batteritilstand ikke kan kompensere
for vedvarende baselinebias ubegrænset. Store systematiske residualer skal
diagnosticeres, før man tilføjer flere kurveparametre. Nul processtøj og et
bestemt x0 giver én bane; det er generelt ikke middelbanen af den ikke-lineære
stokastiske BM. Historiske målefejl/residualer kan godt findes, selv om den
latente dynamik i G0–G5 er deterministisk.

### Eksakt og billig reference for lineære kurver

For f(x)=1−2x, g(u)=1−2u og konstante B_j,u_j over et interval af længde h_j,
sæt q_j=1−u_j og gamma_j=1−B_j, hvis x_j<q_j, ellers gamma_j=B_j. Da fås

$$x_{j+1}=q_j+\frac{1}{k}\operatorname{arsinh}\!\left(\sinh(k(x_j-q_j))\exp[-k\Xi\gamma_jh_j/C]\right).$$

Formlen er allerede dokumenteret i F1 og implementeret som et fast benchmark.
G1 generaliserer om nødvendigt den lille intervalrutine til parameterforslag
og inputarrays; den importerer ikke benchmarkets hårdkodede parametre til
produktionsfit. Kald med store/små argumenter kontrolleres for cancellation
og overflow. Xi=0 og x_j=q_j skal give konstant tilstand.

For intervalmiddeldata kan efterspørgslen beregnes direkte fra balancen:

$$\widehat y_j=B_j+\frac{C(x_{j+1}-x_j)}{h_j}.$$

Det kræver ingen fin kvadratur under ovenstående forudsætninger. For energi
er observationen i stedet integralet, med korrekt fysisk effektskala.
For punktdata bruges D ved det deklarerede måletidspunkt og den korrekte side
af et inputskift. Intervalgrænser opdeles ved alle B/u-skift.

En konstant-timebaseline må ikke indføres alene for at bruge formlen, hvis
den aftalte WM-sammenligning kræver en kontinuerlig baseline. G0 fastlåser
inputrepræsentationen for begge modeller. G0–G3 bruger ved varierende B en billig skalær RK4-vej med de samme BM1-
ligninger, verificeret mod eksisterende segmenteret `solve_ivp`. Første fit har
60 trin/time for at ramme knækkene i WM's fine, lineært interpolerede baseline;
120 trin/time er numerikkontrollen. Et grovere grid afviste en syntetisk kandidat
under uændret præcisionsgrænse. Dette er en kontrolleret numerisk specialisering,
ikke en konstant-baseline-ændring. Stykvise f/g er endnu ikke implementeret i
G-fitteren; den eksisterende ODE-vej er fortsat tilgængelig.

## 3. Parameterestimation

Første fit har theta=(C,k,Xi) og eventuelt x0 som en fjerde ukendt:

$$\min_{\theta,x_0}\frac{1}{2N}\sum_{i\in\mathcal T}\left(\mathcal H_i[D_{\theta,x_0}]-y_i\right)^2.$$

H_i er punktværdi, intervalmiddel eller energiintegral efter datakontrakten.
Observationernes tidsopløsning bestemmer residualernes antal. Et fint internt
grid eller interpolation af y skaber ikke nye uafhængige observationer.
Priser og baseline er faste inputs under fit; A/aggregatorkøb indgår ikke i
dette estimationsobjective. En enkelt samlet theta bruges på alle træningsdage.

Praktisk valg: `scipy.optimize.least_squares(method="trf", loss="linear")`
med bounds, skalerede variable og to-punkts-Jacobian i første implementation.
Brug log C/log k eller eksplicit parameterskalering; dokumentér endelige
fysiske bounds i G0 før optimering. Grænser bestemmes af enheder, model og
udviklingsdata, aldrig af sluttest. Funktionen har et knæk ved delta=0;
komplekse skridt er ikke en gyldig automatisk differentiation her. Ved senere
kurvefit kommer også knæk i f/g. Ingen egen global optimizer eller fuld
Hessian er nødvendig; sensitivitetssystemer tilføjes kun ved målt behov.

Begynd med højst fire deklarerede startpunkter. Inden for samme model vælges
den bedste gyldige, konvergerede TRAIN-kandidat; valg mellem modelvarianter
sker på DEVELOPMENT. Hvis et numerisk check ikke består, registreres fejlen
og forsøget afklares. Ingen skjult udskiftning af modellen ved fejlede kald.

BM-fit er generelt ikke-konvekst, fordi theta ændrer både tilstanden og
responsen. WM4's constrained drift-regression kan være en konveks QP, når
rekonstruerede states er fastholdt; det gør ikke BM-fit til en QP og garanterer
heller ikke, at WM4's regressionsscore er optimal for en hel ugeprognose.
Sammenlign begge modellers held-out forbrugsscore, ikke deres interne fitmål.

### Små identifikationskontroller

- Undersøg skaleret Jacobian/SVD, aktive bounds og forskelle mellem de højst
  fire startpunkter. Det er lokal diagnostik, ikke et globalt entydighedsbevis.
- Nær delta=0 er fleksibilitetsamplituden omtrent proportional med Xi*k;
  parametrene kan være svære at skille ad. Ved svag excitation kan C også
  være dårligt bestemt. Xi=0 gør dynamikparametre uidentificerbare fra demand.
- Brug højst én erklæret reduceret variant med k fastholdt, hvis fuldt fit er
  ustabilt. Et godt forecast må rapporteres med svag parameteridentifikation.
  Ingen store profilgrids, bootstraps eller recovery-populationer i G-kernen.
- På udviklingsdata kan kumuleret y−B og den krævede statebevægelse bruges som
  billig feasibility-diagnose. Observeret støj gør ikke dette til et eksakt
  fysisk certifikat eller en erstatning for simulation-fit.

## 4. Data, initialisering og retfærdig WM4-sammenligning

**Aktiv bestilling:** hele januar 2022 → næste 168 timer, MED, neutral start
og nul warm-up. Genbrug det rapporterede januarfit, når råfiler/kontrakt er
verificeret; snapshot-adapteren er en ny opgave. Ugeprotokollen nedenfor er
den oprindelige reference og vælges ikke automatisk i stedet for månedsfittet.

Den oprindelige standard er WM4's ugeprotokol: én population (først MED), én historisk fituge
med 24 timers warm-up og en prognose på næste 168 fysiske timer. Baselinefit
bruger samme tilgængelige historik før origin som WM4. Datoer og historik
fastlægges i G0. Et 28-dages fit er en senere fælles variant, ikke BM-default.
Et eventuelt dagsbenchmark bruger samme fem matchende ugedage som WM4 og
rapporteres særskilt. Kurver er fælles på tværs af de historiske træningsdage
og fastlåste under forecast. Første integration bruger en lille udviklingsuge;
fuld deklareret træning køres inden den tilhørende sluttest. Afvigelser fra
24 timer per lokalt døgn håndteres eksplicit ved DST.

Der bruges en tidsmæssig TRAIN/DEVELOPMENT/TEST-opdeling. Februar-eksempler,
som allerede er inspiceret, er udviklingsdata. Holdout er en særskilt endnu
uåbnet periode, hvis en sådan findes; ellers kaldes studiet udviklingsbenchmark.
Ved månedlig rolling-origin-test låses hele fitopskriften før første score.
Tidligere testugers data kan indgå ved senere origins efter den låste opskrift;
de må ikke bruges til manuel ændring af modelvalg eller budgetter under testen.

**Populationsaudit:** den nuværende G-import genbruger `consumption_series`,
som gennemsnitter de tilgængelige målere per time med skip-NaN. Det er ikke
F4-H2's faste populationskontrakt. Gem medlemsliste/antal per time og bevar
legacy-status. I en ny fast-populationsvariant giver et manglende medlem en
manglende aggregatmåling for begge modeller. Metadataaudit alene bekræfter
ikke enheder, måletype eller tidszone; se [H2_DATA_CONTRACT.md](H2_DATA_CONTRACT.md).

G0-manifestet fastlåser:

1. Hvilken population/måler, fysisk power-/energienhed, intervalbetydning,
   timestamp-side og timezone hver observation har. Gaps maskeres eksplicit;
   gap-perioder kræver kendte inputforløb for statepropagation eller separate
   segmenter med deklareret initialisering. Ingen fremtidig udfyldning.
2. Fælles WM4-baseline B=O fra `statistical_output`, estimeret uden data efter forecast-origin. Baselinens
   egen træningshistorik og in-sample/cross-fit-konvention gemmes. Udvid ikke
   til en ny baseline-modelsøgning i G. En baseline uden dynamisk pris som
   feature er ikke i sig selv bevis for en kausalt prisfri baseline.
3. Samme rå pris-/temperaturforløb og informationstidspunkt. WM4 bruger aktuelt
   signed U i [−1,1], BM u i [0,1]. Ved samme normalisering kan u=(U+1)/2
   bruges; det skal verificeres på det konkrete input. Sammenlign i fælles
   fysiske forbrugsenheder, ikke ukritisk på hver models egen normalisering.
4. Træningsfast effektskala uden skjult klipning eller held-out genestimation.
   Hvis gyldige BM-inputs ikke kan opnås, registreres domæneproblemet. Original
   WM-resultat fra en anden skala/kontrakt må kun stå som separat reference.
5. Finite parameterbounds, starter, numerik, modelvarianter, stopregler,
   kriterium for kompleksitet og både per-fit- og samlet beregningsbudget.

Kausal initialisering: standarden er fast neutral x0=(1-INIT_F)/2=0.5,
parallelt med WM4's aktuelle INIT_F=0 og R0=I0-O0. BM/WM-states har
forskellig fysisk betydning; det fælles er neutral signed koordinat.
Alternativt fit højst ét x0 på træningen med `--initial-state fit`.
Viderefør den simulerede sluttilstand til forecast-origin.
En deklareret warm-up før det scorede træningsvindue kan bruges. Ingen frie
daglige state-resets og ingen fit af x0 til fremtidige y. WM4 anvender samme
informationsgrænse med sin egen state-dimension. Hvis en state-only opdatering
fra et kort historisk prefix senere ønskes, valideres den som en separat variant.

WM4-noternes optimering af starttilstand mod hele testdagen er en ORACLE-diagnose.
Den må ikke stå som prognose over for kausal BM. En eventuel fælles oracle-
sammenligning rapporteres separat og fritager ikke for hovedbenchmarkets cutoff.
Kendt faktisk pris og temperatur i en historisk test giver en betinget prognose.
En operationel ugeprognose kræver pris-/vejrinputs, der faktisk var tilgængelige
ved origin, og behandles først, når sådanne inputs findes.

## 5. Fleksible f/g i et lille ekstra trin

### Første G4-pilot: matematisk definition og forsvar

**Valget er to stykker i g, lineær f og ét fælles nonlinear least-squares-fit.**
Koden findes allerede; denne præcisering dokumenterer den implementerede
metode. Kun én ny kurveparameter tilføjes til den lineære reference:
$a=g(0.5)\in[-1,1]$. Med normaliseret pris $u\in[0,1]$ sættes

$$
f(x)=1-2x,\qquad
g_a(u)=\begin{cases}
1+2(a-1)u, & 0\le u\le 0.5,\\
a-2(a+1)(u-0.5), & 0.5\le u\le1.
\end{cases}
$$

Begge stykker giver a ved knuden; endpoints er 1 og −1. Hældningerne er
$2(a-1)$ og $-2(a+1)$, begge ikke-positive præcis når $-1\le a\le1$.
Dermed giver en enkelt box-bound kontinuitet og monotonicitet uden
sortering, clipping, splines eller ekstra constraints. $a=0$ genskaber
præcis $g(u)=1-2u$; referencefamilien er indeholdt i udvidelsen.
Knuden ligger i **pris**, ikke tid: samme kurve bruges alle januars timer.

Den eksisterende BM1-dynamik bevares, med tid i timer:

$$
\delta_\theta(t)=\tanh\!\left(\frac{k}{2}[1-2x_\theta(t)+g_a(u(t))]\right),
$$
$$
D_\theta(t)=B(t)+\Xi\delta_\theta(t)
\begin{cases}1-B(t),&\delta_\theta(t)\ge0,\\B(t),&\delta_\theta(t)<0,\end{cases}
\qquad \dot x_\theta(t)=\frac{D_\theta(t)-B(t)}{C_{\rm hours}}.
$$

For $B,\Xi\in[0,1]$ ligger demand i [0,1]. Ved x=0 er f+g≥0,
så state kan kun bevæge sig indad; ved x=1 er f+g≤0. Det begrunder den
kontinuerte models stateinterval. Den numeriske solver skal stadig kontrollere
sine RK4-stadier; den må ikke skjule en for grov tidsdiskretisering med clipping.

**Hvad fittes?** $\theta=(\log C_{\rm hours},\Xi,a)$ over hele det
sammenhængende TRAIN-forløb. C og Xi tilpasses samtidigt med a, fordi demand
afhænger af alle tre og af den resulterende statebane. K holdes på originalt
G2-fit; x0=0.5 er fast i den aktive wm-neutral-variant. Kun hvis originalarkivet
brugte `initial-state fit`, medtages ét fælles x0 som fjerde koordinat.
Baseline, WM, skala, tidsgrid, målemaske og observationskontrakt genbruges.
Standardbounds er C_hours∈[1,744], Xi∈[0,1], a∈[−1,1]; arkivets bounds
er autoritative. Log-C sikrer positiv kapacitet og er alene et koordinatskift.

Lad I være de målte TRAIN-timer, N=|I| og y_i deres normaliserede demand.
For den originale hourly-point-approximation er r_i demand ved time-start:

$$
r_i(\theta)=\frac{D_\theta(t_i)-y_i}{\sqrt N},\qquad
L_\lambda(\theta)=\sum_{i\in I}r_i(\theta)^2+\lambda a^2.
$$

SciPy får residualvektoren $(r_i)_{i\in I}$ med én ekstra række
$\sqrt\lambda a$. Dens interne faktor 1/2 ændrer ikke minimizeren;
eksporten rapporterer MSE og $\lambda a^2$ hver for sig. Manglende demand
giver ingen residual, men state simuleres gennem timerne uden dagsreset.
Division med √N gør datasummen til MSE, så straffens skala ikke afhænger
direkte af antallet af målte timer. λ=0 er almindelig least squares;
λ=0.001 trækker den ekstra knude mod den lineære reference a=0.
Styrken er et forudvalgt pilotvalg, ikke udledt af bogen eller et kendt støjniveau.

**Hvorfor SciPy TRF?** g er lineær i a ved fast u, men hele forwardmodellen
er nonlinear gennem tanh og statebanen. `numpy.linalg.lstsq` eller
`scipy.optimize.lsq_linear` kan derfor ikke løse hele fittet direkte.
`least_squares(method="trf", loss="linear")` håndterer netop residualer
og box-bounds. Grundideen er den lokale approksimation
$r(\theta+p)\approx r(\theta)+Jp$: et lille least-squares-delproblem med
begrænsede skridt, som ikke kræver en egen fuld Hessianimplementation.
Der bruges den eksisterende two-point-Jacobian og to starter a=0/0.25.
Konvergens betyder et lokalt numerisk stop, ikke et bevis for globalt optimum
eller identificerbare fysiske parametre.

**Forsvar og begrænsning.** En ekstra frihedsgrad er den mindste udvidelse,
der kan skelne prisresponsen under og over 0.5 uden at ændre BM's dynamik.
Fast k begrænser kompensation mellem responsstyrke og kurveform; det beviser
ikke identifikation. Data-Jacobianens singularværdier/rang, active bounds og
prisernes dækning skal læses sammen med forecastet. Straffens residual må
ikke tælle som ekstra dataevidens for rang. Ved Xi=0 har g ingen effekt;
ved lille k kan især Xi og k være svære at adskille.

Modellen flytter også energi: $\int_0^T(D-B)\,dt=C[x(T)-x(0)]$, hvis
absolutværdi er højst C. En mere fleksibel g kan derfor ikke frit rette en
vedvarende baselinebias. Dette er en kontinuert modelidentitet, ikke en
intervalvalidering af de gamle punktobservationer. Februarugens positive bias
beviser ikke i sig selv, hvor fejlen opstår.

Metoden er inspireret af Nocedal & Wright, *Numerical Optimization*, 2. udg.,
kap. 10.3, især Gauss–Newton s. 254, og Alba Gonzalo Primo, *Adaptive demand-side
flexibility exploitation through dynamic price signals* (2025), kap. 6.3.1,
s. 96–98: en enkel responsfamilie og trinvis estimation med fast k i kurvetrinnet.
Albas EKF/likelihood er ikke dette deterministiske LS-fit, og hendes frie affine
kurvers skalainvarians overføres ikke direkte til vores faste endpoints.
[SciPy-dokumentationen](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
beskriver den valgte bounded TRF-metode. Der hævdes ingen konveksitet eller
historisk gevinst. Implementation: `fit_response` i
`src/method/bm_1_deterministic_response.py`; arkiv/eksport i
`src/tools/bm_1_deterministic_response.py`; [kørsel og modelvalg](G_JANUARY.md).

### Afgrænsning og efterfølgende kurvevalg

Genbrug `BM1PiecewiseResponse` med endpoints 1,−1. Undersøg først to stykker
(indre knude 0.5), og kun ved relevant udviklingsgevinst fire stykker med
knuder 0,0.25,0.5,0.75,1. Første kurveforsøg udvider g med f lineær; næste
kan udvide f. Det er én sekventiel, på forhånd budgetteret vej, ikke alle
mulige kombinationer. En negativ to-stykkers pilot kan afslutte G4 med
lineær reference og en begrundet anbefaling om at stoppe.

Hold k på G2's udviklingsvalgte værdi i den første kurvepilot for at begrænse
kompensation mellem hældning og kurveform. Fit C, Xi og kurveværdier samlet;
bevar G2's initial-state-policy: wm-neutral holder x0=0.5, mens kun varianten
`initial-state fit` estimerer træningens x0. To fire-stykkers kurver giver højst seks ekstra frihedsgrader.
Alle stykker indgår samtidigt i samme residualvektor over hele træningen.

Monotonicitet håndhæves ved parameterisering eller en optimizer, der faktisk
understøtter de nødvendige constraints. TRF understøtter box-bounds, ikke
arbitrære ordensuligheder. Genbrug en gyldig eksisterende parameterisering;
hvis positive normaliserede fald bruges, fastlås én logit for at fjerne
softmax-skiftinvarians. Eksakte flade stykker ligger da eventuelt kun i grænsen,
hvilket skal oplyses. Undgå sortering/clipping af kandidatknuder under fit.

En lille, dimensionsløs straf mod lineære kurver eller knæk kan tilføjes som
ekstra residualer, med styrke fastlagt på udviklingsdata og gemt i manifestet.
Test højst to på forhånd valgte styrker inklusive nul inden for G4's samlede
budget. Rapportér både datafejl og regulariseret mål. Den enklere model vælges,
hvis udviklingsgevinsten ligger inden for den på forhånd fastlagte praktiske
tolerance, med numerisk fejl medregnet. Pris-/stateintervaller uden tilstrækkelig
dækning får ingen claim om estimeret kurveform.

Før åbning af de nye G4-forecastresultater låses første pilots praktiske regel:
pooled MSE på den fælles DEVELOPMENT-målemaske er primær metric. Behold lineær
BM, hvis dens MSE ligger inden for 1 % af den bedste numerisk checked kandidats
MSE, inklusive den numeriske fejl. 1 % er et konservativt praktisk valg for
denne lille pilot, ikke et estimeret signifikansniveau. Ved større forskel
reviewes identifikation, bounds/kurveform og MAE/bias før et begrundet valg.
Ingen automatisk vinder eller krav om stor O-gevinst; ændring af reglen efter
åbning mærkes som ny udviklingsbeslutning. Reglen gemmes i pilot/snapshot.

G4 genbruger H's kurvegeometri, men H's likelihood-fit og stokastiske accept
er selvstændige. Parametre/kurver overføres ikke ukritisk mellem støjfri og
stokastisk model. Den eksisterende obligatoriske F4-H-f/g-leverance bevares.

## 6. Beregningsbudget og stopregler

Tallene er **foreslåede loftværdier for første implementation**, ikke målte
køretider eller løfter om konvergens. Før en studiekørsel låses de i manifestet.
Det første nåede loft stopper ny beregning; en pause er ikke en bestået fit.

| Arbejde | Første loft | Hvad der tælles |
|---|---|---|
| G1/G2 runtimepilot | 30 residualkald og højst 60 sekunder | Repræsentativ fuld træningslængde; opvarmning/opsætning måles særskilt. |
| Ét G2-fit | 4 starter, højst 250 residualkald per start, højst 1.000 samlet, højst 5 minutter | Inklusive numerisk Jacobian; ikke kun optimizerens rapporterede nfev. |
| G4-kurveundersøgelse | Højst 3.000 residualkald og 10 minutter i alt | Alle familier, starter, strafstyrker og genstarter deler samme loft. |
| G6a-varianspilot og benchmark | Pilot: højst 60 sekunder/30 varianspropagationer. Derefter højst 500 propagationer og 5 minutter samlet for BM/WM-usikkerhedsfit, numerik og scoring | Højst to støjfamilier, to starter per familie; finite differences tæller med. Genbrug gemte deterministiske baner. Ekstra nødvendige strukturelle fits skal budgetteres inden kørsel. |
| G5-månedsbenchmark | Højst 20 minutter samlet som første driftsmål/loft | Alle deklarerede ugeorigins, BM-fit, fælles baseline og WM4-fit/forecast; delresultater gemmes. |

Et residualkald kan kræve flere baner, hvis data er opdelt i segmenter. Gem
derfor både residualkald, faktisk antal forward-solves, RHS-evalueringer og
wall-time. Numerikchecks og final evaluering har egne eksplicitte små kvoter
og tælles i det samlede studiebudget; et tidsstop må ikke bruge den sidste
kandidat som skjult færdig model. Hvis ingen start konvergerer, markeres fit
som uafsluttet, og baseline kan stadig rapporteres.

Begynd enkeltprocesseret med BLAS=1. Vectorisér inputs/residualer og genbrug
baseline, prisarrays, splits og de dele af numerikken, som er parameteruafhængige.
Cache ikke tilstandsbaner på tværs af forskellige parametre. Kun hvis målt
overhead tillader det, bruges op til 8 workers på uafhængige starter/ugeorigins
med BLAS=1 og fælles samlet budget; ingen nested parallelisering.

G0–G5's deterministiske fitloop bruger ikke Monte Carlo eller FV/mixtures.
G6b/G7/G8 bruger dem eksplicit på små forsøg. Undgå store Cartesian-grids,
tusindvis af starter, hel sæson før pilot og brede gentagelser af F-studier. Brug den eksakte vej
hvor kontrakten tillader det. Ellers genbrug ODE og forfin kun et lille
deklareret numerikpanel plus den valgte kandidat. Numeriske præcisionstolerancer
tilpasses ikke efter et ønsket vinderresultat. Hvis budgettet ikke rækker,
rapporteres stop, profilering og en konkret afgrænsning; ingen automatisk ekstra natkørsel.

Gem afsluttede starter, konfiguration og bedste gyldige kandidat atomisk.
Løbende forbedringer kan checkpointes med begrænset frekvens. Et fortsat
SciPy-fit fra gemt theta er en ny warm-start, ikke eksakt genoptagelse af
internt optimizer-state; tidligere forbrug skal stadig tælles. Vis grov ETA
ved længere jobs og bevar samme source under aktive kørsler.

## 7. Kodens placering, kontrol og output

Genbrug eksisterende matematik i `src/model/base/bm.py`, ODE i
`src/model/bm_1.py`, kontrakter i `src/tools/bm_1_data.py`, kurveobjekter i
`src/params/bm_1_response.py` og I/O i `src/tools/bm_1_reports.py`.
`src/params/bm_1.py` er allerede F4's observations/Bayes-primitiver og skal
ikke overskrives med G-fitteren. Den eksisterende punkt-observationscontainer
må heller ikke mærkes som intervalmodel; et intervalinterface kræver egen kontrakt.

G0–G3-kode er oprettet; G6a's variansmodul og G6b/G7/G8's adaptere er fortsat
planlagt. De eksisterende F-solvere/filtre/kontrolevaluatorer er genbrug, ikke ny kode:

- `src/params/bm_1_deterministic.py`: lille fitkonfiguration, fitresultat og
  public API til deterministisk estimation.
- `src/method/bm_1_deterministic_fit.py`: residualopbygning, observationsoperator,
  optimizerbudget og fit; en generaliseret eksakt overgang placeres som genbrugelig
  numerisk helper efter inspektion af eksisterende solverstruktur.
- `src/exe/run_model/bm_1/deterministic.py`: tynd indgang til audit, fit,
  predict-week og compare; måneds-/ugekørsel, fit-only og audit-only er nu kørbare.
- `src/validation/bm_1/deterministic_fit.py` og fokuserede tests: små syntetiske
  regressioner og cutoff-kontroller, ingen lange forskningsstudier i CI.
- G6 tilføjer en lille særskilt usikkerhedsmodul/indgang efter inspektion af
  WM's helpers; deterministiske API'er bevares. Resultater gemmes under
  `files/output/bm_1/runs/uncertainty/<run_id>/` med reference til strukturelt fit.
- `files/output/bm_1/runs/deterministic/<run_id>/`: egne resultater med
  `manifest.json`, `fit.json`, `predictions.csv`, `metrics.json` og ugefigurer.

G1 kontrollerer en lille eksakt/ODE-case, inputskift, Xi=0, rande og
timer↔dage. G2 kontrollerer et støjfrit identificerbart eksempel og en bevidst
uidentificerbar case; kræv god prediction frem for recovery, hvor identifikation
ikke findes. G3 tester, at ændrede fremtidige observationer ikke ændrer fit,
baseline, initialtilstand eller forecast, og at punkt/interval ikke forveksles.
Numerisk præcision kontrolleres på outputs/residualer; optimizerens små skridt
må ikke domineres af ODE-fejl. Ændringer i eksisterende interfaces kræver deres
relevante regressioner. Ingen lange benchmarkkørsler gentages for dokumentation alene.

Rapportér per uge og samlet: samme observed/B/WM/BM på fælles tidsstempler,
RMSE, MAE, bias=prediction−observation, MSE-gevinst over baseline og forbrugt tid.
Gem fejl og uafsluttede uger; vælg ikke kun de gode. Vis x for BM og F/R for
WM i egne paneler med deres forskellige betydninger. State-værdier er ikke
direkte sammenlignelige. Statistik fra én måned er beskrivende og dokumenterer
ikke generel sæsonperformance eller kausal priselasticitet.

## 8. Obligatorisk stokastik, kontrol og rekursiv estimation

### G6a: billig variansreference med BM's egen diffusion

Frys først de udviklingsvalgte strukturelle parametre og f/g-kurver. Brug
derefter den eksisterende BM-diffusionsform:

$$dX_t=a(X_t,t)\,dt+b(X_t,t;\eta)\,dW_t,\qquad
a(x,t)=\frac{D(x,t)-B(t)}{C},\quad b(x,t;\eta)=s(t;\eta)x(1-x).$$

Første støjfamilie har kun én parameter, s(t)=sigma≥0. Kun hvis udviklingsdata
begrunder det, prøves én ekstra familie med et fast døgnmønster:
s(t)=sigma_0+sigma_t[1+cos(2*pi*(t-phi)/24)]/2 med ikke-negative koefficienter.
phi fastlægges på træningsdata, ikke med en stor phasesøgning; t er her timer.
Ved dagsskala skal både frekvens og diffusionsenhed omregnes
(sigma_hour=sigma_day/sqrt(24)). WM's støjhypoteser WM-H0/WM-H1/WM-H2 kopieres ikke direkte ind i
BM: faktoren x(1−x) og BM's randstruktur bevares. Xi og C skalerer ikke
automatisk den eksisterende BM-diffusion.

Den billige første beregning er en lineær støjapproksimation langs den
deterministiske bane x_bar:

$$\dot{\bar x}=a(\bar x,t),\qquad
\dot P=2a_x(\bar x,t)P+b(\bar x,t;\eta)^2,\qquad P(t_0)=0.$$

Det kræver én ekstra skalarligning, ikke en Monte Carlo-population eller
Fokker–Planck-løsning. P0=0 betyder betinget kendt initialtilstand; parameter-,
baseline-, fremtidig input- og initialtilstandsusikkerhed er ikke inkluderet.
Dette er en eksplicit approksimation, ikke den eksakte varians for den
multiplikative SDE. Den deterministiske bane er heller ikke automatisk dens
middelværdi. Behandl responsknæk med segmentering og numeriske kontroller.

For punktmålt demand bruges delta-metoden:

$$V_D(t)\approx D_x(\bar x,t)^2P(t),\qquad
\widehat D(t)\pm1.96\sqrt{V_D(t)}.$$

Båndene er nominelle, approksimative 95 %-procesbånd. De klippes ikke skjult
til et fysisk interval. Ved stor spredning, randkontakt eller utilstrækkelig
kalibrering rapporteres begrænsningen; en dyrere metode startes ikke automatisk.
For intervalmiddel/energi propagéres også kovarians med det integrerede demand
i et lille udvidet system; gennemsnit af punktvarianser er ikke intervalvariansen.
Den deterministiske energiformel fra afsnit 2 må ikke bruges til stokastisk
demand uden støjintegralet i statebalancen.

Fit støjparametrene separat efter strukturelt fit, med samme afgrænsede
least-squares-idé som WM: match modelberegnet standardafvigelse mod historisk
timevis spredning fra de fem tidligere matchende ugedage. Brug samme
observationsoperator, datoer og cutoff for BM/WM. Det er en
**WM-kompatibel spredningskalibrering**, ikke likelihood-estimation af isoleret
processtøj: historisk spredning indeholder også variation i baseline, inputs og
målinger. Derfor er god tilpasning af spredningen alene ikke validering af SDE'en.
Ingen ekstra målevariansparameter i første pilot; en senere sådan udvidelse
kræver særskilt identifikations- og observationskontrakt.

Genbrug WM's kode til mål/figurer, hvor muligt: sigma-RMSE/MAE/bias som
diagnostik samt empirisk dækning, gennemsnitlig intervalbredde og intervalscore
på tidsmæssigt adskilte prognoser. Rapportér efter horisont og samlet med
samme nominelle niveau. Vis både alle tidspunkter og den fælles opgørelse uden
origin med P0=0, som WM gør. Vis punktfejl ved siden af usikkerhedsmålene.
WM/BM vurderes begge med initialisering og fit fra historik før origin;
aktuel WM-oracle-kode må ikke levere hovedsammenligningens prognosecenter.

G6 bruger en lille pilot med sigma=0-regression, variansens ikke-negativitet,
timer/dage-omregning og tidsforfining. Intervaldata kræver kontrol af den
udvidede kovariansberegning. Gem støjparametre, split, scope for usikkerheden,
approksimation, stopårsag og dækningstabeller separat fra deterministiske fits.
Hvis støjlaget ikke giver brugbare bånd inden for budgettet, afsluttes den
afgrænsede undersøgelse med dette negative resultat; der hævdes ikke kalibrerede
prognoseintervaller. G6 erstatter ikke F-sporets fulde stokastiske validering.


### G6b: Fokker–Planck, Beta-mixture og Euler–Maruyama

Brug samme a, b, baseline, prisforløb, initiallov og tidsenhed:

$$\partial_t p=-\partial_x(ap)+\tfrac12\partial_{xx}(b^2p),\qquad
p(t,x)\approx\sum_{m=0}^{M}c_m(t)\varphi_m(x),\quad c_m\geq0,\ \sum_m c_m=1.$$

Første panel bruger konstant sigma_x, som de eksisterende FP/mixture-/kontrol-
interfaces understøtter. Tidsvarierende støj kræver særskilt adapterparitet
på alle beregningsveje før sammenligning.
Genbrug BM's nul-flux FP/FV-reference, beta_mixture.py og F2/F3's dynamiske
mixture-evaluatorer. Beskriv den faktiske metode: FV-transport og
mixture-projektion, ikke en påstået gridfri koefficient-ODE. Kontrollér
masse, positivitet, projektionstab, middel/varians og demand-integraler.
Nye f/g får egen lille kontrol.

EM simulerer samme SDE:

$$X_{n+1}=X_n+a(X_n,t_n)\Delta t+
b(X_n,t_n)\sqrt{\Delta t}\,Z_n,\qquad Z_n\sim N(0,1).$$

Genbrug bm_1.euler_maruyama_solver, seedede increments og
coarsen_brownian_increments. Fin/grov EM kobles med summerede increments;
politikker sammenlignes med fælles tilfældige tal.
Kontinuerlig intervalinvarians garanterer ikke invariant rå EM. Registrér
rå randoverskridelser før enhver korrektion; reflektion/clipping må ikke
skjule numeriske fejl eller kaldes validering af den ukorrigerede SDE.
Genbrug F3's boundary='none'-kontrol hvor gyldig; stop/refinér det lille
panel ved overskridelser.

Begynd med én syntetisk 24-timerscase, derefter højst én historisk
udviklingscase. Match initialloven: en Beta-approksimation til en punktmasse
har endelig spredning og er ikke identisk med EM's præcise punktstart.
Opgør initialfejl særskilt. Sammenlign FP/FV, mixture og EM på state-/demand-
middel, varians, udvalgte CDF/kvantiler og trackingtab. Rapportér EM's
samplingusikkerhed sammen med tidsdiskretiseringsfejl.
Integrér det ikke-lineære D over fordelingen; D(E[X]) er ikke generelt E[D].
G6a's varianslag er reference, ikke erstatning for dette trin.

### G8: rekursiv estimation og handoff til kontrol

Genbrug run_fv_filter, run_mixture_filter og run_parameter_bank i
src/method/bm_1_estimation.py. Start med kendte parametre og tilstandsfilter;
fortsæt med rho=1/C og 5–9 hypoteser inden for
udviklingsfastlagte bounds. Den eksisterende én-parameterbank understøtter
rho eller sigma_x; k kræver en ny adapter og er ikke første pilot.
Øvrige parametre holdes eksplicit faste.
Det syntetiske design skal give tilstrækkelig prisvariation; svag identifikation
rapporteres frem for automatisk større grids.
Hver hypotese har sin egen betingede tilstandsfordeling.

For den syntetiske punktobservation y_n og statisk parameter theta:

$$p_n^-(x,\theta)=\int K_\theta(x\mid z,u)p_{n-1}(z,\theta)\,dz,\qquad
p_n(x,\theta)\propto L(y_n\mid x,\theta)p_n^-(x,\theta).$$

FP/mixture giver prædiktion; likelihood og normalisering giver opdateringen.
Ved manglende måling propagéres uden Bayes-opdatering. G's interpolerede input
må ikke blive observationsmål. Målemodel og sigma_y fastlægges særskilt;
dag-til-dag-spredning er ikke automatisk ren proces-/måleusikkerhed.
G0's punkt-/intervalkontrakt skal passe til likelihood-API'et; et nyt
intervalfilter kræver sin egen observationstilstand/operator. For timeenergi
må likelihooden ikke blot afhænge af sluttilstanden: brug en fælles
intervalkerne K_theta(x,y_n|z,u), så opdateringen er proportional med
integralet af K_theta(x,y_n|z,u) p_{n-1}(z,theta) over z. Den skal afledes og
kontrolleres for G's valgte observation; F4-H2 er mulig metodegenbrug.

Demonstrér mindst én komplet lille rekursiv parameterestimation med
mixture-backend og FV-reference. G2's batch-LS er initialisering/reference,
ikke dette krav. Genbrug F4-evidens med konkrete filer/commits, og lav kun
en kort ny integration for den valgte G-model.

Gem p(x,theta | y_1:n) og cutoff ved origin. Genbrug BM1ForecastOrigin og
control_problem hvor API'et passer. Det eksisterende rho-bank-handoff er
ikke automatisk generelt parameterbank-handoff; en lille adapter eller
hypotesevis evaluering kræves. Bevar parameter–state-afhængigheden i kontrol.
Et billigere MAP-valg bruger nodens egen conditional posterior og mærkes
som betinget approksimation. Gennemsnitsparametre må ikke kombineres med
en fremmed statefordeling.
Vis estimation→frosset origin→kontrol→ny observation på et kort syntetisk
forløb. En omfattende økonomisk closed-loop-undersøgelse er ikke ekstra gate.

### G7: obligatorisk stokastisk prisstyring med single shooting

Fastlås målprofil A, parametre/parameterposterior, initiallov, baseline og
prisdomæne før optimering. Begynd med ét døgn og 2–6 prisblokke:

$$\min_{0\leq u_j\leq1}J(u)
=\int_{t_0}^{T}\mathbb E[(D(X_t,u_t)-A(t))^2]\,dt
+J_{\mathrm{terminal}}+J_{\mathrm{prisglathed}}.$$

Øvrige led kan være nul; vægte og enheder deklareres.
E[(D-A)^2]=(E[D]-A)^2+Var(D): middelbanens fejl alene er et andet objective.
Ved parameterusikkerhed vægtes over hypotesernes egne conditional states.

Genbrug BM1ControlProblem, DynamicMixturePriceEvaluator,
FVStochasticPriceEvaluator/FastFVStochasticPriceEvaluator og multistart_powell
i src/control/bm_1.py. **Single shooting:** kun prisvektoren optimeres;
fordelingen fremadpropageres fra samme initiallov ved hvert objective-kald.
En deterministisk referencepolitik og en mixture/FP-optimeret stokastisk
politik indgår. Kontrollér inputinterpolation, kurve-API og enheder:
C_days=C_hours/24, sigma_days=sqrt(24)*sigma_hours.

Sammenlign neutral/fast referencepris, observeret pris hvis relevant,
deterministisk optimum og stokastisk optimum under samme stokastiske
evalueringsproblem. Genbrug finere FP/mixture og parret EM med
samplingusikkerhed for de få valgte politikker. Rapportér trackingtab,
priser, forventet demand, usikkerhed, randdiagnoser og runtime.
Lokalt optimizerstop er ikke bevis for global optimalitet.

A vælges ex ante med hensyn til modellens praktiske fleksibilitet.
Syntetisk kontrol kan dokumentere algoritmen, selv hvis historisk
prisrespons ikke er identificeret. Historisk fit alene dokumenterer ikke
kausal effekt af nye priser eller økonomisk profit.
Indkøb/profit er en mulig senere udvidelse; målprofilstyring er obligatorisk.
WM-kontrol sammenlignes kun med fælles objective, information, enheder og
constraints; ellers rapporteres resultaterne separat.

### Små budgetter og afslutningskriterier

Følgende er foreslåede loftværdier, ikke målte tider eller ændringer af F-gates.
Design og tolerancer fastlægges før kørsel.

- G6b: 60 sekunders runtimepilot; først 256 EM-baner, højst 2.048 parrede
  baner på to tidsniveauer. Højst to FP-grids og to mixture-niveauer på de
  højst to cases; samlet første loft 10 minutter.
- G8: 24 timer, 5–9 parameterhypoteser og én parameter; højst 10 minutter
  inklusive FV/mixture-sammenligning og kort handoff-kontrol.
- G7: højst to starter per politikfit, 100 objective-kald og fem minutters
  optimering samlet for begge politikfits; højst fem yderligere minutter
  til fælles FP/mixture-/EM-krydsevaluering. Alle forward-solves tælles.
- Samlet G6b/G8/G7-panel: højst 30 minutter plus højst tre 60-sekunders
  runtimepiloter. G6a beholder sit særskilte femminuttersloft.
  BLAS=1; op til 8 workers kun ved målte uafhængige opgaver.

Genbrug råresultater og checkpoints. Numerisk fejl, EM-standardfejl og
faktisk runtime afgør, om piloten er tilstrækkelig.
Budgetstop er uafsluttet/negativt forsøg, ikke automatisk opfyldt bachelorkrav.
Afgræns næste forsøg efter evidensen; ingen automatisk eskalering til store
parameterbanker, hele sæsoner eller natkørsler.
Skeln i slutrapporten mellem implementeret, syntetisk demonstreret,
numerisk kontrolleret og historisk valideret.

## Beslutning 5. oktober: genbrug WM's ugeprognose med fælles langt fit

Johan ønsker WM's enkle ugeprognoseprincip genbrugt i G og et fælles
fit over et helt år, hvis datadækningen tillader det. Implementér kun
yderligere dele, når de er nødvendige i den aktuelle G-etape.
Dette ændrer den planlagte hovedvariants træningsvindue; det gamle
januarforsøg og WM's oprindelige én-fituges-variant bevares som særskilte
udviklingsreferencer. G4's lille januarbudget udvides ikke af denne beslutning.

### Det konkrete WM-princip og genbrug

Reviewet i main's src/exe/run_model/wm_4/predict_weekly.py:
1. statistical_output.fit(frame, fit_end) fitter O på tilgængelig historik
   før cutoff; O's historik er ikke begrænset til dynamikkens ene fituge.
2. qp_params.fit fitter WM-dynamikken på én uge med 24 timers warm-up.
3. F/R fra den simulerede fitbanes præcise slutpunkt bruges som prognosestart.
4. O og r fryses, og deterministic_solver simulerer næste 168 timer.
   Næste uges demand indgår ikke i selve forward-ligningerne.
5. Observeret demand anvendes bagefter til fejlmål og figurer.

WM's deterministiske ligninger er dF/dt=R, dR/dt=r_F F+r_U U+r_R R,
I=O+R, med tid i dage. G genbruger princippet, baselineimplementationen,
cutoff/historik, metrics og rapportering, mens BM beholder egne ligninger,
state x, timer, normalisering og LS-estimator. Overtag ikke WM's F/R-state
eller QP-regression som BM-state/estimator.

G3's snapshot/forecast og G4's response-snapshot implementerer allerede
frysning og videreføring af egen cutoff-state. Ingen ny forecast-solver
eller kopi af WM-runneren er nødvendig for at genbruge princippet.
Baseline B=O skal være samme frosne objekt/forløb for matched BM/WM.

### Informationsgrænsen i ugeprognosen

WM's prediction_setup læser historiske fremtidige U/Temp og kræver også
I til sin combined setup/scoring, selv om I ikke indgår i forward-solve.
Det er en betinget historisk prognose, ikke en estimator af fremtidig pris
eller temperatur. G beholder sin allerede implementerede adskillelse af
forecast-input og optional scoring; forecast skal kunne gemmes uden fremtidig y.

Ved origin t0 beregnes Bhat(t|t0)=b_beta(kalender(t),That(t|t0)),
og BM propagerer sin gemte x(t0) med frosne theta og uhat(t|t0) i 168 timer.
For en virkelig ugeprognose bruges kun origin-tilgængelig prisplan,
prisprognose/scenarie og temperaturprognose. Registrér inputtype og
tilgængelighedstid; observerede fremtidige inputs mærkes conditional.
Der indføres endnu ingen ny pris-/vejrestimator. En dagligt opdateret
prognose er et særskilt forsøg, ikke en frosset ugeprognose.

WM's dagsvariant --optimize-prediction-state fitter initialstate med
målte observationer fra selve måldagen. Den er oracle-diagnostik og
overføres ikke til G's egentlige prognose.

### Hvornår yderligere implementation bliver nødvendig

- Nu: færdiggør den afgrænsede G4-pilot med fælles C/Xi/g-midterværdi,
  lineær f og fast k; genbrug eksisterende gemt-fit-forecast.
- Før langt fit: kontrollér fast population, målebetydning, pris/temperatur/
  demanddækning og fastlæg TRAIN, DEVELOPMENT og senere TEST.
  Januararkivets audit angiver demand til 2. november 2022; et fuldt år
  er endnu ikke dokumenteret. Hold skalaerne TRAIN-only.
- Lang variant: udvid først dato-/arkivadapter og nødvendige runtimeindstillinger,
  hvis eksisterende entrypoints ikke kan udtrykke vinduet. Fit baseline og
  lineær BM-reference på samme erklærede lange historik; brug reference-k
  fast i det efterfølgende fælles kurvefit. Ingen automatisk arvet januar-k.
  Ét fælles kurvepar og én kontinuerlig statebane, ingen daglige resets.
- Ugetest: træningsåret skal ligge før testugerne. Hvis kun ét års data findes,
  bruges et kortere forudgående træningsvindue med senere uger holdt ude;
  fit på hele året er ellers kun in-sample analyse.
  BM/WM sammenlignes med samme lange træningsvindue i den matched variant.
  WM's oprindelige korte dynamikfit rapporteres separat.
- Senere: rullende vindue, kausal state-opdatering og G6/G8/G7 genbruger
  den samme origin/handoff-idé. Årsfit og bred prognosebenchmark bliver
  ikke en ny gate, der blokerer G6b→G8→G7.

Dette er en dokumenteret beslutning og kodereview, ikke et udført årsfit
eller ny prognosevalidering. Ingen numeriske tests/kørsler kræves for
denne rene planændring.

## 9. Kilder og første konkrete opgave

Planen er udarbejdet mod main `fe53cce858979afabeaa7ab1e36b9e44874b0fc5`.
Ved implementation læses nyeste AGENTS/G og berørt WM-kode igen; F læses
kun ved konkret genbrug eller ændring af fælles interfaces. Bevar
andres samtidige arbejde og brug branch i eksisterende checkout, ikke nye worktrees.

- [BM-reference](BM_REFERENCE.md): ligninger, eksakt F1-løsning, feasibility og enheder.
- [BM-guide](BM_GUIDE.md): genbrugelige interfaces og eksisterende outputs.
- `src/model/wm_4.py` og `src/exe/run_model/wm_4/predict_weekly.py` på ovenstående
  commit: signed WM4 og deterministisk strukturelt fit/forecast.
- Johan's `WM4_notes(1).zip`, `main.tex`, dateret 27. september: deterministisk
  fit og oracle-initialisering; ældre fire-koefficient/[0,1]-parameterisering.
  Den skal ikke blandes med den nyere tre-koefficient/signed kode.
- Nocedal & Wright, Numerical Optimization, 2. udgave, kapitel 10: least-squares-
  struktur og lokale metoder; ingen påstand om global konveksitet af BM-fit.
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html):
  bounds/TRF, numerisk Jacobian og separate tællinger af derivativevalueringer.
  Kontrollér installeret SciPy før eventuel brug af nyere callback/workers-API.

**Næste opgave:** Den afgrænsede G4-januarpilot og februarreview er gennemført;
lineær BM er valgt efter den forudfastlagte regel. Genbrug originalt G3-fit
uden refit. Undersøg datadækning/split og nødvendig adapter til fælles langt
fit, eventuelt helår, med senere testuger. G6b→G8→G7 kan fortsætte fra det
frosne lineære valg; årsfit er ingen ny gate. Se G4-reviewet for råresultater.
Review de gemte januar-/februarartefakter, når de er tilgængelige; ingen
genkørsel af det oprindelige G2-fit. En pris uden for frossent TRAIN-domæne
afvises uden refit.
G3-snapshot understøtter endnu kun legacy-hourly-point; en fælles BM/WM-
timeintervalvariant kræver en særskilt deklareret kontrakt og egne fits.
P0 beskriver åbne datafelter; P3–P6 har adapterkort, syntetisk forsøgsudkast,
afhængigheder, budgetter og acceptkontroller for G4/G6b/G8/G7.
G0–G3 fungerer i den rapporterede lokale udviklingskæde. G4's kodekontrol
giver ikke historisk modelaccept. Bred måneds-/sæsonoptimering tilføjes ikke
automatisk; historisk råreview og G6–G8's nye integration/accept mangler.
G's senere posterior-handoff og stokastiske kontrolobjective får et afgrænset
matematisk review; se G-overdragelsen. F4-H2's Astra XHigh-opgave tilhører
F-sporet og er ikke næste G-opgave.

### Januar: håndtering af huller, 5. oktober 2026

G-januar håndterer nu manglende demand/temperatur med den eksisterende import-
interpolation afgrænset til TRAIN. Rå demand beholdes som NaN; baseline/BM-fit
og fælles BM/WM-fejlmål bruger kun målte timer. WM-tilstandsrekonstruktion bruger
udfyldt demand; QP-overgange over manglende endpoints maskeres. G kræver stadig
mindst 168 målte demand-timer i TRAIN og komplette importerede priser. 10 tests består;
én syntetisk januar med 27 demandhuller og 3 temperaturhuller gennemført
på ca. 19 s (717 målte timer for begge modeller). Historisk fit er rapporteret gennemført lokalt; råreview mangler.

Udfyldning bruger `import_input.interpolate_time_gaps` på aggregeret demand og
temperatur inden for træningsvinduet; februar bruges aldrig til januarhuller.
CSV og plot markerer udfyldte værdier. Døgn uden målinger får samples=0 og
udelades fra performance-sammenfatninger. Priser udfyldes ikke yderligere i G.
Dette er en eksplicit G-adapter; WM4's selvstændige fit-entrypoints ændres ikke.


