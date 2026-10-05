# G3 og frem — grundlag og overdragelse til Sol

Forberedt 5. oktober 2026 mod `main` ved
`d9f95a0513605d58f0fe01464067e8f0a2a0eeeb` efter PR #69/#70.
**Implementation 5. oktober:** P1/P2 er nu implementeret og syntetisk
kontrolleret: `deterministic --from-fit`, versioneret JSON-snapshot og
kontrolleret 07fad9f-import. Fælles BM/WM/origin/baseline og varig eksport før
optional scoring er verificeret. 20 G-tests plus tre relevante layout/import-
kontroller består på ca. 6.7 s. Historisk råreview/prognose og G4–G8 mangler.
Beskrivelsen nedenfor bevarer designet; denne status erstatter ældre formuleringer
om manglende P1/P2. Kommando: [G_JANUARY.md](G_JANUARY.md).
Læs nyeste `AGENTS.md`, [G.md](G.md), [G_JANUARY.md](G_JANUARY.md) og den
aktive status i [F4.md](F4.md) ved genoptagelse. Bevar samtidige H/WM-ændringer.

## Opgaven og det allerede kendte

Bachelorens projekt 111866 handler om at optimere prissignaler, så en gruppes
forbrug følger en ønsket profil, med en stokastisk demand-response-model.
FP-approksimation med mixtures, stokastisk single shooting og rekursiv
tilstands- **og** parameterestimation er centrale leverancer. EM betyder her
Euler–Maruyama og bruges til simulation/krydskontrol. Et godt deterministisk
fit eller forecast afslutter ikke bachelorens metodekrav.

Johan ønsker nu forberedelsen færdig, så Sol kan implementere senere. Start
med G3's genbrug af gemt fit, ikke en ny januaroptimering eller et stort studie.
Den aktive historiske variant er **hele januar 2022 → næste 168 timer**, MED,
neutral start, nul warm-up; den ældre uge/24-timers-warm-up-protokol er en
anden deklareret variant. G4's lille forsøg ændrer ikke F4-H's obligatoriske
fælles f/g-leverance eller dens acceptkrav.

| Kendt evidens | Status og begrænsning |
|---|---|
| Januarfit på `07fad9f` | Johan har rapporteret `status=complete`, 624 målte timer af 744; 120 demandhuller. |
| BM-parametre | C=1.63492382 timer, k=0.22688551, Xi=0.23120648; x0=0.5 fast. Tallene er afrundet terminaloutput, ikke input til genstart. |
| Normaliseret MSE | BM 0.0066056846, WM 0.0072743310, baseline O 0.0066347027. In-sample; BM's gevinst over O er ca. 0.44 %. |
| Råartefakter | Run `g_january_fit_20261005_154824` er ikke modtaget/råreviewet her. `fit.png` alene er utilstrækkelig. |
| G3 | Selvstændig forecast fra gemt fit er implementeret og syntetisk kontrolleret. Historisk prognose/råreview mangler. |
| G4/G6/G8/G7 | Ny G-integration er planlagt. Eksisterende F2/F3/F4-kode/evidens genbruges med konkrete kildehenvisninger. |

Arbejd i branch i eksisterende checkout, ikke nyt worktree. Skift/pull ikke
under aktive kørsler. BLAS=1; ingen ny historisk fitrunde alene for dokumentation.
En nødvendig ændring af den videnskabelige datakontrakt kan kræve et nyt,
særskilt fit; det gamle resultat må hverken overskrives eller ommærkes.

## Fund der skal håndteres før historisk G3-accept

1. Den oprindelige fit-mode kalder `run_benchmark`. Ny `--from-fit` indlæser
   fitmappen uden baseline/BM/WM-fit og foretager kun forward propagation.
   `checkpoint.json` er kandidat/status, ikke en genoptagelig SciPy-session.
2. Forecast-inputs bruger kun pris/temperatur. G3 gemmer nu prognoser før
   observationslæsning/scoring; manglende demand giver særskilt scoringstatus.
3. `consumption_series` i `wm_3/fit_weekly.py` bruger `mean(axis=1)` med
   pandas' skip-NaN-adfærd. En delvist observeret time kan få en anden
   population. 624 aggregerede måletimer beviser ikke 624 komplette MED-timer.
4. H2's metadataaudit har verificeret 13 MED-huse med elvarme og 140 UDEN-huse
   uden elvarme. Johan har senere deklareret timeforbrug, foreløbigt kWh,
   hour-start og foreløbig Copenhagen-tid; det er arbejdsantagelser, ikke
   uafhængig metadata. H2's interval-likelihood er implementeret men endnu
   ikke numerisk fit-ready. Se [H2_HOURLY_INTERVALS.md](H2_HOURLY_INTERVALS.md).
5. Prisnormalisering er frosset på TRAIN. Februarpriser uden for domænet
   giver korrekt afvisning; de må ikke klippes eller bruges til efterfølgende
   ændring af januarfittets skala. Faktiske DK1-spotpriser er en deklareret
   proxy, ikke dokumentation for husstandenes faktiske tariff/kausale respons.
6. Nye snapshots gemmer alle baselineindstillinger og BM/WM-cutoff-states.
   Legacy-import genskaber dette fra verificeret source og finegrid-CSV.

Historisk G3 kan først mærkes valideret, når den valgte datakontrakt er
dokumenteret. En videreført legacy-punktapproksimation kan fortsat vises som
udviklingsdemo med sin oprindelige populations-/observationskontrakt. Den
afklarer ikke H2. Syntetisk adapterarbejde behøver ikke vente på råmappen.

## P0 — datakontrakt, fitarkiv og inputaudit

**Filer:** `src/tools/bm_1_deterministic_data.py`, den nye G-snapshot-adapter
beskrevet under P1, G's manifest; genbrug H2's dokumentation og læsegrænser.
Ret ikke WM's selvstændige entrypoints som bivirkning.

- Arkivér original `manifest.json`, `fit.json`, `result.json`, audit/checkpoint
  og `fit_signals_normalized.csv` samt hourly-CSV'er uændret, med SHA256.
  Læs fuld præcision fra filerne; brug ikke terminalens afrundede parametre.
- Auditér de valgte MED-kolonner og antal tilgængelige medlemmer per time i
  den deklarerede TRAIN. Hold sluttest afskærmet. For ny fast-populations-
  kontrakt er aggregatet NaN, hvis et medlem mangler. Gem medlemsliste,
  antal, råmaske og kontraktversion. Legacy-skip-NaN forbliver særskilt.
- Dokumentér måleenhed, observationstype, timelabel og tidszone med kilde.
  Hourly middel/energi kræver korrekt operator for både BM og matched WM,
  ikke blot en ny etiket. Deterministiske intervalintegraler og stokastisk
  interval-likelihood er forskellige implementeringsopgaver; sidstnævnte
  kræver særskilt design/validering før historisk G8.
- Frys træningshistorik, skala, observation/operator, baseline og prisproxy.
  Registrér eksponering: januar og allerede sete februarperioder er udvikling.
  Uåbnet sluttest skal udpeges og protokollen låses før åbning.
- Auditér G3's 168 timers pris-/temperaturinputs og nødvendige endepunkter
  uden fit eller brug af fremtidig demand. Hold sidste input ved slutkanten
  som den eksisterende adapter, medmindre en ny version deklarerer andet.
  Pris uden for frossent domæne stopper denne prognose; ingen skjult refit.

**Leverance:** kort data-/arkivrapport med konkrete åbne felter. Ingen
opdigtede metadata, ingen ubegrundet `historical_fit_ready=True`.

## P1 — gemt fit → frosset forecast-origin

**Implementeret fil:** `src/tools/bm_1_deterministic_snapshot.py`.
**Berørte eksisterende filer:** `src/method/bm_1_deterministic_benchmark.py`,
`src/exe/run_model/bm_1/deterministic.py`, G's validering og små tests.
Designets schema er implementeret som validerede JSON-data med almindelige
helpers; der er ikke indført et ekstra solver-/klasseframework.

Det versionerede snapshot har følgende kontrakt:

| Feltgruppe | Skal gemmes og valideres |
|---|---|
| Identitet | Schema-version, original run-ID/commit, hashes for afhængige numeriske kilder og input/artefakter, softwareversioner, migrationsproveniens. |
| Data/information | Population, observationstype/enhed, timezone, TRAIN-start/slut, baselinehistorik, cutoff, prisproxy, alle frosne skalaer og manglende-data-politik. |
| Baseline | coef, feature_mean/scale, metadata samt base_temperature, hour_harmonics, season_harmonics, ridge, huber_delta, max_iter; features/version, clipping og interpolation. |
| BM | Uafrundede C_hours,k,Xi, kurver, initial-state-mode, valgt trænings-x0, eksakt simuleret x ved cutoff. |
| WM | Uafrundet r, fitkonvention, modelversion og simuleret F,R ved cutoff. Ingen reset til INIT_F eller I−O ved forecast-start. |
| Numerik | Grid/trin, tolerance og faktisk kontrolresultat, tids-/priskonvention og endepunktsregel. |

Implementerede operationer: `export_fit_snapshot(result)`,
`load_fit_snapshot(path)` og eksplicit `migrate_legacy_fit(directory)`.
Snapshot skal kunne indlæses uden optimizer, workbook eller netværkskald.
Genskab WM-baselineobjektet med de gemte indstillinger og koefficienter uden
`.fit()`. JSON må ikke indeholde pickle/eksekverbar kode.

Legacy ved `07fad9f`: `result.json` har `x_forecast_origin`, `wm_fit.r` og
numerik; `manifest.json` har baselinevektorer og skalaer; finegrid-CSV'en har
F/R på cutoff. Kræv nøjagtigt det forventede sluttimestamp, finitte states,
overensstemmelse mellem gemt BM-origin og finegrid-endpoint og bestået numerik.
Sidste hourly-fitpunkt er **ikke** forecast-origin. Manglende baselineindstillinger
må kun rekonstrueres fra den verificerede originale implementation på denne
kendte sourceversion og kontrolleres mod gemt baseline. Ingen aktuelle defaults
som tavs erstatning. Migreringen skriver ny fil i ny mappe og bevarer originalen.

Afvis ukendt/incompatible schema, manglende states eller uverificerbar migration
med præcis årsag. Et nyt docs-commit gør ikke i sig selv numerikken inkompatibel;
kontrollér relevante afhængigheder og registrér kompatibilitetsbeslutningen.
Et manglende felt er ikke tilladelse til automatisk re-fit.

**Accept:** lille syntetisk fit eksport/import giver samme baseline, BM/WM-origin
og forecast inden for fastlagt round-trip-tolerance; ingen fitfunktion kaldes
under import/forecast. Korrupte skalaer, parameterrækkefølge, endpoint og
inkompatibel baseline afvises. Lås kildedata mod utilsigtet overskrivning.

## P2 — G3-prognose, varig eksport og særskilt scoring

**Implementeret:** `forecast_from_snapshot(snapshot, raw_inputs)` i
benchmarkmodulet får kun frosset model og inputs. `deterministic --from-fit`
vælger denne vej og afviser eksplicitte fitindstillinger. Den faktiske
kommando og output står i januarvejledningen.

- BM starter med gemt x ved cutoff. WM starter med gemt F/R. Samme fælles
  O/B-samples, held-price-forløb og enhedsomregning som den oprindelige kæde.
- Skriv predictions, states, inputaudit, provenance og numerisk status atomisk
  før optional scoring. Et scoringproblem må ikke slette en gyldig prognose.
- `score_forecast(predictions, observations)` bruger fælles målte timer for
  BM/WM/O. Ingen udfyldt demand tæller som måling. Gem expected/observed/scored
  samples, coverage, maske og årsag til eksklusion. Døgn uden målinger får
  samples=0 og NaN-metrics; nul observationer giver `not-scored`, ikke fiasko
  i selve simulationen. Ufuldstændig coverage mærkes tydeligt og er ikke en
  fuldt observeret uge. Ingen ny arbitrær coverage-gate skjult i kode.
- Genbrug WM's MSE/RMSE/MAE/bias og performanceopgørelser, og hold pooled score
  adskilt fra gennemsnitlige dagsscorer. Kræv samme maske for sammenligning;
  hvis WM mangler, rapporteres BM særskilt uden matched-accept.
- Manifest adskiller `fit_status`, `forecast_status`, `numerics_status`,
  `scoring_status` og `evidence_scope`. En færdig prognose er ikke automatisk
  historisk/videnskabelig accept. `status=complete` fra et gammelt fit bevares
  i originalen og fortolkes i sin oprindelige betydning.

**Små, men meningsfulde kontroller:** frosset replay mod eksisterende syntetisk
fit+forecast; ændrede/fjernede fremtidige y ændrer kun scoring; delvis/ingen
demand bevarer forecast; startstate kommer fra eksakt cutoff; inputdomain-fejl
giver tydelig afvisning uden optimizerkald. Bevar G's 1e-5 demand-refinement-
tolerance. Ingen ny hel-måneds-smoke alene for denne refaktorering.

Historisk næste handling efter P0–P2: importer den originale januarmappe,
auditér de frosne inputs og kør kun 168-timers-forwarddelen. G3 kan godt give
negativ baselinegevinst; det udløser ikke automatisk nye fits.

## P3 — lille G4 og ét fælles modelinterface

**Genbrug:** `src/params/bm_1_response.py`, `src/method/bm_1_response.py` og
`BM1Parameters` i `src/control/bm_1.py`. H's likelihood-fit er ikke G's LS-fit.
**Adapterarbejde:** `src/method/bm_1_deterministic_fit.py` har en lineær,
specialiseret kerne. Tilføj en afgrænset kurvevej; lineær default/paritet bevares.

1. Et frosset G-modelspec gemmer C_hours,k,Xi, f/g-knuder/værdier, skala,
   observationstype og tidsenhed. Konstant sigma_x tilføjes i G6; sigma_y er
   separat. Callables dannes fra validerede data, ikke fra serialiseret kode.
2. Første pilot: to-stykkers g, lineær f, k fast på G2's værdi, C/Xi og g's
   indre knude estimeres samlet over hele TRAIN. Bevar samme x0-policy som
   G2: wm-neutral holder x0=0.5; kun `initial-state fit` estimerer x0.
3. Endpoints/monotonicitet valideres, ingen sorting/clipping i objective.
   Udvidelse til f/fire stykker kræver den deklarerede udviklingsbeslutning.
   Lineær løsning er et legitimt negativt pilotresultat, ikke H-f/g-accept.
4. Krydstjek drift og demand på indre/boundary-state-, pris- og baselinepunkter
   mellem G-kerne og `bm_1` med samme kurver. Kontrollér retning, signs og
   tidsenheder før nye stokastiske studier. Undgå et ekstra solverframework.

Budget fra G-planen: samlet højst 3.000 residualkald/10 minutter for alle
varianter, strafstyrker og starter. Kodeadapters syntetiske paritet kommer først.
Gem datascore og regulariseret objective separat; DEVELOPMENT vælger model,
TRAIN alene gør det ikke. Frosset kurvevalg overføres til G6/G8/G7.

## P4 — G6b: lille FP/mixture/EM-panel

**Genbrug:** `src/model/bm_1.py`, `src/method/fokker_planc.py`,
`src/method/beta_mixture.py`, `src/method/euler_maruyama.py` og F3's
fælles Monte Carlo-evaluering i `src/control/bm_1.py`.
Foreslået G-studierunner er en tynd adapter, ikke en ny FP- eller EM-solver.
G6a's variansreference er separat; WM4's varianskernel er ikke fuld FP/EM.

Følgende er et **registreringsudkast for en syntetisk udviklingscase**, ikke
en kørt undersøgelse eller fysiske estimater. Lås et faktisk manifest før
generering/evaluering; alle ændringer efterfølgende får et nyt forsøgs-ID.

| Felt | Første afgrænsede forslag |
|---|---|
| Tid | Timer, horizon [0,24], hourly rapportering, alle prisbreakpoints eksplicit. |
| Model | Lineær f/g-reference; C=4 timer, k=2, Xi=0.4; konstant sigma_x=0.2 per sqrt(time). Den G4-valgte kurve får senere sin egen paritetscase. |
| Inputs | B(t)=0.5+0.1 sin(2 pi t/24); u=[0.3,0.7,0.3,0.7] på [0,6,12,18,24]. Ingen historiske targets bruges. |
| Initiallov | Beta(20,20) for FV og EM; mixture-init fra samme lov med særskilt projektionstab. Ikke punktstart forklædt som samme prior. |
| EM | Seedede initialstates/increments; start 256, loft 2048 baner; fin/grov koblet via summerede increments. Fast seedliste registreres før kørsel. |
| Numerik | To FV-grids og to mixturegrader; genbrug F2/F3's definitioner af celler/grader. Vælg eksplicitte små niveauer i 60-sekunders pilot og lås dem før sammenligningen. To tidsniveauer deles af de relevante veje. |
| Output | Masse/positivitet, initial-/løbende projektionstab, E[X], Var(X), E[D], Var(D), udvalgte CDF/kvantiler, trackingtab, MC-standardfejl, randoverskridelser, runtime. |

Bevar eksisterende sandsynligheds-/randguards. Fastlæg moment-, CDF- og
objective-tolerancer i samme manifest før hovedpanelet og i forhold til den
praktiske størrelse, ikke efter den observerede metodeforskel. Runtimepiloten
er ikke videnskabelig accept. Hvis to niveauer eller MC-budgettet ikke kan
opløse forskellen, rapportér `inconclusive`; udvid ikke automatisk studiet.
Ingen claim om generel intervalkalibrering fra ét døgn/en sand bane.

Timer er G's kanoniske tid. F-kode omregner ikke automatisk: alle inputtider,
breakpoints, drift/diffusion og objective-integraler skal bruge samme enhed.
Ved dag-adapter: C_days=C_hours/24 og sigma_days=sqrt(24)*sigma_hours;
integreret trackingtab omregnes også (J_hours=24*J_days for uændret integrand).
Likelihood-sigma_y omregnes ikke som procesdiffusion. Dokumentér enhedstesten.

EM bruger samme SDE og initiallov; rå randoverskridelser registreres før
korrektion. Brug `boundary='none'` og stop ved ugyldige baner frem for at
skjule dem med clipping. Mix-metoden er FV-transport + Beta-projektion, ikke
gridfri koefficient-ODE. Genbrug ikke FV-posterior som skjult restart ved
hvert mixture-tidsskridt. Ikke-lineær demand skal integreres over loven.

Budget: 60-sekunders pilot, derefter højst 10 minutter inkl. valgt forfining,
højst to cases. Syntetisk først; historisk case kun med passende datakontrakt.
G6a's separate 5-minuttersloft ændres ikke.

## P5 — G8: rekursiv state/rho-bank og ærligt posterior-handoff

**Genbrug:** `run_fv_filter`, `run_mixture_filter`, `run_parameter_bank` i
`src/method/bm_1_estimation.py` samt `src/control/bm_1_handoff.py`.
Start med G6b's syntetiske model, kendte parametre og eksplicit punktobservation
D(X,t)+målefejl. Hold sigma_y adskilt fra sigma_x; f.eks. 0.03 i normaliserede
demandenheder i det låste syntetiske manifest, ikke estimeret fra samme testbane.

- Først kendt-parameter-statefilter, dernæst rho=1/C. Forslag til fem noder:
  [0.125,0.1875,0.25,0.375,0.5] per time med uniform prior; sand rho=0.25
  er en kendt syntetisk indre node, ikke et historisk fund. Øvrige parametre
  holdes faste. Simuler observationer med fin EM og separat registreret seed.
- Samme observationssekvens og manglende-data-maske gives til FV/mixture.
  Gem prædiktiv score **før** opdatering, posterior **efter**, log-evidence,
  bankvægte og hver nodes egen conditional state. NaN giver propagation
  uden likelihoodopdatering. Ingen interpoleret observation eller dagligt reset.
- Vis en lille split/genoptagelse med identisk slutposterior; gem mixture-
  koefficienter, ikke kun marginal masse. Rapporter svag identifikation og
  randvægte; én case dokumenterer integration, ikke generel parameterrecovery.
- Det generiske `run_parameter_bank` returnerer en dict. Den eksisterende
  `BM1ForecastOrigin.from_rho_bank_step` kræver `BM1FVRhoBankStep` og vælger
  **MAP**, ikke hele parameterposterioren. Direkte indpassing er ikke nok.
  Skriv en lille valideret G-adapter til frozen origin med noder, vægte,
  conditional masses/coefficients, kurver, parametre, grid og cutoff.
- G7's fulde bankobjective er summen af vægtede nodeobjectives fra disse
  egne states. MAP er en særskilt deklareret billig approksimation med den
  valgte nodes state, ikke gennemsnitsparametre + marginal state. Implementér
  og test 1-node-reduktionen og et 2-node-eksempel før femnodedemonstrationen.

Undgå dobbeltbrug af data: syntetisk sekventiel kontrol starter fra en
deklareret prior og assimilerer hver måling én gang. Hvis historisk G2-fit
bruges til faste parametre/prior, skal dette være en markeret empirisk-Bayes-
approksimation eller bygge på et tidligere adskilt vindue; kald ikke replay
af samme januar en uafhængig posteriorvalidering.

Budget højst 24 timer/5–9 hypoteser/10 minutters samlet beregning inklusive
FV/mixture/handoff-kontroller. Bankens k-akse findes ikke i standard-API'et og
er ikke del af denne første opgave. Historisk intervalfilter er ikke dækket
af den syntetiske punktobservationsdemo.

## P6 — G7: single shooting med samme stokastiske objective

**Genbrug:** `BM1ControlProblem`, deterministisk/FV/dynamisk-mixture-evaluator,
`multistart_powell`, `evaluate_mc_candidates_common_random_numbers` og
`paired_mc_improvement` i `src/control/bm_1.py`.

Første controller bruger 4 prisblokke over 24 timer, bounds u i [0,1], et
fast ex-ante mål A(t) og nul terminal-/glathedsstraffe. Et lille syntetisk
forslag er A(t)=B(t)+0.03 sin(2 pi t/24); frys det før politikoptimering og
rapportér eventuel utilstrækkelig fleksibilitet uden efterfølgende at tilpasse A.
Prisvektoren er optimizerens eneste beslutning; hver evaluering starter fra
samme frosne origin og propagérer fordelingen. Ingen test-demand i objective.

Den deterministiske reference bruger deklareret state fra samme information
(f.eks. posteriorens middel, udtrykkeligt mærket plug-in). Den stokastiske
mixture-politik optimerer E[(D−A)^2], inklusive Var(D). Ved bankhandoff bruges
nodevægte og nodeconditional states; vægtene holdes faste gennem det åbne
kontrolhorizon. Dette er en åben single-shooting-policy, ikke dual control.

Neutral pris, deterministisk policy og stokastisk policy evalueres alle på
samme stokastiske problem med finere FV/mixture og parret EM. Ved bankposterior
samples også parameter-node og dens conditional initialstate konsistent;
samme node/state/increments deles mellem politikkerne. F3's standard-Beta-
initialsampler er ikke automatisk gyldig for en filtreret bankposterior.

Efter G8 vises et kort faktisk loop: observation → posterior → prisvalg →
simuleret anvendt pris → ny observation → ny opdatering. Handoff alene er
ikke et lukket loop. Afgræns antal kontroloptimeringer på forhånd, og tæl dem
i samme budget; en lille demo må ikke snige et helt MPC-studie ind.

**Accept:** fælles objective/enheder/cutoff, ikke-negativ varians, korrekte
prisconstraints, mean/variance-dekomposition, 1-node-bankparitet, ingen
posterior-mutation mellem objective-kald, numeriske/MC-fejl rapporteret.
En stokastisk policy behøver ikke vinde statistisk i én case. Lokalt
optimizerstop er ikke bevis for global optimalitet eller kausal priselasticitet.

Budget fra G-planen: højst to starter per politikfit; **samlet** 100 objective-
kald/5 minutter til de to fits og højst 5 minutter til krydskontrol. Ved
bankobjective tæller ét optimizerkald én kandidat, men alle node-forward-solves
registreres særskilt og er omfattet af væguret. Genbrug cached identiske kald.
G6b/G8/G7 har samlet 30 minutter plus tre 60-sekunders piloter; stop er
uafsluttet/negativt resultat, ikke mulighed for at løsne acceptkrav.

## Leverancer, testscope og næste sessions start

| Pakke | Færdig når | Afhænger af |
|---|---|---|
| P0 | Råarkiv/databetydning og mangler er eksplicit auditeret. | Januarfiler til historisk del; kodeaudit kan laves nu. |
| P1 | Versioneret snapshot og kontrolleret legacy-import uden fit. | Eksisterende fitformat; historisk import afventer råfiler. |
| P2 / G3 | Input-only-forecast, gemning før scoring, matched output og små kontroller. | P1; historisk accept også P0. |
| P3 / G4 | Fælles kurveinterface, paritet og afgrænset udviklingsvalg. | G3-kæden fungerer; budget er målt. |
| P4 / G6b | FP/mixture/EM-panel på låst model med numerisk/MC-status. | Frosset modelinterface; syntetisk først. |
| P5 / G8 | Rekursiv state/rho, conditional posterior, verificeret origin. | P4-paritet. |
| P6 / G7 | Politikoptimering, fælles krydsevaluering og kort estimation/kontrol-loop. | P4/P5; kendt-prior-adapter kan forberedes tidligere. |

G5's bredere forecastbenchmark og G6a's variansreference må ikke blokere
bachelorens centrale FP/estimation/kontrolkæde. Stærk forecastgevinst er
heller ikke adgangskrav. Bevar skellet implementeret / syntetisk demonstreret /
numerisk kontrolleret / historisk valideret i alle statusfiler.

P1/P2's syntetiske round-trip og scoringkontroller er nu gennemført. Næste
skridt er P0/historisk G3 fra Johans originale mappe, derefter P3. G's standard
er de 20 små tests i fit- og snapshotmodulet; kør kun berørte ekstra suites ved delte
interfaces. Ingen fuld test-discovery, historisk månedssmoke eller F1–F4-
genkørsel alene for dokumentation. Lange studier er ikke unit-tests.

Ved hver kodeleverance: opdatér G-status, januarvejledning og den korte
F4/AGENTS-indgang efter faktisk status, med kildecommit, kommando, budget og
råartefakter. Genlæs seneste main før publicering. Denne forberedelse har
oprindeligt ændret dokumentation alene. P1/P2 har nu nye syntetiske tests;
ingen historisk kørsel eller G4–G8-accept følger af dem.

## Reserveret til senere Astra-review

Johan har bedt om, at opgaver som har fordel af dybere matematisk behandling
gemmes til senere. Følgende er anbefalinger om reviewniveau, ikke dokumenterede
modelbenchmarks eller krav om at skifte model for rutineimplementation.

| Opgave | Anbefalet niveau | Konkret afgrænsning |
|---|---|---|
| A1: stokastisk timeinterval-likelihood | **Astra XHigh** | Læs H2_HOURLY_INTERVALS og de gemte negative numerikresultater. Forklar likelihoodfejl/akkumulatordispersion og vurder en billigere tilstrækkeligt præcis metode, f.eks. en afgrænset tilted-generator/Feynman–Kac-vej. Bevar korrekt joint state/interval-observation og nuværende tolerancer. Start med design og en lille uafhængig reference; intet fuldt historisk optimizerfit. |
| A2: posterior → stokastisk prisstyring | **Astra High** | Review parameter–state-afhængighed, mixture/FV-paritet, enheder, forventet kvadratisk tab inklusive varians, posterior-initialiseret EM og cutoff i P4–P6 før samlet metodeaccept. Rutineadaptere kan fortsat implementeres af Sol. |

A1 er nødvendig før historisk intervalbaseret G8/H2-estimation, men blokerer
ikke G3's eksisterende udviklingskontrakt. A2 foretages når adaptere/resultater
er konkrete; et nyt stort studie kræves ikke alene for et review. G4's lille
deterministiske LS-pilot og almindelige kode-/testopgaver kan fortsætte hos Sol.
