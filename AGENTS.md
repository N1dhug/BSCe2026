# Fælles projektkontekst

**Nyt selvstændigt spor, Johan 5. oktober: [G-planen](root/G.md).**
Deterministisk BM1-kerne, afgrænset parameterfit, matched WM4-prognoser og
et efterfølgende billigt stokastisk usikkerhedslag.
Læs G-planen ved dette arbejde; `G-plan/G0`–`G-plan/G8` er ikke `F4-G2a/b/c`.
G0–G3 er første deterministiske leverance. G6 er en obligatorisk senere
stokastisk pilot med FP/mixture og EM; G7 er obligatorisk stokastisk
single-shooting-prisstyring; G8 er rekursiv tilstands-/parameterestimation.
Bachelorens projekt 111866 kræver disse sammenhænge. Genbrug F2/F3/F4-kode
og evidens; G6a's varianslag og G2's batch-LS er ikke erstatninger.
Se kravmatrix, WM-kodereview og små forsøgsbudgetter i G-planens afsnit 0/8.
Grundarbejdet fra G3 ligger i [Sol-overdragelsen](root/G_IMPLEMENTATION_HANDOFF.md):
P1/P2 er nu implementeret: `deterministic --from-fit FITDIR` genbruger et gemt
fit eller kontrolleret legacy-format uden ny optimering og gemmer forecast før
optional scoring. Nye fits skriver `fit_snapshot.json`. 20 G-tests og tre
relevante layout/importkontroller består (23 i alt, ca. 6.7 s). Historisk
råreview/G3-kørsel mangler. Derefter lille G4→G6b→G8→G7 er fortsat planlagt.
G's legacy-aggregering bruger tilgængelige målere per time; fast population og
punkt/interval/enheder skal afklares før historisk validering. Ommærk ikke
januarfittet til en ny datakontrakt. Syntetisk adapterarbejde kan begynde nu.
G som helhed er ikke rent deterministisk. F4-status og dets
obligatoriske H-f/g-krav bevares. G0–G3 er implementeret og syntetisk kontrolleret; Johan har rapporteret historisk januarfit som complete på 07fad9f;
råreview og historisk G3-prognose mangler.
Johan har bestilt hele januar 2022 som fælles BM/WM-fitmåned; næste 168 timer
forudsiges med faste parametre. Læs [januarvejledningen](root/G_JANUARY.md).
Ny januarvisning: `--fit-only --initial-state wm-neutral` bruger de virkelige
DK1-priser og kræver ingen februardata. Standard-x0 er fast (1-INIT_F)/2=0.5
fra WM's neutrale INIT_F=0; kun C,k,Xi fittes. `--initial-state fit` bevarer
den tidligere x0-estimation. Fit.png/ugesnit, CSV og WM-metrics viser in-sample
januarfit, ikke forecast-accept. Ingen historisk kørsel er udført i chatten.
G-januar håndterer nu manglende demand/temperatur med den eksisterende import-
interpolation afgrænset til TRAIN. Rå demand beholdes som NaN; baseline/BM-fit
og fælles BM/WM-fejlmål bruger kun målte timer. WM-tilstandsrekonstruktion bruger
udfyldt demand; QP-overgange over manglende endpoints maskeres. G kræver stadig
mindst 168 målte demand-timer i TRAIN og komplette importerede priser. 10 tests består;
én syntetisk januar med 27 demandhuller og 3 temperaturhuller gennemført
på ca. 19 s (717 målte timer for begge modeller). Historisk fit er nu rapporteret gennemført lokalt; råreview mangler.
G3 er rapporteret complete på main 607dcd7, run g_january_forecast_20261005_204059:
168/168 scored timer. MSE BM=0.06951591504, O=0.06934621735, WM=0.07099789830;
ingen baselinegevinst på den eksponerede udviklingsuge. Råreview mangler.
G4's lille to-stykkers g-adapter/LS er implementeret og syntetisk kontrolleret;
historisk pilot/udviklingsvalg mangler. Brug `deterministic_response --reference-fit`
på original TRAIN-eksport, ingen G2/WM/baseline-refit. K er fast; samme x0-policy.
Lineær G3-kernel er bevaret af hensyn til arkivkompatibilitet. G4 har ni korte
tests i `tests.test_bm1_deterministic_response`; brug dem ved G4-ændringer.
Maks. to strafstyrker inkl. nul, 3.000 residualkald/600 s samlet inklusive
replay/forfining. TRAIN vælger ikke strafstyrke. Derefter G6b→G8→G7;
G5's brede benchmark og en stor baselinegevinst må ikke blokere kernekravene.
Nye G-adaptere får egne korte kontroller; F4/H-status og tidligere gates bevares.
Ved almindelige G0–G3-ændringer er standardkontrollen de 20 tests i
`python -B -m unittest tests.test_bm1_deterministic_fit tests.test_bm1_deterministic_snapshot -q`. Dette erstatter
fuld test-discovery for dette spor. Kør kun berørte ekstra tests ved ændringer
i fælles BM/WM-kode eller entrypoints. Gentag ikke månedssmoke eller historisk
fit alene for dokumentation/testoprydning. Se testpolitikken i januarvejledningen.
Præcisering kl. 12:56: genbrug WM4's `statistical_output`-baseline, samme
fit/predict-protokol, metrics, performanceanalyser og rapportlayout dér, hvor
muligt. Første benchmark følger WM's én-fituges/næste-uges-protokol. Forskelle
skal begrundes; se den bindende kompatibilitetsmatrix i G-planens afsnit 1a.
Astra-opgaver er registreret i G-overdragelsen: XHigh til præcision/runtime
for den stokastiske timeinterval-likelihood; High til senere matematisk
review af posterior→FP/mixture-kontrol. Ingen af dem blokerer legacy-G3.

**H-recovery, 5. oktober:** H1a-v2 er reproduceret på `90211e4` og råarkiveret.
Alle seks betingede H1b-fits er nu færdige på `e7d9089`; to responsive
fire-stykkers fits nåede kandidatloftet. De valgte kurvers 12 units / 48 replays
består egen numerik på `e398a5a`, og 2.304 Bayes-opdateringer er rå-reviewet.
PR #64 supplerer PR #62 med checkpointvalidering og genfundne I/O-/TRAIN-fixes.
Dette er udviklingsevidens, ikke modelvalg, joint recovery eller H2/H3-accept.
H2-metadataaudit har nu verificeret alle 13 MED-ID'er som huse med elvarme
og alle 140 UDEN-ID'er som huse uden elvarme i samme workbook som Johans
header-rapport. Se `root/H2_DATA_CONTRACT.md`; nul observationsværdier er læst.
Johan har senere deklareret timeforbrug, foreløbigt kWh, og hour-start
(03 betyder [03:00,04:00)); Europe/Copenhagen er en arbejdsantagelse.
Separat joint state/akkumulator-likelihood og timeenergiadapter er implementeret
med afgrænset shared f/g-replay. Se `root/H2_HOURLY_INTERVALS.md`.
De første grove BM-grids fejler udviklingens numeriktolerancer trods bevaret
masse; anbefal ikke fuldt historisk fit på disse settings. Ingen faktiske
måleværdier er læst eller historisk/H2/H3-accept opnået. Dokumenteret blindt
split, populationvalg samt pris/skala/baseline/fysiske parametre mangler stadig.
Særskilt H2-characteristic/CN-udviklingskontrol er nu gennemført på samme
syntetiske døgn: log-amplituder løser de små tails; tre positive CN-rum-/tidschecks
består de eksisterende tolerancer. Se `root/H2_INTERVAL_CHARACTERISTIC_CONTROL.md`.
Kun dette faste lineære case er kontrolleret; historisk fitter/reference er uændret.
Fleksible kurver/støj, faktisk tidsvariation, log-checkpointadapter og runtime skal
stadig kontrolleres før historisk anvendelse. H2/H3-readiness forbliver falsk.
Recovery, råartefakter og næste trin: `root/F4_H_CHECKPOINT_AUDIT_STATE.md`.

Ved BM/F4-arbejde: læs først `root/F4.md`, især **Aktiv prioritering — tidsreduktion,
3. oktober 2026**. Det er den autoritative fælles status og to-do-liste; ældre
chatplaner og historiske "næste opgave"-afsnit kan være overhalet.

- Aftalt rækkefølge, Johan 3. oktober kl. 23:08: **F4 færdig → grundig
  prognosediagnose og eventuelle begrundede forbedringer → F5 → F6**. Se
  **Opfølgning efter F4 — demand-prognose og baseline** i `root/F4.md`.
  Nye model-/baselineændringer for de oversete demand-hop starter først efter
  F4; mellemtrinnet er ikke en ny F4-gate. Det tidligere samlede tidsbudget
  skal revurderes med dette mellemtrin.
- Brugerens regel, 3. oktober 2026: brug **branches i eksisterende checkout**.
  Opret ikke flere worktrees, medmindre det er strengt nødvendigt; en ny fase,
  preview eller parallel chat er ikke i sig selv en grund. Bevar source under
  aktive låste kørsler. Før et eksisterende worktree fjernes: bevar branchen,
  egne ændringer og alle nødvendige ignorerede rådata/logs; kontrollér at ingen
  kørsel bruger mappen. Reglen gælder også WM-arbejde.
- I-baselinens origin/forecast/F3-integration er implementeret på det afsluttede
  F-spor; pooled H-f/g-fitprototype og gammel-record budgetpilot er implementeret.
  H1-cases/numerik, H1b-fit og H's data-/splitkontrakt er implementeret.
  H1a/H1b-udviklingsrådata er sikret; næste arbejde er faktiske metadata,
  eksponeringsaudit og H2/H3-design før historisk modelvalg.
  Obligatorisk H-f/g
  skal gennemføres før den endelige samlede I/J-leverance. G2b-v2 er valgfrit
  nat-sidearbejde; G2c er kun en mulig opfølgning efter G2b-accept. Ingen af dem
  må blokere H/I/J eller markeres gennemført uden komplette kontroller.
- G2a-v2 er komplet, rå-reviewet og numerisk bestået på ren `01c7c153`:
  8 units, 42.312 replays, 14 kontroller; nul nye FV-replays i råreviewet.
  Bevar original source og arkiv; løs ikke resume-kontrollen. Det frigiver
  hverken G2b/G2c eller recovery/forecast. Næste aktive arbejde er H1/H's datakontrakt.
- Beslutning fra Johan, 3. oktober kl. 22:26: natkørsler budgetteres til cirka
  otte timer per invocation, med hyppige atomiske blok-checkpoints og genoptagelse.
  Brug den særskilt registrerede `joint-bank-rectangular-robustness` (G2b-v2),
  som råvaliderer alle otte beståede G2a-v2-units før nyt arbejde. Tidsstop er
  en pause, ikke accept; bevarede hypoteser genkøres ikke. Gamle kvadratiske
  G2b/G2c-prerequisites løsnes ikke. Det rektangulære G2c-v2-design er nu
  registreret, men kræver komplet råvalideret G2b-accept. Fordelt PC/chat-kørsel
  bruger én fælles plan og én ejer per shard; se BM_GUIDE. Importeret G2b-arbejde
  kræver identisk numerisk kilde og bevarer originalfiler/source.
  Fastlås nat-source i en branch i samme checkout; skift/pull aldrig
  under en aktiv kørsel. H1/H2/H3 og endelig I/J er fortsat hovedsporet.
- Metode og forsøgsdefinitioner: `root/f4/notes.tex`; kommandoer: `root/BM_GUIDE.md`.
  Opdatér fælles status i `root/F4.md` sammen med en leverance.
- Genbrug F1–F3-interfaces og gennemført evidens. Bevar grids, seeds, observationers
  betydning og numeriske/statistiske gates ved rene performanceændringer.
- Kontrollér nyeste `main` før publicering og bevar andres samtidige ændringer,
  herunder WM-arbejde. Følg brugerens konkrete instruktion om branch/publicering.
- Relevant regression: `python -B -m unittest discover -s tests -q`.
  Noter: `python -B root/f4/build.py`. Lange forskningsstudier er ikke CI-tests.
- Nye lange kørsler skal vise et løbende groft estimat for både resterende og
  samlet tid ved eksisterende statusopdateringer. Genbrug
  `src/exe/run_model/progress.py`; medregn forskellig opgavepris, hvor den er
  kendt, og adskil nyt arbejde fra gemt arbejde ved resume. Ingen ekstra replays
  eller hyppigere checkpoint-skrivning for at måle ETA.
- Følg også F4-planens afsnit **Yderligere effektivisering**: fælles I/F5/F6-
  interfaces, tidlig H-f/g-pilot og forberedelse under beregninger. Brug
  `run_fv_filter(..., keep_history=False)` til rene likelihood-kandidater og
  `evaluate_targets` til flere kvadratiske A-mål ved uændret U/process. Fuld
  historik/validering bevares for valgte modeller; de nye helpers afslutter
  hverken H-f/g, F5 eller F6. Ingen ny spekulativ solveromskrivning på kritisk vej.
- Genbrug `BM1ForecastOrigin` i `src/control/bm_1_handoff.py` til I/F5/F6:
  eksplicit cutoff/tidsenhed, frosne parametre/kurver og konservativ posterior.
  Rho-MAP bruger nodens egen conditional posterior. Et gennemsnitligt parameter-
  plug-in kræver nyt replay eller erklæret approksimation. `i-baseline` er kun en
  gammel-record integrationsdemo; ingen forecast- eller H-f/g-accept.
  `BM1PiecewiseResponse` og `src/method/bm_1_response.py` giver kurve-API og
  pooled replay-fit. Gammel-record prototype er ikke H's endelige fit/accept.
  Bevar deklarerede kandidatbudgetter og negative budgetstops; ingen automatisk
  ekstra fitrunde eller modelvinder efter TRAIN-score. H1's udviklingscases og
  valgte-kurve-numerik er gennemført; faktisk dataaudit, hele den historiske
  træning og blind H3 mangler stadig.

Disse kurveregler gælder BM og ændrer ikke WM-modellernes egne fitkontrakter.

## Beslutning fra Johan, 3. oktober 2026

- Estimation og validering af fleksible `f(x)`/`g(u)` er en **obligatorisk H-f/g-leverance**,
  endnu ikke afsluttet; udviklingsfit er gennemført, mens historisk fit og
  videnskabelig validering mangler. Den må ikke igen klassificeres som en valgfri udvidelse.
- Ét fælles kurvepar skal bestemmes samlet på tværs af træningsdage for en erklæret
  population/gruppe og holdes fast ved validering/forecast. Ingen f/g-fit per dag.
  Daglig baseline og state-estimation kan fortsat variere.
- Undersøg kontinuerte, ikke-voksende stykvise kurver med knuder
  `0, 0.25, 0.5, 0.75, 1`, faste endpoints `1,-1` og fælles estimerede knudeværdier.
  Intervaller vælges efter state/pris, ikke tidspunkt. En enklere pilot og lineær
  reference bevares; alle delstykker indgår i samme likelihood-fit.
- H-f/g skal gennemføres før endelig H→I-handoff og J. Split, normalisering,
  observationskontrakt, numerik og modelvalg dokumenteres; slut-holdout holdes blind.
  Nye kurver kræver egne kontroller. Det faste lineære A--G-design og gemt evidens bevares.
- Budgettér kurveestimationen med en lille pilot og afgrænset optimizer; opret ikke
  et stort Cartesian-grid over kurve- og fysiske parametre. Følg H-f/g-planen.
- Ved ændret status skal `root/F4.md`, nødvendig matematik og vejledning stemme overens.
  Beskriv ikke en planlagt eller blot implementeret estimator som valideret.

Dette er den fælles repoindgang for projektchats. En allerede åben chats indlæste
kontekst opdateres ikke automatisk; ved genoptagelse skal den læse den aktuelle plan.
