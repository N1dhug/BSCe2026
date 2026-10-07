# F4-H2 → I/J: resultatreview, 7. oktober 2026

**Aktuel status, 8. oktober.** Den særskilt godkendte 25/49-fortsættelse har
færdiggjort to-stykkers fit og dets egen fulde base/fine-validering. Firestykkefit
fortsætter fra sin præcise gemte optimizertrace; modelvalg og ugeforecast er
endnu åbne. PR106's separate konturadapter er merged. Blok4 blev afbrudt uden
normal slutrapport, men checkpoint04 sikrer38 komplette kald/780 nye råjobs.
Dens fulde600s-reservation bevares, og blok5 fortsætter under08:00-fristen.
Se [samlet restplan](F4_REMAINING_PLAN_20261008.md). Tidligere negative stops
og kilde-/resultatidentiteter er bevaret; H/I/J/F4 er ikke slutaccepteret.

## Den konkrete opgave

Ét fælles par f/g estimeres på MED-gruppens deklarerede 48 TRAIN-timer.
24 tidligere timer viderefører en Beta(20,20)-prior uden observationslæsning;
24 efterfølgende timer bruges til kompleksitetsvalg. Prognosen er næste
168 timer fra samme frosne origin. Der genfittes ikke per dag.

13 faste MED-huse aggregeres kun ved komplette gruppetimer. Ingen imputation
eller gennemsnit af en delpopulation. Baseline, effekt-/prisskala og pris-
proxy er frosset fra den tidligere periode. De faktiske hushistorier kan
ikke uafhængigt rekonstrueres fra den portable pakke alene; den private
workbook-hash og den tidligere importerprovenance bevares som begrænsning.

De fysiske parametre og støj er nominelle F-betingelser:
C=1, k=2, Xi=1, sigma_x=0,5 og sigma_y=0,04. H2 estimerer her kurver,
ikke husholdningernes fysiske kapacitet eller målerstøj. OLS-baseline fra
før-perioden er en separat estimation; den er ikke samme likelihood-fit.

## Matematik og kode

De oprindelige BM-ligninger bevares:

$$
\delta(x,u)=\frac{2}{1+\exp[-k(f(x)+g(u))]}-1,
$$

$$
D(t,x,u)=B(t)+\Xi\delta(x,u)
\big[(1-B(t))\mathbf1_{\delta\geq0}+B(t)\mathbf1_{\delta<0}\big],
$$

$$
dX_t=\frac{D(t,X_t,U_t)-B(t)}{C}\,dt
      +\sigma_xX_t(1-X_t)\,dW_t.
$$

Det normaliserede timeforbrug modelleres som intervalgennemsnit:

$$
R_i=\frac1{\Delta_i}\int_{t_{i-1}}^{t_i}D(t,X_t,U_t)\,dt,
\qquad Y_i=R_i+\varepsilon_i,
\qquad \varepsilon_i\sim N(0,\sigma_y^2).
$$

Det er ikke en punktobservation af X eller D ved timeslut. Terminal-
likelihooden kombinerer intervalakkumulatoren og terminaltilstanden, så
den filtrerede conditional state kan videreføres korrekt gennem midnat.
Den frosne positive FV/CN-reference og karakteristiske terminalinversion
beregner predictive Y-likelihood i logkoordinater. Guards og egne
base/fine-forløb kontrollerer numerikken; massebevarelse alene er utilstrækkelig.

For m=2 eller m=4 stykker bruges positive fald med samlet fald 2:

$$
q_j=\frac{\exp a_j}{\sum_{\ell=1}^{m}\exp a_\ell},\quad a_m=0,
\qquad c(z_j)=1-2\sum_{\ell=1}^{j}q_\ell.
$$

Her er c enten f eller g, og z_j=j/m. Lineær interpolation giver
kontinuerte, ikke-voksende kurver med endpoints 1 og -1. De to familier
har henholdsvis 2 og 6 frie logits samlet for f/g; lineær reference er
alle logits nul. Endpoints og reference-logit fjerner redundanser i
parameteriseringen, men beviser ikke praktisk identifikation.

Det fælles TRAIN-objective er

$$
Q(a_f,a_g)=-\sum_{i\in\mathrm{TRAIN}}
 \log p_{a_f,a_g}(Y_i\mid Y_{<i},B,U)
 +2\sum_{c\in\{f,g\}}\sum_{j=1}^{m-1}
 \big[c(z_j)-(1-2z_j)\big]^2.
$$

Hver ny kandidat genafvikler hele sin egen kronologiske prior/TRAIN-kæde.
Kun identiske jobs med identisk prior, input, kurve og kilde kan genbruges.
Den samme state-posterior må ikke genbruges på tværs af forskellige kurver.
COBYQA søger under den deklarerede logit-bound [-4,4], med én lineær start
og den særskilt godkendte maxfev25/49-ramme. De gamle17/33-resultater
og terminalmarkører er historik og bevares. Det er en afgrænset lokal søgning; ingen global optimalitet.
`success` og trust-radius-status0 er det erklærede algoritmiske stop.
Et kandidat-/tidsloft dokumenterer et ufuldstændigt fit.

Gaussisk målingsstøj gør ikke nødvendigvis den predictive Y-fordeling
gaussisk: R_i afhænger af en ukendt stokastisk state-bane. Et almindeligt
least-squares-fit til én indsat state-bane ville derfor ændre observations-
modellen og ignorere en del af usikkerheden. Least squares bruges allerede
til før-periodens baseline; H2's kurvefit bruger den deklarerede marginale
interval-likelihood for den latente model.

Lineær, to-stykkers og fire-stykkers familier sammenlignes efter egne
fine-TRAIN/selection-replays. Selection vælger efter predictive score;
scoregab højst0,1 eller base/fine-rangreversal giver den enklere-modelregel
og ingen forbedringsclaim. Slutugen ændrer hverken kurver eller modelvalg.

Kodekort:

| Led | Kode |
|---|---|
| Fysiske BM-ligninger | `src/model/base/bm.py`, `src/model/bm_1.py` |
| Monotone kurver | `src/method/bm_1_response.py` |
| Interval-likelihood og guards | `src/method/bm_1_interval_native.py` |
| Intervalmomenter, CDF og kvantiler | `src/method/bm_1_interval_forecast.py` |
| Immutable jobstore, fit, modelvalg, forecast og finale | `src/exe/run_model/bm_1_h2_compact.py` |
| Oprindelig godkendt budgetadapter | `root/f4/run_h2_same_day.py` |
| Aktuel separate konturfortsættelse | `src/exe/run_model/bm_1_h2_continuation.py`, `root/f4/run_h2_continuation.py` |
| Immutable adoption og checkpointreview | `root/f4/checkpoint_h2_continuation.py` |
| Faktisk H→I-originreview efter freeze | `root/f4/review_h2_i_handoff.py` |
| Frossen prognose → PC-slutdata | `root/f4/export_h2_final_observations.py` |

## I-handoff og finale

Den frosne eksport indeholder begge kurver, de nominelle parametre,
population/skala, origin-tid og hvert grids egen fulde log-state.
`as_forecast_origin` overfører det eksplicit til `BM1ForecastOrigin` og
`BM1Parameters`. F3's konservative masseadapter genbruges. Det er et
betinget plug-in: parameter- og fremtidig inputusikkerhed er ikke inkluderet.
En eksport beviser ikke i sig selv ny historisk F3-policy-optimering.

Åben prognose viderefører state uden nye observationer og leverer
intervalmomenter for168 timer samt fordelinger ved1/6/12/24/48/72/168 timer.
En separat prequential gren starter fra samme origin og opdaterer med
slutugens observationer, efter at den åbne prognose er frosset. De to
informationsmængder skal rapporteres hver for sig. Prognosens kW-middel
og standardafvigelse fås ved at gange normaliserede værdier med den
frosne effektskala; dens fejlmål skal have eksplicit enhed.

Syv PIT-/dækningspunkter er beskrivende diagnose. De er ikke200 uafhængige
sekvenser eller dokumentation for90%-kalibrering på tværs af populationen.
Marginale timefordelinger giver heller ikke en samtidig ugegaranti eller
ugeenergi-varians uden tidskovarianser. Slutugens tidligere eksponering
er ukendt: den skal fortsat kaldes udviklings-holdout, ikke blind H3.

## Faktiske resultater og afslutning

To-stykkers COBYQA nåede status0 efter20 kald, Q=-10.155937279386325.
Den valgte kurves f(0.5)=0.48508748289841463 og
 g(0.5)=0.6331765945295157; endpoints er fortsat1,-1.
Dens fulde egen numeriske validering består:

| Mål | Resultat |
|---|---:|
| TRAIN-score, base/fine | 11.428382211235753 / 11.4253020588032 |
| TRAIN-scoreforskel | 0.003080152432552552 |
| Selection-score, base/fine | 23.267887918742304 / 23.26716786258129 |
| Selection-scoreforskel | 0.0007200561610147815 |
| Maksimal state-L1 | 0.0029276959731513326 |
| Selection-RMSE, fine, normaliseret timeforbrug | 0.07219496766405152 |
| Selection-MAE, fine, normaliseret timeforbrug | 0.060482808455341874 |

Dette er numerisk validering af én fælles fitted familie, ikke modelvalg,
fysisk identifikation eller ugeforecastaccept. Den oprindelige240-job-
validering og dens identity genbruges; ingen replay alene for denne rapport.
Firestykkefit har endnu ingen gemt termination; dets lavere TRAIN-objective
må ikke bruges til at vælge kompleksitet. Lineær/to/fire sammenlignes først
efter fuld egen base/fine-TRAIN og selection for alle relevante familier.

Den ni-konturfortsættelse bevarer den oprindelige fem-konturprefix og
tilføjer fire repræsentationskonturer. Egen24h fine-prior og den oprindeligt
fejlede time består uden at ændre model eller guards. Adopterede jobs har
deres originale tokens/checksums/prior bytes; nye jobs har separat identity.
Kandidat28's gamle terminalfejl og alle crashreservationer bevares fuldt.

Næste resultater, når de foreligger:

- Algoritmisk status for begge nonlinear familier og faktisk forbrug.
- Egne TRAIN/selection-scoreforskelle, state-/moment-/CDF-gates og modelvalg.
- Frosne kurvekoefficienter, origin, sourcecommit og checksums.
- Åben versus prequential slutdiagnose og referencefejl.
- Faktisk demonstreret I-handoff og samlet konklusion til J.

Hvis et fit stopper ved sin cap, skal dette afsnit angive den konkrete
negative konklusion og resterende leverance. Et godt TRAIN-objective
eller et bestået interface må ikke ommærkes til afsluttet H/I/J.
F4-G2b/G2c er droppet. Det separate deterministiske G-spor indgår ikke
i denne F-afslutning.
