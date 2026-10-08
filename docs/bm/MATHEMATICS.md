# BM-F/G: gældende matematik og kode

Denne oversigt beskriver de kørbare udviklingsmodeller. De oprindelige
modelkilder og den afledte numerik er forskellige lag; ingen ny modelantagelse
eller statistisk accept indføres af oprydningen. Den fælles BM-model og WM
bevares uændret. Se også [modelreferencen](../../root/BM_REFERENCE.md).

## Fælles demandbalance

Tilstanden er BM's fyldningsgrad $X\in[0,1]$, prisen $U\in[0,1]$, baseline
$B\in[0,1]$ og relativ fleksibilitet $\Xi\in[0,1]$. Definér

$$\delta(x,u)=\tanh\!\left(\frac{k[f(x)+g(u)]}{2}\right),\qquad
D=B+\Xi\delta\begin{cases}1-B,&\delta\geq0,\\ B,&\delta<0.\end{cases}$$

Den normaliserede energibalance er $dX=(D-B)dt/C$. Den giver $0\leq D\leq1$.
Med ikke-voksende responskurver, endpoints $f(0)=g(0)=1$ og
$f(1)=g(1)=-1$ peger kontinuert drift indad ved begge tilstandsgrænser.
Den fælles implementation er `bm.D`, `bm.dXdt` og `bm.l` i
[base/bm.py](../../src/model/base/bm.py); den korte modelindgang er
[bm_1.py](../../src/model/bm_1.py). Begge er også WM-afhængigheder og beskyttede.
BM's $X$ er ikke WM's $F$ eller $R$.

## F: stokastisk tilstand og intervalobservation

F bruger tid i **døgn**, en fysisk time er $\Delta=1/24$:

$$dX_t=\frac{D(X_t,U_t,B_t)-B_t}{C}\,dt+
\sigma_xX_t(1-X_t)\,dW_t.$$

$C$ har normaliseret tidsenhed døgn, og $\sigma_x$ har enhed
døgn$^{-1/2}$. Det kontinuerte randargument med indadrettet drift og forsvindende diffusion
er ikke et bevis for enhver diskret solvers nøjagtighed. Endelig-volumen-kernen
har særskilte masse-, positivitet- og inversionskontroller.

Den anvendte observationsoperator er

$$Y_j=\frac1\Delta\int_{t_j}^{t_j+\Delta}D(X_s,U_s,B_s)ds+\epsilon_j,
\qquad\epsilon_j\sim N(0,\sigma_y^2).$$

Dette er intervalmiddel, ikke punkt-demand. `freeze_native_operator` og
`native_terminal_update` i [interval_native](../../src/method/bm_1_interval_native.py)
bruger den fælles terminal-state/reward-fordeling. Filtrering betinger på
den observerede time; prognose marginaliserer uden fremtidige targets.
`native_interval_moments` og `native_interval_distribution` i
[interval_forecast](../../src/method/bm_1_interval_forecast.py) beregner
middel/varians og observationernes marginalkvantiler. $\sigma_y^2$ lægges
kun til observationsvariansen. Numerisk inversionsfejl er ikke statistisk usikkerhed.

Det gemte tostykkefit har ét fælles kurvepar for 13 MED-huse. Monotone
stykkevise kurver bygges af positive softmax-fald i
`curve_pair_from_logits` i [response](../../src/method/bm_1_response.py).
De fysiske/støjparametre er nominelle: $C=1$, $k=2$, $\Xi=1$,
$\sigma_x=.5$, $\sigma_y=.04$. De er ikke identificeret af denne startkommando.
Baseline er en frossen ugeprofil. Ny filtrering begynder med Beta(20,20),
ikke H-kampagnens oprindelige posterior. Time med blot én manglende husmåling
propagerer uden opdatering; gruppens observation udfyldes ikke.

F-indgangen [bm_1_f4](../../src/exe/run_model/bm_1_f4.py) følger
`prepare → run → native interval kernel → export`. Under kernen ligger
`interval_terminal_tilt` og `interval_characteristic` i `src/validation/bm_1`:
de er produktionsafhængigheder trods mappenavnet. Kurve- og driftlag genbruger
BM-model, FV og validering. Disse moduler beholdes med deres konkrete ansvar.

De relevante kontroller er [F4 runner-tests](../../tests/test_bm1_f4_runner.py)
(cutoff, populationsmaske, demandisolering, missing og præcis pause/genoptagelse)
og de bevarede native interval-/response-kontroller. Ingen guards lempes.

## G: deterministisk drift, timeoperator og empirisk outputlag

G bruger tid i **timer** og $f(x)=1-2x$. Den valgte $g$ er lineær mellem
$(0,1),(.5,a),(1,-1)$. `BM1ResponseInputs.solve` i
[deterministic_response](../../src/method/bm_1_deterministic_response.py)
integrerer $dX/dh=(D-B)/C$ med kontrolleret RK4; den afviser domænebrud og
klipper ikke tilstanden. For én fysisk time giver energibalancen

$$\overline D_j=\overline B_j+C(X_{j+1}-X_j).$$

For en generel intervallængde skal sidste led divideres med længden.
`point_demand` bevares som en særskilt størrelse. Baselinefeatures integreres
med 60 trapezdelintervaller per time i
[interval_baseline](../../src/method/bm_1_s3_interval_baseline.py).
Fit bruger integrerede lineære features; deployment klipper baseline.
Disse to operatorer er derfor ikke identiske. Rank 20 for 49 features i alle
42 gemte fits er en konkret identificerbarhedsbegrænsning, ikke løst af ridge.

[weekday_rolling](../../src/method/g_weekday_rolling.py) fitter syv baselines
og parametersæt til fem foregående ens ugedage. Skalaen bruger kun femugershistorik.
Alle syv fit fryses ved mandagsorigin. Hver dags G-start er medianen af egne
historiske warmup-sluttilstande: prognosen består af **syv dagsbaner**, ikke én
sammenhængende 168-timers batteribane. Saved export i
[weekday_export](../../src/method/g_weekday_export.py) genbruger uændrede fits
med verificeret 60/120-forfining og egne historiske residualer.

Empiriske fejl $E=Y-\widehat D$ findes ved replay med deployment-startpolitik.
Outputmiddel forskydes med gennemsnittet af $E$; variansen bruger `ddof=0`.
Det er empirisk modelfejl på historiske data, ikke identificeret målerstøj,
parameterposterior eller fysisk diffusion. Det adskilte procesmomentlag findes
stadig i [interval_process](../../src/method/bm_1_s3_interval_process.py).

Frosne bånd bruger $z=|Y-m|/s$, $s>0$, og ordensstatistikken
$q=z_{(\lceil(n+1)0.95\rceil)}$. Senere bånd er $m\pm qs$.
`finite_rank` og `band_metrics` i
[prequential_bands](../../src/method/g_prequential_bands.py) genbruges;
ændret bredde ændrer ikke modellens predictive variance. Den stærke serielle
afhængighed og eksponerede udviklingsperioder giver ingen universel 95%-garanti.

Importkæden er `bm_1_g → g_weekday_export → g_weekday_rolling →`
`bm_1_s3_interval → bm_1_s3_forecast / deterministic_*` samt
`bm_1_s3_residual` og WM's uændrede `output_model.statistical_output`.
Punktmodulet er en genbrugt afhængighed, selv om punktprotokollen er historisk.
Der tilføjes ingen ny kalender eller numerisk model bag den korte kommando.

Relevante tests: [interval](../../tests/test_bm1_s3_interval.py),
[residual](../../tests/test_bm1_s3_residual.py),
[weekday/bands](../../tests/test_g_weekday_bands.py) og
[fælles kalender](../../tests/test_weekday_backtest.py). De forbinder timebalance,
forfining, rå skalering, nulproces, frysning, DST, kalibreringsrank og targetisolering.

## WM-reference og fælles evalueringsgrundlag

WM4's tilstand er $F'=R$, $R'=r_FF+r_UU+r_RR$ og $I=B+R$, tid i døgn.
Den oprindelige weekly-CLI bruger punktoutput og sit eget QP-fit. BM's særskilte
WM-timeadapter integrerer de samme uændrede ligninger med matrixeksponential
og free-run variable projection. For en time er $\overline I-\overline B=24\Delta F$.
Adapterens eget fit/resultat må ikke kaldes det oprindelige WM4-resultat.
`--optimize-prediction-state` bruger fremtidig demand og er oracle-diagnostik.

Demand-enhed, tidszone og hour-start er endnu **arbejdsantagelser**.
Hvis kilden er kWh pr. time, er tallet lig timegennemsnittet i kW; dette
verificerer hverken kildens enhed eller label. Ingen reskalering sker i oprydningen.
F og G har forskellige populationsmasker, prisinformation, baselines, cutoffs
og usikkerhedslag; se [sammenligningen](COMPARISON.md).

## F-weekday: implementeret native fit-adapter

[FWeekdayAdapter](../../src/tools/f_weekday_backtest_adapter.py) er nu koblet
til den eksisterende `weekday_backtest`-kalender. Den laver **nye tostykkefits**
med `fit_shared_curve_pair`; de gemte F4-kurvelogits bruges ikke som fitresultat.
Likelihood, nominelle fysiske parametre, Beta-prior, intervaloperator og
frossen F-baseline bevarer deres egne betydninger.

Kontrakten kræver en eksplicit, hashbundet F-baseline-/skalapakke med cutoff
senest ved hele femugershistorikkens begyndelse. Den medfølgende F4-profil
har cutoff 16. februar; første mulige mandagsorigin for femugershistorik er
derfor 28. marts 2022. Profilen er tidligere frosset, ikke nytilpasset til de
fem mandage. Det er et deklareret valg; den erstattes ikke af G's dagsbaseline.
Den nominelle prior/fysik er betingelser, ikke historisk identificerede parametre.

For hver ugedag starter hver kandidat fra sin egen Beta-prior ved historikkens
begyndelse og propagérer gennem **alle fysiske timer**. `split_weekday` maskerer
targets uden for de fem matchende dage, mens inputs og tilstand fortsætter.
Manglende observationer giver ingen logevidence og intet dagligt state-reset.
Optimeringen bruger hele denne TRAIN-likelihood plus den deklarerede
kurvestraf. `replay_native_record` forbinder responsoptimereren med
`replay_native_candidate`; de seks eksisterende native beregningsfunktioner
er uændrede. Der er ingen ny PDE-solver eller punkt-/Gaussian-mean-erstatning.

Valgte logits skal være identiske med et faktisk succesfuldt optimizerresultat.
Timeout, budgetstop, ufuldstændig replay eller numerisk fejl giver ingen gyldig
fit-score eller forecastaccept. Hele den valgte historik kontrolleres på
base/fine: samlet logscoreforskel højst .05, statemiddelforskel .002 og
konservativ L1 .05 ved hver fysisk time. Forecast kontrollerer tilsvarende
state- og intervalmomentgates. Disse er numeriske gates, ikke et bevis på
identifikation, globalt optimum eller statistisk modelaccept.

Hvert af de syv fits fryser sin egen posterior ved mandagsorigin og føres
gennem hele det targetfri prognosepræfiks frem til sin ugedag. Resultatet
samler syv selvstændige, kontinuerlige kandidatbaner; det er ikke én fælles
168-timers posteriorbane med daglige parameterskift. Den almindelige F4-CLI
har fortsat én åben bane med ét gemt fit. Middel og baseline skaleres med
effektskalaen, varians med dens kvadrat. Native observationsvarians omfatter
den nominelle Gaussian-støj; der påstås ingen kalibreret bånddækning.

Native checkpoints binder backendversion, alle kildehashes, Python/NumPy/
SciPy/Numba/llvmlite-versioner, konkrete kandidatlogits, inputspecifikation,
grid, kronologi, observationer og evidenschecksum. Endelig logmasse bevarer
underflow-haler; sande nuller gemmes som JSON null og indlæses som −∞.
Precise- og native-checkpoints kan ikke ombyttes implicit. Den native kerne
har testet pause/resume. **Hele weekday-CLI'en er stadig fresh-output-only**:
dens fit-/forecastartefakter gemmes til recovery, men læses ikke automatisk
som en fortsat optimizer-/ugekørsel.

De nye [adaptertests](../../tests/test_f_weekday_adapter.py) omfatter rigtig
kort native kurveoptimering, nulmasse/underflow, ændret kandidat/source/runtime,
cutoff, populationskontrakt, DST og targetisolering. En firetimers case med
ikke-nul drift/diffusion, inputskift, gap, midnat og resume stemmer med den
bevarede precise-kerne. 29 nye/berørte native-, response- og kalenderkontroller
består; seks eksisterende F4-tests består også efter integrationen.

**Ingen historisk femugers-fit er kørt.** Et sådant fit kræver omtrent 840
historiktimer pr. kandidat og ugedag (DST kan ændre antallet). De eksplicitte
budgetter i [eksempelkonfigurationen](../../root/f_weekday_adapter.example.json)
er korte stopgrænser, ikke målte produktionsbudgetter. Budgetkontrol sker
kooperativt før/efter nativeblokke og timer; den stopper ikke en igangværende
kernel midt i et trin. En senere konkret fitplan skal have sit eget samlede
budget og eventuelt et eksternt processtop. Populationens komplette måler-
maske er callerens kontrakt og kan ikke verificeres fra et allerede beregnet
`Demand`-gennemsnit alene. Brug og CSV-format står i
[kalendervejledningen](../../root/WEEKDAY_BACKTEST_GUIDE.md).

## Kodestruktur efter gennemgangen

| Ansvar | Beholdt ejer og begrundelse |
|---|---|
| BM-ligninger og fælles input/output | Beskyttede model-/WM-afhængigheder; samme fysiske betydning. |
| F intervalnumerik | Native, precise/characteristic og intervalmomenter har adskilte produktions-/referenceopgaver; referencekontroller er ikke dubletsolvere. |
| Kurvefit | `bm_1_response` ejer kurver og optimizer; solverimports udføres ved replay/fit. Ny native replay er en eksplicit backendtilslutning. |
| G gemt intervalprognose | `g_weekday_export/rolling` og de 20 bundne kilder bevares til eksisterende snapshots. |
| G7/G8 bank og kontrol | Basis og jobbudget er samlet hos basis-/jobejerne; posterior, kontrol og forfining bevarer særskilte ansvar. |
| Ældre S3-indgange | Én `s3`-CLI, eksplicit interval/residual/point-process. Den særskilte `s3_forecast`-API beholdes af hensyn til WM-brugere. |
| H/I/J og valideringsharnesses | Bevar konkrete tests, negative resultater og forskningskontrakter; tre afsluttede H2-kampagnedrivere er i originalarkivet. |
| Kalender og eksport | F/G genbruger den samme kalender, targetisolering, freeze/scoring og rå intervalfelter; ingen ny sammenligningsramme. |

Gamle kildeidentiteter håndteres i [det gendannelige originaltræ](../../archive/bm/ORIGINALS.md).
En pakket kopi tælles ikke som en forenklet kodefil. De faktiske ændringer og
før/efter-tal står i [sammenligningen](COMPARISON.md).

## Metadata og øvrige åbne modelvalg

Anden rundes read-only audit af workbook-docProps, sharedStrings og headerne
på forbrugs-/labelark fandt `Dato`, `Time`, `DateTime` samt MED/UDEN-grupper.
Den fandt ingen leverandørdefinition af fysisk enhed, intervalstart/slut eller
tidszone. Dette supplerer dokumentgennemgangen og begrunder fortsat den mærkede
arbejdsantagelse; hverken et kolonnenavn eller talstørrelsen beviser kWh/kW.
Evidens: `round2/f_runtime/resolved_metadata.json` under recovery-roden i status.

F's gamle H/I/J-accept, manglende firestykkekonvergens og negative
numerikresultater bevares i det tematiske arkiv. G's rankmangel, baselineklipning,
ekstrapolation og empiriske bånd er fortsat begrænsninger, ikke skjult rettet af
filoprydningen. Nye baselinefamilier, støjlag og parameterbanker er forskning.
Den fælles F/G/WM-evalueringskampagne er eksplicit udskudt af Johan.
