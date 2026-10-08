# F: tofamilievalg, slutdiagnose og handoff

700 åbne prognosejobs blev frosset, men kun 345 af 1344 planlagte slutjobs er gemt i den seneste beskrevne status. H/I/J og populationskalibrering er fortsat åbne; det kørbare F4-eksempel er en særskilt leverance.

Status: historisk evidens. Aktuel brug og scope står i [BM-guiden](../../../root/BM_GUIDE.md).

Afsnittene nedenfor samler resultater og begrænsninger uden de gentagne arbejdsordrer. Oprindelige matematisk væsentlige afsnit er bevaret særskilt. Fuld originaltekst, kildehash og erstatningssti findes via [stikortet](../old-paths.json) og den verificerede kildepakke.

<a id="f4-h2-two-family-prefinal-20261008"></a>
## F4_H2_TWO_FAMILY_PREFINAL_20261008.md

Original: `root/F4_H2_TWO_FAMILY_PREFINAL_20261008.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-h2-two-family-prefinal-20261008--f4-h2-godkendt-afgrænsning-modelvalg-og-frosset-ugeprognose"></a>
<a id="f4-h2-two-family-prefinal-20261008--modelvalg-og-betydning"></a>
<a id="f4-h2-two-family-prefinal-20261008--faktisk-implementering-og-verifikation"></a>
<a id="f4-h2-two-family-prefinal-20261008--holdbare-artefakter-og-regnskab"></a>
<a id="f4-h2-two-family-prefinal-20261008--resterende"></a>

Den eksplicit afgrænsede lineær/tostykke-selection valgte tostykkemodellen på begge grids: selection-logscore base23.2678879187 mod lineær13.7893997938, normaliseret RMSE .0721944742 mod .0860703965. Alle700 open-jobs og168 h prognose blev frosset før slut-Y. JSON-repræsentationsfejl, watchdog og hele600 s-reservationen bevares.

<a id="f4-h2-two-family-final-progress-20261008"></a>
## F4_H2_TWO_FAMILY_FINAL_PROGRESS_20261008.md

Original: `root/F4_H2_TWO_FAMILY_FINAL_PROGRESS_20261008.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-h2-two-family-final-progress-20261008--f4-h2-faktisk-slutvurdering-og-næste-admissiongrænse"></a>
<a id="f4-h2-two-family-final-progress-20261008--slutvurderingens-aktuelle-resultat"></a>
<a id="f4-h2-two-family-final-progress-20261008--regnskab-og-konkret-pending-forslag"></a>
<a id="f4-h2-two-family-final-progress-20261008--fortsat-faglig-status"></a>

Tre finalprequential blokke gav123/124/98 nye jobs:345/1344 planlagte. Lineær/base168 h er komplet, øvrige model/grid-strømme og fælles finalmetrics ufuldstændige. V8-ledgeren er H49805.8944273059/72000 s og native1136.465941387039/1140 s;3.534058612961 s er under næste8 s eksportreserve. Det særskilte V4-forslag var approved=false/execution_started=false.

<a id="f4-h2-ij-result-review-20261007"></a>
## F4_H2_IJ_RESULT_REVIEW_20261007.md

Original: `root/F4_H2_IJ_RESULT_REVIEW_20261007.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-h2-ij-result-review-20261007--f4-h2--ij-resultatreview-7-oktober-2026"></a>
<a id="f4-h2-ij-result-review-20261007--den-konkrete-opgave"></a>
<a id="f4-h2-ij-result-review-20261007--matematik-og-kode"></a>
<a id="f4-h2-ij-result-review-20261007--i-handoff-og-finale"></a>
<a id="f4-h2-ij-result-review-20261007--faktiske-resultater-og-afslutning"></a>

Reviewet forbinder den betingede interval-likelihood, fælles monotone f/g-kurver og egne stateforløb med implementationen. Tostykkefit var konvergeret/valideret, men den samlede H/I/J-leverance var fortsat åben. Den oprindelige matematikdel er bevaret nedenfor; de historiske statusgentagelser findes i originalkilden.

### Bevaret fagligt afsnit fra originalen

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

<a id="f4-i-implementation-preparation-20261008"></a>
## F4_I_IMPLEMENTATION_PREPARATION_20261008.md

Original: `root/F4_I_IMPLEMENTATION_PREPARATION_20261008.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-i-implementation-preparation-20261008--faktisk-hi-efter-den-kontrollerede-fortsættelse"></a>
<a id="f4-i-implementation-preparation-20261008--i1-faktisk-origin-frem-for-en-ny-demonstration"></a>
<a id="f4-i-implementation-preparation-20261008--i2-forecastkontrakter-der-skal-holdes-adskilt"></a>
<a id="f4-i-implementation-preparation-20261008--i3-eksisterende-f3-evaluator-med-faktisk-h-belief"></a>
<a id="f4-i-implementation-preparation-20261008--j-og-afslutning"></a>

H→I kræver den faktisk frosne H-origin med egen posterior og cutoff. Nyt Beta-startforløb, prequential diagnose og åben prognose er forskellige opgaver; ingen ommærkning. De fulde konkrete handoff-kontrakter er bevaret nedenfor, fordi de er nødvendig faglig viden.

### Bevaret fagligt afsnit fra originalen

## I1: faktisk origin frem for en ny demonstration

`root/f4/review_h2_i_handoff.py` er et read-only review af det faktiske
`conditional_handoff.npz` og `forecast_frozen.npz`. Det kontrollerer checksum,
manifest, selectionafhængighed og informationscutoff. For hver eksporteret
model og hvert grid går den originale posterior gennem den eksisterende
`as_forecast_origin`/`BM1ForecastOrigin`-bro. Tidspunktet er dag4, svarende til
21.februar2022 kl.00:00 Copenhagen;168h slutter ved dag11.

Reviewet kræver præcis egen log-state/masse, samme f/g og nominelle fysiske
parametre. Den konservative cellelov overføres til401/801 celler med positiv
masse og1e-10 massetolerance. Original log-state og originalt grid bevares i
handoffet. Det er ingen propagation, nyt fit eller populationskontrol.
Koden er syntaxkontrolleret, men reviewet er endnu ikke udført: H-handoffet
findes først efter komplet selection og valideret forecastfreeze.

## I2: forecastkontrakter, der skal holdes adskilt

H's open-week-prognose er for timegennemsnit af demand og Gaussian interval-Y,
fra den faktiske frosne origin med fastlagte B/U og nul fremtidig assimilation.
Den har egne base/fine-forløb og CDF/kvantiler ved1/6/12/24/48/72/168h.
De råresultater genbruges som betinget intervalforecastevidens.

Den eksisterende `forecast_fv` giver derimod endpoint-state og øjeblikkelig
demand. En selvstændig I-backendkontrol skal bruge samme frosne origins og B/U,
men sammenligne endpoint-state med H's gemte endpoint-state. Interval-demand
må ikke sammenlignes direkte med øjeblikkelig demand. H's sigma_y er deklareret
for normaliseret timegennemsnit; `forecast_fv`'s gamle point-Y-varians må ikke
genbruges som kalibreret interval-Y eller som en ny målerstøjmodel.

En fuld selvstændig I-kørsel kræver egen source-/origindigest, atomiske
checkpoints, konkret runtimebudget og numerical reference før launch.
Første afgrænsede målescope er to eksporterede origins, ét open-week-forløb
per origin/grid og få faste F3-policies. Eksakte antal afhænger af H's valgte
model; vælges lineær, udføres den identiske model ikke to gange.
Der er endnu ingen ny numerical I-kørsel eller runtime-admission.

## I3: eksisterende F3-evaluator med faktisk H-belief

Genbrug `origin.control_problem` og F3's objective, constraints og bounded
optimizer. Begge beliefs vurderer de samme faste policies, med samme B/A,
prisskala, startcutoff og horizon; ingen nye målinger under hypotetisk forward.
Efter en kort måling kan et lille optimizerbudget prissættes, og hver valgt
policy får egen fine-vurdering. Ingen skjult genfit eller gennemsnit af optima.

Den lineære H-reference er en nominelt betinget reference, ikke kendt fysisk
truth. Sammenligning med den valgte fitted kurve kan derfor vise betingede
policy- og belief-forskelle, men ikke fysisk parameter-recovery eller faktiske
økonomiske besparelser. Eksisterende kendt/rho-MAP-evidens bevares separat.

## J og afslutning

Samlet rapport skal forbinde A–G's allerede afsluttede evidens med H's actual
resultat, I1-origin, intervalforecast, selvstændig I-backend/F3-kontrol og alle
negative fund. Parameter-/inputusikkerhed og ukendt tidligere slutugeeksponering
mærkes eksplicit. En betinget udviklingsuge er ikke populationskalibrering;
mindst200 uafhængige sekvenser kræves fortsat ved sådanne claims. Et interface-
review eller en bestået pilot afslutter ikke alene I/J/F4.

<a id="f4-remaining-plan-20261008"></a>
## F4_REMAINING_PLAN_20261008.md

Original: `root/F4_REMAINING_PLAN_20261008.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-remaining-plan-20261008--f4-h2-slutvurdering-gemt-i-tre-blokke-eksportgrænse-nået-8-oktober-2026"></a>
<a id="f4-remaining-plan-20261008--f4-h2-godkendt-tofamilie-modelvalg-og-ugeprognose-frosset-8-oktober"></a>
<a id="f4-remaining-plan-20261008--f4-h2-kandidat49-fuldt-sikret-maxfev-stop-8-oktober-2026"></a>
<a id="f4-remaining-plan-20261008--f4-h2-takeover-sikret-importkontrol-kræver-mere-plads-8-oktober"></a>
<a id="f4-remaining-plan-20261008--f4-h2-48-kaldstrace-terminalt-mappetab-8-oktober-2026"></a>
<a id="f4-remaining-plan-20261008--f4-h2-normalt-budgetstop43-kald-sikret-8-oktober"></a>
<a id="f4-remaining-plan-20261008--f4-h2-ny-frist-godkendt-separat-fortsættelse-startet"></a>
<a id="f4-remaining-plan-20261008--aktuel-restplan-for-hele-f4--8-oktober-2026"></a>
<a id="f4-remaining-plan-20261008--én-rækkefølge"></a>
<a id="f4-remaining-plan-20261008--næste-udførbare-trin-er-klart"></a>
<a id="f4-remaining-plan-20261008--beregningstid-og-konkret-beslutning"></a>
<a id="f4-remaining-plan-20261008--i-og-j-efter-h"></a>

Den gentagne restplan er afløst som aktiv arbejdsordre. Seneste beskrevne H2-status er tofamilievalg og345/1344 finalprequential jobs, med utilstrækkelig native-eksportreserve. Faktisk H→I-originreview, selvstændig I/F3-admission og J er åbne. F4_RUN er en separat kørbar forsøgsindgang med frisk erklæret Beta-prior, ikke den gamle H-cutoff-posterior. Ingen automatisk genoptagelse af H/I/J.
