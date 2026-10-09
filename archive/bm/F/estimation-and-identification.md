# F: betinget estimation, identifikation og beregningsgraf

<a id="h2-method-redesign"></a>
<a id="h2-compact-budgeted-search"></a>
<a id="f4-physical-optimizer-pilot"></a>
Dette er en samlet faglig fremstilling af bevaret udviklingsevidens. Tidligere arbejdsordrer og budgetter er historik, ikke nye kørselsordrer. Aktuel brug, accept og opgaver står i [BM-guiden](../../../docs/bm/README.md) og [status](../../../docs/bm/STATUS.txt). Originaler, hashes og genskabelse: [arkivindeks](../INDEX.md).

Historiske statusord som “åben”, “endnu ikke” og “Remaining scope” gælder den pågældende delundersøgelse 5.–8. oktober 2026. De beskriver ikke automatisk den aktuelle F4-kommando eller senere G-timeleverance.

Fysisk parameterfit, fælles kurvefit og online tilstandsestimation er
forskellige opgaver. Det historiske H2-forløb valgte *kurver først* med
faste nominelle fysiske/støjparametre. En hurtig punktestimator er ikke en
erstatning for en parameterbank eller den fulde interval-likelihood.

## Why an optimizer switch or ordinary least squares is insufficient

For the candidate parameters theta, the observation law is

\[
 A_i=\Delta_i^{-1}\int_{s_i}^{t_i}D_\theta(v,X_v)\,dv,
 \qquad Y_i=A_i+\epsilon_i,\quad \epsilon_i\sim N(0,\sigma_y^2).
\]

With the candidate's previous posterior, define the positive joint kernel
\(K_{i,\theta}(y)\) on endpoint state. Then

\[
 q_i=K_{i,\theta}(y_i)p_{i-1},\quad
 Z_i=\mathbf1^Tq_i,\quad p_i=q_i/Z_i,
 \qquad J(\theta)=-\sum_{i\in O}\log Z_i+\lambda R(\theta).
\]

Missing observations apply only the state transition and contribute no score.
Changing theta changes all subsequent posteriors. Optimizer candidates must
therefore replay the full TRAIN; caching another theta's posterior is invalid.

In general \(E[\phi_\sigma(y-A)]\ne\phi_\sigma(y-E[A])\). An ODE least-squares
fit or a Gaussian residual at the predicted mean would change this objective,
and even a Gaussian moment approximation has parameter-dependent variance
and a log-variance term. It is not ordinary unweighted least squares. A formal
sum-of-squares encoding of a shifted negative log likelihood would still need
the same expensive filter and would not produce a computational saving.

## Kurvefamilier, kriterium og accept
## Estimator and acceptance

The original stochastic BM1 demand map, f(x), g(u), diffusion, boundaries,
interval-average Gaussian observation law and full log-state propagation stay
in place. The original linear family remains the reference; the two- and
four-piece monotone families use the existing drop-softmax coordinates, with
2 and 6 free logits, respectively. No G/WM basis or estimator is substituted.

For every candidate theta and observed hour i, use

\[
 q_i=K_{i,\theta}(y_i)p_{i-1},\quad Z_i=\mathbf1^Tq_i,
 \quad p_i=q_i/Z_i,\qquad
 J(\theta)=-\sum_{i\in O_{TRAIN}}\log Z_i+2R(\theta).
\]

Missing hours use only the state transition and have no score. Changing theta
requires its own complete prior lead/TRAIN replay. Do not carry another
candidate's posterior or reset state daily. The same fixed penalty R sums
squared inner-knot departures from the linear curve; on 48 rather than 168
hours it has relatively greater influence. This is an explicit regularized
conditional fit, not evidence that the original seven-day estimate is retained.

Retain SciPy 1.17.0 COBYQA, bounds [-4,4], one linear start per nonlinear family,
initial/final trust radii 0.5/0.02, scale=False and no objective target. Allow
one shared linear TRAIN replay, at most 16 new two-piece and 32 new four-piece
pairs (library maxfev 17/33, including the cached linear start). No multistart,
automatic extension or replacement by the old coordinate panel. Preserve call
order and exact curve/input/source/environment identities for deterministic
cached replay. Count interrupted work and necessary state reconstruction too.

COBYQA's default interpolation set uses 2n+1 points; a call ceiling is not a
termination or convergence guarantee. Require finite valid results and the
declared trust-radius termination from both nonlinear fits. A cap, callback
stop, timeout or numerical failure leaves the fit incomplete; never return a
fabricated infinite objective or present its incumbent as a converged model.
Algorithmic local termination is not a global optimum or identification proof.

All three retained families continue into selection. Rank by unpenalized
sequential predictive Y log-score; record interval-mean RMSE/MAE on the same
observed hours. Run each retained family's own fine prior→TRAIN→selection
stream. Keep total-score difference <=0.05 **per declared window**, maximum
state-mean difference <=0.002 and endpoint mass L1 <=0.05, with the existing
conservative cell-overlap comparison. Do not scale the score tolerance by days.
All hourly mass, conditional and overlap guards remain 1e-10, 1e-8 and 1e-7.
An individual precision failure blocks acceptance. A ranking reversal or score
gap <=0.1 records numerical ambiguity; the existing tie rule prefers fewer
pieces without claiming resolved improvement. A linear outcome is valid only
after both nonlinear families have actually completed the required work.

Report price-bin counts, missingness, daily score components and posterior
state occupation by knot interval. These are limited coverage/identifiability
diagnostics, not independent-day bootstrap intervals or joint parameter recovery.
The broad profiles and population studies remain deferred as already agreed.


## Fysisk optimizerpilot: særskilt betinget udviklingsevidens
De følgende metoder/resultater stammer fra to allerede eksponerede syntetiske
træningsdøgn og to gamle checkdøgn med faste lineære kurver. FV401, h=1/768,
Gauss3 og Beta(20,20) blev brugt; det var ikke et historisk MED-fit.
## Metoder og budgetter

MLE minimerer den negative sum af sekventielle predictive log-evidences.
LS minimerer 0.5*sum(((Y-mu_predict)/0.03)^2), med faste ens vægte.
`mu_predict` er observationens middelværdi FØR den aktuelle måling assimileres.
Den beregnes med Gauss-kvadratur over den predictive cellelov, ikke ved at
indsætte observeret/sand state eller bruge efterfølgende posterior.

Der bruges `minimize(method='L-BFGS-B')` og `least_squares(method='trf')`.
Begge bruger to starter, 0.25 og 0.75 i normaliserede koordinater. Fysiske
responsparametre skaleres til en enhedsboks; støjkoordinater log-skaleres.
Differensskridt 1e-5, ftol 1e-7 og gtol 1e-5 er erklærede pilotindstillinger,
ikke dokumenteret optimale defaults. LS har desuden xtol 1e-7.

Samlet loft per metode er 41 kandidatberegninger ved én parameter, 81 ved to.
Første/anden start får 20/21 eller 40/41; ubrugte første-budgetter overføres
ikke. Loftet tæller også numeriske gradienters faktiske TRAIN-kald og kan
ikke overskrides af SciPy's interne nfev-konvention. Identiske kandidater
genbruger deres egen eksakte score inden for en metode; intet deles mellem
metoder. Én kandidat betyder to komplette uafhængige døgn-replays.

De enkeltdimensionale grids bruger F4-E's 41 noder. Parrene bruger ni noder
per akse over samme støtte (hver femte node fra 41-punktsaksen). Dette 9x9-grid
er en særskilt billig udviklingsreference, IKKE det fulde G2-grid eller en
ændring af G2's eksisterende design. Gridet giver både MLE- og LS-reference
fra samme beregninger. LS bruges ikke til at estimere støj i piloten.

En fælles scorer beregner både likelihood og middelværdier i alle metoder.
Tiderne omfatter kandidat-checkpoints. Et specialiseret likelihood-only-kald
kan være billigere; tabellen er ikke et benchmark af den hurtigste mulige bank.
Samlet tidsloft var 600 s, kontrolleret mellem kandidater/efterkontroller.
Designet blev ikke tilpasset efter resultaterne. Checkpoint gemmes efter hver
kandidat; piloten har ingen resume-CLI og overskriver ikke en eksisterende mappe.

## Målte resultater

Hele piloten, inklusive efterkontroller og data-generering, tog **72.04 s**.
Fit-tider nedenfor udelader efterkontroller. Kandidattal inkluderer begge starter.

| Frie parametre | Grid: kandidater / sekunder | Direkte MLE | Direkte LS |
|---|---:|---:|---:|
| rho = 1/C | 41 / 4.78 | 14 / 1.69 | 20 / 2.34 |
| k | 41 / 5.42 | 35 / 4.38 | 22 / 2.49 |
| rho, Xi | 81 / 10.22 | 66 / 8.20 | 66 / 8.20 |
| sigma_x, sigma_y | 81 / 10.31 | 54 / 6.28 | Ikke forsøgt |

Der blev evalueret **521 forskellige TRAIN-kandidater** fordelt på 11
metodekørsler: 1042 døgn-replays. De 14 valgte resultater (inklusive gridets
separate MLE/LS-valg) fik hver to checkreplays og seks refinementreplays:
112 yderligere døgn-replays, i alt **1154** eksklusive syntetisk datagenerering.

13 af 14 optimizerstarter meldte success. Anden start i k-MLE ramte sit
21-kandidatbudget; første start konvergerede. Budgetstoppet bevares, og
best-evaluated er ikke i sig selv en stationaritets- eller globalitetsgaranti.

| Frie parametre | MLE-parametre | LS-parametre |
|---|---|---|
| rho | 0.997012 | 0.954394 |
| k | 1.989874 | 2.013139 |
| rho, Xi | 1.002807, 0.992264 | 0.954395, 1.000000 (afrundet) |
| sigma_x, sigma_y | 0.455891, 0.021387 | Ikke forsøgt |

Genererende værdier er rho=1, k=2, Xi=1, sigma_x=0.5, sigma_y=0.03.
Der vælges ikke metode efter nærhed til truth. LS og MLE optimerer forskellige
kriterier, så parametrene forventes ikke at være identiske.

På de to separate gamle checkforløb var one-step Y-RMSE:

| Frie parametre | Grid-MLE | Direkte MLE | Direkte LS |
|---|---:|---:|---:|
| rho | 0.043409 | 0.043408 | 0.043390 |
| k | 0.043409 | 0.043536 | 0.043252 |
| rho, Xi | 0.043409 | 0.043625 | 0.043390 |
| sigma_x, sigma_y | 0.043908 | 0.043671 | Ikke forsøgt |

Forskellene er små og to allerede sete checkforløb kan ikke vælge en generel
vinder. Det er observation-Y-fejl, ikke fejl mod støjfri demand, og one-step
prediction med løbende assimilation, ikke åben ugeforecast. For støjparret
blev TRAIN-likelihood bedre end gridet, men check-likelihood blev dårligere:
check-NLL -81.565 mod -81.804. Hurtigere/lavere TRAIN-objective er ikke
automatisk bedre generalisering.

## Numerik og verifikation

Hver valgt kandidat blev kontrolleret separat med halv tidsstørrelse, 801
celler og Gauss5 på træningsforløbene. De fulde forskelle er gemt per kandidat.
Disse er fixed-candidate diagnoser, ikke accept af hele optimizerboksen.
Især rho-MLE's forbedring over gridet var kun 9.67e-6 i base-NLL og skiftede
fortegn ved 801 celler: her var gridpunktet bedre med ca. 1.61e-5.
Resultatet understøtter færre beregninger til et sammenligneligt punktestimat,
ikke en påstand om højere fysisk/numerisk præcision.

Fire fokuserede tests består: scoreparitet med eksisterende filter;
ingen brug af aktuel Y til dens egen prediction; eksakt Gaussian Xi=0 samt
missing-mask; parametertransformationer; begge starter og stramt budget
inklusive gradientarbejde. Flere egenskaber dækkes af samme testmetode.

Råreview genberegnede scores og LS-cost med long-double fra alle 521 gemte
kandidater, kontrollerede udvalg, budgetter, TRAIN/check-ID'er, probability-
diagnoser og 129 sourcehashes. Maksimal objectiveforskel 5.68e-14;
maksimal registreret massefejl 4.15e-14. Ingen nye FV-replays i reviewet.
Dette er målrettet pilotverifikation; fuld repo-regression er ikke kørt lokalt.

## Hvorfor en komplet jobgraf er nødvendig
For hver ny kurve kræves dens egen prior→TRAIN-propagation. Fine-grid-
kontroller, MODEL_SELECTION, åben forecast, intervalmomenter/CDF og en
adskilt final prequential-filtergren er egentlige jobs. Et call-loft er ikke
konvergens, og en lavere median unit-rate finansierer ikke ukendte jobs.

De oprindelige 73 koordinatkandidater og senere 17 parallelle kandidater
var forskellige historiske forslag. Native COBYQA havde i den reviderede
scope højst 49 forskellige kurvepar: én fælles lineær,16 nye tostykkede og32
nye firestykkede; kandidater blev kørt sekventielt, med frekvensarbejde fordelt
inden for kandidaten. Derfor er ceil(kandidater/8) ingen gyldig ny prisning.

Den betingede 8.76 h-illustration for48/24/168 scope var aldrig en målt
runtime-admission. Senere faktiske historiske rates krævede egne budgetvalg.
Det lukkede H-budget og det ufuldstændige finalforløb står i
[valg og overdragelse](h2-selection-and-handoff.md); gamle charges nulstilles ikke.

F's nødvendige native weekday-tilslutning kræver sammenhængende state gennem
mellemperioderne, egen baseline/prior og en genbrugelig TRAIN-likelihood.
Den generiske kalender leverer maskerne; den faste F4-model udfører ikke dette fit.
Ingen sådan historisk fitkampagne startes af dokumentoprydningen.

## Den faste regularisering

Preserve the existing drop-softmax parameterization and endpoints. Two-piece
f/g has two independent logits; four-piece has six. Use one linear start per
family, bounds [-4,4], fixed physical/noise/prior/baseline inputs, and

\[
 R=\sum_{q\in\{f,g\}}\sum_{j\text{ inner}}
       (q(z_j)-(1-2z_j))^2,\qquad\lambda=2.
\]

This is a fixed design regularizer, equivalent to a quadratic preference with
scale 0.5 on each inner value (within the monotone family), not a calibrated
Bayesian uncertainty analysis. It is chosen before historical observations;
there is no lambda grid and TRAIN does not select its value. More inner values
receive more total regularization under the existing sum convention.

## Den oprindelige eksplicitte jobgraf

Dette er den historiske beregningsidentitet, ingen ny budgettilladelse.

## Complete remaining-work graph

Use one 8-worker pool with BLAS=1; no second optimizer pool and no division of
candidate counts by eight. Scalar moments/prior lead may use one worker.
The following conservative graph assumes all declared data hours are observed,
three distinct retained families and a nonlinear winner distinct from linear.
Rates are maximum end-to-end measured times for each actual job type/grid,
including normal IPC/checkpoint/export work. Apply the factor 1.5 once.

| Job | Base units | Fine units | Rate type |
|---|---:|---:|---|
| 49 full 48-hour TRAIN candidates | 2352 | 0 | observed likelihood |
| Each candidate's 24-hour prediction lead | 1176 | 0 | neutral propagation |
| Three 24-hour selection continuations | 72 | 0 | observed likelihood |
| Three retained TRAIN+selection precision streams | 0 | 216 | observed likelihood |
| Their fine prior leads | 0 | 72 | neutral propagation |
| Final prequential winner and linear | 336 | 336 | observed likelihood |
| Selection/final pre-update interval moments | 408 | 408 | real moments |
| Open-week hourly interval moments | 336 | 336 | real moments |
| Seven open distribution horizons, two curves | 14 | 14 | global CF/CDF |

TRAIN candidates do not need moments for optimization. Retained checkpoints
supply filter states before each scored hour for the separate mean/variance
pass. Fine trajectories keep their own states; they never initialize from a
coarse posterior. Open-week states are computed by the moment recursion and
its full log-amplitude, with all saved forecast horizons coming from that path.
Matching completed work is removed from future counts only after raw identity
validation. Count failed/interrupted attempts, discarded pilot conditions and
optimizer reconstruction as actual charged work.

Let b_g,p_g,m_g,c_g be the four measured rates at grid g. With N=49,T=48,S=24,
H=168,L=24,K=7, the priced future work before auxiliary jobs is

\[
 1.5\{(NT+3S+2H)b_b+[3(T+S)+2H]b_f
       +NLp_b+3Lp_f+(3S+4H)(m_b+m_f)+2K(c_b+c_f)\}.
\]

Add the latest shared ledger, all unfinished auxiliary work and 60 seconds for
orderly stop. Reserve 600 seconds for the explicitly itemized auxiliary jobs:
300 for new focused controls/nonreusable historical rate checks, preparation,
imports/startup; 300 for final diagnostics/export/handoff. These allocations
are unmeasured caps, not evidence that the jobs complete within them. The
implementation must list and price each subjob; excess work fails admission.
Required work cannot disappear into an assumed zero or be relabeled optional.

Reserve the **entire** remaining validation graph plus its 300 seconds and
stop allowance before search can spend time. Retain the minimum 7200-second
reserve, but increase it whenever the priced validation is larger. Under the
illustration it is **14431.28 seconds, about 4.01 hours**, so the old two-hour
reserve is insufficient. Do not add the reserve twice to the total: it is a
protected allocation within the same work graph and ledger.



Den tidligere samlede admission krævede
$$T_{charged}+T_{startup}+1.5(N_b\tau_b+N_f\tau_f)+T_{other}+T_{stop}\le36000.$$
Her er N antal base/fine jobs, tau målte kompatible rates, other øvrige
obligatoriske opgaver og stop stopreserven. Ukendte led sættes ikke til nul.
Dette var det daværende 10 h-studieloft; det er ingen ny adgang til lange kørsler.

## Konkrete kode- og kontrolindgange

[fælles kurvefit](../../../src/validation/bm_1/estimation.py), [fysisk optimizerpilot](../../../src/validation/bm_1/parameter_optimization.py), [optimizer-tests](../../../tests/test_bm1_parameter_optimization.py). Historiske CLI- og controllerkommandoer findes kun i det restaurerede originaltræ; disse links peger på de bevarede matematiske komponenter. Tests er links til tidligere eller løbende kontroller, ikke påstået genkørt ved denne dokumentændring.
