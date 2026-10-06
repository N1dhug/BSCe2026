# G8 — punktfilter, rho-bank og posterior til kontrol

6. oktober 2026. Kørt på G-parent a0556d1831858b67e36412e711b0c2f7edde04a2.
Nyeste main ved review: b3ddbe2ca7d9d8b8f3acd23f2acfa03dc73b51a8.
Main har samtidige F/WM-ændringer; ingen G-kildeoverlap i sammenligningen.
Alle 30 importerede eksisterende Python-filer matcher parentens Git-blobs.
Fælles numeriske kerner er uændrede. Kun G-adapter, runner, tre tests og G-dokumentation tilføjes.

## Beslutning og grænser

Den afgrænsede syntetiske integration er gennemført. FV-statefilter, fem-node-bank,
manglende målinger, faktisk checkpoint/resume og fuld-bank-objektiv består egne kontroller.
Mixture M38 og M78 gennemfører uden numerisk exception, men begge afvises af de
uændrede fordelings-/projektionskrav. G8's mixture-gren er derfor ikke accepteret.
Et godt G6b-forward-resultat er ikke tilstrækkeligt efter skarpe måleopdateringer.

FV bruges som reference og kan levere et deklareret posterior til senere G7.
Før mixture→G7 accepteres, skal posterior-repræsentationen løses i en separat lille
opgave. Ingen ekstra grad-/støj-/grid-søgning er kørt. Det brede G8-metodekrav er
ikke markeret afsluttet; historisk intervaloperator, parameteridentifikation og G7
prisstyring mangler. G4's S3/lambda=0 og negative G6a-intervalkonklusion er uændrede.

## Registreret forsøg og valg

En ny 24-timers syntetisk sandhedsbane med C=4, k=2, Xi=.4, sigma_x=.2,
f(x)=1-2x og g(u)=1-2u. Baseline B(t)=.5+.1 sin(2 pi t/24), priser
.3/.7/.3/.7 på 0–6/6–12/12–18/18–24 timer og prior Beta(20,20).
Én ny bane vælges ved seeds, ikke ved fitkvalitet: initial 20261011,
Brownsk støj 20261012 og målefejl 20261013. Målefejl sigma_y=.03 er fast,
separat fra processtøjen; den er ikke estimeret eller gjort større for mixture.

Timesamples 1,...,24, med manglende y ved 6 og 17. Prisen ved måletid er
højregrænsen. Y er en øjeblikkelig normaliseret demandmåling; ikke timeenergi.
FV129/h=1/48 sammenlignes med FV257/h=1/96: fælles rum-/tidsforfining,
ikke særskilt kontinuumsfejlgaranti. Mixture: 129 celler, M38/M78, h=1/48,
CFL=.4. Begge grader blev registreret før observationerne blev genereret.
Likelihood integreres med tre Gauss–Legendre-punkter per celle.

Banken har rho=1/C i [.125,.1875,.25,.375,.5] time^-1, uniform prior og
alle øvrige parametre faste. Sand rho=.25 ligger i banken. Dette er en
integrationstest, ikke recovery på ukendt model eller et parameterfit.

Truth har h=1/192, med koblet h=1/96 ved summation af Brownsk increments.
Nul rå randoverskridelser; maksimal koblet hourly state/demand-forskel .000636408
består den registrerede .001-grænse. Ingen clipping eller refleksion.

Registrerede tolerancer: mean .005, variance .002, CDF .03, kvantiler .02,
posterior-L1 .05, Bayes-projektion-L1 .05, per-måling log-evidence .05,
nodevægt .03, checkpoint 1e-10 og masse 1e-9.
Alle gates bevares også ved negative resultater.

## Matematik: forudsigelse før den nye måling

BM-kernen er

\[
dX_t=\rho[D(X_t,u_t,t)-B(t)]\,dt+
 \sigma_x X_t(1-X_t)\,dW_t,\qquad \rho=1/C.
\]

Responsen er delta=2/(1+exp[-k(f(x)+g(u))])-1 og
D=B+Xi delta[(1-B)1_{delta>=0}+B1_{delta<0}].
Eksisterende kode bruges til både drift og output; sigma_x er diffusionens
amplitude. Solverens Brownsk increment leverer sqrt(dt). Målefejl får ingen sqrt(dt).

Ved node j er b_j=rho_j(D-B) og s(x)=sigma_x x(1-x).
Før y_k kommer, viderefører FP/FV den tidligere posterior:

\[
\partial_t p_j=-\partial_x(b_jp_j)+\tfrac12\partial_{xx}(s^2p_j),
\qquad p^-_{j,k}=\mathcal T_{\rho_j}(t_k,t_{k-1})p^+_{j,k-1}.
\]

Vi viderefører hele fordelingen, fordi D er nonlinear og variance/CDF skal bevares.
E[D(X)] kan ikke generelt erstattes af D(E[X]).
FP bruger nul randflux. FV er den allerede kontrollerede konservative reference,
så der kopieres ingen ny solver.

## Matematik: Bayes og parameterbank

Målemodellen er

\[
Y_k=D(X_{t_k},u_{t_k},t_k)+\varepsilon_k,\quad
\varepsilon_k\sim N(0,\sigma_y^2).
\]

En normaliseret posterior betyder sandsynlighedsmassen efter målingen;
prediktionen er fordelingen før målingen.
Med likelihood L_k(x)=phi_{sigma_y}(y_k-D(x,u,t_k)):

\[
Z_{j,k}=\int L_k(x)p^-_{j,k}(x)\,dx,\qquad
p^+_{j,k}(x)=\frac{L_k(x)p^-_{j,k}(x)}{Z_{j,k}}.
\]

I FV bruges cellens gennemsnitlige likelihood, numerisk med Gauss–Legendre3,
og summen Z_j=sum_i L_i m^-_{j,i}. Dette er Bayes-normalisering; ingen
efterfølgende clipping, kunstig randmasse eller solverreparation.

Bankens vægte opdateres også med Bayes:

\[
w^+_{j,k}=
\frac{w^+_{j,k-1}Z_{j,k}}
{\sum_l w^+_{l,k-1}Z_{l,k}}.
\]

Log-vægte og logsumexp forebygger underflow.
Den predictive score er log(sum_j w_old,j Z_j), beregnet FØR assimilation.
Posteriorens fit til samme y må ikke rapporteres som en forecastscore.
Ved missing propagateres hver node; posterior=prediction og vægte uændrede.
Ingen likelihood anvendes; den manglende score gemmes som NaN, ikke som en observation.

Den fælles posterior er sum_j w_j delta_{rho_j} p_j(x).
Hver node ejer SIN conditional law. En fælles state og en gennemsnitlig rho
giver ikke generelt samme videreførsel. Adapteren kalder den offentlige
run_parameter_bank én måling ad gangen, fordi den API kun returnerer de
sidste node-conditionals. De komplette nodehistories bevares i G8-arkivet.

## Resultater

| Metode | Demand-RMSE før / efter måling | Sum predictive log-evidence |
|---|---:|---:|
| Åben FV129 uden assimilation, genbrugt G6b | .0286866 / samme | 41.63581 |
| Kendte parametre, FV257 | .0208264 / .0162508 | 43.96035 |
| Rho-bank, FV257 | .0212822 / .0164031 | 43.70025 |

RMSE måles mod den syntetiske latente demand over alle 24 reporttider.
Log-evidence summerer kun de 22 målte tider. Før-måling RMSE er en
enkelt-sample udviklingsdiagnose, ikke en uges forecast-accept.
Kendt-parameter-filteret reducerer denne RMSE 27.4002 % mod samme åbne model.
Banken koster lidt præcision her, hvor sand parameter faktisk er kendt til reference;
det er ikke grund til at ignorere parameterusikkerhed i et ukendt system.

FV-forfining består: maksimal posterior-L1 .0255431 (kendt) / .0252735 (bank);
maks. log-scoreforskel .0135374 / .0173852; maks. bankvægtforskel .00346144.
Massefejl højst 5.56e-15 på tværs af alle cases, ingen negativ masse.
Missing-masseændring og missing-logvægtændring er præcis nul.

Slutvægte i FV257: [.249777,.272685,.259411,.158334,.059792].
MAP=.1875, ikke sand .25; entropy=1.51110 mod uniform 1.60944.
Banken har kun svagt reduceret usikkerheden. Vi påstår ikke identificeret C.

| Mixture mod FV257 | Max posterior-L1 | Max Bayes-projektion-L1 | Resultat |
|---|---:|---:|---|
| Kendt M38 | .498924 | .317498 | Afvist |
| Kendt M78 | .139169 | .0922605 | Afvist |
| Bank M38 | .500165 | .363114 | Afvist |
| Bank M78 | .138551 | .128495 | Afvist |

M78's største posterior-demandmean-forskel er .00145385 kendt /
.00142479 bank, begge under .005. Det gør ikke hele fordelingen korrekt:
CDF-/L1-gates fejler, og bankens scoreforskel .0544347 overstiger .05.
Derfor kan lav RMSE eller et vellykket QP alene ikke acceptere mixture.

For kontinuerte, normaliserede basisled Beta(m+1,M-m+1) har ledmean
mu_m=(m+1)/(M+2) og ledvariance mu_m(1-mu_m)/(M+3).
Totalvariance giver for en positiv blanding

\[
\operatorname{Var}(X)=
\frac{\mu(1-\mu)}{M+3}+
\frac{M+2}{M+3}\operatorname{Var}_w(\mu_m)
\geq \frac{\mu(1-\mu)}{M+3}.
\]

Omkring mu=.5 er gulvet .00609756 ved M38 og .00308642 ved M78.
FV's mindste posterior-statevariance er .00244681. Discretely normalized basis
har lidt andre tal; bounden er en forklaring på repræsentationsbegrænsningen,
ikke et eksakt diskret fejlbevis. Observeret M38/M78-minimum .00561291/.00303985
understøtter problemet. En skarp posterior kan ikke altid repræsenteres af denne
positive basis. At hæve graden øger også ill-conditioning; derfor ingen automatisk
grad-optrapning. Næste afgrænsede opgave bør begrunde posterior-repræsentationen,
f.eks. eksplicit FV-fallback eller en lokalt tilpasset positiv basis, før et nyt run.

## Faktisk resume og handoff til G7

JSON-cutoff ved time12 indeholder parametre, rho-værdier, normaliserede logvægte,
hver nodes masser og for mixture også egne koefficienter og den normaliserede basis.
Arrays kopieres og fryses. Resume læser JSON ind i nye objekter/projector og
fortsætter ved time13. FV257 og M78 matcher uafbrudt kørsel med præcis nul
forskel i masser, logvægte, koefficienter og evidence. Det beviser continuation,
ikke M78's numeriske korrekthed. Der genprojecteres ikke ved resume.

Ved cutoff12 og pris .3 på [12,13] evalueres kun et fast kontrolforslag:

\[
J(u)=\sum_j w_j E_j\!\int_{12}^{13}[D_t-A(t)]^2dt
=\int[(\bar D-A)^2+
\sum_jw_jV_j+\sum_jw_j(\mu_j-\bar D)^2]dt.
\]

Det sidste led er variance mellem parameterhypoteser. Der bruges nodens egen
posterior og C=1/rho_j i det eksisterende FV-kontrolinterface.
Vægtene er faste frem over den åbne horisont; ingen fremtidig observation
assimileres, ingen rho-genfit eller prisoptimizer køres.

Fuld-bank J=.00557276728, bias=.00503890622,
inden-nodevariance=.000514587943 og mellem-nodevariance=.0000192731159.
MAP's egne state/rho giver J=.00595314283. MAP er en særskilt approksimation;
den er ikke etiketten for fuld-bank-kontrol. Vægtet sum af node-J og direkte
bankmoment-objektiv kontrolleres med 1e-10-grænse.
Tre nye korte tests dækker én-nodeækvivalens/missing/chronology, to-nodeobjektiv/
owned posterior og JSON-mixture-resumevalidering.

## Kilder, kode og reproducérbarhed

Alba Gonzalo Primo, upload 09-MSC_AGP_FINAL-1.pdf, afsnit6.1.1–6.1.2,
ligninger6.1–6.2, PDF-side102 / trykt90: kontinuert SDE, diskrete nonlinear
demandmålinger med separat Gaussian målefejl og prediction/correction.
Alba anvender EKF; vores fulde FP/FV-Bayesfilter og diskrete rho-bank er egne
implementeringsvalg. Vi kopierer ikke EKF's meanlinearisation eller stateclipping.
Lecture08's prediction/update giver samme princip, ikke vores konkrete bankalgoritme.
Mixture-upload 13-MixtureApproximationsForTheFokkerPlanckEquation-1.pdf
underbygger FP-mixture-approksimation; den beviser ikke at den valgte basis
kan repræsentere denne posterior. Bayes- og totalvariance-ligningerne ovenfor
er udledningen, ikke påståede ordrette formler fra uploads.

Kode: src/method/bm_1_g8.py, src/exe/run_model/bm_1/g8.py,
tests/test_bm1_g8.py. Genbrug: bm_1_estimation, PreparedBetaProjector,
BM1ForecastOrigin og eksisterende stochastic_tracking_objective.
Tre målrettede tests består (.030 s unittest; .90756 s invocation).
Ingen fuld suite eller nye historiske fits. Syntakskontrol af nye filer består.

V2: pilot2.490 s, hovedforsøg46.622 s, fuld invocation49.112 s;
caps60/600 s, single worker/BLAS1. Handoff .511 s er inkluderet.
Første forsøg stoppede ved one-path EM-outputshape før filter eller gemt fixture.
Rettet udelukkende med np.atleast_2d ved aflæsning af eksisterende output.
Første registrering og rekonstruerbar første source bevares; samlet timing
inklusive dette fejlede forsøg er ikke målt.

Råarkiv omfatter registrering før RNG, fixture/increments/seeds, egne nodehistories,
pre-update evidence, basis/koefficienter, cutoff/resume, åbent genbrugt G6b-reference,
sourcehashes/frosne kilder, assessment og review. Ingen personlig måledata.

Kør kun ved et nyt erklæret forsøg; vælg en frisk outputmappe:
python -m src.exe.run_model.bm_1.g8 --output <ny-mappe> --open-reference <G6b-fv_129_h48.npz>

Ingen lokal beregning kræves nu. G7-optimering og historisk ugeforecast er
separate næste leverancer; mixture-blokeringen forbliver eksplicit.

