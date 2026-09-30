# BM-reference — afsluttet F1–F3

Samlet faglig reference fra de tidligere F1/F2/F3-, audit- og performance-noter.
De fulde oprindelige tekster kan læses ved commit
`cb6ede2eedd6dc967013171462c40ce586fc6f99` i Git; der oprettes ikke ekstra arkivkopier.
Denne fil fastholder ligninger, numeriske kontrakter, bærende resultater og reproduktion.
Den er historisk reference, **ikke** en F4-statuslog. Se [F4.md](F4.md) for det aktuelle arbejde.

## 1. Model, enheder og F1-reference

Parameterlisten er `[U,f,g,C,k,Xi]`. Med fast effektskala er
$C=C_{phys}/P_{scale}$ en tidsstørrelse. State $X$, input $U,B$ og demand $D$ er
normaliserede. Den eksisterende model er

$$\delta(x,u)=\tanh\!\left(\tfrac{k}{2}[f(x)+g(u)]\right),\quad
D=B+\Xi\delta\begin{cases}1-B,&\delta\geq0,\\ B,&\delta<0.\end{cases}$$

$$\dot X=(D-B)/C,\qquad C(X_t-X_0)=\int_0^t(D_s-B_s)\,ds.$$

For $C,k>0$, $\Xi\in[0,1]$, $U,B\in[0,1]$ anvender F2 de ikke-voksende responser
$f(0)=g(0)=1$, $f(1)=g(1)=-1$. Endpoint-validering beviser ikke monotonitet/Lipschitz-
regularitet af vilkårlige callables. `stochastic=True` i den gamle derivative laver ikke støj.

For lineære responser $f(x)=1-2x$, $g(u)=1-2u$, konstant $u,B$ på et interval,
$q=1-u$ og $\gamma=1-B$ ved $x_0<q$, ellers $\gamma=B$, fås

$$X(t)=q+\frac1k\operatorname{arsinh}\!\left[
\sinh(k(x_0-q))e^{-k\Xi\gamma(t-t_0)/C}\right].$$

Det følger ved substitution $Y=\sinh(k(X-q))$, hvor $\dot Y=-k\Xi\gamma Y/C$.
Genstart ved hvert inputskift. Formlen gælder ikke vilkårlige PCHIP-/baselineforløb.
`solve_ivp`-wrapperen genstarter ved kendte breakpoints og evaluerer RHS ensidigt.
Outputtider er ikke det samme som solver-breakpoints.

Prisbyggeren er højre-kontinuert på hver fast intervalkant og bruger sidste pris ved
sluttidspunktet. Antal prisbeslutninger er uafhængigt af antal outputpunkter.
Den ældre F1-optimizer bruger bounded L-BFGS-B og sampled Simpson-cost; near-zero
recovery af `[0.72,0.28]` gælder det diskrete benchmark, ikke eksakt tracking mellem
samples. BM's opt-in midpoint-cost splitter derimod ved alle kendte U/B/A-spring.

### Feasibility før optimizerfortolkning

For lineære responser og $x\in[0,1]$ er

$$D_{min}(x)=B[1-\Xi\tanh(kx)],\qquad
D_{max}(x)=B+\Xi(1-B)\tanh(k(1-x)).$$

Et ønsket demand $d$ kræver

$$u=1-x-\frac1k\operatorname{artanh}(z),\quad
z=\frac{d-B}{\Xi\gamma},\qquad
\gamma=\begin{cases}1-B,&d\geq B,\\ B,&d<B.\end{cases}$$

Her kræves $0<B<1$, $\Xi>0$ og $|z|<1$; desuden skal den beregnede pris ligge i
$[0,1]$. Ellers er target infeasible uafhængigt af optimizerens flag.
For konstant target $A_0$ over varighed $S$ sættes

$$E_{excess}=\begin{cases}[(A_0-B)S-C(1-X_0)]_+,& A_0>B,\\
[(B-A_0)S-CX_0]_+,&A_0<B.\end{cases}$$

Energibalance og Cauchy–Schwarz giver $J_{det}\geq E_{excess}^2/S$.
Tilstrækkelig kapacitet er nødvendig, ikke tilstrækkelig; prisen kan stadig være umulig.

## 2. F2: SDE, FV og mixture

$$dX_t=a(t,X_t)dt+b(X_t)dW_t,\quad a=(D-B)/C,\quad b(x)=\sigma_Xx(1-x).$$

Det er en Itô-SDE. Diffusionsrutinen returnerer amplituden $b$, ikke $b^2$, $b/C$
eller $b\sqrt h$. Xi=0 fjerner demand-response og drift, men ikke processtøjen.
Inadrettet randdrift og forsvindende diffusion giver under passende regularitet
kontinuert invarians af [0,1]; et endeligt EM-trin kan stadig overskride randen.

Ved timer→dage: $C_d=C_h/24$, $\sigma_{X,d}=\sqrt{24}\sigma_{X,h}$, og alle
tidsargumenter/breakpoints ændres. Den stokastiske energibalance er

$$C(X_t-X_0)=\int_0^t(D_s-B_s)ds+C\int_0^t b(X_s)dW_s.$$

Sidste led har under integrabilitet middel nul, men er ikke nul på hver path.
Kontrolleret EM bruger

$$X_{n+1}=X_n+a(t_n,X_n)h_n+b(X_n)\Delta W_n,\quad \Delta W_n\sim N(0,h_n).$$

`boundary='none'` vælges eksplicit. Raw violations rapporteres. Samme max_step
sikrer ikke samme faktiske grid, når outputtider ændres. Til stærk konvergens bruges
koblede increments, hvor coarse increments er summen af fine increments.
Zero-noise giver Euler og energibalance ved samme venstre quadrature. Det uafhængige
logistic benchmark $X_t=\operatorname{sigmoid}(\operatorname{logit}x_0+\sigma W_t)$
har drift $\tfrac12\sigma^2x(1-x)(1-2x)$; det er ikke BM1's drift.
EM's forventede stærke orden 1/2 kræver de sædvanlige antagelser/asymptotiske regime.

### Probability-conserving FV

$$\partial_t p=-\partial_x(ap)+\tfrac12\partial_{xx}(b^2p)=-\partial_x\mathcal J,
\quad \mathcal J=ap-\partial_x(\tfrac12 b^2p).$$

Der differentieres på **hele produktet**. Med $m_i\approx\int_{V_i}p\,dx$ og
uniform bredde dx er $p_i=m_i/dx$. Indre rates er

$$r_i^+=\max(a_{i+1/2},0)/dx+b_i^2/(2dx^2),\quad
r_{i+1}^-=\max(-a_{i+1/2},0)/dx+b_{i+1}^2/(2dx^2).$$

$\dot m=Qm$, off-diagonal rates er ikke-negative, $\mathbf1^TQ=0$ og ingen
transitions forlader domænet. Implicit Euler $(I-hQ_n)m^{n+1}=m^n$ med venstre
koefficienttid bevarer derfor masse og positivitet i eksakt aritmetik (M-matrix).
Floating-point fejl overvåges, ikke renormaliseres væk. Stabilitet er ikke accuracy.

FV-reference bruger Thomas-solve. Fast FV bruger samme scheme med LAPACK `dgtsv`,
batchede koefficienter og valgfri cost-only lagring. Tidsevolutionen er stadig sekventiel.
Begge implementationer bevares som uafhængig kontrol. Scalar-time callbacks har en
separat, eksplicit vej; grids/model holdes faste i evaluatorens levetid.

### Samme initialfordeling og mixture-projektion

MC og FV sammenlignes fra samme Beta(20,20)-law: beta-samples i MC og CDF-integrerede
cellemasser i FV. Middel og varians alene definerer ikke samme law. De gamle point-
initialiserede/reflekterede demoer og legacy-FP's clipping er ikke referencekæden.

Den kontinuerte Bernstein-basis er $\phi_j=Beta(j+1,M-j+1)$ for j=0,...,M.
F3 bruger diskret normaliserede centerkomponenter; initial law projiceres eksplicit.
For mean mu har den kontinuerte familie variansgulvet

$$Var(X)=\frac{\mu(1-\mu)}{M+3}+\frac{M+2}{M+3}Var_c(\mu_j)
\geq\frac{\mu(1-\mu)}{M+3}.$$

Et glat plot eller små koefficientændringer beviser ikke korrekt repræsentation.
Den kontrollerede dynamiske mixture laver efter hvert internt trin

$$\widetilde p^{n+1}=\widehat p^n+hL_h\widehat p^n,\qquad
c^{n+1}=\arg\min_{c\geq0,\,1^Tc=1}\tfrac12\|\Phi c-\widetilde p^{n+1}\|_W^2.$$

Raw prediction kræver h gange største samlede outgoing-rate ≤CFL (normalt 0.4).
Ingen skjult clipping/masserepair. Mixture bruger center-gennemsnittet drift ved faces
og eksplicit stepping; FV bruger face-drift og implicit stepping. Deres forskel er
ikke ren projektionsfejl. Total-/best-fit-/dynamisk-gap-normer opfylder trekantsuligheden,
ikke eksakt additiv dekomposition. Ændrede outputtider ændrer også projektionsskemaet.

Med weighted designmatrix A er H=A^T A og kappa(H)=kappa(A)^2 i fuld rang.
Fuld Hessian afhænger af basis/grid, ikke density; aktive constraints kan ændre den
reducerede conditioning. SVD på den reducerede zero-sum basis undgår normal equations.
Primal masse/simplex og dualgap kontrolleres uafhængigt af solverens success-flag.
Fallback løser samme problem, ikke et regulariseret alternativ. Geometri-cache må
ikke genbruge en skjult target-afhængig start. Eksakt konveks L2-projektion er
nonexpansiv i density, selv om koefficienterne kan være følsomme.

## 3. F3: kontrolproblem og validerede forskelle

`BM1Parameters`, `BM1ControlProblem` og `BM1ControlNumerics` adskiller kendte
parametre, initial law, inputs og numerik. Observerede priser er read-only faste
inputs. Kandidatpriser gives eksplicit til evaluatoren; den neutrale template er ikke
observeret historie. `initial_mass_builder(edges)` gør prioren grid-uafhængig.

$$J(u)=\int_0^T E[(D_t-A(t))^2]dt+
\lambda_T E[(X_T-x_{ref})^2]+\lambda_U\sum_j(u_j-u_{j-1})^2.$$

For deterministisk A er tracking-integranden $(E[D]-A)^2+Var(D)$;
terminal-MSE inkluderer ligeledes variansen. Density-momenter beregnes med D(x), ikke
D(E[X]). MC bruger fulde path-costs og paired SE på forskelle mellem samme paths.
Midpoint-quadrature splitter alle kendte input/target-spring. Zero-noise alene fjerner
ikke initialusikkerhed. Lambda=0 giver det oprindelige objective; lambda er ikke
estimerede fysiske parametre. Sammenlign ikke weighted totals fra forskellige lambda-valg.

Single shooting og bounded Powell/multistart giver numeriske kandidater, ikke bevis
for globalt optimum. Konveks indre projektion gør ikke ydre OCP konvekst. Exact-price
cache ændrer ikke priser/tolerancer. MC validerer fastlagte policies, ikke optimizerkald.

### Bærende resultater (historiske, ikke genkørt ved oprydningen)

| Studie | Resultat / afgrænsning |
|---|---|
| F3-B | Adskilt rum/tid/quadrature og koblet EM-reference. Working grid er ikke en eksakt SDE-reference. |
| F3-C | Terminal-MSE og pris-hop-straf; nulvægte reproducerer baselinen. Landmark (0.03,0.003), ikke kalibrering. |
| F3-D | Deterministisk policy (0.72,0.28), stokastisk (0.730609,0.264415). FV-forbedring 1.263081e-4 (~1.82%); paired MC 1.423431e-4, SE 9.273e-6. |
| F3-E | Initial-only (0.723277,0.275548), process-only (0.722198,0.267527); alle fire policies krydsevalueret under samme fulde model og paired MC. Effekter er ikke additive. |
| F3-F før audit | Mixture (0.735904,0.263610); FV-801 optimum (0.732210,0.264413). Refined regret 2.33237524e-6 (~0.0356%); MC FV-minus-mixture -2.2142e-6, SE 6.340e-7. |
| F3-F efter SVD-audit | Mixture (0.735935,0.263606); refined regret 2.36820466e-6 (~0.03611%); paired MC -2.24879020e-6, SE 6.381e-7. Dette er en anden afrundet policy, ikke det samme tal som før audit. |
| F3-G | DK1-input via uændret F3-kontrakt; historik-baseline som default target. 2022-05-17: spot 0.0435152090, neutral 0.00741848718, alternativ 0.00562949463. Ingen kausale/økonomiske spareclaims. |
| F3-H | Numerisk closure-kilde fa38d10854f30bce6f334de3ff9ca514b8233711; dokumenteret 207/207 tests i Python 3.11 og GitHub Actions. |

F3-F's matched 81-cell design fjerner gridforskellen, ikke alle schemeforskelle.
Den tidlige working-grid FV/MC-afvigelse krævede high-resolution validering.
EM-labels 1/48,1/96,1/192,1/384 var **ønskede øvre grænser**; med den nestede
quadrature blev faktisk max-step 1/192,1/384,1/768,1/1536 (194/388/776/1552 trin).
Sidste gapændring ca. 3.83e-8 med SE 3.12e-9 er lille, ikke nul eller en rigorøs fejlgrænse.

F3-G bruger Jan–Oct 2022-sæsonmin/max retrospektivt. Baseline/standardtarget bruger
trimmet same-weekday historik fra fem tidligere uger; current-day consumption er
kun diagnostik. `--target-mode observed` er en eksplicit retrospektiv oracle.
De tre repræsentative dage er ikke et uafhængigt forecasting-testset. Spotpris er ikke
bevis for faktisk forbrugertarif. Faste referenceparametre er ikke estimeret på historien.

### Performance-læringen, ikke endnu en optimeringsplan

Den oprindelige 24-pris fast-FV Linux-test målte 11.65x/12.86x fixed-cost speedup på
201/1001 celler; fuld Powell 463.447→35.579 s med samme 4319 evalueringer og objective.
Senere Windows SVD-audit reducerede mixture-optimering ca. 651→32.1 s med 375/376
objective-kald og stort set samme policy. Det er miljø-/casespecifikke målinger,
ikke universelle hastighedsløfter. Den væsentlige læring er genbrug af geometri,
undgåelse af dårlige normal equations og unødvendig lagring — ikke grovere numerik.

## 4. Reproduktion og provenance

Kanonisk F3: T=[0,1] dag, f=1-2x, g=1-2u, C=1, k=2, Xi=1, sigma_x=0.5,
X0~Beta(20,20), to halve-dags priser. Target genereres fra det deterministiske
benchmark med (0.72,0.28), X0=0.5; scenario-builderen fastlægger B/A.
Working: Nx=201, q_max=h_max=1/96. Fin reference: Nx=801, q_max=1/192, h_max=1/384.
Mixture-design: degree30/Nx81/CFL0.4; held-policy kontroller degree30/38 på 201 celler.

| Studie | Initial-/Brownian-seed | Powell xtol / ftol / maxiter |
|---|---|---|
| B MC / koblet EM | 2061/2062; 2063/2064 | Fixed-policy, ingen optimering |
| D | 2071/2072 | 2e-5 / 1e-9 / 100 |
| E | 2081/2082 | 2e-5 / 1e-9 / 100 |
| F | 2091/2092 | FV: 2e-5 / 1e-9 / 100; mixture: 1e-3 / 1e-5 / 30 |
| G | 2091/2092 | 5e-5 / 1e-9 / 100 |

Fra repositoryroden; lange runs er særskilte fra unit-tests:

```powershell
python -B -m src.exe.run_model.bm_1.verify deterministic
python -B -m src.exe.run_model.bm_1.verify feasibility
python -B -m src.exe.run_model.bm_1.verify em-convergence
python -B -m src.exe.run_model.bm_1.verify mc-fv
python -B -m src.exe.run_model.bm_1.verify projection
python -B -m src.exe.run_model.bm_1.verify conditioning
python -B -m src.exe.run_model.bm_1.refine --mc-paths 20000 --mc-time-paths 5000
python -B -m src.exe.run_model.bm_1.optimize decision --mc-paths 20000
python -B -m src.exe.run_model.bm_1.optimize uncertainty --mc-paths 20000
python -B -m src.exe.run_model.bm_1.optimize mixture-control --mc-paths 20000
python -B -m src.exe.run_model.bm_1.optimize mixture-validate --fv-reference-prices 0.730609 0.264415 --fv-design-prices 0.729329 0.263873 --mixture-prices 0.735935 0.263606 --mc-paths 20000 --mc-time-paths 5000
python -B -m src.exe.run_model.bm_1.real_data --days 2022-05-17 --target-mode baseline --no-mc
```

Reproducer tidligere trykte resultater ved den relevante historiske commit; aktuelle
metoder kan give små ændringer. Retained evidens: `files/output/bm_1/f3_b/` og
`files/output/bm_1/real_data/`. Nye runs går til `files/output/bm_1/runs/` og overskriver
ikke evidens uden eksplicit outputvalg. Genskab detaljeret historik fx med
`git show cb6ede2:root/F3.md`; tilsvarende F1.md, F2.md og de to tidligere auditnoter.

F3-D er afgrænset til en ren same-problem sammenligning, ikke separat deterministisk
optimering i hver 2/4/6/12/24-dimension. F3-F validerer two-price-casen grundigt,
ikke generel ækvivalens ved feedback/højere dimension. F4 bruger det eksisterende
parameter-/initial-law-interface i stedet for en ny kontrolimplementation.

Kilder bag den oprindelige model/metode: Junker et al., Applied Energy 275 (2020),
115096, DOI 10.1016/j.apenergy.2020.115096; Higham, SIAM Review 43(3) (2001),
525–546, DOI 10.1137/S0036144500378302; projektets mixture-afhandling.
De eksakte særtilfælde, feasibility-bounds og diskrete kontrakter ovenfor er
projektets udledninger/implementationsvalg, ikke generelle udsagn fra disse kilder.
