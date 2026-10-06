# G6b — fælles FV/mixture/EM-panel, 6. oktober 2026

G6b's første afgrænsede syntetiske panel er gennemført. FV består begge
registrerede rum-/tidskontroller. Beta-mixture med M=38 ligger tæt på den
fineste FV-reference; M=20 afvises på initialprojektion, statevarians,
statekvantiler og demand-CDF. EM med 2048 fælles baner er forenelig med FV
under de registrerede praktiske tolerancer og punktvise samplinggrænser.
Dette er én dags numerisk udviklingsevidens, ikke historisk prognose-,
intervalkalibrerings- eller parameterrecovery-accept.

FV vælges som primær reference til næste syntetiske G8-statefilter. M=38
bevares som kontrolleret sammenligningsvej i denne forward-case, ikke som
universelt accepteret posteriorrepræsentation. Præcise posteriors kan være
smallere end Beta-basis kan repræsentere. G4's S3/lambda=0 og det negative
historiske G6a-resultat ændres ikke.

## 1. Model og frosset forsøg

Timer er kanoniske. Den allerede implementerede BM1-kernel bruges direkte:

$$dX_t=v(X_t,t)dt+b(X_t)dW_t,\qquad
v(x,t)=\frac{D(x,u(t),B(t))-B(t)}{C},\quad b(x)=\sigma_xx(1-x).$$

Driftsymbolet v er valgt her for at undgå forveksling med G4's kurveparameter a.

$$\delta(x,u)=\tanh\!\left(\frac{k}{2}[f(x)+g(u)]\right),$$

$$D=B+\Xi\delta\begin{cases}1-B,&\delta\geq0,\\B,&\delta<0.\end{cases}$$

| Felt | Registreret værdi |
|---|---|
| Horizon/rapportering | [0,24] timer; 25 hourly tidspunkter inklusive prioren |
| Model | f(x)=1−2x, g(u)=1−2u; C=4 timer, k=2, Xi=0.4 |
| Diffusion | sigma_x=0.2 per sqrt(time), konstant; ingen estimation |
| Baseline | B(t)=0.5+0.1 sin(2 pi t/24) |
| Pris | 0.3/0.7/0.3/0.7 på [0,6,12,18,24] |
| Trackingtarget | A(t)=0.5, syntetisk; ingen historiske målinger |
| Prior | Beta(20,20); E[X0]=0.5, Var(X0)=1/164 |
| FV | 65/129 celler ved h=1/24; 129 celler også h=1/48 |
| Mixture | 129 celler; grader 20/38 ved begge h-lofter; CFL=0.4 |
| EM | 256 pilotbaner, 2048 hovedbaner; faktiske skridt 1/48 og 1/96 timer |
| Seeds | Initialfordeling 20261006; Brownian-increments 20261007 |
| Budget | Pilot højst 60 s; hovedpanel højst 600 s; én case |

Begge EM-grids indeholder trackingkvadraturens midtpunkter. Der genereres én
2048-baners fin incrementmatrix; grovbaner fås ved at summere par, og piloten
bruger de første 256 baner. Ny generering med n_paths=256 ville med den
fælles generators arraylayout give andre baner, selv med samme seed.

### Tolerancer fastlagt før resultaterne

Maksimale hourly forskelle: state-/demandmiddel 0.005,
state-/demandvarians 0.002, CDF 0.03 og kvantiler 0.02 i normaliserede enheder.
Trackingtab må afvige 2 %. Initial cell-mass L1 må højst være 0.05.
Massefejl højst 1e-9; materiel negativ cellemassegrænse 1e-12.
Disse er praktiske udviklingstolerancer, ikke garantier for høj relativ
varianspræcision eller den ukendte kontinuumsløsning.

CDF-punkter er x=0.25/0.5/0.75 og D=0.4/0.5/0.6; kvantiler 5/50/95 %.
EM screens tilføjer tre punktvise standardfejl til moment-/CDF-/objective-
tolerancerne. Varians-SE bruger den empiriske indflydelsesvariabel
(X−mean)^2; kvantil-SE estimeres ikke, så kvantiler bruger den faste tolerance.
Tre SE er ikke en simultan confidence-claim over alle tidspunkter.

## 2. Hvorfor de tre numeriske veje?

### FV som sandsynlighedsreference

Den fremadrettede Fokker–Planck-ligning for standard Brownian motion er

$$\partial_t p=-\partial_x(vp)+\tfrac12\partial_{xx}(b^2p)
=-\partial_x j,\qquad j=vp-\tfrac12\partial_x(b^2p).$$

Ved x=0/1 bruges nul flux. BM1's drift peger indad og b er nul ved begge
rande i denne case. FV bruger den eksisterende konservative upwind-generator
Q og implicit Euler:

$$m^{n+1}=(I-hQ_n)^{-1}m^n.$$

Generatorens kolonnesummer er nul og off-diagonaler ikke-negative; dette
understøtter massebevarelse og positivitet. FV er valgt som reference, fordi
fælles solver allerede findes, og en smal initial/posteriorlov kan ligge på
cellerne uden mixture-basisens variansgulv. Upwind giver stadig numerisk
diffusion; to grids giver en praktisk kontrol, ikke en fejlgaranti.

Initial FV-masse er forskellen mellem Beta-CDF'er ved cellens kanter, ikke
bare sampling af en tæthed og efterfølgende skjult normalisering. Rapporterede
momenter bruger cellernes midtpunkter. Derfor er initial gridvarians
0.006102569 mod den kontinuerte 1/164=0.006097561. State-CDF/kvantiler
interpolerer lineært inden for celler; demand-CDF bruger vægtede celleværdier,
og demandkvantiler interpolerer deres sorterede kumulative vægte.

### Mixture: transport efterfulgt af sandsynlighedsprojektion

$$\widehat p_n(x)=\sum_{m=0}^M c_{n,m}\ell_m(x),\qquad
c_{n,m}\geq0,\quad\sum_m c_{n,m}=1,$$

hvor ell_m er Beta(m+1,M−m+1), diskret normaliseret efter det eksisterende
fælles interface. Først beregnes en eksplicit konservativ FV-fluxopdatering
på **mixturens egen** tæthed, q=p_hat+h L_h p_hat. Derefter løses

$$c_{n+1}=\arg\min_{c\geq0,\,1^Tc=1}
\frac12\sum_i\Delta x\left[\sum_m c_m\ell_m(x_i)-q_i\right]^2.$$

Dette er simplex-begrænset least squares, ikke den tidligere straf på G4's
kurveparameter. Begrænsningerne er nødvendige for sandsynlighedstolkningen.
Der bruges ingen ridge-straf, clipping, renormalisering eller restart fra
FV-referencefordelingen ved hvert trin. QP-kernelens SVD på aktive simplex-
flader bevares; normal equations ville kvadrere condition number.

Hvert accepteret trins endelige optimalitetscheck observeres uden en ekstra
QP. L1/L2 af q minus den projicerede tæthed og optimality gap gemmes.
Summen af lokale L1-tab er diagnostik, **ikke** en dokumenteret global fejlbound.
Antallet af observationer kontrolleres mod antallet af transporttrin.

M=38 vælges allerede i registreringen, fordi Beta(20,20) er basisled m=19.
M=20 er en billig grov reference, der forventeligt ikke kan gengive denne
prior præcist. Dette kan forsvares analytisk: med komponentmiddel mu_m gælder
for den kontinuerte Beta-basis

$$\operatorname{Var}(X)=
\frac{\mu(1-\mu)+(M+2)\operatorname{Var}_c(\mu_m)}{M+3}
\geq\frac{\mu(1-\mu)}{M+3}.$$

Ved mu=0.5 giver M=20 et gulv 0.010869565, væsentligt over priorens
0.006097561; M=38 giver netop 0.006097561. Den diskrete basis har små
kvadraturafvigelser, så det kontinuerte gulv mærkes som sådan i output.

Mixture har max internt h=0.009178 timer på grund af CFL, mindre end begge
nominale h-lofter. Der udføres 2904/3480 trin; ændrede outputknuder ændrer
subdivisionerne. De to nominelle niveauers næsten ens resultater er derfor
ikke et formelt bevis på en bestemt tidskonvergensorden.

### EM som uafhængig samplingkontrol

$$X_{n+1}=X_n+v(X_n,t_n)h+b(X_n)\Delta W_n,
\qquad\Delta W_n\sim N(0,h).$$

Den eksisterende EM-kernel får den samme initiallov, model og prisbreaks.
Fin/grov bruger fælles increments med DeltaW_grov=DeltaW_fin,1+DeltaW_fin,2.
Dermed kan tidsforskellen måles med parret SE frem for at blive domineret af
uafhængig samplingstøj. Boundary='none'; rå proposals registreres, og en
overskridelse stopper forsøget. Kontinuerlig randinvarians er ikke en garanti
for en eksplicit EM-diskretisering.

### Demand og objective integreres over loven

$$\mu_D(t)=\int D(x,t)p(x,t)dx,\qquad
V_D(t)=\int D(x,t)^2p(x,t)dx-\mu_D(t)^2,$$

$$J=\int_0^{24}\mathbb E[(D-A)^2]dt
=\int_0^{24}[(\mu_D-A)^2+V_D]dt.$$

Den fælles midpoint-kvadratur/objective bruges af alle tre veje. D er ikke
lineær; D(E[X]) er derfor ikke generelt E[D]. I den fine FV-kørsel er forskellen
op til 0.00394725. Det er netop en grund til at bruge hele fordelingen.

G6a sammenlignes separat som lokal reference med **samme initiale middel og
varians**: xbar0=0.5 og P0=1/164, ikke det historiske G6a-forsøgs punktstart P0=0.
Den bruger Pdot=2 v_x P+b(xbar)^2 og Var(D) cirka D_x^2 P.
Dens maksimale demandmiddelforskel fra FV er 0.00525123. Den er fortsat en
Gaussian/delta-approksimation, ikke evolution af hele Beta-prioren.

### Enheder

For dagtid s=t/24 skal C_days=C_hours/24 og
sigma_days=sqrt(24)*sigma_hours. Så v_days=24 v_hours og
b_days=sqrt(24)b_hours, mens J_hours=24 J_days for samme integrand.
Den fokuserede enhedstest kontrollerer disse adapterrelationer; den er ikke
en ekstra heldags solver-replay.

## 3. Resultater

Fineste FV: 129 celler, h=1/48; J=0.2386252699 normaliseret demand²·time.

| Metode | J | Maks. demandmiddelafvigelse fra FV | Maks. statevariansafvigelse | Initial L1 | Udfald |
|---|---:|---:|---:|---:|---|
| FV65, h1/24 | 0.2399150290 | 0.00056827 | 0.00080644 | 0 | praktisk refinement består |
| FV129, h1/24 | 0.2386651713 | 0.00002008 | 0.00002837 | 0 | tidskontrol består |
| Mix20, h1/48 | 0.2415675463 | 0.00113013 | 0.00476700 | 0.29166702 | afvist |
| Mix38, h1/48 | 0.2386779220 | 0.00010608 | 0.00006368 | 3.35e-8 | forenelig med FV i denne case |
| EM2048, h1/96 | 0.2376740757 | 0.00200799 | 0.00077824 | samplingprior | forenelig med FV med sampling-SE |

Mix20 har maksimal demand-CDF-forskel 0.07008523 >0.03 og statekvantil-
forskel 0.04325967 >0.02. Mindre tidsskridt reparerer ikke dette prior-/basisproblem.
M20→M38 er derfor ikke bestået gradkonvergens; det er en påvist grov basisfejl
og en finere metode, der stemmer med den uafhængige reference.

Mix38 har demand-CDF-forskel 0.00082798 og relativ J-afvigelse 0.0220648 %.
Største lokale projektionstab er L1=9.1130e-5; summen er 0.0291250.
Tilsvarende M20-tal er 1.15436e-4 og 0.1312567. Begge graders tidsniveauer
ligger tæt, men CFL-fortolkningen ovenfor gælder.

EM-fint J-SE er 0.001232934; forskellen til FV er 0.000951194, mindre end én SE.
Parret grov/fin J-forskel er 3.00910e-6 med SE 5.66953e-6. Maksimal parret
state-/demandmiddelforskel er 4.22646e-5 / 1.20306e-5. EM's state-CDF- og
demandkvantilafvigelser er 0.0232442 / 0.00549493. Nul randoverskridelser
på begge hovedgrids; rå proposals ligger samlet i [0.12826,0.86852].

FV og begge mixturegrader bevarer ikke-negativ masse med højeste massefejl
4.89e-15. Ingen clipping/renormalisering eller QP-fallback. M38's
projektionmatrix-condition number er cirka 2.10e11 og H's cirka 1.11e17;
feasible masser og optimality gaps er kontrolleret, men koefficienterne kan
være dårligt identificerede. Det er ikke en begrundelse for at kalde QP'en
eller dens parametre godt konditionerede.

### Runtime og kontrolomfang

V3-hovedpanelet tager 7.101 s; V3-invocation 8.840 s inklusive generering,
arkivering af inputs, EM-pilot, beregninger og råoutput. Den genbrugte
M20-pilot fra V1 tog separat 1.120 s; dens tid indgår ikke i V3-invocation.
V3's nye pilotarbejde tog 0.203 s. Der er ikke en fuldstændig samlet
runtimeopgørelse for de to fejlede interfaceinvocations eller den indledende
utimerede probe; de må ikke skjules som del af 8.840 s.

De fine FV/M38/EM-solverkald tog henholdsvis 0.692/1.360/0.638 s, én worker,
BLAS=1. Dette er ikke et hardwareuafhængigt throughputestimat. M38 har ingen
målt speedup over FV her, så FV forbliver primær reference.

To fokuserede testmetoder består (0.018 s): timer/dage-adapter og nøjagtig
indlejring/kobling af EM/kvadraturgrids. Efter den konkrete indeksfejl blev
den berørte test udvidet og de to korte metoder genkørt; ingen fuld suite.
Panelets egne sandsynligheds-, initial-, refinement- og MC-checks er særskilt
forsøgsevidens. Alle 28 registrerede runtimekilder er uændrede efter kørslen;
26 eksisterende importerede kilder matcher G-branchens Git-blobs.

## 4. Revisionsspor og råresultater

Der var først en utimeret interfaceprobe af M20. Den havde ingen fulde råarrays
og bruges ikke som registreret evidens. Derefter:

1. V1 låste manifest før sampling; M20 blev gemt komplet. EM-piloten stoppede
   på searchsorted, fordi kvadratur/integration konstruerede samme midpoint
   med få ulps forskel.
2. V2 bruger nærmeste **verificerede** uniform-gridknude, uden interpolation.
   EM blev beregnet og NPZ gemt, men JSON-skrivning stoppede på den legitime
   konfigurationsværdi max_step=inf. Scalar diagnostics var ikke bevaret.
3. V3 serialiserer den som strengen Infinity. Samme seeds, levels og tolerancer;
   V1's M20-resultat genbruges. Den korte EM-pilot gentages for at få komplet
   resultat/diagnostik. Der er ingen ny numerisk kandidat eller efterfølgende
   grid-/grad-/støjfamiliesøgning.

Kildesnapshots for alle tre invocationer er rekonstrueret og verificeret mod
deres oprindelige SHA256-manifester. Tilfældige inputfiler er byteidentiske og
arkiveres én gang med en deduplikeringsoversigt. Arkivet omfatter hourly
momenter, masser, mixturekoefficienter, projektionstab per trin, EM-banernes
hourly states/demand/pathobjectives, seeds, fine increments, originalkilder,
fejlede forsøg og figur. Komplette subhourly EM-statebaner er ikke gemt;
increments og initialstates gør dem reproducerbare.

Kode: `src/method/bm_1_distribution_panel.py` og
`src/exe/run_model/bm_1/distribution_panel.py`. Benchmark:
`files/output/bm_1/benchmarks/g6b_distribution_panel_20261006.json`.

Ny, særskilt invocation fra repositoryroden, efter behov:

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python -m src.exe.run_model.bm_1.distribution_panel --output files/output/bm_1/g6b_local_NEW
```

Der kræves ingen lokal kørsel for denne leverance. Den interne
`--reuse-interface-pilot` er afgrænset til den dokumenterede indeks-/JSON-
reparation; det er ikke generel resume af modificerede solverkilder.

## 5. Kilder og næste trin

- Alexander Elbæk Nielsen, *Approximation of the Fokker–Planck Equation by
  Mixture Distributions*, upload `10-6a727fe98ed8ea201bc32c21-1.pdf`:
  kapitel 4.2, (4.30) for standard FP, (4.34) og (4.42) for constrained
  projektion; trykte s.47–48, PDF s.55–56. Fortegnet i implementeringen
  følger FP-ligningen (4.30). Den konkrete FV-flux og CFL er repositoryets
  diskretisering af princippet, ikke en påstand om ordret kopieret metode.
- Higham (2001), upload `12-Higham2001_AnAlgorithmicIntroductionToNumericalSimulationOfStochasticDifferentialEquations-3.pdf`:
  afsnit 4, (4.3), PDF s.8 for EM; afsnit 5 for stærk/svag konvergens og
  koblede Brownian-increments. Ingen konvergensorden udledes fra dette panel.
- Ritschel, *Mixture approximations for the Fokker–Planck equation*, upload
  `13-MixtureApproximationsForTheFokkerPlanckEquation-1.pdf`, slides 6/8:
  mixture-residual og least squares. I denne implementering fastlægges
  standard Brownian-konventionen og faktoren 1/2 via SDE/FP ovenfor.
- G6a-kovariansreferencen: Lecture08 slide10 (20), og Alba Gonzalo Primos
  upload `09-MSC_AGP_FINAL-1.pdf`, afsnit6.1.2 (6.4)/(6.7).

Døgn, parametre, prior, grader, seeds og tolerancer er vores deklarerede
udviklingsdesign; kilderne er ikke empirisk støtte for disse fysiske værdier.

**Næste er G8:** først kendt-parameter-statefilter med punktobservationer og
separat målefejl; brug samme model og én registreret observationssekvens.
Kontrollér predictive score før opdatering, posterior efter opdatering,
manglende målinger og split/genoptagelse. Vurdér mixture-projektionstabet igen
på posteriors. Derefter den lille rho=1/C-bank og node-egne posteriors til G7.
Historisk timeenergi kræver sin egen observationsoperator; denne syntetiske
punktdemo løser ikke den kontrakt. Følg P5/P6's eksisterende 10-minutterslofter.
