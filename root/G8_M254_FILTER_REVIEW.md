# G8 — eget rekursivt filter med vejlederens Bernstein M254

**Efterfølgende forfining:** Eget M254513-filter består nu alle paired gates og faktisk resume. Se [det nye forfiningsreview](G8_M254_REFINEMENT_REVIEW.md). Det oprindelige 257-forsøg, budgetstop/admission og råresultater nedenfor er uændrede historik.

Dette er den bestilte opfølgning på det statiske M254-review. Vejlederens
basisfamilie er primær; legacy-M38/M78 og den lokaliserede 42-leds-reference
bevares. Statisk repræsentationsevne er ikke i sig selv filteraccept.

## Model og forsøgsafgrænsning

Samme gemte syntetiske G8-døgn, 24 timer, 22 punktobservationer, manglende
målinger ved timerne 6 og 17. Ingen nye historiske observationer, G4-fits,
parametergitre, prisoptimering eller baselinevalg. Tid måles i timer.
Fem faste rho-noder: 0.125, 0.1875, 0.25, 0.375, 0.5; uniform nodeprior.
Sand syntetisk rho er 0.25. Dette er ikke en parameterrecoverytest.

Den eksisterende model bruger

\[
\delta(x,u)=\tanh\{k[f(x)+g(u)]/2\},\qquad
D=B+\Xi\delta\{(1-B)\mathbf1_{\delta\ge0}+B\mathbf1_{\delta<0}\},
\]

\[
dX_t=\rho[D(X_t,t)-B(t)]dt+\sigma_xX_t(1-X_t)dW_t.
\]

Her er k=2, Xi=0.4, sigma_x=0.2, f(x)=1-2x, g(u)=1-2u,
B(t)=0.5+0.1 sin(2 pi t/24). Priserne er 0.3,0.7,0.3,0.7 over
[0,6),[6,12),[12,18),[18,24]. Punktmålinger bruger højre prisgrænse.
C=1/rho og C=4 for den kendte sande model.

SDE'en giver Fokker–Planck-ligningen

\[
\partial_t p=-\partial_x(v_\rho p)+\tfrac12\partial_{xx}(b^2p),
\quad v_\rho=\rho(D-B),\quad b=\sigma_xx(1-x).
\]

Faktoren 1/2 og kvadratet på diffusionsamplituden bevares. Der introduceres
hverken konstant diffusion, clipping eller en ændret fysisk model.
Den eksisterende conservative FV-flux med nul randflux fremskriver
**filterets egen** repræsenterede tæthed; FV-referenceposteriorer indlæses
alene efter kørslen til sammenligning.

## Den oprindelige basis og celleintegration

Vejlederens normaliserede basis er

\[
\ell_{m,M}(x)=(M+1)\binom M m x^m(1-x)^{M-m}
=\operatorname{Beta}(x;m+1,M-m+1),\quad m=0,\ldots,M.
\]

\[
p_M(x)=\sum_{m=0}^{M}c_m\ell_{m,M}(x),\qquad
c_m\ge0,\quad\sum_m c_m=1.
\]

Vi anvender M=254, alle 255 funktioner. Et aktivt sæt med mange nulvægte er
ikke beskæring af ordbogen: alle funktioner er tilgængelige i hver LS-solve.
Den nye numeriske repræsentation er eksplicit:

\[
B_{mi}=\frac1{\Delta x}\int_{e_i}^{e_{i+1}}\ell_{m,254}(x)dx,
\quad P_i=\Delta x\sum_m c_m B_{mi}.
\]

Beta-CDF og komplementær CDF beregner integralerne gennem den eksisterende
`beta_cell_average_basis`. Dermed har hver basisrække diskret masse 1 til
roundoff. Ingen efterfølgende rækkerenormalisering. Dette håndterer smalle
komponenter nær 0/1 bedre end midpoint-evaluering. Det er samme kontinuerte
familie, men en anden deklareret diskretisering; gamle midpoint-checkpoints
må ikke automatisk læses som nye M254-koefficienter. Legacy-gradargumentet
og dets gamle diskrete normalisering er bevaret.

M254 blev valgt i det allerede udførte review. Totalvarians giver

\[
\operatorname{Var}(X)=\frac{\mu(1-\mu)}{M+3}
+\frac{M+2}{M+3}\sum_m c_m(u_m-\mu)^2,\quad
u_m=\frac{m+1}{M+2}.
\]

Den centrale variansbund er 0.25/257=0.00097276, under halvdelen af den
mindste gemte FV513-varians 0.00212277. M38/M78 har for brede komponenter
for de smalle posteriorer. Variansargumentet er et udviklingsargument;
det beviser hverken minimal grad, god konditionering eller filterpræcision.
Det statiske review består på udvalgte eksponerede love, ikke på nye data.

## Eksakt oprindelig prior i nye koordinater

Beta(20,20) er l_(19,38). Den normaliserede gradforhøjelsesidentitet er

\[
\ell_{m,n}=\frac{n+1-m}{n+2}\ell_{m,n+1}
+\frac{m+1}{n+2}\ell_{m+1,n+1}.
\]

Den fås ved at gange den gamle polynomiumform med (1-x)+x og sammenholde
normaliseringskonstanterne. Koordinaterne opdateres derfor med

\[
c_j^{(n+1)}=\frac{n+1-j}{n+2}c_j^{(n)}
+\frac j{n+2}c_{j-1}^{(n)},
\]

hvor ugyldige indeks sættes til nul. De to bidrag fra hver gammel vægt
summer til samme vægt; ikke-negativitet og masse bevares. Gentag fra
n=38 til 253. Dette giver 217 ikke-nul vægte i en ordbog på 255 funktioner.
Vi starter hver node med disse koefficienter, ikke med et nyt LS-fit,
en FV-posterior eller den lokaliserede reference. Den repræsenterede celleprior
kontrolleres mod den oprindelige Beta(20,20)-celleprior.

## LS-projektion og numerisk stabilitet

Efter hver eksplicit FP-substep og hver observeret Bayes-opdatering løses
samme eksisterende begrænsede LS-problem:

\[
\min_{c\ge0,\;\mathbf1^Tc=1}\frac12\|Ac-q\|_2^2,
\quad A=\sqrt{\Delta x}B^T,\quad q_i=\sqrt{\Delta x}\,p_i^{\rm raw}.
\]

Dette er den diskrete L2-fejl for tætheden; den normaliserede posterior og
ikke de observerede demandværdier er LS-target. Simplex-kravene gør
resultatet til en sandsynlighedstæthed. Signed LS kunne passe bedre i L2,
men ville ikke opfylde sandsynlighedsmodellen. Vi tilføjer ingen ridge-straf:
den ville ændre det problem, hvis præcision vi vil vurdere.

Warm start bruger **samme nodes egne tidligere koefficienter**. Den fælles
solver bruger aktivt sæt og SVD i simplexens tangentkoordinater, med eksisterende
fallback og gap-kontrol; ingen nye ligninger, solver eller tolerancer.
Der er stadig meget dårlig konditionering ved M254. Koefficienterne kan være
numerisk mindre identificerbare end den tæthed, de repræsenterer.

For gradient g=A^T(Ac-q) er simplex-gap

\[
G=c^Tg-\min_m g_m\ge0,\qquad F(c)-F(c^*)\le G.
\]

Det følger af konveksitet og g^T c* >= min g_m. Den eksisterende skalerede
gap-gate er 1e-11 for den primære solver og 1e-9 for fallback. Gap beviser
optimeringsnøjagtighed, ikke at basisfamilien kan repræsentere target godt.
Derfor måler vi også masse-L1, moments, CDF, kvantiler, evidence og nodevægte.

Explicit CFL=.4 bevarer den eksisterende positive fluxopdaterings stepgrænse.
257 celler og max_step=1/96 er udgangspunktet; max_step er et loft,
ikke en påstand om 96 faktiske trin/time. Diffusion/CFL kan kræve mange
flere substeps og dermed mange flere LS-projektioner.

## Bayes-opdatering og parameterbank

For hvert punktmål y_r er likelihooden

\[
L_r(x)=\frac1{\sqrt{2\pi}\sigma_y}
\exp\{-\tfrac12[(y_r-D(x,t_r))/\sigma_y]^2\},\quad\sigma_y=0.03.
\]

Den eksisterende GL3-regel tager cellemiddel af likelihooden. Med nodens
predicted cellemasse P^-_(j,i) fås

\[
Z_{jr}=\sum_i L_{ri}P^-_{j,i},\quad
P^{\rm raw,+}_{j,i}=\frac{L_{ri}P^-_{j,i}}{Z_{jr}},\quad
w_j^+=\frac{w_j^- Z_{jr}}{\sum_k w_k^-Z_{kr}}.
\]

Derefter LS-projiceres hver normaliseret conditional posterior, og dens nye
koefficienter føres videre. Evidence beregnes før denne projektion;
approksimationen kan påvirke efterfølgende evidence, hvilket kontrolleres
separat. Ved manglende observation fremskrives tilstanden stadig, men
Bayes-opdatering og ændring af nodevægtene udelades. Marginalen er summen
sum_j w_j p_j; en gennemsnitlig rho eller MAP-node kan ikke erstatte banken
uden at miste state/parameter-sammenhængen.

## Kode, checkpoints, tidsbudget og kontroller

G-only ændringer: `bm_1_g8_bernstein.py`, basisvalg/prior/checkpoint i
`bm_1_g8.py`, runner `g8_bernstein.py`, to nye tests. 33 øvrige importerede
moduler matcher Git-blobbet på main c64198b; fælles FP/FV/LS/Bayes/EM-kode,
G4, F og WM er uændrede.

Den typed basisidentitet er `g8-bernstein-m254-cell-average-v1`. JSON indeholder
identiteten, det faktiske celleintegrerede basismatrix, grid, alle nodes egne
koefficienter/masser, nodevægte, cutoff, parametre, timer og pris-side.
Ved indlæsning af typed M254 verificeres matrix mod den deklarerede eksakte
cellebasis; resume kræver også match af model, noder og øvrige kontrakter.
Gamle origins uden det nye felt bevarer deres legacy-betydning, men kan
ikke genoptage den nye M254-vej. Et resume genskaber basis/cache, men
foretager ingen ny posteriorprojektion eller skjult state-reset.

Registrering sker før nye numeriske kontroller. Højst 300 nye sekunder samlet,
270 s beregningsdeadline og 30 s eksportreserve. De tidligere 9.596 s er
separat statisk evidens. Tre målrettede testmetoder er medregnet. Deres
korte faktisk JSON-resume kontrollerer præcis fortsættelse; der køres ingen
fuld regression eller nye historiske fits.

Piloten er de første to timer af den faktiske fem-node-kørsel, ikke en
separat kendt-parameter-kørsel. Efter to timer vurderes resttid fra den
største af de to målte timetider. Ved admission fortsætter samme filter,
egen posterior og solvercache uden genkørsel af pilotarbejdet.
Piloten har højst 60 s. POSIX SIGALRM kan afbryde et langt LS-kald med
TimeoutError, som ikke opsluges af solverens RuntimeError-fallback.
Det er en Linux/POSIX-runner; ingen lokal Windows-kørsel kræves nu.
Native bibliotekskald kan først levere signalet ved tilbagevenden; derfor
findes eksportreserve, checks og ingen automatisk timeout-retry.

Der skrives atomisk JSON efter hver færdig time. Afbrudt arbejde inden for
en time gemmes ikke som et færdigt fem-node-checkpoint. Referencekørsler
reproduceres ikke; deres gemte output sammenlignes efter eget replay.
En eventuel 513/1-192-forfining og fuld-case-resume er særskilte admissiontrin
inden for samme tidsbudget, aldrig implicit ekstra bevilling.

## Resultater og næste handling

Eget filter er komplet: 24/24 timer, fem egne conditional posteriorer og
22 observerede Bayes-opdateringer per node. Alle udførte gates består;
et positivt udviklingsresultat for 257-konfigurationen.

| Kontrol, maksimum over timer/noder | FV257 | FV513 | Grænse |
| --- | ---: | ---: | ---: |
| Conditional masse-L1 | 0.00735260 | 0.02958112 | 0.05 |
| Conditional state-mean | 0.00043340 | 0.00171520 | 0.005 |
| Conditional log-evidence | 0.00678134 | 0.01748515 | 0.05 |
| Marginal posterior masse-L1 | 0.00232778 | 0.01266852 | 0.05 |
| Marginal log-evidence | 0.00187502 | 0.00734487 | 0.05 |
| Nodevægt | 0.00057527 | 0.00145508 | 0.03 |

Alle prediction/posterior-moment-, CDF- og kvantilgates består også. Mod FV513
har demand-mean maksimum 0.00024351, demand-CDF 0.01113752 og demand-kvantil
0.00070834, mod grænser 0.005/0.03/0.02. Conditional massedefekt højst
7.44e-15, minimum conditional masse 2.06e-211. Manglende måling ændrer ikke
nodevægtene (præcis 0). Predictive RMSE mod den latente syntetiske demandbane
0.0212854 er ikke en historisk forecastscore.

Solveraudit: 51.420 afsluttede LS-projektioner (FP-substeps og Bayes),
155.595 active-face-factorizations og 51.073 cachehits. Maksimal cellemasse-L1
fra raw til projected law 0.000963871 mod 0.05; skaleret gap højst
9.99946e-12; koordinatmassedefekt højst 7.88e-15. Bayes-maxima fra den
uændrede projector stemmer med den transparente audit. Auditten ændrer
ikke solverens target, koefficienter, algoritme eller tolerancer.

Første to timer tog 17.096 s; piloten estimerede 246.276 s til resten og blev
admitted inden for samme beregningsdeadline. Et groft onlineestimat, ikke
målt totalruntime. Efter cachewarming blev kørslen hurtigere: filter 117.873 s,
invocation med tre tests og resultateksport 119.901 s. Tests 1.726 s.

**Fuld-case-resume er faktisk udført.** Efter hovedkørslen blev ét addendum
registreret inden for den oprindelige 300s-ramme. Eget gemt JSON-origin fra
time 22 blev indlæst og timer 23–24 kørt med egen fem-node-state og samme
M254/model. Masser, koefficienter, node-logvægte og log-evidence er præcis
identiske (maksimal forskel 0). Det tog 6.204 s. Samlet målt invocationtid
126.105 s før råarkivering. Et første shell-setupforsøg brugte en forkert
relativ skrive-sti og startede ingen Python/filterberegning; ingen numerisk
retry eller ekstra budgetramme. Den oprindelige hovedrapport ændres ikke;
addendumresultaterne står særskilt i post_review.json.

42-leds-referenceposteriorerne er alene genbrugt til sammenligning på samme
257-grid: conditional L1=0.00112423, marginal L1=0.00082663,
nodevægt=9.60e-6 og log-evidence=8.53e-5, alle maksimale forskelle.
Dette er metodeoverensstemmelse på eksponerede udviklingsdata, ikke ny
sandhed eller anledning til at erstatte den primære vejlederbasis.

**Ikke gennemført:** eget M254-replay på 513 celler med h=1/192.
Runnerens optimistiske admissionestimat inklusive faktor 2 og resume var
255.392 s yderligere; dette kunne ikke rummes i restbudgettet. Fordoblet
rumopløsning kan desuden kræve omkring fire gange så mange explicit
diffusion-substeps (CFL-step proportional dx²), så estimatet er ikke en
pålidelig runtimegaranti. Ingen ekstra kørsel startes for at måle dette.

full_filter_accept og mixture_to_G7_ready står derfor fortsat false:
den aftalte egne mixture-forfiningsgate er ikke afsluttet. Den positive
sammenligning med de allerede refined FV-referencer er ikke et eget
mixture 257→513-replay. Manglende forfining er ikke en målt præcisionsfejl.
Ingen historisk accept, parameterrecovery eller prisoptimering her.

**Næste:** ét særskilt budgetteret 513/1-192-replay og faktisk checkpointkontrol,
samme M254/model/tolerancer; Sol High er tilstrækkelig. En kort pilot skal
måle projektionstakten før fuld admission. 257-runtime er ikke et finruntime-
løfte. Ingen nye gradtal, større støj eller ridge-straf foreslås. Primær
mixture→G7 venter på dette; FV-handoff og 42-leds-reference bevares.
Ingen lokal kørsel kræves nu.

Repoets samlede rårapport er
`files/output/bm_1/runs/g8_m254_filter_20261006.json`. Råpakken bevarer begge
registreringer, frozen executed source, fixture/referencer, NPZ-historik,
alle timeorigins og det genoptagne origin. Kilde-/inputhashes og versioner
står i registreringen.


## Hvor kommer matematikken fra?

Vejlederens tavlefoto 07 er kilden til den oprindelige normaliserede
Bernstein/Beta-familie og mixture-summen. Fotoets identitet og det statiske
review findes i `G8_BERNSTEIN_REVIEW.md`. Vejlederens SDE/FP-tavlemateriale
er grundlaget for det stokastiske lag; den eksisterende model bevares.

Ritschel, *Mixture Approximations for the Fokker–Planck Equation*, slides 6
(ligning 4) og 8 (ligning 7) begrunder mixture og positiv LS med vægtsum 1.
Slide 9's særlige ortogonalitet antages ikke for denne Bernstein-matrix.
Slide 12 henviser til Nocedal/Wright §16.3 for constrained QP/active-set.
Deres metodegrundlag er ikke en påstand om god konditionering her.

Beta-identitet, totalvarians, gradforhøjelse, celleintegraler og gap-bound er
udledt ovenfor. Bayes-formlerne følger af betinget sandsynlighed og diskret
parameterprior. M254, GL3, grid/CFL/budget og checkpointtypen er konkrete
numeriske udviklingsvalg, ikke nye tavleforskrifter eller kildebestemte tal.
Ingen af disse valg dokumenterer historisk ugeforecast eller kausal prisrespons.
