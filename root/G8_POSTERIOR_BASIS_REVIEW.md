# G8 — posterior-basis: udledning, én kandidat og målrettet forfining

6. oktober 2026. G-parent: e77aaafdf2613fa2e846f458386145ce121ae33b.
Dette er en fortsættelse af G8, adskilt fra F4-H2 og WM.
G8's gamle M38/M78-resultater og alle negative gates bevares.

## Beslutning

Brug den nye faste, celleintegrerede Beta-basis med 42 komponenter og
257 FV-celler som udgangspunkt for G7's afgrænsede syntetiske integration.
513-cellerskontrollen består både for bankens marginal og hver nodes egen
conditional law. Det oprindelige 129-cellers grid accepteres ikke til den
strengere conditional-node-kontrakt.

Den store Bayes-projektionsfejl er løst på den eksponerede G8-case. Basisvalget
er udviklet efter at have set denne case, så resultatet er numerisk
udviklingsevidens, ikke uafhængig historisk validering, generel basisgaranti
eller parameterrecovery. G7's prisoptimering er ikke kørt.

FV forbliver primær reference. Den nye mixture er fortsat FV-transport med
positiv LS-projektion ved hvert internt trin; det er ikke en gridfri metode.
I denne case er FV hurtigere. Mixture indgår som den krævede
fordelingsapproksimation, ikke som dokumenteret runtimeforbedring over FV.

## Hvorfor den gamle basis fejlede

Den gamle grad-M-basis var Beta(m+1,M-m+1), m=0,...,M.
Antal komponenter og koncentration var dermed koblet:
K=M+1, kappa=alpha+beta=M+2.
For et Beta-led med mean mu_m og koncentration kappa gælder

\[
E_m[X]=\mu_m,\qquad V_m=\frac{\mu_m(1-\mu_m)}{\kappa+1}.
\]

For en positiv mixture med samme kappa og samlet mean mu giver totalvariance

\[
V=\sum_m c_m V_m+\sum_m c_m(\mu_m-\mu)^2
 =\frac{\mu(1-\mu)}{\kappa+1}
  +\frac{\kappa}{\kappa+1}\operatorname{Var}_c(\mu_m)
 \geq\frac{\mu(1-\mu)}{\kappa+1}.
\]

Ved mu=.5 kan M38 ikke komme under .00609756 og M78 ikke under .00308642.
Den gemte FV257-bank når conditional variance .00216612 ved time16/rho=.5,
mean .510071. Der er derfor en konkret repræsentationsbegrænsning. En bedre
QP-løsning, længere fit eller mere præcis normalisering kan ikke fjerne denne
kontinuerte basisgrænse. Celleapproksimationen har lidt andre momenter; de
kontinuerte grænser bruges som designargument, ikke som et eksakt diskret bevis.

## Den valgte mindste ændring

Vi adskiller placering og bredde:

\[
\mu_r=\frac{r+1/2}{41},\quad r=0,\ldots,40,\qquad
\kappa=256,\qquad
\alpha_r=\kappa\mu_r,\quad \beta_r=\kappa(1-\mu_r).
\]

Dertil tilføjes det oprindelige Beta(20,20)-led. Det giver 42 komponenter
og inkluderer initialloven direkte. Basis er fælles og fast for alle tider
og rho-noder. Kun de positive mixture-vægte ændres.

Kappa=256 er den mindste potens af to, for hvilken det centrale variancegulv
.25/(kappa+1) ligger under halvdelen af den mindste gemte FV-nodevariance.
Det nye gulv er .000972763. Denne regel er eksplicit data-informeret af den
allerede eksponerede udviklingscase. 41 jævnt fordelte means giver afstand
1/41=.02439, mindre end den centrale komponent-standardafvigelse .03119.
Det giver overlappende komponenter med omtrent samme antal koefficienter som
M38. Ingen efterfølgende koncentrations-, antal- eller placeringssøgning køres.

Beta(20,20) er bredere end kappa256-leddene. Derfor har hele ordbogen stadig
V >= mu(1-mu)/257 i den kontinuerte fortolkning; lighed kræver passende
komponentmean. At sænke dette gulv er nødvendigt her, men er ikke alene en
garanti for god repræsentation. Den efterfølgende fulde filterkontrol er nødvendig.

Cellebasen er

\[
B_{mi}=\frac{F_m(e_{i+1})-F_m(e_i)}{\Delta x}.
\]

Her er F_m Beta-fordelingens CDF. Den eksisterende
beta_cell_average_basis bruger complementary CDF i højre hale for at
reducere cancellation. Alle komponenter har ikke-negativ enhedsmasse på
griddet; der anvendes ingen posterior-clipping eller normaliseringsreparation.
Priorens cellemasser repræsenteres af første led til roundoff.

## Least squares, sandsynlighed og bevaret model

Det uændrede projektionsproblem er

\[
c^\star=\arg\min_{c\geq0,\ \mathbf1^Tc=1}
 \frac12\sum_i\Delta x
 \left[\sum_m c_m B_{mi}-p_i^{\rm target}\right]^2.
\]

Målet er enten FV-trinnets forudsagte tæthed eller den rå Bayes-posterior.
LS bestemmer her repræsentationsvægtene c, ikke G4's fysiske parametre eller
straffen på a. G4's S3/lambda=0 ændres ikke.

Positivitet og enhedsmasse følger af basis og simplex-constraints.
Den eksisterende active-set-solver bruger SVD på den aktive simplexflade,
ikke normal equations til løsningen. Hessianen dannes til diagnostik.
En uafhængig optimality-gap-kontrol kontrollerer LS-løsningen; særskilte
L1-, mean-, CDF-, variance-, quantile- og scoregates kontrollerer kvaliteten.

Basis-matricens condition number er stadig stor: ca. 3.92e11 ved 257 celler,
mod ca. 7.00e15 for gammel M78 ved129. Derfor påstås ikke perfekt konditionering.
Den præcise prior er tæt på en kombination af de smalle led, og koefficienter
behøver ikke være robuste, selv når den repræsenterede tæthed er det.
Ingen ridge-straf eller signed mixture indføres. De to kendt-parameterreplays
havde nul QP-fallbacks i alle dynamiske trin; alle registrerede Bayes-projektioner
i de nye banks havde nul QP-fallbacks.

Fysisk drift, sigma_x=.2, sigma_y=.03, punktlikelihood GL3, priser, baseline,
rho-noder, missingmask og seeds er uændrede. Hver node viderefører egne
koefficienter og egen conditional state; ingen FV-reference bruges som restart.

## Alternativer og hvorfor dette valg

- Højere Bernstein-grad: kan sænke gulvet, men øger både antallet af
  koefficienter og næsten lineær afhængighed. Den nye basis adskiller de to valg.
- Adaptiv Beta-basis efter hver måling: mulig, men kræver transport mellem
  forskellige baser, nye checkpointkontrakter og flere adaptive beslutninger.
  Den faste basis kunne afprøves med eksisterende interfaces.
- Større målefejl: ville gøre posterioren bredere ved at ændre den statistiske
  model; det løser ikke repræsentationen af den oprindeligt deklarerede model.
- Signed koefficienter: kan forbedre approximation, men sikrer ikke en gyldig
  sandsynlighedstæthed, som Bayes og næste propagation kræver.
- FV-fallback: den pålidelige mulighed ved fejlede gates. Den skal betegnes FV,
  og erstatter ikke en dokumenteret mixture-demonstration. Den var ikke nødvendig
  for at fuldføre denne nye fine-grid mixture-kontrol.

## Registrering, negativt coarse-resultat og målrettet follow-up

Første registrering låser én basis og grids129/h=1/48 samt257/h=1/96 på den
gemte 24-timers G8-fixture. Pilot er 1.176 s og består. Alle oprindelige
marginal-/projektionsgates består, men en NY, strengere kontrol af hver
rho-nodes conditional law afslører coarse-grid-følsomhed.

| Conditional posterior, maksimal L1 | Forskel | Grænse | Resultat |
|---|---:|---:|---|
| Gemt FV129 → FV257 | .0655479 | .05 | Fejler |
| Ny mixture129 → mixture257 | .0584180 | .05 | Fejler |
| Ny mixture129 → FV129 | .0146962 | .05 | Består |
| Ny mixture257 → FV257 | .00729929 | .05 | Består |

G8's gamle FV-accept omfattede marginalgates, ikke denne nye nodegate.
Den skal derfor ikke beskrives som en tidligere kontrol af hver node.
Første nye studies samlede acceptfelt er false og ændres ikke.

Den observerede fejl begrunder én særskilt registreret follow-up:
præcis FV513 og mixture513 med h=1/192, samme basis og fixture. 257-resultater
genbruges. Højst300 nye sekunder og600 sekunder samlet til begge numeriske
invocations. Ingen yderligere grids eller ny basis, hvis kontrollen fejler.

| Conditional posterior | Maks. L1 | Maks. node-logscoreforskel |
|---|---:|---:|
| FV257 → FV513 | .0365687 | .0239552 |
| Mixture257 → mixture513 | .0328833 | .0203220 |
| Mixture257 → FV513 | .0296379 | .0174841 |
| Mixture513 → FV513 | .00348644 | .00309870 |

Alle består L1<=.05, node-mean<=.005 og node-score<=.05.
Alle oprindelige marginalgates består også, inklusive CDF/quantile/momenter,
pre-update evidence og bankvægte. Dette er praktiske fælles rum-/tidskontroller,
ikke en kontinuumsfejlgaranti.

## Hvor meget bedre er repræsentationen?

| Metode og grid | Max bank-marginal posterior-L1 mod FV257 | Max Bayes-projektion-L1 |
|---|---:|---:|
| Gammel M78 / 129 celler | .138551 | .128495 |
| Ny42 / 129 celler | .0239013 | .00149368 |
| Ny42 / 257 celler | .00230264 | .00155602 |

Den store reduktion i projektionsfejl ses allerede ved samme grid129.
Den yderligere forbedring i anden kolonne ved257 inkluderer gridforfiningen;
hele forbedringen til257 tilskrives derfor ikke basisændringen alene.

Ved513 er mixture/FV marginal-L1 .00132938, maksimal logscoreforskel .00076811,
bankvægtforskel .000272535 og Bayes-projektion-L1 .00156262.
Demand-RMSE før måling er .02132239 mod FV's .02132126.
Mixture reproducerer dermed referenceberegningen tæt; dette er ikke en ny
fysisk demand-model eller dokumenteret forecastgevinst over FV.

Massefejl på de nye replays højst2.34e-15, ingen negative masser.
Missing bevarer predicted/posterior-masser og logvægte præcis.
Faktisk JSON-resume vedtime12 på mixture257 giver præcis nul forskel i
conditional masses, koefficienter, logvægte og evidence frem tiltime24.
To nye fokuserede tests består på .079 s (invocation .673 s).
Ingen fuld suite eller historisk genfit.

Første study42.666 s, follow-up107.420 s, i alt150.086 s til numeriske invocations.
Heraf nye banks: mixture25720.094 s, FV51316.936 s, mixture51389.621 s;
257-resume10.484 s er inkluderet i første study.
De gamle G8-replays blev genbrugt, ikke kørt igen.

## Kilder og præcis tilknytning

Tobias K. S. Ritschel, upload13:
Mixture approximations for the Fokker-Planck equation, 28.september2025,
slide6 ligning4: generel mixture sum c_m ell_m;
slide8 ligning7a–7c: least squares med positive koefficienter og sum1.
Det er direkte grundlaget for at ændre de faste basisfunktioner og bevare
det samme constrained LS-problem. Noterne kræver ikke vores gamle kobling
mellem antal komponenter og Beta-koncentration.

Den konkrete Beta-ordbog, kappa256, cell-integral-repræsentation og
variancegulv-udledningen er vores metodevalg/udledning; de tilskrives ikke
noterne som færdig opskrift. De ikke-ortogonale basisfunktioner kræver vores
generelle LS-solver; noternes særlige ortogonalitetsforenkling bruges ikke.
For standard Brownsk SDE beholder BM-kernen FP-leddet
(1/2) partial_xx(s^2 p); slides' viste diffusionnotation ændrer ikke denne faktor.

Alba Gonzalo Primo, upload09, afsnit6.1.1–6.1.2, ligninger6.1–6.2,
PDF102/trykt90, understøtter separat proces-/målemodel og prediction/correction.
Vores fulde Bayes-densityfilter er ikke hendes EKF.
Alle kildeudsagn bygger på de uploadede dokumenter.

## Implementering, reproduktion og konkret næste G7-opgave

Ny basis er src/method/bm_1_g8_basis.py. Kun G-adapteren bm_1_g8.py får et
eksplicit basis_kind-argument; gamle degree/FV-kald bevarer deres default.
Den eksisterende beta_cell_average_basis, PreparedBetaProjector, FP/Bayes-kode
og kontrolkerner ændres ikke. Alle31 øvrige importerede eksisterende moduler
matcher G-parentens Git-blobs. Ingen ændringer til F/WM-filer.

Runners: src/exe/run_model/bm_1/g8_basis.py og g8_basis_refine.py.
Referencefiler og source hashes låses i registreringerne før hver kørsel.
Den anden runner afviser ændret numerisk source efter første study.
Begge originale registreringer/resultater, den negative coarse-gate, alle
nodehistories, basis, coefficients, evidence, checkpoints og frosne kilder
bevares i råarkivet. Der ændres ikke i gamle G8-arkiver.

Næste opgave er G7's lille single-shooting-kontrol med den accepterede
syntetiske posterior. Brug som start det gemte time12-origin fra
bank_local257 og en separat FV-reference. Mixture-evaluatoren skal eksplicit
modtage originens 42-ledsbasis og hver nodes egne koefficienter; den må ikke
genbygge en standard Bernstein-basis ud fra et gradtal eller projicere til M38.
Vægt objective med de frosne nodevægte; MAP er stadig en særskilt approximation.
Kontrollér det valgte prisforslag mod FV og det erklærede G7-budget.

Den eksisterende weighted_control_evaluation bruger FV-forward fra egne
conditional cell laws og er fortsat korrekt mærket FV. Denne leverance
indeholder ikke en ny mixture-prisoptimizer.

Den nye basis har begrænset mean-/breddeopløsning og er endnu kun kontrolleret
på denne pris-/støjcase. G7's andre priser kan give andre fordelinger;
kontrollerne af det valgte kontrolforslag kan ikke springes over.
Ingen lokal kørsel kræves nu.

