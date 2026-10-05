# Fysiske parametre: direkte optimering og least squares

Udviklingspilot, 5. oktober 2026. Johan bad om at undersøge SciPy-optimering
og least squares også for de fysiske parametre, ikke kun f/g.

## Konklusion og beslutning

Direkte optimering er en konkret kandidat til hurtige betingede punktestimater.
På denne lille pilot bruger både L-BFGS-B på likelihood og TRF på least squares
færre kandidatberegninger end de sammenlignede grids. Least squares indgår nu
som en implementeret udviklingsmetode, ikke blot et forslag. Det er endnu ikke
en integreret eller valideret produktionsestimator.

Bevar F/G-bankerne som reference for parameterusikkerhed og identifikation.
En lokal optimizer erstatter ikke deres posterior, numeriske gates eller
registreringer. H-f/g, H2/H3 og I/J forbliver hovedsporet; ingen eksisterende
kørsel ændres, og ingen stor ny population sættes i gang som følge af piloten.

Næste afgrænsede skridt før operationelt parameterfit er at genbruge den
verificerede likelihood i et egentligt fitinterface med erklærede frie/faste
parametre. Bevar LS som sammenligning for responsparametrene og likelihood for
støj. Bedøm på flere på forhånd valgte udviklingsblokke med eget numerikcheck,
før nogen metode bliver standard. H's endelige holdout forbliver blindt.

## Kilde og afgrænsning

- Beregningskode: commit `7e8a6f8ae0034b35330393471751e4e257be4cc0`, baseret
  på main `06942cf5cd29d457f595e611d5564b5c94bb9fdc`.
- Alle 128 eksisterende Python-filer blev hentet fra dette basecommit og
  verificeret mod deres Git-blob-hashes før beregningen. Ingen eksisterende
  solver, model, filter, gate, observation eller sourcefil blev ændret.
- Ét nyt forsøgsmodul i `src/validation/bm_1/parameter_optimization.py`.
  Det bruger eksisterende `fv_filter_step` og cached-upwind; ingen ny FP-solver.
- Design og hashes for 129 Python-filer blev gemt før fitting og kontrolleret
  uændrede ved afslutningen. Miljø: Python 3.12, NumPy 2.3.5, SciPy 1.17.0,
  Linux, én proces, BLAS/OMP/MKL=1. Se design.json for præcis Python-version.
- To gamle uafhængige syntetiske TRAIN-forløb: ID1326/1327; separate gamle
  DEVELOPMENT-checkforløb: ID1211/1212. Alle er tidligere eksponerede.
  Ingen historiske data, nyt populationsstudie eller blind holdout.
- Faste lineære f/g. De øvrige parametre holdes på de kendte synthetic værdier.
  Resultaterne er derfor betingede, ikke fælles recovery af alle parametre.
- Fælles parametervektor på tværs af de to træningsdøgn, hver med sin deklarerede
  Beta(20,20)-initialprior. Ingen genbrug af posterior fra en anden kandidat.
- FV401, h=1/768, Gauss3. Samme inputs, likelihood og numerik for alle metoder.

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

## Reproduktion og råfiler

Ved integration er CLI og runtime-udskrift flyttet til
`src/exe/run_model/bm_1_parameter_pilot.py`; forsøgsmodulet har en valgfri
progress-callback og importerer ikke længere execution-laget. Den numeriske
scorer, optimizer, budgetter og de arkiverede originale resultater er uændrede.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -B -m unittest discover -s tests -p test_bm1_parameter_optimization.py -v
python -B -m src.exe.run_model.bm_1_parameter_pilot --output files/output/bm_1/runs/estimation/physical_optimizer_reproduction --max-seconds 600
```

Rå design, observationer, kandidatspor, optimeringsstatus og refinement er i
`files/output/bm_1/runs/estimation/physical_optimizer_pilot_v1/`.
Store kandidatrapporter er gzip-komprimerede JSON; summary, design, observationer
og raw_review er almindelig JSON. Alle resultater har scientific_acceptance=False.

Metodisk baggrund: Nocedal & Wright, Numerical Optimization, 2. udg., kap. 7-10;
Alba Gonzalo Primo, Adaptive demand-side flexibility exploitation through dynamic
price signals, afsnit 6.2.1 og 6.3.2. Albas EKF-likelihood/trust-constr er ikke
almindelig LS, og hendes SLSQP-kontroloptimering er et separat problem.
