# BM-audit: F1-A til F3-F

## Status og grundlag

Denne pakke er en **afprøvet kandidat på en separat lokal audit-branch**, ikke en ændring på GitHub/main. Baseline er den mergede commit `a3c0f883de35ee9284284d97dc95caf3a647bea8`. F3-G og F4 er ikke påbegyndt.

Kilden er hentet fra det eksisterende GitHub Actions-artifact for run `36489917469`, artifact `11001357177`. Arkivet indeholder en syntetisk PR-mergecommit, men de tre kilde-træer er verificeret identiske med main:

- `src`: `e0b44ab0988e979294ffecf52cf257236d200b8a`;
- `tests`: `bfbf3cdbe6d2bca333b3a710a3d732f4798d6a2b`;
- `root`: `6537209c7a3b4ca386d9f4fd9449235c65b24647`.

Der findes faktisk en CI-workflow, `.github/workflows/bm-review.yml`. Den tidligere chatbesked om, at ingen CI var konfigureret, var for kategorisk: et tomt combined-status-svar udelukker ikke GitHub Actions. Baseline havde en vellykket Actions-kørsel. Den nye patch er **kun kørt i dette lokale testmiljø**, ikke på brugerens Windows-installation eller i ny remote CI.

## Omfang

Gennemgangen omfatter BM-kæden fra de eksakte/deterministiske F1-benchmarks til F2's SDE, finite volume, mixture, conditioning og objectives samt F3-A–F's fælles kontrolproblem, regularisering, beslutningssammenligninger og numeriske validering. Input-/rapportering og afhængigheder mod den øvrige kode er medtaget. Kildeinventaret indeholder 94 Python-filer inklusive tests og shared/WM-afhængigheder; dette er **ikke** en ny faglig validering af alle WM-modeller og deres fitting-workflows.

| Område | Kontrol og handling |
|---|---|
| F1-A–F | Eksakt benchmark/Euler, prisbrud, ODE, feasibility og optimeringskontrakt dækkes af de bevarede regressionstests; BM-responsevalidering deduplikeret. |
| F2-A–D | SDE-/FV-kernel, diffusion, randregler og Brownian coupling bevaret; sampler- og sammenligningschecks strammet. |
| F2-E–G | Dynamisk simplex-projektion profileret og erstattet med en forberedt direkte least-squares-løsning; statisk shared FP-projektion bevaret. |
| F2-H–I | Samme momentsbaserede objective; fuld F3-F-kontrol og komplekst 2/4/6/12/24-prisforsøg genkørt. |
| F3-A–C | Samme problem-definition; samlet pathwise cost-helper og regression for positive terminal-/smoothing-vægte. |
| F3-D–F | F3-D eksisterende regressioner bevaret; fuld F3-E og F3-F samt fin FV/MC-reference genkørt. |
| Data/rapportering | Syntetiske datatests, strengt JSON, intet nyt datasæt eller nye fysiske antagelser. |

## 1. Målinger

Miljø: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Linux x86_64. `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MPLBACKEND=Agg`. Samme kontrolproblem, prisvektor `[0.735904, 0.263610]`, grids, seeds og precision settings på begge sider. Constructor, import, target-opbygning og den separate profileringskørsel er ikke med i tabellen. Mixture måles efter én opvarmningskørsel, som passer til gentagne evaluator-kald i en optimizer.

Baseline er median af fem gentagelser, release-kandidaten median af syv. Der er ingen samtidige egne benchmarks under målingerne. Rå gentagelser ligger i `evidence/bench_baseline.json` og `evidence/bench_release.json`. Værtsmaskinens variation betyder, at små gevinster skal tolkes forsigtigt; tallene er ikke løfter om identiske faktorer på Windows.

| Beregning | Før, sek. | Efter, sek. | Faktor |
|---|---:|---:|---:|
| Deterministisk evaluering | 0.139532 | 0.117634 | 1.19× |
| Fast FV, 201 celler | 0.010495 | 0.008306 | 1.26× |
| Fast FV, 801 celler, fin tid | 0.086808 | 0.066674 | 1.30× |
| Dynamic mixture M=30, 81 celler | 2.248202 | 0.286354 | 7.85× |
| Dynamic mixture M=30, 201 celler | 9.648240 | 1.450135 | 6.65× |
| Dynamic mixture M=38, 201 celler | 15.400091 | 1.408309 | 10.94× |
| MC, 20.000 baner, to policies | 0.826798 | 0.826304 | 1.00× |


Den store, robuste gevinst ligger i **dynamic mixture**, ikke i en påstand om, at alle dele af projektet bliver lige meget hurtigere. I den gennemførte end-to-end F3-F-kørsel tog selve mixture-optimeringen 117,15 sekunder, og hele F3-F inkl. fælles reevaluering/MC tog 136,83 sekunder. Dette er kandidatens målte wall time, ikke et kontrolleret før/efter-speedup for hele workflowet. Kørselstiden fra brugerens tidligere Windows-log er ikke brugt som nævner.

### Iterationer og stopkriterium

**Runde 1:** Profileringen af en M=30/81-evaluering viste 2,763 af 3,244 profilerede sekunder i den generelle active-set-solver, cirka 85 %. Gentagne matrix-rank-beregninger og fulde KKT-løsninger var hovedproblemet. Den nye forberedte simplex-solver gav den første store forbedring.

**Runde 2:** Fjernede kun dublerede BM-arraychecks, hvor public `l`/`D` allerede validerer de samme værdier. Prisvalideringen bevares, og custom overrides får stadig de oprindelige inputchecks. Parametre og callbacks fastfryses ikke for at gøre en tidsvarierende model billigere.

**Runde 3:** Et forsøg med tre fulde blokke af implicitte FV-matricer blev forkastet. En separat, alterneret 15-runders sammenligning gav ved 801 celler medianerne 0,07712 s for den oprindelige metode, 0,08334 s for allokering af alle stepmatricer og 0,07227 s for en begrænset fælles finiteness-kontrol. Den sidste blev beholdt; de oprindelige per-step matricer og alle efterfølgende mass-/LAPACK-checks er bevaret. Hele massehistorien og objective var bitvis ens mellem de tre varianter.

**Runde 4:** Faktoriseringernes LRU-cache blev afprøvet med 0, 32, 128 og 512 entries. For M=38/201 krævede 128-cache stadig 347 nye faktoriseringer pr. gentaget evaluering, mens 512 kunne rumme de besøgte faces og krævede nul efter opvarmning. En efterfølgende alterneret 512/1024-test viste kun cirka 0,5–1 % tidsforskel; begge krævede allerede nul nye faktoriseringer. 512 blev derfor beholdt.

Stop betyder her: ingen yderligere **overbevisende, afprøvet gevinst blandt disse sikre kandidater**, ikke et bevis for globalt optimal kode. JIT, parallelle optimizers, adjoints og streaming-MC kræver større arkitektur- eller reproducerbarhedsarbejde og er ikke presset ind i denne runde.

## 2. Hessian-konditionering: reel forbedring uden en ændret model

Den dynamiske projektion er stadig

\[
\min_{c\geq0,\;\mathbf1^Tc=1} F(c),\qquad F(c)=\tfrac12\|Ac-b\|_2^2,
\]

hvor A er den vægtede Beta-basismatrice og b den vægtede forudsagte density. Der er ikke tilføjet ridge, `epsilon*I`, negative vægte eller en ændret Beta-grad.

Den tidligere generelle solver arbejdede med `H=A.T@A` og et KKT-system. For fuld kolonnerang er \(\kappa_2(A^TA)=\kappa_2(A)^2\). De høje raw-H-værdier omkring 10^16–10^17 bliver ikke fysisk mindre, blot fordi vi ønsker et pænere tal. Over dette niveau er også selve den rapporterede condition-værdi følsom for floating-point-rang.

Den nye private `_SimplexProjector` i den eksisterende `beta_mixture.py` arbejder på den aktive simplex-face. For de frie koefficienter F bruges en ortonormal nulsumsmatrice Z:

\[
\mathbf1^TZ=0,\qquad Z^TZ=I.
\]

Retningen beregnes direkte fra least-squares-problemet med \(A_F Z\) ved SVD. Lighedsbetingelsen respekteres dermed af retningen; aktive bounds håndteres af active-set-steppet. SVD-faktoriseringerne afhænger af basis og fri-indekser, ikke af target eller pris, og kan derfor genbruges uden at cache target-afhængige løsninger. Cache-state ændrer ikke startvægtene.

Små singulære værdier vurderes med en eksplicit machine-precision-rankgrænse. Det er numerisk ranghåndtering, ikke en ny regulariseret objective. Der findes stadig intrinsisk svagt bestemte koefficientkombinationer. Derfor kontrolleres den repræsenterede density, objective og beslutning frem for at kræve identiske koefficienter.

### Uafhængig førsteordenskontrol

Et `success=True` er ikke alene nok. Med

\[
g=A^T(Ac-b),\qquad G(c)=g^Tc-\min_i g_i
\]

følger for en feasible simplex-vektor ved konveksitet, at

\[
0\le F(c)-F(c^*)\le G(c).
\]

Denne gap beregnes uafhængigt af SVD/SLSQP-stoppet. Den er en floating-point-diagnostik, ikke en intervalaritmetisk certificering af hele FP-forløbet. Primær løsning accepteres ved scaled gap-tolerance 10^-11; fallback ved 10^-9, sammen med finiteness, koefficient-/mass-constraints. Originalens mass-tolerancer er ikke løsnet.

SLSQP er fortsat fallback for præcis samme problem, men dens indre `ftol` er strammet fra 10^-12 til 10^-16. I den tvungne fallback-regression gav de løsere indstillinger succes, selv om den uafhængige gap stadig var cirka 1,77e-8; det ekstra solve-step gav cirka 1,45e-17. **Dette er en ændret intern projektionstolerance**, ikke en ændring af Powell, FP-grid eller fysisk objective. En falsk positiv success-flag afvises nu af en test.

Shared `src/solver/active_set.py` og `src/method/fokker_planc.py` er ikke omskrevet. Statisk initialprojektion og WM's øvrige anvendelser fortsætter som før. Raw Hessian-conditioning bliver stående som diagnostik; forbedringen er, at det dårligt konditionerede Gram/KKT-system ikke længere bruges til den gentagne BM-dynamiske solve.

## 3. Checks: hvad blev beholdt, samlet eller skærpet?

Følgende blev beholdt: fysiske parametergrænser, endepunktskrav, finite værdier, prisbounds, grid-/breakpointkrav, massebevarelse, negativitetskontrol, rå EM-randoverskridelser, solver-return-status og fejl ved ugyldige resultater. Der returneres aldrig en kunstig stor pris-cost for at skjule en fejlet numerisk simulering.

For fast FV konstrueres en Markov-generator med ikke-negative rater. For hvert positivt step h er

\[
1+h\max_i(-Q_{ii})
\]

den største implicitte diagonal og en øvre grænse for alle offdiagonale størrelser. Denne **faktiske matrixgrænse** kontrolleres for hvert step samlet pr. blok. Dermed undgås tre gentagne array-reduktioner inde i Python-loopet, men overflow fanges stadig før LAPACK. En ny regression laver bevidst finite generatorrater og et overflowende hQ-produkt og verificerer, at LAPACK ikke kaldes.

Enkelte overfladisk ens checks kan ikke fjernes: callback-output kan variere over tid, en start kan være gyldig mens evolutionen fejler, og sum(c)=1 er ikke alene et mål for korrekt projektion. De checks er bevidst stadig til stede.

## 4. Korrektheds- og kvalitetsrettelser

### Samme komplette F3-objective i alle MC-sammenligninger

Tidsrefinementet brugte tidligere kun tracking-cost, selv når problemobjektet havde positive terminal-/smoothing-vægte. `run_final_mc_comparison` rapporterede tilsvarende tracking-only total/SE/prefixes. Nu genbruges én pathwise helper for

\[
J^{(r)}=J_{tracking}^{(r)}+\lambda_T(X_T^{(r)}-x_{ref})^2+
\lambda_U\sum_j(u_j-u_{j-1})^2.
\]

Dermed medregnes covariance mellem pathwise tracking og terminalstate i standardfejlen. Det ændrer ikke de allerede gemte F3-F-tal med nulvægte, men retter genbrug med positive F3-C-vægte.

### Sammenligningernes inputkontrakt

MC kræver mindst to paths til en sample-SE. Time-MC er enten slået fra med nul eller har mindst to paths. Ugyldige counts afvises inden dyr optimering. Et kandidatnavn kan ikke overskrive den nye `highres_fv_policy`. Brøkdele af celler, intervalantal eller grader bliver ikke længere stiltiende trunkeret til et andet eksperiment.

En custom initial law må ikke automatisk blive sammenlignet med Beta(20,20)-MC. Standard-Beta-loven kontrolleres på evaluatorens grid; afvigelser kræver en eksplicit passende sampler. Dette grid-check er en praktisk guard, ikke et bevis for lighed mellem vilkårlige kontinuerte fordelinger. Samplerens matematiske overensstemmelse med brugerens mass-builder er fortsat et caller-contract. Returneret sampleshape, finiteness og bounds kontrolleres.

Parrede costs skal være to finite vektorer med samme længde; broadcasting tæller ikke som pairing. 0/0 ved identiske deterministiske costs giver nu en udefineret ratio (`NaN`) frem for et misvisende plus-uendeligt signal. JSON-eksport omsætter udefinerede diagnostikker til `null`, også inde i NumPy-arrays/lister, med `allow_nan=False`. Ikke-finite modelcosts bliver fortsat afvist, ikke skjult som null.

### Faktiske versus ønskede EM-tidsstep

Den gamle rapport viste kun `max_step`, som er en øvre grænse. Outputtidspunkter og den nestede konstruktion kan tvinge finere steps. For F3-F-refinementet med q=1/192 er de faktisk observerede maksimumsteps:

| Ønsket bound | Faktisk maksimum | Antal integrationsteps |
|---|---|---:|
| 1/48 | 1/192 | 194 |
| 1/96 | 1/384 | 388 |
| 1/192 | 1/768 | 776 |
| 1/384 | 1/1536 | 1552 |

Grids eller seed er **ikke ændret**. Rapporten viser nu både faktisk step og stepantal; resultatobjektet indeholder desuden paired SE og SE på ændringen i policy-gap under koblet refinement.

For den gamle mixture-policy er sidste gap-ændring 3,83e-8 med SE 3,12e-9. Det er lille mod policy-tabet, men ikke statistisk nul. Den tidligere formulering om “konvergeret” var derfor for stærk. Desuden varierer både finite sample og de faktiske tidsgrids mellem 20k-singlegrid-checket og 5k-coupled-checket; forskellen bør ikke tilskrives sample-støj alene.

## 5. Numerisk regression og fulde forsøg

Den afsluttende suite består: **200 tests, OK** mod oprindeligt 176. De 24 nye tests omfatter projektionens optimality/feasibility, cache-/rangtilfælde, falsk SLSQP-success, manglende penalized path-cost, ulovlige counts/samplers, referenceoverskrivning, faktisk stepstørrelse, randfejl, overflow, overrides og JSON.

Ved de fastholdte benchmark-inputs er deterministisk state/cost, FV201/FV801's fulde massehistorier/costs samt de fælles MC-path-cost-vektorer **bitvis identiske** med baseline i dette miljø. Mixture er en ændret numerisk solve, så bitvis lighed forventes ikke:

| Mixture-case | Maks. ændring i cellemasse | Ændring i total cost |
|---|---:|---:|
| mixture30_81 | 3.958e-14 | 3.512e-15 |
| mixture30_201 | 2.517e-11 | 1.271e-12 |
| mixture38_201 | 1.404e-08 | 6.927e-10 |


Det er målte forskelle i de fastholdte cases, ikke globale fejlgrænser. Matrix-/koefficient-conditioning håndteres derfor sammen med held-policy reevaluering.

Fuld F3-F: alle tre optimizers lykkedes fra alle tre starts. Der var **nul dynamiske projektion-fallbacks** på hele denne kandidatoptimering og dens reevalueringer. En separat stress-test gennemførte 30 fulde mixture-forløb med reference-, yder- og tilfældige priser ved M30/81 og M38/201, også med nul fallbacks og bevaret sandsynlighedsmasse. De 30 stress-cases er feasibility-/robusthedstests, ikke alle numerisk sammenlignet mod den gamle solver.

Den nye mixture-optimering returnerer `[0.7359350146, 0.2636055093]`, mod den tidligere viste afrundede `[0.735904, 0.263610]`. Største forskel er cirka 3,1e-5, langt under den uændrede ydre Powell-xtol på 0,001. Mixture-evaluatorens egen cost ændres cirka 1,5e-10 ved disse nærliggende kandidater. Det er ikke korrekt at love identiske optimizer-vektorer efter et numerisk backend-skift.

Den gamle held-policy reproducerer under fin FV/MC fortsat regret 2,33237524e-6 og paired MC -2,21420269e-6. **Også den nyfundne policy er valideret** uden ny optimering:

- high-resolution FV-regret: `2.36876799922e-06`, ca. `0.03612%` af referencecost;
- paired MC highres-reference minus ny mixture: `-2.24951301995e-06`, SE `6.38294406824e-07`.

Den oprindelige faglige konklusion er dermed bevaret: nærliggende, ikke identiske policies med en meget lille reference-cost-forskel.


F3-C light-regularization, alle fire F3-E-scenarier og det komplekse 2/4/6/12/24-prisforsøg med MC gennemføres også. 24-pris objective er 0,00575901191445379, samme værdi som det tidligere performance-referenceforsøg. Det er ikke en påstand om, at alle gamle dyre sensitivity-grids er kørt fuldt igen.

## 6. Bevidst ikke ændret og begrænsninger

- Ingen ændringer i drift/diffusion, fysisk kapacitet, initial referencefordeling, Beta-basisfamilie, CFL, tids-/rumgrids, prisparameterisering, seeds eller ydre optimizer-indstillinger.
- Ingen omskrivning af shared FV-, EM-, active-set- eller FP-kernels, shared validators, WM-modeller, package initializers eller parameterfitting.
- Ingen nye runtime-Python-moduler eller kompatibilitets-shims; kun afgrænsede udvidelser i eksisterende BM-filer og tests.
- Ingen færre paths/grøvere grids/løsere masschecks som performance-trick. Den indre mixture-solver og dens stopkontrol er netop ændret og dokumenteret; dens resultater er testet, ikke antaget identiske.
- Historiske rå Excel-workbooks var ikke med i source-artifactet. Syntetiske input-/JSON-tests er kørt; komplet genkørsel af real-data-workbooks er ikke udført.
- Windows/Python3.11/anden BLAS kan give andre timings og små numeriske forskelle. Lokal Windows-regression skal bestå før merge. De nye source-filer er også syntakschecket mod Python3.11-grammatik, men det erstatter ikke runtime-test.
- Der er ikke givet et globalt optimum-bevis, en streng samlet FP-fejlgrænse eller en garanti for alle mulige callbacks og parametre. Shared WM-importtests består, men alle lange WM-fitjobs er ikke genkørt.
- Den eksisterende WM2 docstring-warning om en escape-sekvens er ikke skjult globalt eller ændret i shared kode.

## 7. Evidens og reproduktion

Pakken indeholder benchmarkscript, integrations-/stressscripts, alle udvalgte rå timingserier, before/after-array-snapshots, profiler og de afsluttende testlogs. `numerical_comparison.json` angiver de observerede maksimale forskelle. `source_manifest.json` angiver scope og SHA256 for ændrede filer. `SHA256SUMS.txt` gælder de leverede filer.

Benchmark kan køres i en separat baseline-worktree og audit-worktree med samme miljø og ingen andre egne performance-jobs. Eksempel:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -B .\scripts\bench.py --root <baseline-repo> --out <baseline-output> --repeat 7
python -B .\scripts\bench.py --root <audit-repo> --out <audit-output> --repeat 7
```

Der ændres ingen maskinomspændende thread-settings; ovenstående gælder den pågældende shell. Profilering er valgfri med `--profile` og måles separat fra benchmark-medianerne.

Primære eksterne metodehenvisninger: SciPy-dokumentation for `lstsq` (least squares, singulære værdier og numerisk rang) og `minimize(method='SLSQP')` (ftol's flere stopbetingelser). Selve gap-uligheden ovenfor følger direkte af konveksitet og simplex-constrainten. Repositoryets F1/F2/F3-noter er grundlaget for model- og experimentscope.
