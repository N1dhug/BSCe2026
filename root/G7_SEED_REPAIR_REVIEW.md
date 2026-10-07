# G7: separate RNG-streams og færdig kort kontrol, 7. oktober 2026

## Resultat

Johans instruktion om at fortsætte er udført som én ny præregistreret
seed-rettelse. Den oprindelige neutrale auxiliary-kørsel og dens charge
bevares. Den nye kørsel genbruger alle ti egne M254-forwards og alle ti
færdige FV-forwards. Kun fem coupled EM-par (256 paths/node) og fire
feedback-opdateringer beregnes på ny. Ingen optimizer, ny FP-forward,
historisk observation, uafhængig kvadratur eller WM-kørsel.

Præregistrering/sourcecheckpoint:
9da3aa02276abf14c6a7e227b417328be34c939c.
De gamle 41 native sourcehashes, egne startposteriorer, mål/pris/horizon,
transportgrids, EM-tidsgrids, boundary-policy og tolerancer er bevaret.
Workerens AST ændrer kun de to seed-lookups og tilføjer en collision-guard.
Budget-/stop-/eksportdelen er særskilt registreret til den nye invocation.

| Resultat | Værdi |
|---|---:|
| M254 fine J, genbrugt | .0037155174638558177 |
| FV fine J, genbrugt | .0037153098107086916 |
| EM fine J, nye streams | .0036682534675161536 |
| EM coarse J | .0036683503927112462 |
| Coupled tidsdifference | 9.692519509262379e-8 |
| Fine M254/EM J-difference | 4.726399633966406e-5 |
| Fine FV/EM J-difference | 4.705634319253794e-5 |
| Stratificeret plugin-standardfejl | 8.870328259665031e-5 |
| Raw EM boundary violations | 0 |
| Største feedback conditional L1 | .014528039487282984 |
| Største feedback vægtdifference | .0020073702279112537 |
| Største feedback log-evidence-difference | .012081321618276242 |
| Største M254 feedbackprojektion cell-mass L1 | .00212726906289321 |

Alle registrerede korte forward-/EM-/feedback-/projektionstal består.
M254/FV-værdierne er eksakt de tidligere værdier; nye seeds ændrer EM og det
nye syntetiske endpoint, som forventet. Dette er ikke prisoptimering eller
en historisk test.

## Matematik og streamidentiteter

Den bankvægtede estimator er
\[
\widehat J=\sum_{j=1}^{5}w_j\,{1\over N}\sum_{r=1}^{N}Q_{jr},
\quad
Q_{jr}=\sum_\ell\Delta t_\ell(D_{jr}(t_\ell)-A(t_\ell))^2,\quad N=256.
\]
Med uafhængige node-samples bruges
\[
\widehat{\operatorname{Var}}(\widehat J)=\sum_j{w_j^2s_j^2\over N}.
\]
Dette er en plugin-standardfejl, ikke en kontinuumsfejlgrænse eller en
certifieret confidence-intervalgaranti. FP/EM-differencen svarer i dette
case til cirka .533 standardfejl og ligger under den uændrede faste .001-gate.

Master entropy20261007 giver hver (role,node) sin SeedSequence:
initial=(0,j), Brownian=(1,j), observation=(2,0).
generate_state(1,uint64) omsættes til en Python-integer, som eksisterende
default_rng/Brownian-interface accepterer. Alle11 registrerede integers
er forskellige; tabellen bevares eksakt i seed_repair_registration_20261007.json.
Ingen seed+j-offset bruges i denne worker.

Der bruges separate PCG64-generatorinstanser. Dette retter den konkrete
genbrug af samme RNG-stream. Uafhængighed fortolkes som den almindelige
pseudorandom-samplingantagelse; forskellige streamidentiteter er ikke et
matematisk bevis for fysisk/ægte tilfældighed.

Inden for ét nodepar er starttilstanden fælles, og
\[
\Delta W^c_r=\Delta W^f_{2r}+\Delta W^f_{2r+1}.
\]
Denne tilsigtede tidskobling er bevaret. Den fine startfordeling drawes fra
nodens egne M254-koefficienter via component∼Categorical(c_j), derefter
X0∼Beta(component+1,255-component). FV-cell masses bruges ikke som
substitut for den kontinuerte M254-startdraw.

Truth-node2/rho.25/path0 er prædeklareret, som i den oprindelige fixture.
Nyt endpoint: truth=.4430733698873627, noise=−.042062974062744334,
y=.40101039582461834. Samme y anvendes i alle fire backend/grid-updates.
Ingen historisk measurement eller parameterbank-draw af truth-rho påstås.

## Faktisk kontrol

Én fokuseret stdlib-kontraktkontrol består på .022357413s uden native imports
eller forwards: positive streamidentiteter, collision/changed-identity/missing-
node afvisninger, worker-AST, launcherhashes og komplet bevaret budget.
Ingen gamle numeriske tests genkøres.

Råreviewet tager8.916035462s og importerer ikke en solver. Alle24 nye
receipt-payloads/jobidentiteter og de14 tidligere completed FV/setup-receipts
kontrolleres. De fem initial-component/Beta-draws, fine Brownian-increments
og observationsnoise reproduceres præcist fra registrerede streams.
Begge levels starter fra de gemte egne initialdraws. EM-pathcosts rekonstrueres
identisk; standardfejl og hvert grids egne Bayes-logvægte/log-evidence
kontrolleres. Der propagéres ingen model i reviewet.

independent_mc_standard_error_verified og independent_synthetic_noise_verified
er true under den dokumenterede PRNG-antagelse. Råreviewets scientific_accept
og full_day_budget_admitted er fortsat false. Der er ikke kørt calibration,
interval-likelihood, optimizer eller selvstændig quadratur.

## Journalafvigelse og konservativ reconciliation

Første read-only auditorforsøg standsede, fordi den oprindelige journal ikke
har samme komplette status som de gemte payloads: FV coarse-node0 og1 mangler
journalentries; FV fine-node4 står running, men har en complete receipt.
De øvrige21 entries er complete. Alle24 raw receipts findes, har gyldigt
job-/payloadhash og matcher registrering samt workerens stageidentiteter.

Årsagen til journalafvigelsen er ikke fastlagt. Der påstås derfor ingen
journal-/resume-accept eller testet crash-exact recovery for denne invocation.
Ingen ny forward er kørt for at reparere bogføringen.

Originaljournalen og det første failed-auditorsource bevares uændret. Et
read-only reconciliation-sidecar verificerer alle24 complete payloads.
For manglende/running genbrugsentries bruges mindst deres fulde .5s reservation,
og for hver øvrig entry mindst den oprindelige charge og gemte stage-wall.
Samlet konservativ receipt-charge er19.336431619s. Parentens målte20.996288735s
dækker den fulde sum; derfor reduceres ingen charge og kræves ingen ekstra
modelcharge. Metadata/sidecar er ikke en automatisk omskrivning af NodeJobStore.

journal_integrity_passed=false; receipt_reconciliation_verified=true.
Genoptagelse via den oprindelige aktive node-store er ikke tilladt fra
sidecaret. Journalens writer-/recovery-path skal undersøges og løses i et
G-afgrænset interface før nye optimizer-/resumejobs. Ingen F/shared kernel
er ændret for denne bogføringsafvigelse.

## Samlet budget og råarkiv

Ny scope120s =70 worker +30 stop/eksport +20 review; jobreservationer68s.
Tidligere G1347.654245114s bevares, inklusive alle gamle fejlforsøg/reviews.
Fokuscheck, det fejlede ledgerreview og successful raw review er dækket
af den nye konservative20s reviewcharge; failed runtime er ikke særskilt
instrumenteret og er ikke slettet fra historikken.

Worker-wall20.996288735s, oprindelig eksport12.405701579s,
ekstra reviewpakkeeksport.026505s, review20s:
**ny scopecharge53.428495314s; G1401.082740428s; rest398.917259572s af1800s.**
Der er ingen reset/refund eller ny allowance. Full-day admission er false.

G7_seed_repair_20261007_review.zip er gemt med alle24 råreceipts, original
anomalous journal, streams, sources og begge auditorversioner.
60983504 bytes; SHA256
0bb3c1c5b0ff1afb91906251c126b5cb84b78adb3719a0bf192b8a0dbaf9903f.
Library ID libfile_048717f2dbb081919ec84b19be95e09a.
Final package-timing/budget er separat, så arkivet ikke skal omskrives for
at medtage sin egen eksporttid. Metadata/hash ligger i
seed_repair_complete_archive_index_20261007.json.
Final budget er seed_repair_final_budget_20261007.json.

## Næste konkrete G7-opgave

Seed-rettelsen og den korte numeriske kontrol er afsluttet; de skal ikke
genkøres af vane. Næste trin er et afgrænset journal-/receipt-interface,
der afviser/afklarer manglende journalentries før native arbejde og bevarer
charges og de komplette raw payloads. Genbrug denne evidence uden forwards.

Derefter registreres en budgetmæssigt mulig prisoptimerings-/kvadraturopgave
med præcise dependencies, acceptkrav og stop/export-reserve. Ny policy kræver
nye forwards; neutraldata kan ikke ommærkes til en ny prisplan.
2h-rates giver ingen24h-admission. Historisk intervalestimation og slutbenchmark
er særskilte krav; G7/G som helhed er ikke afsluttet.
Direkte BM/WM-sammenligning venter stadig til BM er færdig; F/WM bevares.
