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

## Hvad der konkret er indført og lavet

M254 repræsenterer hver betinget startfordeling i vejlederens normaliserede
Bernstein/Beta-basis. FV (finite volume) er den alternative transportreference;
EM (Euler–Maruyama) simulerer enkeltbaner til den interne numeriske kontrol.
Denne rettelse ændrer sampling og kontrol af resultaterne. Den ændrer ikke
BM's drift, diffusion, demandfunktion, prisplan eller numeriske tolerancer.

| Del | Konkret ændring og begrundelse | Evidens i denne levering |
|---|---|---|
| RNG-identiteter | Separat initial-, Brownian- og observationsstream; collision/identity-guard før jobs | 11 registrerede seeds; én fokuseret kontraktkontrol |
| EM | Fem nye nodevise coarse/fine-par med fælles start og tilsigtet tidskobling | 256 baner pr. node på hvert level; 1.280 banepar i alt |
| M254/FV | Verificeret genbrug af alle ti resultater fra hver backend | Bevarede kontrakter, sourcehashes, egne origins og payloadhashes |
| Endpoint-feedback | Samme nye syntetiske y til fire backend/grid-opdateringer | Egne priorer, betingede evidences og posteriorvægte kontrolleret |
| Råreview | Reproduktion af tilfældige input og rekonstruktion af cost/SE/Bayes-reduktion | Ingen modelpropagation i reviewet |
| Journal | Separat read-only afstemning af receipts og konservativ charge | Data verificeret; journal/recovery er fortsat uafklaret |

Implementeringen ligger i [runneren](g7/run_auxiliary_seed_repair.py),
[kontraktkontrollen](g7/check_seed_repair_contract.py) og
[råreviewet](g7/review_seed_repair.py). Seedidentiteter og frosne forudsætninger
ligger i [registreringen](g7/seed_repair_registration_20261007.json).

## Matematik, udledning og begrundelser

### Bank, betingede fordelinger og kontrolmål

Lad j=1,…,5 betegne parameterbankens noder, rho_j deres parameterværdier,
w_j≥0 deres gemte cutoff-vægte med sum w_j=1 og p_j tilstandsfordelingen
betinget på den enkelte node. Kodeindeks er 0,…,4. Bankens fælles lov er
\[
 P(X\in dx,\rho=\rho_j)=w_jp_j(x)\,dx.
\]
Hver p_j skal derfor propagéres under sin egen rho_j. Udskiftning af banken
med én gennemsnitlig rho bevarer generelt hverken dynamikken eller
afhængigheden mellem parameter og tilstand. Vægtene er frosne under det
åbne 12→14h-forward; observationen opdaterer dem først ved endpointet.

For demand D_t=D(t,X_t,rho_j,u), mål A_t og den faste neutrale prisplan u er
det stokastiske kontrolmål i denne fixture
\[
 J(u)=\sum_jw_j\int_{12}^{14}E_j[(D_t-A_t)^2]dt.
\]
Med mu_j=E_j[D_t], v_j=Var_j(D_t) og mu=sum_j w_j mu_j giver udvidelse af
kvadratet og E_j[D_t-mu_j]=0 først
\[
 E_j[(D_t-A_t)^2]=(\mu_j-A_t)^2+v_j.
\]
Dernæst er sum_j w_j(mu_j-mu)=0, så
\[
 \sum_jw_jE_j[(D_t-A_t)^2]
 =(\mu-A_t)^2+\sum_jw_jv_j+\sum_jw_j(\mu_j-\mu)^2.
\]
De to sidste led er henholdsvis tilstandsvariation inden for noderne og
variation mellem nodernes middelværdier. De følger af det eksisterende
mål; de er ikke nye strafled. Kun at kvadrere bankens gennemsnitsdemand
ville udelade begge. Denne stokastiske kontrolfixture skal fortsat holdes
adskilt fra BM's deterministiske mål og opgaven med bevaret døgnforbrug.
Her er ingen energilighed eller prisoptimering indført.

### Hvorfor initialsampling svarer til M254-loven

Den ejede startfordeling ved hour12 er
\[
 p_j(x)=\sum_{i=0}^{254}c_{ji}b_{i,254}(x),\quad
 b_{i,254}(x)=255{254\choose i}x^i(1-x)^{254-i},\quad
 c_{ji}\ge0,\quad\sum_ic_{ji}=1.
\]
Beta(i+1,255−i) har normaliseringskonstant
1/B(i+1,255−i)=255!/[i!(254−i)!]=255 binom(254,i).
Dens tæthed er altså netop b_{i,254}. Ved først at trække
I∼Categorical(c_j) og dernæst X_0|I=i∼Beta(i+1,255−i) får vi derfor
\[
 P(X_0\in dx)/dx=\sum_iP(I=i)p(x\mid I=i)=p_j(x).
\]
Sampling er eksakt for den repræsenterede M254-blanding i den matematiske
samplingmodel. Det er ikke et bevis for, at den endelige M254-tilnærmelse
er den eksakte kontinuerte filterlov. FV-cell masses er integrerede
cellesandsynligheder og kan ikke sættes ind som disse basiskoefficienter.
Begge EM-levels bruger samme trukne X_0 for hver bane.

### Estimator og forsvar af standardfejlen

Lad t_ell være de gemte cost-midtpunkter og Delta t_ell deres intervalbredder.
Disse costvægte må ikke forveksles med EM's mindre integrationstrin h.
Alle backends bruger den samme målprofil, prisplan og costpartition. Med
N=256 uafhængige baner pr. node beregnes
\[
 Q_{jr}=\sum_\ell\Delta t_\ell[D_{jr}(t_\ell)-A(t_\ell)]^2,\quad
 \overline Q_j={1\over N}\sum_{r=1}^NQ_{jr},\quad
 \widehat J=\sum_jw_j\overline Q_j.
\]
Linearitet giver E[hat J]=sum_j w_j E[Q_j] for det valgte diskrete
EM-/costproblem. Det hævder ikke unbiasedness for det kontinuerte mål J.
For faste w_j er den generelle varians
\[
 \operatorname{Var}(\widehat J)=
 \sum_jw_j^2\operatorname{Var}(\overline Q_j)
 +2\sum_{j<k}w_jw_k\operatorname{Cov}(\overline Q_j,\overline Q_k).
\]
Under uafhængig sampling mellem noder forsvinder kovariansleddene.
Inden for node j er Var(overline Q_j)=Var(Q_j)/N. Derfor bruges
\[
 s_j^2={1\over N-1}\sum_r(Q_{jr}-\overline Q_j)^2,\qquad
 \widehat{\mathrm{SE}}=
 \sqrt{\sum_j{w_j^2s_j^2\over N}}.
\]
Dette er den stratificerede estimator: alle fem noder simuleres, og deres
gemte vægte anvendes. Det er ikke en estimator baseret på tilfældige
rho-draws. Standardfejlen beskriver Monte Carlo-sampling betinget på
den frosne bank, origin, policy og diskretisering. Den dækker ikke EM-bias,
FP-/basisfejl, costkvadraturfejl eller usikkerheden ved selve bankestimationen.

Master entropy20261007 giver hver (role,node) sin SeedSequence-identitet:
initial=(0,j), Brownian=(1,j), observation=(2,0).
generate_state(1,uint64) omsættes til Python-integer til det eksisterende
default_rng/Brownian-interface. Alle 11 integers er kontrolleret forskellige,
og separate PCG64-generatorinstanser anvendes. Ingen seed+j-offset bruges.
Dette fjerner det konkrete streamgenbrug, som gjorde den tidligere
uafhængighedsantagelse uforsvarlig. Antagelsen om uafhængig sampling er
den sædvanlige PRNG-antagelse; forskellige seeds beviser ikke fysisk
tilfældighed eller universel statistisk uafhængighed.

Den uændrede gate er |J_EM−J_FP|≤max(.001,4 SE). Her er 4 SE≈.000354813,
så .001 er den aktive grænse. Fine M254/EM-differencen er .000047264,
cirka .533 SE. Det er konsistens i denne fixture, ikke en certificeret
confidence-intervalgaranti eller et bevis for eksakt FP-/EM-lighed.

### Hvorfor coarse/fine skal dele Brownian-støj

Skriv den eksisterende nodevise SDE generelt som
\[
 dX_t=b_j(t,X_t,u_t)dt+a_j(t,X_t)dW_t.
\]
Her betegner b_j og a_j de frosne native modelfunktioner; ingen nye
drift-/diffusionsparametre indføres. EM opdaterer
\[
 X_{n+1}=X_n+b_j(t_n,X_n,u_n)h+a_j(t_n,X_n)\Delta W_n,
 \qquad\Delta W_n\sim N(0,h).
\]
Fine h=1/384 time og coarse h=1/192 time kobles ved
\[
 \Delta W^c_n=\Delta W^f_{2n}+\Delta W^f_{2n+1}.
\]
Summen af to uafhængige N(0,h_f)-increments er N(0,2h_f)=N(0,h_c).
Disjunkte par bevarer uafhængige coarse-increments. Dermed har hver
diskretisering den korrekte Brownian-marginal, mens begge beskriver samme
underliggende støjrealisering. Denne tilsigtede afhængighed er inden for
node/banepar; standardfejlen ovenfor bruger kun det fine level.

For en paired costdifference gælder Var(Q_f−Q_c)=Var(Q_f)+Var(Q_c)−2Cov(Q_f,Q_c).
Koblingens formål er at reducere sammenligningsstøj gennem positiv kovarians;
det er ingen universel garanti for et bestemt reduktionsforhold. Den målte
|hat J_f−hat J_c|=9.6925e−8 er under .001-gaten, men ét gridpar kan ikke
certificere tidskonvergens eller udelukke fælles bias. Ingen clipping eller
reflection anvendes; nul randoverskridelser gælder kun de simulerede baner.

### Endpoint-observation og Bayes-opdatering

Truth er den prædeklarerede node med kodeindeks2, rho=.25, fine path0.
Ved t=14 er y=D_truth+epsilon, epsilon∼N(0,sigma_y²), med separat
observationsstream og den ejede sigma_y. Det gemte endpoint er
truth=.4430733698873627, noise=−.042062974062744334,
y=.40101039582461834. Samme y bruges i alle fire updates.
Dette er en kontrolleret syntetisk fixture ved fast rho, ikke et draw af
truth-rho fra banken eller en historisk measurement.

Med h_j(x)=D(14,x,rho_j,u_14), under den gemte price-side-konvention,
er likelihood
\[
 L_j(y\mid x)={1\over\sqrt{2\pi}\sigma_y}
 \exp\!\left[-{(y-h_j(x))^2\over2\sigma_y^2}\right].
\]
For hver backends egen predicted prior p_j^- og egne bankvægte w_j^- giver
Bayes' regel
\[
 Z_j=\int L_j(y\mid x)p_j^-(x)dx,\quad
 Z=\sum_jw_j^-Z_j,\quad
 p_j^{raw,+}(x)={L_j(y\mid x)p_j^-(x)\over Z_j},\quad
 w_j^+={w_j^-Z_j\over Z}.
\]
Normalisering følger direkte: integralet af p_j^{raw,+} og summen af w_j^+
er begge 1. Dette er den kontinuerte Bayes-identitet. Den faktiske native
update anvender egne predicted cell masses m_ji og en trepunkts
Gauss–Legendre-tilnærmelse til den gennemsnitlige likelihood i hver celle C_i:
\[
 \overline L_{ji}\approx{1\over|C_i|}\int_{C_i}L_j(y\mid x)dx,\quad
 Z_j^{grid}=\sum_im_{ji}\overline L_{ji},\quad
 m_{ji}^{raw,+}={m_{ji}\overline L_{ji}\over Z_j^{grid}}.
\]
Det svarer til en stykvis konstant celletæthed; det er ikke eksakt
integration af M254's kontinuerte basisfunktioner. Cellemasser og
likelihoodkvadratur giver en separat numerisk tilnærmelse, også for M254.
Fine/coarse må ikke låne hinandens priorvægte for at få bedre
overensstemmelse. Logberegningen bruger a_j=log w_j^-+log Z_j,
log Z=logsumexp(a), log w_j^+=a_j−log Z for at undgå underflow.
Råreviewet rekonstruerer denne bankreduktion fra hvert grids egne data.

M254's stateposterior projiceres efter beregning af Z_j. Projektionen er
en numerisk repræsentationstilnærmelse og må ikke omdefinere parameter-
likelihooden. Projektionens mass/L1-tal og posteriorernes forskelle
kontrolleres særskilt. En missing observation bevarer predicted origin.
Endpoint-kontrollen validerer dette håndoff; den udfører ikke et efterfølgende
feedback-kontroltrin eller likelihood for historiske timeintervaller.
Native sammenhæng: OwnedBankPriceEvaluator.assimilate i
src/method/bm_1_g7.py og bm1_observation_update/_cell_log_likelihood i
src/method/bm_1_estimation.py, under de registrerede sourcehashes.

### Hvorfor de deterministiske forwards kan genbruges

Et gemt FP-resultat afhænger af egne startfordelinger, parameterbank,
prisplan, horizon, model/sourceversion, backend/grid og costpartition.
Seed-rettelsen ændrer ingen af disse inputs. FP-forwards bruger ikke de
tilfældige input til EM eller det senere syntetiske endpoint. Derfor kan
de ti M254- og ti FV-forwards genbruges under de verificerede kontrakter.
Den nye observation kræver nye posteriorupdates, ikke ny propagation frem
til den samme observationstid. Ny policy eller ny origin kræver derimod
nye relevante forwards; de gemte neutralresultater kan ikke ommærkes.

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

Den efterfølgende matematiske uddybning og dokumentationsregel er en
tekstændring i Git. Den ændrer ikke de frosne sources, råarkivets snapshot,
acceptflags eller ovenstående kørselsbudget. Formler og metodebeskrivelser
er kontrolleret mod runner, råreview og de frosne native EM-/Bayes-interfaces;
der er ikke kørt nye numeriske tests til denne dokumentationsændring.

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
