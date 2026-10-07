# G7: budget- og integrationsdesign efter egne M254-rates

7. oktober 2026. G-sporet. Dette er et færdigt designreview og et maskinlæsbart
forsøgsdesign, **ikke en ny numerisk kørsel**. Ny launcher, input-/sourcefrysning
og særskilt præregistrering mangler før udførelse. Ingen optimizer startes.

## Aktiv afgrænsning: BM færdig først, WM-sammenligning bagefter

Johan, 7. oktober 2026 kl.16:25: G-arbejdet skal udvikle og færdiggøre BM.
Direkte BM/WM-sammenligning, fælles benchmarkkørsler og rangering udskydes,
til BM er færdig. WM-resultater er ikke en adgangsgate for BM's næste trin.

BM skal under udviklingen være meningsfuldt sammenlignelig med WM:
dokumentér tidsakse/horizon, informationscutoff og dataopdeling, input,
baseline/målprofil, normalisering/enheder, prisdefinition og tilladte
kontroller, forbrugsbegrænsninger, metrics og outputformat. Genbrug relevante
WM-konventioner; begrund nødvendige modelforskelle. Det kræver ikke løbende
WM-kørsler eller tilpasning af BM til WM's resultater.

BM's deterministiske kontrolmål og varianten med bevaret samlet forbrug
bevares som BM-opgaver. Stokastiske M254/FV/EM-kontroller er interne
numeriske BM-kontroller og fortsætter. Vejlederbasis/legacy og adskillelsen
mellem G og F bevares.

Tidligere gennemførte BM/WM-forsøg og råresultater bevares som historisk
evidens, men er ikke den aktive udviklingsprioritet eller en endelig
modelsammenligning. Denne afgrænsning overstyrer ældre formuleringer om
"primær BM/WM-sammenligning", også i det gemte auxiliary-designsnapshot.
Det ændrer ikke snapshotbytes, numeriske gates eller beregningsbudget.
Næste G7-opgave er fortsat BM's restore/launcher og afgrænsede FV/EM/feedback-
forsøg. Selve BM/WM-sammenligningen planlægges efter BM's færdiggørelse.

## Beslutning og afgrænsning

BM's deterministiske kontrol med bevaret døgnforbrug bevares som en BM-opgave.
Genbrug den eksisterende BM-evidens og dokumentér mål, input, prisblokke og
normalisering, så en senere WM-sammenligning giver mening. Den tidligere
fælles undersøgelse er historisk evidens; nye WM-benchmarkkørsler og rangering
venter, til BM er færdig. Se G7_ENERGY_NEUTRAL_REVIEW.md for de bevarede resultater.

Den stokastiske G7-opgave bevares. Det næste lille trin skal måle de manglende
FV/EM/feedback-dele på den allerede gemte neutrale to-timerspolicy. Det bør ske
før en optimizer bruger mange nye forwards. Supervisor M254 er primær;
FV er uafhængig transportreference på samme startsandsynlighed. Den lokale
basis er fortsat kun tidligere sammenligningsreference.

Et kort neutralforsøg kan vise numerisk konsistens og endpoint-opdatering,
men er ikke en optimeret kontrolpolicy, et lukket feedbackforløb eller døgnaccept.
Det er heller ikke det deterministiske januarbenchmark: horizon12→14, analytisk
B=.5+.1sin(2πt/24), A=.5 og egne hour12-filterposteriorer er en anden fixture.
Rå J fra de to forsøg må derfor ikke bruges til at rangere BM mod WM.

## Hvilken matematik genbruges og hvorfor

For model M er den primære deterministiske sammenligning
\[
 \min_{u\in[0,1]^4}J_M(u),\qquad
 J_M(u)=\int_0^{24}[q_M(t;u)-A(t)]^2dt,
 \quad E_M(u)=\int_0^{24}[q_M(t;u)-B(t)]dt=0.
\]
Samme modeldemand og kvadraturvægte bruges i J og E. Dermed kan modellen
flytte forbrug mellem tidspunkter, men ikke forbedre tracking ved at ændre
nettomængden. Ligheden udtrykker kravet direkte. En straf λE² ville kræve
en ny λ og tillade mængdeafvigelse; det er en anden opgave. Den oprindelige
opgave uden denne lighed er allerede bevaret som separat reference.
WM's affine QP-certifikat og BM's lokale SLSQP-resultater bevares med deres
forskellige optimalitetsclaims. Der foretages ingen ny søgning her.

Det stokastiske originalmål er med cutoff-vægte w_j
\[
 J_s(u)=\sum_jw_j\int E_j[(D_t(u)-A_t)^2]dt.
\]
Skriv μ_j=E_j[D], v_j=Var_j(D) og μ=Σ_jw_jμ_j. Kvadratudvidelse giver
\[
 E_j[(D-A)^2]=(\mu_j-A)^2+v_j.
\]
Da Σ_jw_j(μ_j−μ)=0, følger
\[
 J_s(u)=\int\!\left[(\mu-A)^2+
        \underbrace{\sum_jw_jv_j}_{\text{within}}+
        \underbrace{\sum_jw_j(\mu_j-\mu)^2}_{\text{between}}\right]dt.
\]
Det er totalvariansloven, ikke en ny regulariseringsstraf. Optimering af kun
(μ−A)² ville udelade begge variansbidrag. Et gennemsnitligt rho-plug-in ville
heller ikke bevare parameter/state-afhængigheden i de egne conditional laws.
En forventet energilighed ∫(μ−B)=0 ville være en *separat senere udvidelse*:
den sikrer ikke samme energi for hver stokastisk realisation. Vi indfører
hverken den eller en ny pathwise begrænsning i dette måleforsøg.

M254 bevares præcis som vejlederens normaliserede Bernstein/Beta-familie:
\[
 p_j(x)=\sum_{i=0}^{254}c_{ji}b_{i,254}(x),\qquad
 b_{i,m}(x)=(m+1){m\choose i}x^i(1-x)^{m-i},
 \quad c_{ji}\ge0,\ \sum_i c_{ji}=1.
\]
Basis, egne c_j og roundoff-certifikater genbruges uændret. For Monte Carlo
samples derfor I~Categorical(c_j), X0~Beta(I+1,255−I); vi genbruger ikke en
ny Beta(20,20)-prior eller FV-celler som M254-koefficienter.

EM genbruger den eksisterende diskretisering
\[
 X_{n+1}=X_n+b(t_n,X_n,u_n)h+\sigma(X_n)\Delta W_n,\qquad
 \Delta W_n\sim N(0,h).
\]
Fine h=1/384h og coarse h=1/192h bruger samme startsamples og
ΔW_coarse=ΔW_fine,1+ΔW_fine,2. Koblingen reducerer støj i tidsforskellen uden
at ændre marginalsandsynlighederne. Ingen clipping eller reflection tillades:
en faktisk randoverskridelse stopper forsøget. 256 paths på hver af fem noder
bevares; vægte fryses under open-loop neutralforward.
For nodevise pathcosts Q_jl og uafhængige samples på tværs af j:
\[
 \widehat J=\sum_jw_j\overline Q_j,\qquad
 \widehat{\mathrm{SE}}=\sqrt{\sum_jw_j^2s_j^2/256}.
\]
Det følger af variansadditivitet for en stratificeret vægtet middelværdi.
Dette er ikke SE for iid tilfældigt valgte rho-noder. Gates
|J_EM−J_FP|≤max(.001,4SE), |J_fineEM−J_coarseEM|≤.001 bevares;
4SE er et praktisk samplingcheck, ikke en simultan konfidensgaranti.

En ny syntetisk punktobservation ved14 vælges på forhånd fra første fine
neutral-path ved rho=.25 plus uafhængig støj med den ejede sigma_y. Seeds
20261021/22/23, nodeindex2/pathindex0 fastholdes. Den gamle prototypemåling
under en valgt stochastic-policy genbruges ikke under den neutrale pris.
Alle fire backend/grid-opdateringer modtager samme nye y og den faktisk
anvendte neutrale pris. Bayes er
\[
 Z_j=\int L(y\mid x,u_{14})p_j^-(x)dx,\quad
 p_j^{raw,+}=L p_j^-/Z_j,\quad
 w_j^+=\frac{w_jZ_j}{\sum_kw_kZ_k}.
\]
Evidence beregnes før M254-stateprojektion, så numerisk LS ikke ændrer
definitionen af parameter-likelihood. Ingen ekstra forward er nødvendig
ved selve endpoint-opdateringen. Dette demonstrerer kun punktobservations-
håndoff; historiske timeintervaller kræver stadig deres egen observationsoperator.
En opdateret origin er ikke i sig selv et udført næste kontroltrin.

## Hvad der konkret skal genbruges

De tidligere ti completed nodeforwards og begge bankfiler fra execution
872e77714d1bc5848d8dce5b1fac058417b796cf bevares read-only. Arkiv- og
bankhashes, begge kontrakter og inputidentiteter står i
g7/auxiliary_integration_design_20261007.json. Original ledger inkluderer
setup og native jobs; kampagnens samlede debit inkluderer også tidligere
fejl, walltid, eksport og review. En ny ledger seeds med *hele* seneste debit,
uden at debitere de gamle jobs igen. Originalledgeren overskrives ikke.

Genskab evaluatorgeometri med en ny setupcharge. Verificér alle41 gamle
sourcehashes, egne origins/basis/coefficients, versionskontrakt, outputpartition
og præcis [.5,.5]-pris. Ny launcher får en separat provenance-manifest;
en ændret numerisk kontrakt må ikke få lov at låne gamle nodejobs ved kun at
sammenligne endpoint eller J. Brug original frossen numerical source.

Banker rekonstrueres gennem den oprindelige combine_nodes med de verificerede
nodepayloads, så evaluatorens ejerrelation bevares til assimilate. Et indlæst
JSON må ikke blot få et frit _owner-felt uden hele kontraktkontrollen.
Old NodeJobStore skriver ikke cross-campaign-imports; en read-only adapter
og dens ejerskabscheck skal derfor implementeres før launch.
Den gamle målelauncher kan ikke bruges som resume: den afviser eksisterende
invocation, og cached setup-receipts indeholder ingen levende Pythonobjekter.

FV257/FV513 starter hver fra det tilsvarende *egne M254-startmass-array*,
med samme rho/vægte, A/B, pris og numerik. Dette isolerer backendtransport;
det er ikke en separat historisk FV-filterposterior. EM starter fra den fine
kontinuerte M254-law, mens FV bruger dens integrerede cellemasser. Diskret
celle- versus kontinuert samplingforskel skal fortsat indgå i fejlfortolkningen.

## Kvadratur skal være en reel særskilt kontrol

src/control/objective.py:midpoint_quadrature returnerer horizonendpoints og
de oprindelige midtpunkter, ikke alle mellemliggende celledelingspunkter.
Begge gamle M254-forwards bruger samme objective_max_step1/96h; forskellen
mellem deres J er derfor ingen uafhængig objective-kvadraturforfining.

Hvis man gennemsnitliggør to gemte midpoint-costs og dobler deres vægt,
får man algebraisk samme sum. Det er ikke en ny quadraturtest. Finere midpoint-
quadratur kræver nye stateværdier på nye tidspunkter og en ny fuld kontrakt.
Outputpunkter påvirker transportens faktiske pause/projektionspartition,
så dette er heller ikke gratis dense output. Den måles senere som eget job
og forbliver ukendt i den uændrede døgnjobgraf. Ingen interpolation eller
omdøbning af den gamle J-difference kan lukke denne gate.

## Budget og konkret næste måleforsøg

Designreview og en read-only struktur-/aritmetikkontrol charges konservativt
10s. Tidligere1224.712123334s → **1234.712123334s**; rest
**565.287876666s** af1800s. F og WM's budgetter/status ændres ikke.

Det foreslåede nye scope er højst200s:150s worker +30s stop/eksport +
20s review. Det er ikke200s udført eller allerede forbrugt. Ved maksimal
scopecharge ville365.287876666s blive tilbage. Ingen ekstra allowance oprettes.

| Ny opgave | Antal | Reservation pr.opgave | Samlet |
|---|---:|---:|---:|
| Ny setup/rehydrering af fire evaluatorer | 4 | 10s | 40s |
| FV513 og FV257 nodeforwards | 10 | 2s | 20s |
| Coupled coarse/fine EM per parameter-node | 5 | 4s | 20s |
| Endpoint-feedback, fire komplette banker | 4 | 10s | 40s |
| Reduktion og interne algebraiske checks | 1 | 10s | 10s |
| Uallokeret worker-headroom | — | — | 20s |
| Stop/eksport og separat review | — | — | 50s |

Reservationerne er administrative stoprammer for en *måling af ukendt runtime*,
ikke målte rates, forventede tider eller matematiske worst-case-grænser.
En langsom feedbackprojektion kan derfor stoppe forsøget med bevaret deldata;
den giver ikke adgang til løsere tolerancer. De sættes ikke ind som24h-rates.
Ordren/dependencies og alle uændrede gates står i JSON-designet.
Se eksisterende G7_NODE_RUNTIME_REVIEW.md for measured2h M254-rates.

Før nyt job kræves hele reservationen i den persistente fælles ledger.
Én serial OS-locked ejer. Gem hver completed FV-node, coupled-EM-node med
fælles inputbytes og komplet feedbackbank atomisk før næste job.
Bankreduktion kræver alle fem egne noder; ingen partial bank-accept.
Parent hard stop skal bevare running reservationer og eksportere partials.
Afbrudte jobs kræver en særskilt eksplicit retryregistrering; ingen auto-retry.
Genoptagelse kan genbruge completed payloads, men levende setup genskabes
med en *ny* invocation-identificeret setupcharge.

Charge=max(worker-wall,ledgerincrement)+målt eksport+20s review;
overshoot registreres også. Stop/eksport-reserven må ikke bruges til flere
modeljobs. Et OS-stop har scheduling-overhead og er ikke et bevis for
et eksakt sekundloft; hvis scope overskrides, bevares den fulde faktiske charge
og nye jobs standses. Den foreslåede måling kan altså fejle uden budgetreset.
Source-versioner og launcher skal først fryses i en særskilt pre-run-registrering.
launch_ready, actual_optimizer_started og scientific_accept er false.

## Døgnopgaven og hvad der mangler

Den fulde14-rækkers24h-plan i bm_1_g7_budget.py ændres ikke. Ingen af dens
kritiske24h-rates er målt. Admission forbliver false.
Illustrationen1.25×3×12×(16.828349+37.961317)=2465.535s for M254-cross alene
er fortsat kun et lineært planscenarie, ikke en rate eller lower/upper-bound.
Det korte auxiliary-design lukker derfor ikke budgetkonflikten.

Rækkefølge efter dette design:
1. Implementér/review read-only restore, nye auxiliary receipts, parent/resume
   og præregistrér det korte neutrale målescope.
2. Udfør scope med stopreserve, eksportér også ved fejl og råreview FV/EM/feedback.
3. Brug faktiske rates til næste beslutning. Prisoptimering, samme-policy
   M254/FV/kvadratur, optimeret-policy-feedback og24h-admission er fortsat åbne.
   Et ændret døgnscope eller større budget kræver særskilt konkret beslutning;
   et2h-resultat kan ikke ommærkes til det oprindelige døgnkrav.
4. Afslut samlet matematisk/resultatmæssig overdragelse og skeln mellem
   syntetisk numerik, historisk forecast, fysisk prisrespons og kontrolbenchmark.

G7 har dermed tre større åbne dele: integreret kort runtimeprøve,
budgetmæssigt mulig policy-/døgnvalidering og samlet overdragelse. Tidligere
skøn2–4 timers aktivt G7-arbejde er kun betinget af, at dette lille design
kan bruges uden solveromskrivning. Det er ikke et tidsløfte om døgnaccept.
G-kernen4–8 timers aktivt arbejde inkluderer dette; faktisk beregning,
nyt helårsfit, ny historisk intervalestimation og større runtimeombygning
er ikke dækket. Sol High er passende; ingen lokal modelkommando kræves nu.

## Kilder, egne valg og faktisk kontrol

- Vejlederbasis/oprindelig FP-model: G8_POSTERIOR_BASIS_REVIEW.md's kildekort,
  FP2025 og tavlematerialet. Basisfamilien og legacy bevares.
- Deterministisk mål/mængdeudledning og oprindelig WM: G7_ENERGY_NEUTRAL_REVIEW.md,
  G7_MATCHED_CONTROL_REVIEW.md og deres konkrete kode-/materialekort.
- Kvadratudvidelse, totalvarians, Bayes og stratificeret samplingvarians:
  standardidentiteter, udledt ovenfor; ingen påstand om nye vejlederformler.
- EM og coupled Brownianincrements: samme eksisterende G7-protokol og kernel
  som G7_OWNED_HANDOFF_REVIEW.md; Higham2001 er den tidligere angivne
  EM-baggrund. Vi har ikke foretaget et nyt PDF-/tavlereview i dette arbejde.
- Neutral auxiliary-policy,200s scope, jobrækkefølge, allowances, read-only
  rehydrering og separat provenance er egne software-/forsøgsvalg. De
  erstatter ingen oprindelig matematisk model eller acceptgate.

Faktisk kontrol i dette scope: én stdlib-only struktur-/aritmetikkontrol af
den publicerede design-JSON mod gemte budget/rates/arkivmetadata og AST-læst
uændret tolerancedictionary. Dependencies, caps, false-acceptflags og original
døgnplans14 ukendte rækker er kontrolleret. Ingen import af native solver,
ingen nye forwards, observationer, optimizer, refits eller gamle tests.
Plan-/tekstændringer får ingen numeriske tests efter AGENTS.md-reglen.
Kildekode uden for G's status/dokumentation er uændret.
