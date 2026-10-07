# G7: egne nodecheckpoints og komplet budgetgraf

7. oktober 2026. Afsluttet implementering og afgrænset stubkontrol; **ingen ny
M254/FV/EM-propagation, fit, optimizer eller numerisk G7-accept**.
Forudregistrering: [nodejobplan](G7_NODE_JOB_CHECKPOINT.md).
Tidligere deterministiske resultater: [bevaret forbrug](G7_ENERGY_NEUTRAL_REVIEW.md).
Vejlederens normaliserede Bernstein/Beta-basis M254 er fortsat primær.
F/WM, deres budgetter og fælles numeriske kerner er uændrede.

## Matematik og kildegrundlag

Lad den gemte parameterbank ved informationscutoff være
\[
 p_0(x,\rho)=\sum_{j=1}^{5}w_j p_{j,0}(x)\delta_{\rho_j}(\rho),
 \qquad w_j\geq0,\quad\sum_jw_j=1.
\]
Hver node har sin EGEN conditional posterior og sine egne M254-koefficienter.
For en åben prisplan u holdes vægtene faste. Hver conditional law følger
den eksisterende transportoperator på samme integrationspartition:
\[
 \partial_t p_j=\mathcal L^*_{\rho_j,u}p_j,\qquad p_j(t_0)=p_{j,0}.
\]
Der er ingen ny observation under candidate-forward, derfor ingen ny
Bayes-opdatering af w. En efterfølgende faktisk observation hører til
feedbackarbejdet, ikke til den fremtidige objectiveberegning.

For \(\mu_j=E_j[D]\), \(v_j=\operatorname{Var}_j(D)\) og
\(\mu=\sum_jw_j\mu_j\), udvides kvadratet direkte:
\[
 \begin{aligned}
 \sum_jw_jE_j[(D-A)^2]
 &=\sum_jw_j\{v_j+(\mu_j-A)^2\}\\
 &=(\mu-A)^2+\underbrace{\sum_jw_jv_j}_{\text{within}}
   +\underbrace{\sum_jw_j(\mu_j-\mu)^2}_{\text{between}}.
 \end{aligned}
\]
Det sidste trin bruger \(\sum_jw_j(\mu_j-\mu)=0\).
Dermed gælder både kontinuerligt og på den fælles gemte objectivepartition:
\[
 J(u)=\sum_jw_jJ_j(u)
 =\sum_n\Delta t_n\left[(\mu_n-A_n)^2+
             \sum_jw_jv_{j,n}+\sum_jw_j(\mu_{j,n}-\mu_n)^2\right].
\]
Dette er kvadratudvidelse og totalvariansloven, ikke en ny straffunktion
eller en matematisk antagelse lånt fra tavlebillederne. Vejledermaterialet
leverer basisfamilien og de bevarede modelgrundlag; det konkrete kildekort
findes i [basisreview](G8_POSTERIOR_BASIS_REVIEW.md) og
[own-handoffreview](G7_OWNED_HANDOFF_REVIEW.md).
Nodecheckpoint og budgetgraf er vores egne software-/forsøgsvalg.

M254 anvender uændret den normaliserede familie
\[
 b_{i,m}(x)=(m+1){m\choose i}x^i(1-x)^{m-i},
 \quad i=0,\ldots,m,\quad m=254,\quad
 p_j(x)\approx\sum_i c_{j,i}b_{i,m}(x).
\]
Det eksisterende diskrete basisarray/projektionsforløb genbruges præcist;
den kontinuerte formel ovenfor er ikke en genberegning eller erstatning
af den gemte numeriske basis.

## Hvorfor checkpoint efter en hel node

Summen over parameterbankens noder gør nodeforwards uafhængige betinget
på den samme prisplan. Derfor kan hele nodebaner gemmes og bagefter
reduceres uden at ændre transportpartitionen. Vilkårlige nye timepauser
ville derimod kunne ændre steps/projektionspunkter. En gennemsnitlig
parameter eller FV-state må ikke erstatte en nodes egne c/posterior.

OwnedBankPriceEvaluator har nu evaluate_node og combine_nodes.
evaluate genbruger samme opdeling. Statisk AST-review viser:
- det oprindelige native mixture-transportkald er identisk;
- bankreduktionen fra weights og frem er identisk;
- assimilate er identisk.

combine_nodes kræver alle fem ordnede, ejede baner med samme priser/tider,
gyldige masser og initialtilstande samt konsistente mixture-koefficienter.
Den oprindelige objective- og totalvarianskontrol bevares.

CheckpointedBankControl er foreløbig begrænset til den eksisterende
analytiske G8-fixture B(t)=.5+.1sin(2pi*t/24), A=.5 og den oprindelige
native evaluator. Dette er den stokastiske syntetiske kontrakt; den
fælles deterministiske januar-BM/WM-opgave har sit særskilte frosne A.
Historiske B/A-callbacks kræver senere en immutable inputkontrakt.
Ingen påstand om automatisk support for vilkårlige cached/custom solvere.

## Identitet, crash og ejerskab

Jobidentiteten indeholder fuld typed origin, conditional mass/c, basis,
vægte, parameterkurver, backend, output-/prispartition, alle relevante
numerikfelter, eksakt prisvektor, NumPy/SciPy-versioner og 36 kildehashes.
Priser afrundes ikke. Nye priser får nye jobs. Kilder og evaluatorens
levende kontrakt kontrolleres også ved hver evaluate.

NodeJobStore bruger én kampagnemappe og en nonblocking OS-writerlås.
Seedens prior debit, samlede loft og max_new_seconds er persistent og
skal matche ved resume. To uafhængige mapper med samme seed er ikke en
fælles ledger; en launcher skal anvende den ene registrerede ejer/mappesti.

Rækkefølge:
1. Gem hel node-reservation i ledger før compute.
2. Beregn den samme native nodeforward.
3. Gem fuld bane, momenter, diagnostics og payloadhash med fsync/atomic replace.
4. Gem completion og faktisk elapsed tid i ledger.

Crash før trin3 efterlader reservationen som charge. Resume markerer
interrupted og stopper; kun et eksplicit retry-ID kan tilføje et NYT
forsøg uden at slette den gamle charge. Crash mellem3 og4 genfinder den
komplette hashverificerede fil og genbruger den uden forward; fuld reserve
beholdes konservativt. Manglende completed filer eller korrupte receipts
er stop, ikke cachemiss og automatisk replay. Completed genbrug har ingen
ny solvecharge; setup/audit/export koster stadig og skal budgetteres.

Deadline kontrolleres ved native driftkald og før/efter forward. En
lang SVD eller eksport kan overskride den kooperative deadline; faktisk
elapsed charge klippes aldrig ned. Dette er ikke et hårdt process-stop.
En kommende målelauncher skal have særskilt stop-/eksportreserve.
Linux-lås/crashmekanik er kontrolleret; Windows-koden er ikke kørt på PC.

## Komplet døgnjobgraf og budget

Den tidligere ufuldstændige admission-formel bevares som historik og må
ikke frigive G7. Den nye graf har 14 eksplicitte jobtyper:

| Arbejde | Antal hele jobs |
|---|---:|
| Setup | 1 |
| Deterministiske objectivekald | 48 |
| FV-objectivekald | 48 |
| M254 coarse/fine policychecks | 3 + 3 |
| FV coarse/fine policychecks | 3 + 3 |
| Quadraturchecks | 3 |
| Coupled EM, 256 paths/node, coarse/fine | 3 |
| M254 coarse/fine feedback inkl. update/projektion | 1 + 1 |
| FV coarse/fine feedback inkl. update | 1 + 1 |
| Audit/eksport | 1 |

Antal er prototype-lofter, ikke udførte jobs eller SciPy-iterationer.
Grafen er 24h/fem noder/tre policies; ingen ny stochastic energy-neutral
udvidelse er smuglet ind. Ingen cache-rabat før konkret råidentitetsaudit.

For målt pris r_k for et helt job og antal N_k er planoverslaget
\[
 T_{\mathrm{plan}}=1.25\sum_kN_kr_k,\qquad
 \mathrm{budget\_admitted}=
 [\text{alle rates råreviewet og matcher job/horizon}]
 \land[T_{\mathrm{plan}}\leq T_{\mathrm{rest}}].
\]
1.25 er en planreserve, ikke en bevist worst-casegrænse. Ingen automatisk
2h→24h-ekstrapolation; den gamle afbrudte finepilot er ikke en completed
rate. Rate-provenancefelter kontrolleres, men den eksterne råaudit skal
udføres, før raw_reviewed sættes. En flagværdi alene er ikke råevidens.

Alle 14 rates er aktuelt ukendte; budget_admitted=false,
scientific_accept=false og actual_optimizer_started=false.
Gemte plan/admission: root/g7/node_job_budget_20261007.json.
Tidligere G-debit1098.580078s + denne konservative charge20s =
**1118.580078s** af1800s; **681.419922s tilbage**. F-budget ændres ikke.
De20s dækker den korte kontraktkontrol, review/eksport og let syntax-CI;
arbejdstid for implementering er ikke en målt modelrate.

## Faktisk kontrol og bevaret evidens

Én fokuseret integreret testmetode blev kørt to gange. Første run bestod;
review fandt manglende persistent delundersøgelsesloft. Den konkrete
rettelse begrundede anden invocation; første receipt er bevaret separat.
Processider:2.0963406199998644s og2.033033269000043s, i alt4.129373889s.
Begge bruger stubtransport, ingen faktisk FP/M254/FV/EM-simulering.

Kontrollen dækker interruption midt i node, eksplicit retry uden charge-
reset, completed genbrug, crash efter artifact før ledgercompletion,
egne forskellige c, unchanged weighted objective/within+between,
ændrede priser/kilder/numerik, to writerlås, payloadkorruption og studycap.
Fuld jobgraf med ukendte rates stopper admission; falsk2h-rate og fjernet
FV-finerække afvises. Toy-rates i testen er tydeligt stub og ingen målinger.

Sourcecheckpoint før sidste test:
de0d507fb8538d79999a00f617ff36f2425a92c9.
Final SHA256-manifest:
7ca1a846a9809c72c358a740bd611a4a5d2cdac297c8517b93b9039184526d07.
Receipts og statisk review ligger i root/g7/node_job_check*_20261007.json
og node_job_static_review_20261007.json. Testens midlertidige dummy-
nodefiler er ikke videnskabelige modelartefakter og blev ikke arkiveret.

## Resterende arbejde og næste trin

Implementeringsdelopgaven er færdig; stokastisk G7 er fortsat åben.
Næste er et særskilt registreret, kort faktisk node-/runtimeforsøg med de
allerede gemte EGNE M254-origins, samlet allowance, exportreserve og
immutable source. Før start skal de gamle råoriginfiler hashverificeres
fra det bevarede pilot-/filterarkiv og inputprovenance fryses.
Ingen nye priors, basisændringer eller gentagelse af færdige filtre.

Mål relevante completed rates; hvis den komplette graf ikke kan
underbygges inden for restbudgettet, stop og dokumentér budgethindringen.
Et kort nodeforsøg er ikke automatisk døgn-admission. Først derefter
kommer policyoptimering/krydsbackend/quadratur, coupled EM og rigtig
observation→feedback med samme priser. Historisk årsfit/kontrol samt
kausal prisrespons er særskilt åbne; de deterministiske gemte resultater
må ikke ommærkes til historisk/stokastisk accept.
Sol High er passende til næste afgrænsede måle-/implementeringstrin.
Ingen lokal modelkommando kræves af denne leverance.
