# G7: konkret forslag til samlet syntetisk kontrolforsøg

7. oktober 2026. Forslag, ikke launchregistrering. Ingen nye forwards
eller optimeringer er startet ved udarbejdelsen.

## Spørgsmål og afgrænsning

Kan G8's egne betingede state-/parameterposteriorer bruges i en
24-timers single-shooting-kontrol, med både deterministisk reference
og stokastisk politik, fælles numerisk krydsevaluering og én faktisk
observation/opdatering efterfulgt af et nyt prisvalg og forward?

Det er en syntetisk algoritme-/numerikleverance. Historisk kausalitet,
profit, historiske timeenergi-likelihoods og direkte WM-sammenligning
indgår ikke. Ingen ny målprofil tilpasses efter forsøgets resultat.
En stokastisk politik behøver ikke vinde over neutral eller deterministisk
for at demonstrere en korrekt integrationskæde.

## Frosset problem

Genbrug den deklarerede analytiske baseline og target fra den eksisterende
native fixture: B(t) er den eksisterende baseline, A(t)=0.5.
Tid er timer, demand og pris er normaliserede; J har enhed normaliseret
demand i anden gange timer. Ingen fysisk kWh- eller kronerfortolkning
uden særskilt skala. Brug egne G8-origins ved t0=12, ikke et nyt filterfit.
Hver backend/grid beholder sin egen conditional state og sine egne vægte.
Det nye åbne horizon er [12,36]; egne origins og source/inputhashes skal
fryses i den senere faktiske launchregistrering.

Fire konstante prisblokke, edges [12,18,24,30,36], bounds [0,1].
Neutral u=(.5,.5,.5,.5). Terminal- og glathedsstraffe er nul.
Ingen stokastisk energilighed tilføjes. Den tidligere deterministiske
variant med bevaret døgnforbrug bevares separat; lige middelpris er
ikke bevaret forventet forbrug.

Den deterministiske policy fittes med den native deterministiske
controller fra deklareret posterior-middelstate og posterior-middelrho,
med sigma=0. Det er eksplicit en plug-in-reference, ikke bankens
eksakte middelbane. Den stokastiske policy fittes med egne M254257
conditional states og den fulde within-/between-varians.

Brug den eksisterende bounded Powell/single-shooting-rutine, højst to
starter per fit: (.5,.5,.5,.5) og (.3,.7,.3,.7). Højst 24 objective-kald
per start og 48 per fit, altså højst 96 samlet; alle nodeforwards tælles
derudover. De gamle samlede 100-kald/300-sekunders optimizerlofter er
stadig øvre grænser, ikke en garanti for at M254-søgning kan gennemføres.
En afbrudt søgning rapporteres som afbrudt med den faktisk udførte
evaluering; den ommærkes ikke som completed optimization.

Identiske kandidater kan genbruges under eksakt kontrakt. Ingen fine/FV
reselection eller ekstra starter efter eksponering af valideringsresultater.
Inden for completed coarse-kandidater vælges laveste J; eksakte ties
afgøres af først registrerede kandidatrækkefølge. Kun observeret lokalt
resultat rapporteres, aldrig kontinuert/global optimalitet.

## Samlet jobgraf

| Job | Højst antal whole jobs | Definition |
|---|---:|---|
| setup | 1 | Source/origin/input/receipt-preflight |
| search_deterministic | 48 | Ét deterministisk objective-kald |
| search_m254 | 48 | Ét fem-hypotese M254257-bankobjective |
| cross_m254_coarse | 3 | Neutral/deterministisk/stokastisk, 24h |
| cross_m254_fine | 3 | Samme politikker, M254513 |
| cross_fv_coarse | 3 | Samme politikker, FV257 |
| cross_fv_fine | 3 | Samme politikker, FV513 |
| quadrature | 3 | M254513 med dobbelt costpartition |
| em_coupled | 3 | 256 baner/hypotese, to tidslevels, tre politikker |
| feedback_update | 4 | Samme syntetiske y; egne fire backend/grid-priorer |
| feedback_policy_search | 3 | Tre M254257-bankkandidater på ny posterior, 2h |
| feedback_m254_coarse | 1 | Valgt ny pris, egen opdateret origin, [36,38] |
| feedback_m254_fine | 1 | Samme valgte pris, egen fine posterior |
| feedback_fv_coarse | 1 | Samme valgte pris, egen FV257-posterior |
| feedback_fv_fine | 1 | Samme valgte pris, egen FV513-posterior |
| audit_export | 1 | Rawreview, tabeller, journal, durable arkiv |

Counts er konservative maksimummer før verificeret genbrug.
Et bankjob omfatter alle fem egne hypoteser; det er ikke en enkelt
billig node. Setup/export og feedback-opdateringer har andre rateenheder.
24h og 2h må ikke låne hinandens rates.

Dette er et eksplicit v2-forslag til den gamle 14-rækkers budgetgraf:
search_fv erstattes af search_m254, og feedback-update/search synliggøres.
Den gamle native full_job_plan/price_job_plan og dens source ændres ikke.
Dens validator afviser denne v2-graf; den kan ikke bruges som skjult
admission. En senere launcher/budgetregistrering skal håndtere v2
udtrykkeligt. FV erstatter ikke primær M254-søgning af runtimehensyn.

## Numeriske kontroller og feedback

Coarse: 257 celler, h_max=1/96h. Fine: 513 celler, h_max=1/192h.
Costpartition 1/96h; den separate refinement er 1/192h, samme fine
h_max. M254 er stadig grad254/255 koefficienter på begge grids.
Costpartitionændringen ændrer også transportens output/pausepunkter;
den er ikke et rent kvadraturfejlcertifikat.

Alle tre politikker skal have komplet gemt projektionsaudit og de
eksisterende forwardgates. Projection cell-mass L1<=.05,
conditional/marginal L1<=.05, mean<=.005, variance<=.002,
CDF<=.03 og quantile<=.02. J-gaten er
abs(Jleft-Jref)<=max(.001,.02 abs(Jref)); masse 1e-9 og
checkpoint 1e-10. Ikke blot terminal J: bevarede fulde arrays,
origin-ejerskab og within-/between-identitet skal kontrolleres.
Tolerancerne er udviklingens interne numeriske gates, ikke universelle
fejlgrænser eller i sig selv en forbedringspåstand.

EM bruger eksisterende native drift/diffusion og samme costpartition,
256 paths pr. hypotese, h_coarse=1/192h og h_fine=1/384h. De tre
politikker deler initialdraw og Brownian-increments pr. hypotese/bane;
coarse-increments er summen af to fine. Separate role-/hypotese-streams,
ny entropy20261008, observation role separat. Ingen automatisk flere baner.
Initialstate samples fra den egne M254-blanding; FV-cell masses må ikke
fortolkes som M254-koefficienter. Raw boundary violations skal være nul;
ellers afvisning, ikke en ny clipping-policy.

For parret forbedring mod neutral:

$$
\widehat\Delta=\sum_jw_j\overline{(Q_j^{neutral}-Q_j^{policy})},\quad
\widehat{SE}_{paired}=
\sqrt{\sum_j w_j^2s_j^2(Q_j^{neutral}-Q_j^{policy})/256}.
$$

Uafhængige PRNG-streams mellem hypoteser er den deklarerede
samplingantagelse. SE dækker sampling betinget på den frosne bank,
ikke diskretiseringsbias eller bankestimation. Rapportér paired SE og
diskretiseringsforskelle; EM/FP-gaten for hver J er max(.001,4SE),
coupled EM-level-difference<=.001. En lille positiv Delta alene er
ikke tilstrækkelig evidens for en nyttig gevinst.

Brug den stokastiske 24h-policys gemte fine EM-bane0 ved
prædeklareret truth-hypotese indeks2 til én ny endpointobservation ved36:
y=D_truth+epsilon, epsilon~N(0,sigma_y²), separat observationsstream.
Samme y går til fire backends' egne predicted priors.
Bayes-evidence beregnes før posteriorens M254-projektion:

$$
Z_j=\int L_j(y|x)p_j^-(x)dx,\quad
w_j^+=w_j^-Z_j/\sum_kw_k^-Z_k,\quad
p_j^{raw,+}=L_jp_j^-/Z_j.
$$

Gem rå evidences, logsumexp-reduktion, posterior, observation/price-side,
update-projektionsdiagnoser og own-origin-hash samlet i feedbackreceipt.
Kontrollér vægtdifference<=.03, log-evidence-difference<=.05,
conditional L1 og posterior projection-L1<=.05. Punktobservationen
er syntetisk; denne operator er ikke historiske timeenergidata.

På den opdaterede M254257-origin vælges én af de tre erklærede nye
2h-planer [.5,.5], [.3,.7], [.7,.3] efter samme coarse-J/tie-regel.
Fine/FV bruger denne valgte pris fra deres egne posteriorer. Fremfør
fra36 til38 og gem de fire auditerede resultater. Dermed er næste
forward faktisk initialiseret fra opdateringen og prisvalget, i stedet
for blot at vise et posteriorhåndoff. Den korte finite søgning er ikke
en anden døgnoptimering eller et stort MPC-studie.

## Budget og adgangskriterium

G-debit efter adapterkontrollen er 1846.3372718589885 s af1910;
rest63.662728141011485 s. Ingen af v2-grafens relevante whole-job-rates
er målt på det registrerede horizon/problemet. De eksisterende 2h
M254-bankmålinger er kontekst, ikke 24h-rates; ingen lineær skalering
eller ukendte nulrates giver admission.

Forslaget gemmer ukendte rates som null. Et konkret senere tilbud skal
have source/input/own-origin-identiteter, målt repræsentativ whole-job-rate
eller eksplicit ny begrænset målescope, alle jobcounts, stop-/eksport-/
reviewreserve og samlet hardcap. 1.25 sikkerhedsfaktor er en
planlægningsreserve, ikke en worst-case-garanti. Gamle optimizer-/
validationlofter og resten af kampagneloftet gælder fortsat.

budget_admitted=false, launch_ready=false og scientific_accept=false.
Ingen ny måle-/døgnkørsel eller automatisk retry er godkendt af designet.
Næste implementation er v2-integration/registrering med både forward-
og update-receipts; næste numeriske kørsel kræver en konkret ramme.
Tidligere completed forwards genkøres ikke for at udfylde dette forslag.
