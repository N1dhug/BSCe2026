# G7: afsluttet knuderecovery og fuld tidsopdelingssammenligning

Johan godkendte højst110 ekstra beregningssekunder den7. oktober2026.
Denne status erstatter den tidligere aktuelle partial-status, men de gamle
registreringer, fejl, journaler og charges bevares. PR114 er allerede merged
som66991a0. PR115 indeholder udviklingen; PR115 er ikke mergegodkendt.

## Konkret udført

Recovery-launcher og separat registration blev committed som
f208bbc3eb774d2f201c31f1243b85969ffc552d før invocation; registrationSHA256
97148f853a01a1a5d7047c2019529f55498c6a7f729275374d5f78fbe6a71e8b
blev læst tilbage fra Git. Launcher/manifestbytes er frosset i
`root/g7/bounded_quad_recovery_20261007/`.

Kun quadnode3 og4 fik nye numeriske forwards. Alle13 tidligere completed
forwards (10 coarse-kandidatknuder,3 quadknuder) blev genbrugt read-only.
Derudover genbruges de tidligere10 neutrale M254-knuder og5 FV-knuder.
Gamle journaler blev ikke genoptaget; den interrupted18s reservation står
uændret i originaljournalen, og det fulde tidligere273s tabsscope refunderes
ikke. Samtlige41 native numerical sourcehashes og begge originale own-inputs
er kontrolleret. Ingen src-, test-, workflow-, F- eller WM-kode er ændret.

Én quad-evaluator blev rehydreret. Tidligere coarse/fine/FV-banker blev
reduceret af den uændrede native `combine_nodes` med lette supports fra
frosne kontrakter; det konstruerer ingen nye projektorer eller forwards.
Den verificerer alle egne startfordelinger, hele masse-/koefficientbaner,
pris-/tidsidentiteter, nodeomkostninger og totalvarians. Sammenligningernes
reference-evaluator bruges kun til samme513-grid-models terminale state/
demand-momenter, CDF'er og kvantiler, aldrig til propagation.

Parent var aktivt overvåget til slut og skrev progress løbende. Den første
ZIP-eksport viste sig ufuldstændig ved slutkontrol. Årsagen er ikke bevist.
ZIP blev genopbygget uden solverimport eller nye forwards, fsyncet med sin
mappe og kontrolleret med CRC. Den gemte version har samme verificerede
slut-SHA256 som den lokale fil; den fejlbehæftede eksport er erstattet.

## Resultater og uændrede gates

| Neutralplan[.5,.5],12→14 | J |
|---|---:|
| M254257,192 midpoints | .003741963573462555 |
| M254513,192 midpoints | .0037155174638558177 |
| M254513,384 midpoints | .003715523422208866 |
| FV513,192 midpoints | .0037153098107086916 |

| Par | Absolut J-forskel | Maks conditional terminal-L1 | Gate |
|---|---:|---:|---|
| coarse→fine | 2.64461096067e-5 | .0111877516183 | bestået |
| fine→384-midpoint | 5.95835304837e-9 | 9.28648394581e-7 | bestået |
| fine→FV | 2.07653147126e-7 | .000256225638651 | bestået |

J-gate er stadig `max(.001,.02*abs(referenceJ))`; terminal masse-L1≤.05,
state/demand-mean≤.005, variance≤.002, CDF≤.03, quantile≤.02. Alle rapporterede
marginal-/conditional- og terminalgates består. Den lille tidsopdelingsforskel
er empirisk stabilitet for denne plan, ikke en universel fejlgrænse.

Den neutrale plan er fortsat bedst blandt de tre komplette coarse-planer:
J([.3,.7])=.017303375697677884 og J([.7,.3])=.010655853331904826.
Ingen ny prisudvælgelse er udført på finer grid. Ingen forbedring eller
kontinuert global optimalitet er dokumenteret. Tilts blev ikke fine-valideret;
vi kan alene erklære deres coarse-rangering, ikke fin-grid optimalitet.

## Matematik og begrundelse

For hver frosset parameterknudeρ_j har vi dens egen cutoff-statefordeling.
Parameterweightsw_j≥0,∑w_j=1 holdes fast under prisforward; fremtidige
observationer indgår ikke i kontrolmålet. Lad μ_j(t)=E[D_t|ρ_j],
v_j(t)=Var(D_t|ρ_j), μ(t)=∑w_jμ_j(t), og et deklareret målA(t).
Da E[(D-A)^2|ρ_j]=v_j+(μ_j-A)^2, fås

\[
\begin{aligned}
J(u)&=\int_{12}^{14}\sum_jw_j E[(D_t-A(t))^2\mid\rho_j,u]dt\\
&=\underbrace{\int(\mu-A)^2dt}_{\text{middelfejl}}
+\underbrace{\int\sum_jw_jv_jdt}_{\text{inden for parameterknuder}}
+\underbrace{\int\sum_jw_j(\mu_j-\mu)^2dt}_{\text{mellem parameterknuder}}.
\end{aligned}
\]

Den sidste lighed følger ved μ_j-A=(μ_j-μ)+(μ-A) og
∑w_j(μ_j-μ)=0, så krydsleddet forsvinder. Det er totalvariansloven,
ikke en valgt ekstra straf. Objektivet beregnes fra første- og andenmomenter
på samme positive midpointweightsΔt_k; samme identitet gælder derfor også
for den diskrete sum. Ingen separat middelmålskørsel er nødvendig for at
rapportere middelfejl fra disse forwards. Et tidligere egentligt
deterministisk trajectorymål er stadig en anden modelberegning i en
ikke-lineær model og bliver ikke erstattet af μ alene.

For hele384-midpointbanken er

\[
J=0.0022613178834201752
+0.0013782185162290082
+0.00007598702255962573
=0.003715523422208866.
\]

Det første led er middelfejl; de to næste summerer til varians
.0014542055387886908. De samme gemte moments og weights reproducerer
identiteten i en stdlib-review med afvigelse under1e-10. Masse og
koefficientsummer afviger højst5.55e-15 fra1 gennem alle fem baner.
De to tilts sænker coarse-variansen, men øger middelfejlen endnu mere;
det forklarer deres dårligere samlede J. Variansreduktion alene er ikke
forbedring efter det valgte forventede kvadratiske tab.

Prisfamilien uδ=(.5+δ,.5−δ),δ∈{0,−.2,+.2}, har fast gennemsnitspris
for de to lige lange blokke. Det er ikke bevaret forventet energi. For den
fine neutrale plan er ∫(E[D]−B)dt=−.010345067191684047 i modellens
normaliserede demand×hour-enheder. Energinormalisering til fysiske enheder
kræver den separat fastlagte skala. Der er ikke tilføjet en energilighed.

Egne conditional laws beskytter parameter–state-afhængigheden. Faste
cutoffweights beskytter informationsgrænsen. M254 viderefører den valgte
supervisorbasis; FV er et særskilt transportreferencebackend og kontrollerer
at resultatet ikke kun skyldes mixtureprojektion. Finere stategrid og flere
midpoints undersøger hver sin numeriske følsomhed uden en dyr ny søgning.
Den finere midpointpartition ændrer også stop-/projektionspunkter i
M254-transporten, selv om solvermaxstep er uændret: forskellen må ikke
omdøbes til en ren kvadraturfejl eller et Richardson-certifikat.

Dette er nyttigt for projektet, fordi vi får kontrolmålets middelfejl og
usikkerhedsbidrag under samme input/cutoff/pris, plus en kontrollerbar
numerisk reference. Et middelmål alene taber netop de to variansled.
Tre kandidatplaner er et budgetafgrænset metodeforsøg; de giver ikke den
optimeringsdækning, historiske validering eller fulddøgnskontrol, som en
endelig anvendelighedsvurdering kræver. BM færdiggøres før direkte WM-
sammenligning; ingen nye WM-resultater er indført i denne recovery.

## Hvad der endnu ikke kan accepteres

Den nye SolverAudit dækker kun node3/4:6916 checked projections,
max projectioncellmassL1=1.6517018793543928e-6≤.05, max scaled gap≈1e-11,
max coefficient mass error1.33e-15. Projection-fit auditmaxima for de gamle
quadnode0/1/2 blev ikke gemt før crashet. Gemte native diagnostics har
masse-, koefficient-, clipping-/renormalization- og timestepchecks, men
ikke forskellen til hvert raw target før projektion. Massebevarelse kan
ikke erstatte denne L1-kontrol; en god projektion kan stadig være unøjagtig.
Den tabte audit kan ikke udledes af slutfordelingen alene.

Derfor: `quad_bank_complete=true`, `forward_gates_passed=true`,
`new_projection_gate_passed=true`, men `quad_projection_audit_complete=false`,
`numerical_validation_passed=false`, `scientific_accept=false`,
`improvement_screen_passed=false` og `full_day_budget_admitted=false`.
Ingen completed forward genkøres for at skjule dette hul. Næste nødvendige
kodearbejde er at gemme projectionaudit atomisk sammen med hver node,
uden at ændre numerikkernen. En eventuel særskilt auditkørsel kræver sit
eget konkret beskrevne scope; denne110s-recovery giver ikke adgang til
replays eller et nyt prisforsøg.

## Budget, faktisk kontrol og råevidens

| Debit | Sekunder |
|---|---:|
| Bevaret tidligere G-debit | 1799.1415078599962 |
| Konservativ forberedelse/check | 10 |
| Parent/worker wall, større end ledgerincrement19.014800452 | 21.533936035993975 |
| Første parenteksport | .661827962998359 |
| Bevaret rawreview/finaleksport-reserve | 10 |
| Ekstra charge i alt | 42.195763998992334 |
| Nyt kumulativt G-forbrug | 1841.3372718589885 |
| Rest af godkendt loft1910 | 68.66272814101148 |

Registreret70s workerhardstop/100s execution blev ikke ramt. Jobwalls var
verify2.958s, quadrehydration2.751s, node3 7.008s, node4 6.929s,
fuld reduktion/parrapport.376s. Der er fem nye complete journalentries og
præcis to nye forwards. Native source og originale inputs er uændrede.
Rawreview/eksport inklusive genopbygning og efterfølgende CRC-kontrol tog
4.784623874s, dækket af den bevarede10s reserve. Ingen gamle charges er
refunderet, og de resterende68.66s er ikke automatisk brugt på nye forsøg.

Det fulde slutarkiv er `G7_bounded_price_complete_quad_review_20261007.zip`,
38829366bytes, SHA256
e546e94fd23e1cf884cb4a5a17ee90921e62b4df400f29d37cf121240914354b.
Det indeholder89 files: nye receipts, alle fem quadraturebaner gennem
3 read-only/2 nye receipts, neutral coarse/fine/FV-bank, own-inputbytes,
41 frosne sources, fulde kontrakter/registreringer, journaler, moments/
terminalorigin, logs, budget og review. De10 uvalgte coarse-receipts er
allerede i det bevarede partialarkiv og hashmanifest og gentages ikke.
Den udvidede arkivindex indeholder timing, biblioteksidentitet og CRC-status.

Kommando:

```bash
python -B g7_bounded_quad_recovery_run/launch_quad_recovery.py \
  --output g7_bounded_quad_recovery_run --previous g7_native_recovered \
  --seed g7_seed_repair_run --partial g7_bounded_price_inputs_fixed_run
```

Dette dokument, de frosne launcherbytes og summary/indexfiler ligger i Git.
Der kræves ingen lokal numerisk kørsel for at læse eller reviewe resultatet.
