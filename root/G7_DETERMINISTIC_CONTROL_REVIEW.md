# G7: deterministisk BM-prisstyring er primær reference

Johan, 7. oktober 2026. Selvstændigt G-spor; F4-H2 og WM ændres ikke.
Base: `1644a55a77c50be0c25cadadb1f7513600dab7a2`, med de to nye
G7-deterministiske Python-filers eksplicitte SHA256 i forsøgsmanifestet.

## Beslutning og vigtig korrektion af anbefalingen

Begge kontrolformuleringer beholdes. Det deterministiske kontrolmål har
førsteprioritet som BM-reference og til den kommende fælles sammenligning
med WM. Stokastisk G7 med FP/mixture, fuld parameterbank og kort feedback
er fortsat en bachelorleverance; det er ikke gjort valgfrit eller afsluttet.
G8's accepterede syntetiske M254-filter og det negative G7-runtimeforsøg
bevares. De skal ikke genkøres for denne deterministiske leverance.

Den foregående chat anbefalede en prisafhængig BM-indkøbsfunktion A(U) som
nødvendig næste ændring. En ny kontrol af originalmaterialet viser, at det
ikke er nødvendigt for BM's oprindelige kontrolproblem: FP2025, afsnit
3.2.3, fysisk PDF-side15 / trykt side14, definerer allerede kendt, exogent
A(t) og optimerer kun priserne. Projektbeskrivelse111866 angiver ligeledes
styring mod en på forhånd ønsket samlet forbrugsprofil. Derfor følger denne
pilot originalens faste A(t). Ingen WM-volumen/indkøbsmap oversættes til
BM's SoC, og der opfindes ingen nye gamma-koefficienter.

WM4-noternes deterministiske mål sum(A_k(U)-I_k(U))^2 er beslægtet, men
A_k kan afhænge af både pris, input og WM-state. Samme symbol A betyder
derfor ikke automatisk samme benchmarkopgave. Før en BM/WM-rangering skal
man erklære enten en fælles fast målprofil, også for WM-referencepolitikken,
eller en særskilt begrundet fælles indkøbskontrakt. WM's oprindelige
køb–forbrug-opgave bevares separat. Ingen WM-policy eller modelrangering
er gennemført her. Den fælles fast-målprofil-reference er næste opgave;
den kræver ikke nye historiske parameterfits.

## Matematik, direkte kildegrundlag og egne numerikvalg

Original BM1 genbruges uændret:

\[
\dot x(t)=\frac{D(x(t),U(t),B(t))-B(t)}{C},\qquad x(0)=x_0,
\]
\[
\delta=\frac{2}{1+\exp[-k(f(x)+g(U))]}-1,
\quad
D=B+\Xi\delta\left[(1-B)\mathbf1_{\delta\ge0}+B\mathbf1_{\delta<0}\right].
\]

Kun u=(u_0,...,u_3) vælges; U(t)=u_b på hver seks-timersblok.
C,k,Xi,f,g,x0,B,A er frosne og fitter ikke til kontrolmålet.
Det oprindelige deterministiske trackingproblem er

\[
\min_{u\in[0,1]^4}J_{\rm det}(u),\qquad
J_{\rm det}(u)=\int_0^{24}[D(x^u(t),u(t),B(t))-A(t)]^2dt.
\]

Single shooting betyder, at kun prisvektoren er beslutningsvariabel;
hele ODE-banen integreres fra samme x0 ved hver kandidat. Tidligere
kandidaters slutstate bruges aldrig som næste kandidats start.

Genbrug eksisterende midpoint-quadratur, der deler præcist ved prisbrud:

\[
J_h(u)=\sum_q h_q[D(x^u(t_q),U^u(t_q),B(t_q))-A(t_q)]^2,
\quad
r_q(u)=\sqrt{h_q}[D_q(u)-A_q],\quad J_h=r^Tr.
\]

q indekserer quadraturceller, ikke de fire prisblokke. Kvadratroden er
en identisk omskrivning: uden den ville ulige tidsvægte ændre problemet.
Ingen straf, robust tabsfunktion eller sandsynlighedsvægt er tilføjet.
SciPy minimerer F=J_h/2 ved loss='linear'; rapporteret fysisk objective
er fortsat J_h. Begge tal og faktor-to-fejlen gemmes.

Den lokale least-squares-model bruger residual-Jacobian R=dr/du:

\[
\nabla F=R^Tr,\qquad
\nabla^2F=R^TR+\sum_qr_q\nabla^2r_q.
\]

R^TR er Gauss–Newton-delen. Den ekstra term og den ikke-lineære BM-dynamik
betyder, at et sum-of-squares-objective ikke automatisk er konvekst.
WM4's affine reduktion/konvekse QP kan derfor ikke overføres alene ved
at genbruge kvadreret tab. To konvergerede starter er lokal evidens,
ikke bevis for global optimalitet eller økonomisk gevinst.

### Hvorfor bounded least squares / TRF?

Det er et standardvalg til et eksplicit residualproblem med prisbounds.
SciPy's Trust Region Reflective holder prisbounds og udnytter Jacobianen.
Det erstatter ikke den originale BM `optimize` (L-BFGS-B), den eksisterende
F3-Powell-adapter eller andre legacy-kodeveje. Ingen ekstra optimizerpanel
er kørt. LM håndterer ikke bounds; dogbox vælges ikke som første metode
ved mulig svag prisfølsomhed/rangmangel. Ingen algoritme hævdes at være
hurtigst uden sammenlignende data.

To-point-differenser med diff_step=1e-4 er et deklareret pilotvalg for de
fire normaliserede priser. Complex step er ikke gyldig for kode med
stykvis demand og real-/domænevalidering. Et sensitivitetssystem kunne
spare gentagne ODE-solves, men er ekstra afledningskode, der endnu ikke
er nødvendig: det målte forsøg er allerede billigt. Der er ikke udført
selvstændig Jacobian-step-konvergens; ftol-stop må ikke omdøbes til en
verificeret KKT- eller global-optimalitetsgaranti.

### Forholdet til det stokastiske objective

Begge veje kan bruge samme faste A og samme fysiske demand:

\[
J_{\rm stok}(u)=\int\mathbb E[(D(X_t,u_t)-A(t))^2]dt
=\int[(\mathbb E[D]-A)^2+\operatorname{Var}(D)]dt.
\]

Dette er ikke generelt J_det plus en konstant: den deterministiske
ODE-bane er ikke nødvendigvis stochastic mean, og varians kan afhænge
af pris. G8's node/conditional-state-afhængighed bevares i stokastisk G7.
Hvis en senere indkøbsfunktion også er stokastisk, skal man i stedet
dekomponere hele Q=D-A, inklusive covariance. Begge politikker skal
senere scores under eksplicit samme deterministiske/stokastiske problem;
minimumstal fra forskellige mål er ikke en fair modelrangering.

## Låst manifest før modellen blev kørt

- Ét syntetisk døgn [0,24] timer; ingen historisk data eller refit.
- C=4 timer, k=2, Xi=.4; f(x)=1-2x, g(u)=1-2u; x0=.5, sigma_x=0.
  Dette er den kendte lineære G6-reference, ikke G4's historiske S3-fit.
- B(t)=.5+.1 sin(2pi t/24), A(t)=.5 valgt ex ante. Fire seks-timersblokke.
- Starter [.5,.5,.5,.5] og [.7,.7,.3,.3]; hver højst44 nye coarse forwards.
- TRF/two-point/linear loss; ftol=xtol=gtol=1e-7; diff_step=1e-4.
- Coarse quadratur1/24 h, ODE max_step1/12 h, rtol1e-9/atol1e-11.
  Fine quadratur1/48 h, ODE max_step1/24 h, rtol1e-10/atol1e-12.
- For hver neutral/rapporteret slutpolitik: |Jc-Jf|<=1e-4+1e-3|Jf|,
  state på coarse-times og demand på coarse-midpoints fejl<=1e-5.
  Fine-state interpoleres kun ved evalueringskontrollen, ikke i ODE'en.
  Demand evalueres med original BM-map, så der ikke interpoleres over
  prisspring. Disse er nye deterministiske gates; gamle FP-gates ændres ikke.
- Alle raw states/priser skal være gyldige; ingen clipping. Terminal- og
  prisglathedsstraffe0; terminalstate er fri. Højst100 totale forward-solves,
  inklusive finite-difference probes og finekontroller; cache kun eksakte
  prisvektorer. SciPy nfev alene er ikke dette budget.
- Pilot60 s; samlet300 s med240 s computedeadline og60 s eksportreserve.
  Linux-process-timer plus cooperative per-forward-deadline. Windows har
  cooperative checks; den aktive ODE kan returnere efter deadline. Ingen
  lokal Windows-kørsel eller hård Windows-afbrydelse hævdes her.
- Alle completed coarse-kald gemmes straks som scalar receipts; neutral
  og slutpolitikkernes fulde coarse/fine trajectories eksporteres før scoring.
  Budgetstop gemmer completed data og åbner ikke en automatisk gentagelse.

## Faktisk resultat

Runtimepilot .260967 s. Invocation14.227233 s før sidste report-export.
65 coarse forwards inklusive probes, 3 fine forwards, 7 cachehits.
Begge starter konvergerer via ftol; 34 og30 nye forwards, nfev7 og6.
Optimality er henholdsvis1.69379e-7 og2.49624e-7; gtol er ikke stopårsagen.

| Politik | Fine J | Prisvektor |
|---|---:|---|
| Neutral | .120000000000 | [.5,.5,.5,.5] |
| Start1 slut | .0209268019832 | [.686689414,.797804949,.503310994,.392193880] |
| Valgt: start2 slut | .0209268019845 | [.686689795,.797805159,.503310238,.392193631] |

Valgreglen er lavest coarse J blandt konvergerede starter; fine-resultaterne
bruges kun til kontrol, ikke nyt modelvalg. De to fine J afviger1.37e-12.
Forbedring mod neutral82.5609983%. Ingen perfekt tracking påstås.
For den valgte politik er max statefejl1.32738e-7, demandfejl5.55150e-8
og |Jc-Jf|6.78073e-7; alle låste gates består. Stateinterval
[.310112478,.5], terminalstate .499999266. At staten her næsten vender
tilbage skyldes denne case; ingen terminalrestriktion er tilføjet.

Neutral giver et uafhængigt analytisk plausibilitetskrydstjek uden nyt solve:
ved x=.5,u=.5 er delta=0 og D=B, så x forbliver .5 og
integral(.1 sin(2pi t/24))^2dt=.01*12=.12. Råresultatet matcher præcist.

Én kort kontrakttest mod en uafhængig affine LS-løsning dækker tidsvægte,
prisbounds, faktor-to, eksakt cache og faktiske Jacobian-probe-budgetstops.
Første assertion på koordinatafstand til en aktiv grænse var for snæver
for deklareret bound-scaled gtol; den fejlede efter.296377 s. Kontrollen
blev rettet til at teste det analytiske minimum med objective-gap og
prisfejl konsistent med samme gtol. Produktsolver/gates blev ikke løsnet.
Samme ene kontrol består efter.269609 s invocation / .004 s testbody.
Ingen fuld testpakke eller ekstra optimizercase.

27 eksisterende importerede Git-blobs matcher basecommit; nye adapter/
runner har deres egne kildehashes. Hele sourceinventory er uændret gennem
invocation. F/WM/shared solvers/models er ikke ændret.

## Kode, rådata og budget

Ny adapter `src/method/bm_1_g7_deterministic.py` genbruger
`DeterministicPriceEvaluator` og `BM1ControlProblem`; ny researchrunner
`src/exe/run_model/bm_1/g7_deterministic.py` genbruger G6's låste
syntetiske baseline/model. Ingen ny model, ODE, FP eller responsekernel.
`tests/test_bm1_g7_deterministic.py` er én fokuseret kontraktkontrol.

Råpakke: `files/output/bm_1/runs/g7_deterministic_control_20261007.json`.
Den indeholder original report/protocol, sourcehashes, alle65 coarse
scalar receipts, alle neutral/start1/valgt coarse/fine state-/demandbaner,
fælles tids-/target-/quadraturgeometri uden tabsgivende afrunding og receipt.
Geometry-deduplikation ændrer ikke nogen numeriske værdier.

Tidligere conservative debit1045.580078 s bevares. Ny invocation rundes op
til15 s, begge testprocesser dækkes af2 s charge. Debit1062.580078 s;
resterende737.419922 s af G-panelets1800 s. Dette er fortsat konservativ
reservation, ikke komplet timing af gamle umålte probes. Der åbnes ikke
automatisk et nyt stort mixture-studie eller nyt panelbudget.

Reproduktion af det færdige forsøg (kun ved konkret behov, ikke næste opgave):

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -B -m src.exe.run_model.bm_1.g7_deterministic --source-commit 1644a55a77c50be0c25cadadb1f7513600dab7a2 --output files/output/bm_1/runs/g7_deterministic_NEW
```

Destination skal være ny. Basecommit plus manifestets hash-identiske nye
Python-filer definerer den udførte source; et nyt HEAD må ikke alene
påstås at være dette gamle forsøg. Ingen lokal kommando kræves nu.

## Hvad er accepteret, og hvad er næste skridt?

**Accepteret:** original fixed-A deterministisk single-shooting på én
syntetisk lineær reference, bounded LS-adapter, lokal optimizerkonvergens,
fælles costidentitet og to-niveau numerisk udviklingskontrol.
**Åbent:** faktisk BM/WM fælles kontrolbenchmark, historisk/kausal
priselasticitet og kontrol, S3-kurveparitet i kontrol, operationel forecast,
fuld stokastisk G7-policy/mixture/EM/feedback og generel global optimalitet.

Næste er at låse en fælles fast A(t), baseline, tidsskala, prisblokke,
prisgrænser og initialiseringsregler for BM/WM-referencepolitikkerne.
Bevar WM's native køb–forbrug-mål som særskilt reference. Genbrug valgte
fits, ingen nye kurvefamilier eller parameterpaneler. G4's svage/flade
prisrespons skal eksplicit diagnosticeres, før en historisk kontrolstrategi
fortolkes som fysisk eller økonomisk forbedring. F/WM-arbejde koordineres
gennem konkrete interfaces; G implementerer ikke F4-H2.

Derefter beholdes deterministisk og stokastisk score side om side på samme
kontrolproblem. Den dyre M254-bankprisforward kræver stadig det allerede
aftalte afgrænsede runtime-/jobdesignreview før et nyt registreret run.
Projektets obligatoriske stokastiske leverance består; deterministisk
førsteprioritet er en rækkefølgeændring, ikke en erstatning.

## Kilder og attribution

1. Vejleder-/originalmateriale: `FP2025-1.pdf`, afsnit3.2.3, trykt side14:
   fixed-A BM-control, prisblokke og single shooting. BM-formler og
   legacykode er bevaret. Rapportens afsnit4/WM og WM4-noternes
   `Quadratic price objective` forklarer den særskilte affine A(U)-opgave.
2. Projekt111866 (STADS-billedet): predetermined desired aggregate energy
   consumption profile; FP-mixture, stokastisk kontrol og rekursiv
   estimation er obligatoriske sammenhænge. Det er ikke et valg af denne
   pilots konstante A, fire blokke eller solver.
3. Nocedal & Wright, Numerical Optimization, 2nd ed.: kapitel4 om trust
   regions og kapitel10 om nonlinear least squares/Gauss–Newton er
   standardgrundlag for residual-/lokalmodeltænkningen. Det præcise SciPy
   TRF/bounds-design henføres til SciPy's dokumentation og Branch–Coleman–Li,
   ikke til tavlebillederne eller en påstand om en ny vejledermodel.
4. SciPy officielle least_squares-dokumentation, hentet7. oktober2026:
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
   Bound-constrained TRF, linear loss, F=J/2 og nfev eksklusive numerical
   Jacobian-prober. Faktisk udført med SciPy1.17.0; ingen1.18-only API bruges.
5. Egne deklarerede valg: syntetisk A=.5, frosne referenceparametre,
   starts, fire blokke, quadratur/steps/tolerancer, diff_step, call/timecaps,
   valgregel og acceptscope. Råmanifestet fastlåser dem før kørslen.
