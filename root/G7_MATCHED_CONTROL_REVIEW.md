# G7: fælles deterministisk BM/WM-kontrakt og frozen-fit-prisrespons

7. oktober 2026. G-sporet; Sol High. Ingen F4-H2-opgave og ingen WM-refit.

## Resultat og afgrænsning

Den fælles fixed-target-opgave er gjort konkret og gemt som en selvstændig
inputpakke. Det oprindelige januarpar og januar-G4 uden straf genbruges.
Tre faste prisniveauer, tre modeller og to grids giver **18 forwards**.
Alle ni coarse/fine-par består de på forhånd fastlagte gates.
Ingen prisoptimering, historisk modelrangering, ny parameterestimation eller
stokastisk propagation er udført. Den tidligere syntetiske G7-optimering
genkøres ikke. M254's negative G7-runtimepilot er fortsat negativ.

En fokuseret ny analytisk kontraktkontrol består. Den afprøver fysisk
time/dag-konvertering, signed pris, WM-affinitet, raw cost og bounds.
Produktkoden er ikke ændret efter packet-frysning eller kørsel.

Det vigtigste fund er, at prisrespons er meget forskellig på de to halvdele
af G4/S3's prisskala. S3's primære fits fra uge 16–18 er næsten flade på
hele det prisinterval, som deres gemte prognoser faktisk bruger.
Dette må ikke skjules ved blot at vise en optimizer's lavere model-cost.

## 1. Hvorfor netop disse gemte fits?

Det arkiverede `G4-deterministic-fit` indeholder også et fuldt
`G-deterministic-fit` med det tidligere BM/WM4-januarpar. Januar-G4's
lambda=0-kandidat genbruges som curved sammenligning; det originale
lineære BM-fit beholdes som legacy-reference. Kandidaten blev ikke valgt
som en ny vinder her. Alle tre bruger træning 1. januar→1. februar 2022,
samme baseline, råskala og præcis origin 1. februar kl. 00.

| Model | Frosne parametre | Frosset start ved origin |
|---|---|---|
| Lineær BM januar | C=1.6349238174094938 h, k=.22688550696235388, Xi=.2312064791761644 | x=.6859311841073318 |
| G4 BM januar uden straf | C=1.4579146988701088 h, samme k, Xi=.19062055809117637, a=-.9999999996559088 | x=.4397562219525716 |
| WM4 januar | r=[-.02471083257232427,-.02471083257232426,-.3932770618786307], modeltid dage | [F,R]=[.3349719233905929,.0009626951655749] |

Den valgte senere **S3/lambda=0**-protokol ændres ikke. S3 har seks
rolling-origin fits i marts–maj; der er ikke et matched WM4-fit til samme
35 træningsdage og samme origins i den gemte G-evidens. At parre ét af dem
med januar-WM og kalde resultatet en matched modelrangering ville ændre
både træningshistorik, baseline og prisskala. Derfor udføres S3's analytiske
audit separat. Ingen ny WM-estimation indføres for at udfylde hullet.

## 2. Den frosne fælles opgave

| Felt | Fælles kontrakt |
|---|---|
| Horisont | 1. februar 2022 kl. 00→2. februar kl. 00, 24 fysiske timer, ingen DST-overgang |
| Beslutninger | fire priser u_j∈[0,1], holdt på [0,6),[6,12),[12,18),[18,24) timer |
| Baseline | B(t)=O(t), samme restaurerede januar-baseline, 60 samples/h og lineær interpolation |
| Temperatur | 25 gemte timesamples fra samme forecast-input; realiseret vejr, betinget udviklingsprøve |
| Mål | konstant døgnmiddel af den frosne baseline, uafhængigt af kandidatprisen |
| Initialisering | hver kandidat starter fra modellens egen gemte origin; ingen neutral reset/nyt fit |
| Cost | kvadratisk tracking-integral, ingen ekstra a-, terminal- eller smoothing-straf |
| Constraints | fælles prisbounds; native BM x∈[0,1], WM F∈[-1,1]; finite trajectories; demand inden for den erklærede normaliserede benchmarkdomæne |
| Numerik | original BM RK4 og original WM4 Euler; fælles fysiske grids 60 og 120 steps/h |
| Observationer | ingen target-demand læses til baseline, kontrolmål eller prisforsøg |

Baselinen ligger her i [.24936384992692504,.5983689705695795]. Restaurering
bruger præcis gemte koefficienter/standardisering/settings, ingen `.fit()`.
Dens rå prediction er allerede i [0,1]; den eksisterende clipping ændrer
ingen samples. Timesamples matcher det gemte forecast inden 1e-12.
Det sammenhængende baseline-input er frosset ved 60 samples/h; fine-grid
løser samme interpolerede funktion, ikke en ny vejr/baseline-model.

### Målprofilens udledning og begrundelse

For dette lille kontrolforsøg vælger vi en flad desired profil med samme
samlede input som baselinen:

\[
 A(t)=\bar B,\qquad \bar B=\frac1T\int_0^T B(s)\,ds,
 \qquad \int_0^T A(t)\,dt=\int_0^T B(t)\,dt.
\]

Integral af den deklarerede piecewise-lineære baseline beregnes ved den
eksakte trapezsum på dens egne knuder. Her er A=.39900697422105735
normaliseret. Dette er **vores erklærede benchmarkvalg**, ikke en ny formel
tilskrevet vejlederens tavle eller en estimeret indkøbsmodel.

En flad profil gør det muligt at undersøge belastningsflytning; at bevare
baseline-integralet undgår at definere opgaven som en vilkårlig reduktion
af samlet efterspørgsel. Valget A=B ville primært måle ophævelse af modellens
egen fleksibilitetskorrektion. En købt eller separat ønsket fysisk profil
kan senere erstatte A, men den skal erklæres før kandidatpriser og være
identisk for BM/WM. A er ikke faktisk observeret forbrug i testdøgnet.

Fremtidstemperaturen var allerede realiseret i denne udviklingspakke;
"exogent/frosset før priserne" betyder ikke et blindt operationelt forecast.

## 3. Samme pris og enheder — ikke samme tal i native koordinater

Lad p være den fælles DK1 spot-proxy i DKK/MWh og u den fælles
beslutningskoordinat. Med original januar-normalisering:

\[
 p=p_{\min}+s_pu,\qquad
 u_{BM}=u,\qquad U_{WM}=2u-1,
\]

\[
 p_{\min}=.74,\qquad s_p=2604.02001.
\]

Tre prøvepriser u=.25,.5,.75 betyder derfor præcis samme råpris for begge
modeller: 651.7450025,1302.750005,1953.7550075 DKK/MWh. WM får native
U=-.5,0,.5. En ens numerisk u-værdi sendt uomregnet til begge ville give
to forskellige råpriser. Der ændres ikke g-knuder eller WM-koefficienter.
Spot-proxy er ikke en verificeret kundetarif eller kausal prisintervention.

For demand/target bruges samme affine map:

\[
 D_{raw}=d_{\min}+s_DD,\qquad A_{raw}=d_{\min}+s_DA,
 \qquad d_{\min}=.40230769230769237,\ s_D=2.44.
\]

\[
 D_{raw}-A_{raw}=s_D(D-A),\qquad J_{raw}=s_D^2J.
\]

Råtarget er 1.3758847094070722 i de oprindelige aggregatenheder.
Inputimporten beregner MED-husholdningernes gennemsnit. Den præcise
point/interval-semantik fra historiske workbookmålinger er stadig uafklaret
i G's legacy-kontrakt. Denne leverance kalder derfor **ikke** J_raw et
verificeret kW²h-tal. Kontrolcost har enheden normaliseret-demand²·timer;
raw cost har rå-demand-enhed²·timer. En senere fysisk energitolkning kræver
metadatasemantik, ikke et automatisk lån fra F4-H2's særskilte datakontrakt.

WM's modeltid er dage: tau=h/24. Følgelig

\[
 \frac{dF}{dh}=\frac R{24},\qquad
 \frac{dR}{dh}=\frac{r_FF+r_U(2u-1)+r_RR}{24},\qquad I=O+R.
\]

Adapteren giver den eksisterende WM-solver et [0,1]-døgn-grid, mens
output/cost/timestamps fortsat er i timer. Koefficienterne divideres ikke
også med 24; det ville tælle konverteringen to gange. Native F/R og BM's
x er forskellige states og sættes ikke lig med hinanden.

## 4. Kontrolmatematik og numeriske valg

BM beholdes uændret:

\[
 \dot x=\frac{D(x,u,B)-B}{C},\qquad
 \delta=\tanh\!\left(\frac k2[f(x)+g(u)]\right),
\]
\[
 D=B+\Xi\delta\left[(1-B)\mathbf1_{\delta\ge0}
                         +B\mathbf1_{\delta<0}\right],\qquad f(x)=1-2x.
\]

For både D_BM og I_WM er den fælles opgave

\[
 \min_{u\in[0,1]^4}J_M(u),\qquad
 J_M(u)=\int_0^{24}[D_M(h;u)-A(h)]^2\,dh.
\]

**Denne kørsel løser ikke minimeringen.** Den evaluerer tre på forhånd
erklærede konstante politikker med alle fire blokpriser ens.
Sådan undgår vi at bruge en optimizer som første diagnose af et næsten
prisuafhængigt fit. En faktisk fireblok-Jacobian/reachability er stadig åben.

Residualerne er samme mål, blot egnet til least squares:

\[
 r_q(u)=\sqrt{\Delta h_q}\,[D_M(h_q^{mid};u)-A(h_q^{mid})],
 \qquad J_h=r^Tr.
\]

q tæller fine quadratureceller, ikke de fire prisblokke. BM-state ved
midtpunkter interpoleres mellem RK4-endpoints før den oprindelige demandmap;
WM's R interpoleres mellem Euler-endpoints. Begge integreres ved samme
midtpunkter. Dette er en ny **G-evalueringsadapter**, ikke en erstatning af
den gamle hourly-point historiske observationskontrakt. Den kontinuerte
kontrolprofil og målingernes ukendte interval-semantik er forskellige ting.

For frosne WM-koefficienter og origin er Euler-systemet lineært i state
og affint i de holdte priser. Derfor findes et c og H med

\[
 I(u)=c+Hu,\qquad J_h(u)=\|W^{1/2}(c+Hu-A)\|^2,
 \qquad \nabla^2J_h=2H^TWH\succeq0.
\]

Det giver en convex QP/LS-mulighed til en senere WM-optimizer. Native
state/flow-admissibility kan udtrykkes ved lineære inequalities for dette
frosne affine system. Denne egenskab gælder ikke generelt nonlinear BM.
Den er ikke en påstand om globale minimummer fra tidligere BM-TRF-starter.

## 5. Faktisk prisresponsprøve og gates

På forhånd: tre modeller × tre konstante niveauer × grids60/120 =18
actual forwards; ingen optimizer/Jacobian-prober/refits. Samlet
cooperative loft30 s med kontrol før/efter hver forward, ingen automatisk
retry. Dette er ikke en hård preemptiongaranti inde i en forward.
BLAS=1, én proces; parallelisering ville være unødvendig her.

Gates fra den afsluttede G7-reference bevares:

- max coarse/fine statefejl≤1e-5;
- max demandfejl ved samme fysiske midtpunkter≤1e-5;
- |J_c−J_f|≤1e-4+1e-3|J_f|;
- ingen ugyldige states, demand-domain failure eller clipping.

Alle ni par består. Største statefejl=2.4309725806803684e-6,
demandfejl=1.0257320618434207e-6, costfejl=2.744987028513446e-6.
Kørsel før final export=.21155951299806475 s; process wall=.504847401 s.

| Model | Fine J ved u=.25 | Fine J ved u=.5 | Fine J ved u=.75 | RMS ændring i demand, .25→.75 |
|---|---:|---:|---:|---:|
| Lineær BM januar | .3067884915035636 | .3040810155614405 | .29891386458236846 | .009651475769838214 |
| G4 BM januar uden straf | .30717838142719356 | .30018738528236033 | .30018738528093636 | .008031465639990857 |
| WM4 januar | .30982467003807723 | .3064754447997665 | .30495536506339327 | .012346205563982549 |

RMS-demandændring i raw units er henholdsvis .02355,.01960,.03013.
Tallene er **modelbetingede følsomheder**, ikke observerede interventions-
effekter eller en modelrangering. At have stærkere respons er ikke i sig
selv bedre. G4 januar ændrer næsten intet fra .5→.75, fordi a≈−1 giver
næsten flad øvre g-halvdel. WM's affine midpoint-identitet består til
1.11e-16; nonlinear BM er ikke affint (afvigelser .000731 og .005416).

### Sluttilstand og samlet forbrug skal vises sammen med J

Ens target-integral sikrer ikke ens faktisk demand-integral. Ingen
terminalconstraint er skjult i cost. Fra de originale modelbalancer fås

\[
 \int_0^T(D_{BM}-B)\,dh=C[x(T)-x(0)],\qquad
 \int_0^T(I_{WM}-O)\,dh=24[F(T)-F(0)].
\]

Ved u=.75 ligger actual normalized input-areal under target-arealet med
ca .18815 for lineær BM, .15820 for G4 BM og .19865 for WM. Retningen og
størrelsen stemmer med deres ændrede sluttilstande; numerisk midpoint-
balance er en tilnærmelse og en ny eksakt-gate er ikke indført bagefter.

En optimizer kan således delvis forbedre J ved at udnytte startlager og
slutte et andet sted. Det er lovligt i den nu erklærede original fixed-A-
reference, men må ikke markedsføres som ren døgnflytning med uændret energi.
F/R's og x's kapacitetskonventioner er desuden ikke fælles fysiske batteri-
constraints. Fremtidige resultater skal vise J, energiunderskud og terminal
state separat. Et cyklisk energy-neutral mål er en særskilt begrundet
benchmarkvariant, som skal bevare originalen; ingen tilfældig strafvægt
tilføjes for at skjule forskellen.

## 6. Analytisk audit af de primære S3-fits

S3-kurven er lineær gennem (0,1),(.5,a),(1,−1). Direkte fra interpolation:

\[
 g'(u)=\begin{cases}2(a-1),&0<u<.5,\\-2(1+a),&.5<u<1.\end{cases}
\]

Ved fast state og B, væk fra g-knuden og deltategnskiftet:

\[
 \frac{\partial D}{\partial u}
 =\Xi H_B\frac k2(1-\delta^2)g'(u),\quad
 H_B=\begin{cases}1-B,&\delta>0,\\B,&\delta<0.\end{cases}
\]

H_B≤1 og 1−delta²≤1 giver en øvre bound
|∂D/∂u|≤Xi*k*|g'|/2. For råpris er
∂D_raw/∂p=(s_D/s_p)∂D/∂u. Bounden er ikke en målt elasticitet og udelader
state-feedback over tid. Ved delta=0 er ensidig afledt relevant, og ved
u=.5 er g normalt ikke differentiabel. Vi påstår ingen central Jacobian dér.
En præcis flad g-strækning giver imidlertid identisk BM-forcing og dermed
identisk trajectory ved priser, der bliver på denne strækning, når alle
andre inputs/origin er ens. Dette er en strukturel observation, ikke et fit.

| Uge | a | g' nedre halvdel | g' øvre halvdel | Gemte forecastpriser u_min→u_max |
|---|---:|---:|---:|---:|
| 13 | −.823260 | −3.646519 | −.353481 | .092992→.450962 |
| 14 | −.729809 | −3.459617 | −.540383 | .074074→.376228 |
| 15 | −.754762 | −3.509524 | −.490476 | .091960→.422057 |
| 16 | .9999999902091905 | −1.95816e-8 | −3.99999998 | .077049→.415006 |
| 17 | .9999999999999999 | −2.22045e-16 | −4 | .235908→.410326 |
| 18 | .9999999999999999 | −2.22045e-16 | −4 | .166758→.416786 |

Alle 168 priser per uge ligger på den nedre halvdel: **1008/1008**.
Deres S3-map er p=−500+6500u; knækket u=.5 er 2750 DKK/MWh. De seks
fits er verificeret mod både lokale resultater og den allerede publicerede
G4-validation-artifacts parametre. Kun pris-kolonnen bruges i denne audit;
ingen target-demand scoring/refit foretages. Gemte forecastpriser er
realiserede, allerede eksponerede udviklingsinputs, ikke nye blinde data.

For uge16–18 er den direkte normalized-response-bound på den observerede
halvdel ca 3.99e-8,1.11e-15,1.11e-15. Uge13–15 er ikke flade på den
samme halvdel; "alle G4-fits er prisuafhængige" ville være forkert.
Uge16–18 kan formelt reagere kraftigt over 2750 DKK/MWh, men god forecast-
MSE ved lave priser dokumenterer ikke netop denne højprisrespons. At en
curvefamilie har endpoints 1/−1 er ikke kausal identifikation af mellemrummet.

## 7. Kilder, kodeejerskab og provenance

- **Original BM/fixed A:** `FP2025-1.pdf`, afsnit3.2.3, fysisk PDF-side15/
  trykt14: known availability profile, piecewise price control og
  single-shooting-optimering. Den oprindelige BM-map og gamle optimizer
  bevares. Et prisafhængigt A(U) er ikke en nødvendig BM-udvidelse.
- **WM4:** `WM4_notes(1)(1).zip/main.tex` og `src/model/wm_4.py`:
  F'=R,R'=r_FF+r_UU+r_RR,I=O+R. Notes' native indkøbsfunktion
  A(U,I,F,PE) er fortsat en separat legacy-opgave. Den bruges ikke som
  fælles target her; den ville gøre målet modelafhængigt.
- **Standardmatematik:** kædereglen h↔dage, affine koordinatmaps,
  energy-balanceintegration, piecewise-lineær differentiation og
  kvadratisk Hessian-PSD er udledt ovenfor. Residual-LS passer til
  Nocedal/Wright, *Numerical Optimization*, kapitel10; vi tilskriver ikke
  bogens resultater vores nye valg af fladt target eller diagnosepriser.
- **Egne G-valg:** 24h/four blocks, target=baseline-mean, tre konstante
  prisniveauer, adapter med fælles midpoint-cost og begrænset kontroltest.
  Ingen ny basis, modelinnovation på tavlefamilien, noise-fit eller F-kode.

Nye filer er kun G-adapter, G-runner, én G-test, disse noter og to
selvbærende JSON-artifacts. 18 eksisterende importerede kildeblobs matcher
executionbase `2846a5474027101a45b43aa66e9ec111a8727628`.
De to nye numerical files har execution-SHA256 i packet/report.
Original full G4/reference snapshot, weather25, baseline1441, target,
scales, origins og alle S3-auditfelter er bevaret i packet.

WM4's deterministiske constructor/equivalent_q/_solve_deterministic og
initial-condition/constructor-validering har identisk AST med det
verificerede H5-referencecommit `f92ea632aa1a17ccdc18256a0e8606cfcf573d33`.
H5-modelfilens SHA256 matcher det gamle snapshots hash. Den nyere
validering tilføjer r_F+r_U≥r_R−tol; frosne r består med margin .3438554.
De øvrige ændringer er diffusion/fitter-kode, som ikke kaldes her.
Dette er en konkret scoped source review, **ikke** en generel bypass eller
ændring af det eksisterende snapshots strict hash-validator.

Den nye fokuserede unittest består, processmåling .302911655 s.
Ingen gammel testpakke, parameterpanel, modelpilot eller post-merge
numerisk genkørsel er udført. Råreport bevarer fuld-grid scalar costs og
precisionmetrics, timevise states og første demand-midpunkt i hver time.
Dense forward traces er ikke eksporteret; artifact er derfor ikke en
uafhængig dense-precision-genberegning. Selvstændig reproduktion kan bruge
den frosne packet og source uden private workbookdata eller refits.

Konservativ ny debit6 s dækker preparation/import, enkeltcheck,
invocation/export og kildeaudit. Tidligere G-debit1062.580078 s bliver
1068.580078 s, rest731.419922 s af1800. F/WM-budgetter ændres ikke.

## 8. Hvad næste trin må og ikke må hævde

1. Næste lille G7-opgave er en afgrænset **fireblokoptimering af det gemte
   januarpar** under denne identiske fixed-target-kontrakt. Vis både neutral,
   optimeret J, raw/normalized profil, energi-gap og terminal states.
   WM kan udnytte affine sensitivities/convex LS; BM bruger bounded nonlinear
   LS. Native trajectoryadmissibility skal håndhæves for alle kandidater,
   ikke antages ud fra tre konstante policyprober.
2. Det er modelbetinget udviklingskontrol, ikke en kausal historisk rangering.
   S3's senere prisplateauer er stadig åbent problem; en fysisk/historisk
   optimering af dem må ikke flyttes stiltiende til u>.5 for at vise gevinst.
   Diagnose dokumenteres før eventuel ny curve-estimation, ikke via en
   ubudgetteret refit eller automatisk genindførelse af a-straf.
3. Skal sammenligningen være energineutral døgnflytning, skal et fælles
   terminal/energy-constraint begrundes som særskilt benchmarkvariant.
   Original fixed-A-reference og WM's native indkøbsopgave bevares.
4. Stokastisk G7 er fortsat en obligatorisk senere leverance. M254 runtime/
   jobgraf skal vurderes før nye priceforwards; ingen retry her. Vejlederens
   M254-basis er primær og 42-leds-basis kun sammenligningsreference.

Sol High er passende til dette næste lille adapter-/optimizertrin.
Der er ikke behov for lokal kodekørsel nu.
