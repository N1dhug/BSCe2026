# G7: deterministisk kontrol med bevaret døgnforbrug

7. oktober 2026. G-sporet; ingen F4-H2-ændringer. Executionbase
`4f9ea30abee6a2673a33868ffad30e9ff82cc429`; registreret kode/protokol før run
på `3bcba23c16630f09edc67f7cd07e6d2b54d798ce`. Råcheckpoint efter run:
`4a3c4148b308fd2e74e3b7cc590a0de591b349b0`.

**Resultat:** seks valgte politikker består de registrerede coarse/fine-gates.
Én fælles januarbaseret 24h-opgave er gennemført, både med det oprindelige
fixed-A-mål og med en særskilt lighedsbegrænsning på døgnforbruget. Dette er
et betinget numerisk kontrolbenchmark, ikke kausal prisvalidering, forecast-
modelvalg, hele G7 eller hele G-planens afslutning. De oprindelige modeller,
WM's native A(U)-mål, vejlederens basis og tidligere negative resultater bevares.

## 1. Frosset forsøgsgrundlag

Genbrug `files/output/bm_1/runs/g7_matched_contract_20261007.json` uændret.
Origin er 1. februar 2022 kl.00, fire prisblokke [0,6,12,18,24] timer.
Januarens lineære BM-reference, G4/lambda0-udviklingskandidat og matched WM4
bruger deres egne gemte origins. Der er ingen historisk refit, ny observation
eller genkørsel af det afsluttede prisresponspanel. Januar-G4 er ikke en ny
udnævnelse af den senere S3-protokol som kontrolmodel.

Fælles normaliseret baseline B og fast mål

\[
 A(h)=\bar B=\frac1{24}\int_0^{24}B(s)\,ds=0.39900697422105735
\]

kommer fra den tidligere frosne pakke. A vælges uden næste døgns demand-Y.
Baseline er betinget af de allerede deklarerede realiserede vejrinputs;
dette er fortsat ikke en operationel forecast med ukendt fremtidigt vejr.
Samlet referenceforbrug er 24A=9.576167381305376 normaliserede demand-timer.

Den fælles pris er u∈[0,1]. WM modtager U=2u−1; native WM-tid er dage,
så fysiske timer h indgår gennem τ=h/24. Rå scales bevares:
D_raw=.40230769230769237+2.44D og p_raw=.74+2604.02001u.
Priserne er DK1-spotproxy, ikke en dokumenteret kundetarif/intervention.
Legacy-demandens punkt-/interval-/populationssemantik er ikke løst af dette
forsøg. Derfor hævdes ingen fysisk kW²h-enhed eller sammenlignelig lagerkapacitet.

## 2. Kontrolmålet og hvorfor bevaret forbrug er en lighed

Lad q_M(h;u) være modellens deterministiske demand (BM:D, WM:I). Originalen er

\[
 \min_{u\in[0,1]^4}J_M(u),\qquad
 J_M(u)=\int_0^{24}[q_M(h;u)-A(h)]^2\,dh.
\]

Den nye sammenligning løser samme opgave under

\[
 E_M(u)=\int_0^{24}[q_M(h;u)-B(h)]\,dh=0.
\]

Fordi integral A=integral B er det ækvivalent at bruge q−A i E. En fast
samlet mængde kan dermed flyttes i tid; en lavere J må ikke købes ved blot
at ændre døgnets nettomængde. J er tracking, ikke elregning eller velfærd.

En straf J+λE² ville tillade afvigelser afhængigt af en ny vilkårlig λ,
ændre tradeoff og kræve et nyt parameter-/skaleringsvalg. Her er bevaret
forbrug netop sammenligningens krav, så en lighed er mere præcis. Original-
opgaven bevares særskilt; der tilføjes ingen skjult terminal-/prisglathedsstraf.

Den brugte midpoint-quadratur, h_i=t_{i+1}−t_i, er

\[
 r_i(u)=\sqrt{h_i}[q_M(t_{i+1/2};u)-A_i],\quad
 J_h=r^Tr,\quad E_h=\sum_i h_i(q_{M,i}-A_i).
\]

Samme q og vægte bruges i begge udtryk. Prisblokgrænser ligger på gridpunkter;
coarse/fine-demand sammenlignes ved gennemsnittet af de to fine midtpunkter
inden for den samme coarsecelle. Ingen interpolation hen over prisspring.
Rå objektivværdi er J_raw=s_D²J; rå energiforskel er s_D E, da offsetten
forsvinder ved samme horizon og reference. Man må ikke sammenligne rå J på
forskellige skalaer eller glemme WM's faktor24.

## 3. Udledning af lagerbalancen og den numeriske WM-korrektion

BM-kernen er uændret:

\[
 \dot x=\frac{D-B}{C},\quad f(x)=1-2x,\quad
 \delta=\tanh\!\left(\frac{k}{2}[f(x)+g(u)]\right),
\]
\[
 D=B+\Xi\delta\left[(1-B)\mathbf1_{\delta\ge0}+B\mathbf1_{\delta<0}\right].
\]

Integrér første ligning direkte:

\[
 E_{BM}=C[x(24)-x(0)].
\]

Dermed er kontinuert energineutralitet lig uændret slut-SoC. Til en
feasible start bruges g(u_eq)=−f(x0)=2x0−1; så δ=0, D=B og x er konstant.
Den anden på forhånd fastlagte start er u_eq+d(1,−1,−1,1), hvor
 d=min(.15,u_eq/2,(1−u_eq)/2). Der tilføjes ingen ekstra starter efter resultatet.

WM's bevarede time-dynamik er

\[
 \dot F=R/24,\quad
 \dot R=[r_FF+r_U(2u-1)+r_RR]/24,\quad I=B+R.
\]

Derfor er

\[
 E_{WM}=\int_0^{24}R\,dh=24[F(24)-F(0)].
\]

R(24)=R(0) følger ikke af energibalancen. At kræve både F og R tilbage
ville være en stærkere periodisk-state-opgave, som vi ikke indfører her.

Original WM bruger eksplicit Euler: F_{i+1}−F_i=hR_i/24. Adapterens demand
bruger midpoint-R=(R_i+R_{i+1})/2. For konstant fysisk step h teleskoperer
forskellen præcist:

\[
 E_{h,WM}=24(F_N-F_0)+\frac h2(R_N-R_0).
\]

At kræve F_N=F_0 alene bevarer altså ikke præcis den samme diskrete
forbrugsmængde som rapporteret i J. Vi håndhæver E_h=0 og rapporterer
slutstate og forfiningsfejl separat; vi omskriver ikke den originale solver.
Når h→0, falder korrektionen bort. BM's RK4-state og midpoint-demand har
også en lille quadraturforskel; begge identiteter auditeres på gemte baner.

På fineste WM-neutrale bane er E=1.4657932e−6,
F_N−F_0=2.3086787e−7 og R_N−R_0=−.00097800859. Identiteten ovenfor stemmer
til 1.47e−14. Fineprisen er den samme som den coarseoptimerede; ingen skjult
fine-genoptimering foretages. Coarse E er numerisk nul, fine E er under
præregistreret 1e−5. Det er bevaret forbrug inden for deklareret numerisk
præcision, ikke et påstået eksakt nul på alle grids.

## 4. Metodevalg, kode og hvad optimalitet betyder

### BM: lille ikke-lineær single shooting

Kun fire priser optimeres; alle fysiske parametre, g og origins holdes faste.
Hver kandidat fremskrives fra samme origin med eksisterende RK4 via
`MatchedDeterministicControl`. SLSQP anvendes til både originalen og
lighedsvarianten, med bounds[0,1], ftol1e−11, absolute FD-step1e−5 og
højst60 iterationer/400 faktisk nye forwards per start. To starter per variant.
Eksakt float-tuple-cache genbruger samme objective-/constraint-evaluation;
alle Jacobianprober tælles, ikke kun SciPy's nfev.

Begrundelse: SLSQP håndterer den ene ikke-lineære lighed direkte med lokale
kvadratiske delopgaver og linearisering af E. Almindelig bounded residual-LS
kan ikke direkte håndhæve E=0. En stor straffaktor er ikke nødvendig.
En ny adjoint/sensitivitykerne ville være uforholdsmæssig for fire kontroller.
Residualkvadratet bevares præcist; ingen robust loss eller regularisering.

g er stykvis lineær, og D har et grenvalg ved δ=0. SLSQP's glatte lokale
konvergensteori er derfor ikke et globalt bevis her. Begge starter rapporterer
success, men free-starterne gav forskellige slutpunkter. Vi vælger den laveste
J blandt gennemførte, feasible succesfulde starter og kalder den **bedste
fundne kandidat**. Ingen nonlinear KKT/global accept eller ekstra søgning
hævdes. Neutrale starter giver praktisk samme J, men det er heller ikke et bevis.

### WM: affin respons og fire-dimensional konveks QP

Euler-dynamikken er affin i u med faste r, B og origin. Derfor kan residualen
skrives r(u)=c+Hu. Fem native forwards (u=0 og de fire enhedsvektorer) giver
c og hver søjle H_j=r(e_j)−r(0), uden parameterfit eller approksimativ surrogate.
E(u)=e0+a^Tu er ligeledes affin. Dermed

\[
 J=\|c+Hu\|^2,\quad Q=2H^TH\succeq0,\quad
 E=0\iff a^Tu=-e_0.
\]

Alle state-/demandkoordinater er affine. Hele prisboksens koordinatvise
minimum/maximum beregnes eksakt som v0+sum_j min(0,v_j) og
v0+sum_j max(0,v_j). Her er demand i [.2229618,.6098922] og
F i [.3212726,.3429841] for ALLE u∈[0,1]^4. Native pathconstraints er derfor
redundante i netop denne pakke; koden afviser at udelade dem, hvis certifikatet fejler.

Vi gennemgår alle 3^4=81 faces: hver pris er fri, på0 eller på1. På hvert
face løses de frie stationaritetsligninger, eventuelt KKT-systemet

\[
 \begin{bmatrix}Q_{ff}&a_f\\a_f^T&0\end{bmatrix}
 \begin{bmatrix}u_f\\\lambda\end{bmatrix}
 =\begin{bmatrix}-q_f-Q_{fb}u_b\\-e_0-a_b^Tu_b\end{bmatrix},\quad q=2H^Tc.
\]

Feasible kandidater sammenlignes i J. Dette er et lille udtømmende aktivt-
face-søg, ikke 81 native modelreplays. Q's egenværdier er
[.000170468,.0003529998,.0011353557,.0085948708], condition≈50.42:
normal equations er her kontrollerbart velkonditionerede. Dette begrunder
ikke normal equations i den ill-konditionerede M254-basis, hvor SVD bevares.

Endelig kontrolleres gradienten Q u+q+λa: nul på frie priser, ≥0 ved0,
≤0 ved1. KKT-residual er numerisk0 i begge WM-opgaver. Native forward på
valgt pris stemmer med den affine residual til højst3.47e−17. Med positiv
definit Q og konveks feasible mængde er WM-prisen entydigt global for den
DISKRETE opgave, inden for de rapporterede floating-point checks.
Dette er ikke et globalt optimum for den kontinuerte ODE uden discretiseringsfejl.

## 5. Faktiske resultater

J nedenfor bruger fine120steps/time. Baseline-only J_B=.308609546074;
den er en fælles referenceprofil, ikke nødvendigvis en feasible konstant
WM-prispolitik med det givne R0. Forbrugsbalance er relativ til B/A.

| Model | Original J | J med bevaret forbrug | E, original | E, bevaret | Reduktion mod J_B, bevaret |
|---|---:|---:|---:|---:|---:|
| BM lineær | .268818599455 | .275408887552 | −.0656625984 | −4.60e−10 | 10.7581% |
| BM G4 | .273316332715 | .276532279602 | +.0604490720 | −3.88e−10 | 10.3941% |
| WM4 | .299704871421 | .303629020154 | −.2259380037 | +1.47e−6 | 1.61386% |

Ingen invarians kræver samme kontrolmængde i originalopgaven: G4 øger her
nettoforbruget, mens lineær BM og WM sænker det. Derfor er energineutral-
sammenligningen væsentlig. Den ændrer ikke rækkefølgen i dette ene case.
Den fleksiblere G4-kurve forbedrer ikke denne kontrolscore over lineær BM.
Det omstøder ikke tidligere forecast-valg; forecast-fit og betinget kontrol-
tracking er forskellige spørgsmål.

Neutrale u for blokke0–6/6–12/12–18/18–24:

| Model | u0 | u1 | u2 | u3 |
|---|---:|---:|---:|---:|
| BM lineær | 0 | 1 | 0 | .4749984451 |
| BM G4 | 0 | .9468691847 | .2187762763 | .8849911735 |
| WM4 | .4085658674 | 0 | 1 | 0 |

G4's januar-a≈−1 gør den øvre prishalvdel næsten flad. De to høje priser
er derfor repræsentanter for næsten samme forcing, ikke præcist identificerede
økonomisk optimale tariffer. Mange aktive prisgrænser og modelbaseret respons
understreger behovet for kausal/økonomisk validering før anvendelse.

![Fælles døgnopgave og neutral sammenligning](g7/night_control_20261007.svg)

Største coarse/fine statefejl2.56890e−6, demandfejl1.06182e−6,
J-fejl3.00612e−6. Alle seks består state/demand1e−5 og
J-absolute1e−4+relative1e−3; neutrale fine-E består1e−5. Dette er en parret
tidsforfining, ingen universel kontinuumsfejlgrænse. Ingen finepris er refittet.

## 6. Genbrugt S3-evidens, ingen ny strafbeslutning

Den tidligere råaudit i G7_MATCHED_CONTROL_REVIEW.md genbruges:1008/1008
prisinputs i uge13–18 ligger underu=.5. For g gennem(0,1),(.5,a),(1,−1):

\[
 g'(u)=\begin{cases}2(a-1)&u<.5,\\-2(1+a)&u>.5.\end{cases}
\]

I uge16–18 er a≈+1, så responsen er næsten flad NETOP i den observerede
nedre prishalvdel. Ved fastholdt state/B og væk fra knæk:

\[
 |\partial D/\partial u|\le \Xi k|g'(u)|/2.
\]

Dette er en direkte betinget afledt, ikke fuld dynamisk elasticitet.
Hvis g er præcis konstant på et interval, giver priser inden for intervallet
samme forcing og dermed samme statebane fra samme origin. Stærk respons
uden for dette interval er ikke dokumenteret af disse prisobservationer.
Uge13–15 er ikke flade; udsagnet må ikke generaliseres til alle S3-fits.

Vigtigt: den gamle januar-G4 har a≈−1 og dermed flad ØVRE halvdel. Dette er
ikke samme fit eller plateau som de senere S3-uger. Genbrug af januarparret
skyldes tilgængelige matched WM-origins, ikke en påstand om identiske kurver.
Den tidligere forecast-score blev dårligere med straf (ca2.4934%); ingen
ny straf genindføres blot fordi en kontrolkurve er flad. Separat prisidentifikation,
matched origins og frosset historisk kontrolprotokol kræves før næste rangering.

## 7. Kilder og ejerskab af matematikken

| Del | Grundlag | Hvad er vores valg? |
|---|---|---|
| BM-dynamik, f/g og fixed-A single shooting | Vejlederens tavler og uploadet FP2025, afsnit3.2.3; eksisterende BM-kode | Ingen modelændring; fire prisblokke og konkret frossent case |
| WM-tilstande og native A(U) | Uploadet WM4_notes/main.tex og uændret wm_4.py | Separat fælles fixed-A-adapter, ikke erstatning af native objective |
| Kvadratisk tracking/LS, konveks QP/KKT, SQP | Uploadet Nocedal–Wright, Numerical Optimization, kapitler10,16,18 | SLSQP til4D-BM; udtømmende81-face-søg til4D-WM |
| SLSQP stop-/FD-parametre | SciPy officiel dokumentation, lokalSciPy1.17.0 | eps/ftol/startbudget er registrerede numerikvalg, ikke vejlederkonstanter |
| Integral-/lagerbalance | Direkte integration af de bevarede ODE'er ovenfor | Nyt separat energineutralt sammenligningskrav bestilt af Johan |
| Euler-midpoint-korrektion | Direkte teleskopering, vist ovenfor | Samme diskrete E/J for fair rapportering; solver bevares |
| M254 og celler | Vejlederens normaliserede Bernstein/Beta-basis og G8-review | Ingen basisændring her; ny basis er fortsat kun reference |

Officiel SLSQP-reference:
https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
Dokumentationens nuværende side er1.18; brugte options er til stede lokalt1.17.
Målprofilen og energineutraliteten er egne eksplicitte forsøgsvalg, ikke
formler som vi påstår står på vejlederens tavlebilleder.

## 8. Kode, kontrol, checkpoints og budget

- `src/method/bm_1_g7_energy.py`: lille G-adapter, cache, QP og gates.
- `root/g7/night_control.py`: fast protokol, unit-ledger, atomiske checkpoints.
- `root/g7/night_protocol_20261007.json`:23 kildehashes, packet-hash og alle budgets.
- `tests/test_bm1_g7_energy.py`: én selvstændig analytisk QP-/Euler-balancekontrol;
  består på .476894410s process/.002s body. Ingen gammel suite genkørt.
- `root/g7/review_night.py`: uafhængig read-only J/E/raw-scale-/byteaudit og figur;
  ingen model-forward, ny optimizer eller fit.19 råfiler rekonstrueres med egne SHA256.
- `files/output/bm_1/runs/g7_night_raw_20261007.json`: komprimeret JSON med alle
  fulde coarse-/fine-baner, optimizertraces, fem WM-affine probes og ledger.
- `root/g7/night_audit_20261007.json`: læsbar resultattabel og auditreceipt.

Den numeriske runner bruger POSIX signal-timer i dette Linux-miljø; der
hævdes ingen Windows-launcher. Read-only audit/udtræk kræver ikke denne timer.
Faktisk crash/resume er ikke numerisk replaytestet i nat; completed-unit-
identiteter, fulde råfiler og hashes er auditeret uden et nyt modelrun.

Der er374 faktiske nye forwards inklusive BM-Jacobianprober, fem WM-basis-
forwards, to native WM-winnerchecks og seks finechecks. Run-process3.775397722s;
17 completed units bruger3.322515748s (delmængde af processen, ikke ekstra charge).
Råauditbody .183326009s; audit+figurprocess cirka2.54s. Ingen timeout/failed
modelunit. En forkert arbejdsmappe ved oprettelse af review-script blev rettet
før filen fandtes; ingen numerisk kørsel blev startet af den fejl.

Nattens samlede konservative charge er30s inkl. process/check/komprimering,
read-only audit, figur, source-/CI-/publikationskontroller og øvrig lille overhead.
Det er en reservation/konservativ debitering, ikke30s målt modelruntime.
Tidligere G-debit1068.580078s →1098.580078s; rest701.419922s af1800s.
Nattens loft300s er overholdt; F/WM-budgetter ændres ikke.

Source og protocol blev Git-checkpointet før run; alle inputs/imports matches
før første forward. Runneren skriver en fuld tidsreservation før hver unit,
atomisk fsync+replace på completion, og hver valgt coarsebane før fine.
Ved crash bliver ufærdig unit konservativt debiteret med hele reservationen
og genkøres ikke automatisk. Genoptagelse genbruger completed IDs under identisk
protokol/kildehash. Hard crash mellem unitfil og report kan kræve read-only
reconciliation af den bevarede fil, ikke blind genkørsel. En enkelt BM-start
kan højst miste30s arbejde; der er ikke et påstået iterat-exakt SQP-resume.

Råbundlen kan udtrækkes uden modelberegning:

```bash
python root/g7/review_night.py --bundle files/output/bm_1/runs/g7_night_raw_20261007.json --packet files/output/bm_1/runs/g7_matched_contract_20261007.json --output /tmp/g7_readonly_audit.json --extract /tmp/g7_night_restored
```

Kun en ufuldstændig invocation kan behøve runner-resume. Dette run er komplet;
kommandoen ovenfor er dokumentation, ikke en opfordring til at gentage studiet.
Læs `G7_NIGHT_CHECKPOINT.md` før fortsættelse. Stokastisk runtime-/jobdesign:
[G7_M254_RUNTIME_JOB_REVIEW.md](G7_M254_RUNTIME_JOB_REVIEW.md).
