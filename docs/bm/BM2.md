# BM2 og sammenligning med BM1

BM2 har samme respons og observerede forbrug som BM1, men tilf�jer tr�ghed.
Begge modeller bruger de samme input, historiske baselines, normalisering,
fem tr�ningsvinduer, f/g-familier og prognosetidspunkter i ugesammenligningen.
Resultaterne ligger i separate mapper under `files/output/bm_2/`.

## Matematik og parameteridentitet

Den historiske deterministiske BM2 var

\[
\dot X=V,\qquad m\dot V=(D-B)/C_{\mathrm{original}}-\gamma V.
\]

Kun kombinationerne `tau=m/gamma` og `C=gamma*C_original` indg�r efter
division med gamma. Derfor s�ttes gamma=1 i implementeringen, og tau samt
den effektive C estimeres. Vi fors�ger ikke at identificere tre parametre,
som kan �ndres langs en f�lles skala uden at �ndre modellen.

Responsen er u�ndret fra BM1:

\[
\delta=\tanh\{k[f(X)+g(U)]/2\},\quad
q=\delta\begin{cases}1-B,&\delta\geq0,\\ B,&\delta<0,\end{cases}
\qquad D=B+\Xi q.
\]

For en lille, fast beregningsboks anvendes `R=C*V/Xi` internt:

\[
dX=(\Xi/C)R\,dt+\sigma_X X(1-X)\,dW+dL_0-dL_1,
\qquad dR=(q-R)\,dt/\tau.
\]

X ligger i [0,1], og R ligger i [-1,1]. Sidstn�vnte interval er invariant,
n�r R0 ligger d�r, fordi q altid ligger i [-1,1]. Startv�rdier uden for dette
interval er ikke med i denne version. V0 gemmes ogs� i sin oprindelige skala.

St�jen er samme Ito-koefficient som i BM1, p� den samme X-tilstand. Der er
�n Wiener-proces og �n sigma_X. R bliver usikker gennem koblingen til X;
der er ikke indf�rt en ekstra fri st�jstyrke for R. I den stokastiske model
er V den systematiske �ndringshastighed, ikke en almindelig afledt af den
ikke-differentiable X-bane.

BM2-tr�ghed sikrer ikke i sig selv X i [0,1]. Derfor er refleksion en eksplicit
del af denne afgr�nsede BM2: L0 og L1 virker kun ved randen og modvirker
udadg�ende bev�gelse. V/R vendes ikke kunstigt ved et st�d. Det deterministiske
tilf�lde bruger den tilsvarende projektion, mens Fokker-Planck bruger nul flux.
Dette er en specificeret randbehandling ud over den gamle indre BM2-ODE.
Tilstanden er en adf�rdstilstand; modellen opfylder ikke BM1's direkte balance
C*dX/dt=D-B. For lille tau n�rmer den indre deterministiske dynamik sig BM1
efter en eventuel kort starttransient.

## Fit og usikkerhed

Alle seks familier underst�ttes: linear, piecewise-linear, pchip, bernstein,
beta-cdf og logistic. �t f�lles parameters�t estimeres p� fem historiske
24-timers vinduer. Hvert vindue f�r sit eget par (X0,R0). Fittet veksler mellem
parameterfit og fit af de fem par, som BM1's nuv�rende alternerende opskrift.
To startkomponenter kr�ver en todimensional bounded least-squares-s�gning.
De f�lles BM1-parametre og X0'er bruges som startg�t; R0 starter ved q(X0).
Dette genbruger alene information fra de samme tr�ningsvinduer.

tau's fitinterval er 0,05-48 timer. De �vrige gr�nser f�lger BM1, inklusive
C>=0,05 ved MPC. Gamma er fast. Et parameterfit eller et startpar accepteres
kun ved u�ndret/lavere tr�nings-SSE inden for BM2. Der er ingen garanti for,
at BM2's tr�ningsfejl eller prognosefejl bliver lavere end BM1's.

Hvis BM1-referencen kommer fra `compare_fg_week` med f�lles efteroptimering,
f�r BM2 samme ekstra budget til f�lles efteroptimering af parametre og ti
startkomponenter. `converged` f�lger da det f�lles f�rsteordenskriterium;
ellers f�lger det den alternerende opskrifts stopregel. `bm1_variant` og
`joint_fit` registrerer valget. Et optimizer-success alene er ikke et bevis
p� f�rsteordens optimalitet eller et globalt minimum.

D�gnprognosen f�lger den eksisterende BM1-uges ops�tning: fem deterministiske
baner med v�gt 20% og usikkerhed fra de fem startpar. Der kalibreres ikke
processt�j i denne d�gnprognose.

MPC f�lger BM1's procedure ved hvert origin:

1. Fit fem tidligere vinduer med samme ugedag og klokkesl�t.
2. Estim�r sigma_X fra tr�ningsresidualerne med den samme ikke-negative
   variansregression. BM2 bruger en 2x2 kovarians P og overgangens Jacobian A:
   `P_next=A P A.T+diag(X^2(1-X)^2*dt,0)` for enhedsv�rdi af sigma_X^2.
   Forbrugssensitiviteten er `[dD/dX,0]`. Det er en diskret line�r
   st�japproksimation, som i BM1, ikke eksakt SDE maximum likelihood.
3. Byg en f�lles KDE af de fem PAR (X0,R0), med v�gt 20% pr. normaliseret
   komponent. De fem X0'er og R0'er kombineres ikke til 25 uafh�ngige par.
   Minimumsb�ndbredder er 0,025 i X og 0,05 i R; som i BM1 bruges mindst
   halvdelen af den historiske standardafvigelse p� hver akse.
4. Opdat�r hele fordelingen med likelihood for I(origin). Observationen
   afh�nger af X, og kan ogs� opdatere R gennem afh�ngigheden i den f�lles lov.
5. Beregn prognoser p� leads 1-8 timer. Fremtidigt forbrug anvendes kun ved scoring.

Fokker-Planck-ligningen har to tilstande:

\[
\partial_t p=-\partial_x[(\Xi/C)rp]
-\partial_r[((q(x)-r)/\tau)p]
+\tfrac12\partial_{xx}[\sigma_X^2x^2(1-x)^2p].
\]

Transporten bruger minmod-begr�nset rekonstruktion og SSPRK2 med CFL-kontrol;
diffusionen behandles med implicit Euler i to halve trin. Det reducerer kunstig
transportspredning i forhold til f�rsteordens upwind. Sandsynlighed og
positivitet kontrolleres uden efterf�lgende klipning eller renormalisering.
Samlet er tidsmetoden ikke andenordens, da diffusionstrinene er f�rsteordens.
En fuldt implicit f�rsteordens reference findes som `evolve_reference`.
Ved sigma=0
udvikles de v�gtede gitterpunkter deterministisk, s� der ikke tilf�jes kunstig
FV-diffusion. Den eksisterende WM-solver �ndres ikke: dens �bne rand og
diffusionsretning svarer ikke til denne BM2.

## K�rsel fra main

Kort kontrol af 5. maj med alle familier, d�gn og MPC kl. 00:

```powershell
python -B -m src.exe.run_model.compare_bm_week --start 2022-05-05 --end 2022-05-05 --hours 0 --workers 8 --output files/output/bm_2/runs/pilot_2022-05-05
```

Hele ugen 2.-8. maj, alle seks familier, d�gn og alle MPC-origins:

```powershell
python -B -m src.exe.run_model.compare_bm_week --start 2022-05-02 --end 2022-05-08 --workers 8 --mode both --families linear piecewise-linear pchip bernstein beta-cdf logistic --reference files/output/bm_1/runs/fg_week_2022-05-02_2022-05-08 --output files/output/bm_2/runs/week_2022-05-02_2022-05-08
```

Dette er 1050 BM2-jobs. Hver worker har �n BLAS-thread. De gemte BM1-prognoser
genbruges. Manglende referencejobs fra den oprindelige BM1-k�rsel f�rdigg�res
under den NYE outputmappes `reference_completed/` med den eksisterende
BM1-opskrift og beta-CDF-rettelse. En f�lles efteroptimeret BM1-reference
skal v�re komplet i forvejen; v�lg dens mappe med `--reference` og en anden
`--output`, hvis netop den variant skal sammenlignes.

Efter afbrydelse forts�tter samme kommando fra f�rdige jobs. Kode, input,
referencer og indstillinger indg�r i hashes. �ndringer kr�ver en ny outputmappe.
Opdat�r ikke koden under en aktiv k�rsel. Referencemappen overskrives aldrig.

## Kontrol og fortolkning

Standardgitteret er 101 X-celler og 41 R-celler med h�jst 0,25 time pr. FP-trin.
Hvert MPC-job kontrolleres ogs� ved 201x81 celler og 0,125 time. Tr�ningsbaner
og d�gnprognoser bruger adaptiv RK4 med trinfordobling, et stabilitetstrin for
relaksationen og projektion ved X-randen. De kontrolleres med mindst 32
undertrin og en strammere lokal fejltolerance. Tolerancen genbruges
fra referencen (1e-4 i den eksisterende uge). Kontrollen af BM2 omfatter b�de
rum og tid; BM1-referenceflaget f�lger dens oprindelige kontrol og m� ikke
fortolkes som en identisk rumlig kontrol.

`comparison.csv` indeholder parvis RMSE/MAE, originalskala, baseline, antal jobs,
konvergens og numeriske flag. `fit_comparison.csv` viser de enkelte fits,
tau, randbes�g og tid. `comparison_by_lead.csv` opdeler MPC pr. lead.
`comparison.png` viser BM1/BM2 og baseline. Hvert job gemmer parametre,
startpar, st�jestimat, prognoser og kontrolresultater i `fit.json` og CSV.
Et gennemf�rt job er ikke n�dvendigvis et konvergeret eller numerisk best�et job.

Pilotkontrollen p� 5. maj viste stabile tr�ningsbaner, men ogs� gitterf�lsomme
MPC-prognoser, is�r beta-CDF ved X-randen. Standardgitteret er derfor ikke en
garanti for tilstr�kkelig n�jagtighed. `all_paired_numeric_checks_passed` og
`bm2_max_forecast_refinement_difference` i sammenligningen g�r dette synligt.
RMSE-rangeringen omfatter alle f�rdige par og er forel�big, n�r kontrollerne
ikke best�r. En st�rre k�rsel kan bruges til screening; en endelig konklusion
kr�ver afklaring af de relevante numeriske afvigelser. St�rre gitre v�lges med
`--x-steps` og `--r-steps` i en ny outputmappe.

BM2-worker-tid inkluderer nyt BM2-fit, prognose og finere numerisk kontrol.
Startg�ttets allerede udf�rte BM1-fit er genbrugt; en fuld k�rsel fra r� data
skal ogs� betale for at fremskaffe det. Sammenlign ikke disse tider som et
rent m�l for forholdet mellem de to differentialligningers l�setid.

Relevant kode: `src/model/bm_2.py`, `src/params/bm_2.py`,
`src/exe/run_model/bm_2/predict.py`, `src/exe/run_model/compare_bm_week.py`.
