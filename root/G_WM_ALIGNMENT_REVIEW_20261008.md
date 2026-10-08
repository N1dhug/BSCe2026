# G/WM: sidste genvurdering før implementering

8. oktober 2026. Kildebase `a06a2fcd3640b141dc89edf9d07554bc97d3cd33`.
Dette supplerer [kode-/matematikreviewet](G_CODE_MATH_REVIEW_20261008.md).
Den [aktive plan](G_COMPLETION_PLAN_8H_20261008.md) er autoritativ for rækkefølge
og scope. Ingen historisk model er genfittet, og ingen tidligere numerisk
underkendelse er ændret. F er fortsat et særskilt projektspor.

## 1. Hvad WM faktisk gør

Den konkrete reference er WM4's additive procesusikkerhed efter et
deterministisk fit, ikke alle tidligere WM-varianter under ét.
[Modellen](../src/model/wm_4.py) har, i dage,

\[
dF=R\,dt,\quad dR=(r_FF+r_UU+r_RR)\,dt+\sigma\,dW_t,
\qquad I=O+R.
\]

Med frosne koefficienter og eksogene signaler er middelbanen deterministisk.
For Euler-overgangen A og støjretningen e_R er kovariansrekursionen

\[
A=\begin{pmatrix}1&h\\hr_F&1+hr_R\end{pmatrix},\quad
P_{j+1}=AP_jA^T+h q_j e_Re_R^T.
\]

`process_variance_kernel` beregner nøjagtigt den tilhørende diskrete kernel
K med P0=0. Ved konstant q=σ² er procesvariansen kq, k=K·1.
Det er eksakt for den valgte Euler-diskretisering, ikke en garanti for
kontinuert tidsnøjagtighed. `sensitivity` giver J, så initialbidraget er JP0Jᵀ.

[`constant_diffusion_qp`](../src/params/wm_4.py) tilpasser **varians**:

\[
\min_{q\ge0}\;\frac1n\sum_{i\in M}(k_iq+v_{0i}-v_i)^2
                     +\frac{\lambda}{2}q^2,
\quad q^*=\max\!\left(0,\frac{k^T(v-v_0)}{k^Tk+n\lambda/2}\right).
\]

Summationerne bruger fitmasken; λ er WM's numeriske ridge 1e-10.
Ved nul følsomhed må en ny G-adapter rapportere manglende identifikation,
selv om regularisering kan returnere tallet nul.
Variansmålene er samplevarians (ddof=1) af fem tilsvarende historiske
ugedages **rå normaliserede demand**, jf.
[`uncertainty_frame`](../src/exe/run_model/wm_4/analyze_observation_uncertainty.py).
Det er ikke varians af modelresidualer givet identiske priser/temperaturer.
Dermed kan spredningen også rumme inputvariation, startvariation og modelfejl.
Den estimerede σ er en kalibreret spredningsparameter, ikke isoleret fysisk støj.

## 2. WM-varianterne må ikke blandes umærket

| Vej | Initialisering og usikkerhed | Konsekvens for G |
|---|---|---|
| [Proces alene](../src/exe/run_model/wm_4/fit_constant_diffusion_qp.py) | Historisk medianstate; P0=0; q til rå varians | Relevant første, betingede reference. P0=0 betyder valgt start uden usikkerhed, ikke kendt sand/oracle-start. |
| [Proces + initial](../src/exe/run_model/wm_4/predict_all_days_uncertainty.py) | Historisk middelstate; samplekovarians P0; q fittes først med v0=0, derefter lægges JP0Jᵀ til | Mulig dobbelt tilskrivning af samme empiriske spredning. Ikke et fælles variansfit; kopieres ikke som sådan. |
| [Kun initial](../src/exe/run_model/wm_4/predict_initial_state_uncertainty.py) | Historisk middel og kovarians, normalbånd via sensitivitet | Wrapperens returværdi refererer til ubundet `deterministic` på linje 80. Statisk konstateret; hele wrapperen er ikke kørt her. |

P0 i WM er samplekovarians af historiske states, ikke kovarians af et
estimeret gennemsnit P0/n. Proces+initial-wrapperens oracle-fit bruger
måldagens observationer til en særskilt diagnostik; dette gør ikke i sig
selv dens historisk beregnede middelprognose lækkende, men wrapperen er
derfor ikke en ren fremadrettet prognoseindgang.

[`prediction_setup`](../src/exe/run_model/wm_4/predict_daily.py) kræver også
fremtidig I sammen med U/Temp. Ny forecast-kode skal kunne nøjes med
eksogene signaler og et frosset snapshot; observationer tilføjes ved scoring.
WM-wrapperfejlen registreres her, men en generel oprydning i WM er ikke
forudsætning for G's næste trin. Genbrug de testede kerner gennem adaptere.

## 3. Yderligere sammenligningsproblemer

[`aligned_data`](../src/exe/run_model/wm_3/fit_weekly.py), som WM4 bruger,
normaliserer demand med Jan–Okt-periodens min/max og WM4-pris med hele 2022.
Det er kodefakta. Disse skalaer kan bruge oplysninger efter en konkret
prognoseorigin og er ikke samme kontrakt som G's træningsbaserede demand-
skalering og erklærede eksogene prisdomæne. Historiske WM-rapporttal må
derfor ikke sættes direkte op imod ny G som et fælles leakage-frit benchmark.

Den fælles adapter skal bruge samme rå population, observationstidsstempler,
cutoff, demand-skala, deklarerede prisgrænser og pris/temperaturscenarie.
WM får signed pris U=2u−1. Prognoser fryses på samme mandag; et WM-fit
opdateret senere i ugen er en anden informationsprotokol.
WM's eksisterende fem dagsfit uden G's 24h warm-up er en model-/fitforskel,
der oplyses, ikke skjules. G's syv dagsbaner er ikke én kontinuert uge-state.

Bevar målte observationers maske. Manglende observationer må ikke gøres
til målte ved interpolation; råskalaens fysiske enhed og punkt/timeinterval-
fortolkning er stadig metadata, som skal verificeres. Udviklingseksemplet
kan bruge den eksisterende mærkede punktapproksimation, uden at kalde den
historisk validering af en interval-likelihood.

## 4. Konkrete valg for G's støjlag

G beholder sin model

\[
dX=\mu(X,u,B)\,dt+\sigma_xX(1-X)\,dW_t,\qquad
\mu=(D-B)/C,\quad t\text{ i timer}.
\]

Kun amplituden er konstant; diffusionen er stateafhængig. Ingen ekstra
Xi- eller 1/C-faktor tilføjes til diffusionen. Ved ren tidsenhedskonvertering
er σ_day=√24 σ_hour og C_day=C_hour/24. WM's og G's støjamplituder kan
alligevel ikke fysisk sidestilles: de virker på forskellige states.

Første afgrænsede forsøg fastholder S3 og P0=0 ved hver valgt dagsstart.
Der fittes én fælles q=σ_x² for den mandagsfrosne uge fra de 35 historiske
forløb, uden refit af C,k,Xi,a eller baseline. Bevar G's eksisterende
ugedag/time-profiler: gennemsnit af de fem historiske enhedsvarianser
per bin. Fit q til de tilsvarende empiriske variansbin, målte værdier,
mindst to observationer; time 0 udelades fra q-fit og rapporteres særskilt.
Dette er en eksplicit forskel fra WM-wrapperens q per dag og alle 24 timer.
En streng fælles kalibreringskontrol kan poole WM-kerner under samme
en-q/uge-maske; ændr ikke gamle WM-resultater til denne nye protokol.

Den billige første kandidat er eksisterende G6a-enhedspropagation med et
**nyt navngivet variansfit**, ikke en ændring af den gamle std-fitfunktion:

\[
\dot P=2\mu_xP+q[X(1-X)]^2,\quad
\operatorname{Var}(D)\approx D_x^2P.
\]

Denne lokale approksimation arver ikke WM's eksakte middel-/kovarians-
egenskab: generelt E[D(X)]≠D(E[X]), og E[X] følger ikke den deterministiske
bane. Den gamle [G6a-undersøgelse](G6A_LINEAR_NOISE_REVIEW.md) fejlede med
statevarians op til ca. 3,935 mod det nødvendige maksimum 0,25 på [0,1].
Et ændret fitkriterium kan ikke på forhånd antages at reparere dette.
P≤0,25 er nødvendigt, men ikke tilstrækkeligt for en brugbar approksimation.

Derfor kun én budgeteret pilot efter S3-snapshot: bevar forfiningskrav
1e-5 for center og 1e-4 for std/profil; kontrollér endelige tal, q=0,
stategrænser/varians og sammenhold en repræsentativ bane med eksisterende
ikke-lineær FV-propagation ved frosne parametre. Frys konkret pilot,
grid, tolerancer og runtime før kørsel. Ingen klipning, nedjustering af σ
eller efterfølgende tolerancetilpasning for at få accept.

Hvis den lokale vej fejler, mærkes den **rejected approximation**.
En kort vurdering af fast-parameter FV som propagation af samme SDE må
ske inden for støjlagets tidsramme; det er ikke en parameterbank og har
ikke automatisk et lineært q-fit. Et bredt FV-kalibreringsstudie eller nye
støjmodeller startes ikke automatisk. Hvis intet gyldigt lag kan afsluttes
inden for rammen, leveres den fungerende deterministiske del med støjlaget
åbent og en konkret restopgave. Hele målet erklæres ikke færdigt.

## 5. Output, styring og accept

Fælles tabel skelner `deterministic_prediction`, `predictive_mean`,
`process_variance`, `initial_variance`, `total_variance`, bånd og status.
En LNA-center er mærket approksimation; et deterministisk tal omdøbes ikke
til eksakt stokastisk middel. Normalbånd er punktvise nominelle 95%-bånd,
ikke 95% samtidig dækning af døgnet. Procesbånd indeholder ikke automatisk
parameter-, baseline-, input- eller observationsusikkerhed.

Samme scoremaske, MSE/RMSE/MAE/bias, dækning, middelbredde og intervalscore
bruges for G og WM. Intervalscore er U−L + 2/α·(L−y)+ + 2/α·(y−U)+.
Hvor normalbånd passer bruges eksisterende `interval_metrics`; andre
kvantiler scores direkte med samme formel. Ingen kunstig [0,1]-klipning.
Manglende/underkendte bånd får status, ikke nulbredde eller godkendt dækning.

Den [deterministiske prisstyring](G7_DETERMINISTIC_CONTROL_REVIEW.md) og
[fælles fixed-target-kontrakt](G7_MATCHED_CONTROL_REVIEW.md) bevares.
Parameterbankens udsættelse fjerner ikke dette allerede udførte arbejde.
Svagt identificeret prisrespons i historisk S3 betyder stadig, at lavere
model-cost ikke dokumenterer faktisk historisk styringsgevinst.

## 6. Kontroller udført i dette review

[Maskinlæsbar kvittering](g_wm_alignment_checks_20261008.json) indeholder
kildehashes og seks små kontroller: WM-kernel mod direkte kovariansrekursion
(maksfejl 2,78e-17), QP mod skalarformel (3,10e-10), initialbidragets
subtraktion i kernefit, q≥0-randen, forskellen mellem std-/variansfit samt
statisk ubundet navn i WM-wrapperen. Sidste kontrol bekræfter et fund,
ikke en bestået historisk wrapperkørsel. Ca. 0,025 s eksklusive imports.
En første ugyldig syntetisk r blev korrekt afvist af WM's domænekontrol;
testinput blev rettet, produktkode blev ikke ændret.

De 39 tests i det foregående review er tidligere evidens; de er ikke
genkørt eller talt som nye her. Ingen fuld WM/G-prognose eller stokastisk
accept er kørt i dette review. Den næste implementering er afgrænset i
[arbejdsinstruksen](G_NEXT_STEP_20261008.md).
