# G6a — billig variansreference, 6. oktober 2026

## Beslutning og scope

Den afgrænsede G6a-reference er implementeret og numerisk kontrolleret. **De kalibrerede procesbånd er ikke accepteret som brugbare 95 %-prognoseintervaller.** En enkelt konstant støjfamilie matcher ikke den observerede spredning tilstrækkeligt, og lineariseringen bryder fysisk sammen i uge 13. Ingen ekstra støjfamilie, sigma-bounds, målefejl eller P0 er tilføjet efter scoring. Næste trin er det planlagte **G6b-panel med FP/FV, Beta-mixture og Euler–Maruyama**, som skal bruge en fast, lille syntetisk støjstyrke og sammenligne med denne reference. De store historisk kalibrerede sigmaer er ikke godkendte SDE-parametre.

Der bruges kun gemte **S3, lambda=0** fra ISO2022 uge 13 og 18, valgt som første/sidste af G4's seneste blok før G6a-scoring. Strukturelle parametre og f/g genfittes ikke. Hver mandag bruger 35 historiske dage, og alle syv dagsprognoser fryses med dette cutoff. G4 har allerede brugt disse uger til modelvalg: dette er udviklingsdiagnostik på eksponerede uger, ikke en uafhængig slutvalidering. Realiseret pris/temperatur, den oprindelige frosne demand-/prisskala og legacy hourly-point approximation bevares.

## Matematik og begrundelser

Skriv drift som mu for at skelne fra G4-kurvens knudeparameter a:

$$dX_t=\mu(X_t,t)dt+\sigma X_t(1-X_t)dW_t,\qquad \mu=(D-B)/C.$$

G4's eksisterende demandfunktion genbruges:

$$\delta=\tanh\left(\frac{k}{2}[1-2x+g_a(u)]\right),\qquad
D=B+\Xi\delta\gamma,\quad
\gamma=\begin{cases}1-B&\delta\ge0\\B&\delta<0.\end{cases}$$

Her er f lineær, og g er den fælles to-stykkers kurve fra det gemte fit. På hver glat gren:

$$D_x=-k\Xi\gamma(1-\delta^2),\qquad \mu_x=D_x/C\le0.$$

Ved delta=0 kan D_x have et knæk, når B ikke er 1/2. Koden bruger gren delta>=0 ved lighed, og forfining kontrollerer følsomheden over faktiske tegnskift. Priser holdes inden for hver fysisk time, baseline er G4's faste interpolerede 60/time-input. Ingen efterfølgende state- eller variansklipning.

Lineær støjapproksimation: skriv X cirka x_bar+epsilon og behold første orden i drift og nulte orden i diffusion:

$$d\epsilon_t\approx\mu_x(\bar x,t)\epsilon_tdt+\sigma\bar x(1-\bar x)dW_t.$$

Itôs formel for epsilon² og E[epsilon]=0 giver

$$\dot{\bar x}=\mu(\bar x,t),\qquad
\dot P=2\mu_x(\bar x,t)P+\sigma^2[\bar x(1-\bar x)]^2,\qquad P(t_0)=0.$$

Denne approksimation omfatter ikke højere ordens diffusionstermer eller tilbagekobling af variansen til middelbanen. G4-banen er derfor ikke påstået at være E[X] for den nonlinear SDE. P0=0 betyder en kendt betinget initialtilstand. Usikkerhed i parameter, baseline, initialtilstand, kommende input og en separat målefejl er udeladt.

Historiske statebaner har præcis 24 fysiske warmup-timer, som i G4. P sættes først til nul ved den historiske target-midnat. Warmup bestemmer x, men tilføjer ikke varians. Det gør historiske og kommende dagsbaner sammenlignelige under samme P0-policy. Hver af ugens syv prognosedage starter særskilt fra medianen af de fem relevante historiske x ved 24 timer; der hævdes ingen kontinuerlig 168-timers stokastisk statebane.

Delta-metoden giver punkt-demand-variansen og nominelle bånd:

$$V_D\approx D_x^2P,\quad s_D=|D_x|\sqrt P,\quad
\widehat D\pm z_{0.975}s_D.$$

Båndene klippes ikke til [0,1]. Dette er **punktvarians**. Timegennemsnit/timeenergi kræver kovarians med et demand-integral; gennemsnit af punktvarianser er ikke korrekt. Den nye funktion afviser hourly-mean i stedet for at ommærke legacy-data.

### Billig, positiv numerik

Genbrug G4's RK4-statebane med fine samples. For et trin h evalueres A=2mu_x og q=[x(1-x)]² ved midtpunktet (x fås ved lineær interpolation af fine endpoints). Den eksakte overgang for lokalt frosne koefficienter er

$$P_{j+1}=e^{A_jh}P_j+q_j\frac{e^{A_jh}-1}{A_j},$$

med q_j h når A_j=0. expm1 bruges tæt ved nul. A<=0 og q>=0 giver ikke-negativ varians uden klipning og en stabil overgang ved stor lokal drift. Dette er en midpoint-approksimation til tidsvarierende koefficienter, ikke en eksakt generel overgang. Prisernes spring ligger på trinrande. Knæks påvirkning undersøges med to tidsopløsninger; ingen universel kontinuerlig fejlgaranti påstås.

G4-grids er 12/time i uge13 og 24/time i uge18. Forfining bruger henholdsvis 24 og 48/time, samme baseline-grid og samme frosne parametre. Historiske state-medianer og sigma beregnes også på den finere grid; demandcenter, historisk std-profil og forecast-std kontrolleres. Låste grænser er 1e-5 normaliseret demandcenter og 1e-4 normaliseret standardafvigelse. Sidstnævnte er 0.000244 i rå demand-enhed med den gemte span 2.44. Ingen ekstra grid-søgning efter resultatet.

### Kalibrering med et globalt least-squares-minimum

For konstant sigma og P0=0 er P_sigma=sigma²P_1. Det giver s_model=sigma*w. For hver weekday/clockhour beregnes:

- s_emp: sample-standardafvigelse, ddof=1, over de fem tidligere matchende ugedage, kun målte observationer.
- w: kvadratroden af gennemsnittet af de fem betingede unit-demand-varianser på samme clockhour.

Spring-DST matches på lokal klokktime: manglende 02:00 interpoleres ikke. Uge13 har ét bin med fire observationer; øvrige bins har fem. Mindst to målte observationer kræves. Kun h=1..23 indgår i fit, fordi modelvariansen er nul ved hvert target-origin. Det giver 7*23=161 bins pr. mandag.

$$\widehat\sigma=\arg\min_{\sigma\ge0}\frac1M\sum_j(\sigma w_j-s_j)^2
=\max\left(0,\frac{\sum_jw_js_j}{\sum_jw_j^2}\right).$$

Dette er den eksakte globale løsning af den én-dimensionale konvekse LS-opgave. SciPy TRF er unødvendig her; der er ingen nonlinear parametersøgning, finite-difference-prober eller to seed-fits. Ved nul model-sensitivity rapporteres unidentifiable, ikke et godkendt sigma=0-fit. Den generelle G4-TRF ændres ikke.

Observed spredning indeholder variation i baseline, input og måling, mens w er betinget procesvarians omkring separate meanbaner. Mellem-meanvariansen er **ikke** lagt ind. Sigma er derfor en WM-kompatibel spredningskalibrering, ikke likelihood-estimation af isoleret processtøj; et stort fitted sigma kan være kompensation for forkert mean/sensitivity eller udeladte usikkerhedskilder. Vi tilføjer ikke en ekstra målevariansparameter efter et dårligt resultat.

Timer/dage: C_days=C_hours/24, sigma_days=sqrt(24)*sigma_hours. Sigma er amplituden i diffusionen, ikke demand-standardafvigelsen. Den analytiske kontrol verificerer samme varians ved fysisk samme tid på begge enheder.

## WM-genbrug og comparator

WM4 `uncertainty_frame` bruger sample-std ddof=1; samme definition genbruges. Coverage, bredde og intervalscore har samme definition som WM4:

$$IS_\alpha=(U-L)+\frac2\alpha(L-y)1_{y<L}+\frac2\alpha(y-U)1_{y>U}.$$

Lavere intervalscore er bedre: det belønner smalle bånd, men straffer manglende dækning. Alpha=.05. Rapportér både alle timer og uden hver dags første time samt pr. horisont. Punkt-MSE står ved siden af båndmålene.

Seneste main blev kontrolleret på `7b426dcdd1cac0c35c057cdec9031d1104b03bb9`. WM4's `test_diffusion_hypotheses.py` (blob `4788394ff329e97b21055a70b74c0e17be4323d4`) er ændret siden G4, men bruger fortsat `optimize_prediction_initial_state` fra det fulde target-døgn og beskriver selv oracle-centered diagnostik. Den bruges ikke som en kausal G6a-comparator. Et fuldt matched WM-støjbenchmark ville kræve særskilt budgetterede WM-fits og er ikke gennemført her. Ingen WM- eller F-kode ændres.

Den billige comparator bruger s_emp fra de fem historiske weekdays omkring **samme frosne G4-forecastcenter**. Den tester om BM-procespropagation giver bedre intervaller end direkte historisk spredning. Den er ikke WM's procesmodel og er heller ikke automatisk et kalibreret 95 %-interval.

## Faktiske resultater

Alle 336/336 fysiske target-timer scores; observationer under den gamle demandskala og negative råværdier beholdes. Forecasts og støjparametre er gemt før target-scoring. Samme center giver samme pooled punkt-MSE 0.1734819077 for begge bånd.

| Bånd | Dækning alle 336 | Dækning uden 14 dag-origins | Bredde alle | Intervalscore alle |
|---|---:|---:|---:|---:|
| BM lineær procesapproksimation | 45.83 % | 47.83 % | 0.84149 | 6.84732 |
| Historisk empirisk spredning, samme center | 88.99 % | 88.82 % | 1.43883 | 2.14559 |

| Uge | Sigma per sqrt(time) | Training sigma-RMSE, normaliseret | BM-dækning | Empirisk dækning | BM intervalscore | Empirisk intervalscore |
|---|---:|---:|---:|---:|---:|---:|
| 13 | 1.65525 | 0.098763 | 60.71 % | 82.74 % | 5.50056 | 2.38652 |
| 18 | 5.61439 | 0.127489 | 30.95 % | 95.24 % | 8.19407 | 1.90466 |

Training sigma-bias er -0.04115 og -0.07364 normaliseret; én amplitude fanger ikke profilformen. Uge18 har en næsten flad/mettende G4-demandrespons og state tæt på randen, som gør conditional demand-sensitivity lille. Stor sigma kan ikke rette dette eller mean-fejlene.

**Fysisk modeksempel til accept:** En random variable i [0,1] har Var(X)<=1/4. Uge13's approksimerede P når 3.93538 og overskrider 1/4 på 91.67 % af forecast-timerne. Det kan ikke være den faktiske varians for den invariant bounded SDE. Uge18's max P er 0.00092845, men state ligger så tæt på randen, at det symmetriske 95 %-statebånd stadig forlader [0,1] på 95.83 % af timerne. Begge uger har 95.83 % sådanne randoverskridende statebånd. Dette er diagnostik af Gaussian/linearisering, ikke en numerisk state-overshoot i RK4; state selv klippes aldrig.

Der hævdes ikke normalfordelte demandfejl, 95 %-dækning, identificeret fysisk sigma, causal prisrespons eller korrekt historisk intervalenergi. To eksponerede uger tillader ingen generel statistisk konklusion eller uafhængighedsbaseret signifikanstest. Negativt resultat er en fuldført afgrænset undersøgelse, ikke accept af en fungerende intervalmodel.

## Kontroller og runtime

- Tre fokuserede unittest-metoder består, 0.007 s: analytisk equilibrium-OU-varians inklusive warmup-reset, sigma=0 og time/day-skala; analytisk demandderivative mod finite difference og afvisning af intervaloperator; eksakt nonnegative LS samt intervalscore.
- Beholdt runtimepilot: første fem historiske unit-propagationer, 0.214 s, ingen gentagelse. Loft60 s/30 propagationer.
- Samlet historisk prøve: **168 propagationer, 2.283 s**, under loft500/300 s. 2 uger * (35historier+7forecasts) * 2grids. Ingen nye strukturelle fits, ingen Monte Carlo eller FP.
- Uge13 center-/std-/training-profil-gridfejl 2.6634e-9 / 2.8513e-8 / 2.1396e-8.
- Uge18 center-/std-/training-profil-gridfejl 2.5513e-6 / 1.0666e-5 / 6.0569e-6.
- Begge numerikchecks består. G4-centerreplay ændrer højst 9.86e-17 normaliseret demand. Der er 43 registrerede historiske tegnskift i uge13 og nul i uge18.
- Fuld unittest discovery og tidligere G4/WM-studier er ikke genkørt. CI-status rapporteres separat efter publicering.

## Kode og reproduktion

`src/method/bm_1_linear_noise.py` giver den lille G6a-kernel, derivative, analytic LS og intervalmetrics. `src/exe/run_model/bm_1/linear_noise.py` fryser første/sidste G4-uge, kontrollérer exact raw-fingerprint/scales og saved-fit-hashes, og gemmer pointforecasts før scoring. G4's kerner ændres ikke. Source SHA256 i protocol.json matcher den kørte kode.

Kør fra repo-root med originalt G4-råarkiv pakket ud og eksisterende G4-run:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python -B -m src.exe.run_model.bm_1.linear_noise --raw-csv "PATH/raw_inputs.csv" --saved-g4 "PATH/run" --output "files/output/bm_1/runs/g6a_new"
```

Output skal være en ny mappe; ingen automatisk genfit eller retuning. Den gemte G6a-prøve genbruges ved review, så Johan behøver ikke køre denne kommando nu. Ved budget-/numerikstop bevares filer; stop er ikke accept. Den første G6a-run har ingen resume-mekanisme, fordi hele panel er målt til få sekunder. G6b/G8/G7's selvstændige kontrolkrav og budgetter bevares.

## Kildepræcisering fra efterfølgende matematisk gennemgang

Lecture08 slide10, (20), giver den lineære kovariansligning Pdot=AP+PA^T+GG^T. Alba Gonzalo Primos upload `09-MSC_AGP_FINAL-1.pdf`, afsnit6.1.2, (6.4)/(6.7), viser BM-EKF-processkovarians og outputvarians. Higham2001 afsnit8, (8.3), giver Itô-korrektionen bag generator-/momentargumentet. Den konkrete five-weekday-std-kalibrering af sigma og startpolitikken P0=0 var vores udviklingsvalg, ikke en kildevalideret procedure. G6b bruger en særskilt samme-prior-reference med P0=1/164; den ændrer ikke denne historiske kørsels definition eller negative resultat.
