# G: matematik, argumentation og udført arbejde

Samlet 8. oktober 2026 efter genoptagelsen af den slettede chat. Grundlag: repositoryversion `56f7fc0a184d2ea3b624f64f5de6f681d29a48ea` samt de tilgængelige, uændrede projektkilder. Dette dokument samler og forklarer den gemte matematik; det rapporterer ingen nye modelkørsler. De oprindelige reviews er bevaret særskilt med deres daværende status og råtal.

**Det gemte faglige grundlag er fundet.** Det omfatter modellen, estimationen, prognoserne, støjapproksimationen, fordelingsmetoderne, filteret og styringsforsøgene. Formuleringer og mellemregninger, som kun stod i den slettede chat, kan ikke identificeres som genfundne. Denne sammenstilling er derfor en dokumenteret gennemgang af det bevarede arbejde, ikke en kopi af den slettede samtale.

## 1. Læserute og evidensstatus

| Del | Hvad vi har lavet | Status og vigtigste originalnoter |
|---|---|---|
| G0–G1 | Fælles inputkontrakt, deterministisk BM-forward, enheder, balance og eksakt konstant-input-reference | Implementeret; [G-planen](G.md), [januargrundlaget](G_JANUARY.md), [BM-reference](BM_REFERENCE.md) |
| G2–G3 | Januarfit og frosset 168-timers prognose fra egen sluttilstand | Gennemført udviklingsforsøg; historisk målesemantik/population stadig et forbehold; [januar](G_JANUARY.md) |
| G4 | To-stykkers prisrespons, strafprøve, weekday- og 35-dagesstrategier | Afgrænset punktprotokol gennemført; S3 uden straf valgt; [uger 13–18](G4_VALIDATION_WEEKS_13_18_REVIEW.md) |
| G5 | Figurer, fejlmål, forklaringer og faglig dokumentation | Udviklet løbende; denne samling er ikke en særskilt afsluttet slutrapport eller et mundtligt forsvar |
| G6a | Lineær støjapproksimation og spredningskalibrering | Gennemført negativ undersøgelse: båndene er ikke accepteret som brugbare 95 %-intervaller; [review](G6A_LINEAR_NOISE_REVIEW.md) |
| G6b | Fokker–Planck/FV, mixture og Euler–Maruyama på samme syntetiske case | Numerisk sammenligning gennemført; M20 afvist, M38 brugbar i denne åbne case; [review](G6B_DISTRIBUTION_PANEL_REVIEW.md) |
| G8 | Punktfilter, fem-node-parameterbank, posterior-basis og M254-forfining | Eget M254513-filter og faktisk resume består deklarerede syntetiske gates; [forfining](G8_M254_REFINEMENT_REVIEW.md) |
| G7, deterministisk | Fireblok-prisstyring og særskilt variant med bevaret døgnforbrug | Små forsøg gennemført; lokale BM-løsninger, ikke generel optimalitet; [kontrol](G7_DETERMINISTIC_CONTROL_REVIEW.md), [balancekrav](G7_ENERGY_NEUTRAL_REVIEW.md) |
| G7, stokastisk | Egne posterior-forwards, kort EM/feedback-kontrol, tre-plan-søgning og implementeret døgnstyring | Korte numeriske forsøg og orchestration med stubkerner gennemført. **Det fulde numeriske døgnforsøg er ikke gennemført eller accepteret**; [seneste kontrol](G7_DAY_COMPLETION_REVIEW.md), [recovery](G7_CHAT_RECOVERY_20261008.md) |

Den aktuelle prioritet er at færdiggøre BM. De tidligere BM/WM-kontrolforsøg nedenfor bevares som historik; de er ikke en endelig rangering af modellerne. F4-H2 er et andet arbejdsforløb og indgår ikke i G's resultater eller tab.

Det komplette filindeks findes i [kildeindekset](G_MATHEMATICS_SOURCE_INDEX_20261008.md). Det ledsagende manifest registrerer version, filstørrelser og SHA256, så originalerne kan kontrolleres uden at gentage beregningerne.

## 2. Notation, normalisering og observationer

Vi bruger fysisk tid i timer, tilstand $x\in[0,1]$, normaliseret pris $u\in[0,1]$, baseline $B$ og demand $D$. $C>0$ er en tidsstørrelse, $k>0$ en responsstejlhed og $\Xi\in[0,1]$ fleksibilitetsamplituden. I parameterbanken er $\rho=1/C$. $\sigma_x$ betegner processtøjens amplitude; den er ikke demand-standardafvigelsen. $A$ er en på forhånd fastlagt målprofil til kontrol.

Når der findes en fysisk effektskala, er $C=C_{\mathrm{phys}}/P_{\mathrm{scale}}$. G's legacy-data giver ikke i sig selv grundlag for at kalde et estimeret $C$ en identificeret fysisk batterikapacitet. En rå-demand-enhed er heller ikke automatisk verificeret kW eller kWh.

Med en affin normalisering

$$p=p_{\min}+s_pu,\qquad D_{\mathrm{raw}}=d_{\min}+s_DD$$

skal målprofilen omregnes med samme map. Derfor er

$$D_{\mathrm{raw}}-A_{\mathrm{raw}}=s_D(D-A),\qquad J_{\mathrm{raw}}=s_D^2J.$$

Januargrundlaget bruger $p_{\min}=0.74$, $s_p=2604.02001$, $d_{\min}=0.40230769230769237$ og $s_D=2.44$. WM's signed priskoordinat er $U_{\mathrm{WM}}=2u-1$. Ens normaliserede tal sendt direkte til begge modeller ville derfor ikke nødvendigvis betegne samme råpris. Spotprisen er en proxy, ikke dokumentation for en kundetarif eller kausal prisintervention.

Observationstypen bestemmer operatoren $\mathcal H_i$:

$$\mathcal H_i[D]=D(t_i),\quad
\frac{1}{h_i}\int_{t_i}^{t_i+h_i}D(t)\,dt,\quad\text{eller}\quad
P_{\mathrm{scale}}\int_{t_i}^{t_i+h_i}D(t)\,dt.$$

Det er henholdsvis punktværdi, intervalmiddel og energi. G's gennemførte historiske fits bevarer deres **hourly-point approximation**. Arbejdsantagelser om timeforbrug/kWh/hour-start ændrer ikke retrospektivt denne numeriske operator. G6a's punktvarianser og G8's punktlikelihood er heller ikke intervalenergimodeller.

G-importen gennemsnitter tilgængelige målere med skip-NaN. 624 aggregerede januartimer er dermed ikke dokumentation for 624 komplette timer for hver af de 13 MED-huse. En ny fast-populations- eller intervalvariant kræver sin egen dokumenterede kontrakt; F4-H2's kontrakt kan ikke automatisk tilskrives det gamle G-fit.

Ved timer til dage gælder

$$C_d=C_h/24,\qquad \sigma_{x,d}=\sqrt{24}\,\sigma_{x,h},\qquad J_h=24J_d.$$

Diffusionens kvadratrodsfaktor følger af Brownsk tidsskalering. Tid, inputbrud og integrationsmål skal omregnes samtidig.

Kilder: [G-kontrakten](G.md), [januargrundlaget](G_JANUARY.md), [matched kontrol](G7_MATCHED_CONTROL_REVIEW.md), [BM-reference](BM_REFERENCE.md).

## 3. Batterimodellen, balance og gennemførlighed

Den bevarede BM1-model er

$$\delta(x,u)=\tanh\left(\frac{k}{2}[f(x)+g(u)]\right),$$

$$\gamma(\delta,B)=\begin{cases}1-B,&\delta\geq0,\\B,&\delta<0,\end{cases}
\qquad D=B+\Xi\gamma\delta,\qquad \dot x=\frac{D-B}{C}.$$

De kontinuerte, ikke-voksende responser opfylder $f(0)=g(0)=1$ og $f(1)=g(1)=-1$. Ved $x=0$ er $f(x)+g(u)\geq0$, så driften peger indad; ved $x=1$ er summen ikke-positiv, så driften igen peger indad. Under den nødvendige regularitet bevarer den kontinuerte model derfor intervallet. Det er ikke et løfte om, at ethvert endeligt solvertrin gør det. Numeriske stateoverskridelser kontrolleres, frem for at blive skjult ved clipping.

Integrér tilstandsligningen:

$$C[x(t_b)-x(t_a)]=\int_{t_a}^{t_b}(D-B)\,dt.$$

En tilstand i $[0,1]$ giver en begrænset akkumuleret korrektion. Modellen kan flytte forbrug i tid, men kan ikke rette en vedvarende baselinebias ubegrænset. Dette begrunder baseline-/feasibilitydiagnostik før mere fleksible responskurver.

For lineære kurver $f(x)=1-2x$, $g(u)=1-2u$ er de øjeblikkeligt mulige demandgrænser

$$D_{\min}(x)=B[1-\Xi\tanh(kx)],\qquad
D_{\max}(x)=B+\Xi(1-B)\tanh(k(1-x)).$$

Et ønsket demand $d$ kræver, når $0<B<1$, $\Xi>0$,

$$z=\frac{d-B}{\Xi\gamma},\qquad
u=1-x-\frac{\operatorname{artanh}(z)}{k},$$

hvor $\gamma=1-B$ ved $d\geq B$ og ellers $B$. Både $|z|<1$ og $u\in[0,1]$ skal gælde. Et optimizer-successflag gør ikke et umuligt mål muligt.

For konstant baseline og target $A_0$ over varighed $S$ giver lagerrammen et overskud

$$E_{\mathrm{excess}}=\begin{cases}
[(A_0-B)S-C(1-x_0)]_+,&A_0>B,\\
[(B-A_0)S-Cx_0]_+,&A_0<B.
\end{cases}$$

Balance og Cauchy–Schwarz giver den nedre trackinggrænse $J_{\mathrm{det}}\geq E_{\mathrm{excess}}^2/S$. Kapacitet er kun en nødvendig gennemførlighedsbetingelse; prisbounds og responsen kan stadig forhindre tracking.

### Eksakt konstant-input-overgang

På et interval med konstant $B,u$ sættes $q=1-u$ og $z=x-q$. Da $\delta=-\tanh(kz)$,

$$\dot z=-\frac{\Xi\gamma}{C}\tanh(kz),\qquad
\frac{d}{dt}\sinh(kz)=-\frac{k\Xi\gamma}{C}\sinh(kz).$$

Dermed er

$$x_{j+1}=q+\frac1k\operatorname{arsinh}\left[
\sinh(k(x_j-q))e^{-k\Xi\gamma h_j/C}\right].$$

Vælg $\gamma=1-B$ hvis $x_j<q$, ellers $B$. Banen krydser ikke sin ligevægt, så grenen er konstant på intervallet. Ved $\Xi=0$ eller $x_j=q$ er tilstanden konstant. Genstart ved hvert inputskift.

For et intervalmiddel er balanceidentiteten

$$\widehat y_j=B+\frac{C(x_{j+1}-x_j)}{h_j}.$$

Den eksakte formel er en billig, uafhængig kontrol og en mulig intervaloperator under dens antagelser. Den må ikke anvendes som erstatning for januarbaselinens faktisk varierende, lineært interpolerede forløb. Det gennemførte januarfit bruger skalær RK4 med 60 trin/time og kontrol på 120; max-demand-forskellen skal være højst $10^{-5}$. Den eksisterende segmenterede `solve_ivp` og konstant-input-løsningen er separate referencer.

Kilder: [BM-reference, afsnit 1](BM_REFERENCE.md), [G, afsnit 2–3](G.md), [januarnumerik](G_JANUARY.md). Implementation: [BM-kernen](../src/model/bm_1.py), [deterministisk benchmark](../src/method/bm_1_deterministic_benchmark.py).

## 4. Estimation, baseline og frosset prognose

G2 estimerer $\theta=(C,k,\Xi)$, eventuelt også én $x_0$ for det sammenhængende træningsforløb:

$$\min_{\theta,x_0}\frac1{2N}\sum_{i\in\mathcal T}
[\mathcal H_i(D_{\theta,x_0})-y_i]^2.$$

Der er én residual pr. faktisk observation. Et finere internt grid skaber ikke flere observationer. Priser og baseline er faste inputs; kontrolmålet $A$ er ikke et fitmål. Standardjanuarfittet har $x_0=0.5$ og ingen warmup. Udviklingsbounds er $1\leq C\leq744$ timer, $0.05\leq k\leq10$, $0\leq\Xi\leq1$; et eventuelt fittet $x_0$ ligger i $[0,1]$.

Vi bruger logkoordinater for $C,k$ og bounded nonlinear least squares/TRF. Dette håndterer bounds og residualstruktur; det gør ikke problemet konvekst. Two-point-Jacobianen er et erklæret numerikvalg. Complex step passer ikke automatisk til kode med grenvalg og realdomænevalidering. Fire faste starter og alle faktiske residual-/Jacobianprober tælles; bedste gyldige, konvergerede TRAIN-kandidat vælges inden for samme model. Modelvarianter vælges på udviklingsdata.

Nær $\delta=0$ er demandkorrektionen omtrent proportional med $\Xi k$, så de to parametre kan være svære at adskille. Ved $\Xi=0$ bliver demand $B$, og de dynamiske parametre kan ikke identificeres fra demand. Jacobianens singularværdier, rang og aktive bounds er lokal diagnostik; fuld lokal rang er ikke bevis for stærk eller global identifikation. En eventuel strafrække må ikke tælles som ekstra dataevidens i rangdiagnosen.

Baselinen genbruger det eksisterende statistiske output med temperatur-, klokke-, sæson-/ugedagsfeatures og robust ridge/Huber-fit. Gemte koefficienter, featurestandardisering og settings genbruges præcist. Defaults dokumenteret i G er temperaturbase 18, fire timeharmoniske, to sæsonharmoniske, ridge $10^{-3}$ og Huberparameter 1.5. Den eksisterende baselinebegrænsning til $[0,1]$ er en erklæret del af kontrakten og auditeres; der indføres ikke ny clipping af målinger eller state.

G3 fryser fit, baseline, skala og egen simuleret sluttilstand ved cutoff. De næste 168 timer fremskrives uden target-demand, og prognosen gemmes før scoring. Realiseret fremtidig pris/temperatur gør studiet til en **betinget historisk prognose**, ikke en blind operationel prognose. Et snapshot er genbrug af den gamle model; et nyt fit fra gemte parametre ville være et nyt warm-start-forsøg.

På den eksponerede februaruge er den gemte rå-MSE baseline $0.06934621735$, lineær BM $0.06951591504$ og WM $0.07099789830$. BM er her cirka 0.245 % dårligere end baseline. Dette negative baseline-resultat er bevaret. Der er 168 scorede timer; historisk målesemantik og populationsaudit begrænser accepten.

Kilder: [januar og snapshot](G_JANUARY.md), [overdragelse P0–P2](G_IMPLEMENTATION_HANDOFF.md). Implementation: [fit](../src/method/bm_1_deterministic_fit.py), [snapshot](../src/tools/bm_1_deterministic_snapshot.py), [data](../src/tools/bm_1_deterministic_data.py).

## 5. G4: fleksibel prisrespons og hvorfor straf ikke blev beholdt

Den første udvidelse holder $f$ lineær og tilføjer én indre knude $a=g(0.5)$:

$$g_a(u)=\begin{cases}
1+2(a-1)u,&0\leq u\leq0.5,\\
a-2(a+1)(u-0.5),&0.5<u\leq1,
\end{cases}\qquad -1\leq a\leq1.$$

Kurven er kontinuert, har de gamle endpoints og ikke-positive hældninger. $a=0$ giver den gamle lineære kurve. $a=1$ giver flad respons i den nedre halvdel; $a=-1$ i den øvre. Dermed kan en god prognosescore optræde med næsten ingen prisfølsomhed i det faktisk observerede prisområde.

Vi sammenlignede

$$L_\lambda=\frac1N\sum_i(D_i-y_i)^2+\lambda a^2,$$

med residualer $(D_i-y_i)/\sqrt N$ og $\sqrt\lambda a$. SciPy's cost er halvdelen af denne sum og har samme minimizer. Skriv $g_a=g_0+a h$ med $h(u)=2u$ før midten og $2(1-u)$ efter. Da

$$\int_0^1(g_a-g_0)^2\,du=\frac{a^2}{3},$$

er $\lambda a^2$ en straf mod den lineære kurve, ækvivalent til $3\lambda$ gange denne kurveafstand. Den er et udviklingsvalg, ikke en fysisk lov. Ved lokal linearisering $D_i(a)\simeq D_i(0)+q_i a$, og med øvrige parametre fastholdt, fås

$$a_\lambda\simeq
\frac{\sum_i q_i[y_i-D_i(0)]}{\sum_i q_i^2+N\lambda}.$$

Det forklarer shrinkage og mulig bias; det lover ikke bedre forecast.

### De gennemførte valg

Januarpiloten holdt $k=0.22688550696235388$ og $x_0=0.5$ fast og fittede $C,\Xi,a$ for $\lambda=0$ og $0.001$, med to faste starter. 624 observerede TRAIN-timer og 168 februar-scoretimer blev brugt. Ustraffet G4 gav forecast-MSE $0.06950545924$, straffet $0.06951595226$, mod lineær BM $0.06951591504$. Gevinsten over lineær BM var kun cirka 0.015 %. Den på forhånd låste 1 %-regel beholdt derfor den enklere lineære pilotmodel.

Den senere weekday-protokol bruger fem tidligere matchende ugedage. Hver historisk bane er 48 timer: 24 fysisk warmup, dernæst 24 målte fit-timer. Der propagéres ikke gennem de seks dages huller mellem matchende weekdays. Hver forecastdag starter separat fra medianen af modellens egne fem historiske states efter warmup. Et fælles mandagscutoff fryser hele næste uge.

Tre strategier blev sammenlignet:

| Strategi | Træning | Fortolkning |
|---|---|---|
| S1 | Strukturelle parametre på 12-ugersblokken, dernæst weekday-$g$ | Retrospektiv udviklingsdiagnose; fremtidig information i forhold til tidlige origins |
| S2 | Fælles $C,k,\Xi,a$ på fem tidligere dage for hver weekday | 7 særskilte weekdayfits |
| S3 | Fælles $C,k,\Xi,a$ på 35 tidligere dagsbaner | Én pooled fitopskrift pr. cutoff |

På uger 6–12, 1175 målte timer, ændrede straf S2-MSE fra $0.2358689857$ til $0.2144786038$ og S3 fra $0.2065071888$ til $0.2052253468$; baseline var $0.2076682236$. Gevinsten var følsom over for uge 10. Den uge blev beholdt i hovedresultatet. Overlappende træningsvinduer er ikke uafhængige gentagelser.

Den låste S3-opfølgning på uger 13–18 gav 1008/1008 scoretimer:

| Model | Rå pooled MSE |
|---|---:|
| Baseline | 0.1806157175 |
| S3, $\lambda=0$ | 0.1798347601 |
| S3, $\lambda=0.001$ | 0.1843188395 |

Straffen forværrede MSE 2.4934 %, og uden straf forbedrede baseline blot 0.4324 %. Derfor blev **S3 uden straf** valgt. 207 targets var under den frosne demandskalas minimum; 107 var negative i rå enheder. De indgik uændret i score, selv om output i $[0,1]$ ikke kan følge dem alle.

I de sidste tre ustraffede fits lå $a\simeq1$, $k\simeq10$, mens alle forecastpriser var under 0.5. Her er $g'_a=2(a-1)\simeq0$. På en glat demandgren gælder

$$\frac{\partial D}{\partial u}=
\frac{\Xi\gamma k}{2}(1-\delta^2)g'_a(u),\qquad
\left|\frac{\partial D}{\partial u}\right|\leq\frac{\Xi k}{2}|g'_a(u)|.$$

Det er en direkte grund til at skelne god forecastscore fra identificeret priselasticitet og brugbar styring. Alle 12 valgte ugefits bestod den endelige $10^{-5}$-forfining; uge 17–18 krævede 24 mod 48 trin/time efter en bevaret negativ 12/24-kontrol. DST håndteres i fysisk tid; et lokalt døgn kan have 23 timer.

Kilder: [januarpilot](G4_JANUARY_PILOT_REVIEW.md), [weekday](G4_WEEKDAY_DEVELOPMENT_REVIEW.md), [strategier](G4_STRATEGIES_12_WEEKS_REVIEW.md), [uger 13–18](G4_VALIDATION_WEEKS_13_18_REVIEW.md), [prisfølsomhed](G7_MATCHED_CONTROL_REVIEW.md). Implementation: [responsfit](../src/method/bm_1_deterministic_response.py), [strategier](../src/method/bm_1_deterministic_strategy.py).

## 6. G6a: lineær støjapproksimation og dens negative resultat

Den stokastiske BM er en Itô-SDE:

$$dX_t=\mu(X_t,t)\,dt+\sigma_xX_t(1-X_t)\,dW_t,\qquad \mu=(D-B)/C.$$

På en glat gren med lineær $f$ er

$$D_x=-k\Xi\gamma(1-\delta^2),\qquad \mu_x=D_x/C\leq0.$$

Sæt $X\simeq\bar x+\varepsilon$; behold første orden i drift og nulte orden i diffusion:

$$d\varepsilon\simeq\mu_x(\bar x,t)\varepsilon\,dt+
\sigma_x\bar x(1-\bar x)\,dW_t.$$

Itô på $\varepsilon^2$ giver

$$\dot{\bar x}=\mu(\bar x,t),\qquad
\dot P=2\mu_xP+\sigma_x^2[\bar x(1-\bar x)]^2.$$

Det historiske forsøg bruger $P_0=0$ ved hvert target-origin, altså en kendt betinget state. Varians tilbagekobles ikke til middelbanen. Initial-, parameter-, baseline-, input- og måleusikkerhed er udeladt. Den deterministiske bane er derfor ikke påstået at være den eksakte stokastiske middelbane.

For unit-støj, lokalt frosne $A=2\mu_x$ og $q=[\bar x(1-\bar x)]^2$, er det positive trin

$$P_{n+1}=e^{Ah}P_n+q\frac{\operatorname{expm1}(Ah)}{A},$$

med $qh$ ved $A=0$. Ved generel støj multipliceres kildetermen med $\sigma_x^2$. Dette er eksakt for de frosne koefficienter, men kun en midpoint-approksimation til tidsvarierende koefficienter. Delta-metoden giver punktvarians $V_D\simeq D_x^2P$ og nominelle bånd $\widehat D\pm1.96|D_x|\sqrt P$. Timeenergi kræver tidskovarianser og kan ikke findes ved blot at gennemsnitte punktvarianser.

Da $P_\sigma=\sigma^2P_1$, kalibreres $s_{\mathrm{model},j}=\sigma w_j$ mod empirisk weekday-standardafvigelse $s_j$ fra fem tidligere matchende dage:

$$\widehat\sigma=\max\left(0,\frac{\sum_jw_js_j}{\sum_jw_j^2}\right).$$

Her er $w_j$ en unit-støjfølsomhed, ikke en parameterbankvægt. Nul nævner betyder uidentificerbar amplitude. Dette er et globalt minimum af en én-dimensional konveks LS-opgave; ingen nonlinear optimizer er nødvendig. Det er spredningskalibrering, ikke likelihood-estimation af isoleret fysisk processtøj.

På de gemte uger 13 og 18, 336 timer, var dækningen 45.83 % for BM-bånd mod 88.99 % for empirisk historisk spredning omkring samme center. Intervalscore var henholdsvis 6.84732 og 2.14559, med

$$IS_\alpha=(U-L)+\frac2\alpha(L-y)\mathbf1_{y<L}
+\frac2\alpha(y-U)\mathbf1_{y>U},\qquad\alpha=0.05.$$

De fitted amplituder var 1.65525 og 5.61439 pr. kvadratrod time. I uge 13 nåede den approksimerede $P$ 3.93538, mens enhver variabel i $[0,1]$ har varians højst $1/4$. Dermed kan denne approximation ikke være den bounded SDE's faktiske varians. Numerikchecks bestod, men den videnskabelige intervalmodel blev afvist. De store sigmaer blev ikke accepteret som fysiske parametre til G6b.

Kilde: [hele G6a-reviewet](G6A_LINEAR_NOISE_REVIEW.md). Implementation: [linear noise](../src/method/bm_1_linear_noise.py).

## 7. G6b: Fokker–Planck, FV, mixture og Euler–Maruyama

For drift $v$ og diffusionsamplitude $b(x)=\sigma_xx(1-x)$ er

$$\partial_tp=-\partial_x(vp)+\frac12\partial_{xx}(b^2p)
=-\partial_x\mathcal J,\qquad
\mathcal J=vp-\frac12\partial_x(b^2p).$$

Hele produktet $b^2p$ differentieres; faktoren $1/2$ følger af standard-Brownsk Itô-konvention. Med nul randflux bevares total sandsynlighed. Xi=0 fjerner drift/demandresponsen, men ikke processtøjen.

Integreret langs en stokastisk bane er balanceidentiteten

$$C(X_t-X_0)=\int_0^t(D_s-B_s)ds+C\int_0^t b(X_s)dW_s.$$

Under integrabilitet har Itô-integralet middel nul, så den deterministiske balanceform gælder for forventninger med fast C. Støjleddet er derimod ikke nul på hver bane. Man må derfor ikke bruge den deterministiske endpointidentitet som et pathwise energicertifikat for SDE'en. Tilsvarende er $dE[X]/dt=E[\mu(X,t)]$ generelt forskellig fra $\mu(E[X],t)$; det forklarer, hvorfor en deterministisk middelstate-reference er en approksimation.

Den syntetiske døgncase bruger $C=4$, $k=2$, $\Xi=0.4$, $\sigma_x=0.2$, $B(t)=0.5+0.1\sin(2\pi t/24)$ og priser 0.3/0.7/0.3/0.7 i seks-timersblokke. Prioren er Beta(20,20), middel 0.5 og varians $1/164$. Den er særskilt fra G6a's $P_0=0$.

### FV-reference

Cellemasserne er $m_i=\int_{V_i}p\,dx$, og $p_i=m_i/\Delta x$. Den eksisterende upwindgenerator har ikke-negative off-diagonale transitionsrater og $\mathbf1^TQ=0$. På uniformt grid er de indre rater

$$r_i^+=\frac{\max(v_{i+1/2},0)}{\Delta x}+
\frac{b_i^2}{2\Delta x^2},\qquad
r_{i+1}^-=\frac{\max(-v_{i+1/2},0)}{\Delta x}+
\frac{b_{i+1}^2}{2\Delta x^2}.$$

Implicit Euler $(I-hQ)m^{n+1}=m^n$ bevarer masse og positivitet i eksakt aritmetik gennem M-matrix-egenskaben. Dette er stabilitet og conservation, ikke i sig selv kontinuert accuracy. Upwind giver numerisk diffusion. Beta-prioren integreres med CDF-forskelle; cellemidtpunktsmomenter er stadig diskrete approksimationer til dens kontinuerte momenter.

### Dynamisk mixture

Mixturemetoden repræsenterer $\widehat p=\sum_m c_m\ell_m$, $c\geq0$, $\mathbf1^Tc=1$. Den udfører faktisk

$$\widetilde p^{n+1}=\widehat p^n+hL_h\widehat p^n,\qquad
c^{n+1}=\arg\min_{c\in\Delta}\frac12
\sum_i\Delta x[(\Phi c)_i-\widetilde p_i^{n+1}]^2.$$

Der bruges egen eksplicit diskret FP-transport efterfulgt af LS-projektion, med CFL 0.4. Det er ikke en gridfri koefficient-ODE og ikke gentagne fits til FV-referencens nye law. Mixture/FV har også forskellige tids-/fluxdiskretiseringer, så deres forskel er ikke ren basisfejl. Ingen ridge, negative vægte eller clipping indføres for at skjule fejl. SVD på det aktive simplex-face bruges, fordi normal equations kvadrerer konditionstallet.

### EM og det korrekte kontrolmål

Euler–Maruyama simulerer

$$X_{n+1}=X_n+v(t_n,X_n)h+b(X_n)\Delta W_n,\qquad
\Delta W_n\sim N(0,h).$$

Der bruges hverken reflection eller clipping; rå randoverskridelser rapporteres. Kontinuert invarians garanterer ikke diskret EM-invarians. Samme max-step alene garanterer heller ikke samme grid, hvis output-/pausepunkterne ændres.

Demandmomenter beregnes ved $D(x)$:

$$\mu_D=\int D(x)p(x)\,dx,\qquad
v_D=\int D(x)^2p(x)\,dx-\mu_D^2.$$

Generelt er $\mu_D\neq D(E[X])$. Trackingcost er

$$J=\int[(\mu_D-A)^2+v_D]dt=\int E[(D-A)^2]dt.$$

I G6b var fine FV-J $0.2386252699$, M38-J $0.2386779220$ og EM-J $0.2376740757$ med SE $0.001232934$. M38's relative J-afvigelse var cirka 0.0221 %. M20 blev afvist, blandt andet initial-L1 0.291667 mod 0.05. EM havde nul rå randoverskridelser i de simulerede baner. M38 bestod denne åbne case, men den senere snævre filterposterior gjorde samme basis utilstrækkelig.

Registrerede praktiske kontroller omfattede middel 0.005, varians 0.002, CDF 0.03, kvantiler 0.02, relativ J 2 %, masse $10^{-9}$ og initial-L1 0.05. Punktvise EM-standardfejl er ikke simultane konfidensbånd, og god agreement på én syntetisk case er ikke historisk modelvalidering.

Kilder: [G6b-review](G6B_DISTRIBUTION_PANEL_REVIEW.md), [BM-reference, afsnit 2–3](BM_REFERENCE.md). Implementation: [distribution panel](../src/method/bm_1_distribution_panel.py), [fælles BM-control](../src/control/bm_1.py).

## 8. G8: rekursivt Bayes-filter og parameterbank

Den syntetiske bank bruger $\rho\in\{0.125,0.1875,0.25,0.375,0.5\}$ med uniform prior, $C=1/\rho$, og samme øvrige parametre som G6b. Sand syntetisk $\rho$ er 0.25. Der er 22 punktobservationer på 24 timer, med manglende målinger ved 6 og 17 timer og måle-støj $\sigma_y=0.03$. Dette demonstrerer filtrering, ikke sikker recovery af den sande parameter.

Den fælles posterior er

$$P(X\in dx,\rho=\rho_j)=w_jp_j(x)\,dx.$$

Hver $p_j$ fremskrives under sin egen $\rho_j$. Ved ny punktmåling er

$$L_j(y\mid x)=\frac1{\sqrt{2\pi}\sigma_y}
\exp\left[-\frac{(y-D_j(x,t))^2}{2\sigma_y^2}\right],$$

$$Z_j=\int L_jp_j^-\,dx,\qquad
p_j^{\mathrm{raw},+}=\frac{L_jp_j^-}{Z_j},\qquad
w_j^+=\frac{w_j^-Z_j}{\sum_kw_k^-Z_k}.$$

På cellegrid bruges GL3 til cellemiddel-likelihood og $Z_j=\sum_iL_{ji}m_{ji}^-$. Logsumexp stabiliserer nodevægtene. **Evidence beregnes fra den rå prediction før stateprojektion.** En efterfølgende mixtureprojektion må ikke ændre den evidence, der allerede indgår i Bayes-vægtene. Ved manglende måling propagéres state, men likelihood-opdateringen springes over.

Bankens marginal er $p=\sum_jw_jp_j$. En gennemsnitsparameter med en enkelt state-law mister generelt parameter/state-afhængigheden. For demand gælder

$$\mu=\sum_jw_j\mu_j,\qquad
\operatorname{Var}(D)=\sum_jw_jv_j+
\sum_jw_j(\mu_j-\mu)^2.$$

Det sidste led er variation mellem parameterhypoteser, ikke en frivillig ekstra straf.

Den første FV257-bank gav predictive/posterior demand-RMSE 0.0212822/0.0164031 mod åben FV-reference 0.0286866. M38 og M78 blev derimod afvist: bankens største posterior-L1 var 0.500165 og 0.138551 mod 0.05. En god marginal mean var ikke nok til at godkende den fulde law.

Kilde: [punktbankreview](G8_POINT_BANK_REVIEW.md). Implementation: [filter og bank](../src/method/bm_1_g8.py).

## 9. Bernstein/Beta-basis: udledning og M254-valget

Vejlederens oprindelige normaliserede familie er

$$\ell_{m,M}(x)=(M+1){M\choose m}x^m(1-x)^{M-m}
=\operatorname{Beta}(x;m+1,M-m+1).$$

Hvert led har integral 1. Derfor er $p_M=\sum_mc_m\ell_{m,M}$ en tæthed, når $c_m\geq0$ og $\sum_mc_m=1$. Sæt $u_m=(m+1)/(M+2)$, $\mu=\sum_mc_mu_m$ og $S=\sum_mc_m(u_m-\mu)^2$. Beta-momenter giver

$$\operatorname{Var}(X\mid m)=\frac{u_m(1-u_m)}{M+3}.$$

Da $E_c[u(1-u)]=\mu(1-\mu)-S$, giver totalvarians

$$\operatorname{Var}(X)=\frac{\mu(1-\mu)}{M+3}
+\frac{M+2}{M+3}S\geq\frac{\mu(1-\mu)}{M+3}.$$

Positive blandinger af brede komponenter kan altså ikke skabe vilkårligt snævre posteriorer. For præcis kontinuerlig matching af middel $\mu$ og varians $v$ er

$$M\geq\left\lceil\frac{\mu(1-\mu)}v-3\right\rceil$$

en nødvendig betingelse. Den er ikke tilstrækkelig til en given L1-tolerance og er ikke et bevis for nødvendigheden af præcis M254.

Den snævre gemte FV513-posterior havde varians 0.00212276596 og mean 0.511328337; momentbetingelsen krævede mindst M115. M254 blev valgt som én udviklingskandidat med $M+2=256$: det centrale variansgulv $0.25/257=0.000972763$ er under halvdelen af denne varians. Valget giver 255 af **samme oprindelige basisfunktioner**; det er ikke en ny model eller et bevist mindste nødvendigt M.

### Eksakt prior ved gradforhøjelse

Beta(20,20) er $\ell_{19,38}$. For grad $n$ gælder

$$\ell_{m,n}=\frac{n+1-m}{n+2}\ell_{m,n+1}
+\frac{m+1}{n+2}\ell_{m+1,n+1}.$$

De nye koefficienter er derfor

$$c_j^{(n+1)}=\frac{n+1-j}{n+2}c_j^{(n)}
+\frac{j}{n+2}c_{j-1}^{(n)},$$

med nul uden for de gamle indeks. Positivitet, vægtsum og den kontinuerte tæthed bevares. Gentagelse fra 38 til 254 giver den gamle prior i nye koordinater uden et nyt priorfit. Den kontrollerede prior-L1 var cirka $3.18\cdot10^{-14}$.

### Celleintegration, LS og solvercertifikat

M254 bruger celleintegrerede komponenter

$$B_{mi}=\frac1{\Delta x}\int_{V_i}\ell_{m,254}(x)dx,
\qquad m_i=\Delta x\sum_mc_mB_{mi}.$$

Beta-CDF og komplementær CDF anvendes i halerne. Legacy-basis med evaluerede/renormaliserede midtpunkter har en særskilt identitet; dens koefficienter kan ikke automatisk læses som de nye M254-koefficienter.

På uniformt grid er density-LS med $A=\sqrt{\Delta x}B^T$ og mass-LS med celleintegralmatrixen ens op til en positiv skalar. Den konkrete native skalering bevares i hver kodevej. For

$$F(c)=\tfrac12\|Ac-q\|^2,\qquad g=\nabla F=A^T(Ac-q)$$

er simplex-gap'et

$$G=c^Tg-\min_mg_m\geq0.$$

Konveksitet giver $F(c)-F(c^*)\leq G$: minimeringen af $g^Tz$ over simplekset vælger den mindste gradientkomponent. Gap'et kontrollerer optimizerfejl i dette LS-problem, **ikke** basisadequacy eller akkumuleret filterfejl. Derfor kontrolleres også L1, momenter, CDF, kvantiler, evidence, vægte og masse.

M254 er stadig svært konditioneret: det statiske review rapporterede designmatrix-kondition cirka $1.30\cdot10^{15}$ på 513 celler og $4.18\cdot10^{17}$ på 257. Individuelle koefficienter kan være følsomme, selv når den repræsenterede law er præcis. SVD og separate primal-/gapkontroller bevares; graden løste ikke konditioneringsproblemet.

### Alternativ reference og faktisk rekursiv accept

Den udviklede lokale reference har 41 Beta-komponenter med $\mu_r=(r+0.5)/41$, koncentration $\kappa=256$, parametre $\alpha_r=\kappa\mu_r$, $\beta_r=\kappa(1-\mu_r)$, plus Beta(20,20)-prioren. Hvert leds varians er $\mu_r(1-\mu_r)/(\kappa+1)$. Placeringer, koncentration og ekstra priorled er vores referencevalg. De er ikke vejlederens foreskrevne basis og erstatter ikke den valgte primære Bernstein-familie.

Efter statiske kontroller blev eget M254257-filter gennemført. Den særskilte 513-forfining gennemførte samme 24 timer fra egen prior, med egne 22 Bayes-opdateringer. Største conditional-L1 mod FV513 var 0.00354065417; marginal posterior-L1 0.00108110760; nodevægtforskel 0.000265417496. Største substep-/Bayes-projektion-L1 var 0.000971943317. Alle deklarerede gates og faktisk JSON-resume time 22→24 bestod.

Dette er numerisk accept af den rekursive repræsentation på den ene syntetiske case. Det er ikke historisk intervalaccept, fysisk parameteridentifikation eller accept af en senere G7-optimizer.

Kilder: [Bernstein-udledning](G8_BERNSTEIN_REVIEW.md), [posterior-basis og reference](G8_POSTERIOR_BASIS_REVIEW.md), [M254257](G8_M254_FILTER_REVIEW.md), [M254513](G8_M254_REFINEMENT_REVIEW.md). Implementation: [typed M254](../src/method/bm_1_g8_bernstein.py), [cellebasis](../src/method/bm_1_g8_basis.py).

## 10. G7: deterministisk prisstyring

Fire holdte priser vælges på seks-timersblokke:

$$\min_{u\in[0,1]^4}J_{\mathrm{det}}(u),\qquad
J_{\mathrm{det}}=\int_0^{24}[D(x^u(t),u(t))-A(t)]^2dt.$$

Single shooting betyder, at hver prisplan integreres fra samme frosne startstate; en tidligere kandidats slutstate bliver aldrig den næstes start. Fysiske parametre og responskurver genfittes ikke til kontrolmålet.

Midpoint-cost og residualer er

$$J_h=\sum_qh_q(D_q-A_q)^2=r^Tr,\qquad r_q=\sqrt{h_q}(D_q-A_q).$$

Pris-/inputbrud splitter quadraturen. $\sqrt{h_q}$ er nødvendigt for de rigtige tidsvægte. TRF minimerer $F=J_h/2$; rapporteret J er fortsat $J_h$. Med residual-Jacobian $R$ er

$$\nabla F=R^Tr,\qquad
\nabla^2F=R^TR+\sum_qr_q\nabla^2r_q.$$

Gauss–Newton-leddet er positivt semidefinit, men den ekstra term og dynamikken gør ikke automatisk BM-opgaven konveks. To starter med næsten ens J er lokal numerisk evidens, ikke global optimalitet.

Den lille syntetiske deterministiske reference gav neutral J 0.12 og bedste fundne J cirka 0.0209268019845 med priser cirka $(0.686689795,0.797805159,0.503310238,0.392193631)$, svarende til 82.56 % lavere J i netop denne opgave. Det er tracking, ikke elregning, profit eller kausal effekt.

### Historisk matched kontrakt og bevaret forbrug

Et særskilt tidligere januar-origin-forsøg bruger frosne fits og fladt mål

$$A=\bar B=\frac1T\int_0^TB(t)dt=0.39900697422105735.$$

Baselineintegralet beregnes på dens egne lineære knuder. Valget bevarer målprofilens total, men **tvinger ikke kandidatens demand-total til at være den samme**. Derfor blev en særskilt lighed indført:

$$E(u)=\int_0^{24}[D(t;u)-B(t)]dt=0.$$

BM-balancen giver $E=C[x(24)-x(0)]$, så kontinuert energineutralitet svarer til uændret slut-SoC. Et kvadratisk strafled $\lambda E^2$ ville ændre tradeoff og kræve en ny vilkårlig vægt; ligheden udtrykker det bestilte krav direkte. SLSQP blev brugt til BM's fire variable og ene ikke-lineære lighed. De bedste gennemførte feasible kandidater er lokale resultater.

For den historiske WM-reference er time-dynamikken $\dot F=R/24$, $I=B+R$, så $E=24[F(24)-F(0)]$. Original Euler bruger venstre R, mens cost bruger midpoint-R. På uniformt timegrid h teleskoperer det til

$$E_{h,\mathrm{WM}}=24(F_N-F_0)+\frac h2(R_N-R_0).$$

Derfor håndhæves samme diskrete $E_h$ som i rapporteret cost; $F_N=F_0$ alene er ikke den samme diskrete betingelse. Der kræves ikke også $R_N=R_0$.

Frosset WM er affint i u: residual $r=c+Hu$, $E=e_0+a^Tu$. Det giver en konveks QP med $Q=2H^TH$. Alle $3^4=81$ frie/nedre/øvre faces blev undersøgt algebraisk; KKT-kontroller og redundans af pathbounds gav en global løsning af **den diskrete WM-opgave**. Dette argument kan ikke overføres til nonlinear BM eller den kontinuerte limit uden yderligere kontrol.

| Historisk frosset fit | J uden lighed | J med bevaret forbrug |
|---|---:|---:|
| Lineær BM | 0.268818599455 | 0.275408887552 |
| G4 BM | 0.273316332715 | 0.276532279602 |
| WM4 | 0.299704871421 | 0.303629020154 |

BM's fine neutrale energirester var cirka $-4.60\cdot10^{-10}$ og $-3.88\cdot10^{-10}$; WM's $1.47\cdot10^{-6}$, under $10^{-5}$-gaten. De nye constraints var dyrere i J, som forventeligt ved indsnævring af feasible muligheder. Dette er bevaret historisk kontrol, ikke den aktuelle endelige BM/WM-sammenligning.

Kilder: [deterministisk kontrol](G7_DETERMINISTIC_CONTROL_REVIEW.md), [matched kontrakt](G7_MATCHED_CONTROL_REVIEW.md), [energineutral udledning og KKT](G7_ENERGY_NEUTRAL_REVIEW.md). Implementation: [deterministisk evaluator](../src/method/bm_1_g7_deterministic.py), [energy](../src/method/bm_1_g7_energy.py).

## 11. G7: stokastisk mål, korte resultater og feedback

For egne betingede posteriorer og faste cutoff-vægte bruges

$$J(u)=\int\sum_jw_j\int_0^1[D_j(t,x,u)-A(t)]^2p_j(t,x;u)dx\,dt.$$

Udvid kvadratet med $\mu_j=E_j[D]$, $v_j=\operatorname{Var}_j(D)$, $\mu=\sum_jw_j\mu_j$:

$$J=\int(\mu-A)^2dt+
\int\sum_jw_jv_jdt+
\int\sum_jw_j(\mu_j-\mu)^2dt.$$

Krydsleddet forsvinder, fordi $\sum_jw_j(\mu_j-\mu)=0$. Middelfejl, within- og between-varians er derfor en eksakt dekomposition af ét objective, ikke tre valgte straffe. En separat deterministisk ODE ved middelstate/MAP-node er generelt ikke bankens mean; den fundne deterministiske pris skal også scores under det fulde stokastiske J ved en fair kontrolsammenligning.

### Kort tre-plan-forsøg, time 12→14

De tre på forhånd valgte planer er $u(\delta)=(0.5+\delta,0.5-\delta)$ med $\delta=0,-0.2,+0.2$. Lige middelpris betyder ikke bevaret forbrug. Coarse-grid valgte laveste komplette bank-J, med tie-rækkefølge neutral først. Fine/FV/ny partition validerede valget og blev ikke brugt til efterfølgende reselektion.

Neutral vandt: coarse J 0.003741963573462555 mod 0.017303375697677884 for $(0.3,0.7)$ og 0.010655853331904826 for $(0.7,0.3)$. Neutral fine M254-J var 0.0037155174638558177, FV-J 0.0037153098107086916 og den finere partition-J 0.003715523422208866. De tilsvarende forward-pargates bestod. Tilts blev ikke fine-valideret; resultatet er en coarse-rangering af tre planer, ikke et kontinuerligt optimum.

Ny partition halverede objective-step fra 1/96 til 1/192 time, med 192→384 forskellige midtpunkter. Den ændrede samtidig transportens output-/pausepunkter. Derfor er forsøget partitionfølsomhed, ikke et isoleret quadraturfejl- eller Richardsoncertifikat. Den fine partitions J blev rekonstrueret som

$$0.0022613178834201752+0.0013782185162290082
+0.00007598702255962573=0.003715523422208866.$$

De gamle per-node-projektionsmaxima for quadnode 0/1/2 var gået tabt før den aktuelle chatsletning. Komplette fordelingsbaner kan ikke i sig selv genskabe samtlige mellemprojektioners maxima. Derfor blev den samlede numerical/scientific accept ikke erklæret, selv om bank- og terminalpargates bestod.

### EM-sampling, uafhængighed og parret SE

Fra den repræsenterede M254-law trækkes først $I\sim\mathrm{Categorical}(c_j)$ og dernæst $X_{12}\mid I=i\sim\mathrm{Beta}(i+1,255-i)$. Dette sampler den kontinuerte mixture, ikke FV-cellemasser som koefficienter. Hver node har 256 baner. Initialdraws, Brownsk støj og observationsstøj har separate role/node-RNG-identiteter. Den konkrete tidligere streamgenbrug blev rettet; forskellige seeds er ikke et fysisk bevis for universel uafhængighed.

Med pathcost $Q_{jr}=\sum_qh_q(D_{jr,q}-A_q)^2$ er

$$\widehat J=\sum_jw_j\overline Q_j,\qquad
\widehat{\mathrm{SE}}(\widehat J)=
\sqrt{\sum_j\frac{w_j^2s_j^2}{256}}.$$

Formlen følger af uafhængige betingede strata; alle fem noder er simuleret med faste vægte. De 1280 paths er ikke 1280 iid draws fra marginalbanken. SE dækker MC-sampling betinget på bank og numerik, ikke EM-bias, FP-/LS-fejl, bankestimation, modelusikkerhed eller optimizerfejl.

Coarse Brownian increments er summen af to fine increments. Samme initialdraws og Brownian input bruges på tværs af policy, så forbedringen $d_{jr}=Q_{jr}(u_0)-Q_{jr}(u)$ har

$$\widehat\Delta=\sum_jw_j\overline d_j,\qquad
\widehat{\mathrm{SE}}(\widehat\Delta)=
\sqrt{\sum_j\frac{w_j^2s_{d,j}^2}{256}}.$$

Covariance er dermed inkluderet i forskellens varians. Inputhashes kontrolleres før parring. Kobling kan reducere sammenligningsstøj, men lover ikke altid mindre SE. I den korte seed-rettede prøve var EM fine J 0.0036682534675161536 og SE $8.870328259665031\cdot10^{-5}$, med nul rå randoverskridelser. Fine M254/EM-afvigelsen $4.7264\cdot10^{-5}$ bestod den praktiske gate $\max(0.001,4SE)$.

Den fælles syntetiske endpointmåling gav fire egne backend/grid-Bayes-opdateringer, med største conditional-L1 0.0145280 og nodevægtforskel 0.00200737. Dette var et kort forward-/EM-/endpoint-update-forsøg, ikke hele det efterfølgende feedbackstyringsdøgn.

### Implementeret døgnstyring og seneste faktiske status

Den nye driver søger fire priser på $[12,18),[18,24),[24,30),[30,36]$ med fast $A=0.5$ og den analytiske baseline. Den genbruger bounded Powell med to faste starts/callcaps; fulde kandidater og konvergensstatus bevares særskilt. Maxfev-stop er ikke konvergens, og bedste observerede kandidat er ikke et bevist optimum.

Designet har M254257/513 og FV257/513 fra **hver backends egen filtrerede origin**, neutral/deterministisk/stokastisk policyreview, partitionkontrol, parret EM og samme måling ved 36. Rå evidence opdaterer egne laws/vægte; de videreføres til 38. Det er posteriorbaseret open-loop-styring efterfulgt af et feedbacktrin, ikke dual control eller en Bellman-PDE. Der er ikke tilføjet stokastisk energilighed, terminal- eller smoothingstraf.

Den fulde **stubgraf** blev gennemført; et separat saved-fixture-review autentificerede 132 receipts og fire gemte Bayes-opdateringer uden compute. Det kontrollerer orchestration og algebra. Stubtransport, stub-likelihood og stubprojektion er ikke et numerisk døgnresultat. Den historisk fejlede originalinvocation bevares som fejlet.

Én ægte standalone deterministisk 24h-neutral måling blev kørt: 2.3313609979995817 s, J 0.12120382930681177, MAP-node 3 og egen middelstate ved 12 på 0.5062519458461637. Den er én-origin-timing, ikke et kompatibelt fire-origin-ratepacket eller en fuld prissøgning.

Kilder: [seed-rettelse og SE](G7_SEED_REPAIR_REVIEW.md), [quad-recovery](G7_COMPLETE_QUAD_RECOVERY_REVIEW.md), [feedback](G7_FEEDBACK_INTEGRATION_REVIEW.md), [døgndriver](G7_FULL_DAY_DRIVER_REVIEW.md), [seneste fixturekontrol](G7_DAY_COMPLETION_REVIEW.md). Implementation: [døgndriver](g7/day_control.py), [råreview](g7/day_review.py), [auditeret feedback](g7/audited_feedback.py).

## 12. Hvorfor ejerskab, versionsidentitet og budget er fagligt relevante

Et genbrugt forward er kun samme eksperiment, når model, observation/cutoff, egen prior, node, policy, tids-/cellegrid, basis, numerik og sourceidentitet matcher. Et cache-hit med forkert state eller grid kan ændre J, evidence og kandidatvalg. SHA256, receipts og fulde kontrakter beskytter dermed den matematiske sammenligning; de beviser ikke, at modellen er fysisk sand.

M254 og FV skal beholde hver deres egen law. En state-law kan overføres konservativt til sammenligning, men må ikke derefter kaldes den anden backends egen filterposterior. Den gamle auxiliary-FV-forward fra en M254-origin er en dokumenteret reference; den er ikke en uafhængigt FV-filtreret origin til det nye døgnforsøg.

Den planlagte beregningsramme er

$$B_{\mathrm{plan}}=B_{\mathrm{reserve}}+1.25\sum_kn_kr_k,$$

hvor $r_k$ skal være kompatible målte whole-job-rater. En ukendt rate forbliver ukendt, og budgetestimat/admission forbliver null/false. Faktor 1.25 er en planlægningsreserve, ikke et matematisk worst-case-bevis. Korte 2h-rater og stubtider giver ikke automatisk en 24h-rate. Parent-/node-deadlines og varig registrering er separate værn.

Ved seneste afsluttede G-review var den bevarede debit 1927.3372718589885 s, registreret loft 1930 s og rest 2.6627281410114847 s. Alle 16 kompatible døgnrater var stadig ukendte. Denne dokumentindsamling genkører ingen numeriske kerner og giver ingen nye modelberegningstider eller nulstilling af tidligere charges.

Kilder: [day completion](G7_DAY_COMPLETION_REVIEW.md), [day check/rate](G7_DAY_CHECK_AND_RATE_REVIEW.md), [recovery](G7_CHAT_RECOVERY_20261008.md), [budgetintegration](g7/integrated_budget.py).

## 13. Oprindelse: vejleder, litteratur og vores valg

Dette er attributionen i de bevarede reviews. Samlingen foretager ikke en ny uafhængig litteraturvalidering eller hævder at have genlæst alle bog-/PDF-sider.

| Matematik/metode | Dokumenteret kildegrundlag | Egen udledning eller konkret valg |
|---|---|---|
| BM-dynamik og $f/g$ | Vejlederens tavler, FP2025 afsnit 3.2.3, eksisterende BM1 | Datakontrakter, inputrepresentation, bounds og konkrete casevalg |
| Normaliseret Bernstein/Beta-familie | Vejlederens upload07, aktuelt `wSnLZuio(1).jpg` | Variansgulvets samlede udledning, M254-kandidat, eksakt gradforhøjet prior og typed cellebasis |
| Mixture og positiv LS | Ritschel Lecture08 slide 6 eq. 4 og slide 8 eq. 7a–7c; mixture/FP-materialet | Faktisk FV-baseret diskret transport/projektion, active-face-SVD og numeriske gates |
| Kovariansligning/linear noise | Lecture08 slide 10 eq. 20; Alba Gonzalo Primo afsnit 6.1.2 eq. 6.4/6.7 | Five-weekday-spredningskalibrering, historisk $P_0=0$, scopes og comparator |
| Itô og Euler–Maruyama | Higham 2001, herunder afsnit 8 eq. 8.3, samt bevaret BM-reference | Brownian coupling, RNG-roller, grids, paths og registrerede praktiske tolerancer |
| LS/Gauss–Newton, QP/KKT, SQP | Nocedal–Wright kapitler 10, 16 og 18; dokumenteret SciPy-brug i originalreviews | TRF/SLSQP/Powell-setup, finite-difference-steps, starts og caps |
| Balance, totalvarians, Bayes og stratificeret SE | Direkte udledninger vist i G/BM-reviews | Anvendelsen på egne nodeposteriorer, policyvalg og kontrolarkitektur |
| Lokal 42-leds Beta-reference | Samme almindelige Beta-identiteter | 41 placeringer, koncentration 256 og priorled; ikke vejlederens foreskrevne primærfamilie |
| Fladt target og energilighed | Direkte balanceargument, historisk WM-noter | Benchmarkmålet og det særskilt bestilte bevaret-forbrug-krav |

De tilgængelige uploadfiler er medtaget uændret i pakken under `reference-material/`, med SHA256 og eventuelle identiske dubletnavne i manifestet. En inkluderet PDF er kildemateriale, ikke dokumentation for et gennemført ekstra forsøg.

## 14. Hvad mangler, og hvad samlingen gør muligt

1. Det fulde numeriske G7-døgnforsøg er åbent. Implementeret driver og bestået stubkontrol må ikke omtales som gennemført science-run.
2. To egne hour12-M254-origins er byteverificeret genfundet. De tilsvarende **egne FV-filtrerede origins** er ikke fundet i det tilgængelige repositoryarkiv. Library-arkivet `G8_M254_513_forfining_raaresultater_20261006.zip` er navngivet i de gamle noter, men dets indhold er ikke hentet i denne recovery. Manglende adgang er ikke bevis for sletning.
3. Tre ældre quad-projektionsmaxima mangler fortsat; det hul opstod før den slettede chat. Det bevares eksplicit frem for at blive udfyldt med antagelser.
4. De originale noter henviser til flere råarkiver, som ikke alle er tilgængelige her. Pakken indeholder alle fundne G-noter, relevant repositorykode/-evidens, de tilgængelige uploadkilder og det faktisk gendannede arkiv; den er ikke et løfte om, at alle historiske råarrays er genfundet.
5. Historisk måleoperator, råenhed, fast population, fysisk/kausal identifikation og en uafhængig sluttest er ikke afsluttet gennem denne dokumentindsamling.

Samlingen kan bruges til at genoptage arbejdet fra de faktiske kontrakter, skrive rapportens metode-/resultatafsnit og efterprøve argumentationen. De daterede originalreviews og filmanifestet er autoritative for de enkelte kørslers præcise setup, resultater og begrænsninger.
