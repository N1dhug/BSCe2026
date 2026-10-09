# G: dynamik, punktprognoser og det historiske S3-valg

<a id="s3-transition-history"></a>
<a id="legacy-development"></a>
<a id="g-s3-standalone-guide-20261008"></a>
<a id="g-next-step-20261008"></a>
<a id="g-mathematics-recovery-20261008"></a>
<a id="g-completion-plan-8h-20261008"></a>
<a id="g-code-math-review-20261008"></a>
<a id="g4-weekday-development-review"></a>
<a id="g4-validation-weeks-13-18-review"></a>
<a id="g4-strategies-12-weeks-review"></a>
<a id="g4-january-pilot-review"></a>
Dette er en samlet faglig fremstilling af bevaret udviklingsevidens. Tidligere arbejdsordrer og budgetter er historik, ikke nye kørselsordrer. Aktuel brug, accept og opgaver står i [BM-guiden](../../../docs/bm/README.md) og [status](../../../docs/bm/STATUS.txt). Originaler, hashes og genskabelse: [arkivindeks](../INDEX.md).

Historiske statusord som “åben”, “endnu ikke” og “Remaining scope” gælder den pågældende delundersøgelse 5.–8. oktober 2026. De beskriver ikke automatisk den aktuelle F4-kommando eller senere G-timeleverance.

Denne udviklingshistorik bruger den oprindelige punktoperator. Den senere
timeintervalmodel, residualer og weekday-bånd har egne gældende fremstillinger:
[interval](../../../docs/bm/G_INTERVAL_MODEL.md),
[usikkerhed](../../../docs/bm/G_UNCERTAINTY.md) og
[weekday-studie](../../../docs/bm/G_WEEKDAY_STUDY.md).

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

Kilder: [G-kontrakten](forecast-development.md#legacy-development), [januargrundlaget](forecast-development.md#legacy-development), [matched kontrol](control-and-evidence.md#g7-matched-control-review), [BM-reference](../../../root/BM_REFERENCE.md).

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

Kilder: [BM-reference, afsnit 1](../../../root/BM_REFERENCE.md), [G, afsnit 2–3](forecast-development.md#legacy-development), [januarnumerik](forecast-development.md#legacy-development). Implementation: [BM-kernen](../../../src/model/bm_1.py), [deterministisk benchmark](../../../src/method/bm_1_deterministic_benchmark.py).

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

Kilder: [januar og snapshot](forecast-development.md#legacy-development), [overdragelse P0–P2](forecast-development.md#legacy-development). Implementation: [fit](../../../src/method/bm_1_deterministic_fit.py), [snapshot](../../../src/tools/bm_1_deterministic_snapshot.py), [data](../../../src/tools/bm_1_deterministic_data.py).

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

Kilder: [januarpilot](forecast-development.md#g4-january-pilot-review), [weekday](forecast-development.md#g4-weekday-development-review), [strategier](forecast-development.md#g4-strategies-12-weeks-review), [uger 13–18](forecast-development.md#g4-validation-weeks-13-18-review), [prisfølsomhed](control-and-evidence.md#g7-matched-control-review). Implementation: [responsfit](../../../src/method/bm_1_deterministic_response.py), [strategier](../../../src/method/bm_1_deterministic_strategy.py).


## Fra punkt-S3 til den senere intervalleverance
Den gemte standalone-punktmilepæl havde18.400s fit,168prognosetimer,
RMSE.377435 mod baseline.378180 og bestået demandforfining. Den er et
eksponeret udviklingseksempel, ingen uafhængig sluttest. Den senere time-
operator bruger sit eget fit/snapshot; et gammelt punktfit ommærkes aldrig.
Gamle planer om otte timers arbejde og nye parameterbanker er afløste
arbejdsordrer. Aktuel kalender/snapshotbrug står i BM-guiden; nye kampagner
kræver en særskilt konkret afgrænsning.

For SDE/FP se [fordelinger](stochastic-distributions.md), for Bayes/LS/basis se [posterior](posterior-and-basis.md), og for kontrol og negative forsøgsfund se [styring](control-and-evidence.md).

## Konkrete kode- og kontrolindgange

[fit](../../../src/method/bm_1_deterministic_fit.py), [fit-tests](../../../tests/test_bm1_deterministic_fit.py). Historiske CLI- og controllerkommandoer findes kun i det restaurerede originaltræ; disse links peger på de bevarede matematiske komponenter. Tests er links til tidligere eller løbende kontroller, ikke påstået genkørt ved denne dokumentændring.
