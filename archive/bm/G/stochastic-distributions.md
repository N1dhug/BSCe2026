# G: processtøj, FV, mixture og negative båndresultater

<a id="g6b-distribution-panel-review"></a>
<a id="g6a-linear-noise-review"></a>
Dette er en samlet faglig fremstilling af bevaret udviklingsevidens. Tidligere arbejdsordrer og budgetter er historik, ikke nye kørselsordrer. Aktuel brug, accept og opgaver står i [BM-guiden](../../../root/BM_GUIDE.md) og [status](../../../docs/bm/STATUS.txt). Originaler, hashes og genskabelse: [arkivindeks](../INDEX.md).

Historiske statusord som “åben”, “endnu ikke” og “Remaining scope” gælder den pågældende delundersøgelse 5.–8. oktober 2026. De beskriver ikke automatisk senere G-timeleverancer.

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

Kilde: [hele G6a-reviewet](stochastic-distributions.md#g6a-linear-noise-review). Implementation: [linear noise](../../../src/method/bm_1_linear_noise.py).

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

Kilder: [G6b-review](stochastic-distributions.md#g6b-distribution-panel-review), [BM-reference, afsnit 2–3](https://github.com/N1dhug/BSCe2026/blob/archive/f-main-20261009/root/BM_REFERENCE.md). Implementation: [distribution panel](../../../src/method/bm_1_distribution_panel.py), [fælles BM-control](../../../src/control/bm_1.py).


## Tolerancer og fortolkning
### Tolerancer fastlagt før resultaterne

Maksimale hourly forskelle: state-/demandmiddel 0.005,
state-/demandvarians 0.002, CDF 0.03 og kvantiler 0.02 i normaliserede enheder.
Trackingtab må afvige 2 %. Initial cell-mass L1 må højst være 0.05.
Massefejl højst 1e-9; materiel negativ cellemassegrænse 1e-12.
Disse er praktiske udviklingstolerancer, ikke garantier for høj relativ
varianspræcision eller den ukendte kontinuumsløsning.

CDF-punkter er x=0.25/0.5/0.75 og D=0.4/0.5/0.6; kvantiler 5/50/95 %.
EM screens tilføjer tre punktvise standardfejl til moment-/CDF-/objective-
tolerancerne. Varians-SE bruger den empiriske indflydelsesvariabel
(X−mean)^2; kvantil-SE estimeres ikke, så kvantiler bruger den faste tolerance.
Tre SE er ikke en simultan confidence-claim over alle tidspunkter.


Procesvarians, en syntetisk laws numeriske reference og empirisk forecast-
residual er forskellige objekter. Det aktive G-residuallag beskrives derfor
separat i [G_UNCERTAINTY](../../../docs/bm/G_UNCERTAINTY.md).
Ritschel/Alba-kilderne giver kovarians- og mixturegrundlag; anvendelsen af
femweekday-spredning og denne cases kalibreringsmål var konkrete egne valg.
Ingen numerisk agreement giver fysisk identifikation af sigma_x.

## Konkrete kode- og kontrolindgange

[linear noise](../../../src/method/bm_1_linear_noise.py), [distribution panel](../../../src/method/bm_1_distribution_panel.py), [panel-tests](../../../tests/test_bm1_distribution_panel.py). Historiske CLI- og controllerkommandoer findes kun i det restaurerede originaltræ; disse links peger på de bevarede matematiske komponenter. Tests er links til tidligere eller løbende kontroller, ikke påstået genkørt ved denne dokumentændring.
