# G: Bayes-bank, Bernstein/Beta og posteriorens opløsning

<a id="g8-posterior-basis-review"></a>
<a id="g8-point-bank-review"></a>
<a id="g8-m254-refinement-review"></a>
<a id="g8-m254-filter-review"></a>
<a id="g8-bernstein-review"></a>
Dette er en samlet faglig fremstilling af bevaret udviklingsevidens. Tidligere arbejdsordrer og budgetter er historik, ikke nye kørselsordrer. Aktuel brug, accept og opgaver står i [BM-guiden](../../../docs/bm/README.md) og [status](../../../docs/bm/STATUS.txt). Originaler, hashes og genskabelse: [arkivindeks](../INDEX.md).

Historiske statusord som “åben”, “endnu ikke” og “Remaining scope” gælder den pågældende delundersøgelse 5.–8. oktober 2026. De beskriver ikke automatisk den aktuelle F4-kommando eller senere G-timeleverance.

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

Kilde: [punktbankreview](posterior-and-basis.md#g8-point-bank-review). Implementation: [filter og bank](../../../src/method/bm_1_g8.py).

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

Kilder: [Bernstein-udledning](posterior-and-basis.md#g8-bernstein-review), [posterior-basis og reference](posterior-and-basis.md#g8-posterior-basis-review), [M254257](posterior-and-basis.md#g8-m254-filter-review), [M254513](posterior-and-basis.md#g8-m254-refinement-review). Implementation: [typed M254](../../../src/method/bm_1_g8_basis.py), [cellebasis](../../../src/method/bm_1_g8_basis.py).


## Udledning af den særskilte lokale Beta-reference
Den følgende generalisering er diagnostisk reference, ikke en erstatning af
vejlederens primære Bernstein-familie. Tabellen og de negative M38/M78-fund
ovenfor fastlægger hvorfor ekstra opløsning blev undersøgt.
### Den nye reference udledes fra samme Beta-familie

Dette afsnit beskriver en referenceudvidelse, ikke en erstatning af tavlens
foreskrevne basis. Lad beta-normaliseringsfunktionen være
\[
\mathrm B(\alpha,\beta)=\int_0^1x^{\alpha-1}(1-x)^{\beta-1}\,dx
=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}.
\]
For alpha,beta>0 er
\[
\ell_{\alpha,\beta}(x)=
\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
x^{\alpha-1}(1-x)^{\beta-1}.
\]
Den er ikke-negativ og integrerer til 1 ved definitionen af B.
Gammafunktionen har Gamma(n)=(n-1)! for positive heltal.
Indsættelse af alpha=m+1,beta=M-m+1 giver præcis vejlederens formel.
Den nye reference ændrer altså valget af Beta-parametre, ikke
normaliseringsprincippet eller sandsynlighedskravene.

Beta-integralerne giver momentformlerne direkte:
\[
E[X^q]=\frac{\mathrm B(\alpha+q,\beta)}
{\mathrm B(\alpha,\beta)},\quad
E[X]=\frac{\alpha}{\alpha+\beta},\quad
E[X^2]=\frac{\alpha(\alpha+1)}
{(\alpha+\beta)(\alpha+\beta+1)}.
\]
Her bruges q=1,2; generelt kræves alpha+q>0.
Sæt kappa=alpha+beta og mu=alpha/kappa. Ved at trække E[X]^2
fra E[X^2] fås
\[
\operatorname{Var}(X)=\frac{\mu(1-\mu)}{\kappa+1}.
\]
Dermed kan placering mu og koncentration kappa vælges særskilt:
alpha=kappa*mu,beta=kappa*(1-mu). Det er hele den matematiske
idé bag referenceudvidelsen.

For referencekomponenterne bruges r=0,...,40,
mu_r=(r+1/2)/41 og kappa=256:
\[
\widetilde\ell_r(x)=
\frac{\Gamma(256)}
{\Gamma(256\mu_r)\Gamma(256(1-\mu_r))}
x^{256\mu_r-1}(1-x)^{256(1-\mu_r)-1}.
\]
Parametrene er generelt ikke heltal, så Gamma erstatter fakulteter.
Ved r=20 fås mu=.5 og den konkrete funktion
255!/(127!127!) * x^127(1-x)^127.

Den fulde reference-mixture er
\[
\widetilde p(x)=c_{\rm prior}\ell_{20,20}(x)
+\sum_{r=0}^{40}\widetilde c_r\widetilde\ell_r(x),
\quad c_{\rm prior},\widetilde c_r\geq0,\quad
c_{\rm prior}+\sum_r\widetilde c_r=1.
\]
Her betegner ell_(20,20) Beta-formparametrene, ikke et Bernstein-gradtal.
Beta(20,20) tilføjes for at repræsentere den oprindelige initiallov direkte:
initialt c_prior=1 og de øvrige vægte 0. I kodens 42-leds-array er prioren
komponent0, mens referencekomponent r ligger på indeks r+1.
De 42 repræsentationsvægte er forskellige fra parameterbankens fem rho-vægte.


## Kilder, numeriske valg og åben grænse
Vejlederens upload07 (`wSnLZuio(1).jpg`) er direkte kilde til normaliseret
Bernstein på[0,1]. Ritschel Lecture08 slide6 eq4/slide8 eq7a–7c understøtter
mixture og positiv LS. Celleintegreret basis, FV-transport, active-face-SVD,
41placeringer/koncentration256 og de konkrete gates er vores valg.
Tavlens FP-faktor1/2 og standard-Brownsk konvention bevares; slides' notation
uden eksplicit1/2 ændrer ikke modellen. Ortogonalitetsforenklingen fra slide9
gælder ikke disse overlappende baser.

G8's faktiske JSON-resume time22→24 bestod kun for den registrerede case.
En FV-forward fra M254-input er ingen FV-filtreret posterior. To egne
hour12-M254-origins er genfundet byte-identisk; de tilsvarende egne
FV-filterorigins var ikke fundet i repositoryets tilgængelige råarkiv.
Manglende adgang til et navngivet Library-arkiv er ikke bevis for sletning.

## LS-solverfejl versus repræsentationsfejl


Konveksitet giver \(f(c)-f(c^*)\leq\mathcal G\), fordi den mindste lineære
værdi over simplex er den mindste gradientkomponent. Optimalitet af c* og
den kvadratiske identitet giver desuden

\[
\|A(c-c^*)\|_2\leq\sqrt{2\mathcal G}.
\]

På enhedsintervallet er tæthedens L1-afvigelse fra det optimale basisfit
højst den tilsvarende vægtede L2-afvigelse. Det afgrænser solverfejl,
**ikke** repræsentationsfejl over for måltætheden eller ophobet filterfejl.
Beviset gælder eksakt feasible aritmetik; flydende massetjek/gap er en
praktisk numerisk kontrol og ikke et intervalaritmetisk certifikat.

## Genindlæsning af en verificeret basis

Genberegning af samme deklarerede cellebasis er her ikke bitvis identisk
med de bevarede arrays, selv om kildehash og versionsnavne er verificeret.
Fine:77 elementer ændret, max abs3.019806627e-14. Coarse:34 elementer,
max abs7.105427358e-15. Den konkrete system-/specialfunktionsårsag er ikke
isoleret; versionsnavne alene beviser ikke bitvis numerisk reproduktion.

Vi udskifter IKKE de oprindelige basisværdier med nye værdier.
Standard from_record uden certificate kræver fortsat np.array_equal.
Den eksplicitte from_verified_bytes(payload,expected_sha256) gør:
1. Verificér det eksternt registrerede SHA256 for HELE originalfilen.
2. Bevar originalt basisarray og grid; beregn deres eksakte float64-bytehash.
3. Verificér supervisor M254 og en særskilt roundoff-only familiegrænse.
4. Bevar de eksisterende krav om realitet, positivitet, masse og c→mass-paritet.
Certificate følger med typed forecast-/feedback-origins og checkpointkontrakten.
Default origins/legacy-recordformat ændres ikke; certificate er et eksplicit
optional metadatafelt. Ingen fysisk parameter, projektion eller tolerance
for modelaccept er lempet.

For basisarrayets cellemiddel b_ik og lokal generator \hat b_ik er grænsen
\[
 \eta=\max_i\sum_k\Delta x_k|b_{ik}-\hat b_{ik}|
 \leq64\epsilon_{\mathrm{float64}}=1.4210854715\cdot10^{-14}.
\]
Den målte eta er1.1426856862e-16 fine og5.5295154534e-17 coarse.
Dette er vores eget portability-kriterium, ikke en værdi fra vejlederen.
Det bruger integreret komponentmasse frem for en løs relativ cellegrænse
i tails. For simplexkoefficienter er
\[
 \left\|\sum_i c_ib_i-\sum_i c_i\hat b_i\right\|_{1,h}
 \leq\sum_ic_i\|b_i-\hat b_i\|_{1,h}\leq\eta,
 \qquad c_i\geq0,\quad\sum_ic_i=1.
\]
Denne ulighed begrunder familiecheckets skala. Den er ikke en ny fejlgaranti
for transporten, og hverken ny generatorbasis eller nye c anvendes i runnet.
Bytehashen håndhæver den faktisk ejede basis; store familieforskelle
afvises også med et nyt whole-record-hash. Loadercheck verificerer begge
afvisninger, default strict guard og roundtrip med identiske basis/c.

M254 er fortsat vejlederens normaliserede Bernstein/Beta-familie
\[
 b_{i,m}(x)=(m+1){m\choose i}x^i(1-x)^{m-i},\quad m=254.
\]
Cellerepræsentationen, oprindelige legacy-veje og LS/FP-kernel bevares.
Det nye indlæsningsinterface og toleranceforsvar ovenfor er egne softwarevalg.


## Konkrete kode- og kontrolindgange

[basis og Bernstein](../../../src/method/bm_1_g8_basis.py), [bankfilter](../../../src/method/bm_1_g8.py), [basis-tests](../../../tests/test_bm1_g8_bernstein.py), [reload-tests](../../../tests/test_bm1_g8_owned_reload.py). Historiske CLI- og controllerkommandoer findes kun i det restaurerede originaltræ; disse links peger på de bevarede matematiske komponenter. Tests er links til tidligere eller løbende kontroller, ikke påstået genkørt ved denne dokumentændring.
