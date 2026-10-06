# G8 — posterior-basis: udledning, én kandidat og målrettet forfining

6. oktober 2026. G-parent: e77aaafdf2613fa2e846f458386145ce121ae33b.
Dette er en fortsættelse af G8, adskilt fra F4-H2 og WM.
G8's gamle M38/M78-resultater og alle negative gates bevares.

## Aktiv basisbeslutning — Johan, 6. oktober 2026

Vejlederens normaliserede Bernstein/Beta-basis er G's primære mixture-basis:

\[
\ell_m(x)=\frac{(M+1)!}{m!(M-m)!}x^m(1-x)^{M-m},
\qquad m=0,\ldots,M.
\]

Den nye faste 42-leds Beta-basis beholdes alene som sammenligningsreference.
FV er fortsat den numeriske reference; metodeprioritet og numerisk accept er
forskellige beslutninger. De gemte negative M38/M78-resultater og positive
42-leds-resultater bevares uændret. M38's åbne G6b-propagation består, men
M38/M78 er ikke accepteret til G8's posterior→G7-kæde.

**G8-review gennemført, 6. oktober:** Én statisk kandidat, M254 med 255 af
vejlederens oprindelige Bernstein/Beta-funktioner og eksplicit celleintegration,
består syv afgrænsede LS-kontroller på gemte love. Maksimal posterior-L1 er
1.27144e-6; invocation9.596 s. Gradforhøjelse bevarer Beta(20,20)-prioren
med L1-fejl3.18e-14. De eksisterende produktionskerner og legacy-defaults er
uændrede. Dårlig konditionering består; dette er repræsentationsaccept på
udvalgte eksponerede love, ikke accept af et rekursivt filter eller G7.

**Næste: Sol High** implementerer den navngivne celleintegrerede Bernstein-
variant, eksakt prior og checkpointkontrakt, derefter én afgrænset egen
fem-node-filterkørsel. Højst300 nye numeriske sekunder inklusive højst60 s
pilot og30 s eksportreserve; forfining/resume kun inden for samme budget.
Ingen automatisk gradtalssøgning. Se [fuld udledning, resultater og stopregler](G8_BERNSTEIN_REVIEW.md).
Ingen lokal kørsel kræves nu.

G7's primære mixture-handoff skal komme fra en kontrolleret kørsel med den
oprindelige basisfamilie og hver nodes egne koefficienter/posterior.
Det gemte bank_local257-origin må kun bruges som eksplicit 42-leds-reference,
ikke som primær origin eller som skjult restart for Bernstein-filteret.
Bevar modellen, likelihooden, simplex-constraints og accepttolerancer;
ingen større målefejl, signed koefficienter eller FV-fallback forklædt som
mixture for at få den primære metode til at bestå. Hvis den primære basis
ikke består inden for praktisk budget, dokumenteres begrænsningen.

Denne beslutning erstatter tidligere anbefalinger om 42-leds-basis som
G7's primære startkonfiguration. De tidligere resultater bevares uændret.
Det efterfølgende M254-review ovenfor tilføjer kun statisk evidens.

## Historisk udviklingsanbefaling — erstattet som metodevalg

Den tidligere anbefaling var at bruge den nye faste, celleintegrerede Beta-basis med 42 komponenter og
257 FV-celler som udgangspunkt for G7's afgrænsede syntetiske integration.
513-cellerskontrollen består både for bankens marginal og hver nodes egen
conditional law. Det oprindelige 129-cellers grid accepteres ikke til den
strengere conditional-node-kontrakt.

Den store Bayes-projektionsfejl er løst på den eksponerede G8-case. Basisvalget
er udviklet efter at have set denne case, så resultatet er numerisk
udviklingsevidens, ikke uafhængig historisk validering, generel basisgaranti
eller parameterrecovery. G7's prisoptimering er ikke kørt.

FV forbliver primær reference. Den nye mixture er fortsat FV-transport med
positiv LS-projektion ved hvert internt trin; det er ikke en gridfri metode.
I denne case er FV hurtigere. Mixture indgår som den krævede
fordelingsapproksimation, ikke som dokumenteret runtimeforbedring over FV.

## Vejledergrundlag og referenceudvidelse — dokumentationsregel

Johan har 6. oktober præciseret, at de uploadede tavlebilleder er materiale
fra vejlederen. De behandles som det oprindelige faglige grundlag. En
udvidelse skal have en konkret begrundelse, udledning, kildeangivelse,
sammenligning og tydelig forskel fra legacy-modellen. De oprindelige
formler, kodeveje og resultater bevares; en alternativ reference må ikke
overtage den primære metode stiltiende.

### Hvor kommer den oprindelige basis fra?

Tavlebilledet wSnLZuio(1).jpg (upload07, lokal kopi
07-wSnLZuio-1-.jpg) viser både mixture-summen og den normaliserede
Bernstein-basis på [0,1]. Det er den direkte vejlederkilde til vores
primære basis; Ritschels slides er et supplerende grundlag for mixture/LS.

Bernstein-polynomiet er
\[
b_{m,M}(x)=\binom{M}{m}x^m(1-x)^{M-m}.
\]
Dets integral er 1/(M+1), så en sandsynlighedskomponent er
\[
\ell_{m,M}(x)=(M+1)b_{m,M}(x)
=\frac{(M+1)!}{m!(M-m)!}x^m(1-x)^{M-m}.
\]
Tavlens normaliseringsfaktor er derfor nødvendig: de almindelige
Bernstein-polynomier er ikke hver især sandsynlighedstætheder.

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

### Hvorfor referenceudvidelsen bruges, og hvorfor netop disse tal?

Formålet er at undersøge, hvor meget den observerede posterior-fejl skyldes
basisfamiliens opløsning, når den fysiske model og målemodel holdes fast.
For vejlederens basis er kappa=M+2, og derfor kobles antal og bredde.
Variansgulvet udledes i næste afsnit. M38/M78's negative G8-resultater
er grunden til sammenligningen, ikke en grund til at slette originalmodellen.

Den gemte FV-nodevariance er mindst cirka .00216612. Det udviklingsvalgte
designmål for referencen var et centralt komponentgulv under halvdelen:
.25/(kappa+1)<.00108306. Kappa256 er den mindste potens af to, der
opfylder dette mål: .25/257=.000972763. Valget af en potens af to og
halveringsfaktoren er praktiske designregler, ikke et teorem eller en
opskrift fra vejlederen. De er valgt efter eksponering af udviklingscasen.

41 placeringer giver cirka samme komponentantal som M38 og spacing
1/41=.0243902, mindre end den centrale standardafvigelse
sqrt(.25/257)=.0311891. Komponenterne overlapper derfor centralt.
Dette er ikke et krav om overlap i halerne eller et bevis for optimal
dækning. Antallet41 er et enkelt fast referencevalg, ikke et estimeret optimum.
Også den nye familie har et variansgulv og begrænset opløsning.

Kappa256 kunne matches af vejlederens familie med M254, men denne ville
have255 komponenter i stedet for42 og en anden placering af deres means.
Det er kun en analytisk sammenligning; M254 er hverken valgt eller testet.
Vi holder derfor originalfamilien primær og undersøger dens gradtal og
konditionering separat, med den nye reference som diagnostisk sammenligning.

### Hvad er fra kilderne, og hvad er vores valg?

| Del | Oprindelse og status |
|---|---|
| Normaliseret Bernstein-basis på [0,1] | Vejlederens upload07; primær legacy-familie. |
| SDE, Euler–Maruyama og FP med faktor1/2 | Upload05/07 viser disse relationer; behold den deklarerede standard-Brownsk-konvention. |
| Mixture-sum og positiv LS med sum1 | Mixture-sum: upload07 og Ritschel upload13 slide6 eq4. LS/constraints: upload13 slide8 eq7a–7c. |
| Beta-normalisering og momentudledning ovenfor | Udledning af de almindelige Beta-identiteter; forbindelsen til tavlens formel er vist eksplicit. |
| Uafhængige mu/kappa, 41 placeringer, kappa256, ekstra priorled | Vores begrundede referencevalg; ikke tilskrevet tavlen/slides som foreskrevet model. |
| Celleintegreret basis, FV-transport og efterfølgende LS-projektion | Den faktiske genbrugte numeriske implementering; ikke påstået identisk med slides' eksplicitte residualtidsstep. |
| Referenceaccept på den ene syntetiske case | Gemte udviklingsresultater; ingen generel garanti, historisk forecastaccept eller metodeprioritet. |

At slides skriver en generel ell_m giver plads til en sammenligningsreference,
men ophæver ikke Johans beslutning om tavlens konkrete basis som hovedmetode.
Ved en mulig uoverensstemmelse mellem kilder dokumenteres konventionen:
tavlen viser FP-faktoren1/2, mens upload13's viste SDE/FP-notation ikke
har denne faktor eksplicit. Vi ændrer ikke BM's Brownsk-konvention stiltiende.
Slide9's ortogonalitetsforenkling bruges ikke til disse overlappende baser.

Kontrollerede originalkilder (SHA256):

- upload07: 858904ad879ce1e8b7c910cac0fb91d00a2ed5e431f9448476635f7e5a276fa3
- upload13: ddb18be6af9e9a23dc546ff1dcbe2c86ae062beb22ebf4a1140d874f1a94f395

### Referencekørsel og bevarelse af legacy

Den nye reference er allerede kørt på samme G8-fixture, model, likelihood,
seeds og rho-bank som den oprindelige M38/M78-sammenligning. Resultaterne
i tabellerne nedenfor genbruges; denne noteudvidelse udfører ingen ny kørsel.
Samme129-grid sammenlignes særskilt fra forbedringen ved finere grid.
Original M38/M78-evidens fra g8_run_v2 og referencens g8_basis_run/
g8_basis_refine_run bevares i deres oprindelige arkiver og registreringer.

Den generelle degree-vej er stadig tilgængelig; referencebasis kræver
eksplicit basis_kind='g8-local-beta-41-k256-prior20-v1'. Der ændres ikke
defaults eller checkpoints for legacy. Hvis senere nye forsøg kræver en
ændring, erklæres hvad der ændres, hvorfor, og hvordan den sammenlignes med
den bevarede original. Original posterior må ikke udskiftes med en
referenceposterior under primær filtrering eller kontrol.

## Hvorfor den gamle basis fejlede

Den gamle grad-M-basis var Beta(m+1,M-m+1), m=0,...,M.
Antal komponenter og koncentration var dermed koblet:
K=M+1, kappa=alpha+beta=M+2.
For et Beta-led med mean mu_m og koncentration kappa gælder

\[
E_m[X]=\mu_m,\qquad V_m=\frac{\mu_m(1-\mu_m)}{\kappa+1}.
\]

For en positiv mixture med samme kappa og samlet mean mu giver totalvariance

\[
V=\sum_m c_m V_m+\sum_m c_m(\mu_m-\mu)^2
 =\frac{\mu(1-\mu)}{\kappa+1}
  +\frac{\kappa}{\kappa+1}\operatorname{Var}_c(\mu_m)
 \geq\frac{\mu(1-\mu)}{\kappa+1}.
\]

Ved mu=.5 kan M38 ikke komme under .00609756 og M78 ikke under .00308642.
Den gemte FV257-bank når conditional variance .00216612 ved time16/rho=.5,
mean .510071. Der er derfor en konkret repræsentationsbegrænsning. En bedre
QP-løsning, længere fit eller mere præcis normalisering kan ikke fjerne denne
kontinuerte basisgrænse. Celleapproksimationen har lidt andre momenter; de
kontinuerte grænser bruges som designargument, ikke som et eksakt diskret bevis.

## Den valgte mindste ændring

Vi adskiller placering og bredde:

\[
\mu_r=\frac{r+1/2}{41},\quad r=0,\ldots,40,\qquad
\kappa=256,\qquad
\alpha_r=\kappa\mu_r,\quad \beta_r=\kappa(1-\mu_r).
\]

Dertil tilføjes det oprindelige Beta(20,20)-led. Det giver 42 komponenter
og inkluderer initialloven direkte. Basis er fælles og fast for alle tider
og rho-noder. Kun de positive mixture-vægte ændres.

Kappa=256 er den mindste potens af to, for hvilken det centrale variancegulv
.25/(kappa+1) ligger under halvdelen af den mindste gemte FV-nodevariance.
Det nye gulv er .000972763. Denne regel er eksplicit data-informeret af den
allerede eksponerede udviklingscase. 41 jævnt fordelte means giver afstand
1/41=.02439, mindre end den centrale komponent-standardafvigelse .03119.
Det giver overlappende komponenter med omtrent samme antal koefficienter som
M38. Ingen efterfølgende koncentrations-, antal- eller placeringssøgning køres.

Beta(20,20) er bredere end kappa256-leddene. Derfor har hele ordbogen stadig
V >= mu(1-mu)/257 i den kontinuerte fortolkning; lighed kræver passende
komponentmean. At sænke dette gulv er nødvendigt her, men er ikke alene en
garanti for god repræsentation. Den efterfølgende fulde filterkontrol er nødvendig.

Cellebasen er

\[
B_{mi}=\frac{F_m(e_{i+1})-F_m(e_i)}{\Delta x}.
\]

Her er F_m Beta-fordelingens CDF. Den eksisterende
beta_cell_average_basis bruger complementary CDF i højre hale for at
reducere cancellation. Alle komponenter har ikke-negativ enhedsmasse på
griddet; der anvendes ingen posterior-clipping eller normaliseringsreparation.
Priorens cellemasser repræsenteres af første led til roundoff.

## Least squares, sandsynlighed og bevaret model

Det uændrede projektionsproblem er

\[
c^\star=\arg\min_{c\geq0,\ \mathbf1^Tc=1}
 \frac12\sum_i\Delta x
 \left[\sum_m c_m B_{mi}-p_i^{\rm target}\right]^2.
\]

Målet er enten FV-trinnets forudsagte tæthed eller den rå Bayes-posterior.
LS bestemmer her repræsentationsvægtene c, ikke G4's fysiske parametre eller
straffen på a. G4's S3/lambda=0 ændres ikke.

Positivitet og enhedsmasse følger af basis og simplex-constraints.
Den eksisterende active-set-solver bruger SVD på den aktive simplexflade,
ikke normal equations til løsningen. Hessianen dannes til diagnostik.
En uafhængig optimality-gap-kontrol kontrollerer LS-løsningen; særskilte
L1-, mean-, CDF-, variance-, quantile- og scoregates kontrollerer kvaliteten.

Basis-matricens condition number er stadig stor: ca. 3.92e11 ved 257 celler,
mod ca. 7.00e15 for gammel M78 ved129. Derfor påstås ikke perfekt konditionering.
Den præcise prior er tæt på en kombination af de smalle led, og koefficienter
behøver ikke være robuste, selv når den repræsenterede tæthed er det.
Ingen ridge-straf eller signed mixture indføres. De to kendt-parameterreplays
havde nul QP-fallbacks i alle dynamiske trin; alle registrerede Bayes-projektioner
i de nye banks havde nul QP-fallbacks.

Fysisk drift, sigma_x=.2, sigma_y=.03, punktlikelihood GL3, priser, baseline,
rho-noder, missingmask og seeds er uændrede. Hver node viderefører egne
koefficienter og egen conditional state; ingen FV-reference bruges som restart.

## Alternativer og hvorfor dette valg

- Højere Bernstein-grad: kan sænke gulvet, men øger både antallet af
  koefficienter og næsten lineær afhængighed. Den nye basis adskiller de to valg.
- Adaptiv Beta-basis efter hver måling: mulig, men kræver transport mellem
  forskellige baser, nye checkpointkontrakter og flere adaptive beslutninger.
  Den faste basis kunne afprøves med eksisterende interfaces.
- Større målefejl: ville gøre posterioren bredere ved at ændre den statistiske
  model; det løser ikke repræsentationen af den oprindeligt deklarerede model.
- Signed koefficienter: kan forbedre approximation, men sikrer ikke en gyldig
  sandsynlighedstæthed, som Bayes og næste propagation kræver.
- FV-fallback: den pålidelige mulighed ved fejlede gates. Den skal betegnes FV,
  og erstatter ikke en dokumenteret mixture-demonstration. Den var ikke nødvendig
  for at fuldføre denne nye fine-grid mixture-kontrol.

## Registrering, negativt coarse-resultat og målrettet follow-up

Første registrering låser én basis og grids129/h=1/48 samt257/h=1/96 på den
gemte 24-timers G8-fixture. Pilot er 1.176 s og består. Alle oprindelige
marginal-/projektionsgates består, men en NY, strengere kontrol af hver
rho-nodes conditional law afslører coarse-grid-følsomhed.

| Conditional posterior, maksimal L1 | Forskel | Grænse | Resultat |
|---|---:|---:|---|
| Gemt FV129 → FV257 | .0655479 | .05 | Fejler |
| Ny mixture129 → mixture257 | .0584180 | .05 | Fejler |
| Ny mixture129 → FV129 | .0146962 | .05 | Består |
| Ny mixture257 → FV257 | .00729929 | .05 | Består |

G8's gamle FV-accept omfattede marginalgates, ikke denne nye nodegate.
Den skal derfor ikke beskrives som en tidligere kontrol af hver node.
Første nye studies samlede acceptfelt er false og ændres ikke.

Den observerede fejl begrunder én særskilt registreret follow-up:
præcis FV513 og mixture513 med h=1/192, samme basis og fixture. 257-resultater
genbruges. Højst300 nye sekunder og600 sekunder samlet til begge numeriske
invocations. Ingen yderligere grids eller ny basis, hvis kontrollen fejler.

| Conditional posterior | Maks. L1 | Maks. node-logscoreforskel |
|---|---:|---:|
| FV257 → FV513 | .0365687 | .0239552 |
| Mixture257 → mixture513 | .0328833 | .0203220 |
| Mixture257 → FV513 | .0296379 | .0174841 |
| Mixture513 → FV513 | .00348644 | .00309870 |

Alle består L1<=.05, node-mean<=.005 og node-score<=.05.
Alle oprindelige marginalgates består også, inklusive CDF/quantile/momenter,
pre-update evidence og bankvægte. Dette er praktiske fælles rum-/tidskontroller,
ikke en kontinuumsfejlgaranti.

## Hvor meget bedre er repræsentationen?

| Metode og grid | Max bank-marginal posterior-L1 mod FV257 | Max Bayes-projektion-L1 |
|---|---:|---:|
| Gammel M78 / 129 celler | .138551 | .128495 |
| Ny42 / 129 celler | .0239013 | .00149368 |
| Ny42 / 257 celler | .00230264 | .00155602 |

Den store reduktion i projektionsfejl ses allerede ved samme grid129.
Den yderligere forbedring i anden kolonne ved257 inkluderer gridforfiningen;
hele forbedringen til257 tilskrives derfor ikke basisændringen alene.

Ved513 er mixture/FV marginal-L1 .00132938, maksimal logscoreforskel .00076811,
bankvægtforskel .000272535 og Bayes-projektion-L1 .00156262.
Demand-RMSE før måling er .02132239 mod FV's .02132126.
Mixture reproducerer dermed referenceberegningen tæt; dette er ikke en ny
fysisk demand-model eller dokumenteret forecastgevinst over FV.

Massefejl på de nye replays højst2.34e-15, ingen negative masser.
Missing bevarer predicted/posterior-masser og logvægte præcis.
Faktisk JSON-resume vedtime12 på mixture257 giver præcis nul forskel i
conditional masses, koefficienter, logvægte og evidence frem tiltime24.
To nye fokuserede tests består på .079 s (invocation .673 s).
Ingen fuld suite eller historisk genfit.

Første study42.666 s, follow-up107.420 s, i alt150.086 s til numeriske invocations.
Heraf nye banks: mixture25720.094 s, FV51316.936 s, mixture51389.621 s;
257-resume10.484 s er inkluderet i første study.
De gamle G8-replays blev genbrugt, ikke kørt igen.

## Kilder og præcis tilknytning

Tobias K. S. Ritschel, upload13:
Mixture approximations for the Fokker-Planck equation, 28.september2025,
slide6 ligning4: generel mixture sum c_m ell_m;
slide8 ligning7a–7c: least squares med positive koefficienter og sum1.
Dette understøtter mixture/LS-grundlaget for en særskilt referenceudvidelse.
Tavlens konkrete normaliserede Bernstein-basis er fortsat hovedmetoden;
slides' generelle notation er ikke en begrundelse for at erstatte den.

Den konkrete Beta-ordbog, kappa256, cell-integral-repræsentation og
variancegulv-udledningen er vores metodevalg/udledning; de tilskrives ikke
noterne som færdig opskrift. De ikke-ortogonale basisfunktioner kræver vores
generelle LS-solver; noternes særlige ortogonalitetsforenkling bruges ikke.
For standard Brownsk SDE beholder BM-kernen FP-leddet
(1/2) partial_xx(s^2 p); slides' viste diffusionnotation ændrer ikke denne faktor.

Alba Gonzalo Primo, upload09, afsnit6.1.1–6.1.2, ligninger6.1–6.2,
PDF102/trykt90, understøtter separat proces-/målemodel og prediction/correction.
Vores fulde Bayes-densityfilter er ikke hendes EKF.
Alle kildeudsagn bygger på de uploadede dokumenter.

## Implementering, reproduktion og konkret næste G7-opgave

Ny basis er src/method/bm_1_g8_basis.py. Kun G-adapteren bm_1_g8.py får et
eksplicit basis_kind-argument; gamle degree/FV-kald bevarer deres default.
Den eksisterende beta_cell_average_basis, PreparedBetaProjector, FP/Bayes-kode
og kontrolkerner ændres ikke. Alle31 øvrige importerede eksisterende moduler
matcher G-parentens Git-blobs. Ingen ændringer til F/WM-filer.

Runners: src/exe/run_model/bm_1/g8_basis.py og g8_basis_refine.py.
Referencefiler og source hashes låses i registreringerne før hver kørsel.
Den anden runner afviser ændret numerisk source efter første study.
Begge originale registreringer/resultater, den negative coarse-gate, alle
nodehistories, basis, coefficients, evidence, checkpoints og frosne kilder
bevares i råarkivet. Der ændres ikke i gamle G8-arkiver.

Den tidligere G7-anbefaling brugte det gemte time12-origin fra bank_local257.
Efter Johans basisbeslutning er dette alene en 42-leds-reference. Næste
primære opgave er repræsentationsreview og derefter en afgrænset kontrol
inden for den oprindelige Bernstein/Beta-familie. Den primære G7-origin
skal stamme fra denne families egen kontrollerede rekursive filterkørsel.
Genbrug ikke referenceposteriors som skjulte restarts. Den senere
mixture-evaluator skal respektere den faktisk gemte basis og hver nodes
egne koefficienter; vægt objective med de frosne nodevægte.
MAP er fortsat en særskilt approximation.

Den eksisterende weighted_control_evaluation bruger FV-forward fra egne
conditional cell laws og er fortsat korrekt mærket FV. Denne leverance
indeholder ikke en ny mixture-prisoptimizer.

Den nye basis har begrænset mean-/breddeopløsning og er endnu kun kontrolleret
på denne pris-/støjcase. G7's andre priser kan give andre fordelinger;
kontrollerne af det valgte kontrolforslag kan ikke springes over.
Ingen lokal kørsel kræves nu.

