# G8 — review af vejlederens Bernstein-basis

6. oktober 2026. G-parent: `7ecea53f289edbc2b86c7e7ff3e273694075aa73`.
Seneste main kontrolleret ved reviewstart: `fa6c60fa3d1b4843d6e82028163a7374d011afcf`.
Dette er G's matematiske repræsentationsreview; F/WM og historiske fits ændres ikke.

## Beslutning og præcis status

**M=254, altså 255 af vejlederens oprindelige basisfunktioner, vælges som én
kandidat til næste afgrænsede rekursive G8-forsøg.** Den statiske kontrol af
gemte fordelinger består. Et helt filter med denne kandidat er endnu ikke
kørt, og mixture→G7 er derfor stadig ikke accepteret.

Reviewet ændrer ingen eksisterende produktionskerne eller legacy-default.
Den nye fil `root/g8/bernstein_review.py` er en afgrænset videnskabelig
diagnose. Dens ene invocation tog 9.596362 s, inklusive syv LS-kontroller og før den
afsluttende komprimerede array-eksport.
Rapporten står i `root/g8/bernstein_review_report.json`; alle arrays, input,
registrering og frosne importerede kilder er bevaret i råarkivet
`G8_Bernstein_review_20261006.zip`.

Den lokaliserede 42-leds Beta-ordbog forbliver sammenligningsreference.
M38/M78-resultaterne og deres negative posterior-gates bevares. Der er ikke
kørt nye historiske prognoser, parameterfits eller prisoptimering.

## 1. Samme basisfamilie, større grad

Vejlederens tavle (upload07, `07-wSnLZuio-1-.jpg`) giver

\[
\ell_{m,M}(x)=(M+1)\binom{M}{m}x^m(1-x)^{M-m}
=\operatorname{Beta}(x;m+1,M-m+1),\quad m=0,\ldots,M.
\]

Vi repræsenterer en tæthed ved

\[
p_M(x)=\sum_{m=0}^M c_m\ell_{m,M}(x),\qquad c_m\geq0,\quad\sum_m c_m=1.
\]

Hvert led har integral 1. Positivitet og simplex-betingelsen sikrer derfor
en sandsynlighedstæthed. At øge M ændrer opløsningen inden for den oprindelige
familie; det introducerer ikke den alternative lokaliserede ordbog.

Sæt \(u_m=(m+1)/(M+2)\). Beta-momenterne giver

\[
E[X\mid m]=u_m,\qquad
\operatorname{Var}(X\mid m)=\frac{u_m(1-u_m)}{M+3}.
\]

Med \(\mu=\sum_m c_mu_m\) og \(S=\sum_m c_m(u_m-\mu)^2\) fås fra
totalvarians og \(E[u(1-u)]=\mu(1-\mu)-S\):

\[
\boxed{\operatorname{Var}(X)=\frac{\mu(1-\mu)}{M+3}
+\frac{M+2}{M+3}S\ \geq\ \frac{\mu(1-\mu)}{M+3}.}
\]

Blandinger af for brede positive komponenter kan altså ikke skabe vilkårligt
smalle posteriorer. Flere forskellige vægte alene fjerner ikke dette gulv.
Hvis en kontinuert måltæthed med middel \(\mu\) og varians v skal matches
præcist i begge momenter, er det nødvendigt, at

\[
M\geq\left\lceil\frac{\mu(1-\mu)}{v}-3\right\rceil.
\]

Dette er kun et nødvendigt momentkriterium. Det garanterer hverken L1-form,
halepræcision eller filtreringsscore. Det er heller ikke en nødvendig bound
for blot at bestå den løsere, absolutte variansgate 0.002.

På de gemte FV513-love er den største ratio 117.710418 ved time16, rho=0.5:
middel 0.511328337, varians 0.00212276596. Derfor kræves mindst M115 for
præcist kontinuert momentmatch. FV257 giver M113. FV-variansen her beregnes
for den stykvist konstante celletæthed: midpoint-variansen plus dx²/12.
Bounden er et kontinuert designkriterium, ikke en eksakt ulighed for enhver
midpoint-diskretisering.

Vi vælger M254 som **én udviklingskandidat**, ikke som bevist minimalt gradtal.
Da M+2=256, har den samme koncentration som den allerede bevarede smalle
reference. Det centrale variansgulv er 0.25/257 = 0.000972763, under halvdelen
af den smalleste gemte FV513-varians. M115 ville kun ligge ved en nødvendig
grænse; der ville ikke være tilsvarende spillerum til formtilpasning.
Dette valg udnytter eksponerede udviklingsresultater og er ikke blind validering.

## 2. Eksakt prior gennem gradforhøjelse

Den oprindelige prior Beta(20,20) er præcis \(\ell_{19,38}\).
Identiteten \(1=x+(1-x)\) giver for den normaliserede basis

\[
\ell_{m,M}=
\frac{M+1-m}{M+2}\ell_{m,M+1}
+\frac{m+1}{M+2}\ell_{m+1,M+1}.
\]

Vægtene opdateres således, med vægte uden for indeksområdet sat til nul:

\[
c^{M+1}_j=\frac{M+1-j}{M+2}c^M_j
+\frac{j}{M+2}c^M_{j-1},\qquad j=0,\ldots,M+1.
\]

Begge bidrag er positive, og hver gammel vægt fordeles med sum 1.
Tætheden er uændret. Gentaget fra M38 til M254 giver dette en eksakt
repræsentation af samme prior, uden en ekstra Beta(20,20)-komponent.
Den målte prior-L1-fejl på 513 celler er 3.18e-14 og vægtsumfejlen 2.22e-16.
217 vægte er positive; dette er en initialisering, ikke posterior-fit.

Identiteten viser også, at de kontinuerte positive familier er indlejrede:
et gammelt fit kan bevares ved højere M. På samme celleintegrerede grid kan
det optimale LS-tab derfor ikke vokse med M i eksakt aritmetik.
Det lover hverken bedre konditionering eller mindre runtime.

## 3. Celleintegraler er et eksplicit numerisk valg

For celle \(I_j=[e_j,e_{j+1}]\) af bredde dx anvendes

\[
B_{mj}=\frac{1}{dx}\int_{e_j}^{e_{j+1}}\ell_{m,M}(x)\,dx,
\qquad p_j=\sum_m c_mB_{mj}.
\]

Integralerne beregnes med den eksisterende `beta_cell_average_basis`, som
bruger Beta-CDF og komplementær CDF til stabile differenser. Dermed matcher
repræsentationen FV's cellemasser, og den lineære gradforhøjelsesidentitet
bevares også efter integration. Den kontinuerte basisformel er uændret.

Legacy-gradvejen sampler derimod i cellemidtpunkter og normaliserer hver
basisrække diskret. Denne vej bevares. Den nye celleintegrerede variant skal
være et navngivet, opt-in G-valg; den må ikke erstatte gammel adfærd stiltiende.
Kontinuerte gradforhøjelsesvægte må heller ikke kaldes eksakte for den gamle
række-normaliserede matrix uden at håndtere rækkernes skaleringsfaktorer.

Ved M254 er den største komponentvise L1-forskel mellem de to
diskretiseringer 0.01470 på 513 celler og 0.05472 på 257 celler, nær randen.
For samme statisk fittede centrale posterior-vægte er forskellen højst
0.00007198 hhv. 0.00028118. Det sidste er ikke et refit eller et filterbevis;
det dokumenterer blot, hvor stor kvadraturændringen er i de valgte eksempler.

## 4. LS-koden og hvad dens stopkriterium siger

Med \(A_{jm}=\sqrt{dx}B_{mj}\), \(b_j=\sqrt{dx}p_j^{\rm target}\) løses

\[
\min_{c\geq0,\,1^Tc=1}f(c),\qquad f(c)=\tfrac12\|Ac-b\|_2^2.
\]

Den anvendte `PreparedBetaProjector` løser aktive flader med SVD i et
nulsum-underrum. Den danner ikke normale ligninger til selve fladeløsningen.
Den gamle separate legacy-`active_set`-vej må ikke forveksles med denne.

For \(g=\nabla f(c)\) kontrolleres simplex-gappet

\[
\mathcal G=c^Tg-\min_m g_m.
\]

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

De eksisterende tolerancer er uændrede: primær skaleret gap 1e-11 og
eventuel SLSQP-fallback 1e-9. Alle syv kontroller brugte den primære løsning;
største skalerede gap var 9.615e-12. Der blev ikke tilføjet ridge-straf,
negative vægte, ændret likelihood eller løsere acceptkrav.

Konditioneringen er fortsat vanskelig. På 513 celler har A konditionstal
1.30e15 og numerisk rang119 af255 ved den registrerede SVD-tærskel;
på 257 celler er tallene 4.18e17 og rang115. De beregnede Gram-konditionstal
er omkring 1e18 og upålidelige som præcise størrelser nær maskinpræcision.
De viser ikke, at højere M har løst konditioneringsproblemet. Tæthedsfit kan
være præcist, mens koefficienterne er følsomme eller numerisk uentydige.
Derfor vurderes fordelinger, gap, masse og runtime frem for koefficienternes
individuelle fortolkning. Ingen ny solver er begrundet af denne statiske test.

## 5. Den udførte, afgrænsede kontrol

Registreringen blev skrevet før kandidatmatricen blev bygget. Der blev
valgt én gemt posterior per rho-node, med størst mu(1-mu)/v på FV513, plus
den globalt vanskeligste FV257-posterior og den eksakte prior. Alle fem
nodevalg faldt på time16. De er udvalgt som vanskelige momenteksempler;
de er ikke et tilfældigt eller uafhængigt udvalg, og de dækker ikke alle
mulige former, randtilfælde eller fremtidige posteriorer.

| Gemte mål | Antal LS-kontroller | Største L1-fejl | LS-tid |
|---|---:|---:|---:|
| Beta(20,20), 513 celler | 1 | 3.18e-14 | 0.0004 s |
| FV513, fem rho-noder ved time16 | 5 | 1.27144e-6 | 1.50–1.68 s per fit |
| FV257, rho=0.5 ved time16 | 1 | 1.11957e-6 | 0.898 s |

Største middelfejl er 3.597e-8, variansfejl 5.548e-9 og CDF-fejl 2.419e-7.
Vægtsum-/tæthedsmassefejl er højst 4.441e-16. Alle består de uændrede gates
L1=.05, middel=.005, CDF=.03, varians=.002 og masse=1e-9.
De fem fine posteriorfits ender med 20–26 aktive koefficienter, men **alle
255 basisfunktioner er tilgængelige**; der er ikke indført en reduceret ordbog.

Hvert posterior-fit starter fra den eksisterende uniforme default. Statisk
cache_size=32 lagrer kun løsningsgeometri. Et dynamisk forløb bruger egne
forrige koefficienter og mange projektioner per FV-tidstrin; 1.5 s per koldt
fit kan derfor hverken bruges som målt filtertid eller som sikker prognose
for hele kørslen. Reviewet accepterer kun statisk repræsentation.

Reproduktion fra arkivets rod (nyt outputnavn kræves):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B frozen_source/root/g8/bernstein_review.py \
  --repo frozen_source --fv257 inputs/bank_fv257.npz \
  --fv513 inputs/bank_fv513.npz --legacy78 inputs/bank_mix78.npz \
  --output rerun
```

Det udførte script har SHA256
`1c21420142cd4e4bb93c2865f701c475ecfdcd5ba866930c9dad3f2a392733c3`.
Den fulde manifest med input- og kildehashes ligger i rapport/råarkiv.
Scriptet er kørt én gang; ingen ekstra modelkørsler eller fuld testsuite.

## 6. Konkrete næste implementeringsopgave — Sol High

1. Tilføj et eksplicit G-valg for celleintegreret Bernstein M254 og den
   gradforhøjede Beta(20,20)-prior. Bevar legacy-gradvejen og 42-leds-reference.
   Basisidentitet og kvadratur skal fremgå af checkpoint; gamle checkpoints
   må ikke fortolkes som nye koefficienter. Genbrug eksisterende LS-kerne.
2. Kontroller kun den nye kontrakt: prioridentitet/masse og faktisk JSON-resume
   med egne nodekoefficienter. Ingen januarrefits eller F/WM-testpakker.
3. Kør den eksisterende fem-node-case fra t0-prioren på 257 celler, h=1/96,
   CFL=.4. Første del er runtimepilot og fortsættes fra eget checkpoint,
   så samme arbejde ikke gentages. Ingen initialisering fra FV-posteriorer
   eller 42-leds-reference. Gem per time og bevar eventuel delvis/negativ kørsel.
4. Kun hvis præcision/runtime tillader det inden for budget: kør samme M254
   fra prior på 513 celler, h=1/192, og udfør eksisterende conditional-node-,
   marginal-, evidence-, vægt-, projektions- og resume-gates. FV257/FV513 og
   den lokaliserede reference genbruges som reference, ikke som egen state.

**Ny numerisk budgetramme: højst 300 s samlet, inklusive højst 60 s pilot,
eventuelle filterforfininger og resume-kontrol; reservér 30 s til eksport.**
Reviewets allerede målte 9.596 s er separat registreret. Budgettet er en
stopgrænse, ikke et løfte om, at alle trin kan nås. En deadline skal også
kunne stoppe et langt LS-kald; nuværende drift-callback alene er utilstrækkelig.
Vurder den målte projektionstakt tidligt. Hvis det ikke kan nås, standses
med eksplicit uafsluttet/negativ status. Ingen automatisk M-søgning, ekstra
budget, større observationstøj eller lempede gates. Dette er et delbudget
inden for den eksisterende samlede G-ramme, ikke en nulstilling af den.

Et statisk positivt resultat giver ikke automatisk adgang til G7.
Først efter eget kontrolleret filter kan samme banks egne posteriorer blive
primær G7-origin. Det næste arbejde er implementering af en afklaret kontrakt;
Sol High er en arbejdsanbefaling, ikke en målt modelsammenligning.
Johan behøver ikke køre noget lokalt nu.

## Kilder og hvad der er vores udledning

- Supervisorens upload07: den normaliserede Bernstein-formel og mixture-sum.
  Originalfotoets SHA256 er
  `858904ad879ce1e8b7c910cac0fb91d00a2ed5e431f9448476635f7e5a276fa3`.
- Ritschel, *Mixture Approximations for the Fokker–Planck Equation*,
  28. september 2025, slide6 ligning4 og slide8 ligning7: mixture og positiv
  LS-projektion med vægtsum1. Slide9's særlige ortogonalitet antages ikke her.
- Beta-momenter, totalvarians, gradforhøjelse og konveks gap-bound er vores
  eksplicitte algebra ovenfor. M254, cellekvadratur og forsøgsbudget er vores
  dokumenterede udviklingsvalg; de tilskrives ikke vejlederens tavle.
- Kodegrundlag: uændret `src/method/beta_mixture.py` og
  `src/method/fokker_planc.py`, plus de fem importerede compatibility-/solverfiler
  i manifestet. Det gamle kildegrundlag og alle modeller bevares.

Se også [den tidligere referenceudledning og kildekort](G8_POSTERIOR_BASIS_REVIEW.md).
