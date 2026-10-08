# G-S3 grundversion med afgrænset procesusikkerhed

Den deterministiske fit/snapshot/predict/score-vej er færdig. G kan nu også
beregne en ikke-lineær procesfordeling fra samme fitted model og eksportere
middel, varians, punktvise kvantiler og fælles fejlmål. **Historisk kalibrering
af støjstyrken er afvist i udviklingseksemplet**; den fungerende procesvej
kræver en eksplicit scenariestyrke. Ingen totalusikkerhed eller fysisk
95%-dækningsaccept er erklæret. Dette er første brugbare grundversion.

Se [deterministisk brug](G_S3_STANDALONE_GUIDE_20261008.md),
[protokol før kørsel](G_S3_UNCERTAINTY_PROTOCOL_20261008.md),
[modelkode](../src/method/bm_1_s3_uncertainty.py) og
[indgang/eksport](../src/exe/run_model/bm_1/s3_uncertainty.py).

## Matematik og fortolkning

Alle dynamiske tider er timer. U er normaliseret til [0,1], f(x)=1−2x,
g er snapshotets kontinuerte monotone to-stykkers respons. For baseline B(t):

\[
\delta(x,t)=\tanh\{k[1-2x+g(U(t))]/2\},\qquad
D(x,t)=B(t)+\Xi\delta(x,t)
\begin{cases}1-B(t),&\delta\ge0,\\B(t),&\delta<0.\end{cases}
\]

\[
dX_t=\frac{D(X_t,t)-B(t)}{C}\,dt+
\sigma_xX_t(1-X_t)\,dW_t,\quad \operatorname{Var}(dW_t)=dt.
\]

Hver dag starter fra sit eget frosne historiske medianstate med P0=0.
Der assimileres ingen fremtidige målinger. Den numeriske procesvej genbruger
den eksisterende konservative Fokker–Planck-FV på [0,1] med zero-flux-rand,
upwind-drift og implicit Euler. Formelt er ligningen
\(\partial_t p=-\partial_x(ap)+\tfrac12\partial_{xx}(b^2p)\).
Sandsynlighedsmasser er de numeriske ukendte, ikke densityværdier som vægte.
Ingen klipning eller renormalisering reparerer et forløb. Dirac-starten
repræsenteres i én celle; afvigelsen fra ønsket start rapporteres særskilt.
Numerisk forfining er en udviklingskontrol, ikke et bevis for en eksakt lov.

Output er E[D(X)] og Var[D(X)] beregnet direkte fra masserne. E[D(X)] er
generelt forskellig fra D(E[X]) og ODE-prognosen. Båndgrænserne er den
diskrete generaliserede inverse CDF's .025/.975-kvantiler af D(X), ikke
Gauss-bånd. Ved sigma=0 genbruges præcis S3's RK4-prognose med nul varians.
Den særlige nulgrænse undgår kunstig FV-advektionsspredning.

Den separate historiske diagnose bruger lokal lineær støjpropagation:
\(P'=2(D_x/C)P+[x(1-x)]^2\) ved enhedsamplitude, P0=0 efter warm-up.
For delta>=0 er \(D_x=-k\Xi(1-B)(1-\delta^2)\); for delta<0 erstattes
1−B af B. Ved knækket bruges den højre demandgren som i eksisterende kode.
Demand-enhedsvariansen er \(D_x^2P\). Med matching-weekday samplevarianser
v (ddof=1) og gennemsnitlige fem-historie enhedsprofiler w tilpasses
\(q=\max(0,w^Tv/(w^Tw))\). Nulsensitivitet er uidentificerbar.
Time 0 udelades. Dette fit identificerer ikke fysisk processtøj alene.
Varians over 1/4 strider mod enhver lov på [0,1]; understøttelsesbrud i
frosne D([0,1]) viser uforenelighed med et observationsfejlfrit processpor
under den erklærede udviklingsapproksimation. Afvist q bruges aldrig skjult.

## Brug

Kør fra repository-roden med det eksisterende videnskabelige Python-miljø:

```text
python -B -m src.exe.run_model.bm_1.s3_uncertainty diagnose --snapshot S3_SNAPSHOT.json --raw-csv HISTORY.csv --output NEW_AUDIT_DIR --seconds 60
python -B -m src.exe.run_model.bm_1.s3_uncertainty predict --snapshot S3_SNAPSHOT.json --inputs-csv FUTURE.csv --sigma-x 0.2 --output NEW_PROCESS_DIR --observations-csv OBSERVED.csv
```

.2 er kun en illustrationsstyrke per sqrt(time), ikke et accepteret fit.
API: `diagnose_calibration`, `process_forecast`, `score_process_forecast`.
Inputs/snapshot/kalender følger S3-kontrakten. Forårs-DST bevarer 167 fysiske
timer; efterårets gentagne naive lokale timer afvises. Forecast åbner ingen
observationsfil før CSV/JSON er gemt. Outputmappen skal være ny.
CLI returnerer 0 for komplet processcenarie; 2 for numerisk fejl/ugyldigt
input eller afvist diagnose. En diagnose-kandidat er ikke kalibreringsaccept.

CSV har eksisterende baseline/ODE/tid/pris plus `process_mean`,
`process_variance`, `process_lower95`, `process_upper95`, `process_state_mean`
og `process_state_variance`. Rå demandvarians er normaliseret varians gange
skala²; middel/kvantiler transformeres affint. Observationer/måltemaske
tilføjes kun ved scoring. Udeladte usikkerhedskilder er deklareret i JSON:
parameter-, baseline-, input- og observations/modelfejl. Procesvarians er
derfor ikke en estimeret totalvarians.

Fælles scorer er MSE/RMSE/MAE/bias på samme finite måltemaske for procesmiddel,
ODE og baseline. Båndets intervalscore er bredde+40·underskridelse+
40·overskridelse for alpha=.05. Dækning er punktvis og ikke samtidig over
ugen. Ingen refit eller støjjustering ud fra scoring. Disse felter gør WM-
sammenligning praktisk, når population, information, enheder og origin
matches; der er endnu ikke udført et fælles WM-benchmark.

## Faktisk Feb7-udviklingskørsel

Protokol/kode frøs ved commit 1bba8e7 før begge nye kørsler. Det oprindelige
S3-fit genbruges; nul nye optimizerkald. Første FV-grid bestod, så den
betingede ekstra forfining blev ikke brugt. Tal her er aggregerede:

| Kontrol/resultat | Værdi |
|---|---:|
| Historisk diagnose | 1,127 s; numerik bestået, fysisk fortolkning afvist |
| Understøttelsesbrud | 657/720 historiske målinger, 91,25 % |
| Fine fitted q / sigma pr. sqrt(time) | 25,388687 / 5,038719; afvist |
| Største LNA-statevarians | 28,744177; nødvendig grænse 0,25 |
| Historisk demand / enheds-std / fitted profil-std fejl | 7,245e−8 / 6,123e−8 / 9,271e−8 |
| FV processcenarie | 2,368 s; sigma=.2; 168 timer; alle gates bestået |
| Grids / maksimumtrin | 128/256 celler; 1/24 og 1/48 time |
| Største mean-fejl, demand / state | 6,607e−5 / 0,001955 |
| Største variansfejl, demand / state | 1,922e−7 / 0,000225 |
| Største demandkvantil / state-CDF fejl | 0,000201 / 0,016145 |
| Procesmiddel / ODE / baseline RMSE | 0,377679 / 0,377435 / 0,378180 |
| Dækning / middelbredde / intervalscore | 4,1667 % / 0,027203 / 10,872159 |

Johan præciserede efter kørsel, at hver værdi er forbrug over en hel time,
og at Demand efter hans erindring er Power. Det peger på timegennemsnit
af effekt; fysisk enhed og effekt/energi-konvention er endnu ikke verificeret.
Koden og snapshotet bruger fortsat den eksplicit erklærede punktapproksimation.
Støttebrud og dækningsafvisning ovenfor gælder dette point-output-kontrakt;
de er ikke en afvisning af enhver intervalbaseret G-model. Et fysisk timeoutput
kræver en intervaloperator: for effektsignal er \(\bar D_h=\int_h^{h+1}D_tdt\)
divideret med intervallets varighed. Deterministisk kan eksisterende G-kode
bruge \(\bar B_h+C(X_{h+1}-X_h)/\Delta t\). I SDE følger i stedet
\(\int Ddt=\int Bdt+C\Delta X-C\int b(X)dW\), så man må ikke genbruge
den deterministiske endpointidentitet som stokastisk intervalreadout.
Det kræver også tidskovarians, ikke blot punktvise FV-marginaler.

Scorer er i uverificerede rå workbook-enheder. Realiseret pris/temperatur
er betingede inputs, ikke en blind vejrprognose. Båndene er meget smallere
end målespredningen: numerisk gyldighed er tydeligt adskilt fra brugbar
observeret dækningskalibrering. En større sigma kan ikke flytte demand uden
for D([0,1]); vi starter derfor ingen bred støjsøgning for at skjule dette.

## Slutstatus og senere arbejde

Otte oprindelige S3-tests og otte nye proceskontroller består. De første
15 blev kørt samlet i 2,474 s; efter tilføjet ikke-nul spring-DST-check
bestod de otte proceskontroller i 2,595 s. Kontroller dækker variansfit,
native demandformel, nulgrænse uden refit/demandleak, konservativ ren
diffusion, budgetstop, måltemaske/skalering, numerisk afvisning, kalender
og forecast bevaret før scoringfejl. Windows-sandboxens atomiske testfil-
blokering blev løst ved autoriseret lokal testkørsel uden at svække I/O.

Grundimplementeringen er færdig til undersøgelse og senere udbygning.
Kalibreret observeret usikkerhed er åben. Næste faglige prioritet er at
verificere den præcise demand-enhed og implementere intervaloperator ud fra
Johans timepræcisering, derefter tage stilling til
baseline/modelfejl og støjfortolkning i et særskilt afgrænset arbejde.
Ingen obligatorisk bankdemo, BM2-implementering eller F-undersøgelse følger
automatisk. Eksisterende deterministisk prisstyring bevares; historisk
closed loop og styringsgevinst er ikke valideret af denne prognosekørsel.

Komplet pakke med inputs, snapshots, tidsserier, figur og kontrolkvittering
bevares lokalt. Remote leverance er kode, syntetiske tests og aggregerede
resultater under Johans allerede givne tilladelse. Tid/budget bevares som
samme forløb, ikke en ny otte-timers-ramme. Denne G-chat på Sol High er
tilstrækkelig til næste almindelige implementering; ingen ny chat kræves.
