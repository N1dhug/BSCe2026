# G/WM: fælles timebenchmark er kørbar

Den afgrænsede sammenligning er implementeret i
[metoden](../src/method/g_wm_hourly.py) og
[indgangen](../src/exe/run_model/bm_1/g_wm_hourly.py). Se
[fast protokol og nødvendig eksportkorrektion](G_WM_HOURLY_PROTOCOL_20261008.md).
G/time/residual/proceskode, eksisterende WM-modeller/fits og F er uændrede.

Sammenligningen bruger **samme frosne baseline**, fem historiske uger,
720 finite measured targets, mandagFeb7-cutoff, demand-skala, signedprice,
meterpopulation, realised price/weather-scenarie og fysisk timegennemsnit.
Forecast har168 timer og gemmes før målinger åbnes. G's eksisterende
forecast/residualsnapshot genbruges uden fit. WM er en ny tydeligt mærket
timeadapter, ikke det gamle punkt-QP-resultat eller WM's selvstændigt
optimerede baseline. Formålet er at isolere forskellen i dynamik under
fælles baseline og information. Dette er en betinget udviklingssammenligning.

## Matematik til WM-timeadapteren

WM4's eksisterende model, tid i dage, er uændret:

\[
\dot s=As+bU,\quad s=(F,R)^T,\quad
A=\begin{pmatrix}0&1\\r_F&r_R\end{pmatrix},\quad b=(0,r_U)^T,
\qquad I=B+R.
\]

For en fysisk time er h=1/24 dag og U konstant. Indfør Z'=R og en konstant
forcingstate U. Exponential af den augmented4x4-matrix giver i ét kald
\(s_{j+1}=Ts_j+vU_j\) og \(\bar R_j=Hs_j+wU_j\). Observationen er
\(\bar I_j=\bar B_j+\bar R_j\). Z sættes til nul ved hver time; state
fortsætter gennem dagen. Ingen inverseA/eigenværdedivision er nødvendig,
så kritisk dæmpning/degenerateinput ikke kræver særlige divisionsformler.
Tidsenheden24 er eksplicit. Energibalancen er
\(\bar I_j-\bar B_j=24(F_{j+1}-F_j)\), for hele fysiske timer.

WM's drift er affin og r konstant; denne operator er kontinuert eksakt
under piecewise-held U. Det er en anden numerisk implementering end den
eksisterende Euler-solver. Baselineintegralet er stadig fælles G-minutegrid
med den allerede dokumenterede clipping/statistical-design-approksimation.
Den nye adapter genbruger native WM4's coefficient/initial-state-validator.

## Fit, information og erklærede forskelle

For givet r fås hver historisk dags observationsdesign
\(\bar I=\bar B+H_{day}s_0+v_{day}\). Kun finite measured timer tæller.
Initials s0 findes ved least squares med F0∈[-1,1], R0 fri (som native WM's
initial-condition-validator). Det unconstrained minimum og begge F-boundary
minima dækker den eksakte løsning; R bestemmes analytisk på hver rand.
Rank-deficient designs håndteres med lstsq, uden inverseH'H.

Ydre least_squares tilpasser tre strukturelle størrelser under native
WM-domæne: r_F<0,r_U<=0,r_R<0,r_F<=r_U,r_R<=r_F+r_U. Sæt
r=[−a,−a·rho,−a(1+rho)−d],a>0,d>=0,rho∈[0,1]. Faste numeriske bounds/seeds,
to starts og samlet60s/400rawcalls står i protokollen; kun laveste
konvergerede trænings-MSE vælges. Ingen modelvalg på forecastugen.

**Dette er free-run hourly variable-projection fit**, ikke WM's hidtidige
reconstructed-drift QP-fit. WM's historiske initialstates estimeres direkte
ved targetdagens start; G bruger24h-warmup med sin faste neutrale historiske
start og median af egne efter-warm-up-states. Begge prognoser har syv
mandagsfrosne uafhængige dagsbaner. WM har tre globale r og to initials for
hver af30 tilgængelige targetdage (60 lokale størrelser); G har fire fælles
fitstørrelser (C,k,Xi,g_mid) og afledte medians. Forskellige fitfrihedsgrader
oplyses, ikke skjules som en kontrolleret complexity-matched sammenligning.

Fem targetdage er helt umålte. De imputeres ikke og får ingen fitted WM-start.
WM's frozen medians bruger4/5/5/4/4/4/4 tilgængelige dage; G's startsregel
bevares uændret. Samme720 historical measuredtargets indgår i scorer/fits.
WM-initials/inputs/fit bruger ingen målinger fra targetforecastugen.

Initial F0 bounds er ikke en fullpath-stateconstraint. Ingen state clipper.
Forecasts maksimale |F| ved timeendpoints rapporteres; intervalernes indre
ekstrema eller fysisk population-/enhedskalibrering erklæres ikke bevist.
Fysisk time-startlabel/enhed er stadig Johans working assumptions.

## Sammenligneligt residuallag og score

Begge modeller har særskilt puremean og en fælles **opskrift** til outputfejl:
pooled empirical Y−model fra replay af egne frozenmedianstarts på de fem
historiske uger. Fejlbankerne er modelspecifikke; middel/varians og inverse
empiricalCDF-bånd beregnes med samme alpha=1−.95-konvention.
G's bank genbruges præcist; WM's bank gemmes i eget snapshot. Begge er
in-sample udviklingslove, ikke identifiedmeasurement noise eller independent
calibration. Ingen stateprocesdiffusion, Gaussian substitution, clipping,
proces/residualdoublecount eller prisstyringsændring er med.

Samme rå finite mask giver MSE/RMSE/MAE/bias og marginal95%-dækning,
meanwidth og intervalscore. Manglende måling er ikke målt nul. Bånd bliver
ikke bredere, fordi slutugen underdækker. Fælles168h-dækning hævdes ikke.

## Kørsel og snapshots

```text
python -B -m src.exe.run_model.bm_1.g_wm_hourly fit --g-snapshot G_FIT/interval_snapshot.json --raw-csv HISTORY.csv --seconds 60 --max-calls 400 --output NEW_WM_FIT
python -B -m src.exe.run_model.bm_1.g_wm_hourly predict --g-snapshot G_FIT/interval_snapshot.json --wm-snapshot NEW_WM_FIT/wm_snapshot.json --g-residual-snapshot G_ERRORS/residual_snapshot.json --inputs-csv FUTURE.csv --observations-csv OBSERVED.csv --output NEW_COMPARISON
```

Nye outputmapper kræves. Own WM snapshot binder præcist parentfingerprint,
trainingfingerprint, sourcehashes, r, starts og errorbank. Strict load refitter
eller migrerer ikke automatisk. JSON/checkpoints er atomiske; CSV/plots og
komplette snapshots forbliver lokale. Remote leverance indeholder kun kode,
syntetiske tests og Johans godkendte aggregater fra samme udviklingscase.

Denne udviklingskørsel har ét oprindeligt fit941f4f1 og en eksplicit reviewed
**export-only snapshotmigration** til a184d2e-kilder. Den ændrer kun kvantil-
konventionen til samme floatingalpha som standaloneG: ved720 samples gav
literal .025 mod alpha/2 forskel på én rang. Original fit/forecast er bevaret;
alle parametre/initials/errorbanks og øvrige kildehashes er identiske.
Den lokale journal beviser eksakt tolinje-replacement i hele kildefilen;
ingen optimizer genkørt. Finalforecast er gemt separat. Første snapshot
kræver941f4f1-kilder; final bruger export_migration/wm_snapshot.json.
Et nyt fit på nuværende kode producerer direkte den korrekte eksportregel.

## Faktiske resultater

| Middelmodel | RMSE | MAE | Bias |
|---|---:|---:|---:|
| Fælles baseline |0,379144|0,285199|+0,060967|
| PureG |0,378396|0,284756|+0,063882|
| WM4 adapted hourly |0,379297|0,285675|+0,052424|
| G + empirical mean error |0,377901|0,284091|+0,060879|
| WM + empirical mean error |0,380183|0,286973|+0,058494|

| Nominelt95%-bånd | Dækning | Middelbredde | Intervalscore |
|---|---:|---:|---:|
| G empirical output error |120/168=71,4286%|0,736655|3,617927|
| WM empirical output error |121/168=72,0238%|0,727822|3,711246|

Én WM-fit2,527086s/148rawcalls, begge seeds konvergerer til omtrent samme
trainingMSE.005414061. G-refit0, WM-refit0 efter første fit. r i dage:
[-2,810914,−1,520e−11,−2,810914]; rho ved nedre rand, ekstra damping≈6,0e−8.
Den fitted price response er praktisk nul i dette fit; dette er ikke et
bevis for, at pris ikke virker fysisk. G's allerede dokumenterede g_mid
ved rand/weakidentification står også uændret.

Final input-only forecast0,661687s. Max|WM_F| ved timeendpoints.033645,
største WM-integralenergi-balancefejl1,110e−16 <=1e−9. G-forfining består
som tidligere. Original forecast plus nødvendig korrigeret export er begge
lokale; ingen score-, grid-, tolerance-, model- eller budgetændring skjules.

Puremeans og bånd er meget ens, og baseline dominerer. G's RMSE-forskel
fra WM er0,000901 (cirka0,24%); denne ene allerede sete betingede uge er
ikke grundlag for generel modelrangering. Begge nominelle95%-bånd underdækker.
Residualjusteringen beholdes selv hvor den forringer WM's RMSE; der er ikke
fravalgt bias efter scoring. Ingen kalibrerings-, holdout- eller kontrolaccept.

## Kontroller og resterende arbejde

Fem nye fokuserede tests dækker nativeWM equilibrium/Euler-limit, independent
integralenergi-balance/halfstepcomposition, independent boundedLS, synthetic
known-law recovery, egne source/cutoff/input-only kontrakter og exactG-output/
sharedbaseline/scaling/scoremasks. Alle34 kontroller bestod17,554s før
science; efter kvantilrettelsen består alle fem nye på4,794s, inklusive
standaloneG-band-regression. De oprindelige29 kerner er uændrede.
Figuren er visuelt kontrolleret; raw-artefakter sikres lokalt med CRC/SHA256.

Grundkode, godkendte udvidelser og denne afgrænsede benchmark er udført.
Der mangler **ingen implementering til disse afgrænsede leverancer**.
Senere arbejde kan være flere perioder/origins med præregistreret split,
uafhængig kalibrering og robust state-/baseline-/price-identifikation eller
fysisk closedloop. Det er udvidelser, ikke nye automatiske G/F/bankopgaver.
Otte-timers-rammen nulstilles ikke; tid fra tidligere review er umålt.
Denne G-chat på Sol High er passende, når Johan vælger næste udvidelse.
