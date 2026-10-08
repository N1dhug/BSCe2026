# G/WM: fælles timebenchmark før faktisk WM-fit

Pas startet8.oktober2026 kl.19:49:38 UTC (21:49:38 Copenhagen). Johan bad
om fortsættelse og resterende tid. Første estimat45–90min for denne leverance;
G-grundkode/residuallag er færdige. Dette er samme oprindelige budgetforløb,
ikke en ny8h-ramme. Kalibrerings-/F-/bankstudier startes ikke.

Eksisterende WM4-wrapper er punktbaseret, bruger andre skalaer og kræver
fremtidigt I. Den kan derfor ikke bruges umærket til den nye G-timekontrakt.
Den nye adapter følger WM4's **samme** dF=Rdt, dR=(r_F F+r_U U+r_R R)dt,
I=B+R, tid i dage. U holdes i hver fysisk time; augmented matrixexponential
beregner eksakt transition og integreret R. Baseline/timefeatures, skalaer,
cutoff, rå meterpopulation og inputscenarie genbruges fra G's frozen parent.
Dette isolerer dynamikken under shared baseline; det er ikke WM's bedst
mulige selvstændige baseline eller en gengivelse af gamle WM-resultater.

## Fast fit og kørsel

1. G b2a6776-snapshot og afa9af0-residualsnapshot læses uændret; ingen G-fit.
2. Én WM-fitinvocation på samme Jan1–Feb14 lokale udviklingsinput,
   cutoffFeb7. Fem egne historiske targetdage pr.weekday, samme measuredmask,
   ingen interpolation af manglende Demand. Dage med<2 finite targettimer
   giver ingen estimeret start; forecastmedian bruger tilgængelige dage.
3. WM-fit er **nyt hourly free-run least squares** med variable projection
   af hver historisk [F0,R0]. Det er ikke det gamle reconstructed-drift QP.
   F0∈[-1,1], R0 fri som WM's native initial-condition-validator. Starts
   estimeres direkte ved historisk targetstart; ingen WM24h-warmup. Dette
   er en erklæret fitting-/initialiseringsforskel fra G, ikke skjult paritet.
4. Én fælles r/uge. a=exp(theta0),d=expm1(theta1),rho=theta2,
   r=[−a,−a·rho,−a(1+rho)−d]. Bounds a∈[1e−6,1e4],d∈[0,1e4],rho∈[0,1]
   respekterer native restoring/damping/price/equilibrium constraints.
   To faste seeds (a,d,rho)=(1,1,.25),(100,10,.5). Samlet max60s/400 raw
   residualcalls, x_scale=jac,ftol=xtol=gtol=1e−8. Mindst én konvergeret
   kandidat kræves; laveste converged trænings-MSE vælges. Provisional,
   calls/checkpoints/stops bevares. Ingen budgetreset/nyfit/boundændring.
5. Median af egne historiske fitted targetstarts pr.weekday fryses mandag.
   Hver forecastdag er egen24h-bane som G; sommer-DST bruger fysiske timer.
   Numerik er exact frozen-U WM hourtransition; energibalance<=1e−9 kontrolleres.
   State-F's timeendpointdomæne rapporteres uden clipping/fullpath-garanti.
6. WM's empirical error bank fås fra replay af de frozen medianstarts på
   samme targettimer. Samme iid pooled empirical output-error-opskrift som G,
   inklusive mean og inverse-CDF95%-kvantiler; in-sample udviklingsmodel.
   Ingen stateprocesdiffusion, parameter-/inputstøj eller Gauss-erstatning.
7. Én input-only168h forecast før observationsfil åbnes; derefter fælles
   raw finite mask/MSE/RMSE/MAE/bias,95%-dækning/width/intervalscore. Ingen
   model- eller båndtuning på den allerede sete udviklingsugeFeb7.
8. Kode/syntetiske tests/agg.resultater merges under Johans tilladelse;
   raw fits/states/errorbank/scoredseries/figurer/pakke forbliver lokale.

Synthetic referencechecks: constant-U equilibrium, ∫R=ΔF, half-hour split
composition, native Euler convergence, independent boundedLS solver,
known-law recovery, own snapshot/cutoff/refit/future-demand independence,
shared baseline/scoring mask/raw variance scaling. En test fandt forkert
koefficientrækkefølge i den første augmented matrix; det er rettet før
sciencekørsel. Ingen faktisk WM-fit er genkørt for at skjule dette.

Resultatet er **conditional shared-baseline development comparison**. Det
beviser ikke generel modelrangering, blind holdout, fysisk enhed,95%kalibrering
eller prisstyringsgevinst. Labels/enhed er stadig brugerens working assumptions.

## Nødvendig eksportkorrektion efter første sammenligning

WM-fit941f4f1 er gennemført én gang,2.527086s/148calls, og dets parametre,
historiske states og fejlbank bevares. Første forecast/scoring bevares også.
Sammenligningsadapteren brugte literal .025 ved lower empiricalquantile,
mens G's eksisterende companion bruger alpha=1−.95,alpha/2. Ved720 samples
landede afrundingen på hver sin side af en empiricalCDF-atomgrænse og gav
en forskel på én rang. Puremeans, baseline, fit og fejlbank er uændrede.

Før sidste eksport ændres **kun** add_empirical til G's eksisterende regel.
En ny regressionstest sammenholder G-bånd/mean med standalone-companion.
Ingen WM-refit: det allerede fitte snapshot migreres eksplicit lokalt efter
kontrol af præcis disse to kildeændringslinjer. Migrationjournal bevarer
original snapshotfingerprint, originale/nye sourcehashes og viser, at alle
andre snapshotfelter er bit-identiske. Strict load migrerer aldrig automatisk.
Én nødvendig ny input-only forecast/scoring gemmes i ny mappe; det er ikke
støjtuning, modelvalg eller budgetnulstilling. Originale outputtabeller er
stadig bevaret med deres oprindelige kildehashes. G's egne snapshots ændres ikke.
