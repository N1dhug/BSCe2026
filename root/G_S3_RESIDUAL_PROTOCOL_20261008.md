# G: afgrænset empirisk outputfejl, før faktisk kørsel

Pas startet8.oktober2026 kl.19:36:59 UTC (21:36:59 Copenhagen).
Johan har godkendt et enkelt residuallag, tydeligt mærket udviklingsmodel.
Det eksisterende G-fit, timeoperator, baseline og kontrol ændres ikke.

Modellen er Y_h=G_h+E_h. E_h har en fælles empirisk fordeling af de endelige
historiske replayfejl, inklusive deres middel. Det er model-/baseline-/måle-
residual samlet, ikke identificeret fysisk processtøj eller separat målefejl.
E_h antages iid alene som en enkel udviklingsmodel. Historisk lag1-korrelation
rapporteres; den kan vise, at iid er utilstrækkelig. Ingen AR/OU-model vælges
automatisk ud fra dette eller forecastugen. WM har additiv diffusion i R;
outputfejlen her er en særskilt statistisk approksimation, ikke WM's SDE.

1. Én historisk replay, max30s, med uændret interval-snapshot fra b2a6776
   og den samme lokale Jan1–Feb14 udviklingscase. Kun parentens fem uger
   førFeb7 og samme måltemaske bruges. Ingen optimizer eller baselinefit.
   Hver historisk dags targetbane starter fra den gemte mandagsfrosne
   weekdaymedian, så initialization svarer til den leverede forecastregel.
   Alle fem uger deltager allerede i parentfit; dette er **in-sample**.
2. Coarse/fine med parentens egne12/24 steps/h og starts; maxmeanerror<=1e−5.
   Støttebrud, afstand til fysisk støtte (RMSE lower bound for frossen model),
   baselineeffekt, residualmiddel/varians og fysisk tilstødende lag1-par gemmes.
   Missing timer/daglige resets må ikke tælles som tilstødende par.
3. Én input-only ugeforecast med empiriske95%-kvantiler (inverse empiricalCDF),
   gemt før observationsfilen åbnes. PureG, residualmiddel og observeret
   middel/varians/bånd er særskilte kolonner. Sigma_x=0; ingen sammenblanding
   med integrerede procesmomenter, ingen Gauss- eller fysisk clipping.
4. Derefter samme168 udviklingstimer scores med fælles rå MSE/RMSE/MAE/bias,
   bånddækning, bredde og intervalscore. Ingen modelvalg/biasfravalg/ændret
   kvantil på baggrund af slutugen. Det er en allerede set udviklingsuge,
   ikke en blind kalibreringsaccept eller kausal prisstyringstest.
5. Egne sourcehashes og exact parentfingerprint binder residualsnapshot.
   Rå fejl/replay/snapshots/forecast/figurer/pakke forbliver lokale. Kun kode,
   syntetiske tests og godkendte aggregerede resultater fra samme case merges.

Ingen ny model/baseline-/parameter-/sigma-søgning, bank/BM2/F-opgave eller
budgetnulstilling. Original otte-timers-ramme og tidligere charges bevares.
Næste accept kræver senere data, korrekt fysisk metadata og kontrol af
bias/stationaritet/seriel afhængighed; udviklingsbånd giver ikke denne accept.
