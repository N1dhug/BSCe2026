# G-S3 timegennemsnit: protokol før faktisk fit og proceskørsel

Pas startet 18:41:05 UTC, 8. oktober 2026. Johan bad om fortsættelse efter
grundversionen og oplyste, at Demand dækker forbrug over en hel time og
efter hans erindring er Power. Kl.10 tolkes foreløbigt som kl.10–11 efter
hans antagelse; det er ikke verificerede kildemetadata. F holdes adskilt.

Den nye vej er et selvstændigt interval-snapshot. G's punktkode/snapshots
ændres ikke og migreres ikke skjult. S3 lambda=0, lineær f, to-stykkers g,
fælles C/k/Xi/a, 35 egne histories med 24 timers warm-up, egne syv medianstarts
og mandagsfrossen origin bevares. Eksogen WM-baseline genbruges som frosset
kontinuerlig inputmodel; det er ikke en ny baseline- eller F-undersøgelse.

## Fast kørsel

1. Én ny `fit_interval` på det eksisterende lokale Jan1–Feb14 input med
   origin Feb7 og demand-label hour_start. Kun komplette intervaller med
   end<=origin indgår. Maksimum 120 sekunder og 1000 residualkald samlet,
   750 per start, eksisterende 12/24/48 RK4-grids og demandforfining<=1e-5.
   Ingen punkt-fitparametre genmærkes som intervalfit. Ingen modeludvælgelse.
2. Én input-only deterministisk intervaluge, gemt før scoring på 168 timer.
   CSV indeholder point-readout særskilt fra hourly-average forecast og
   integrated baseline. UTC-start/end deklarerer de fysiske intervaller.
3. Én intervalprocesuge med eksplicit illustrations-sigma=.2 per sqrt(time),
   128/256 celler, trin<=1/24 og 1/48 time og 120 sekunders loft. Samme
   konservative FV-generator og implicit tridiagonal solve som punktvejen.
   Middel/varians af timeintegralet beregnes med tre joint state/rewardfelter.
   Kontroller normaliseret demand/state meanfejl<=.005 og variansfejl<=.002,
   masserror<=1e-10, finitte felter, ikke-negativ varians. Ingen klipning,
   observationstøj, normalisering eller efterfølgende sigmajustering.
   Kun hvis første grid numerisk afvises: én yderligere 256/512-forfining
   med trin<=1/48 og 1/96 og 120-sekunders loft. Ingen flere retries.
4. Én historisk intervalstøjdiagnose, højst60s. Matching-weekday raw sample-
   varians ddof=1, mindst to målte værdier/bin. Unitprofiler bruger joint
   lineær (state,integreret-output)-kovarians med positiv 3-node Gauss-
   støjkvadratur, P0=0 efterwarmup; hver timers outputaccumulator nulstilles.
   Time0 indgår nu: den første HELE time har processpredning selv ved P0=0.
   q>=0 varians-LS bruger alle24 timer. Historisk meanforfining<=1e-5,
   unit- og fitted profil-stdforfining<=1e-4. P>1/4 og målte gennemsnit uden
   for den frosne integrerede demandstøtte er hver sin afvisningsgrund.
   En kandidat kræver særskilt ikke-lineær validering; q bruges aldrig
   automatisk i den illustrative procesprognose.

Output er E[hourly-average D] og Var[hourly-average D], ikke middel/varians
af endpoint-D og ikke gennemsnittet af marginalvarianser. Intervalkvantiler
og dækning beregnes ikke fra kun to momenter; punktbånd genbruges ikke som
timebånd. Score råenheds MSE/RMSE/MAE/bias på samme finite måltemaske efter
forecastexport. Intet bredt støjfit, parameterbank eller nyt F-studie.

Kontroller før historisk kørsel omfatter snapshotroundtrip, point/mean
adskillelse, future-demand-uafhængighed, endlabel-cutoff, forårs-DST,
integreret Brownsk kovarians, uafhængig Van-Loan-reference, analytisk
telegraph-integral, konservativ FV og nulstøj/nulfleksibilitetsgrænser.
Rå input/resultattidsserier bevares lokale. Johans tilladelse til kode,
syntetiske tests og aggregerede resultater/merge gælder samme eksempel.
