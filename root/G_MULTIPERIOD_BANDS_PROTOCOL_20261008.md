# G: bestilte flere perioder og usikkerhedsbånd, fast protokol

Pas startet8.oktober2026 kl.20:32:13 UTC (22:32:13 Copenhagen). Johan bestilte
flere testperioder og usikkerhedsbånd efter estimat1–2h/2–4h. F, bank/BM2,
priskontrol og verificering af fysiske metadata er ikke dette pas. Samme
oprindelige8h-forløb bevares; tidligere reviewtid er umålt, ikke nulstillet.

## Kalender før nye resultater

- Kalibreringsforecastorigins: **2022-02-14,2022-02-21,2022-02-28**.
- Evalueringsforecastorigins: **2022-03-07,2022-03-14,2022-03-21**.
- Hver origin er mandag; forecast slutter næste mandag. Sidste uge har
  167 fysiske timer ved forårs-DST; resten168. Ingen erstatningsuger vælges
  efter resultater. Feb7's allerede sete eksempel er reference, ikke kalibrering.
- Januar/uger6–18 har tidligere været inspiceret. Dette er en ny protokolfast
  kronologisk udviklingsevaluering, **ikke et hidtil ulæst/blindt holdout**.

Samme MED mean over tilgængelige målte målere, ingen demandimputation,
hour_start som usikker arbejdsantagelse, råeffektenhed uverificeret, realised
historisk DK1 price/countrytemperature. Betinget scenarie, ikke weatherforecast.
Weather hentes/genbruges i separate history/weekchunks, så den eksisterende
imputation ikke bruger en forecastuges temperatur til at udfylde før dens cutoff.
Data-/cache-/workbookhashes og coveragekontrol gemmes lokalt før første fit.

## Fit og fælles information

Én G-intervalfit, ownresidualreplay og WM-hourlyfit per origin. Eksisterende
kerner/baselinefeatures/S3 lambda0/familier/starts/gates beholdes. G's
normaliseringsvindue er de fem foregående uger; baseline må bruge Jan1→origin
som i den eksisterende adapter. G har35 histories med24h-warmup. WM deler
præcis parentbaseline/skala/inputs/maske, og har sin allerede dokumenterede
exacthourly/free-run variable-projection fit uden warmup. Forskellen er åben.

- G120s/1000rawresidualcalls per origin, eksisterende12/24/48 grids og <=1e−5.
- G residualreplay30s per origin, nul nye optimizerkald.
- WM60s/400calls per origin, samme toseeds/bounds og convergencekrav.
- Samlet højst6*210=1260s for disse modeltrin, eksklusive download/I/O/tests.
- Ingen automatiske retries, længere caps, nye families/starts eller gate-
  ændringer. Failure/provisional/checkpoints bevares; ufuldstændige perioder
  må ikke omdøbes til komplette. Rollende refits er seks nye egne fits,
  ikke en omskrivning af de tidligere receipts eller af gamle G7-budgetter.

Alle forecasts gemmes med own hashes før targetobservationer åbnes for score.
Senere origins må bruge tidligere målte uger i almindelig kausal træning.
Fitting/rawfiles indeholder derfor ikke en streng common-training split.
Ingen hyperparameter/modelvalg styres af evalueringsresultaterne. De tre
evalforecastpakker fryses før samlet metricsrapport; deres egne target-Y
kan ikke påvirke egen forecast eller den frosne kalibreringsregel.

## Ét afgrænset forbedret båndforsøg

For hver model/origin er m den eksisterende mean plus egen in-sample empirical
meanerror; s er egen historical empirical residualSD i **råenheden**, kendt
ved origin. Ingen ny mean-/state-/sigma-model tilpasses. s skal være finite
og positiv; manglende skala afvises, ikke skjult floor/tuning.

På de tre tidligere kalibreringsforecastuger beregnes score
z=|Y−m|/s på finite measuredtimer. Separate pooled scorebanker for G/WM,
samme maske og regel. Kræv mindst300scores/model og alle tre perioder.
For nominal95% er k=ceil((n+1)*.95); q er k'te sorteret score. Ved k>n er
båndet uendeligt og ikke deployable; der vælges ikke en mindre rang.
Kalibreringsrecord fryses med cutoff **2022-03-07**, parentforecast/datahashes,
rank/scores/scale/fingerprint. Evaluerings-Y kan ikke bruges i denne record.

For alle tre evaluger er det nye bånd [m−q*s,m+q*s], med samme frosne q og
rollende originens egen previsible s. Ingen retuning på evalugerne. Det er
**prequential finite-rank calibration**, ikke eksakt split-conformal guarantee:
tidsafhængighed, modelrefits, sæson-/populationvariation kan bryde score-
exchangeability. God empirisk dækning på tre uger er heller ikke universel95%.
Seriel afhængighed rapporteres; der startes ingen AR/OU/sigma/bankundersøgelse.

Gamle empiricalbands bevares ved siden af nye calibrationbands. Mean/pureG/
WM/baseline ændres ikke. q er en båndmultiplikator, ikke diffusion eller ny
varians: intervalgrænser bestemmer ikke en fuld fordeling. Ingen Gaussian-
substitution, varianceinflation forklædt som exactmoments, fysisk clipping,
proces/residualdoublecount eller ændret kontrol. Negative rågrænser tælles.

Rapport: per origin/phase og pooled evaluation på fælles finite rawmask,
RMSE/MAE/bias,95%coverage,width,intervalscore for begge gamle/nye bånd.
Pool RMSE fra alle squared errors, ikke gennemsnit af ugentlige RMSE.
Coverage per weekday/day og helugefailurecount er descriptive; ingen95%
simultaneous weekcoverage eller smalle iidconfidenceintervals hævdes.

## Leverance og privat materiale

Kørbar periodedriver, tests, protokol, brugsvej og lokale forecasts/fits/
kalibreringsrecord/scorer/figurer/pakke. Kode/protokol/generisk vejledning må
sendes/merges under eksisterende tilladelse. Nye perioders numeriske resultater,
rå inputs/states/errorbanks og komplette pakker holdes lokale; tidligere
konkrete uploadtilladelse dækkede det gamle udviklingseksempels aggregater.
Ingen ny ekstern metricpublicering er nødvendig for at udføre bestillingen.
Rådata og synkroniserede sources ændres ikke.
