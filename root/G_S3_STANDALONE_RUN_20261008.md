# S3-grundversion: registreret historisk udviklingseksempel

8. oktober 2026. Denne kørsel er en del af den autoriserede praktiske
G-afslutning. F, gamle G-ledgers og tidligere underkendte G6a/G7-forsøg er
ikke berørt. Registreringen skrives sammen med koden før modelkørslen.

- Origin: mandag 7. februar 2022 kl.00, lokal Europe/Copenhagen.
- Input: originale Git-workbooks, MED-gennemsnit af tilgængelige målte
  målere per time, DK1-priser og eksisterende DMI country mean_temp importer.
- Inputperiode: 1. januar til 14. februar, opdelt ved cutoff for temperatur-
  hentning. Ingen demand-imputation. Enheden og punkt/timeinterval-
  fortolkningen er endnu ikke fysisk verificeret; legacy punktapproksimation.
- Normalisering: 3. januar til 7. februar; demand-skala kun historisk,
  eksogene prisgrænser [-500,6000]. Baseline på målte data før cutoff.
- Model: kun S3 lambda=0; eksisterende to-start `fit_strategy` med default
  reference [12 h,2,.5,.5]. Ingen lambda-/modelvalg eller WM-fit.
- Fitloft inklusive nye gridforsøg: 600 s, 3000 residualkald, højst 750 per
  start. Grids 12/24/48 steps/h, hver med eget dobbelt grid. Ingen automatisk
  retry; hvis fit/forfining ikke afsluttes, bevares unfinished.
- Efter konvergens: gem selvstændigt S3-snapshot med hele baseline og egne
  coarse/fine dagsstarter. Genindlæs og forecast 7.–14. februar uden fit.
- Forecast-check: syv uafhængige dagsbaner; demandforfining <=1e-5 i
  normaliserede enheder; ingen klipning eller future demand i forecast.
- Eksportér forecast før observationsfil åbnes. Score målte timer med
  MSE/RMSE/MAE/bias på råskala og vis baseline/pris/observationer i figur.
- Forudsigelsens realiserede priser/temperaturer er betingede scenarieinputs.
  Denne allerede inspicerede periode er et udviklingseksempel, ikke blind test.

Input-/datakvittering og de fulde råfiler gemmes i arbejdsområdets
`G-S3-example-20261008`. Modeloutput får egne nye undermapper `fit` og
`predict`. Kilden er den efterfølgende commit af denne registrering og de
tre nye Python-filer; den eksakte commit og faktisk forbrug føjes til
resultatnotatet efter kørslen. Ingen accept/resultater er foregrebet her.

Seks nye fokuserede S3-interfacekontroller består (1,098 s). Første sandbox-
kørsel havde to atomiske filgemninger afvist i Windows' midlertidige mappe;
kontrollerne blev derefter kørt med tilladt rigtig fil-I/O og bestod.
Ingen numeriske kerner eller deres checks blev ændret for at omgå dette.
