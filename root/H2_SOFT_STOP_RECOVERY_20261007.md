# F4-H2: kontrolleret tidligt blokstop

Blok 009 sluttede 7. oktober kl. 16:16:26 UTC med exit 124 uden sin normale
slutrapport. De 600 s forbliver reserveret og fuldt debiteret. Ingen gammel
charge eller reservation ændres. Der var 3.161 færdige råjob, 20 komplette
firestykkekald og 34 gemte timer i næste kandidat. Final-Y er stadig ulæst.

Råaudit består: alle jobchecksums, 69 frosne kilder, kronologiske TRAIN-
streams og præcis cached COBYQA-replay. Der er ingen gemte numeriske failures.
Checkpoint 09 bevarer hele tilstanden. `extension_report_009.json` er en
udtrykkeligt ekstern afbrydelsesrapport, ikke en opfundet controllerrapport.
Den oprindelige `extension_interruption_009.json` bevares også.

Kildegennemgangen viser, at worker-watchdogs kalder `os._exit(124)` ved
samme deadline, hvor hovedprocessen skal afbryde og lukke sin process-pool.
Et overlap ved pool-nedlukningen er en plausibel forklaring på den hårde
afbrydelse; den faktiske Python-stack er ikke tilgængelig. Ingen numerisk
fejl hævdes at være bevist fraværende i den afbrudte, ugemte operation.

Den oprindelige controller og alle numeriske kilder ændres ikke. Fremtidige
blokke startes med GNU timeout som ydre livscykluskontrol:

```sh
timeout --foreground --preserve-status --signal=INT --kill-after=75s 480s \
  python -u root/f4/run_h2_cap_extension.py \
  --run files/output/bm_1/runs/estimation/f4_h2_cap_extension_20261007 \
  --campaign files/output/bm_1/runs/estimation/f4_h2_same_day_campaign_20261007.json \
  --source-commit 4f4e35ae64cbbf6e1f75f8e10462baacbadf2561 --stage prefinal
```

480 s er en tidligere frivillig pause inden for den samme 600 s reservation.
Kun controlleren får SIGINT; dens eksisterende KeyboardInterrupt-håndtering
gemmer paused-status og lukker arbejdere før deres 540 s deadline. Et inert
parent/child-signalcheck verificerer denne signalafgrænsning og exitstatus;
det udfører ingen modelberegning og beviser ikke fravær af alle pool-races.
Efter hver blok kontrolleres procesexit, rapport, ledger og budget, og der
gemmes checkpoint før en udtrykkelig genoptagelse. Fejl/nyt kandidatloft
udløser ingen automatisk retry. Samme execution-identitet og 25/49-caps.

Denne kommando bruges kun, når mindst 600 s er tilbage i både det godkendte
kumulative budget og dagsfristen. Hvis mindre er tilbage, foretages ny
admissionkontrol; 480 s må ikke overskride en kortere reservation.

Native allowance belastes af det ene kontrolforsøg. Checkpoints og produktion
debiteres som hidtil; 18 h og stop 23:55 Copenhagen er uændrede. F/G/WM og
historiske negative resultater holdes adskilt. PR106 forbliver draft.
