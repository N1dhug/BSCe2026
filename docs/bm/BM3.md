# BM3: bløde tilstandsgrænser

BM3 er en forsøgsmodel med samme logistiske demand-respons, baseline og historiske informationsgrundlag som BM1. Tilstanden X kan overskride 0 og 1. Jo større overskridelsen er, desto kraftigere bliver tilbageføringen. Demand D forbliver i [0,1]. Modellen er endnu ikke fysisk eller statistisk valideret som en forbedring af BM1.

## Implementeret matematik

Tid måles i timer. Ligningen er

```text
dX = [(D(X,B,U)-B)/C + R(X)] dt + sigma_x dW
R(X) = lambda * [max(-X,0)^2 - max(X-1,0)^2]
```

R er nul i [0,1], positiv under 0 og negativ over 1. Styrken `lambda` er positiv og har enheden 1/time. Standardværdien 10 er en eksplicit forsøgsindstilling, ikke et estimat fra data.

BM1's logistiske tilstandsrespons fortsættes konstant udenfor intervallet: f(X)=1 for X<0 og f(X)=-1 for X>1. Kun responsens værdi mættes; selve tilstanden klippes aldrig. Indenfor [0,1] er f uændret. Prisrespons g, tanh-kobling og demand-formel er de samme som i BM1:

```text
delta = tanh(0.5*k*(f(X)+g(U)))
D = B + Xi*delta*(1-B)  hvis delta >= 0
D = B + Xi*delta*B      ellers
```

Støjen er additiv med konstant amplitude sigma_x, også ved X=0 og X=1. BM1's amplitude er sigma_x*X*(1-X). De to sigma_x-værdier har derfor forskellig betydning og kan ikke flyttes direkte mellem modellerne.

Tilbageføringen ændrer energibalancen. I den deterministiske model er

```text
C*dX/dt = D-B + C*R(X).
```

`auxiliary_flow_mean` viser C*E[R(X)] separat. Dette bidrag er ikke inkluderet i den demand, der scores mod målingerne. Med støj tilkommer C*sigma_x*dW. BM1's identitet mellem timegennemsnit og C gange tilstandsændringen må derfor ikke bruges uændret udenfor intervallet. En fysisk fortolkning af denne ekstra energistrøm er et åbent modelspørgsmål.

## Genbrug af BM1

Den eksisterende least-squares fitter genbruges ved arv. Den samme fælles alpha, beta, k, Xi og C og fem historiske X0 estimeres med samme fitkontroller. X0-fit er fortsat begrænset til [0,1]; de stokastiske prognoser kan bevæge sig udenfor. Lambda fastholdes under fit.

Dagsprognosen genbruger BM1's baselinefit, dataforberedelse, fem matching weekdays, fitløkke og CSV/JSON-skrivning via modelhooks. MPC bruger samme fem forskudte historiske 24-timers vinduer, samme posterioropdatering fra I(origin), samme lead 1..8 og samme historiske variansregression. Den additive støj kræver et nyt enhedsstøj-kernel med variansbidrag dt i stedet for X²(1-X)²dt. Det er fortsat en lineær støjapproksimation og en procesvarians-proxy; strukturfejl kan bidrage til estimatet.

Fremtidig demand bruges først til scoring. Fremtidige priser og temperaturer er erklærede input, præcis som i BM1.

Når støjen er nul og de deterministiske tilstande er i [0,1], er BM3's dynamik den samme som BM1's. Derfor kan eksisterende historiske logistiske BM1-fits genbruges. Ugerunneren kontrollerer denne lighed numerisk før genbrug og bevarer reference- og inputidentiteter. En ikke-konvergeret BM1-fit mærkes fortsat som ikke-konvergeret; genbrug er ikke en ny optimering eller et konvergensbevis.

## Parallel mappe- og filstruktur

| Formål | BM1 | BM3 |
|---|---|---|
| Model | `src/model/bm_1.py` | `src/model/bm_3.py` |
| Fit og støjkalibrering | `src/params/bm_1.py` | `src/params/bm_3.py` |
| Kørsler | `src/exe/run_model/bm_1/` | `src/exe/run_model/bm_3/` |
| Modelkontrol | `src/validation/model/bm_1.py` | `src/validation/model/bm_3.py` |
| Dagsprognose | `files/output/bm_1/prediction/DATO/` | `files/output/bm_3/prediction/DATO/` |
| MPC | `files/output/bm_1/prediction/mpc/DATO/` | `files/output/bm_3/prediction/mpc/DATO/` |
| Studier | `files/output/bm_1/runs/NAVN/` | `files/output/bm_3/runs/NAVN/` |
| Tests | `tests/test_bm1_*.py` | `tests/test_bm3_pipeline.py` |
| Forklaring | `docs/bm/THOMAS_PIPELINE.md` | `docs/bm/BM3.md` |

Dagsprognosen skriver `hourly.csv`, `training_days.csv`, `iterations.csv`, `fit.json` og `forecast.png` som BM1. Derudover gemmes `distribution.npz`, `demand_heatmap.png` og `state_boundary.png`, når plotting er valgt. NPZ-filen gør det muligt at tegne den faktiske gemte fordeling uden et nyt fit.

MPC skriver `bm3_mpc_forecasts.csv`, `historical_diffusion.csv`, `skipped_origins.csv`, `rmse_coverage_by_lead.csv` og `settings.json`. Forecastkolonnen hedder `I_BM3` i stedet for `I_BM1`. RMSE, MAE, bias og 68/95%-dækning følger samme definitioner.

Ugerunneren følger BM1's jobstruktur:

```text
files/output/bm_3/runs/NAVN/DATO/logistic/daily/forecast/
files/output/bm_3/runs/NAVN/DATO/logistic/mpc/origin_HH/
```

Hvert job gemmer prognose-CSV, fit-JSON og iterationer. Roden indeholder `prepared/`, `logs/`, `settings.json`, `runtime.json`, `daily_summary.csv`, `mpc_summary.csv`, `week_summary.csv`, `mpc_by_lead.csv`, `fit_comparison.csv`, `comparison.csv` og `comparison.png`. Eksisterende BM1-resultater ændres aldrig. Detaljerede numeriske kontrolflag gemmes selv ved overskridelse af tolerancen; de er ikke automatisk godkendte resultater.

## Kørsel

Kør fra projektets rod med projektets eksisterende afhængigheder. Første version bruger logistiske f/g.

Dagsprognose med kendt additiv støjstyrke:

```powershell
python -B -m src.exe.run_model.bm_3.predict_daily --day 2022-05-02 --sigma-x 0.1 --restoring-strength 10 --score --save
```

Med `--sigma-x 0` har dagsprognosen kun usikkerhed fra de fem starttilstande, som BM1's standarddagsprognose. En forskel i den bløde grænse forventes ikke her, hvis alle baner bliver inde i intervallet.

MPC med historisk estimeret additiv støj:

```powershell
python -B -m src.exe.run_model.bm_3.predict_mpc --day 2022-05-02 --restoring-strength 10 --x-steps 601 --fp-max-step 0.125
```

Prøv kun to timeorigins ved at tilføje `--hours 0 12`.

Sammenlign på samme gemte uge med BM1-fits og forberedte input:

```powershell
python -B -m src.exe.run_model.bm_3.compare_week --workers 8 --mode both --state-min -6 --state-max 7 --x-steps 2601 --fp-max-step 0.125 --output files/output/bm_3/runs/logistic_week_2022-05-02_2022-05-08_wide
```

Standardreferencen er `files/output/bm_1/runs/fg_week_2022-05-02_2022-05-08`. Brug `--reference` ved en anden placering. Den skal have gemte, komplette logistiske jobs og `prepared/DATO.pkl`. Referencedatoer, forecastgrids, målinger og inputhashes skal matche. Checkpoints genbruges kun med uændrede kilder og indstillinger; brug en ny outputmappe ved ændring.

`--refit` udfører et nyt BM3-fit med BM1-parametrene som startgæt. Uden dette flag genbruges de kontrollerede historiske fits, hvilket sparer tid. `--joint-max-nfev` kan bruges sammen med `--refit` til den eksisterende fælles efteroptimering.

For at undersøge modstandens styrke køres samme kommando med f.eks. `--restoring-strength 1`, `10` eller `100` og forskellige outputmapper. Det sammenligner forsøgsindstillinger uden at tilpasse lambda til testugen. `--daily-sigma-x` åbner for støj også i ugesammenligningens dagsprognoser; dette ændrer usikkerhedsmodellen i forhold til BM1's gemte standarddagsprognose og skal fremgå af fortolkningen.

## Numerik og kontrol

Fokker–Planck-solveren genbruges med en udvidet numerisk tilstandsakse, som som standard er [-1,2]. Denne afskæring er ikke en fysisk grænse. Der beregnes sandsynlighed i de yderste 2% af cellerne på begge sider. Hvis den samlede randmasse på et outputtidspunkt overstiger `--tail-tolerance` (standard 1e-4), afvises prognosen med besked om at udvide området. Brug `--state-min` og `--state-max` og flere celler for at fastholde opløsningen. En lav randmasse er en nødvendig kontrol af afskæringen, ikke et generelt bevis for fuld numerisk nøjagtighed.

Ugerunneren sammenligner prognosen med dobbelt antal celler og halvt tidsskridt. `numeric_forecast_difference` og `numerically_passed` viser resultatet med standardtolerance 1e-4 i normaliseret demand. Sandsynlighedsbevarelse, positiv masse, krydsning af de fysiske grænser, tilbageføringens styrke, kausalitet og outputfiler kontrolleres i `tests/test_bm3_pipeline.py`.

Den fælles responskode anvender nu tre små geometri-checks i `src/validation/params/bm_1_response.py`. Det reparerer dens afhængighed af den arkiverede `src.validation.estimation` uden at gendanne den gamle pipeline.

## Gennemført prøve, 12. oktober 2026

Alle 175 jobs for 2.-8. maj 2022 blev gennemført med 8 workers, BLAS=1, lambda=10, numerisk område [-6,7], 2601 celler og maksimalt tidsskridt 0,125 time. BM1's gemte logistiske fits blev genbrugt efter kontrol af den deterministiske historiske lighed. Dagsprognoserne brugte sigma_x=0; MPC estimerede BM3's additive støj på de historiske vinduer. Tiden var 47,29 sekunder inklusive numerisk forfining og eksport i denne kørsel. Det er ikke et tidsmål for nye parameterfits.

| Metode | BM1 RMSE | BM3 RMSE | Baseline RMSE | Observationer |
|---|---:|---:|---:|---:|
| 24-timers dagsprognose | 0,10532540 | 0,10532221 | 0,10106988 | 168 |
| MPC, alle lead 1..8 timer | 0,09684165 | 0,09684567 | 0,10192142 | 1344 |

Dette viser ingen klar prognosegevinst ved BM3. Dagsprognosens meget lille forskel skyldes numerik; tilbageføringen er nul på dens indre deterministiske baner. For MPC var middelværdien af P(X<0 eller X>1) over prognosetidspunkterne 0,1163. Det er sandsynlighed ved de enkelte tidspunkter, ikke sandsynligheden for nogensinde at krydse en grænse i hele prognoseperioden. MPC's glatte prior kan også have masse udenfor [0,1] allerede ved start.

Alle numeriske randkontroller bestod; største yderkantmasse var 2,59e-25. Området [-1,2] var utilstrækkeligt ved to origins med stor estimeret støj, hvilket udløste kontrolfejl før udvidelsen. Dette er en numerisk afskæring, ikke en ændring af de bløde niveauer 0 og 1.

Alle 7 dagsprognoser og 89/168 MPC-prognoser bestod forfiningskontrollen ved tolerance 1e-4. Største MPC-prognoseforskel ved forfining var 0,007623. De øvrige 79 MPC-prognoser er gemt med `numerically_passed=False`. De samlede RMSE-tal er derfor foreløbige forsøgsresultater, og deres lille indbyrdes forskel er ikke grundlag for modelvalg. Kun 1/175 genbrugte referencefits havde sit oprindelige konvergensflag sat; de øvrige fitbegrænsninger bevares og skjules ikke.

33 relevante kontroller består: BM3, BM1's aktive pipeline, fælles joint-fit, responsfamilier, kendt-input-fit og BM2. En ekstra samlet discovery over den lokale kopis 70 tests var ikke grøn: to eksisterende Windows-checkpointtests ramte adgangsfejl i dette sandboxmiljø, og tre lokale, ikke-versionerede WM-testfiler henviste til moduler/funktioner, som ikke fandtes i kildekopien. De filer og WM-koden er ikke ændret eller inkluderet i BM3-ændringen.
