# BM på virkelige data, med WM4-lignende output

Dette script kører **kun BM**. Det bruger `input.xlsx` og `prices.xlsx` fra dit
BSCe2026-repository. Ingen WM4-model importeres, kalibreres eller køres. Der
genereres ingen kunstige målinger.

Det er et foreløbigt bud på F4-H/I/J-output, ikke en færdig eller valideret F4.
Scriptet kan ikke forudsige den endelige models præcision.

## Kør fra testbranchen i PowerShell

Kør følgende fra din eksisterende BSCe2026-mappe. Der oprettes en separat
worktree, så din nuværende branch og eventuelle beregninger bevares:

```powershell
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch fejlede." }

git worktree add -b test-bm-wm4-output ../BSCe2026-bm-wm4-output origin/preview/bm-real-wm4-output
if ($LASTEXITCODE -ne 0) { throw "Preview-mappen kunne ikke oprettes." }

Set-Location ../BSCe2026-bm-wm4-output

python -B bm_real_wm4_preview.py --repo . --start auto
if ($LASTEXITCODE -ne 0) { throw "BM-preview fejlede; læs fejlbeskeden ovenfor." }
```

Brug det Python-miljø, du normalt bruger til projektet. Der kræves numpy,
scipy, pandas, matplotlib, openpyxl og tzdata på Windows. Ved manglende pakker:

```powershell
python -m pip install numpy scipy pandas matplotlib openpyxl tzdata
```

Kørslen skriver en ny resultatmappe og viser dens fulde placering. Du skal
ikke opdatere eller skifte branch i checkoutet til en igangværende G2a-kørsel.
Hvis worktree-mappen eller den lokale branch allerede findes, stopper
kommandoerne; de sletter ikke dine eksisterende filer.

## Uge og historik

`--start auto` finder den første komplette mandag–søndag med fem tidligere
forekomster af samme ugedag/time, også for den foregående kalibreringsuge.
Den bruger syv kalibreringsdage og 24 timers opvarmning før prognosen. Med
komplet historik fra 1. januar 2022 er den første mulige uge **14.–20. februar
2022**. Den kan ligge senere, hvis der mangler originale målinger. Scriptet
skriver de faktisk valgte datoer. Autosøgningen stopper før 1. marts; den
scanner ikke resten af sæsonen for at finde en uge med gode resultater.

For at vælge en bestemt prognoseuge:

```powershell
python -B bm_real_wm4_preview.py --repo . --start 2022-02-14
```

Dette er prognoseugens mandag. WM4's `predict_weekly --week` angiver i stedet
**kalibreringsugens** mandag og prognosticerer den efterfølgende uge. En senere
WM4-kørsel for samme målperiode skal altså bruge datoen syv dage tidligere.
Der medfølger ingen WM4-kørsel.

Januar kan ikke prognosticeres med denne historikregel, hvis workbooken kun
indeholder 2022. Manglende 2021-data erstattes ikke af kunstige værdier.

## Hvad kommer ud?

Resultaterne placeres i en ny mappe under
`files/output/bm_1/preview_wm4/<uge>_<kørselstid>/`.

| Fil | Indhold |
| --- | --- |
| `weekly_prediction.png` | Tre paneler som WM4: målt/prognosticeret forbrug og baseline; pris/fleksibilitetsændring; modeltilstand/fleksibilitet. |
| `demand_original_units.png` | Forbrugsgraf i de oprindelige workbook-enheder til sammenligning mellem modeller. |
| `hourly_predictions.csv` | 168 tidsstempler med faktiske målinger, prognose, baseline, pris, BM-tilstand og fleksibilitet. |
| `fine_predictions.csv` | Samme prognose på en finere intern tidsakse, som i WM4. |
| `weekly_metrics.csv` | MSE, RMSE, MAE og bias for BM og baseline, på både normaliseret og oprindelig skala. |
| `daily_metrics.csv` | De samme fejlmål pr. dag. |
| `density_and_response_curves.png` | BM-tilstandsfordeling og estimerede fælles, stykvise f/g-kurver. |
| `run_metadata.json` | Datoer, datafiler/hash, kolonner, skalaer, alle parameterhypoteser og forbehold. |
| `shared_scales.json` | Fast normalisering, som kan genbruges i en sammenligning. |
| `posterior_and_density.npz` | Parameter-vægte og start-/slutfordelinger til reproduktion. |
| `training_inputs.csv` | De konkrete historiske målinger, baselines og priser brugt ved kalibrering. |

Bias defineres ligesom i WM4: gennemsnittet af **prognose minus måling**.
MSE er middelværdien af de kvadrerede fejl; RMSE er kvadratroden af MSE; MAE
er middelværdien af de absolutte fejl. Daglige mål beregnes på 24 timepunkter,
ugentlige på 168. De fine modelpunkter tæller ikke som ekstra observationer.

## Samme data og sammenlignelige tal

Forbrugsserien er præcis WM4's definition: rækkegennemsnit af de tilgængelige,
oprindelige `MED`-kolonner. Antallet af tilgængelige kolonner eksporteres pr.
time. Priserne er originale DK1-priser i DKK fra samme workbook; udfyldte
priser anvendes ikke. Prognoseugens forbrug bruges først til evaluering,
efter BM-prognosen er beregnet. Den indgår ikke i parameterestimation,
baselinen, initialtilstanden eller fremadpropageringen.

**Normalisering er den væsentlige forskel:** den gennemgåede WM4-kode bruger
min/max fra hele januar–oktober til forbrug og fra hele 2022 til pris. Dette
preview fryser som standard min/max fra 1. januar og frem til prognosestart,
uden at inddrage målugen eller senere forbrugsdata. Prisintervallet inkluderer
også nul efter en fast regel, så en senere nulpris kan repræsenteres uden
at bruge fremtidige data til at fastlægge skalaen. Pris vises i den signerede
WM4-lignende koordinat [-1,1], mens BM internt bruger [0,1].

Derfor skal du sammenligne **oprindelige enheder** eller bruge præcis samme
faste skalaer i begge modeller. Samme målinger og samme metric-formler giver
ikke automatisk samme normaliserede skala. En tilstandsvariabel i BM er heller
ikke fysisk identisk med en tilstandsvariabel i WM4.

Hvis du allerede har sammenligningens faste normalisering, kan du levere den
uden at køre WM4:

```json
{
  "consumption_min": 0,
  "consumption_max": 1,
  "price_min": 0,
  "price_max": 1
}
```

Tallene ovenfor viser kun filstrukturen; **erstat dem med jeres faktiske
skalaer**, og gem filen som `comparison_scales.json`.

```powershell
python -B bm_real_wm4_preview.py --repo . --start auto --scales-json .\comparison_scales.json
```

Baseline og pris skal ligge inden for den faste modelskala. Hvis en målpris
falder udenfor, stopper scriptet med de konkrete grænser. Det klipper ikke
prisen og udleder ikke nye grænser af målugen. Målugens forbrug må gerne ligge
uden for den normaliserede træningsskala; det evalueres uden klipning.

## Hvad dette preview antager

BM1 bruger `delta = tanh(k/2 * (f(X) + g(u)))`,
`D = B + Xi * delta * (1-B hvis delta >= 0, ellers B)` og
`dX = rho * (D-B) dt + sigma_x X(1-X) dW`, med tid i dage.

Der kalibreres en lille fælles bank over rho, Xi, sigma_x og monotone, stykvist
lineære f/g-kurver med knuder ved 0, .25, .5, .75 og 1. Kurverne har faste
endepunkter +1 og -1. De estimeres samlet over kalibreringsugen og genbruges
uændret på alle prognosedage. Det er en grov demonstration af den planlagte
fleksible f/g-estimation, ikke den fulde fremtidige estimationsmetode. Med
`--fg linear` kan du bruge de oprindelige lineære referencekurver.

Modellen bruger en konservativ finite-volume-fordeling på [0,1] og fryser alle
parameter-vægte ved prognosestart. Der anvendes ingen efterfølgende
assimilation af målinger. `sigma_y` er en erklæret arbejdsantagelse på .06
normaliserede enheder, som kan ændres med `--sigma-y`; den er ikke estimeret
fra målugen. Den Gaussiske likelihood er foreløbig: den endelige sondring
mellem punktmåling, intervalmiddel og energi er ikke valideret her.

Baseline B er middelværdien af fem tidligere tilsvarende ugedage/timepunkter,
som i F4's historiske reference. Den er ikke WM4's vejrafhængige O-regression.
Fremtidige faktiske spotpriser behandles som kendte input, ligesom i WM4's
retrospektive prognosekode. Dette er ikke en forudsigelse af spotpriser en
uge frem. Previewet laver heller ingen prisoptimering, da det ville ændre
inputtene i sammenligningen.

Det viste 90%-bånd er et **foreløbigt latent modelbånd**, ikke et dokumenteret
90%-interval for målte observationer. BM's X og ændringen af E[D-B] må ikke
tolkes som WM4's F og R_dot. Ændringen beregnes inden for hver time og
udelader spring i baseline/pris ved timeskift. DST-uger afvises eksplicit.

## Kontrol før levering

Koden er kontrolleret med et separat kunstigt testdatasæt: workbook-import,
autovalg af uge, alle outputfiler, metric-formler, sandsynlighedsmasse og
uafhængighed af prognoseugens målinger. Testdata medfølger ikke som resultater.
Den komplette virkelige `input.xlsx` var ikke tilgængelig som lokale bytes
her, så den faktiske BM-kørsel og de faktiske tal beregnes på din computer.
WM4 er ikke kørt under kontrollen.
