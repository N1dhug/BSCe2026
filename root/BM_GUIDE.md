# G: brug af den deterministiske grundmodel

Main indeholder G's deterministiske grundmodel samt WM og nødvendige fælles moduler.
Den læsbare matematik står i [grundmodellen](../docs/bm/G_INTERVAL_MODEL.md).

Installer projektets afhængigheder med `python -m pip install -r requirements-bm.txt`.
Kør kommandoerne fra projektets rod, og brug en ny outputmappe hver gang.

## Fit

```powershell
python -m src.exe.run_model.bm_1_g fit --raw-csv historik.csv --origin 2022-03-28 --demand-label hour_start --output g-fit
```

CSV skal have Datetime, Price, Temp og Demand. Angiv målerpopulation og
verificeret enhed med `--population` og `--demand-unit`, når disse er kendt.
Der bruges fem tidligere uger, egen opvarmning og syv gemte ugedagsstarter.
Parametrene er C, k, Xi og a. `--seconds` og `--max-calls` begrænser fitbudgettet.
Kun et færdigt fit producerer et brugbart `interval_snapshot.json`.

## Prognose fra eget gemt fit

```powershell
python -m src.exe.run_model.bm_1_g predict --snapshot g-fit/interval_snapshot.json --inputs-csv prognoseinput.csv --output g-prognose
```

Inputs indeholder Datetime, Price og Temp for prognoseperioden.
Forecast og baseline gemmes som timegennemsnit. Et billede viser forbrug og pris.
Eventuelle målinger læses først bagefter, via `--observations-csv`, til scoring.
Fremtidig pris og temperatur er erklærede scenarieinput.

Der er ingen parameterbank, Bayes-filter, ekstra kurvebasis, processtøj,
residualmodel eller usikkerhedsbånd i denne G-indgang.
Gamle snapshots og kildebundne checkpoints bruges med deres egne arkivkilder.

G-udvidelser: [arkivbranch](https://github.com/N1dhug/BSCe2026/tree/archive/g-udvidelser-20261009).
F: [F-arkiv](https://github.com/N1dhug/BSCe2026/tree/archive/f-main-20261009).
Status: [STATUS.txt](../docs/bm/STATUS.txt).
