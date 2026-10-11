# G: brug af den deterministiske grundmodel

Main indeholder G's deterministiske grundmodel samt WM og n�dvendige f�lles moduler.
Den l�sbare matematik st�r i [grundmodellen](../docs/bm/G_INTERVAL_MODEL.md).

Installer projektets afh�ngigheder med `python -m pip install -r requirements-bm.txt`.
K�r kommandoerne fra projektets rod, og brug en ny outputmappe hver gang.

Fit til en kendt demand med kendt baseline og pris, hvor X0 estimeres sammen
med C/k/Xi (eller fastholdes, hvis den er kendt):
[enkel fitter med line�re f/g](../docs/bm/KNOWN_INPUT_FIT.md).

## Fit

```powershell
python -m src.exe.run_model.bm_1_g fit --raw-csv historik.csv --origin 2022-03-28 --demand-label hour_start --output g-fit
```

CSV skal have Datetime, Price, Temp og Demand. Angiv m�lerpopulation og
verificeret enhed med `--population` og `--demand-unit`, n�r disse er kendt.
Der bruges fem tidligere uger, egen opvarmning og syv gemte ugedagsstarter.
Parametrene er C, k, Xi og a. `--seconds` og `--max-calls` begr�nser fitbudgettet.
Kun et f�rdigt fit producerer et brugbart `interval_snapshot.json`.

## Prognose fra eget gemt fit

```powershell
python -m src.exe.run_model.bm_1_g predict --snapshot g-fit/interval_snapshot.json --inputs-csv prognoseinput.csv --output g-prognose
```

Inputs indeholder Datetime, Price og Temp for prognoseperioden.
Forecast og baseline gemmes som timegennemsnit. Et billede viser forbrug og pris.
Eventuelle m�linger l�ses f�rst bagefter, via `--observations-csv`, til scoring.
Fremtidig pris og temperatur er erkl�rede scenarieinput.

Der er ingen parameterbank, Bayes-filter, ekstra kurvebasis, processt�j,
residualmodel eller usikkerhedsb�nd i denne G-indgang.
Gamle snapshots og kildebundne checkpoints bruges med deres egne arkivkilder.

G-udvidelser: [arkivbranch](https://github.com/N1dhug/BSCe2026/tree/archive/g-udvidelser-20261009).
F: [F-arkiv](https://github.com/N1dhug/BSCe2026/tree/archive/f-main-20261009).
Seneste oprydning: [F-rester og ekstra indgange](https://github.com/N1dhug/BSCe2026/tree/archive/f-og-ekstra-grundmodel-20261009),
med komplette kilder fra main 4011c10 og tilh�rende tests.
Status: [STATUS.txt](../docs/bm/STATUS.txt).

## Thomas-opskrift p� main

Dagsprognose, timevis MPC, usikkerhed og resultatgrafer er beskrevet i
[THOMAS_PIPELINE.md](../docs/bm/THOMAS_PIPELINE.md).
Logistiske f/g kan ogs� v�lges i eksisterende fit med
`--fit-response --response-family logistic`.

Valgfri f�lles efteroptimering og uge-sammenligning med alle seks f/g-par:
[JOINT_FIT.md](../docs/bm/JOINT_FIT.md).

BM2 med tr�ghed, f�lles usikkerhed for to tilstande og parvis sammenligning
med de gemte BM1-prognoser: [BM2.md](../docs/bm/BM2.md).
