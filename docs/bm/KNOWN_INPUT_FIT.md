# BM1-fit med kendt demand, pris, baseline og X0

Denne indgang fitter kun C, k og Xi. f(X)=1-2X og g(u)=1-2u er faste.
Baseline og pris er kendte input, og X0 er en fast startværdi i [0,1].
Ét fælles parametersæt og én sammenhængende tilstandsbane bruges over perioden.
Der er ingen dagsreset, warmup eller stateopdatering med den observerede demand.

## Matematik

delta = tanh(k*(1-X-u)).
For delta >= 0: D = B + Xi*(1-B)*delta.
For delta < 0: D = B + Xi*B*delta.
dX/dt = (D-B)/C, X(0)=X0.

SciPy least_squares(method="trf", loss="linear") minimerer summen af
(modelleret demand - observeret demand)^2. C og k optimeres i logkoordinater.
Standardgrænserne følger den eksisterende G-kode: C i [1,744] timer,
k i [0.05,10], Xi i [0,1]. De kan ændres i Python-kaldet.
Fitteren er lokal; konvergens garanterer hverken et globalt minimum eller
entydige parametre. Eksempelvis kan C/k ikke bestemmes fra demand, hvis Xi=0.

Pris holdes konstant inden for hver time. Baseline interpoleres lineært
mellem de angivne støttepunkter. Den eksisterende kontrollerede RK4-simulator
genbruges. Ved timegennemsnit bruges
D_mean = B_mean + C*(X_end-X_start)/1 time.
Til sidst kontrolleres demand én gang med dobbelt så mange integrationstrin.
Kun konvergens med bestået numerisk kontrol får status "converged".

## Python

Kør fra repoets rod med projektets afhængigheder installeret:

```python
from src.method.bm_1_known_input_fit import fit_known_inputs

result = fit_known_inputs(
    baseline_time, baseline, prices, observed_demand, x0=0.23,
    initial=(12.0, 2.0, 0.5),
    observation_contract="hourly-mean",
)
print(result["parameters"], result["rmse"], result["status"])
```

prices og observed_demand har N værdier for N sammenhængende fysiske timer.
baseline_time og baseline dækker [0,N] og indeholder alle hele timegrænser;
finere støttepunkter er tilladt. Alle modelinput er normaliseret til [0,1].
Demand og baseline skal bruge samme skala. Tid måles i timer, X0 er dimensionsløs.
Manglende input og observationer afvises; der udføres ingen udfyldning.
Ved punktmålinger vælges observation_contract="hourly-point": demand ved
timebegyndelsen med den nye times pris. Det er et andet observationsmål.

Resultatet indeholder parametre, demand, state (N+1 værdier), residuals
(model minus observation), rmse, initial_rmse, optimizer og numerics.
RMSE fra Python-kaldet er i normaliserede demand-enheder.
max_nfev begrænser SciPys evalueringer; residual_calls tæller også
evalueringerne til numerisk differentiering. Der er ét startgæt pr. kald.
Et andet initial kan bruges i et særskilt kald ved behov.

## CSV og kommando

CSV har time_hours,baseline,price,demand. Hver række angiver en timebegyndelse.
Sidste række angiver periodens slutgrænse og kun baseline:

```csv
time_hours,baseline,price,demand
0,0.40,0.20,0.50
1,0.42,0.70,0.35
2,0.41,0.40,0.45
3,0.40,,
```

Tallene viser formatet. Mindst tre observationer kræves; brug en længere periode
med varieret pris for at give parametrene et bedre grundlag.
Baseline i CSV er værdier ved timegrænser, ikke timegennemsnit.
time_hours skal være sammenhængende fysiske timer; starten kan være forskellig fra 0.

```powershell
python -m src.method.bm_1_known_input_fit --input bm_inputs.csv --x0 0.23 --output bm-fit
```

Standardinput er allerede normaliseret. Rå enheder kan bruges med eksplicitte
skalaer: D_norm=(D_raw-demand_offset)/demand_scale, og tilsvarende for baseline;
u=(price_raw-price_offset)/price_scale. Eksempel:

```powershell
python -m src.method.bm_1_known_input_fit --input bm_inputs.csv --x0 0.23 --demand-offset 0 --demand-scale 1000 --price-offset -500 --price-scale 6500 --output bm-fit
```

Skalaerne er kendte input og estimeres ikke. Vælg dem så modelinput ligger i [0,1].
Demand, baseline og price klippes ikke. Modellen bruger forudbestemte skalaer,
og C_hours er en kapacitets-/tidsskala i denne normalisering.

Outputmappen skal være ny. fit.json indeholder parametre, fast X0, normalisering,
sluttilstand, RMSE, stopårsag og numerisk kontrol. trajectory.csv indeholder
observeret/fitted demand, baseline, pris, X og residualer i de angivne inputenheder.
Ved hourly-mean er outputbaseline et timegennemsnit. Den sidste række viser X
ved periodens slutning. Exitkode 0 betyder "converged"; 1 betyder et ufærdigt
eller numerisk utilstrækkeligt fit, hvis resultater stadig gemmes til inspektion.
