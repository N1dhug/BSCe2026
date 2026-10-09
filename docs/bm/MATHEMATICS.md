# BM-G: matematisk grundlag

## Fælles demandbalance

Tilstanden er BM's fyldningsgrad $X\in[0,1]$, prisen $U\in[0,1]$, baseline
$B\in[0,1]$ og relativ fleksibilitet $\Xi\in[0,1]$. Definér

$$\delta(x,u)=\tanh\!\left(\frac{k[f(x)+g(u)]}{2}\right),\qquad
D=B+\Xi\delta\begin{cases}1-B,&\delta\geq0,\\ B,&\delta<0.\end{cases}$$

Den normaliserede energibalance er $dX=(D-B)dt/C$. Den giver $0\leq D\leq1$.
Med ikke-voksende responskurver, endpoints $f(0)=g(0)=1$ og
$f(1)=g(1)=-1$ peger kontinuert drift indad ved begge tilstandsgrænser.
Den fælles implementation er `bm.D`, `bm.dXdt` og `bm.l` i
[base/bm.py](../../src/model/base/bm.py); den korte modelindgang er
[bm_1.py](../../src/model/bm_1.py). Begge er også WM-afhængigheder og beskyttede.
BM's $X$ er ikke WM's $F$ eller $R$.


## G: deterministisk drift, timeoperator og empirisk outputlag

G bruger tid i **timer** og $f(x)=1-2x$. Den valgte $g$ er lineær mellem
$(0,1),(.5,a),(1,-1)$. `BM1ResponseInputs.solve` i
[deterministic_response](../../src/method/bm_1_deterministic_response.py)
integrerer $dX/dh=(D-B)/C$ med kontrolleret RK4; den afviser domænebrud og
klipper ikke tilstanden. For én fysisk time giver energibalancen

$$\overline D_j=\overline B_j+C(X_{j+1}-X_j).$$

For en generel intervallængde skal sidste led divideres med længden.
`point_demand` bevares som en særskilt størrelse. Baselinefeatures integreres
med 60 trapezdelintervaller per time i
[interval_baseline](../../src/method/bm_1_s3_interval_baseline.py).
Fit bruger integrerede lineære features; deployment klipper baseline.
Disse to operatorer er derfor ikke identiske. Rank 20 for 49 features i alle
42 gemte fits er en konkret identificerbarhedsbegrænsning, ikke løst af ridge.

[weekday_rolling](../../src/method/g_weekday_rolling.py) fitter syv baselines
og parametersæt til fem foregående ens ugedage. Skalaen bruger kun femugershistorik.
Alle syv fit fryses ved mandagsorigin. Hver dags G-start er medianen af egne
historiske warmup-sluttilstande: prognosen består af **syv dagsbaner**, ikke én
sammenhængende 168-timers batteribane. Saved export i
[weekday_export](../../src/method/g_weekday_export.py) genbruger uændrede fits
med verificeret 60/120-forfining og egne historiske residualer.

Empiriske fejl $E=Y-\widehat D$ findes ved replay med deployment-startpolitik.
Outputmiddel forskydes med gennemsnittet af $E$; variansen bruger `ddof=0`.
Det er empirisk modelfejl på historiske data, ikke identificeret målerstøj,
parameterposterior eller fysisk diffusion. Det adskilte procesmomentlag findes
stadig i [interval_process](../../src/method/bm_1_s3_interval_process.py).

Frosne bånd bruger $z=|Y-m|/s$, $s>0$, og ordensstatistikken
$q=z_{(\lceil(n+1)0.95\rceil)}$. Senere bånd er $m\pm qs$.
`finite_rank` og `band_metrics` i
[prequential_bands](../../src/method/g_prequential_bands.py) genbruges;
ændret bredde ændrer ikke modellens predictive variance. Den stærke serielle
afhængighed og eksponerede udviklingsperioder giver ingen universel 95%-garanti.

Importkæden er `bm_1_g → g_weekday_export → g_weekday_rolling →`
`bm_1_s3_interval → bm_1_s3_forecast / deterministic_*` samt
`bm_1_s3_residual` og WM's uændrede `output_model.statistical_output`.
Punktmodulet er en genbrugt afhængighed, selv om punktprotokollen er historisk.
Der tilføjes ingen ny kalender eller numerisk model bag den korte kommando.

Relevante tests: [interval](../../tests/test_bm1_s3_interval.py),
[residual](../../tests/test_bm1_s3_residual.py),
[weekday/bands](../../tests/test_g_weekday_bands.py) og
[fælles kalender](../../tests/test_weekday_backtest.py). De forbinder timebalance,
forfining, rå skalering, nulproces, frysning, DST, kalibreringsrank og targetisolering.

## WM-reference og fælles evalueringsgrundlag

WM4's tilstand er $F'=R$, $R'=r_FF+r_UU+r_RR$ og $I=B+R$, tid i døgn.
Den oprindelige weekly-CLI bruger punktoutput og sit eget QP-fit. BM's særskilte
WM-timeadapter integrerer de samme uændrede ligninger med matrixeksponential
og free-run variable projection. For en time er $\overline I-\overline B=24\Delta F$.
Adapterens eget fit/resultat må ikke kaldes det oprindelige WM4-resultat.
`--optimize-prediction-state` bruger fremtidig demand og er oracle-diagnostik.

Demand-enhed, tidszone og hour-start er endnu **arbejdsantagelser**.
Hvis kilden er kWh pr. time, er tallet lig timegennemsnittet i kW; dette
verificerer hverken kildens enhed eller label. Ingen reskalering sker i oprydningen.
G og WM har forskellige fit- og observationsprotokoller.
