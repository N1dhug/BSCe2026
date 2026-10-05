# G4: fem foregående matchende ugedage, tre prognoseuger

## Resultat og status, 6. oktober 2026

Johans bestilling er gennemført på 7.–27. februar 2022: 21 daglige prognoser,
tre fulde mandag–søndag-uger og 504/504 målte demand-timer. De to modeller bruger
samme data, baseline, kurvefamilie, k, startpolitik, skalering og numerik.
Kun straffen varierer: lambda=0 eller 0.001. **Samlet laveste prognose-MSE fås
uden straf, lambda=0.** Det er et udviklingsvalg for denne dagligt opdaterede protokol.
Den tidligere januarpilot med én fast ugeprognose bevares uændret.

| Prognoseperiode | Baseline O MSE | G4 uden straf MSE | G4 med straf MSE | Strafgevinst mod uden |
| --- | ---: | ---: | ---: | ---: |
| 2022-02-07–2022-02-14 ekskl. | 0.1414454676 | 0.1400247363 | 0.1390401161 | 0.7032 % |
| 2022-02-14–2022-02-21 ekskl. | 0.1118341593 | 0.1036472936 | 0.1045548688 | -0.8756 % |
| 2022-02-21–2022-02-28 ekskl. | 0.2582457058 | 0.2480953017 | 0.2562090330 | -3.2704 % |
| Alle 504 timer | 0.1705084443 | 0.1639224439 | 0.1666013393 | -1.6342 % |

Positiv strafgevinst betyder lavere MSE med straf; negativ betyder højere MSE.
MSE er i **rå workbook-enheder i anden potens**. En fysisk kWh-/populationsfortolkning
er ikke etableret af dette studie. Samlet vinder straf på 10 af 21 dage
og 1 af tre uger. Modelvalget følger samlet forecast-MSE,
ikke TRAIN-score eller den gamle januarpilots 1 %-simplicitetsregel.
MAE, RMSE, bias, alle dagsresultater, parametre og numerik står i
`files/output/bm_1/benchmarks/g4_weekday_development_20261006.json`.

## Hvad WM-koden faktisk gør

Reviewet bruger `src/exe/run_model/wm_4/predict_daily.py` og `fit_daily.py`
fra main `c39c57b44aa9e7c83657140a2080010c066047b5`:

- `fit_history`: vælger t−7, t−14, t−21, t−28 og t−35 dage, hver med 24 timers
  opvarmning; `qp_params.fit_many` fitter ét fælles WM-parametervektor på disse
  fem selvstændige forløb. Dagsparametrene kan ændre sig ved næste forecastorigin.
- `fit_expected_output`: WM's `statistical_output.fit(frame,t)` bruger alle
  målte historiske I/Temp før t. Det er ikke alene fem mandage til baseline.
  Flaget `--baseline-weeks` styrer i praksis parameterhistorikken.
- `historical_prediction_state`: median af de fem simulerede [F,R]-tilstande
  ved afslutningen af opvarmningen, ikke median af deres sluttilstande.
- Standardforecast: frosne dagsparametre og denne historiske median gennem
  næste døgn med oplyst pris og temperatur. `--optimize-prediction-state`
  bruger target-day I til startfit og er alene en oracle-diagnose; den kopieres ikke.
- WM's `prediction_setup` kræver I i sin validering, men standardmodellen
  bruger det ikke til dynamikken. G-versionen kræver alene pris/temperatur
  til forecast og åbner target-demand til scoring efter den gemte prognose.

Dette er derfor **21 rolling-origin døgnprognoser**, ikke tre forecast på
168 timer lavet hver mandag. Tirsdagens historik indeholder allerede mandagens
observationer. En ægte prognose af hele kommende uge fra én cutoff er et
selvstændigt forsøg; dette studie afslutter ikke det forsøg.

## Matematisk overførsel til G4

Lad t være prognosedagens midnat og j=1,...,5. Historisk forløb j starter ved
t−7j dage−24 timer og varer 48 timer. I hvert forløb sætter vi x_j(0)=0.5,
simulerer 24 timers opvarmning og bruger kun den følgende dags **målte** demand
i parameterfittet. Ingen state føres over de seks kalenderdage mellem disse
træningsforløb. Hvert forløb har egne priser og egen temperatur/baseline.

G4-kernen er uændret, med fysisk tid i timer:

\[
\dot x=\frac{D-B}{C_{\rm hours}},\qquad
\delta=\tanh\!\left(\frac{k}{2}[f(x)+g_a(u)]\right),
\]
\[
D=B+\Xi\delta
\begin{cases}1-B,&\delta\ge0,\\ B,&\delta<0.\end{cases}
\]

Her er f(x)=1−2x og

\[
g_a(u)=\begin{cases}
1+2(a-1)u,&0\le u\le\tfrac12,\\
a-2(1+a)(u-\tfrac12),&\tfrac12<u\le1,
\end{cases}\qquad -1\le a\le1.
\]

Kurven er kontinuert og ikke-voksende med endpoints 1 og −1; a=0 giver den
lineære reference. Med B,x,u i [0,1], Xi i [0,1] og positive C,k er D i [0,1].
Ved x=0 er f+g >=0, så tilstanden drives indad; ved x=1 er f+g <=0.
Derfor bevarer den kontinuerte ligning [0,1]. RK4 kontrollerer sine stages;
ingen numerisk state-/pris-clipping skjuler en domænefejl.

For hver origin og hver strafstyrke fitter vi **ét fælles** (C,Xi,a) på alle
fem historiske måldage:

\[
\widehat\theta_{\lambda,t}=
\underset{C,\Xi,a}{\operatorname{argmin}}\;
\frac{1}{N_t}\sum_{j=1}^5\sum_{h=0}^{23}
m_{j,h}\,[D_{\theta,j}(24+h)-y_{j,h}]^2+\lambda a^2,
\quad N_t=\sum_{j,h}m_{j,h}.
\]

m markerer en målt time, og y er normaliseret med den faste demand-skala.
K er låst til den tidligere, præ-cutoff januarværdi
0.22688550696235388. Bounds er C_hours i [1,744], Xi i [0,1], a i [−1,1].
Januars C/Xi er startsæd, ikke frosne parametre. Hver styrke har a-starter
0 og 0.25. SciPy TRF arbejder i (log C,Xi,a) med samme eksisterende G4-Jacobian
og tolerancer. Resultatet er et lokalt bounded LS-fit, ikke et globalt optimum.

Efter fit beregnes prognosestarten fra hver kandidats **egne** fem forløb:

\[
\widehat x_t=\operatorname{median}_{j=1,\ldots,5}x_j(24).
\]

Denne state og kandidatens faste parametre bruges til næste 24 timer.
Det svarer til WM's median ved slutningen af opvarmningen, men BM's skjulte
x kan ikke sættes lig WM's observerede flowresidual. WM initialiserer
historisk [F0,R0]=[0,I0−O0]; BM bevarer neutral x0=0.5 og lader opvarmningen
forme tilstanden. Dette er en eksplicit state-approksimation og ingen
target-day stateestimation. Medianen antager, at de historiske starter er
relevante for næste tilsvarende ugedag; den antagelse vurderes via forecastfejl.

Straffen foretrækker a nær nul. Den forudsiger ikke selv noget:

\[
\int_0^1[g_a(u)-g_0(u)]^2\,du=\frac{a^2}{3}.
\]

Den er altså en kvadratisk præference for den lineære prisrespons. Det kan
reducere støjfølsomhed eller give bias. Om den hjælper demand-prognoser skal
afgøres på senere observationer. Lambda=0.001 er den tidligere erklærede
følsomhedsværdi; det er ikke en estimeret optimal strafstyrke.

## Data, grænser og afbrudt pilot

Originale input-/pris-workbooks er genfundet lokalt og deres Git-blob-SHA
verificeret mod repositoryet: `373f924089b3c2f603a8e88bff1018e06616bfa1`
og `36224ea1521a00d18614e221f41e8ee82bbc6f1e`.
MED-gennemsnittet bevarer den tidligere legacy-population: tilgængelige målere
per time. Det matcher januararkivets målte demand til 8.89e−16.
DMI har 1.392 rå, unikke timer 1. januar–28. februar ekskl., ingen huller
og ingen ændring mellem rå temperatur og studieinput. Der er derfor ingen
temperaturinterpolation over en forecastgrænse i dette datasæt.

Demand-skalaen fryses fra 1. januar–7. februar ekskl.:
min=0.4023076923076923, bredde=2.44. Baseline refittes med kun præ-origin
observationer, men hverken skala eller lambda vælges på target-demand.
Prisdomænet er deklareret til [0,2604.76001] DKK/MWh, med øvre grænse fra
samme kalibreringshistorik og en eksplicit nulprisgrænse. Den gamle positive
minimumspris 0.74 afviste ellers korrekte nulpriser 19. februar.
Ingen future maksimumsværdi, clipping eller efterfølgende dagsskalering bruges.
Domænet er kun egnet til dette ikke-negative prisforløb; negative/højere priser
kræver et andet foruddeklareret domæne, ikke skjult reskalering.

En første pilot med gammel skala og 200 residualkald/start blev standset
15. februar: begge strafstarter nåede loftet. Otte foreløbige dagsresultater
og den fejlede fitdag er bevaret som `aborted_pilot`; de indgår ikke i tabellen
og gav intet endeligt modelvalg. Det korrigerede studie bruger samme domæne
og samme eksisterende G4-budget for begge kandidater: 750 kald/start,
3.000 samlet pr. origin, 180 s pr. origin. Ingen tolerance blev løsnet.
Der er højst to starter og to styrker. De tre uge-shards bruger tre processer
med BLAS=1, fælles kalibreringscutoff og ingen delte mutable modelobjekter.

De fem hele missing-demand-dage 13.–17. januar bliver liggende i historikken,
men bidrager ikke til LS-tab. Derfor er N_t 96 eller 120; alle 504 targettimer
er målt. WM's oprindelige dagshelper afviser sådanne huller, så G bruger sin
eksisterende measured-only policy. Der erstattes ikke en manglende dag med
en sjette ugedag, og manglende demand bliver ikke kunstige fitobservationer.

## Kode og faktisk kontrol

Nye moduler:

- `src/method/bm_1_deterministic_weekday.py`: en lille fem-forløbsadapter til
  den eksisterende `fit_response`. Kun observationsvektorer sammenlægges;
  hvert ODE-forløb starter selvstændigt ved 0.5. Ingen kopieret RK4/optimizer.
- `src/exe/run_model/bm_1/deterministic_weekday.py`: expanding WM-baseline,
  frosne skalaer, fit, historisk median, døgnforecast, numerik og gemt forecast
  før optional scoring. Referencefittets manifest skal slutte før første origin.
- `tests/test_bm1_deterministic_weekday.py`: to fokuserede tests består,
  ca. 0.16 s. Den ene kontrollerer independent replay, warm-up-/missing-maske
  og antallet af forløb. Den anden ændrer al target/future demand til 1e9 og
  kontrollerer byteidentisk forecast med stuboptimizer. Den laver ikke et
  ekstra historisk fit. Ændrede Python-filer består syntakskontrol.

Kernerne `bm_1_deterministic_fit`, `bm_1_deterministic_response`,
`bm_1_response`, `bm_1_deterministic_data` og WM-baseline har samme
Git-blob-SHA i kørslen som ved main-reviewet. Original G3/G4/WM/F-kode,
snapshots og tidligere resultater er uændrede. Kildepinnings øvrige afhængigheder
og hashes bevares i `source_map.json` i råarkivet.

Alle 42 valgte kandidater er genafspillet med 60 og 120 deltrin/time, inklusive
deres fem historiske forløb, medianstate og targetforecast. Største forskel
i normaliseret demand er 1.349e-14; grænsen er uændret 1e−5.
Der er 84 optimizerstarter, 82 med optimizerkonvergens.
Hver styrke på hver dag har mindst én konvergeret kandidat; budgetstoppede
starter er bevaret og tæller ikke som konvergerede fits.
Samlet 7414 residualkald; fit-sekundsum 745.14 s.
Uge-shards varer 121.82, 309.05, 329.17 s (tre parallelle processer; summen er ikke
en måling af wall time). Tidligere afbrudt pilot: ca. 179.47 s fit-sekundsum.
Ingen fuld testpakke, ekstra WM-fit eller genfit af januarreference.

## Reproduktion og fortolkning

Efter checkout af leverancens branch:

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
python -m src.exe.run_model.bm_1.deterministic_weekday --raw-csv PATH/raw_inputs.csv --reference-fit PATH/fit.json --start 2022-02-07 --days 21 --seconds-per-day 180 --output files/output/bm_1/g4_weekday_development
```

`raw_inputs.csv` indeholder Datetime,Demand,Price,Temp i originale enheder.
`fit.json` og dets tilstødende `manifest.json` er den gamle januarreference.
Ét sekventielt 21-dages run bruger samme kalibreringscutoff som de tre
parallelle syvdagesruns. Ved sharding skal `--calibration-end 2022-02-07`
gives eksplicit på alle shards. Et nyt outputnavn kræves; eksisterende
forsøg overskrives ikke. Ingen lokal kørsel er nødvendig for dette resultat.

Lavest udviklings-MSE giver en kandidat til næste forsøg. Disse uger er
eksponerede, korrelerede og dækker kun én vintermåned. Ingen p-værdi,
kausal priselasticitet, fysisk kapacitet eller generel helårsgevinst hævdes.
De oplyste fremtidige temperaturer/priser er realiserede inputs: dette er
betingede prognoser. Kalender og baseline kan stadig forklare det meste af
demand, og state/price-leddet har begrænset amplitude. Et stærkere fit kan
derfor ikke automatisk fange alle observerede demand-hop.

Næste prognosevalg skal bruge efterfølgende ueksponerede perioder med frosset
protokol. Hvis målet bliver en hel uge fra én origin, skal samme sammenligning
gentages med 168 timer uden observationer/refits undervejs. G6b→G8→G7 venter
ikke på en bred benchmark; det historiske G-datakontraktreview består særskilt.
