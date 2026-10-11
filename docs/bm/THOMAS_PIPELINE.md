# Thomas' BM-opskrift på main

Kilde: thomas_main, commit 7af6247f529261b6b45616561eb777d05b7cdfea.
Kildebranchen læses kun. Overførslen omfatter estimator, dagsprognose, MPC,
usikkerhed, plotting, fælles evaluering og nødvendige valideringsmoduler.
WM, fælles dataimportere og den eksisterende deterministiske BM-solver bevares.
src.tools.baseline genbruger WM's eksisterende statistical_output.

## Matematik og fit

f(X)=tanh(alpha*(0.5-X)/2)/tanh(alpha/4) og
g(u)=tanh(beta*(0.5-u)/2)/tanh(beta/4), alpha,beta i [0.05,20].
Begge går fra +1 til -1 og krydser nul ved 0.5.
delta=tanh(k*(f(X)+g(u))/2),
D=B+Xi*delta*((1-B) hvis delta>=0 ellers B),
dX/dt=(D-B)/C, med tid i timer.

Dagsopskriften bruger fem tidligere matchingugedage. Ét fælles
(alpha,beta,k,Xi,C) estimeres på de fem døgn ved alternerende LSQ;
hvert historisk døgn får sit eget X0. De fem tilstande bruges som ligevægtet
startfordeling for måldagen. Baseline for hvert træningsdøgn fittes før dette
døgn; baseline for måldagen fittes før måldagen.
Normalisering kalibreres før det ældste træningsdøgn og værdier uden for
referenceintervallet klippes som i kildekoden.

MPC genestimerer på fem historiske 24-timers vinduer ved hvert timeorigin,
kalibrerer sigma_x fra historiske residualer, opdaterer statefordelingen med
I(origin) og forudsiger lead 1..8. Procesloven bruger
sigma_x*X*(1-X); fordelingen udvikles med den eksisterende finite-volume-solver.
Dagsprognosens standard sigma_x=0; positiv sigma_x er brugerangivet.
MPC estimerer sigma_x selv. Variansfit er en lineær-støj-approksimation;
strukturel fejl er ikke adskilt fra observationsstøj.

Kildeopskriften scorer punktværdier ved timetimestamps. Hvis observationen
mærket 10:00 dækker 10-11, er den først tilgængelig efter intervallet;
origin-assimilering kræver derfor særskilt afklaring af målingens tidslabel.
Ingen sådan tidsforskydning indføres stiltiende ved denne overførsel.
Fremtidige temperaturer og priser er givne input i disse historiske kørsler.
Kurverne genestimeres ved nye fits; dette er ikke ét globalt kurvepar for
alle prognosestarter og ikke en kontinuerlig ugeprognose uden opdateringer.

## Kør en dagsprognose

Fra repoets rod på main:

~~~powershell
python -m src.exe.run_model.bm_1.predict_daily --day 2022-05-05 --score --save
python -m src.exe.run_model.bm_1.plot_prediction --day 2022-05-05
~~~

Input hentes fra files/input/data/input.xlsx, prices.xlsx og den eksisterende
DMI-cache/API via WM-importerne; gruppen er MED medmindre --tag ændres.
Resultater ligger i files/output/bm_1/prediction/2022-05-05:
hourly.csv, training_days.csv, iterations.csv, fit.json og grafer.

## Kør MPC

~~~powershell
$outputDir = "files/output/bm_1/runs/thomas_mpc_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python -m src.exe.run_model.bm_1.predict_mpc --day 2022-05-05 --output $outputDir
~~~

Dette er 24 separate fits og op til 8 leads per origin. Det er dyrere end én
dagsprognose. Fitstatus og optimizergrænser gemmes; et endeligt tal er ikke
automatisk et konvergeret fit. MPC bruger som standard mindst C=0.05 time,
mens dagsestimatorens grænse er 0.001. Ingen lange kampagner startes ved import.

## Evaluering

compare_model_predictions og compare_all_model_predictions reproducerer
kildens BM-dagprognose versus WM-MPC; deres informationsmængder er forskellige.
compare_mpc_predictions reproducerer sammenligningen ved samme origins/leads,
med checkpoints og Windows-låsehåndtering. Indgangene kan køres med --help.
MPC-resultater føres tilbage til WM's normaliserede skala før scoring.
Dagsopskriftens RMSE er på BM's referencebaserede skala og må ikke sammenlignes
direkte med RMSE i originale dataenheder. Kontroller også baseline-RMSE.

## Brug samme logistiske kurver i vores eksisterende kontinuerlige fit

~~~powershell
$outputDir = "files/output/bm_1/runs/fit_logistic_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python -m src.exe.run_model.bm_1.bm_1_known_input_fit --start 2022-01-01 --end 2022-02-05 --tag MED --fit-response --response-family logistic --output $outputDir
~~~

Dette bevarer vores hourly-mean-opskrift, kontinuerlige statebane, baseline,
kapacitetsgrænser og RMSE i originale enheder. Kun kurvefamilien ændres;
alpha/beta estimeres samlet med C/k/Xi/X0 uden strafled. Det er dermed
ikke en genetikettering af Thomas' punkt-/dagsfit.
--compare-responses medtager nu også logistic. Gemte logistiske kurver kan
rekonstrueres med response_from_record og bruges af eksisterende grafer.

## Korte kontroller

~~~powershell
python -B -m unittest tests.test_bm1_thomas_pipeline tests.test_mpc_checkpoint_windows tests.test_bm1_known_input_fit tests.test_bm1_deterministic_response tests.test_bm1_g_basic -q
~~~

Kontrollerne omfatter kurveparitet, isolering af måldemand, stateprognose,
gem/genindlæs/grafer, checkpoints og eksisterende deterministisk BM.
De bruger syntetiske data og starter ingen historisk benchmarkkampagne.
# Fælles efteroptimering

En valgfri fælles optimering af modelparametre og fem starttilstande samt en
parvis uge-sammenligning med gemte fits er beskrevet i [JOINT_FIT.md](JOINT_FIT.md).
