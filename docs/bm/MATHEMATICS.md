# BM-F/G: matematik, observationer og kode

Denne fil ejer de aktuelle modelantagelser og forbindelsen til kode/tests.
[Startguiden](README.md) ejer kommandoerne; [sammenligningen](COMPARISON.md)
ejer målinger og før/efter-tal. De oprindelige ligninger står i den beskyttede
[BM-reference](../../root/BM_REFERENCE.md). Historiske forsøg og negative
resultater findes via [arkivindekset](../../archive/bm/INDEX.md).

## Data og målergrupper

Workbookens installationsmetadata identificerer **13 MED-målere som huse med
elvarme og 140 UDEN-målere som huse uden elvarme**. De 164 kolonne-ID'er blev
matchet til metadata; de 13 medlemmer i F's gemte model matcher MED præcist.
Det verificerer ikke solcelleudstyr. Tredje rundes metadataaudit genbrugte den
tidligere audit og kontrollerede den mod samme workbook-SHA samt rå metadata;
ingen observationsværdier blev indlæst. Kvittering: `round3/f/metadata_receipt.json`
under recovery-roden i [status](STATUS.txt).

Fysisk enhed, demandtidszone, intervalstart/slut og netto-/bruttoforbrug er
fortsat **ukendte fra leverandørmetadata**. Kørsel antager lokale Copenhagen
hour-start-labels. Hvis målingen er kWh pr. fysisk time, er dens numeriske værdi
lig intervalmiddel i kW; dette er en betinget fortolkning, ikke verificeret metadata.
De historiske F-feltnavne med `kW`/`kWh` beholdes for kompatibilitet, med tydelig
unit_status. Der foretages ingen ny reskalering eller stiltiende tidskorrektion.

F bruger en fast komplet målerliste: én manglende måler maskerer hele timen.
G's gamle fit bruger middel over tilgængelige MED-målere. Det gemte G-fit har
træningsfingerprint og gruppelabel, men **ingen præcis målerliste eller rå maske**.
Eksport til en brugbar model bevarer derfor `legacy-unverified`; konvertering
kan ikke efterfølgende bevise historisk population. `--group` afviser skift til
en anden gruppe. F's model/config/run binder gruppe, målerrækkefølge, modelhash,
skala, baselinehash og information-cutoffs. Det gamle bærbare F-snapshot bærer
packetproveniens, men ikke en fuld eksplicit TRAIN-periode; den opfindes ikke.

Nye G-weekday-fit kræver en eksplicit MED-erklæring med unik målerliste,
tilgængelig-måler-aggregation, enheds-/scenarieantagelser og præcis CSV-SHA.
Den isolerede historik, træningsperioden, skalaer, syv baselines og begge
snapshots bindes i kvitteringer før forecastfreeze. Det beviser byteidentitet,
men ikke aggregatets rå medlemskab: status er `declared-unverified`.
Den beskyttede MED-fitkerne afviser UDEN i den nye indgang. Den gamle models
manglende medlemskab ommærkes ikke. Se [fitkontrakten](WEEKDAY_BACKTEST.md).

## Fælles normaliseret BM-balance

Tilstanden er fyldningsgrad $X\in[0,1]$, pris $U\in[0,1]$, baseline
$B\in[0,1]$ og relativ fleksibilitet $\Xi\in[0,1]$:

$$\delta(x,u)=\tanh\!\left(\frac{k[f(x)+g(u)]}{2}\right),\qquad
D=B+\Xi\delta\begin{cases}1-B,&\delta\geq0,\\B,&\delta<0.\end{cases}$$

Dermed $0\leq D\leq1$ og $dX=(D-B)dt/C$. Monotone ikke-voksende kurver med
endpoints $f(0)=g(0)=1$, $f(1)=g(1)=-1$ giver indadrettet kontinuert randdrift.
Det er ikke i sig selv en garanti for en diskret solver. Bevarelse, positivitet
og grænser kontrolleres særskilt. BM's $X$ omdøbes ikke til WM's $F$ eller $R$.
Den fælles [model](../../src/model/bm_1.py) og
[balance](../../src/model/base/bm.py) er WM-afhængigheder og uændrede.

## F: stokastisk tilstand og Gaussian intervalobservation

F bruger **døgn**, en fysisk time er $\Delta=1/24$:

$$dX_t=\frac{D(X_t,U_t,B_t)-B_t}{C}dt+\sigma_xX_t(1-X_t)dW_t,$$
$$Y_j=\frac1\Delta\int_{t_j}^{t_j+\Delta}D(X_s,U_s,B_s)ds+\epsilon_j,
\qquad\epsilon_j\sim N(0,\sigma_y^2).$$

$C$ har normaliseret tidsenhed døgn, $\sigma_x$ døgn$^{-1/2}$.
$Y_j$ er ét skaleret gruppemiddel, ikke 13 selvstændige huslikelihoods.
Den latente integrand er begrænset, men **Gaussian $Y$ har støtte på hele
reallinjen**. $\sigma_y$ er nominel gruppestøj, ikke estimeret husvis støj.

F4's kontrol af hver råmålers fortegn er derfor en **forbrugsdatakontrakt**,
ikke en matematisk følge af likelihoodstøtten. Kontrollen forhindrer, at
positive målere skjuler en uafklaret eksportkanal ved aggregering. Den gamle
24-timers filterperiode har 17 negative råceller i seks timer; over hele dens
192 timer er der 122 negative celler i 44 timer, men kun tre negative
gruppemidler. Middelkontrol alene ville ændre optagelseskriteriet.

Hvis kanalen er netforbrug, repræsenterer ikke-negativ latent konsumtion ikke
import minus produktion. En konkret forsvarlig udvidelse er en eksplicit
produktions-/eksportkomponent i observationsoperatoren, med dokumenteret
enhed og tilgængeligt scenarieinput; baseline og identifikation skal revurderes.
En direkte signeret nettomodel er også mulig, men kræver en ny balance-/
støttekontrakt og selvstændig validering.
Hvis negative tal derimod er bekræftede målefejl omkring bruttoforbrug, kan
Gaussian gruppelikelihood i princippet tillade negative $Y$, men admission og
frossen træningspopulation skal vurderes eksplicit. Ingen af disse valg indføres
her. Klipning, vilkårligt offset, målerudeladelse og populationsskift er ikke løsninger.

### Normal F-beregningsvej

[F4](../../src/exe/run_model/bm_1_f4.py) følger:
`prepare → fixed_population_mean → frozen scale/profile → Beta prior → advance_hour → export`.
`advance_hour` beregner prædiktive intervalmomenter **før** betingning på en
historisk observation. Kun timer før forecast-origin assimileres. Eventuelle
fremtidige observationer åbnes separat til evaluering efter prognosen.

Det gemte tostykkefit har ét fælles kurvepar, frossen ugeprofil og nominel
$C=1,k=2,\Xi=1,\sigma_x=.5,\sigma_y=.04$, med Beta(20,20) ved ny filtrering.
Disse fysiske/støjparametre er ikke identificeret af startkommandoen.
Inputvalideringen håndhæver nu korrekt $0\leq\Xi\leq1$; tidligere blev nul
afvist tidligt, mens $\Xi>1$ kunne nå numerikken; rewardguard afviste først
et faktisk domænebrud.

Den korte [likelihood-ejer](../../src/method/bm_1_interval_likelihood.py) har
de fælles Thomas-faktorer, Fourier-/alias-/tailguards og konturbank. Tre
primitives er flyttet AST-identisk fra historiske referenceharnesses.
[Native kernen](../../src/method/bm_1_interval_native.py) ejer operatorfrysning,
frequency blocks og terminal-state-betingning; alle ti eksisterende definitioner
er AST-identiske med PR146. [Intervalmomenter](../../src/method/bm_1_interval_forecast.py)
ejer middel, varians og marginalkvantiler. Observationsvariansen tillægger
$\sigma_y^2$ én gang. Nulmasse bevares som logmasse $-\infty$ / JSON null.

Normal F4-indlæsning kræver ikke længere karakteristik-/terminalforsøgsdrivere.
Native TRAIN-resume bruger stadig precise-referenceværktøjets checkpoint/units;
den afhængighed påstås ikke fjernet. Historiske guards accepterer kun deres
eksakte originale kildeversioner, herunder krævet LF-repræsentation.

### Nye F-ugedagsfits og genoptagelse

[F-adapteren](../../src/tools/f_weekday_backtest_adapter.py) laver syv **nye**
tostykkefits via `fit_shared_curve_pair` og native interval-likelihood. Gemte
F4-logits ommærkes aldrig som nye fits. Baseline/skala/prior/fysik er eksplicit
tidligere frosne; cutoff må ikke ligge efter hele femugershistorikkens begyndelse.
Den leverede baseline har cutoff 16. februar, så første mulige mandagsorigin er
28. marts 2022. Præaggregeret CSV kan ikke bevise den rå komplette målerliste;
den deklarerede aggregeringskontrakt skal være opfyldt før indlæsning.

Hver kandidat starter fra egen Beta-prior og propagérer gennem alle fysiske
historiktimer. Kun fem matchende ugedages targets indgår; mellemrum fortsætter
tilstanden uden reset. Valgte logits skal høre til et faktisk succesfuldt
optimizerresultat. Budgetstop og ufuldstændig replay er ikke accepterede fits.
Hele TRAIN kontrolleres base/fine: logscoreforskel højst .05, statemiddel .002
og konservativ L1 .05 ved hver time. Forecast har state-/momentgates.

Hvert fit beholder sin egen mandagsposterior og gennemløber et targetfrit
præfiks frem til sin ugedag. Syv kontinuerlige kandidatbaner samles; det er
**ikke** én 168-timers statebane med daglige parameterskift. Middel/baseline
skaleres med effektskalaen og varians med dens kvadrat.

`--resume` genbruger hashverificerede færdige fits, dagsprognoser og hele uger.
Source/runtime/model/config/gruppe/kalender/historik/input skal matche. Et
uafsluttet fit får ét nyt forsøg med oprindeligt budget i en ny attemptmappe;
gamle partials bevares. Der er ingen optimizer- eller partial-prefix-resume.
F4's egen intervalvise pause/resume er fortsat separat. Alle uger fryses før
scoring. Se [kalendervejledningen](WEEKDAY_BACKTEST.md).

## G: timebalance og et reelt frosset modelartefakt

G bruger **timer**, $f(x)=1-2x$, og $g$ interpolerer
$(0,1),(.5,a),(1,-1)$. Den beskyttede
[RK4-responskerne](../../src/method/bm_1_deterministic_response.py) afviser
domænebrud uden stateklip. For en fysisk time:

$$\overline D_j=\overline B_j+C(X_{j+1}-X_j).$$

For andre intervallængder divideres sidste led med længden. Punkt-demand
bevares som en anden størrelse. Timebaseline integreres på 60 delintervaller;
60/120-forfining kontrollerer efterfølgende samme $10^{-5}$ demandtolerance.
Baselinefit på lineære integrerede features og deployment med klip er
forskellige operatorer. Rank 20 ud af 49 features i alle 42 gamle fits er en
reel identifikationsbegrænsning, som ridge ikke ophæver.

G bruger den træningsbundne affine råkoordinat
$Y_{raw}=demand\_min+demand\_scale\,D$. Den kan rumme negative råtal, når de
ligger i det frosne område. Det gør ikke G's $C\cdot demand\_scale$ til en
identificeret fysisk batterikapacitet, og det er ikke F's rent multiplikative
skala. Prisdomænet klippes eller nytilpasses ikke ved ny prognose.

`export-model` validerer det gamle [weekday-export](../../src/method/g_weekday_export.py)
og bevarer alle originale snapshotbytes/hashes. Den nye
[G-prognoseejer](../../src/method/bm_1_g_forecast.py) indlæser kun frosne dagsfits,
baselines, skala og residualbanke; den normale sti kalder ingen gammel
fit-/studie-CLI. `forecast` kan bruge en **senere mandagsuge** med nyt deklareret
Price/Temp-scenarie. Fit-cutoff og forecast-origin holdes adskilt.

Der opdateres ingen parameter, baseline, skala eller initialisering. Hver dag
starter fra sin egen oprindelige historiske medianstate. Det er syv separate
dagsbaner; alder, sæson-ekstrapolation, baselineklipning og gamle residualers
relevans skal fortsat vurderes. `replay`/den gamle `--snapshot`-form er bundet
til den oprindelige uge og inkluderer den særskilte WM-timeadapter.

G's empiriske outputlag er frosne historiske fejl, ikke diffusion i $X$,
parameterusikkerhed eller kalibreret fremtidig dækning. Den særskilte
[båndprotokol](G_WEEKDAY_STUDY.md) og dens kilder/resultater er bevaret; dens kalibrering
må ikke automatisk overføres til den nye G-only deploykontrakt.

## Kontroller, output og faglige grænser

| Område | Kode-/testforbindelse |
|---|---|
| F gruppe, cutoff, inputisolering, resume og output | [F4-tests](../../tests/test_bm1_f4_runner.py) |
| Fourier-/rand-/positivitetskontroller | [characteristic](../../tests/test_bm1_interval_characteristic.py), [terminal](../../tests/test_bm1_interval_terminal_tilt.py) |
| Ny F-likelihood-fit, native/precise, gaps og nulmasse | [F-adaptertests](../../tests/test_f_weekday_adapter.py) |
| Fælles kalender/DST, targets og freeze-before-score | [kalendertests](../../tests/test_weekday_backtest.py) |
| G ny origin, gruppe, kilde-/inputmutation og numerisk ækvivalens | [deploytests](../../tests/test_bm1_g_forecast.py) |
| Historisk G-snapshot og bånd | [G-runner](../../tests/test_bm1_g_runner.py), [G-bånd](../../tests/test_g_weekday_bands.py) |

F/G eksporterer entydige fysiske intervalkanter/UTC, rå prediction/baseline,
inputscenarie, fit-cutoff, model- og kildeidentitet samt runtime. G's
`horizon_start_hours/end_hours` gør intervallead eksplicit; det ældre
`forecast_horizon_hours` er startlead. F's `horizon_hour` er slutlead, første
forecasttime=1. Modelspecificikke statefelter bevarer deres betydning.
RMSE, MAE og bias er absolutte råenhedsfejl; MAPE bruges ikke ved negative/nul targets.

Der skelnes mellem numerisk forfiningsfejl, procesusikkerhed, observationsstøj
og empiriske fejl-/kalibreringsbånd. Ingen af dem er automatisk parameter- eller
inputusikkerhed. Realiseret pris/temperatur er et betinget scenarie, medmindre
tilgængelighed ved origin dokumenteres; fremtidig demand må kun bruges til scoring.
Original WM4 og G's WM-timeadapter bruger forskellige observations-/fitprotokoller.
Oracle-statevalg er ikke almindelig prognose. Den fælles evalueringskampagne og
rangering er udskudt; population, enhed, periode og information skal først afstemmes.

Nødvendigt restarbejde er datakanalernes betydning, verificeret G-medlemskab og
et særskilt historisk F-fitbudget med konvergens/runtime. H/I/J-accept,
firestykkefit, nye baselinefamilier og parameterbanke er bevaret forskning,
ikke skjulte acceptkriterier for oprydningen. Johans praktiske brugerprøve afventer.
