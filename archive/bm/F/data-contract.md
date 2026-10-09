# F: fast population, metadata og frosne inputs

<a id="h2-data-contract"></a>
<a id="h2-compact-data-foundation"></a>
Dette er en samlet faglig fremstilling af bevaret udviklingsevidens. Tidligere arbejdsordrer og budgetter er historik, ikke nye kørselsordrer. Aktuel brug, accept og opgaver står i [BM-guiden](../../../root/BM_GUIDE.md) og [status](../../../docs/bm/STATUS.txt). Originaler, hashes og genskabelse: [arkivindeks](../INDEX.md).

Historiske statusord som “åben”, “endnu ikke” og “Remaining scope” gælder den pågældende delundersøgelse 5.–8. oktober 2026. De beskriver ikke automatisk den aktuelle F4-kommando eller senere G-timeleverance.

F bruger et fast middel af de 13 auditerede MED-huse. Det senere G-spors
middel over tilgængelige målere er en anden population. Følgende metadata er
faktisk fundet; kWh/time, hour-start og tidszone for demand forbliver mærkede
arbejdsantagelser. Ingen metadata er blevet stærkere af dokumentoprydningen.

## Fundet populationsmetadata
## Confirmed population metadata, 5 October 2026

A read-only join of the consumption header IDs with `data_installation`
establishes the workbook's labels:

- **MED:** all 13 installation IDs are categorised as
  `Parcel-, række- m.v. huse med elvarme` (houses with electric heating).
- **UDEN:** 140 installation IDs are categorised as houses without electric
  heating: 126 `Parcel-, række- m.v. huse uden elvarme` and 14
  `Parcelhuse u/elvarme`.
- The two NULL, seven ?, one postal/telecommunications and one street-lighting
  columns remain separate categories; they are not members of the MED group.

The exact MED members are saved in the
[metadata audit](../../../files/output/bm_1/runs/estimation/h2_workbook_metadata_audit_20261005.json). All 164 header IDs were
matched against installation metadata. The audited input workbook SHA256 is
`3ce534b533e0da472cd72ae599916fa6209eccd10ad4c021a76ac5cfba3586d4`,
identical to Johan's earlier header-inspection report. Only `Labels 2022`,
`data_installation`, installation IDs and the two demand header rows were
decoded; **zero observation values were read**. The audit is reproducible
from commit `9de6d0d2e6f878ff9611f12acb776cdfb9ceae47` and
[its successful run](https://github.com/N1dhug/BSCe2026/actions/runs/37322217260).

This resolves the meaning and membership of those tags. A fitting contract
must still declare the selected fixed population. The inspected metadata sheets
provide no definition of consumption units, point versus interval measurements,
demand timezone or hour start/end. The price/proxy decision, chronological
split/exposure audit, frozen scales and baseline also remain prerequisites.
Johan's later hourly choices are working assumptions, not workbook evidence.
`historical_fit_ready=False` and scientific acceptance remain unchanged.


## Datagrænse og manglende værdier
## Implemented data boundary

`src/tools/bm_1_history_io.py` reads explicit XLSX cells instead of using the
legacy whole-workbook/interpolating loaders. Timestamp metadata and raw XML
bytes are traversed to locate the window; only the chosen population's
measurement cells inside it are decoded as observations. Numeric values
outside TRAIN, including formula cells, cannot influence its output. Missing
hourly rows/cells stay NaN; no imputation or normalization is performed.
Formatting-only rows are ignored. Invalid selected values, selected formulas,
duplicate/off-grid times and ambiguous/nonexistent demand hours fail explicitly.
Excel day fractions use microsecond precision to avoid false off-grid times
from binary floating point. UTC price times retain the distinct DST hours.

`load_historical_training_workbooks` requires the existing
`BM1HistoricalCurveContract`, a strictly earlier prior origin, demand timezone
and price area. It returns the entire TRAIN measurement table and raw prices
from prior origin through TRAIN end, plus digests/contract provenance. Workbook
digests must remain unchanged while loading. The spot-file route accepts only
an explicit `spot_proxy` / `DKK/MWh` contract. Consumer tariff input needs its
own documented adapter. Declared units are prerequisites, not proof supplied
by the loader.

Callers separately supply the complete history-only baseline and frozen physical
parameters to `prepare_historical_curve_training`. It preserves the first TRAIN
observation, fixed-population missing mask and elapsed UTC model time.
`read_raw_power_window` can load an explicitly declared earlier baseline-history
window through the same boundary; the training loader does not choose that
history or construct a baseline. Neither route provides an H3 holdout evaluator.


## Den faktisk frosne kompakte H2-kontrakt
De tidligere syvdagesforslag blev afløst af det udtrykkelige 48/24/168-timers
forsøg. Tabellen er det betingede historiske setup, ikke ny parameterestimation.
## Frosne valg

| Felt | Deklaration og provenance |
|---|---|
| Population | De 13 allerede auditerede `_MED`-kolonner; fast gruppemiddel. Én manglende husmåling gør hele gruppetimen manglende. UDEN indgår ikke. |
| Tid/enhed | Hour-start, kWh per time; gennemsnitseffekt kW. Europe/Copenhagen og enheder er dokumenterede arbejdsantagelser, ikke bevist målerkalibrering. Modeltid er dage. |
| Historik | 1. januar til 16. februar 2022 kl. 00, halvåbent vindue, 1104 timer. Intervalslut skal være senest cutoff; cutoff ligger strengt før prior-origin 17. februar. |
| Baseline | 168 weekday/hour-intercepts, separat mindste kvadrater på komplette tidligere MED-gruppetimer. Mindst 3 komplette historiske timer per bin; ellers STOP. Frosset kalenderprofil hele forløbet. |
| Effektskala | 1.1 × største komplette gruppemiddel i denne historik; ingen min-subtraktion, klipning eller fremtidsdata. Negative komplette gruppemidler giver STOP. |
| Pris | DK1 SpotPriceDKK, DKK/MWh, eksplicit spot_proxy efter F3-konventionen; ingen slutning om husenes geografi eller faktisk tarif. Historik-min/max med 10% af historikspændet som padding på begge sider. |
| Fysik | C=1, k=2, Xi=1, sigma_x=.5 fra `src/exe/run_model/bm_1/real_data.py::_parameters` og F3-referencen i `root/BM_REFERENCE.md`. Faste nominelle betingelser; ingen historiske MED-estimater. |
| Likelihood-støj | sigma_y=.04 i normaliseret timegennemsnitseffekt; særskilt deklareret nominalantagelse, ikke en estimeret målerstøj. |
| Prior | Beta(20,20) fra F-referencecases; prediction-only 17.–18. februar, ikke historisk estimeret prior. |
| Split | TRAIN 18.–20. februar (48 h), MODEL_SELECTION 20.–21. februar (24 h), slutuge 21.–28. februar (168 h). Ingen daglige kurvefits/state-resets. |
| Åbne ugeinput | Frosset baseline og priser fra 14.–21. februar gentaget fra forecast-origin 21. februar. Ingen fremtidige realiserede priser læses af denne adapter. |

Baselinevalget er nyt og eksplicit. De tilgængelige komplette timer giver den
analytiske OLS-løsning i hver kalenderbin. Det ændrer ikke den gamle helper
`previous_weekday_mean` med fem eksakte foregående uger. Manglende målinger
bliver hverken interpoleret, erstattet med baseline eller delvist gennemsnittet.
Ingen søgning efter bedre baseline, ny prisregion eller nye skalaer på target-Y.
Gamle F3-helseason-skalaer og G/WM-estimater genbruges ikke.

## Numerik og tid

Nominal sigma_x=.5 kræver mindre tidsskridt for positiv CN end de tidligere
betingelser. På uniformt fitted FV er diffusion <= sigma_x²/32 og |drift| <=1/C,
fordi B og BM1's bounded reward D ligger i [0,1]. Bernoulli-facevægtens bound
`D/dx² + max(one-sided drift,0)/dx` giver en konservativ outgoing-rate-grænse
`sigma_x² Nx²/16 + 2 Nx/C`. Kravet er `h × rate / 2 < 1`.

Vi deklarerer derfor base **1921 celler, h=1/32768** og fine **2561 celler,
h=1/65536**. Dette er en fysisk begrundet time-grid-revision, ikke ny solver
eller løsere precision. Frekvenser 64, span 2, contours og eksisterende gates
1e-10/1e-8/1e-7 samt score .05, state mean .002/L1 .05 og forecast .002 bevares.
Boundet viser kun positiviteten; historisk likelihood/base-fine-præcision er
stadig ukontrolleret. De gamle 40-timers kontroller og råartefakter bevares.

Det tidligere betingede **8.76 h**-overslag må ikke anvendes på denne condition.
Tidsskridtene kræver cirka en tredjedel flere substeps; en ren proportional
opskalering af den gamle proxy kan overstige 10 h. Faktiske repræsentative
historiske base/fine/moment/CDF-rates og prissat samlet jobgraf skal derfor
bestå før et fit. Der er ikke givet tilladelse til en lang kørsel. Samlet H-loft
10 h og mål 4–6 h er uændret; fit/numerik/runtime/accept forbliver false,
mens conditional_data_ready nu er true.

## Eksponering

Den gamle F3 real-data-applikation normaliserede med dataderiverede skalaer fra
hele sæsonen. Det er dokumenteret i `src/tools/bm_1_data.py` og
`files/output/bm_1/real_data/bm_1_real_data_daily_results.json` (Git-blob
`01fc5af661b7148214961672f3551d7548a958fa`). Vi ved ikke præcist, hvilke
slutugemålinger tidligere er inspiceret; nul nye læsninger er ikke bevis på
historisk blindhed. Slutugen fastholdes derfor som **udviklings-holdout med
ukendt tidligere eksponering**. Blind H3/videnskabelig accept forbliver false.
Det eksponerede/ukendte interval optages i contract.split.exposure_intervals,
så den eksisterende require_blind_holdout også afviser denne kontrakt.


## Bevarede datakontroller og evidensgrænse
Den gemte PC-pakke havde 1104 historiske gruppetimer: 984 komplette og 120
manglende. Alle 168 baselinebins havde 5–7 komplette observationer; skalaen
var 3.1265384615384613 under den deklarerede kWh/time-antagelse. Prispadding
gav −259.66200100000003 / 2865.162011 DKK/MWh. Datapreparationens gemte
30.4501879 s er ikke en ny måling af den nuværende F-opstart.

Den historiske audit kontrollerede packet identity, 60 sourcehashes, tidsgrænser,
profil/counts og prisrekonstruktion. Den modtagne receipt indeholdt ikke demand-
workbook, så hushistorik, OLS-koefficienter og historisk effektmax blev dengang
ikke uafhængigt genberegnet. Dette hul må ikke skjules som ny kalibrering.

Den tidligere syntetiske OLS-kontrol havde maksimal fejl 1.78e-15. Poison-
formler uden for cutoff, manglende gruppemedlem, utilstrækkelig historik,
ændret population/hash og blindhedsclaim blev afvist. Implementationskontrol
etablerer ingen leverandørmetadata. Det aktuelle loaderforløb står i
`src/tools/bm_1_history_io.py`; dets tests bevarer disse afgrænsninger.

## Konkrete kode- og kontrolindgange

[kode](../../../src/tools/bm_1_history_io.py), [loader-tests](../../../tests/test_bm1_history_io.py), [datakontrakt-tests](../../../tests/test_bm1_history_contract.py). Historiske CLI- og controllerkommandoer findes kun i det restaurerede originaltræ; disse links peger på de bevarede matematiske komponenter. Tests er links til tidligere eller løbende kontroller, ikke påstået genkørt ved denne dokumentændring.
