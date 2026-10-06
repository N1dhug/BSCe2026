# F4-H2: betinget datagrundlag, 6. oktober 2026

Aktiv datapolicy er `root/f4/h2_compact_data_policy_20261006.json`. Johan valgte
**kurver først**: fælles f/g estimeres betinget på faste F-referenceparametre,
ikke samtidig historisk identifikation af de fysiske parametre. Implementeringen
er færdig; de faktiske workbook-afledte tal skal stadig fryses på PC og råreviewes.
`input.xlsx` findes ikke i chat-workspace, og den store private GitHub-binærfil
kan ikke hentes korrekt gennem den tilgængelige forbindelse. Ingen faktiske MED-
forbrugsværdier er læst i denne implementering.

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
10 h og mål 4–6 h er uændret; alle fit/readiness-flags forbliver false.

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

## Lokal forberedelse

`root/f4/run_h2_compact_data.ps1` eksporterer `src`, `root/f4` og den aktuelle
budgetseed fra et præcist Git-commit til en ny midlertidig sourcefolder. Den
ændrer ingen aktiv branch/worktree og udfører ingen G/WM-kørsel. Brug eksisterende
Python-miljø med NumPy/pandas/SciPy/openpyxl. Der installeres ikke automatisk.
Workbook-hashes skal matche header-auditen før én måleværdi læses.

Runneren bruger én proces/BLAS=1, den eksisterende fælles H-ledger og native-
allowance, default højst **180 s inklusive 60 s stopreserve**. Imports og
forberedelse tælles konservativt i budgettet; ingen fit eller FV/JIT-replay.
Hver preparation er immutable, med exclusive lås, input/sourcehashes før/efter,
source/environment-manifest, rå baseline.csv, inputs.npz, packet.json, source.zip,
rapport, afsluttet ledger og review-ZIP. `inputs.npz` bruger det eksisterende
BM1-array-checkpointformat (ikke numpy.load direkte). Target-Y er fraværende.
Ved manglende bin/negativ effekt/ud-af-range pris/ændret workbook/budgetstop:
bevar output og failure; ingen automatisk gentagelse, retuning eller imputation.

Næste trin er råreview af den lokale ZIP: baseline-bins, complete/missing counts,
reelle frosne skalaer, provenance og ledger. Derefter kan en særskilt kort faktisk
rate/gate planlægges under det resterende budget; optimizeren startes aldrig af
datapreparationen. Den gamle delvise packet og metodebeslutning bevares som
historik. Sol High er tilstrækkelig til dette data-/implementeringsarbejde.

## Afgrænset kontrol

Én syntetisk workbook-/OLS-kontrol er kørt under fælles reservation: 4.06 s
inklusive imports og source-export; ingen modeltimer eller historiske værdier.
168-bin-designmatrix/OLS giver maksimal fejl 1.78e-15. Poison-formler efter
historik-cutoff, TRAIN og final samt finalpriser dekodes ikke. Fixed-group-NaN,
utilstrækkelig historik, ændret population/hash, overwrite og blindhedsclaim
afvises. Nye CN-load-bounds .93845/.82094 består <1; de gamle steps afvises.
Rårapport/manifest/source/ledger er gemt i
`files/output/bm_1/runs/estimation/f4_h2_compact_data_control_20261006_review.zip`.
Tre ændrede Python-filer består syntakskontrol. Windows-launcheren er
statisk gennemgået, ikke eksekveret her (PowerShell er ikke tilgængelig).

Ledger efter kontrollen: **1498.45822 s H**, **473.82710/900 s native**,
rest **426.17290 s** samlet på tværs af PC/chat/resumes. Den nye seed udvider
den fulde kendte historik; gamle seeds/rapporter må ikke nulstille tiden.
