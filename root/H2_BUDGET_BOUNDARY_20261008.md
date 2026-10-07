# H2: budgetstop efter43 kald, 8. oktober

Blok6 sluttede normalt med `Search stopped to protect complete remaining validation`.
43 firestykkekald er komplette. Kandidat44 har38 af72 timer gemt; alle1166 nye
jobs er sikret i checkpoint01–06. Seneste checkpoint indeholder195 nye jobs og
verificerer450 adoptioner. Der er0 gemte numerical jobfailures. Ingen proces
kører. Blok4's fulde600s-reservation og alle ældre charges bevares.
PR118 er merged med checkpoint01–05 og opdateret I/J-matematikreview.

En enkelt cached COBYQA-diagnose reproducerer43 calls og den præcise næste
kandidat uden propagation. Sidst rapporterede resolution er0.1; krævet slutradius
er0.02. Der foreligger ingen convergence eller model selection. De små parser-/
bogføringsfejl i read-only diagnostic/pricing er bevaret og konservativt charged;
ingen af dem er en model failure eller anledning til ændring af likelihood/guards.

## Tidsbehov og konkret forslag

Til højst49 calls mangler120 prediction- og274 observed-basejobs. Nye gemte
jobmiddelrater er0.33385s/3.90317s:18.49min modelarbejde,27.74min med faktor1.5.
De uændrede registered maxrates med faktor1.5 og360s hjælpereserve giver82.42min.
Dette er extrapolation fra faktisk gemt arbejde, ikke en universel upper bound
eller garanti for optimizerconvergence inden49 calls.

Forslag: ét separat budgetsupplement på højst7200s; samlet H-loft64800→72000s
(18→20 timer), samme8.oktober08:00 Copenhagen-frist, native1080s,25/49-caps,
73-distinct-pair-cap og600s-blokke. De20955.85668s restvalidation beskyttes
fortsat; hele jobgrafen er ikke garanteret færdig før fristen. Ny budgetapproval
er påkrævet; `approved=false`. H/I/J/F4 er stadig åbne. Ingen refund, restart,
capforøgelse, fallbackmodel, numerical retry eller ændring af final-data-gaten.

Den konkrete proposal er `root/f4/h2_continuation_budget_supplement_proposal_20261008.json`.
V3-campaign og original execution bevares. V4 adopterer alle oprindelige entries
og den separate5s preparation charge; den samme numerical Engine/genbrugsstore
og præcise optimizertrace bruges. Nativekilder og original konturadapter er urørte.

## Forberedt implementation og kontrol

`root/f4/run_h2_budget_supplement.py` og `checkpoint_h2_budget_supplement.py`
er syntaxkontrolleret. Én registered offline invocation består7 scoped checks:
unapproved afvises før reservation, korrekt v4-prefix/sum accepteres, og tampered
prefix/loft/deadline/total/validationreserve afvises.0 propagationer og0 production
reservations. Source-/control-/preparationhashes er fastlagt i proposal.
Godkendelse af merget implementation er ikke godkendelse af ekstra modelkørsel.

Efter udtrykkelig budgetapproval gemmes den i en separat approvalfil med ny
decisiondigest. V3-proposal/approval/ledger og alle gamle checkpoints bevares.
Ny controllercommit registreres; launch bruger separat v4-campaign og supplemental
execution/report. Kontrollér ingen aktive processer/låse og samme miljø/kilder først.
Én bounded blok → charged review/checkpoint → durable save før næste blok.
Budget-/fristsstop er pause; cap eller numerical failure er terminalt.

Original input.xlsx er fortsat nødvendig efter den faktisk validerede forecastfreeze.
Den blokerer ikke det nu pauserede fit. I's faktiske handoffreview kører først efter
H-handoffet; selvstændig I/F3-evidens er fortsat en separat registreret leverance.
F4-G2b/G2c forbliver droppet, og separat G/WM-arbejde er uberørt.
