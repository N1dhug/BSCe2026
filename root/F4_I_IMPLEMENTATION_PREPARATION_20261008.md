# Faktisk H→I efter den kontrollerede fortsættelse

Denne forberedelse ændrer hverken H's låste kilder, forsøgsbudget eller gates.
Den gamle I-baseline og dens råreview genbruges som interfaceevidens; de køres
ikke igen. H-fittet, dets validering og forecastfreeze er stadig i gang.

## I1: faktisk origin frem for en ny demonstration

`root/f4/review_h2_i_handoff.py` er et read-only review af det faktiske
`conditional_handoff.npz` og `forecast_frozen.npz`. Det kontrollerer checksum,
manifest, selectionafhængighed og informationscutoff. For hver eksporteret
model og hvert grid går den originale posterior gennem den eksisterende
`as_forecast_origin`/`BM1ForecastOrigin`-bro. Tidspunktet er dag4, svarende til
21.februar2022 kl.00:00 Copenhagen;168h slutter ved dag11.

Reviewet kræver præcis egen log-state/masse, samme f/g og nominelle fysiske
parametre. Den konservative cellelov overføres til401/801 celler med positiv
masse og1e-10 massetolerance. Original log-state og originalt grid bevares i
handoffet. Det er ingen propagation, nyt fit eller populationskontrol.
Koden er syntaxkontrolleret, men reviewet er endnu ikke udført: H-handoffet
findes først efter komplet selection og valideret forecastfreeze.

## I2: forecastkontrakter, der skal holdes adskilt

H's open-week-prognose er for timegennemsnit af demand og Gaussian interval-Y,
fra den faktiske frosne origin med fastlagte B/U og nul fremtidig assimilation.
Den har egne base/fine-forløb og CDF/kvantiler ved1/6/12/24/48/72/168h.
De råresultater genbruges som betinget intervalforecastevidens.

Den eksisterende `forecast_fv` giver derimod endpoint-state og øjeblikkelig
demand. En selvstændig I-backendkontrol skal bruge samme frosne origins og B/U,
men sammenligne endpoint-state med H's gemte endpoint-state. Interval-demand
må ikke sammenlignes direkte med øjeblikkelig demand. H's sigma_y er deklareret
for normaliseret timegennemsnit; `forecast_fv`'s gamle point-Y-varians må ikke
genbruges som kalibreret interval-Y eller som en ny målerstøjmodel.

En fuld selvstændig I-kørsel kræver egen source-/origindigest, atomiske
checkpoints, konkret runtimebudget og numerical reference før launch.
Første afgrænsede målescope er to eksporterede origins, ét open-week-forløb
per origin/grid og få faste F3-policies. Eksakte antal afhænger af H's valgte
model; vælges lineær, udføres den identiske model ikke to gange.
Der er endnu ingen ny numerical I-kørsel eller runtime-admission.

## I3: eksisterende F3-evaluator med faktisk H-belief

Genbrug `origin.control_problem` og F3's objective, constraints og bounded
optimizer. Begge beliefs vurderer de samme faste policies, med samme B/A,
prisskala, startcutoff og horizon; ingen nye målinger under hypotetisk forward.
Efter en kort måling kan et lille optimizerbudget prissættes, og hver valgt
policy får egen fine-vurdering. Ingen skjult genfit eller gennemsnit af optima.

Den lineære H-reference er en nominelt betinget reference, ikke kendt fysisk
truth. Sammenligning med den valgte fitted kurve kan derfor vise betingede
policy- og belief-forskelle, men ikke fysisk parameter-recovery eller faktiske
økonomiske besparelser. Eksisterende kendt/rho-MAP-evidens bevares separat.

## J og afslutning

Samlet rapport skal forbinde A–G's allerede afsluttede evidens med H's actual
resultat, I1-origin, intervalforecast, selvstændig I-backend/F3-kontrol og alle
negative fund. Parameter-/inputusikkerhed og ukendt tidligere slutugeeksponering
mærkes eksplicit. En betinget udviklingsuge er ikke populationskalibrering;
mindst200 uafhængige sekvenser kræves fortsat ved sådanne claims. Et interface-
review eller en bestået pilot afslutter ikke alene I/J/F4.
