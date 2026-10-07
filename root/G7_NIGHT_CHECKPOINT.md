# G7 nattecheckpoint — færdigt deterministisk delstudie, 7. oktober 2026

Status: numerisk studie og read-only review færdige. Publicering følger G7-branchens PR; numerik skal ikke gentages.
Aktiv overdragelse: [matematik og resultater](G7_ENERGY_NEUTRAL_REVIEW.md).
Råresultater er sikret på commit4a3c4148b308fd2e74e3b7cc590a0de591b349b0.
Alle17 units/19 råfiler er komplette. Ingen modelgenkørsel ved genoptagelse.
Seks policies består precision;374 forwards;3.775397722s process.
Én uafhængig QP/Euler-test og byte-/J/E-audit består. Ingen M254-retry.
Konservativ nattecharge30s, G1098.580078s, rest701.419922s af1800s.
Begge deterministic objectives bevares, F og legacy uændrede.
Næste: [nodecheckpoint og komplet M254-jobgraf](G7_M254_RUNTIME_JOB_REVIEW.md),
inden en særskilt registreret måling/optimizer. Hele stokastiske G7 er åbent.

## Oprindelig registrering, bevaret som historik

# G7 nattecheckpoint — 7. oktober 2026

Status: registreret før nye beregninger. Executionbase: 4f9ea30abee6a2673a33868ffad30e9ff82cc429.
Ejer: G. F4-H2, WM/shared-kilder og vejlederens legacy-basis ændres ikke.

## Autoriseret arbejde
1. Genbrug immutable g7_matched_contract_20261007.json og tidligere prisrespons.
2. Optimér fire 6h-prisblokke på 24h for januar-BM lineær, januar-G4 og matched WM4.
3. Bevar original fixed-A-opgave; sammenlign separat lighedsbegrænsning:
   integral(demand-baseline) dh = 0. Ingen terminalstraf.
4. Review S3's gemte flade g og M254-runtime/jobdesign uden genfit eller pilotretry.
5. Dokumentér matematik, kilder, præcision, begrænsninger; review og merge egne færdige dele.

## Budget og accept
Seed G=1068.580078s, rest=731.419922s af 1800s.
Denne nat: højst 300s NY beregning/kontrol, også fejl og resumes.
Præregistrér detaljeret protokol og sourcehashes før første forward.
Genkør ingen afsluttet historisk studie. Ingen nye observationer eller parameterfit.
Gem hver færdig optimizer-delopgave og valgte fulde trajectory før næste del/fine.
Atomisk checkpoint med kumulativt budget, hashes og completed-id'er.
Afbrydelse må ikke nulstille budget eller slette fejlforsøg.
Afslut ved færdige leverancer eller konkret blokering, ikke efter et kunstigt tidsmål.

## Matematisk beslutning før forsøg
Kontrolpriser er u in [0,1]^4, WM bruger 2u-1 og dage internt.
Fast A er frossent baseline-døgnmiddel; pris/efterspørgselskalaer genbruges.
Kontinuert BM: integral(D-B)=C(x_T-x_0).
Kontinuert WM: integral(I-B)=24(F_T-F_0); R_T=R_0 kræves ikke.
WM's Euler-F og midpoint-demand har en eksplicit quadratureforskel.
Bevar solverne; brug samme diskrete demandintegral i objective og energilighed.
Rapportér terminalfejl og coarse/fine separat; ingen fysisk/kausal modelrangering.

## Genoptagelse
Læs dette checkpoint og nyeste G7_NIGHT_PROTOCOL/rapport, før nyt arbejde.
Afsluttede delopgaver genbruges; started uden completion regnes som afbrudt,
ikke numerisk accepteret. Kontroller kildehashes og ledger før genoptagelse.
