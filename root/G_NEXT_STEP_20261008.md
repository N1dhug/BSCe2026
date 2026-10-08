# Seneste holdested: grundimplementering afleveret, intervaloperator næste faglige trin

S3 fit/snapshot/predict/score og en afgrænset, kontrolleret FV-procesvej er
nu implementeret. Se [samlet brug, matematik og faktisk status](G_S3_UNCERTAINTY_GUIDE_20261008.md).
Historisk støjfit er afvist; eksplicit valgt sigma er et processcenarie,
ikke et accepteret kalibreret bånd for målingerne. Johans nyeste oplysninger:
timeværdien dækker forbrug over en hel time; Demand er efter hans erindring
Power. Verificér præcis enhed og brug derefter en timeintervaloperator.
Punktoutput er fortsat deklareret udviklingsapproksimation. Stokastiske
timegennemsnit kræver tidskovarians/integreret output; de kan ikke rekonstrueres
korrekt fra FV-marginalbånd alene. Ingen sådan udvidelse er udført nu.

Den første praktiske milepæl er færdig. Denne G-chat, Sol High, er fortsat
passende til almindelig implementering. F/bank/BM2 er særskilte spor.
Instruksen nedenfor er historisk og allerede udført, ikke en bestilling om refit.

# Historisk arbejdsinstruks: selvstændig S3 fit/snapshot/predict

Seneste præcisering fra Johan: Førsteprioriteten er fungerende G-kode,
som kan køres, undersøges og senere udvides med BM2 eller flere parametre.
G optimeres med metoder egnet til G inden for tidsrammen. WM er en konceptuel
reference, ikke et krav om 1:1-behandling. Solvere, fitkriterier, intern
skalering og modelspecifikke checks må variere med en kort faglig begrundelse.
En sammenligning skal have sammenlignelig opgave, information og horisont
og vurderes i fælles enheder på et tydeligt observationsgrundlag.
Første milepæl er en fungerende deterministisk grundversion. Støjlaget har
særskilt status og må ikke blokere denne milepæl alene af hensyn til WM.
Variansfit er en kandidat, ikke en binding til WM's metode. Ingen lange
metodesøgninger, fuld WM-paritet eller generel BM2-arkitektur nu; enkle
modulgrænser og et versionsmærket snapshot er tilstrækkeligt.

Klar arbejdsinstruks, 8. oktober 2026. Brug denne G-chat med **GPT-6.1 Sol,
High**. En ny chat er ikke nødvendig; hvis kontekstskift er praktisk, kopier
denne instruks ind i en særskilt G-chat og læs repoets nyeste AGENTS først.
F-chatten og F's undersøgelser er ikke denne opgave. Astra/Extra High er
ikke et generelt krav; brug kun dyrere ræsonnement ved et konkret matematisk
problem, som den afgrænsede implementering ikke kan afklare.

## Bestilling

Implementér et tyndt selvstændigt indgangslag for den allerede valgte
historiske G-S3-model, lambda=0. Brug [aktiv plan](G_COMPLETION_PLAN_8H_20261008.md)
og [WM-review](G_WM_ALIGNMENT_REVIEW_20261008.md). Sigt mod to timers arbejde
for denne del; registrér faktisk tid fra start og genvurdér ved konkret
blokering. Det er en del af den eksisterende otte-timers-ramme.

Start fra seneste main på en egen G-branch. Kontrollér for nyere ændringer;
bevar brugerens/F's arbejde. Brug repoets eksisterende kodekonventioner.
Foreslået placering er `src/method/bm_1_s3_forecast.py` og en lille
`src/exe/run_model/bm_1/s3_forecast.py`; navnene er planlagte, ikke eksisterende
funktionalitet eller et krav om at oprette unødvendige lag.

## Genbrug præcist disse dele

- `deterministic_validation.week_job`: 35 historiske dagsforløb og syv
  mandagsfrosne fremtidsdage; fremtidig demand indgår ikke i selve jobbet.
- `deterministic_strategies.calendar_inputs`: 24 fysiske warm-up-timer,
  fast baselinegrid og scenarieinputs. Vær opmærksom på lokal DST.
- `BM1PooledInputs` og `fit_strategy`: samme målfunktion, parametre,
  to-stykkers prisrespons og kun lambda=0. Kopiér ikke `checked_task`'s
  to-lambda-undersøgelse eller hele det gamle seksugers studie.
- `wm_4.output_model.statistical_output`: gem også `coef`, `feature_mean`,
  `feature_scale` og alle constructorindstillinger; metadata alene er
  ikke et selvstændigt fitted snapshot.
- G's eksisterende træningsbaserede skalering. Brug ikke WM's sæson-/
  helårskalering fra `aligned_data` til en fælles forecast-kontrakt.

## Leverance og fokuserede checks

1. `fit`: én mandagsorigin med erklæret compute-loft; eksisterende
   optimizerkonvergens og coarse/fine-krav bevares. Delvise fits får status
   unfinished. Mulighed for at importere komplet verificeret gammelt fit;
   ingen rekonstruktion af manglende baselinekoefficienter ved gæt.
2. Versionsmærket, valideret `snapshot`: hele baseline, normalisering,
   C/k/Xi/a, syv egne historiske starts, grids, cutoff, tidszone,
   observationstype, input-/kildehashes, træningsmaske og numerikstatus.
   Snapshotgrids bevarer både coarse/fine-initialisering, hvis dette er
   nødvendigt for at gentage det oprindelige forfiningscheck.
3. `predict`: snapshot + fremtidige priser/temperaturer; intet demand-krav
   og intet fit. Syv separate dagsbaner, aldrig skjult kontinuerlig 168h-bane.
   Rå og normaliserede output, baseline, pris, tid, protokol og status.
4. `score`: valgfri efterfølgende observationstabel og fælles måltemaske.
   Eksportér MSE/RMSE/MAE/bias og figur. Ingen fremtidig observation i
   inputfingerprint for prognosen; et separat scoringfingerprint er tilladt.
5. Kontraktchecks: rundtur gem→indlæs giver samme forecast; fitfunktioner
   blokeres under reload/predict; ændring/fjernelse af future demand ændrer
   kun scoring; baselinefeatures/skalering/cutoff og starts bevares.
   Finitte tal og demand-forfining ≤1e-5. Snapshot med forkert version,
   ugyldige grids, manglende felter eller scenarie uden for kontrakt afvises.
6. Kalender: ordinær uge og kendt forårs-DST kontrolleres. Efterårs-DST med
   dublerede lokale tider skal enten bevares entydigt med UTC-offset eller
   afvises tydeligt i første version. Ingen skjult drop/merge af timer.

Find i samme arbejdspas det historiske udviklingseksempels inputs. Input-
og prisworkbooks findes i Git, men sparse checkout har dem ikke lokalt;
temperaturcache og den fulde gamle fitted pakke er endnu ikke verificeret.
Dette er en reel adgangsopgave, ikke dokumentation for tabt kode. Byg og
kontrollér adapteren med små syntetiske kontrakteksempler imens. Behov for
ukendt fysisk enhed afklares med Johan; opfind den ikke. Temperaturer brugt
som realiseret fremtid mærkes som betinget scenarie, ikke vejrudsigt.

## Afslutning af denne del

Gem fungerende kode, fokuserede testresultater, dokumenteret kommando og
faktisk tidsforbrug. Historisk eksempel må ikke erklæres kørt på baggrund
af syntetiske tests. Hvis komplet input er tilgængeligt, kør det ene
budgeterede historiske eksempel; ellers skriv præcis manglende input og
fortsæt uafhængige kontraktchecks. Merge færdig egen branch efter relevante
kontroller som autoriseret af Johan. Ingen ny generel plangodkendelse kræves.

Derefter: det enkleste fagligt gyldige G-støjlag inden for tidsrammen.
Det beskrevne variansfit er en kandidat, ikke et krav om at kopiere WM. Ingen parameterbank, G7-døgnkørsel eller F-forsøg i dette
første implementeringstrin. G6a's tidligere failure står uændret.
