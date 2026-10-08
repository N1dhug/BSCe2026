# Afgrænset S3-proceslag: protokol før historisk kørsel

8. oktober 2026. Implementeringspas startet 18:15:43 UTC. Dette er G's
eksisterende Ito-SDE, ikke et nyt F-studie. S3 lambda=0, fitted baseline,
normalisering, to-stykkers g og syv egne mandagsfrosne starts bevares.
Ingen ny optimizer, parameterbank, observationsstøj eller modeludvælgelse.
Den foreløbige støtteanalyse er allerede set: 657/720 historiske målinger
ligger uden for det frosne D(X)-interval. Denne kørsel er udvikling, ikke blind.

## Kørsel og stopregler

1. Én træningsdiagnose på det lokale Feb7-snapshot og præcis dets historik.
   Maksimum 60 sekunder. Fem historiske målte timer per ugedag/time, sample-
   varians ddof=1, mindst to målinger. Time 0 udelades fra amplitudefit.
   Én q>=0 per uge: q=max(0,w·v/(w·w)), w er gennemsnittet af fem
   enhedsvarianser. Coarse/fine bruger snapshotgrids. Demandfejl <=1e-5,
   enheds- og fitted profil-standardafvigelsesfejl <=1e-4. P>1/4 og målinger
   uden for D([0,1]) er hver sin afvisningsgrund. q anvendes aldrig skjult
   som en accepteret støjstyrke. Numerisk bestået er ikke fysisk accept.
2. Én input-only FV-uge med udtrykkelig **illustrationsstyrke sigma=.2 per
   sqrt(time)**. Den er ikke et historisk støjfit. P0=0 ved hver egen start.
   128/256 celler, maksimumstrin 1/24 og 1/48 time, budget 120 sekunder.
   Hver prisgrænse segmenteres. Eksisterende konservativ zero-flux FV med
   implicit Euler genbruges; sandsynligheder klippes/renormaliseres ikke.
   Kontroller normaliserede middelfejl (state/demand) <=.005, variansfejl
   <=.002, demandkvantilfejl <=.005 og state-CDF-fejl <=.05 ved .25/.5/.75
   efter time 0. Disse er udviklingstolerancer, ikke en rigorøs fejlgrænse.
   Dirac-starten ligger i én celle: centreringsfejlen rapporteres særskilt.
3. Kun ved numerisk FV-afvisning tillades én på forhånd afgrænset forfining:
   256/512 celler, trin 1/48 og 1/96 time, samme 120-sekunders loft. Ingen
   sigmaændring, gateændring eller automatisk yderligere søgning.
4. Prognosen gemmes før observationerne åbnes. Derefter score samme endelige
   måltemaske i råenheder: MSE/RMSE/MAE/bias, punktvis bånddækning, bredde og
   intervalscore. Kvantiler beregnes direkte fra D(X)-sandsynlighedsmasser,
   ikke som klippede Gauss-bånd. E[D(X)] holdes adskilt fra ODE-prognosen.

Diagnosen bruger P'=2(D_x/C)P+[x(1-x)]² ved enhedsamplitude, P0=0 efter
24 timers warm-up. Dette er en lokal approksimation; stor fitted q giver
ikke automatisk en gyldig ikke-lineær lov. Rå historisk spredning indeholder
også ændrede inputs/baseline og identificerer ikke alene fysisk processtøj.

Historisk kalibrering får særskilt status. En numerisk god proces-scenarie-
vej kan afleveres selv om kalibrering af observeret totalusikkerhed afvises.
Ingen garanti om 95% observed-coverage eller samtidig ugedækning.
Rå inputs, tidsserier, snapshots og fulde outputpakker forbliver lokale.
Kode, syntetiske tests og aggregeret dokumentation må sendes og merges efter
Johans allerede givne tilladelse. F's kode, budget og accept ændres ikke.
