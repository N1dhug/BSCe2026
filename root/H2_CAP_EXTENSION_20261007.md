# F4-H2: eksplicit fortsættelse efter kandidatloftet, 7. oktober 2026

Johan bad kl. 16:28:18 om at fortsætte med revisionen af søgeplanen efter
det dokumenterede 17-kaldsstop. Denne instruktion anvendes til en afgrænset
metoderevision inden for det allerede godkendte 18-timersloft og stop kl.
23:55 Europe/Copenhagen. Den giver ikke et nyt tidsbudget.

## Beslutning og begrundelse

COBYQA nåede resolution 0,02 efter kald 16, men afsluttede ikke sine
afsluttende lokale skridt før maxfev=17. En cached-only rekonstruktion
med maxfev=25 reproducerer alle 17 kald nøjagtigt og anmoder derefter om
punktet [-1.0746176949041604,-1.513408573994702]. Ingen model blev genkørt.
De næsten ens sidste objectiveværdier er ikke selv et konvergensbevis.

Loftet bliver 25/49 objectivekald for familierne med 2/4 stykker: højst
24/48 nye ikke-lineære kurvepar, samlet 73 inklusive den fælles lineære.
Dette er den konkrete udvidelse beskrevet i terminalreviewet. Den større
fire-stykkers familie har seks variable mod to; en større maxfev giver
plads til flere trust-region-skridt, men garanterer ingen konvergens.
Det fulde 73-par-forløb er ikke finansieret/admitted. Søgearbejdet stopper
før det bruger den beskyttede resterende validering eller dagsfristen.

Samme pinned SciPy 1.17.0 COBYQA, lineære start, bounds [-4,4], start-
radius 0,5, slutradius 0,02 og regularisering anvendes. Den tidligere
radiusdiagnose findes i f4_h2_cap_extension_diagnostic_20261007.json.
SciPys beskrivelse af maxfev og final_tr_radius:
https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyqa.html
Den installerede 1.17.0-kode og cached trace bestemmer faktisk adfærd.

## Bevaring, genbrug og ny identitet

Den terminale gamle mappe bevares. En ny run kopierer alle 1.402 immutable
jobs og nødvendige frosne input/sourcefiler. Gamle resultat-/tracefiler
bevares uændret. Den gamle failure kopieres som predecessor-failure;
den nye execution bruger egne extension_optimizer_* filer og registry.
Den oprindelige manifestidentitet bevares som job-cache-namespace, fordi
hver jobs model, parametre, observation, grid og inputstate er identisk.
Manifestets gamle 17/33-caps beskriver alene forgængeren; den nye frosne
execution-identitet og metodebeslutning deklarerer 25/49. Der hævdes ikke
uændret optimizerprotokol. Alle 69 oprindelige sourcefiler forbliver frosne.

Genoptagelsen konstruerer optimizerens state ved præcis replay af gemte
objectivekald. Det er ikke en ny start fra den bedste kandidat. Åbne
reservationer og alle charges bevares i samme v2-campaign. Hver blok har
600 s reservation og 60 s stop-/eksportreserve; numeric failure og nyt
kandidatloft er terminale og må ikke udløse automatisk retry.

## Rækkefølge og accept

2-fit → egen base/fine-TRAIN/selection → 4-fit → egen validering → modelvalg
→ frossen 168 h åben prognose → PC-eksport af slut-Y → prequential finale.
Alle 1/2/4-familier kræves fortsat. Konvergens betyder kun COBYQA success
med status 0 og endelig objective; det er hverken global optimalitet,
identifikation eller accept af numerikken. Alle oprindelige præcisions-
gates og valideringsreserver gælder. F/G/WM holdes adskilt. Sammenligning
med WM udføres først, når BM-arbejdet er færdigt.

Kommando: root/f4/run_h2_cap_extension.py --run RUN --campaign V2_LEDGER
--source-commit EXTENSION_COMMIT --stage prefinal. Slutdata forbliver ulæst
indtil model og prognose er frosset. De gældende nominelle fysiske parametre
og ukendt tidligere slutugeeksponering rapporteres i I/J som før.
