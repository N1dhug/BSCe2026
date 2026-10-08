# Genoptagelse af G7 efter slettet chat

G7 slap efter den godkendte kontrol af gemte døgnkvitteringer den 8. oktober kl. 15.59 Copenhagen. Døgnstyringen er implementeret, og den sammensatte fixturekontrol består. Det numeriske døgnforsøg er fortsat åbent. Den aktuelle opgave er fire egne startfordelinger ved time 12, kompatible runtime-målinger og samlet admission før en separat numerisk præregistrering.

## Genfundet holdepunkt

[Det seneste completion-review](G7_DAY_COMPLETION_REVIEW.md) er indgangen. 132 gemte kvitteringer og fire posterioropdateringer er kontrolleret med stubbed numeriske kerner. Den deterministiske standalone-døgntiming er 2.331360998 s ved neutralpris. Den timing dækker én origin og udgør ikke runtimegrundlaget for den samlede graf. Alle 16 kompatible whole-job-rater er stadig ukendte.

Historisk debit er 1927.3372718589885 s af loftet 1930 s; rest 2.6627281410114847 s. 1800 s var oprindeligt et planforslag, ikke en genfundet eksplicit brugergrænse. De konkrete +110 s og +20 s var godkendt. Denne recovery ændrer ingen gammel charge, invocation eller approval og indebærer ingen ny modelkørsel.

## Gendannede bytes

`root/g7/recover_day_origins.py` bruger kun standardbiblioteket og det Git-gemte `files/output/bm_1/runs/g7_node_runtime_failed_20261007.raw.zlib`. Det kontrollerer komprimeret og ukomprimeret arkivhash, alle 50 filers størrelse/hash, 41 oprindelige sourcehashes og de to inputhashes fra den senere verificerede registrering. De oprindelige fejlkørselsfiler og ledger bevares som historik; fejlinvocationen genoptages ikke.

| Gendannet origin | Celler | Cutoff | Oprindelig SHA256 |
| --- | ---: | ---: | --- |
| M254 coarse | 257 | 12 h | `1cd343248607cc27baa61d4e52303fbfbf5c08ca7674fac0b9b3dce43649b3bc` |
| M254 fine | 513 | 12 h | `746e45c4603e724824687e535c2e7235f824e19ecbb6adcdd1c6ee0c0a7593e0` |

Begge har `g8-bernstein-m254-cell-average-v1`, fem rho-hypoteser og 255 koefficienter per hypotese. De gendannes byte-identisk uden nyt filter, basisfit eller projektion. Den aktuelle integrationsmanifest er også kontrolleret: 41 af 41 kildehashes matcher efter lokal materialisering med de registrerede LF-bytes. Windows-checkoutets oprindelige CRLF-bytes blev opdaget som en filidentitetsforskel, ikke en numerisk kodeændring. Git-tracked kilder skal ikke indgå i recovery-committen.

Forskellen fra det ældre arkivs source til integrationsmanifesten omfatter præcis `bm_1_g8.py`, `bm_1_g7.py` og `measure_owned_node_rates.py`. De gamle kildebytes bevares; gammel source må ikke forveksles med den nyere registrerede integrationssource.

## Hvorfor to FV origins stadig mangler

Den samlede driver kræver `m254_coarse`, `m254_fine`, `fv_coarse` og `fv_fine`. For hver hypotese j skal FV- og M254-filteret videreføre sin egen posterior `p_j(x | y_1:12)` og sine egne Bayes-vægte. Backendvalget for en fremtidig forward ændrer ikke, hvilket filter der skabte startfordelingen.

I `run_auxiliary_measurement.py` indlæser både mixture- og FV-evaluatorerne de samme M254-inputfiler. Disse tidligere FV-referenceforwards er gyldige som transport fra den deklarerede M254-startlov, men de leverer ikke selvstændige FV-filterposteriorer ved time 12. De må ikke ommærkes til driverens to egne FV origins. Arkivet i Git indeholder kun de to M254-inputfiler.

Råarkivet `G8_M254_513_forfining_raaresultater_20261006.zip` er ifølge [forfiningsreviewet](G8_M254_REFINEMENT_REVIEW.md) gemt i ChatGPT Library med timeorigins, fixture og frozen references. Hent dette arkiv og undersøg de originale FV-referencer/checkpoints; brug den oprindelige FV257/FV513-kæde, hvis arkivet ikke indeholder begge. Browseren i denne chat var ikke logget ind i Library. De registrerede Library-identiteter for nyere G7-arkiver er bevaret i de eksisterende JSON-filer; adgang til dem er ikke bevis for, at de manglende egne FV-filtertilstande findes i dem.

## Næste trin

1. Importér og hashkontrollér de originale egne FV257- og FV513-startfordelinger ved time 12. Bevar deres oprindelige references/proveniens; et NPZ-summary må ikke behandles som et fuldt origin uden en dokumenteret, entydig recordrekonstruktion.
2. Bind alle fire origins, integrationssource og det faktiske numeriske miljø i én context. Linux-rater kan ikke bruges som Windows-rater. Den lokale Python 3.11 har NumPy 1.26.4 og SciPy 1.17.0; ingen numerisk kørsel blev udført.
3. Forbered én konkret afgrænset runtime-måling og nyt samlet budget med alle 16 jobtyper. Bevar de eksisterende gates, searchcaps og stop/export-reserver. Et gammelt approvals færdiggjorte scope er ikke nyt målebudget.
4. Efter admission: commit og læs en særskilt immutable numerisk præregistrering tilbage før døgnforsøget. Genbrug kun præcise kompatible receipts; de gamle manglende projektionsaudits forbliver manglende.

BM færdiggøres før direkte WM-sammenligning. F4-H2 er et andet spor. Recovery giver hverken scientific accept, historisk validering eller et gennemført numerisk døgnforsøg.
