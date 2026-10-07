# G7 M254: runtime- og jobdesignreview efter deterministisk nattearbejde

7. oktober 2026. **Read-only review; ingen ny M254/FV/EM-kørsel.**
G8's accepterede egne M254257/M254513-filterposteriorer bevares, ligesom
G7's negative60s-pilot og den oprindelige vejlederbasis. F4-H2 er ikke en
adgangsgate og får ingen nye opgaver eller budgetdebit fra dette review.

## Hvad den eksisterende evidens faktisk siger

`g7_owned_handoff_20261007.json` og G7_OWNED_HANDOFF_REVIEW.md viser:
coarse to-timersbank var færdig ved afrundet17.61s elapsed. Den første
finebank blev afbrudt ved samlet60.001028361s. Den oprindelige v1 gemte
ikke coarsebanen før fine; derfor kan den ikke efterfølgende ommærkes til
et råvalideret completed forward eller bruges som genstartsposterior.

17.61s er elapsed siden invocation inklusive opsætning, ikke præcis
coarse-solver-only runtime. Fine-forwardens fulde runtime er ukendt;
60s er hverken dens målte tid eller en velunderbygget døgnrate. Det gamle
illustrative3×12×60≈2160s er ikke et formelt lower/upper-bound. Filterets
327.83s døgnrun giver heller ikke automatisk prisforwardens rate: antal
outputknuder, projections, kontrolforløb og initial law er forskellige.

V2 gemmer hver completed bank straks og projection-audits i finally.
Den rettelse er syntakskontrolleret; den har ikke givet ny numerisk accept.

## Gennemgået konkret kode og fund

`OwnedBankPriceEvaluator.evaluate` i `src/method/bm_1_g7.py` løber gennem
fem rho-noder sekventielt. Hver node starter fra sin EGEN frosne c_j.
`PreparedBetaProjector` genbruger geometri og aktive-face-SVD; cache er
allerede implementeret. Et generelt forslag om blot at cache geometri
ville derfor gentage eksisterende kode uden dokumenteret gevinst.

`evolve_projected_density` bevarer den eksisterende transport, CFL=.4 og
simplexprojektion på255 M254-koefficienter. Outputknuder og pris-breakpoints
indgår i integrationspartitionen. At skære i vilkårlige timeblokke kan
ændre de faktiske step/projektionspunkter og er derfor ikke automatisk et
rent checkpointfix. Nye basisfunktioner, færre knuder, større steps eller
andre støjparametre må ikke kaldes en runtimeoptimering af samme forsøg.

**Konkret mangel i den gamle admission-formel:** døgnets `day_checks`
prissætter tre policyers to mixture-grids og en fast30s-reserve, men mangler
et eksplicit led for de tre fine FV-policychecks, coupled EM, feedback og
quadraturkontrol. `day_search` bruger en FV-rate som proxy også for andre
search-/setupomkostninger. Prototypens formel har flere led, men30s-reserven
beviser ikke omkostningen til umålte EM/feedbackjobs. Det gamle
`admitted`-flag må ikke bruges som fuld G7-admission, selv hvis det engang
kunne blive true. Ingen search blev faktisk startet af den gamle pilot.
Legacyrunneren ændres ikke som sideeffekt af dette dokumentationsreview;
ny admission-adapter skal eksplicit erstatte dens utilstrækkelige jobestimat.

## Afgrænset næste implementering

1. Tilføj en G-specifik durable node-jobadapter omkring de eksisterende
   nodeforwards. Gem hver completed rho-node straks med fuld tidsbane,
   c/masser, objectivebidrag, raw diagnostics, hashes og timeledger. En
   completed node for samme kandidat må ikke genberegnes ved resume.
2. Jobidentiteten indeholder originhash, node, basis/grid, solver/output-
   partition, frosne parametre, target/baseline, eksakt prisvektor og source.
   Geometry-cache kan genskabes; koefficienter må ikke hentes fra en anden
   prisvektor eller fra FV. Bankresultatet dannes først, når alle fem
   navngivne nodejobs er komplette og samme informationscutoff er bevaret.
3. Før en node startes, reserveres dens fulde tilladte tid i den fælles
   G-ledger. Ved hard crash beholdes reservationen; ingen budgetreset eller
   automatisk gentagelse. Den nye adapter skal gemme også mislykkede jobs.
   Parallelisering er en særskilt fremtidig performancebeslutning, ikke
   antaget speedup eller noget startet i dette arbejde.
4. Vælg den mindste syntetiske kontraktkontrol for jobejerskab, fingerprints,
   pause/genoptagelse og bankreduktion med en billig stub. Brug ikke et nyt
   døgn eller de gennemførte G8-filtre som automatisk regressionstest.
5. Registrér først derefter en specifik måleopgave med et samlet loft,
   konkrete manglende rates og eksportreserve. Det er ikke en automatisk
   genkørsel af den gamle60s-pilot. Finegrids/egne origins og gates bevares.

Nodegranularitet er førstevalg, fordi den følger en allerede eksisterende
uafhængig sum og ikke ændrer en nodes integration. Hvis én node selv er for
dyr, kræver finere pausepunkter et nyt matematisk/numerisk review af den
oprindelige integrationspartition. Resume af SQP/Powell iterationer og
resume af en sandsynlighedsforward er forskellige problemer.

## Komplet jobgraf før numerisk admission

Lad t_{b,H} være målt omkostning for en HEL fem-nodebank ved backend b og
horizon H på dens egne frosne origins. Der skal også være målt setup,
objective-/quadratur- og eksporttid. For tre politikker (neutral, deterministisk,
stokastisk) er minimumsopdelingen

\[
 T_{\rm search}=N_d t_{d,H}+N_f t_{FV257,H},
\]
\[
 T_{\rm cross}=3(t_{M257,H}+t_{M513,H}+t_{FV513,H}),
\]
\[
 T_{\rm total}=T_{\rm setup}+T_{\rm search}+T_{\rm cross}
 +T_{\rm quad}+T_{\rm EM}+T_{\rm feedback}+T_{\rm audit/export}.
\]

Tidligere præregistreret96 objectivekald (48+48) er et prototype-loft, ikke
96 afsluttede fits. Antallet må ikke behandles som SciPy-iterationer uden
at tælle faktiske nye forwards. Eksakt-cachehits koster ikke et nyt solve.
Parrede samme-policy replays kan genbruges, hvis deres komplette rådata og
fingerprints matcher; den gamle v1-coarse uden eksport opfylder ikke dette.

Der skal være særskilte rækker for den valgte policyens feedback-propagation
og observation/projektion på de relevante egne grids; ingen fremtidig Y
må indgå i open-loop objective. Coupled EM følger den registrerede
255-komponent-mixture-draw, fælles Brownianincrements og de eksisterende
rand-/sampling-/timegates, ikke et gennemsnitligt parameterplug-in.

Horizon-ekstrapolation fra2h til24h er kun et planoverslag: initialisering,
CFL, aktive-face-skift og outputpartition kan være ikke-lineære i H og
policy. En1.25- eller1.5-faktor er en eksplicit planreserve, ikke en
bevist runtimegrænse. Alle umålte kritiske led markeres ukendte; admission
kan ikke blive true ved at sætte dem til0 eller gemme dem i en vilkårlig30s-post.

Efter denne nat er restbudget **701.419922s**, ikke en ny600/900/1800s-pulje.
Der er endnu ikke et komplett målt prisoverslag inden for denne rest.
Ingen fuld stokastisk optimizer er derfor frigivet af dette review.

## Matematisk mål og hvorfor dette arbejde stadig er nødvendigt

For den gemte fælles posterior p(x,rho_j)=w_j p_j(x) fryses w_j under en
open-loop policy, mens hver conditional law fremskrives fra sin egen origin.

\[
 J_{stoch}(u)=\int\sum_jw_j E_j[(D-A)^2]dt
 =\int\left[(\mu-A)^2+\sum_jw_jv_j+
 \sum_jw_j(\mu_j-\mu)^2\right]dt.
\]

Den deterministiske natteopgave løser tracking på modelbaner. Den afslutter
ikke within-/between-variance-leddene, M254↔FV-krydstjek, EM-validering
eller estimation→kontrol→ny observation. G8-filterets basisaccept er ikke
et bevis for denne kontrolpipeline. Begge mål beholdes, med deterministisk
BM/WM som den primære modelsammenligning efter Johans prioritering.

Formlen følger ekspansion af kvadratet og totalvariansloven, som udledt i
G7_OWNED_HANDOFF_REVIEW.md; M254 følger vejlederens tavle og G8-review.
Nodecheckpoint/jobgraf/admissionreglen er egne software-/forsøgsvalg, ikke
nye modelantagelser fra vejlederen. Ingen ny matematisk basis indføres.
