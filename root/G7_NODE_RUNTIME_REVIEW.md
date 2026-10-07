# G7: kort runtimeforsøg fra egne gemte M254-posteriorer

7. oktober 2026. Det særskilt registrerede forsøg er gennemført og råreviewet.
Ti faktiske native nodeforwards: fem på M254513 og fem på M254257, alle på
12→14h med priser [.5,.5]. Ingen optimizer, FV/EM-replay, feedback eller
historiske observationer. Den deterministiske BM/WM-reference med bevaret
forbrug bevares. Stokastisk døgn-G7 er fortsat uafsluttet.

## Registrering, fejl og faktisk udførelse

Før source/kørsel: node_runtime_registration_20261007.json på b9fee8f.
Første source bcd3d357b64e2e3743378976d970733a27260fce.
Loft200s:150 worker +30 stop/eksport +20 konservativ reviewcharge.
De oprindelige fine/coarse hour12-filer matcher SHA256
746e45c4603e724824687e535c2e7235f824e19ecbb6adcdd1c6ee0c0a7593e0
og1cd343248607cc27baa61d4e52303fbfbf5c08ca7674fac0b9b3dce43649b3bc.

Første invocation stoppede i setup ved den eksisterende bit-exact basisguard:
0 actual forwards, worker1.067494422s, eksport.618243206s, charge inkl.review
21.685737628s. Original source/input/log/ledger/registration-bytes er gemt
i files/output/bm_1/runs/g7_node_runtime_failed_20261007.raw.zlib med
index node_runtime_failed_archive_20261007.json. Ingen charge slettet.
Det var en loadingfejl, ikke en observeret transport-/basispræcisionsfejl.

En særskilt addendum-registrering bevarer denne debit og tillader én ny
invocation, ikke automatisk retry af modeljobs. Ny scope højst160s:
120 worker +20 stop/eksport +20 review; sammen med første charge højst
181.685738s under det oprindelige samlede200s-loft.
Execution source:872e77714d1bc5848d8dce5b1fac058417b796cf.
41 importerede/runner-kilder og begge inputhashes er frosset/verificeret.
Ét billigt loadercheck består på1.608989s process/.838s body, uden transport.
De tidligere node-/crashtests genkøres ikke. Deres sourcefixture læses nu
fra aktuel kode under testen; de historiske receipt-manifester bevares.

Fine grid måles først, fordi den gamle60s-pilot manglede et completed fine
forward. Derefter coarse. Baseline, target, priser, basis og solvertrin
ændres ikke: B=.5+.1sin(2pi*t/24), A=.5; objectivepartition1/96h,
solvermax1/192h på513 og1/96h på257. Hver rho-node bruger sine EGNE gemte c.
BLAS1 og én serial writer. Modelworker har et eksternt subprocess-stop
ved120s; kooperative per-node-reservationer er20s fine/5s coarse.
Ingen hard stop eller node-retry blev nødvendig.

## Hvorfor verified-byte-indlæsning var nødvendig

Genberegning af samme deklarerede cellebasis er her ikke bitvis identisk
med de bevarede arrays, selv om kildehash og versionsnavne er verificeret.
Fine:77 elementer ændret, max abs3.019806627e-14. Coarse:34 elementer,
max abs7.105427358e-15. Den konkrete system-/specialfunktionsårsag er ikke
isoleret; versionsnavne alene beviser ikke bitvis numerisk reproduktion.

Vi udskifter IKKE de oprindelige basisværdier med nye værdier.
Standard from_record uden certificate kræver fortsat np.array_equal.
Den eksplicitte from_verified_bytes(payload,expected_sha256) gør:
1. Verificér det eksternt registrerede SHA256 for HELE originalfilen.
2. Bevar originalt basisarray og grid; beregn deres eksakte float64-bytehash.
3. Verificér supervisor M254 og en særskilt roundoff-only familiegrænse.
4. Bevar de eksisterende krav om realitet, positivitet, masse og c→mass-paritet.
Certificate følger med typed forecast-/feedback-origins og checkpointkontrakten.
Default origins/legacy-recordformat ændres ikke; certificate er et eksplicit
optional metadatafelt. Ingen fysisk parameter, projektion eller tolerance
for modelaccept er lempet.

For basisarrayets cellemiddel b_ik og lokal generator \hat b_ik er grænsen
\[
 \eta=\max_i\sum_k\Delta x_k|b_{ik}-\hat b_{ik}|
 \leq64\epsilon_{\mathrm{float64}}=1.4210854715\cdot10^{-14}.
\]
Den målte eta er1.1426856862e-16 fine og5.5295154534e-17 coarse.
Dette er vores eget portability-kriterium, ikke en værdi fra vejlederen.
Det bruger integreret komponentmasse frem for en løs relativ cellegrænse
i tails. For simplexkoefficienter er
\[
 \left\|\sum_i c_ib_i-\sum_i c_i\hat b_i\right\|_{1,h}
 \leq\sum_ic_i\|b_i-\hat b_i\|_{1,h}\leq\eta,
 \qquad c_i\geq0,\quad\sum_ic_i=1.
\]
Denne ulighed begrunder familiecheckets skala. Den er ikke en ny fejlgaranti
for transporten, og hverken ny generatorbasis eller nye c anvendes i runnet.
Bytehashen håndhæver den faktisk ejede basis; store familieforskelle
afvises også med et nyt whole-record-hash. Loadercheck verificerer begge
afvisninger, default strict guard og roundtrip med identiske basis/c.

M254 er fortsat vejlederens normaliserede Bernstein/Beta-familie
\[
 b_{i,m}(x)=(m+1){m\choose i}x^i(1-x)^{m-i},\quad m=254.
\]
Cellerepræsentationen, oprindelige legacy-veje og LS/FP-kernel bevares.
Se kildekort/udledning i G8_POSTERIOR_BASIS_REVIEW.md.
Det nye indlæsningsinterface og toleranceforsvar ovenfor er egne softwarevalg.

## Kontrolmålet og hvorfor nodecheckpoint er korrekt

Vægtene w_j er posterior-sandsynligheder ved cutoff. De fryses under åben
prisplan; ingen fremtidig observation læses. Med mu_j=E_j[D], v_j=Var_j(D)
og mu=sum_j w_j mu_j gælder
\[
 J(u)=\sum_jw_jJ_j(u)
 =\int\left[(\mu-A)^2+\sum_jw_jv_j+
                 \sum_jw_j(\mu_j-\mu)^2\right]dt.
\]
Det følger af kvadratudvidelse og totalvariansloven. Ingen ny straf opfindes.
Hele nodeforwards er uafhængige betinget på samme u og startlaw, derfor
gemmes de uden nye integrationspauser. Native transportkald og objective-
beregning er uændrede; den eneste G7-metadataændring viderefører basis-
certificate til typed predicted/assimilated origins. En faktisk observation
og ny vægtopdatering er en separat senere opgave.

## Faktiske resultater og begrænsninger

| Måling | M254257 coarse | M254513 fine |
|---|---:|---:|
| Hele bankens walltid, inkl.nodefiler/reduktion, ekskl.cold setup | 16.828349s | 37.961317s |
| J | .00374196357346 | .00371551746386 |
| Biasbidrag | .00227896673988 | .00226131112437 |
| Variansbidrag | .00146299683358 | .00145420633949 |
| Maks.single-projection L1 | 6.14684084e-6 | 1.65157879e-6 |

Fine-nodeforwards:5.210170/9.140471/8.054094/7.471967/6.601046s.
Coarse:3.520098/4.689202/3.950143/2.229641/1.608936s.
Disse tider er fra den faktisk udførte rækkefølge med genbrugt geometrycache;
de er ikke fem uafhængige cold-start-rates. Ingen målt parallel speedup.

| Parret kontrol | Målt | Uændret gate |
|---|---:|---:|
| J-forskel | 2.64461096e-5 | .001 |
| Conditional slutmasse-L1 efter konservativ transfer | .011187752 | .05 |
| Marginal slutmasse-L1 | .005225045 | .05 |
| Slut-state-middel | .000345296 | .005 |
| Slut-demand-middel | .000123426 | .005 |
| Slut-state-varians | 4.85685428e-5 | .002 |
| Slut-demand-varians | 6.41210422e-6 | .002 |
| State-/demand-CDF | .001660634/.001120996 | .03 |
| State-/demand-kvantil | .000803836/.000265689 | .02 |

Alle disse korte gates består. Coarse og fine starter fra deres egne
tidligere filterposteriorer; forskellen omfatter derfor inherited filter-
repræsentation og nyt forward, ikke ren tidsdiskretiseringsfejl.
En neutral to-timerspris er ikke et bevis for andre policies, et døgn,
FV-paritet, EM, feedback, historisk likelihood eller fysisk/kausal prisrespons.

Worker57.762549938s, dataeksport6.683757768s. Charge inkl.review84.446307706s.
Begge invocations samlet106.132045334s. G-debit1224.712123334s,
**575.287876666s tilbage** af1800s. F-budget ændres ikke.
Dataaudit/ny ZIP/.838s loadercheck og let syntax-CI dækkes konservativt af
de registrerede reviewcharges; modelkørsler er ikke gentaget i reviewet.

## Råreview og recovery

Read-only audit:12 completed ledgerentries (2 setup,10 native noder), ingen
duplikerede attempts. Alle12 payload-/jobhashes og to kontrakter er kontrolleret.
Egne initial-c matcher eksakt. Demandmomenter er genberegnet fra de gemte
masser uden transport: max difference0. J/bankreduktion og alle lagrede
mass/c-baner matcher. 41 sourcehashes matcher frossen source. Ingen nye forwards.

G7_M254_node_runtime_20261007_data.zip indeholder fulde data, kontrakter,
ledger, logs og final invocation. ZIP-bytes og hver originalfil er hash-
verificeret; SHA256 ad730a148e770fffeeccdfcfff13af30fc33510d0bbcba9b988c15767af1e80d.
Dataarkivet er gemt separat; identity/index ligger i
root/g7/node_runtime_raw_audit_20261007.json. Kode ligger i executioncommit,
originale input/sourcebytes i det første fejlarkiv i Git, så kode ikke
duplikeres i dataarkivet. Begge fejl-/succesregistreringer og receipts bevares.
Det tidlige rå-zlib-eksportarkiv og ZIP har forskellige emballager;
ZIP indeholder den endelige invocation med eksporttid/charge.

Faktisk interrupt→resume-mekanik genbruges fra den allerede beståede
stubkontrol. Dette run blev ikke bevidst afbrudt eller genkørt som resume-test.
Setup-receipts er metadata, ikke serialized evaluator-objekter: en senere
launcher skal reetablere geometri med en NY setupcharge, mens native completed
nodejobs genbruges under samme fulde kontrakt/ejerledger. Den nuværende
målelauncher nægter automatisk resume; den er en single-invocation måler.
Dette softwarearbejde er åbent før en lang resumable kontrolintegration.

## Komplet budgetgraf og resterende arbejde

De to2h-rates er gemt som to2h-rates. Ingen af dem kvalificerer som en24h-rate
i den komplette14-jobgraf. Alle døgnrates er derfor stadig ukendte og
budget_admitted=false; optimizer/full_day/scientific_accept=false.

Som tydeligt mærket planillustration, uden admission:
\[
 1.25\cdot3\cdot12(16.828349+37.961317)
 =2465.535\ \mathrm{s}\approx41.1\ \mathrm{min}.
\]
Dette dækker KUN tre policies på to M254-grids, før FV/search/EM/feedback.
Lineær horizon-ekstrapolation er hverken en målt rate, en lower bound eller
et worst-case-bevis. Illustrationen peger dog på en mulig konflikt med de
resterende9.59min; fuld døgnkørsel frigives ikke automatisk.
Det oprindelige30min-kampagneloft udvides ikke af denne konklusion.

Næste konkrete G7-opgave er et budgetteret integrationdesign med de gemte
noder som reference: reetablering af setup uden replay af completed noder,
måling/prissætning af FV/EM/feedback og få policychecks med fulde egne states.
En afgrænset2h-integration kan demonstrere estimation→kontrol→observation,
men erstatter ikke det registrerede døgnkrav. Afklar fuldt døgn-design og
budget, før optimizer startes. Bevar den gamle negative60s-pilot og alle gates.

Resterende G7-leverancer:
1. Budgetadgang/rategrundlag og samlet launcher/recovery for hele jobgrafen.
2. Prisoptimering og samme-policy M254/FV-/quadraturchecks.
3. Coupled EM med samplingusikkerhed og faktisk observation→feedback.
4. Samlet matematisk/resultatmæssig overdragelse; ingen globaloptimalitetsclaim.

Planlægningsskøn for aktivt kode-/review-/notearbejde: **2–4 timer for G7**,
hvis der findes et godkendeligt lille design uden solveromskrivning.
Det er et eget arbejdsskøn, ikke en målt runtime eller et løfte om døgnaccept.
For G-kernen inklusive afsluttende prognose-/BM/WM-rapportering og
historisk data-/enhedsafklaring: **4–8 timers aktivt arbejde som grov ramme**.
Faktisk beregning er ekstra og endnu ikke komplet prissat. Nyt helårsfit,
nyt historisk intervalfilter eller større runtimeombygning er ikke i dette
skøn. G4/G6/G8 gennemført evidens skal genbruges; F4-H2 er en anden opgave.
Sol High er fortsat passende; ingen lokal modelkørsel kræves nu.
