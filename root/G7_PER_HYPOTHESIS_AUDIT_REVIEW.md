# G7: atomisk projektionsaudit og næste samlede forsøg

Status 7. oktober 2026. Filnavnenes suffix 20261008 er et artefakt-ID;
kontrollen blev udført i sessionen dateret 7. oktober. Den allerede gemte
præregistrerings datoetiket bevares uændret som provenance.

## Resultat og merge

PR115 er merged som d5b061195b9f88c22a375a3d343cd12ac1c00f1e.
Merge-træet 1628932e9090d8d22557062be7b50434e4bd6bd2 matcher det
reviewede PR-head. Fjorten øvrige gamle G-branches var allerede ancestors
af main. Den ene resterende G-notebranch indeholdt unikke historiske noter.

Disse noter er merged via PR116 som
02812cab540b414d59ad7d06124e46d5bca039d2, træ
546655bbb7dd6441caa32ff7d39d1b9b89ca7ddb. Konflikten i G_JANUARY blev
løst med den aktuelle status bevaret og de tidligere auxiliary-noter
markeret som historik. Originale manifest-/next-step-bytes er bevaret;
et historikindeks henviser til de nyere reviews. Ingen gamle budgetter
eller acceptstatusser blev genaktiveret. Merge-træet matcher det løste
head. F's åbne PR106 er draft og blev ikke merged. Ingen branches slettet.

Den nye G-adapter blev committed før kontrollen:
b58434384f7c7ad7b452784835f1ca259a6406f5. Registreringen blev læst tilbage
byte-identisk. Kode og den faktiske kontrol er særskilte fra gamle
numeriske resultater.

## Hvad der er indført

AuditedBankControl er en udtrykkeligt tilvalgt underklasse af den
eksisterende CheckpointedBankControl. Den kalder fortsat den samme native
forward, bankreduktion, sourceguard, deadline og NodeJobStore. Native
src/tests/workflows og de 41 frosne numeriske kildefiler er uændrede.

For hver parameterhypoteses mixture-forward opsamler en ny SolverAudit
antal solver-kald og kontrollerede projektioner, maksimal cellemasse-L1,
skaleret optimalitetsgap, koefficienternes massefejl og solverdiagnoser.
Den underliggende solver og dens geometriske cache genbruges; kandidaten
starter fortsat fra sin egen frosne conditional origin.

Audit lægges inde i samme payload som forwardens arrays. Native
NodeJobStore hasher og gemmer denne samlede payload med sin eksisterende
atomiske filskrivning, fsync og journalrækkefølge. Der er ingen særskilt
auditfil, som kan gå tabt mellem to writes. Ved reuse læses de gemte
målinger; en ny tom live-audit må ikke erstatte dem.

Kontrakten får et eksplicit audit-policy-felt og den nye adapterhash.
Gamle unauditerede receipts kan derfor ikke blive samme jobidentitet.
Mixture kræver positive solver-kald og checked_projections=solver_calls.
Manglende, forkert ejet, ikke-endelig eller tom audit afvises før
bankkombination. FV gemmer i stedet et eksplicit “not applicable”:
FV anvender ikke mixture-projektion, så nul solver-kald er ikke en
bestået mixture-gate.

Det er en forward-audit. Den auditerer ikke automatisk
assimilate/Bayes-opdateringens særskilte posteriorprojektioner. Disse
diagnoser og de rå evidences skal også gemmes i det senere feedbackjob.
Adapteren gør heller ikke den gamle launcher til en døgnlauncher.

## Matematik og forsvar

Vejlederens basis består af 255 normaliserede tæthedsfunktioner:

$$
b_{i,254}(x)=255{254\choose i}x^i(1-x)^{254-i}
=\operatorname{Beta}(x;i+1,255-i),\quad i=0,\ldots,254.
$$

Hver betinget lov repræsenteres som
$p_j(x)=\sum_i c_{ji}b_{i,254}(x)$ med $c_{ji}\geq0$ og
$\sum_i c_{ji}=1$. Basisfunktionerne integrerer til 1, så
$\int p_j=1$ følger af koefficientsummen. Dette beviser normalisering
af repræsentationen, men siger ikke, hvor tæt den er på transportmålet.

Lad $Q_{\ell i}$ være basisfunktion $i$'s masse i celle $\ell$, og
$\widetilde m$ den rå cellemasse efter et transporttrin. Den eksisterende
projektion er det simpleksbegrænsede least-squares-problem

$$
c^\star=\arg\min_{c\geq0,\;\mathbf1^\top c=1}
\frac12\|Qc-\widetilde m\|_2^2.
$$

På vores uniforme grid er density-LS og cell-mass-LS proportionale med
en positiv konstant; derfor har de samme minimizer. Projektionen kan
bevare masse, selv om residualen er stor. Det relevante særskilte
fitmål er derfor

$$
e_j=\max_{n\text{ i forward }j}
\|Qc^\star_{jn}-\widetilde m_{jn}\|_1.
$$

Dette er maksimum for ét forward, ikke summen af fejl gennem hele
horizonet og ikke en garanti for den kontinuerte FP-fejl. Vi gemmer
desuden solverens eksisterende skalerede gap og
$|\mathbf1^\top c^\star-1|$ som forskellige diagnoser.
Auditens rolle er at bevare målingerne; en registreret forsøgsreview
skal derefter sammenligne dem med sine uændrede tolerancer.
“complete audit” betyder komplet dækning, ikke automatisk numerisk accept.

Hvis $R_j$ er forwardresultatet og $a_j$ auditrecorden, gemmes

$$
P_j=(R_j,a_j),\qquad H_j=\operatorname{SHA256}(\operatorname{encode}(P_j)).
$$

Jobidentiteten binder også origin, parameter, pris, backend, numerik,
source og auditpolicy. Hashen verificerer byteintegritet under
kontrakten; den er ikke et matematisk korrekthedsbevis eller en signatur
mod en angriber med skriveadgang.

Ved process-tab før en komplet receipt beholdes reservationen, og jobbet
er uafsluttet. Ved tab efter komplet receipt, men før ledger-commit,
findes både $R_j$ og $a_j$; restart kan genbruge dem og beholde den fulde
reservation. Det er præcis den verificerede crash-position.
Process-tab midt i forwarden giver stadig ingen komplet audit.

Valget er nyttigt her, fordi G's bank har dyre individuelle forwards:
kontrollen skal overleve samme afbrydelse som resultatet. En samlet
live-audit uden per-hypotese persistence kan miste de tidligere målinger;
en særskilt auditfil ville kræve endnu en transaktionsmekanisme.
Indlejring genbruger den eksisterende durable receipt uden at ændre
M254-basis, dynamik eller kontrolmål.

Det stokastiske mål bevares:

$$
J(u)=\int\sum_jw_jE_j[(D_t-A_t)^2]\,dt
=\int\left[(\mu_t-A_t)^2+\sum_jw_jv_{jt}
+\sum_jw_j(\mu_{jt}-\mu_t)^2\right]dt.
$$

Her $\mu_{jt}=E_j[D_t]$, $v_{jt}=\operatorname{Var}_j(D_t)$ og
$\mu_t=\sum_jw_j\mu_{jt}$. Identiteten følger ved at udvide kvadratet,
bruge $E_j[D_t-\mu_{jt}]=0$ og
$\sum_jw_j(\mu_{jt}-\mu_t)=0$. Mean tracking samt within-/between-varians
rapporteres fra samme forwards; de er ikke tre forskellige modelkørsler.
Den gamle deterministiske kontrol er fortsat en separat reference og
kan ikke genskabes ved blot at fjerne variansleddet fra denne bank.

## Den faktisk udførte kontrol

Én stub-invocation: seks checks bestod, proceswall 0.782377885 s,
check-body 0.008191904 s, nul native BM/FP/EM/prisforwards. Den bruger
native receiptstore, kontraktadapter og SolverAudit, men erstatter den
videnskabelige forward og bankkombination med små stubs. Det er en
afgrænset persistence-/kontroltest, ikke en ny numerisk G7-validering.

1. Wrappers og driftguard gendannes efter process-loss-undtagelsen.
2. Hash-verificeret resultat og komplet audit findes før ledger-commit.
3. Restart genbruger dem uden ekstra forward eller reservationsrefusion.
4. Hash-valid receipt uden audit afvises uden replay.
5. Mixture-audit med nul kald afvises.
6. FV kræver det eksplicitte not-applicable-record.

En parent-preflight med forkert filpræfiks stoppede før imports og før
kontrolstart; kun parentens path-resolution blev rettet. Adapter/check
og præregistreringshashes blev ikke ændret. Intet gammelt numerisk
resultat eller testpanel blev genkørt.

Konservativ charge 5 s beholdes trods kortere faktisk wall:
G-debit 1846.3372718589885 s af godkendt 1910 s;
rest 63.662728141011485 s. Alle gamle charges er bevaret.
Registreringen/resultatet findes i g7/audit_check_*_20261008.json.

Reproduktion fra en checkout med de registrerede 41 native sourcebytes:

    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -B root/g7/check_audited_node_control.py

Kommandoen er dokumentation, ikke en opfordring til at gentage kontrollen.

## Begrænsning og næste trin

De gamle quad-forward 0/1/2 har stadig ingen gemte projection-fit-maxima.
Vi har hverken genskabt dem fra banernes normalisering eller genkørt dem.
Deres tidligere åbne numeriske/scientific status forbliver åben.

Det næste samlede forsøgsforslag er beskrevet i
[G7_INTEGRATED_EXPERIMENT_DESIGN.md](G7_INTEGRATED_EXPERIMENT_DESIGN.md)
og den maskinlæsbare g7/integrated_experiment_proposal_20261008.json.
Det er et konkret design, ikke en launchregistrering eller automatisk
admission. M254 bevares som primær stokastisk søgerepræsentation;
FV/EM er interne referencer. 24h-rates og samlet budget er ukendte.
BM færdiggøres før direkte WM-sammenligning. F-sporet bevares.
