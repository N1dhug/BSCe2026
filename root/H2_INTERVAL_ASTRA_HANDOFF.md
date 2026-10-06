# H2 — konkret Astra XHigh-overdragelse

**Status efter Astra XHigh, 5. oktober 2026:** Opgaven nedenfor er nu behandlet.
Læs [metode og resultat](H2_TERMINAL_TILT_REVIEW.md) før nyt arbejde.
Terminalvis eksponentiel tilting består de små uafhængige referencer og
gennemfører begge smalle BM-døgn uden den oprindelige cancellation-fejl.
Tre fokuserede tests samt faktisk log-state-pause/resume er kontrolleret.
Den samlede tids-/rumforskel er dog .08754758 i log-likelihood mod grænsen .05;
mean/L1 består. Der er fortsat ingen historisk fit-readiness eller H2/H3-accept.
Den endelige rårapport er bevaret, inklusive det negative præcisionsresultat.

**Sol-krydskontrollen er nu færdig:** 1281 celler ved h=1/24576 tog 433.43 s.
Tids-scoreforskel .00216009 består; rum-scoreforskel .08538749 fejler .05.
Slut-mean/L1 består begge; begge gamle døgn er genbrugt uden genberegning.
Den ene 2561-cellers reference ved h=1/49152 er nu færdig: sammenligningen
med det gemte 1921-cellers døgn består score .03055462, slut-mean .00001396
og masse-L1 .00117516 mod .05/.002/.05. Runtime 1503.30 s; 15 timer er
genbrugt efter budgetpause med identisk log-state-fingerprint. Genkør ikke
denne reference. Næste er én parret timekontrol af 48 frekvenser/periode 1.5
versus 64/2 på det gemte 1921-grid, med samme guards; Sol er tilstrækkelig.
Se [råresultat og det afgrænsede næste trin](H2_TERMINAL_TILT_FINE_REFERENCE.md).
Det er en udviklingsreference før praktisk runtime/backend-valg. Historisk integration,
tidsvariation/breakpoints og data-/split-/baselinevalg er stadig åbne.
Tilting har ingen universel garanti for adskilte betingede blandinger, og
roundoffguard er heuristisk; se det konkrete analytiske modeksempel i reviewet.
En ny matematisk metodebeslutning kan kræve Astra, rutineintegrationen gør ikke.

Resten af dokumentet bevarer det oprindelige opdrag og den negative evidens
fra 5. oktober. Det er ikke en bestilling på at gentage de færdige kontroller.

Læs `AGENTS.md`, den aktuelle top af `root/F4.md`,
`root/H2_INTERVAL_CHARACTERISTIC_CONTROL.md` og `root/H2_HOURLY_INTERVALS.md`.
G-sporet arbejder samtidigt; bevar dets kode og legacy-januarkontrakt.

## Hvad allerede virker, og hvad der ikke virker

Observationen er normaliseret gennemsnitseffekt over en time:

\[
Y_i=\Delta_i^{-1}\int D(t,X_t)\,dt+N(0,\sigma_y^2).
\]

Johans råmåling er timeforbrug, foreløbigt kWh, hour-start. Det er ikke
slutpunkts-demand. Historiske population-/split-/eksponerings-/pris-/skala-/
baseline-/priorbeslutninger er fortsat prerequisites; ingen faktiske
historiske observationsværdier er læst i dette spor.

Den konservative joint state/akkumulator-implementation er historisk reference.
Dens gemte grove og finere kontroller fejler scoregrænsen 0.05. Den særskilte
characteristic-implementation er fortsat kun en syntetisk udviklingskontrol.
Log-amplituder bevarer underflowede tails. Positiv CN med symmetriske reward-
halvtrin består det faste lineære døgn og det nye fire-stykkers/brede støjcase.
B/U er konstante inden for hver time. Der er ingen generel andenordensclaim
for tidsvarierende fields, ingen historisk adapter og intet nyt optimizerfit.

De nye fixtures bruger samme saved observationsstream, Beta(2,3), C=0.15 dag,
k=3, Xi=0.8 og sigma_x=0.4. Kurverne holdes fælles over hele døgnet:

| Fixture | f-værdier | g-værdier | sigma_y | Resultat |
|---|---|---|---:|---|
| To stykker, knuder 0,.5,1 | 1,.3,-1 | 1,-.25,-1 | .04 | Alle fem replays afbryder i første time |
| Fire stykker, knuder 0,.25,.5,.75,1 | 1,.7,.2,-.55,-1 | 1,.45,-.1,-.8,-1 | .16 | Alle fem numeriske comparisons består |

To fælles kurve/støj-fixtures er ikke en isoleret undersøgelse af hver faktors
effekt, recovery, modelvalg eller validering af hele parameterdomænet.

## Den konkrete fejl

`src/validation/bm_1/interval_characteristic.py` lagrer den tiltede transform
som log-amplitude og conditional characteristic function for hver terminalcelle.
`characteristic_gaussian_update` inverterer hver række med en periodisk
Gauss-Fourier-serie. Log-amplituder løser underflow, men fjerner ikke subtraktion
af næsten ens Fourier-led i de små betingede tætheder.

Det smalle støjcase giver negative tætheder omkring -1e-14 til -1e-13.
Finere tid/rum, 64→128 frekvenser og periode 2→4 løser det ikke. Alle fem
replays stopper før første likelihoodbidrag. Råfilen er
`files/output/bm_1/runs/estimation/h2_interval_two_narrow_control_20261005.json`.

En uafhængig tre-state-reference har ingen drift/diffusion og konstante rewards
(.05,.5,.95), prior (.2,.5,.3), y=.6, c=.5, sigma_y=.04 og tilt=62.5.
Likelihood er præcis Gaussian per reward; total log-evidence er
-1.5182098888964164. Conditional kernel phi_sigma(c-reward) har tails cirka
3.2821e-27. Den integrerede transform giver negative værdier omkring -6.66e-15.
Se `h2_interval_cancellation_witness_20261005.json` i samme run-mappe.

En endnu mindre reference danner phase-kolonnerne direkte som
exp(i*omega*(reward-c)), uden state-solver eller tidstrin. Ved periode 2 og
128 frekvenser er truncation-bound 9.94e-58, mens computed tails er positive
3.33e-16 og 1.11e-15. Positivt fortegn er derfor heller ikke et bevis på korrekte
tails. Den eksisterende globale series-guard afviser dette vægtede case med
0.01397 mod 1e-8; der er ikke accepteret en forkert posterior.
Se `h2_interval_exact_phase_cancellation_20261005.json`.

## Oprindelig Astra-opgave (behandlet; aktuelt næste trin ovenfor)

1. Review disse to analytiske referencer og skellet mellem samlet evidence-
   error, terminal posterior-error og den rekursive brug af små log-tails.
2. Vælg én begrundet stabil/positiv eller tilstrækkeligt certificeret interval-
   likelihood, som kan demonstreres på disse små referencer. Forklar eventuelle
   approximationer og deres fejlbudget før en ny BM-panelkørsel. Sammenlign med
   eksisterende joint state/akkumulator- og characteristic-kode; undgå en bred
   spekulativ omskrivning eller et nyt Cartesian-grid.
3. Implementér kun en lille development-kontrol først med uafhængig reference,
   sigma_y=.04 og de eksisterende grænser. Bevar negativ evidens, state/process-
   definition, shared kurver og observationsstream. Ingen tail-clipping,
   probability floors, posterior-repair, cost-substitution, stille time-adaptation
   eller løsning ved at erklære snævrere støj ulovlig.
4. Før historisk adapter: korrekt tidsvariation/breakpoints, checkpoint med
   faktiske log-posteriors og et afgrænset runtimebudget skal kontrolleres.
   De nye unit-checkpoints er pilotresultater, ikke produktionens log-state.

Genkør ikke det beståede lineære CN-panel eller det brede støjpanel alene for
review. Ingen fuld historisk optimering. F4/H/I/J-gates og rå holdout bevares.
Rutineintegration kan fortsætte på Sol, når metoden er begrundet.

## Source, tests og genoptagelse

Numerisk kernel, begge control-moduler, CLI og de da gældende tests ligger i
[source-arkivet ecbb8d9](https://github.com/N1dhug/BSCe2026/commit/ecbb8d972de7d8d40a84e25846c50955736a7319).
Rårapporter bevarer lokal snapshot 77cc66b og fuld SHA256-map. Dette er matching
experiment-source, ikke et claim om identiske hele lokale/remote checkouts.
Final levering tilføjer en analytisk guard-test uden at ændre replay-kernelen.

Korte kontroller:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m unittest \
  tests.test_bm1_interval_characteristic tests.test_bm1_interval_curve_control -q
```

Den nye CLI er `src.exe.run_model.bm_1_interval_curve_control` med cases
runtime, linear-period, two-narrow og four-wide. Resume kræver original frossen
source/input/environment og `--resume`; færdige reports overskrives ikke.
Den parrede én-time-runtime var 4.8893→3.9805 s i median, 1.23x; dette er ikke
historisk optimizer-throughput. Et færdigt smalt case returnerer exit 2 som
deklareret negativ udviklingsevidens, ikke en CLI-crash.
