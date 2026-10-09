# BM: faglig historik og bevarede originaler

<a id="g-mathematics-source-index-20261008"></a>
Aktuel brug: [BM-guide](../../docs/bm/README.md). Gældende F/G-overblik:
[matematik](../../docs/bm/MATHEMATICS.md). Seneste faktisk verificerede arbejde:
[status](../../docs/bm/STATUS.txt). Arkivtekster giver ingen ny kørselsordre.

## F: model, numerik og historisk accept

- [Population, metadata og frosne inputs](F/data-contract.md)
- [Interval-likelihood, positive operatorer og terminale tilts](F/interval-likelihood.md)
- [Intervalmomenter, CDF og åben prognose](F/interval-forecast.md)
- [Estimation, identifikation og beregningsgraf](F/estimation-and-identification.md)
- [Numeriske fejl og kontroller](F/numerical-evidence.md)
- [Tidlige H2-afbrydelser](F/h2-early-recovery.md), [terminale stop](F/h2-terminal-recovery.md), [budgetvalg](F/h2-runtime-decisions.md), [slutvalg og H→I/J](F/h2-selection-and-handoff.md)

De fire sidstnævnte recoverytemaer bevarer lukkede ledgers, mistede jobs,
source-/checkpointadmission og H→I/J-kontrakten. De fem fagtemaer ovenfor
beskriver data, operatorer, estimation og numeriske fejl; gamle kørselsordrer
genaktiveres ikke, og budget-/genskabelsesoplysninger gentages ikke som ny matematik.

## G: udvikling, fordelinger og styring

- [Dynamik, punktprognoser og S3-valget](G/forecast-development.md)
- [Processtøj, FV/mixture/EM og negative båndresultater](G/stochastic-distributions.md)
- [Bayes-bank og Bernstein/Beta-basis](G/posterior-and-basis.md)
- [Prisstyring, energiligevægt og uafsluttet stokastisk evidens](G/control-and-evidence.md)
- Senere timeprotokol: [interval](../../docs/bm/G_INTERVAL_MODEL.md), [residual/usikkerhed](../../docs/bm/G_UNCERTAINTY.md), [weekday/bånd](../../docs/bm/G_WEEKDAY_STUDY.md)

## Genskabelse og sporbarhed

[Originaltræet](ORIGINALS.md) indeholder de præcise tidligere scripts, noter,
ledgers, src og tests i deres oprindelige indbyrdes stier. [Stikortet](old-paths.json)
forbinder alle fjernede/forkortede noter med emner og originalhashes. Historiske
JSON-manifests omskrives ikke. ZIP-pakning er arkivering og tælles med sit
udfoldede filantal; faglig konsolidering opgøres separat.

## Beregningsindgange og historiske drivere

Almindelig F/G-brug har tre indgange i [startguiden](../../docs/bm/README.md)
og [fitkalenderen](../../docs/bm/WEEKDAY_BACKTEST.md): gemt F-fit, gemt G-model
og nyt weekday-fit. De øvrige bevarede BM-CLI'er nedenfor har afgrænsede
reference-, eksport- eller forsøgsopgaver. At de findes, er ingen ordre til
at gentage en gammel kampagne. Deres egne kilde-/budgetkontrakter gælder.

| Bevaret modul | Selvstændig opgave |
|---|---|
| `src.exe.run_model.bm_1_h2_precise`, `src.exe.run_model.bm_1_history` | Afgrænsede H2-/TRAIN-likelihoodreferencer; hverken implicit baselinefit, populationsvalg eller holdoutevaluering. |
| `src.exe.run_model.bm_1_interval_characteristic`, `src.exe.run_model.bm_1_interval_curve_control`, `src.exe.run_model.bm_1_interval_terminal_tilt` | Syntetisk interval-/kurvekontrol og terminal-contour-reference med egne checkpoints. |
| `src.exe.run_model.bm_1_parameter_pilot` | Begrænset fysisk parameterpilot; ingen automatisk historisk kampagne. |
| `src.exe.run_model.bm_1.verify`, `src.exe.run_model.bm_1.refine` | Numeriske referencecases og forfining af faste planer. |
| `src.exe.run_model.bm_1.optimize`, `src.exe.run_model.bm_1.real_data` | Pris-/F3-referenceberegninger, adskilt fra normal prognose. |
| `src.exe.run_model.bm_1.estimate`, `src.exe.run_model.bm_1.curves` | Bevarede F-estimations-/H1-referencespor. |
| `src.exe.run_model.bm_1.outputs` | Eksport og gentegning af gemte I-/figurresultater. |
| `src.exe.run_model.bm_1.deterministic`, `src.exe.run_model.bm_1.deterministic_response` | G0–G4-referencefit og frosne udviklingsprognoser. |
| `src.exe.run_model.bm_1.deterministic_strategies`, `src.exe.run_model.bm_1.deterministic_validation`, `src.exe.run_model.bm_1.deterministic_weekday` | Registrerede strategi-/uge-/rolling-referenceforsøg. |
| `src.exe.run_model.bm_1.linear_noise`, `src.exe.run_model.bm_1.distribution_panel` | G6-støjdiagnose og afgrænset distributionspanel. |
| `src.exe.run_model.bm_1.s3`, `src.exe.run_model.bm_1.s3_forecast` | Særskilte ældre punkt-/interval-/residuallag; sidstnævntes API har WM-brugere. |
| `src.exe.run_model.bm_1.g_weekday_export`, `src.exe.run_model.bm_1.g_weekday_continue` | Kildebundet numerisk eksport og historisk continuation; ingen automatisk fitgenstart. |
| `src.exe.run_model.bm_1.g_wm_hourly`, `src.exe.run_model.bm_1.g_wm_period`, `src.exe.run_model.bm_1.g_wm_solver_extension` | Beskyttede G/WM-timeprotokoller; uændret kode og oprindelige kontrakter. |

De afsluttede G7/G8-drivere `g7`, `g7_deterministic`, `g7_matched_response`,
`g8`, `g8_basis`, `g8_basis_refine`, `g8_bernstein`, `g8_bernstein_refine`
er pensioneret fra aktiv CLI. Det samme gælder F's fire faste forsøgsdrivere
`bm_1_interval_search_pilot`, `bm_1_interval_search_stream`,
`bm_1_interval_joint_cn`, `bm_1_interval_joint_cn_day`.
Præcise originale bytes og tilhørende versioner findes i [originaloversigten](ORIGINALS.md#round3-retirements);
de genbrugelige metodekerner og deres faglige tests bliver aktive.

## Lokale resultater og nødvendige kopier

Resultaterne bliver ved deres registrerede stier under `files/output/bm_1`:

| Familie | Bevaringsgrund |
|---|---|
| `examples` og korte F-runtime-runs | Aktuel brugerprøve og afgrænset korrektheds-/runtimeevidens. |
| `benchmarks`, `f3_b`, `real_data` | Tidligere referencekontroller og selvstændige aggregater/figurer. |
| `runs/estimation` | Likelihood-, grid-, kandidat-, budget- og failurekæder; unikke delresultater bevares. |
| `runs/deterministic` og tidligere preview-runs | Egne fits, forecasts og proveniens; samme arrays betyder ikke samme forsøgsidentitet. |
| `g2b_distributed_v1` | Planen binder inputs, shards og checkpoints ved sti/hash; gentagne payloads er nødvendige lokale inputkopier. |

Tredje gennemgang fjernede kun tre fritstående byteidentiske resultataliasser;
deres kanoniske kopier og oprindelige backupmedlemmer er verificeret.
De øvrige identiske filer bevares på grund af job-, bundle-, checkpoint- eller
proveniensroller; enkelte alias-/readerbindinger er endnu ikke fuldt afklaret.
Unik eller uklar evidens bliver ikke slettet alene på grund af alder eller størrelse.
Det fulde lokale fil-/hashkort og restorekvitteringer findes under recovery-roden
i [status](../../docs/bm/STATUS.txt), `round3/structure`. Der er ikke offentliggjort
nye rådata eller private numeriske resultater, og ingen ny resultat-ZIP er oprettet.
