# BM: faglig historik og bevarede originaler

<a id="g-mathematics-source-index-20261008"></a>
Aktuel brug: [BM-guide](../../root/BM_GUIDE.md). Gældende F/G-overblik:
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
