# F4-0: sådan bruges forberedelsen

## Status

Pakken er et lokalt, testet forberedelsespatch mod `5a4793cd57d70a46beae2a6c758b4cd4e0556536`.
Det er ikke en færdig F4-estimator. Koden er ikke pushet/merged til GitHub.
`../F4.md` er den opdaterede arbejdsplan og statuskilde. `main.tex` inkluderer
`preparation.tex` og er den matematiske læsevej. Den genererede PDF ligger i
leveringspakken, ikke som en ny binær kildefil i repositorypatch'et.

F4-0 indeholder funktionelle kodeprimitiver og tests, ikke tomme API-stubs:

- `BM1ObservationSeries`: owned/read-only punktmålinger, eksplicit time unit,
  sidekonvention og missing-mask. Intervaldata afvises indtil en model er valgt.
- `gaussian_log_likelihood`, `mass_to_log`, `bayes_update`: ét korrekt
  observations-/hypoteseupdate med log-evidence og bevaret log-posterior.
- `conservative_mass_transfer`: overlapbaseret overførsel ved samme support.
- `PreparedBetaProjector`, `beta_cell_average_basis`: opt-in interface og basis
  til den eksisterende numerical kernel. F3-defaults er uændrede.

Et filter kræver fortsat en korrekt prediction/update-loop, data-/problemkontrakt,
streaming, forecasting og validering. Parameterbank kræver eget conditional
state-filter for hver hypotese. Disse ting påstås ikke implementeret her.

## Kør fra repositoryroden

```powershell
python -B -m unittest tests.test_bm1_estimation -v
python -B -m unittest discover -s tests -q
python -B root/f4/verify_preparation.py
```

Den sidste kommando viser reproducerbare statiske beregninger. Den skriver ikke
filer uden `--output`. Et eksplicit eksempel på ny resultatplacering er:

```powershell
python -B root/f4/verify_preparation.py --output files/output/bm_1/runs/f4_0/preparation_checks.json
```

Det er en videnskabelig kontrol af enkeltmålingen og basisrummet, ikke en
parameterfit- eller ugeforecastkommando. Der er bevidst ikke tilføjet et tomt
femte `estimate.py` endnu.

## Byg matematiknoterne

```powershell
cd root/f4
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Alternativt fra repositoryroden:

```powershell
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build root/f4/main.tex
```

Upload begge `.tex`-filer til Overleaf og vælg `main.tex`, hvis noterne redigeres
der. Ingen eksterne figurer, BibTeX-kørsler eller medleverede fontfiler kræves.
Genererede LaTeX-filer i `root/f4/build/` skal ikke committes.

## Matematik → kode → test

| Regel | Kode | Kontrol |
|---|---|---|
| Punktmåling kan ligge uden for [0,1] | ObservationSeries + Gaussian helper | Physical-range- og missing-mask-tests. |
| Bayes på masser uden ekstra dx | `bayes_update` | Direkte diskret reference og grid-uafhængig konstant likelihood. |
| Tiny logvægt er ikke ægte nul | `BayesUpdate.log_mass` | Vægt under exp-underflow kan genvinde betydning. |
| Konservativ pce-overførsel | `conservative_mass_transfer` | Analytiske overlap og tilfældige nonuniforme grids. |
| QP-gap kontrollerer solver, ikke basis | `PreparedBetaProjector.project` | Samme løsning som eksisterende primitive; stor statisk basisfejl kan stadig forekomme. |
| CDF-integration er en ny diskret basis | `beta_cell_average_basis` | Analytiske Beta(2,1)/(1,2)-integraler, højre hale og smalle komponenter. |
| Variansgulv | Eksisterende `beta_mixture_variance_floor` | Ny enkeltmålingstest plus gammel F2-test. |
| Forecastvarians kan falde | Itô-udledning i noter | Uafhængig quadrature i `verify_preparation.py`. |

## Miljø og test-evidens

Baseline: 207 tests. Kandidat: 253 tests, heraf 46 nye. Lokalt runtime:
Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0. Python 3.11-grammatik er checket;
dette er ikke en 3.11-runtimekørsel eller en ny GitHub Actions-kørsel.
Der er ikke ændret på `root/environment.yml`. Det nye komplementære beta-CDF-kald
kræver SciPy >= 1.11 (den testede version opfylder dette).

Testlogs, quadrature/projektionsresultater, miljøversioner, source-manifest og
checksums er i leveringspakkens `evidence/`. Den eksisterende WM2-docstring giver
en SyntaxWarning i Python 3.13; den er ikke skjult eller ændret af F4-0.
Inputworkbooks og lange historiske scientific runs er ikke genkørt.

## Næste trin: F4-A

Læs noteafsnit 1–5, og fastlås første forsøgsdefinition: initialprior, kendte/frie
parametre, observationstider, price-side, synthetic inputfamilie, forecast-origin
og præregistrerede acceptmål. Herved omsætter vi det forberedte matematiske
udkast til en konkret F4-A-kontrakt. A–J-statusfelter markeres først færdige efter
det faktisk gennemførte trin.
