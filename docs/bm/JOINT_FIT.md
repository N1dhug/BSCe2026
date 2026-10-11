# Fælles efteroptimering af Thomas-fittet

Den oprindelige metode skifter mellem fælles modelparametre og fem individuelle starttilstande. Den nye, valgfrie afslutning optimerer alle disse variable samlet med samme deterministiske forbrugsresidualer og samme grænser. SciPy `least_squares(method='trf', x_scale='jac')` starter fra det gemte fit. Der tilføjes ingen parameterstraf.

Et resultat accepteres kun, hvis den genberegnede trænings-SSE ikke stiger. Ved tidsgrænse eller et ugyldigt forslag beholdes det bedste brugbare punkt. For stykvis lineær og Bernstein prøves også det eksisterende lineære referencefit, indlejret i den fleksible familie; den bedste af de to starter bruges.

Førsteordens optimalitet måles med SciPys grænsekorrigerede `optimality`. Det nye `converged` kræver `optimality <= joint_gtol`. Et `ftol`/`xtol`-stop kan derfor have `optimizer_success=True`, men `first_order_passed=False`. Den gamle konvergensstatus byggede på forbedringen mellem alternerende runder. Kolonnerne er navngivet forskelligt i sammenligningen, fordi kriterierne ikke er identiske. Ingen af dem er et bevis på et globalt minimum eller bedre holdout-prognoser.

Standardbudgettet er 80 funktionsiterationer (`max_nfev`, uden alle numeriske Jacobian-kald), 20 sekunder pr. efteroptimering og `joint_gtol=1e-5`. Tidsgrænsen kontrolleres mellem residualberegninger og er derfor blød. Optimeringen kan stoppe tidligere på SciPys `ftol`/`xtol`-kriterier. De egentlige residualkald og stopårsager gemmes særskilt.

## Sammenlign med den gemte uge, 2.–8. maj 2022

Kør fra repoets rod på main med det eksisterende Python-miljø:

```powershell
python -B -m src.exe.run_model.bm_1.compare_fg_week --reference files/output/bm_1/runs/fg_week_2022-05-02_2022-05-08 --output files/output/bm_1/runs/fg_week_joint_2022-05-02_2022-05-08 --workers 8
```

Standardvalgene er alle seks f/g-par, syv døgnprognoser pr. familie og MPC fra hver af de 168 timer med lead 1–8. Otte worker-processer bruger hver én BLAS-thread.

De gemte parameterfits, starttilstande og forberedte historiske data genbruges. Der importeres ikke nye træningsdata eller fremtidigt forbrug til fittet. Eventuelle manglende referencejobs færdiggøres med den gamle alternerende opskrift under den nye outputmappes `reference_completed/`; den oprindelige mappe er skrivebeskyttet for denne runner. Beta-CDF bruger rettelsen af hældningsgrænsen ved shape-parametre tæt på 1. Derfor registreres både referencefilens trænings-RMSE og referenceparametrenes genberegnede trænings-RMSE under den aktuelle numerik.

Efteroptimeringen bruger kun de samme fem træningsvinduer. Døgnprognosen bruger igen den empiriske blanding af de fem nye starttilstande. MPC genestimerer diffusionen fra de nye træningsresidualer, assimilerer observationen ved origin og genberegner prognosen. Den numeriske kontrol er den samme som i referencekørslen.

## Resultater

- `comparison.csv`: gammel/ny RMSE på præcis samme par af færdige jobs, antal observationer, træningsforbedring, konvergensmål, numerisk kontrol og tidsforbrug.
- `fit_comparison.csv`: sammenligning pr. fit inklusive genberegnet referencefejl, stopårsag, optimalitet og valget af startpunkt.
- `comparison_by_lead.csv`: MPC-sammenligning pr. lead 1–8 timer.
- `comparison.png`: figur med gammel/ny RMSE og baseline.
- `daily_summary.csv`, `mpc_summary.csv`, `week_summary.csv`, `mpc_by_lead.csv`: den nye kørsels egne oversigter.
- `runtime.json`: faktisk vægtid og worker-antal for hver fortsættelsessession.
- Mapper pr. dato/familie/origin med parametre, `joint_fit`, prognoser og numeriske kontroller. Worker-logs ligger i `logs/`.

`n`, `paired_jobs` og `expected_jobs` viser, om en sammenligning er komplet. Originalskalaen bør også bruges til sammenligning på tværs af døgn, fordi hvert døgn har historisk beregnet normalisering.

## Tidsforbrug og genoptagelse

Den fulde nye fitmetode koster den oprindelige estimation plus efteroptimeringen. I dette forsøg genbruges første del fra checkpoints, så den målte fortsættelsestid kun omfatter efteroptimering, nye prognoser, kontroller og eventuelt manglende referencefits.

`estimated_full_method_worker_ratio = 1 + sum(joint_seconds)/sum(old_worker_seconds)` er et omtrentligt mål for den ekstra beregningstid. Det er ikke en målt vægtidsratio: prognosens beregningstid kan ændres med de nye parametre, og worker-belastningen kan variere. `runtime.json` måler den faktiske fortsættelseskørsel.

Kør samme kommando igen efter en afbrydelse. Færdige jobs springes over. Kodeindhold, indstillinger, inputfiler og referencer har hashguards; ændringer kræver en ny outputmappe. Antallet af workers kan ændres ved genoptagelse. Opdatér ikke kildekoden eller referencefilerne under en aktiv kørsel.

## API

`lsq_params.fit_many(..., joint_max_nfev=80)` slutter det logistiske fit med fælles efteroptimering; standarden 0 bevarer den oprindelige opførsel. `lsq_params.refine_joint(datasets, fitted, ...)` kan bruges direkte på et eksisterende fit. `FamilyFit(model, family=...).fit_many(...)` og `.refine_joint(...)` giver samme mulighed for de øvrige familier. Den fælles numeriske del ligger i `src/params/bm_1_joint.py`.

En enkelt logistisk døgnprognose kan også køres med:

```powershell
python -B -m src.exe.run_model.bm_1.predict_daily --day 2022-05-05 --joint-max-nfev 80 --joint-max-seconds 20 --score --save
```
