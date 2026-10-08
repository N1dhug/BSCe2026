# F: H2-runtime, budget og scopebeslutninger

De forskellige budgetter og deadlines nedenfor er historiske beslutninger. Oprydningen nulstiller dem ikke og autoriserer ingen nye H2-fitjobs.

Status: historisk evidens. Aktuel brug og scope står i [BM-guiden](../../../root/BM_GUIDE.md).

Afsnittene nedenfor samler resultater og begrænsninger uden de gentagne arbejdsordrer. Oprindelige matematisk væsentlige afsnit er bevaret særskilt. Fuld originaltekst, kildehash og erstatningssti findes via [stikortet](../old-paths.json) og den verificerede kildepakke.

<a id="f4-same-day-remaining-plan-20261007"></a>
## F4_SAME_DAY_REMAINING_PLAN_20261007.md

Original: `root/F4_SAME_DAY_REMAINING_PLAN_20261007.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-same-day-remaining-plan-20261007--f4-restarbejde-og-mulighed-for-afslutning-7-oktober-2026"></a>
<a id="f4-same-day-remaining-plan-20261007--hvad-der-faktisk-er-gjort"></a>
<a id="f4-same-day-remaining-plan-20261007--tidsgrundlag"></a>
<a id="f4-same-day-remaining-plan-20261007--korteste-forsvarlige-udførelse"></a>
<a id="f4-same-day-remaining-plan-20261007--hvorfor-der-ikke-er-omskrevet-en-solver"></a>
<a id="f4-same-day-remaining-plan-20261007--konkret-budgetvalg-endnu-ikke-godkendt"></a>
<a id="f4-same-day-remaining-plan-20261007--afslutningens-faglige-scope"></a>

Fra 1076 verificerede jobs blev resterende fuld kandidatramme illustreret som 4.85 h ved middelrater, 7.19 h med faktor1.5 og 14.01 h ved maksimumrater. Scenarier er ikke målt kode-speedup eller konvergensløfter. Metoden prioriterede eksakt cached trace, kontrolleret miljø og hele den beskyttede restvalidering.

<a id="f4-h2-terminal-and-next-decision-20261007"></a>
## F4_H2_TERMINAL_AND_NEXT_DECISION_20261007.md

Original: `root/F4_H2_TERMINAL_AND_NEXT_DECISION_20261007.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="f4-h2-terminal-and-next-decision-20261007--f4-h2-terminalstatus-og-konkret-fortsættelsesforslag-7-oktober-2026"></a>
<a id="f4-h2-terminal-and-next-decision-20261007--faktisk-resultat"></a>
<a id="f4-h2-terminal-and-next-decision-20261007--budget-og-bevaret-historik"></a>
<a id="f4-h2-terminal-and-next-decision-20261007--hvad-der-mangler"></a>
<a id="f4-h2-terminal-and-next-decision-20261007--konkret-metodeændring-til-godkendelse--ikke-udført"></a>

17 komplette tostykkekald og 1402 råjobs var sikret, men maxfev17 gav status5/converged=false. Den gamle run var terminal. Forslaget om 25/49 maxfev var på dette tidspunkt en særskilt metodeændring, ikke en allerede udført fortsættelse; senere godkendelse ændrer ikke den oprindelige negative evidens.

<a id="h2-budget-boundary-20261008"></a>
## H2_BUDGET_BOUNDARY_20261008.md

Original: `root/H2_BUDGET_BOUNDARY_20261008.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="h2-budget-boundary-20261008--h2-budgetstop-efter43-kald-8-oktober"></a>
<a id="h2-budget-boundary-20261008--tidsbehov-og-konkret-forslag"></a>
<a id="h2-budget-boundary-20261008--forberedt-implementation-og-kontrol"></a>

43 firestykkekald og 1166 nye råjobs var sikret; kandidat44 var delvis. Blok6 stoppede rent for at beskytte restvalideringen. Forslaget om H18→20h og native1080 s skulle autoriseres separat. Estimeret tid var ikke garanti for kandidatloft eller konvergens.

<a id="h2-cap-extension-20261007"></a>
## H2_CAP_EXTENSION_20261007.md

Original: `root/H2_CAP_EXTENSION_20261007.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="h2-cap-extension-20261007--f4-h2-eksplicit-fortsættelse-efter-kandidatloftet-7-oktober-2026"></a>
<a id="h2-cap-extension-20261007--beslutning-og-begrundelse"></a>
<a id="h2-cap-extension-20261007--bevaring-genbrug-og-ny-identitet"></a>
<a id="h2-cap-extension-20261007--rækkefølge-og-accept"></a>

Den særskilte 25/49-caps-fortsættelse bevarede cached COBYQA-trace, kilder og gamle reservationscharges. Cap/failure var terminal; invocation- og budgetstop var pause. Større kandidatloft ændrede hverken numeriske tolerancer eller kravet om fuld egen base/fine-validering og frosset slutprognose.

<a id="h2-compact-staged-run"></a>
## H2_COMPACT_STAGED_RUN.md

Original: `root/H2_COMPACT_STAGED_RUN.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="h2-compact-staged-run--f4-h2-etapeopdelt-natplan-og-kørselskontrakt"></a>
<a id="h2-compact-staged-run--hvorfor-rækkefølgen-ændres"></a>
<a id="h2-compact-staged-run--fast-rækkefølge"></a>
<a id="h2-compact-staged-run--hvad-prøverne-kan-og-ikke-kan-dokumentere"></a>
<a id="h2-compact-staged-run--crash-og-aflevering"></a>

Etapeplanen krævede først fuld lineær base/fine-reference, dernæst kontrolleret nonlinear optimering, egne valideringer/modelvalg, frosset 168 h prognose og til sidst separat slutdiagnose. En kort prefixrate dokumenterer ikke nonlinear worst-case eller fuld vinduespræcision. Atomiske checkpoints og bevarede charges gælder ved hvert stop.

<a id="h2-same-day-execution-20261007"></a>
## H2_SAME_DAY_EXECUTION_20261007.md

Original: `root/H2_SAME_DAY_EXECUTION_20261007.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="h2-same-day-execution-20261007--f4-h2-godkendt-dagsfortsættelse-7-oktober-2026"></a>
<a id="h2-same-day-execution-20261007--budget-og-uændret-model"></a>
<a id="h2-same-day-execution-20261007--admission-og-rækkefølge"></a>
<a id="h2-same-day-execution-20261007--reproduktion"></a>

Dagscontrolleren brugte 600 s invocations under eksplicit 18h-loft og stopfrist, med lukket ledger og deltaarkiv efter hver blok. Gamle 69 numeriske kilder, cached optimizerkald og crashreservation blev bevaret. Slut-Y skulle forblive afskærmet indtil faktisk forecastfreeze; senere budgetbeslutninger er separate.
