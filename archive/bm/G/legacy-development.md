# G: januargrundlag og afløste fælles arbejdsordrer

De tre gamle fælles indgange gentog status, udførte opgaver og afløste næste-trin. Den sammenhængende faglige forklaring findes allerede i [G-matematikgennemgangen](../../../root/G_MATHEMATICS_RECOVERY_20261008.md), der indekserer de originale noter. Historiske punktforsøg må ikke ommærkes til den senere intervalmodel.

Status: historisk evidens. Aktuel brug og scope står i [BM-guiden](../../../root/BM_GUIDE.md).

Afsnittene nedenfor samler resultater og begrænsninger uden de gentagne arbejdsordrer. Oprindelige matematisk væsentlige afsnit er bevaret særskilt. Fuld originaltekst, kildehash og erstatningssti findes via [stikortet](../old-paths.json) og den verificerede kildepakke.

<a id="g"></a>
## G.md

Original: `root/G.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="g--aktuel-g-plan-og-næste-implementering-8-oktober-2026"></a>
<a id="g--aktiv-g-afslutning-otte-timers-grundversion-8-oktober-2026"></a>
<a id="g--samlet-matematik-og-argumentation-8-oktober-2026"></a>
<a id="g--g7-genoptaget-efter-slettet-chat-8-oktober-2026"></a>
<a id="g--g7-gemte-døgn-fixturekvitteringer-kontrolleret-8-oktober2026"></a>
<a id="g--g7-døgn-fixture-timing-og-budgetstop-8-oktober-2026"></a>
<a id="g--g7-samlet-døgnstyring-implementeret-ikke-kørt-8-oktober"></a>
<a id="g--g7-durable-feedback-og-v2-budgetgate-kontrolleret-8-oktober"></a>
<a id="g--g7-pr115116-merged-atomisk-audit-kontrolleret-7-oktober"></a>
<a id="g--g7-recovery-afsluttet-fuld-quadbank-åben-projektionsaudit-7-oktober"></a>
<a id="g--g7-pr114-merged-coarse-prissøgning-færdig-quad-partial-7-oktober"></a>
<a id="g--g7-pr114-merged-launcher-og-særskilt-præregistrering-færdige-7-oktober"></a>
<a id="g--aktiv-afgrænsning-bm-færdig-først-wm-sammenligning-bagefter"></a>
<a id="g--g7-merges-verified-bounded-pricequad-design-ready-7-oktober"></a>
<a id="g--g7-receipt-genbrug-og-writer-guards-færdige-7-oktober"></a>
<a id="g--g7-seed-rettelse-og-kort-emfeedback-kontrol-færdig-7-oktober"></a>
<a id="g--g7-neutral-auxiliary-kørsel-færdig-råreview-begrænser-accept-7-oktober"></a>
<a id="g--g7-auxiliary-budget-og-integrationsdesign-færdigt-7-oktober"></a>
<a id="g--g7-eget-m254-nodeforsøg-gennemført-7-oktober"></a>
<a id="g--g7-egne-nodecheckpoints-implementeret-7-oktober"></a>
<a id="g--g7-deterministisk-døgnkontrol-med-bevaret-forbrug-7-oktober"></a>
<a id="g--g7-fælles-bmwm-kontrakt-og-frozen-fit-prisrespons-7-oktober"></a>
<a id="g--aktiv-kontrolprioritet--johan-7-oktober-2026"></a>
<a id="g--aktiv-basisbeslutning--johan-6-oktober-2026"></a>
<a id="g--g-plan--deterministisk-bm-kerne-stokastisk-udvidelse-og-wm4-sammenligning"></a>
<a id="g--arbejdsregel-fra-johan-5-oktober-adskil-gf-og-minimér-kontrolarbejdet"></a>
<a id="g--grundarbejde-til-næste-implementation-5-oktober"></a>
<a id="g--afgrænsningsreview-5-oktober-2026"></a>
<a id="g--aktuel-g0g3-leverance-5-oktober"></a>
<a id="g--0-bachelorens-formål-og-krav"></a>
<a id="g--hvad-wm-faktisk-gør"></a>
<a id="g--1-leverancer-og-rækkefølge"></a>
<a id="g--1a-bindende-genbrug-fra-wm4"></a>
<a id="g--2-første-trin-samme-bm1-uden-processtøj"></a>
<a id="g--eksakt-og-billig-reference-for-lineære-kurver"></a>
<a id="g--3-parameterestimation"></a>
<a id="g--små-identifikationskontroller"></a>
<a id="g--4-data-initialisering-og-retfærdig-wm4-sammenligning"></a>
<a id="g--5-fleksible-fg-i-et-lille-ekstra-trin"></a>
<a id="g--første-g4-pilot-matematisk-definition-og-forsvar"></a>
<a id="g--afgrænsning-og-efterfølgende-kurvevalg"></a>
<a id="g--6-beregningsbudget-og-stopregler"></a>
<a id="g--7-kodens-placering-kontrol-og-output"></a>
<a id="g--8-obligatorisk-stokastik-kontrol-og-rekursiv-estimation"></a>
<a id="g--g6a-billig-variansreference-med-bms-egen-diffusion"></a>
<a id="g--g6b-fokkerplanck-beta-mixture-og-eulermaruyama"></a>
<a id="g--g8-rekursiv-estimation-og-handoff-til-kontrol"></a>
<a id="g--g7-obligatorisk-stokastisk-prisstyring-med-single-shooting"></a>
<a id="g--små-budgetter-og-afslutningskriterier"></a>
<a id="g--beslutning-5-oktober-genbrug-wms-ugeprognose-med-fælles-langt-fit"></a>
<a id="g--det-konkrete-wm-princip-og-genbrug"></a>
<a id="g--informationsgrænsen-i-ugeprognosen"></a>
<a id="g--hvornår-yderligere-implementation-bliver-nødvendig"></a>
<a id="g--9-kilder-og-første-konkrete-opgave"></a>
<a id="g--januar-håndtering-af-huller-5-oktober-2026"></a>

Den oprindelige G-plan skelnede deterministisk kerne/fit/forecast (G0–G4), støj og fordelinger (G6a/G6b), rekursiv parameter/stateestimation (G8) og prisstyring (G7). Den blev gentagne gange opdateret under udviklingen. BM-X bevarer egen betydning, prisnormalisering og timer/dage omregnes eksplicit, og WM er sammenligningsreference. G6a gav negativ båndkalibreringsevidens; fuldt stokastisk G7-døgn er fortsat ikke numerisk accepteret. Den senere afsluttede timeprognosemilepæl ændrer ikke disse historiske begrænsninger.

<a id="g-implementation-handoff"></a>
## G_IMPLEMENTATION_HANDOFF.md

Original: `root/G_IMPLEMENTATION_HANDOFF.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="g-implementation-handoff--aktuel-g-plan-og-næste-implementering-8-oktober-2026"></a>
<a id="g-implementation-handoff--fg-afgrænsning-præciseret-8-oktober-2026"></a>
<a id="g-implementation-handoff--aktiv-g-afslutning-otte-timers-grundversion-8-oktober-2026"></a>
<a id="g-implementation-handoff--genfundet-matematisk-grundlag-8-oktober-2026"></a>
<a id="g-implementation-handoff--g7-genoptaget-efter-slettet-chat-8-oktober-2026"></a>
<a id="g-implementation-handoff--g7-gemte-døgn-fixturekvitteringer-kontrolleret-8-oktober2026"></a>
<a id="g-implementation-handoff--g7-døgn-fixture-timing-og-budgetstop-8-oktober-2026"></a>
<a id="g-implementation-handoff--g7-samlet-døgnstyring-implementeret-ikke-kørt-8-oktober"></a>
<a id="g-implementation-handoff--g7-durable-feedback-og-v2-budgetgate-kontrolleret-8-oktober"></a>
<a id="g-implementation-handoff--g7-pr115116-merged-atomisk-audit-kontrolleret-7-oktober"></a>
<a id="g-implementation-handoff--g7-recovery-afsluttet-fuld-quadbank-åben-projektionsaudit-7-oktober"></a>
<a id="g-implementation-handoff--g7-pr114-merged-coarse-prissøgning-færdig-quad-partial-7-oktober"></a>
<a id="g-implementation-handoff--g7-pr114-merged-launcher-og-særskilt-præregistrering-færdige-7-oktober"></a>
<a id="g-implementation-handoff--aktiv-afgrænsning-bm-færdig-først-wm-sammenligning-bagefter"></a>
<a id="g-implementation-handoff--g7-merges-verified-bounded-pricequad-design-ready-7-oktober"></a>
<a id="g-implementation-handoff--g7-receipt-genbrug-og-writer-guards-færdige-7-oktober"></a>
<a id="g-implementation-handoff--g7-seed-rettelse-og-kort-emfeedback-kontrol-færdig-7-oktober"></a>
<a id="g-implementation-handoff--g7-neutral-auxiliary-kørsel-færdig-råreview-begrænser-accept-7-oktober"></a>
<a id="g-implementation-handoff--g7-auxiliary-budget-og-integrationsdesign-færdigt-7-oktober"></a>
<a id="g-implementation-handoff--g7-eget-m254-nodeforsøg-gennemført-7-oktober"></a>
<a id="g-implementation-handoff--g7-egne-nodecheckpoints-implementeret-7-oktober"></a>
<a id="g-implementation-handoff--g7-deterministisk-døgnkontrol-med-bevaret-forbrug-7-oktober"></a>
<a id="g-implementation-handoff--g7-fælles-bmwm-kontrakt-og-frozen-fit-prisrespons-7-oktober"></a>
<a id="g-implementation-handoff--aktiv-kontrolprioritet--johan-7-oktober-2026"></a>
<a id="g-implementation-handoff--aktiv-basisbeslutning--johan-6-oktober-2026"></a>
<a id="g-implementation-handoff--g3-og-frem--grundlag-og-overdragelse-til-sol"></a>
<a id="g-implementation-handoff--ny-beslutning-langt-fit-og-genbrug-af-wms-ugeprognose"></a>
<a id="g-implementation-handoff--opgaven-og-det-allerede-kendte"></a>
<a id="g-implementation-handoff--fund-der-skal-håndteres-før-historisk-g3-accept"></a>
<a id="g-implementation-handoff--p0--datakontrakt-fitarkiv-og-inputaudit"></a>
<a id="g-implementation-handoff--p1--gemt-fit--frosset-forecast-origin"></a>
<a id="g-implementation-handoff--p2--g3-prognose-varig-eksport-og-særskilt-scoring"></a>
<a id="g-implementation-handoff--p3--lille-g4-og-ét-fælles-modelinterface"></a>
<a id="g-implementation-handoff--p4--g6b-lille-fpmixtureem-panel"></a>
<a id="g-implementation-handoff--p5--g8-rekursiv-staterho-bank-og-ærligt-posterior-handoff"></a>
<a id="g-implementation-handoff--p6--g7-single-shooting-med-samme-stokastiske-objective"></a>
<a id="g-implementation-handoff--leverancer-testscope-og-næste-sessions-start"></a>
<a id="g-implementation-handoff--reserveret-til-senere-astra-review"></a>

Den gamle overdragelse beskriver allerede implementerede snapshot-, responsefit-, filter- og kontroldele. Gemte fits forudsiger uden ny optimering og fryser forecast før optional scoring. Modelinterne stater og egne parameterbankposteriorer bevares. Historiske næste-opgaver er afløst; originale implementationdetaljer og sourceversioner findes i den verificerede kildepakke, mens gennemgået matematik står i G_MATHEMATICS_RECOVERY og de bevarede emnereviews.

<a id="g-january"></a>
## G_JANUARY.md

Original: `root/G_JANUARY.md` ved `89e8337b6857`. Historiske instruktioner er afløst.

<a id="g-january--aktuel-g-plan-og-næste-implementering-8-oktober-2026"></a>
<a id="g-january--aktiv-g-afslutning-otte-timers-grundversion-8-oktober-2026"></a>
<a id="g-january--matematikken-samlet-efter-chatsletning-8-oktober-2026"></a>
<a id="g-january--g7-genoptaget-efter-slettet-chat-8-oktober-2026"></a>
<a id="g-january--g7-gemte-døgn-fixturekvitteringer-kontrolleret-8-oktober2026"></a>
<a id="g-january--g7-døgn-fixture-timing-og-budgetstop-8-oktober-2026"></a>
<a id="g-january--g7-samlet-døgnstyring-implementeret-ikke-kørt-8-oktober"></a>
<a id="g-january--g7-durable-feedback-og-v2-budgetgate-kontrolleret-8-oktober"></a>
<a id="g-january--g7-pr115116-merged-atomisk-audit-kontrolleret-7-oktober"></a>
<a id="g-january--g7-recovery-afsluttet-fuld-quadbank-åben-projektionsaudit-7-oktober"></a>
<a id="g-january--aktiv-afgrænsning-bm-færdig-først-wm-sammenligning-bagefter"></a>
<a id="g-january--historisk-g7-neutral-auxiliary-før-seed-rettelsen-7-oktober"></a>
<a id="g-january--g7-auxiliary-budget-og-integrationsdesign-færdigt-7-oktober"></a>
<a id="g-january--g7-eget-m254-nodeforsøg-gennemført-7-oktober"></a>
<a id="g-january--g7-egne-nodecheckpoints-implementeret-7-oktober"></a>
<a id="g-january--g7-deterministisk-døgnkontrol-med-bevaret-forbrug-7-oktober"></a>
<a id="g-january--g7-fælles-bmwm-kontrakt-og-frozen-fit-prisrespons-7-oktober"></a>
<a id="g-january--aktiv-kontrolprioritet--johan-7-oktober-2026"></a>
<a id="g-january--g0g4-januar-fit-frosset-ugeprognose-og-lille-kurvepilot"></a>
<a id="g-january--g4-fra-original-train-eksport--gennemført-pilot-og-bevarede-kommandoer"></a>
<a id="g-january--g3-kommandoen-bevaret-til-frosset-replay"></a>
<a id="g-january--vis-fittet-for-januar"></a>
<a id="g-january--kørsel-på-eksisterende-checkout"></a>
<a id="g-january--fit-og-beregningslofter"></a>
<a id="g-january--data-og-evidensgrænse"></a>
<a id="g-january--output-og-kontroller"></a>
<a id="g-january--minimumskontrol-for-g0g3"></a>

Januar2022-fit og efterfølgende168 h blev gennemført som eksponeret udviklingsforsøg. På den gemte uge var MSE BM=.06951591504, baseline=.06934621735 og WM=.07099789830; ingen baselinegevinst eller generel modelvinder. G-januar bruger den historiske tilgængelige-målere-aggregering og punktoperator; det er ikke F4's faste population eller senere G-timeoperator. Manglende rå demand beholdes som maskeret i metrics, og interpolation begrænses til den tilladte træningsperiode. Fit/forecast/input- og normaliseringskontrakterne er bevaret i G_MATHEMATICS_RECOVERY og G4-reviewene.
