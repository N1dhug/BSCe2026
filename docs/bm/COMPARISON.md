# Aktuel modelafgrænsning

| Område | På main efter beskæring |
|---|---|
| G | Deterministisk respons, baseline, energibalance, timegennemsnit, C/k/Xi/a-fit og prognose |
| G-udvidelser | Arkiveret: bank, Bayes-filter, Bernstein/Beta, støj, residualer, bånd, kontrol- og G/WM-studier |
| WM og WM-MPC | Bevarede med nødvendige fælles afhængigheder |
| F | Tidligere arkiveret på archive/f-main-20261009; genindføres ikke af G-beskæringen |

G-kernen er uændret. De fælles moduler kan fortsat indeholde stokastisk numerik,
som WM eller historiske referencekontroller bruger. Det er ikke en del af G's aktive indgang.
Der udføres ingen ny fælles modelrangering eller historisk fitkampagne.

Læs [grundmodel](G_INTERVAL_MODEL.md), [brug](../../root/BM_GUIDE.md) og
[arkivoversigt](../../archive/bm/INDEX.md).
