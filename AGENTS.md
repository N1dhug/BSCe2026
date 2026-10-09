# Arbejdsregler for BSCe2026

Læs [aktuel opgave](docs/bm/STARTPROMPT.txt) og [status](docs/bm/STATUS.txt).
Kontrollér faktisk branch, diff og aktive processer før genoptagelse.
Ældre arbejdsordrer og budgetter er historik, ikke nye kørselsordrer.

- Bevar lokale ændringer og unikke/uklare resultater. Arbejd på aftalt branch;
  ingen automatisk merge eller genstart efter afbrudt chat.
- WM/WM4/WM-MPC, fælles afhængigheder og deres data/dokumentation/resultater
  er beskyttede. Kontrollér [manifestet](docs/bm/wm-protection.json) efter ændringer.
  Synkroniserede `sources/` og vejledermateriale er skrivebeskyttede.
- Skift ikke population, observationstype, grids, tolerancer eller seeds som
  skjult oprydning. Bind fits til gruppe/liste/skala/baseline/cutoff og deklarér
  inputs/usikkerhed. Numerisk kontrol er ikke statistisk eller fysisk accept.
- Genbrug gyldig evidens. Kør kun berørte tests og korte kontroller inden for
  aftalt samlet budget; ingen automatisk historisk kampagne eller dyrt fit.
- Bevar source/input/runtime-identiteter og checkpoints. Brug originalkilder
  til gamle runs; svæk ikke hashguards. Skriv atomisk, med entydig status.
- Publicér ikke lokale rådata eller private resultater. Opdater én kanonisk
  status efter delmål, og lav små sammenhængende commits med relevante kontroller.

Brug: [BM-guide](docs/bm/README.md). Faglig kontrakt: [matematik](docs/bm/MATHEMATICS.md).
Historik/restore: [arkiv](archive/bm/INDEX.md). WM læses som reference og ændres ikke.
