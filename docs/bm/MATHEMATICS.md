# G: grundlæggende matematik

Den aktuelle G-model er beskrevet med almindeligt læsbare ligninger i
[G_INTERVAL_MODEL.md](G_INTERVAL_MODEL.md) og [G_GRUNDMODEL.txt](G_GRUNDMODEL.txt).

Modellen bevarer lineær tilstandsrespons, to rette prisresponsstykker, en
tanh-respons, forbrug omkring baseline og energibalance. Tid måles i timer.
C, k, Xi og a tilpasses ved mindste kvadrater på timegennemsnit.
Baseline, skalaer og starttilstande fryses før prognosen.

Den deterministiske respons, fitkerne, RK4-integration og timeoperator er
uændrede af beskæringen. Ingen støj-, filter-, bank- eller båndmodel indgår.
Parameteridentifikation, baselinebegrænsninger og ekstrapolation er fortsat
modelspørgsmål; beskæringen er ikke en ny statistisk validering.

WM og fælles afhængigheder bevares. BM's lagertilstand og WM's tilstande har
deres egne betydninger. De oprindelige F-referencer findes på F-arkivbranchen.

Brug: [BM-guide](../../root/BM_GUIDE.md).
Arkiver: [oversigt](../../archive/bm/INDEX.md).
