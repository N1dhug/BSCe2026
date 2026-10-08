# Arbejdsregler for BSCe2026

Den aktuelle BM-F/G-opgave står i [startprompten](docs/bm/STARTPROMPT.txt).
Ved genoptagelse: læs [status](docs/bm/STATUS.txt), kontroller faktisk branch,
Git-diff og aktive processer, og fortsæt fra seneste verificerede delmål.
Ældre daterede arbejdsordrer er historik; [arkivindekset](archive/bm/INDEX.md)
bevarer kilde-, resultat- og beslutningskæder.

- Arbejd på én branch i det eksisterende fulde checkout. Nye worktrees kræver
  reel nødvendighed. Bevar lokale/ignorerede resultater; skift/pull aldrig under
  aktive låste beregninger. Merge ikke historiske previewbranches blindt.
- WM, WM-MPC, deres dokumentation, data/resultater og fælles afhængigheder
  ændres ikke ved BM-oprydning. Kontroller beskyttelsesmanifestet efter relevante
  delmål. BM-navngivne filer kan også være WM-afhængigheder.
- Synkroniserede `sources/`, vejledermateriale og tavlegrundlag er skrivebeskyttede.
  Skeln mellem oprindelig matematik, afledte resultater og egne numeriske valg.
- Genbrug G's færdige timeoperator, residuallag, kalender og gemte resultater.
  F og G har selvstændige formål, budgetter og gates. Et fast F4-fit er ikke et
  femugers ugedagsfit. BM's X må ikke ommærkes til WM's F/R.
- Bevar grids, seeds, observationstype, tolerancer, negative resultater og alle
  historiske charges/failures. Numerisk kontrol er ikke statistisk accept.
  H/I/J og historisk fælles f/g-estimation har fortsat dokumenterede restpunkter;
  oprydning genstarter ikke kampagnerne.
- Ingen ny historisk fit, bred benchmarkkampagne, parameterbank eller gentagelse
  alene efter konteksttab. Forelæg konkret behov, genbrugsmulighed og tidsbudget
  før lange kørsler. En pause ved tidsloftet er ikke en færdig prognose.
- Bevar checkpoints og eksakte kilde-/inputidentiteter. Svæk aldrig hashguards.
  Gamle F4-runs binder hele `src`; brug original kildepakke til gammel resume.
  Brug atomiske kvitteringer, nye outputmapper og én ejer per kørsel.
- Kør mindste relevante kontrol. Rene dokumentændringer kræver linkkontrol,
  ikke numeriske studier. Fuld discovery og TeX-build er begrundede tilvalg;
  behold den eksisterende lette CI-regel. Slet ikke tests for pænere filtal.
- Brug eksisterende fremdriftsfunktioner; vis nyt/gemt arbejde, tid og
  færdig/pauset/fejlet. Opdater én status før lange operationer og efter delmål;
  lav små sammenhængende commits efter beståede relevante kontroller.
- Demand-enhed/hour-start er arbejdsantagelser indtil metadata verificeres.
  Deklarér forecast-inputs, maskering, cutoffs og usikkerhed; fremtidig demand
  og oracle-diagnostik må ikke indgå som ordinære prognoser.
- Lokale rå tidsserier publiceres ikke af oprydningen. Publicering/merge følger
  brugerens konkrete autorisation; gamle tilladelser er ikke nye fitbudgetter.

Brug: [BM-guide](root/BM_GUIDE.md). Matematik: [F/G](docs/bm/MATHEMATICS.md)
og uændret [BM-reference](root/BM_REFERENCE.md). Evidens:
[sammenligning](docs/bm/COMPARISON.md). Historiske kilder bevares i Git og den
verificerede lokale recovery-pakke; et arkiv er ikke i sig selv simplere kode.
