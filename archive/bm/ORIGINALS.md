# Præcist genskabelige BM-originaler

<a id="original-tree"></a>
Kildepakken [source-7249bf9.zip](originals/source-7249bf9.zip) bevarer 766
versionsstyrede startfiler fra `7249bf9c318b20f45aa0c3a64b3c6c3b75afaa82`:
hele `src`, `tests`, `root`, dokumentation og konfiguration. Den indeholder
7.732.773 bytes udfoldet og 2.560.128 bytes pakket. **Dette er arkivering,
ikke matematisk eller kodemæssig deduplikering.**

Bytes er rekonstrueret fra de verificerede Windows-startbackups, ikke hentet
fra en senere ændret arbejdskopi. Hver fil blev kontrolleret mod råstartens
SHA256 og Git-indholdet med kun LF/CRLF-normalisering i den sidste kontrol.
Et separat nyt restoretræ blev derefter kontrolleret fil for fil.
[Manifestet](originals/source-7249bf9.json) registrerer alle præcise råhashes.

Historiske controllers, ledgers og JSON-referencer er uændrede inde i pakken.
Når et script bruger `parents[2]`, en oprindelig `root/f4`-/`root/g7`-sti eller
alle `src`-hashes, genskabes hele den indbyrdes struktur. Ingen hashguard
omgås, og en ny sourceversion påstås ikke at være et gammelt checkpoint.

## Verificér eller genskab

Kør fra repositoryets rod med standardbiblioteket i Python:

```powershell
python -I -S archive/bm/restore_original.py --destination ../bm-original-7249bf9 --verify-only
python -I -S archive/bm/restore_original.py --destination ../bm-original-7249bf9
```

Destinationen skal være ny og må ikke være det aktive checkout. Programmet
validerer hele ZIP/manifestet og alle sikre relative stier før skrivning,
og genlæser derefter hver fil. Det starter ingen model, installerer intet,
ændrer intet eksisterende træ og tilføjer ingen nye beregningsbudgetter.

`files/` er udeladt: private inputs og beholdte resultater publiceres ikke i
kildepakken. En faktisk gammel resume kræver også originalt input, config,
environment og eksisterende resultatmappe med deres egne receipts. De lokale
fulde backups og Git-bundles er registreret i [status](../../docs/bm/STATUS.txt).
Der loves ikke, at enhver historisk ekstern råpakke er genfundet, eller at
7249-source passer et checkpoint, som oprindeligt bandt et tidligere commit.
Sådanne checkpoints bruger deres egne bevarede sourcepakker.

## Hvad blev flyttet, og hvad blev samlet?

[Root-filplanen](root-file-plan.json) dækker alle 416 versionsstyrede root-filer,
deres disposition, begrundelse og observerede indgående referencer.
[Stikortet](old-paths.json) finder de faglige erstatninger for noter og de
oprindelige ZIP-medlemmer for historiske drivers og registreringer.
De seks lokale TeX-buildfiler er beholdt; de tælles ikke som slettede kilder.

Den læsbare viden findes i [emneindekset](INDEX.md). Aktuel brug sker gennem
[BM-guiden](../../docs/bm/README.md); gamle reproduktionskommandoer køres kun i
et passende genskabt originaltræ, aldrig som skjult genstart af en kampagne.

<a id="round3-retirements"></a>
## Tolv yderligere historiske drivere (09c602f)

Tredje gennemgang pensionerede otte afsluttede G7/G8-CLI'er og fire faste
F-search/joint-CN-kampagner; [indekset](INDEX.md) angiver navnene. Genbrugelige
metodekerner og deres faglige tests er stadig aktive. Dette er færre aktive
drivere, ikke sletning af den oprindelige kode eller en ny modelaccept.

Ti filer er byteidentiske med deres medlemmer i `source-7249bf9.zip`.
De to Bernstein-drivere havde ændrede imports ved 09c602f; deres præcise
seneste bytes findes i [delta-ZIP'en](originals/g-driver-delta-09c602f.zip)
med [hashmanifest](originals/g-driver-delta-09c602f.json).
[Stikortet](old-paths.json) angiver archive/member/SHA256 for alle tolv.
Deltaen indeholder kun de to filer og er **ikke et selvstændigt køreklar træ**.
En genskabelse af 09c602f kræver også de matchende afhængigheder fra det
bevarede Git-commit/startsource og de oprindelige private inputs/resultater.
Man må ikke lægge deltaen oven på 7249 og antage hashkompatibel resume.
Den komplette 7249-pakke og dens restorevej ovenfor er uændret.
