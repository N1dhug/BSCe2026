# F4 nattearbejde, 7. oktober 2026

Aktiv branch: `feature/f4-h2-night-runner-20261007`.
Udgangspunkt: lokal `b1a702a`, GitHub main `4f9ea30abee6a2673a33868ffad30e9ff82cc429`.
Kun eksplicitte F-filer publiceres oven på seneste main; samtidig G/WM bevares.

## Genoptagelse

1. Læs denne fil, `H2_COMPACT_RATE_REVIEW.md` og den aktuelle Git-status.
2. Genbrug den råreviewede PC-ZIP og dens 72 frosne observationer. Ingen ny
   workbook-preparation, ingen gentagelse af 27-timers pilot.
3. Færdiggør en afgrænset runner til fælles kurvefit, egne base/fine streams,
   selection, frosset open-week og separat final prequential evaluering.
4. Kontroller checkpoint/resume, budget, datagrænser og negative gates med
   små kontroller. Bevar fejl og charges; ingen automatisk ny modelprøve.
5. Publicér checkpoints, review egen ændring, merge når checks består.

## Bindende udgangspunkt

H: 1671.0408590929223 s; native: 646.4097384530473 / 900 s.
Native-rest: 253.59026154695266 s. Validation-reserve: 11840.471907224273 s.
Samlet H-loft: 36000 s. Samme 1/2/4-piece familier, 49 unikke TRAIN-par,
48/24/168 timers split, separate MED-huse, uændrede numeriske gates.

Den lineære pilot er kun 24 prior + 3 observerede timer. Et fuldt base/fine
TRAIN+selection-vindue kan ikke dokumenteres ved at gange dens fejl med 24;
filtreringen er observationafhængig og nonlinear. En sådan kontrol kan ikke
finansieres af 253.59 s ved de målte rates. Koden skal derfor skelne mellem
implementeret, kørselsadmitted og faktisk numerisk/videnskabeligt accepteret.
Ingen flytning af udviklingskontroller til en anden budgetetiket.

## Status

- [x] Aktuel kilde, råpilot, budget og seneste GitHub-main læst.
- [ ] Konkret admission-/valideringsbeslutning dokumenteret.
- [ ] Genbrug af frossen receipt og portable inputs.
- [ ] Fit, egne valideringsstreams og atomiske checkpoints/resume.
- [ ] Prognose og betinget I/J-handoff.
- [ ] Afgrænsede kontroller og status/budget opdateret.
- [ ] Publiceret og reviewet.

Historisk fit, runtime-admission og videnskabelig accept er fortsat false.
