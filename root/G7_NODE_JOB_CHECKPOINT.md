# G7 nodecheckpoint — registrering 7. oktober 2026

Scope: implementér durable node-jobs og fuld admission-jobgraf. G alene.
Base: 9c2943cc12e895cd275669cee577ada2fce1e475. Tidligere G-debit1098.580078s, rest701.419922s.
Højst20s konservativ charge for denne implementerings-/stubkontrolleverance.
Ingen faktisk M254/FV/EM-kørsel, historisk fit eller gamle checks genkøres.
Bevar vejlederens M254, gamle evaluators, kerner og tidligere negative pilot.

Plan:
- Del eksisterende G-evaluation i samme nodeforward og samme bankreduktion.
- Nodecheckpoint indeholder fuld bane/momenter/diagnostics og identitet for
  origin, model/numerik/inputs/price/source. Eksakt resume, ingen ny state.
- Atomisk store + write-ahead budgetreserve. Afbrudt node holder sin fulde
  charge; ingen automatisk retry. Completed nodegenbrug har ingen solvecharge.
- Genoptag reducering først efter alle fem ejede nodejobs.
- Komplet eksplicit jobgraf incl.search, M254257/M254513/FV513, quadratur,
  coupled EM, feedback, setup/audit/export. Ukendte rates blokerer admission.
- Én fokuseret billig test med stubtransport; ingen faktisk pilot.
- Dokumentér/ review/merge efter eksisterende authorization. Gem checkpoints.

Status: registreret før implementering/test. Numerisk G7-accept fortsat false.
