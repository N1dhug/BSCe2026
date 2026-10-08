# Historisk BM-evidens

Dette arkiv samler afløste arbejdsordrer og resultathistorik efter emne. Gamle instruktioner er ikke aktive kørselsordrer. Aktuelle indgange: [BM-guide](../../root/BM_GUIDE.md), [F4-status](../../root/F4.md).

Originale noter ligger uændret i Git ved `89e8337b6857b148c48e6880473984c3ae4574a0` og i den verificerede lokale sourcepakke `cleanup_work/recovery/start-source-89e8337.zip`. Rådata ligger i den separat verificerede lokale backup; ingen rådata publiceres her. [Stikortet](old-paths.json) angiver hver originalsti, hash, erstatning og anker.

Kildebundne scripts, registreringer og manifests i `root/f4` og `root/g7` bliver på deres originale stier. De er logisk en del af historikken, fordi deres path/hashkontrakter er nødvendige for genskabelse. Historiske JSON-referencer gælder den oprindelige kildepakke og ændres ikke til resuméernes stier.

F- og G-matematik, herunder negative numeriske resultater, er bevaret i de eksisterende fagnoter. Denne runde samler 28 status-/overdragelsesnoter i 7 historikfiler og 8 G-guider/protokoller i 3 fagnoter; den flytter ikke numeriske kerner eller tests.

- [F: tidlige H2-afbrydelser og recovery](F/h2-early-recovery.md)
- [F: terminale H2-stop og kildegendannelse](F/h2-terminal-recovery.md)
- [F: H2-runtime, budget og scopebeslutninger](F/h2-runtime-decisions.md)
- [F: tofamilievalg, slutdiagnose og handoff](F/h2-selection-and-handoff.md)
- [G: kontrolcheckpoints og genfundet evidens](G/control-checkpoints.md)
- [G: S3-milepæl og afløste udvidelsesplaner](G/s3-transition-history.md)
- [G: januargrundlag og afløste fælles arbejdsordrer](G/legacy-development.md)

## Samlede G-fagnoter og forsøgsprotokoller

- [G: timeoperator, baseline og integrerede procesmomenter](../../docs/bm/G_INTERVAL_MODEL.md)
- [G: procesusikkerhed og empirisk outputresidual](../../docs/bm/G_UNCERTAINTY.md)
- [G: ugedagsfit og det historiske seksugers båndstudie](../../docs/bm/G_WEEKDAY_STUDY.md)

## Kontrolleret bevaringsprøve

Recovery-rodens faktiske placering står i [status](../../docs/bm/STATUS.txt).
Stier med `cleanup_work/` nedenfor er relative til ChatGPT-projektet, ikke dette repository.

- F: `f4_delivery_final` forbinder manifest, gemt model, inputidentiteter og resultat.
  Den særskilte `cleanup_work/recovery/f4_delivery_source` bevarer præcis alle
  215 manifestbundne sourcefiler, model, konfiguration og begge inputfiler.
  `HISTORICAL_EXPORT.txt` forklarer brugen; eksporten er historisk og må ikke
  udvikles videre. Begge origin-states er genindlæst uden modelberegning.
  Audit: `cleanup_work/f4_historical_export_audit.json`. F4_code er urørt.
- G: `G-weekday-bands-20261008` har 316 verificerede manifestposter; uge,
  matematik/protokol, input-/sourceidentiteter og resultater kan følges via
  [G-ugedagsstudiet](../../docs/bm/G_WEEKDAY_STUDY.md) og den originale pakke.
  Audit: `cleanup_work/artifact_integrity.json`, afsnittet `G_package`.
- De 36 fjernede noters originalbytes/hash/anker findes i [stikortet](old-paths.json)
  og start-source-pakken. Originale AGENTS/F4/BM_GUIDE findes ved samme Git-base.
  Ingen test, numerisk kerne, checkpoint eller rå tidsserie er slettet.
