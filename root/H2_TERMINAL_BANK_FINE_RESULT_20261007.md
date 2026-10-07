# H2 terminal bank: own fine control passed, 7 October 2026

Johan approved the proposed additional 180 native-development seconds at
23:47:41 Europe/Copenhagen ("de er godkendt"). The native ceiling is now
1080 seconds; the campaign ceiling remains 64800 seconds and the numerical
stop remains 7 October 23:55 Copenhagen. The unapproved proposal remains
historical; the separate approval references its exact SHA256.

The separate fine control completed in 41.78445 seconds. Candidate 28 used
2561 cells and step 1/65536 with its own Beta prior and 24 prediction-only
hours, followed by observed hour 24. No base prior interpolation was used.
All nine full contours, all 24 prior checkpoints and the posterior are saved.

| Check | Result | Existing limit |
|---|---:|---:|
| Base/fine one-hour log-score difference | 0.0000106323474 | 0.05 |
| Posterior mean difference | 0.00000011589485 | 0.002 |
| Conservative posterior L1 difference | 0.000834816732 | 0.05 |
| Maximum relative conditional guard | 2.38537e-11 | 1e-8 |
| Maximum contour overlap log difference | 1.94623e-8 | 1e-7 |
| Maximum neutral mass error | 3.01981e-14 | 1e-10 |

Fine log evidence: 0.8592776768548576; base: 0.859288309202249.
The original five-center prefix leaves 32 fine cells unresolved. Center
0.875 resolves these, matching the base diagnostic's mechanism. All 2561
active cells pass the original guards. All 69 frozen numerical sources
remain unchanged. The optional extended-bank adapter is still separate.

Read-only review verified every raw SHA256, the exact own-prior chain,
recomputed each conditional candidate from saved characteristic arrays,
and reconstructed the score and full posterior exactly. It used zero native
propagations. Its first attempt had an additional relative-per-cell mass
assertion that was unsuitable for negligible tails: absolute differences
were at most 5.43e-16 and total difference below 9.41e-14. That failure and
its charge are preserved. The corrected review uses the existing total-mass
scale 1e-10; no model or numerical guard was changed.

Campaign after computation, review and ZIP export: 40472.90629690291/64800 s.
Native development: 942.3103597460481/1080 s. This approval used 44.22399 s
of its 180 s scope. All old reservations and charges remain in the ledger.

This validates one failed hour, not optimizer convergence or the selected
family's full TRAIN/selection validation. Original 3655 jobs, 27 completed
four-piece calls and terminal failure markers remain preserved. No fit
restart, model selection, final-Y reading or forecast occurred.

The next concrete step is a separate continuation adapter that verifies
old source identities, candidate tokens and exact priors before adopting
old successful checkpoints. It must preserve their original identities,
record the new bank and source identity for new jobs, and establish a new
run rather than reopen the terminal run. A budget/time admission must cover
remaining fit and full validation before any numerical continuation. The
current same-day deadline must not silently roll over. PR106 stays draft.
