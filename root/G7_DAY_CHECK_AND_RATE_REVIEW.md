# G7 day graph fixture and standalone native timing

## Result and limits

The native day driver completed its entire fixture graph in the third separately registered scope: deterministic and M254 searches, frozen neutral/deterministic/stochastic policies, own coarse/fine M254/FV forwards and objective quadrature, coupled EM comparison, one endpoint observation, four own posterior updates and continuation. The saved day_result.json has all fixture numerical gates true. These values come from scientific-kernel stubs. They establish orchestration coverage, not numerical scientific acceptance.

The encompassing check is not passed. Its auxiliary reuse assertion reconstructed an evaluator with default numerical settings instead of the exact saved contract, and was correctly rejected. check_day_control.py now uses day.make_control for that reconstruction; the correction has only been statically parsed. The subsequent saved-receipt completion review timed out before a pass record. Do not report this package as a completed full-graph check or launch the scientific experiment.

## Immutable registrations and charges

Each scope was committed and all registration/source files read back exactly before worker invocation. There was no automatic numerical retry. Failed stores were not reused as new scientific stores.

| Scope | Preregistration commit | Parent outcome | Charge (s) |
|---|---|---|---:|
| Original full-graph stub | 1437257d31fa1ba76821801ac6234b53ceac1c79 | Node reservation exceeded | 15 |
| Smaller fixture/longer node reservation | cc06198f8104efa52d055df883d376a5b8cdcc6b | Parent hard stop at6s | 8 |
| Same corrected fixture, larger parent cap | 6d71ee0119bff530270c675edfc0b8596b23876b | Graph completed; auxiliary reuse contract rejected | 20 |
| Read-only saved completion review | 355d2780b9a5235f4d34f7bfc5371c343f8f0d71 | Parent hard stop at3s | 5 |
| One native deterministic24h timing | 4a9fc67eb5a0a1771afb4234c2feba172e29a3c2 | Completed, raw receipt reviewed | 8 |

An initial parent metadata preflight used the wrong registration key and failed before launching the corrected stub worker. Its empty invocation/preflight failure is preserved. It did not launch a second worker. Scope caps are charged in full; shorter successful or failed wall times do not recover budget.

Preserved prior1851.3372718589885 +56 =1907.3372718589885s. Ceiling1910s, remaining2.6627281410114847s. Source reads, documentation and artifact export are not scientific kernel invocations. Fixture ledger seed0/cap300 is a test ledger, not a replacement G campaign balance.

## One genuine rate

The registered native day_control.deterministic_job ran once for24h from its hash-verified archived own coarse M254 posterior at t=12. It chose the MAP hypothesis j=3 (zero-based), with conditional mean initial state0.5062519458461637. Prices(.5,.5,.5,.5), production257 cells, ODE/objective step1/96, native ODE and cost kernels unchanged. Complete call including job setup, solve and receipt write took2.3313609979995817s. J=0.12120382930681177. Parent wall3.932990571999653s; campaign charge8s.

The standalone identity deliberately differs from the integrated four-origin context; whole_graph_compatible=false. The genuine timing is useful evidence for measurement planning but is not inserted into the full admission packet. All16 compatible rates remain unknown. No linear extrapolation from earlier2h forward rates or stub times is permitted.

For perspective only,48 calls at this single observed rate with factor1.25 would allocate139.8816598799749s for deterministic search alone. This is not a worst-case bound or a complete graph estimate; candidate/runtime variation and all other work remain unmeasured. It cannot fit the current2.66s remainder.

## Mathematics and why the checks matter

For fixed hypothesis weights over a planned horizon, total expected tracking cost is

\[
J(u)=\int_{12}^{36}\!\left[(\mu(t)-A(t))^2+\sum_jw_jv_j(t)+\sum_jw_j(\mu_j(t)-\mu(t))^2\right]dt,
\quad\mu(t)=\sum_jw_j\mu_j(t).
\]

This is the exact law-of-total-variance expansion of E[(D-A)^2], not an added penalty. The deterministic benchmark instead solves the native ODE for the MAP hypothesis from its own conditional mean state. Its J value therefore has a different objective interpretation from the stochastic expected cost; measuring one neutral candidate does not establish optimizer superiority.

At an observed demand y, the native point likelihood yields raw evidence and Bayes weights

\[
Z_j=\int L_j(y\mid x)p_j^-(x)dx,\qquad
w_j^+=\frac{w_j^-Z_j}{\sum_kw_k^-Z_k},\qquad
\widetilde p_j^+(x)=\frac{L_j(y\mid x)p_j^-(x)}{Z_j}.
\]

Evidence must be captured before M254 projection. M254 represents state laws with255 normalized Beta/Bernstein basis functions at degree254. The state projection solves the unchanged constrained least-squares simplex problem; convexity of that projection does not imply convexity of the price objective. Each backend continues from its own updated law, which the fixture graph exercised. Stubs cannot verify likelihood/projection numerical accuracy.

Common initial draws and Brownian increments enable paired policy comparisons. With256 paths per hypothesis, d_ji=C_ji(reference)-C_ji(candidate),

\[
\widehat\Delta=\sum_jw_j\bar d_j,\qquad
\mathrm{SE}=\sqrt{\sum_jw_j^2s_{d,j}^2/256}.
\]

Positive Delta means lower candidate cost. The SE quantifies sampling uncertainty, not discretization bias; separate FV/M254/time refinements remain necessary. Exact receipt/input identity prevents mismatched policies or solver grids from being called reused or paired.

Admission requires all compatible measured whole-job rates r_k:

\[
B_{\mathrm{plan}}=B_{\mathrm{reserve}}+1.25\sum_kn_kr_k.
\]

Unknown/incompatible rates leave the estimate null and admission false. Planning factor1.25 is not a mathematical worst-case guarantee; hard parent/node deadlines remain necessary. The scientific day launch must receive a separate immutable registration after compatible origins/rates and full admission exist.

## Durable evidence and concrete next proposal

All five run directories, raw receipts, ledgers, registrations, hashes, logs and failed parent records are preserved in day_check_artifacts_20261008.zip. Its durable identity and SHA256 are in day_check_artifact_location_20261008.json. This is an evidence archive, not a passed scientific result.

day_check_completion_budget_proposal_20261008.json proposes +20s solely for one read-only completion review with18s worker and2s stop/export reserve. It is not approved and authorizes no model evolution. The current2.6627s cannot cover that scope. After a completed control review, recover/pin all four own origin records and design separately bounded representative measurements. Full graph budget and preregistration remain blocked. G/F/WM and original native source bytes are preserved.
