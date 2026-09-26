# F2-I: stochastic dynamic-price optimization

F2-I completes the stochastic BM1 workflow by optimizing the price signal
against the stochastic tracking objective introduced in F2-H.

The decision variable is a vector of piecewise-constant normalized prices

\[
U=(U_1,\ldots,U_N),
\qquad
0\le U_j\le1.
\]

For fixed interval edges, F2-I solves

\[
\boxed{
\min_{U\in[0,1]^N}
J_{\mathrm{stoch}}(U)
}
\]

with

\[
J_{\mathrm{stoch}}(U)
=
\int_0^T
\mathbb E[(D_t-A(t))^2]\,dt
=
J_{\mathrm{bias}}(U)
+
J_{\mathrm{variance}}(U).
\]

The primary optimizer model is the pure finite-volume Fokker--Planck reference.
The dynamic Beta mixture is used as an independent surrogate optimizer-model
check, and Monte Carlo is reserved for final common-random-number validation.

No Water Model file is modified.

## 1. Non-trivial optimization benchmark

The F2-H constant target is useful for validating the objective but is not an
ideal optimization benchmark. F2-I instead reuses the structure of the
deterministic F1-D verification.

The horizon is one day with two price intervals,

\[
[0,0.5),
\qquad
[0.5,1].
\]

The deterministic reference prices are

\[
\boxed{
U_{\mathrm{det}}
=
[0.72,\ 0.28].
}
\]

Starting from the deterministic point state

\[
X_0=0.5,
\]

these prices generate a non-trivial target demand trajectory A(t).

The stochastic optimization problem then uses the same target but starts from

\[
X_0\sim\operatorname{Beta}(20,20)
\]

and process noise

\[
\sigma_X=0.5\ \mathrm{day}^{-1/2}.
\]

Therefore the stochastic optimum is not expected to reproduce [0.72,0.28]
exactly. The variance contribution can shift the optimum even when the
deterministic reference tracks the target perfectly in the zero-uncertainty
benchmark.

## 2. Primary pure-FV objective evaluator

For any candidate price vector U, the evaluator:

1. builds the right-continuous piecewise-constant price signal;
2. solves the pure one-dimensional Fokker--Planck equation from the common
   Beta initial law;
3. computes E[D_t] and E[D_t^2] from the FV cell masses;
4. evaluates the F2-H stochastic tracking objective.

Thus the optimizer sees a deterministic map

\[
\boxed{
U
\longmapsto
J_{\mathrm{FV}}(U).
}
\]

There is no Monte Carlo sampling noise inside this primary optimization loop.

## 3. Bounded Powell multistart

The price-to-objective map is nonlinear and no global convexity claim is made.

F2-I therefore does not rely on one local derivative-based run from one start.
The reference problem has only two decision variables, so the workflow is:

1. evaluate a coarse objective surface over a Cartesian price grid;
2. select several lowest-cost coarse points;
3. add the deterministic reference and a neutral initial price vector;
4. launch bounded Powell from all unique starts;
5. retain the best successful solution.

The default price bounds are

\[
0\le U_1,U_2\le1.
\]

Powell is derivative-free and is used here because the objective includes a
full stochastic density solve and later also a projected mixture solve.

The coarse surface is retained as a diagnostic showing whether the multistart
solution sits in the expected low-cost region.

## 4. Dynamic-mixture optimizer-model check

F2-F established dynamic Beta-mixture evolution and F2-G showed that density
and BM moments remain robust even though high-degree coefficient vectors are
ill-conditioned.

F2-I therefore performs a secondary optimization with the dynamic mixture.

The optimization surrogate uses

\[
M=30
\]

on a reduced spatial grid for computational efficiency. It is started from

- the pure-FV optimum, and
- the deterministic reference price vector.

This produces

\[
U_{\mathrm{mix}}^*.
\]

The mixture optimum is not accepted on its own model's objective. It is
reevaluated with the pure-FV reference,

\[
J_{\mathrm{FV}}(U_{\mathrm{mix}}^*).
\]

Conversely, the pure-FV optimum is also evaluated with high-resolution
mixtures at

\[
M=30
\quad\text{and}\quad
M=38.
\]

This separates optimizer-model approximation from the actual reference
objective.

## 5. Final Monte Carlo validation

Monte Carlo is intentionally not used inside Powell.

After candidate prices have been selected, F2-I evaluates

\[
U_{\mathrm{det}},
\qquad
U_{\mathrm{FV}}^*,
\qquad
U_{\mathrm{mix}}^*
\]

with Euler--Maruyama.

All candidates use the same sampled initial states and exactly the same Brownian
increments. This common-random-number design turns the final comparison into a
paired experiment.

For each path r, define

\[
J_r(U)
=
\sum_k
w_k
(D_{r,k}(U)-A_k)^2.
\]

The paired pathwise improvement of a stochastic candidate over the
deterministic reference is

\[
\Delta J_r
=
J_r(U_{\mathrm{det}})
-
J_r(U_{\mathrm{candidate}}).
\]

F2-I reports

\[
\overline{\Delta J}
\]

and

\[
\operatorname{SE}(\overline{\Delta J})
=
\frac{s_{\Delta J}}{\sqrt N}.
\]

Positive \(\overline{\Delta J}\) means the stochastic candidate improves the
expected tracking cost.

Using common random numbers removes much of the comparison noise that would
arise if every candidate were simulated with independent paths.

## 6. What is compared

The reference run reports three candidate price vectors:

\[
U_{\mathrm{det}},
\qquad
U_{\mathrm{FV}}^*,
\qquad
U_{\mathrm{mix}}^*.
\]

For each candidate, pure FV reports

\[
J,
\qquad
J_{\mathrm{bias}},
\qquad
J_{\mathrm{variance}}.
\]

High-resolution M=30 and M=38 dynamic mixtures reevaluate all candidates.

Monte Carlo finally reports

\[
\hat J_{\mathrm{MC}},
\qquad
\operatorname{SE}(\hat J_{\mathrm{MC}})
\]

and the paired improvement relative to the deterministic reference.

The important question is not merely whether Powell returns success. The
candidate is convincing only if

1. multiple FV starts converge to the same low-cost region;
2. pure FV confirms a lower stochastic objective than the deterministic
   reference;
3. the mixture optimizer returns a nearby competitive decision;
4. Monte Carlo confirms the improvement with common random numbers.

## 7. Repository structure

New BM-only functionality is placed in

src/method/bm_1_stochastic_optimization.py

It contains

- the pure-FV stochastic price evaluator;
- candidate-price validation;
- bounded multistart Powell;
- the two-price coarse objective surface.

The controlled experiment is

src/exe/run_model/bm_1/stochastic_optimization_reference.py

The dynamic-mixture optimizer wrapper remains BM-specific to the experiment
and reuses the F2-F projected mixture evolution.

Tests are in

tests/test_bm_1_f2_stochastic_optimization.py

No shared Water Model method, objective, solver or validation implementation is
modified.

## 8. Run

From the repository root:

    python -B -m unittest discover -s tests -p "test_bm_1_f2_stochastic_optimization.py" -v
    python -B -m unittest discover -s tests -v
    python -B -m src.exe.run_model.bm_1.stochastic_optimization_reference

The full reference writes

files/output/graphs/bm_1/bm_1_stochastic_optimization_reference.png

The complete run is substantially more expensive than the preceding tests
because it performs a coarse FV objective surface, several Powell
optimizations, high-resolution mixture reevaluations and final 50,000-path
Monte Carlo validation.

## 9. F2 completion criterion

F2-I completes the planned F2 development when the workflow establishes a
reproducible stochastic price candidate and validates it independently.

At that point the project has implemented and checked the chain

\[
\boxed{
\text{SDE}
\rightarrow
\text{EM}
\rightarrow
\text{Fokker--Planck/FV}
\rightarrow
\text{MC--FV comparison}
\rightarrow
\text{Beta approximation}
\rightarrow
\text{dynamic mixture}
\rightarrow
\text{conditioning}
\rightarrow
\text{stochastic objective}
\rightarrow
\text{stochastic optimization}.
}
\]

Further work such as larger price vectors, adaptive mixture bases, parameter
estimation, economic objectives or alternative optimizers should then be
treated as extensions rather than required F2 completion steps.
