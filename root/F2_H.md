# F2-H: stochastic BM1 tracking objective

F2-H introduces the stochastic objective that will later be minimized in the
dynamic-price optimizer.

This is new BM functionality, not another solver-only verification step.

For deterministic target/available demand A(t) and stochastic BM1 demand D_t,
the relevant tracking objective is

\[
\boxed{
J(U)
=
\int_0^T
\mathbb E\!\left[(D_t-A(t))^2\right]
\,dt .
}
\]

F2-H implements this objective independently of the stochastic solver and then
evaluates the same fixed price path with Monte Carlo, pure finite volume and
the dynamic Beta-mixture approximation.

No stochastic price optimization is performed yet.

## 1. Bias--variance decomposition

Because A(t) is deterministic,

\[
\mathbb E[(D_t-A)^2]
=
\mathbb E[D_t^2]
-
2A\mathbb E[D_t]
+
A^2 .
\]

Writing

\[
\operatorname{Var}(D_t)
=
\mathbb E[D_t^2]
-
\mathbb E[D_t]^2
\]

gives

\[
\boxed{
\mathbb E[(D_t-A)^2]
=
(\mathbb E[D_t]-A)^2
+
\operatorname{Var}(D_t).
}
\]

Hence

\[
J(U)
=
J_{\mathrm{bias}}(U)
+
J_{\mathrm{variance}}(U),
\]

where

\[
J_{\mathrm{bias}}
=
\int_0^T
(\mathbb E[D_t]-A(t))^2\,dt
\]

and

\[
J_{\mathrm{variance}}
=
\int_0^T
\operatorname{Var}(D_t)\,dt.
\]

This distinction matters. Replacing stochastic demand by only its mean would
compute J_bias and would omit the uncertainty contribution.

## 2. Demand moments from Monte Carlo

With N simulated demand paths D_t^(r),

\[
\widehat{\mathbb E[D_t]}
=
\frac1N
\sum_{r=1}^N D_t^{(r)}
\]

and

\[
\widehat{\mathbb E[D_t^2]}
=
\frac1N
\sum_{r=1}^N
(D_t^{(r)})^2 .
\]

The population-moment convention is intentional. It guarantees the exact
empirical identity

\[
\frac1N
\sum_{r=1}^N
(D_t^{(r)}-A_t)^2
=
(\bar D_t-A_t)^2
+
\left(
\overline{D_t^2}-\bar D_t^2
\right).
\]

For every Monte Carlo path, F2-H also forms the pathwise cost

\[
J^{(r)}
=
\sum_k
w_k
(D_k^{(r)}-A_k)^2 .
\]

The Monte Carlo objective estimate is

\[
\hat J_{\mathrm{MC}}
=
\frac1N
\sum_{r=1}^N J^{(r)}
\]

and its sampling standard error is

\[
\boxed{
\operatorname{SE}(\hat J_{\mathrm{MC}})
=
\frac{s_J}{\sqrt N},
}
\]

where s_J is the sample standard deviation of the pathwise objectives.

This standard error automatically respects temporal correlation along each
path because uncertainty is evaluated after integrating each complete path.

## 3. Demand moments from a probability density

For a finite-volume or mixture representation on cells i, let m_i(t) denote
cell probability mass and

\[
D_i(t)=D(x_i,U_t,B_t).
\]

Then

\[
\mathbb E[D_t]
\approx
\sum_i D_i(t)m_i(t)
\]

and

\[
\boxed{
\mathbb E[D_t^2]
\approx
\sum_iD_i(t)^2m_i(t).
}
\]

Therefore

\[
\operatorname{Var}(D_t)
\approx
\sum_iD_i(t)^2m_i(t)
-
\left(
\sum_iD_i(t)m_i(t)
\right)^2.
\]

F2-H evaluates these same formulas for pure finite-volume cell masses and
dynamic Beta-mixture cell masses.

The objective code therefore does not need to know which density solver
produced the probability masses.

## 4. Breakpoint-safe midpoint quadrature

The reference price jumps from

\[
0.75
\longrightarrow
0.25
\]

at 12.5 h.

Demand can therefore jump at this time even though the state distribution
itself remains continuous.

F2-H does not integrate a polynomial interpolant across a price discontinuity.
Instead, it constructs quadrature edges such that every known price breakpoint
is an edge.

Each resulting interval [t_i,t_(i+1)] contains no input discontinuity. The
objective is evaluated at

\[
\tau_i
=
\frac{t_i+t_{i+1}}2
\]

and integrated as

\[
\boxed{
J
\approx
\sum_i
(t_{i+1}-t_i)
\,
\mathbb E[(D_{\tau_i}-A_{\tau_i})^2].
}
\]

The default maximum objective interval is 15 minutes.

The stochastic solvers return states/densities at all quadrature midpoints,
while their internal integration still receives the 12.5 h price breakpoint
explicitly.

This separates the quadrature grid used to evaluate J from each solver's own
internal numerical stepping.

## 5. Common F2-H reference experiment

All three methods use the same F2 reference case:

\[
T=1\text{ day},
\qquad
B=0.3,
\qquad
C=1,
\qquad
k=2,
\qquad
\Xi=1,
\]

\[
\sigma_X=0.5\ \mathrm{day}^{-1/2},
\]

and

\[
X_0\sim\operatorname{Beta}(20,20).
\]

The target is the model's existing deterministic availability signal A(t).
In the controlled reference case, A=B=0.3.

Monte Carlo uses uncorrected Euler--Maruyama with boundary="none". Any raw
boundary violation remains a failure of the reference experiment.

Pure finite volume uses the conservative F2-C solver without clipping or
renormalization.

The dynamic mixture uses the F2-F projected evolution. The default objective
study reports M in {20,30,38}.

## 6. Cross-method validation

F2-H reports

\[
J_{\mathrm{MC}},
\qquad
J_{\mathrm{FV}},
\qquad
J_{\mathrm{mix},M},
\]

and for each one the decomposition

\[
J
=
J_{\mathrm{bias}}
+
J_{\mathrm{variance}}.
\]

For Monte Carlo it additionally reports SE(J_MC).

The difference |J_MC-J_FV| is shown both absolutely and in Monte Carlo
standard-error units.

This is diagnostic rather than an assertion that FV must lie inside a pure
Monte Carlo confidence interval: EM and FV still have their own time/space
discretization errors.

For each mixture degree, F2-H reports |J_mix,M-J_FV|.

## 7. Deterministic/zero-variance limit

The objective implementation itself satisfies the exact limit

\[
\operatorname{Var}(D_t)=0
\quad\Longrightarrow\quad
J
=
\sum_i
w_i(D_i-A_i)^2.
\]

That is the deterministic squared-tracking objective evaluated with the same
quadrature.

This test is deliberately formulated as a zero-demand-variance statement.

Simply setting sigma_X=0 in the standard F2 reference case is not sufficient
to create a deterministic problem because X_0~Beta(20,20) still contains
initial uncertainty.

## 8. New BM functionality

The new reusable module is

src/method/bm_1_stochastic_objective.py

It provides breakpoint-safe midpoint quadrature, Monte Carlo demand moments,
discrete-density demand moments, stochastic bias/variance objective evaluation,
and pathwise Monte Carlo objective uncertainty.

The module is solver-independent at the objective level so the same code can
be used by the later F2-I optimizer.

The controlled executable is

src/exe/run_model/bm_1/stochastic_objective_reference.py

Tests are in

tests/test_bm_1_f2_stochastic_objective.py

No existing Water Model file or shared objective implementation is modified.
The deterministic src/method/objective.py remains unchanged.

## 9. Run

From the repository root:

    python -B -m unittest discover -s tests -p "test_bm_1_f2_stochastic_objective.py" -v
    python -B -m unittest discover -s tests -v
    python -B -m src.exe.run_model.bm_1.stochastic_objective_reference

The reference experiment writes

files/output/graphs/bm_1/bm_1_stochastic_objective_reference.png

## 10. Completion criterion and F2-I

F2-H is complete when

1. the bias--variance identity is verified numerically;
2. the zero-variance objective reduces to deterministic squared tracking;
3. the breakpoint-safe quadrature is verified;
4. MC, pure FV and dynamic mixture evaluate the same fixed price problem with
   documented differences;
5. Monte Carlo reports pathwise sampling uncertainty for the integrated
   objective.

Only after this map

\[
\boxed{
U
\longmapsto
J_{\mathrm{stoch}}(U)
}
\]

is verified should it be called repeatedly by an optimizer.

That later step is F2-I.
