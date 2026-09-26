# F2-F: dynamic Beta-mixture evolution

F2-F evaluates the **dynamic** Beta-mixture approximation for stochastic BM1.

F2-E answered a static approximation question:

> If the pure finite-volume density at a given time is known, how accurately
> can a degree-M Beta mixture represent it?

F2-F asks the next question:

> If the Beta mixture itself is propagated forward in time, how much additional
> error is introduced by the repeated prediction/projection dynamics?

Only BM-specific experiment/test/documentation files are added. Shared
Water-Model method, solver and tool implementations are not changed.

## 1. Dynamic mixture state

For degree $M$, the density approximation is

$$
hat p^n(x)
=
sum_{j=0}^{M} c_j^n ell_j(x),
$$

where

$$
c_j^nge 0,
qquad
sum_{j=0}^{M}c_j^n=1.
$$

The fixed basis is

$$
ell_j(x)
=
(M+1)inom{M}{j}x^j(1-x)^{M-j}
=
operatorname{Beta}(j+1,M-j+1).
$$

F2-F uses

$$
Min{20,30,38},
$$

because F2-E showed that these levels span the transition from a visibly
representation-limited basis to the degree-38 space that contains the initial
$operatorname{Beta}(20,20)$ density.

## 2. Initial condition

The same reference law is retained:

$$
X_0simoperatorname{Beta}(20,20).
$$

For every degree, the exact finite-volume initial density is first projected
onto the corresponding Beta-mixture space by the same constrained $L^2$
problem used in F2-E.

Thus F2-F and F2-E start from exactly the same degree-$M$ mixture at $t=0$.

For $M<38$, this initial mixture already contains representation error.
For $M=38$, the initial representation is essentially exact on the numerical
grid.

## 3. Explicit Euler prediction

Suppose the current mixture density is $hat p^n$ at time $t_n$.

The Fokker--Planck equation is

$$
partial_t p
=
-partial_x(ap)
+
rac12partial_{xx}(b^2p),
$$

with the BM1 coefficients

$$
a(t,x)
=
rac{D(x,U_t,B_t)-B_t}{C},
$$

and

$$
b(x)
=
sigma_X x(1-x).
$$

Using the existing conservative one-dimensional grid operator, F2-F forms the
raw explicit-Euler prediction

$$
oxed{
	ilde p^{n+1}
=
hat p^n
+
h,L_{FP}(t_n)hat p^n.
}
$$

The raw prediction generally does not lie in the fixed Beta-mixture space.

No clipping or renormalization is applied to $	ilde p^{n+1}$.

## 4. Projection after every internal step

The next dynamic mixture is found from

$$
oxed{
c^{n+1}
=
argmin_{cge0, mathbf 1^Tc=1}
rac12
left|
Phi c-	ilde p^{n+1}
ight|_W^2
}
$$

with

$$
W=Delta x I.
$$

Then

$$
hat p^{n+1}
=
Phi c^{n+1}.
$$

This projection is performed after **every internal Euler step**, not only at
the requested output times.

Therefore the method implemented in F2-F is

$$
oxed{
hat p^n
ightarrow
	ilde p^{n+1}
ightarrow
hat p^{n+1}
}
$$

repeated over the full horizon.

The constrained QP is warm-started from $c^n$, because the previous
coefficient vector is already feasible and is normally close to the next
solution.

## 5. Internal time step and positivity

The dynamic mixture uses an explicit Fokker--Planck prediction, so the internal
step is restricted by a positivity-oriented CFL condition.

At each cell, F2-F estimates the total outgoing conservative transition rate
from upwind advection and diffusion. The step satisfies

$$
h,r_{mathrm{out},i}
le
mathrm{CFL}
$$

for every cell.

The default is

$$
mathrm{CFL}=0.40.
$$

The actual internal step is

$$
h
=
min
left(
h_{mathrm{remaining}},
h_{max},
h_{mathrm{CFL}}
ight).
$$

Output times and the 12.5-hour price discontinuity are hard integration
boundaries, so no internal step crosses them.

The raw prediction is checked before projection. A materially negative raw
density causes the experiment to fail rather than being clipped.

## 6. Static representation error versus dynamic error

Let

$$
p^{FV}_k
$$

be the pure finite-volume reference at output time $t_k$.

From F2-E, let

$$
p^{*,M}_k
$$

be the independently optimized **best static degree-$M$ projection** of the
same finite-volume density.

F2-F produces the actual propagated dynamic mixture

$$
p^{dyn,M}_k.
$$

Two different errors are therefore reported.

### Total dynamic mixture error

$$
E_{mathrm{total}}^M(t_k)
=
left|
p^{dyn,M}_k-p^{FV}_k
ight|.
$$

This contains both

- finite-basis representation limitations, and
- error caused by dynamic repeated prediction/projection.

### Dynamic gap to the best same-degree representation

$$
E_{mathrm{gap}}^M(t_k)
=
left|
p^{dyn,M}_k-p^{*,M}_k
ight|.
$$

This measures how far the propagated mixture is from the best mixture that
could have been chosen at the same time if the pure-FV density were known.

This is not an additive decomposition:

$$
E_{mathrm{total}}

eq
E_{mathrm{static}}+E_{mathrm{gap}}
$$

in general.

However, by the triangle inequality,

$$
left|
p^{dyn,M}-p^{FV}
ight|
le
left|
p^{dyn,M}-p^{*,M}
ight|
+
left|
p^{*,M}-p^{FV}
ight|.
$$

The three curves therefore distinguish whether poor performance is primarily
caused by an insufficient basis or by the dynamic evolution.

## 7. Error measures

The same metrics as F2-E are retained:

- density $L^2$ error,
- cell-mass $L^1$ distance,
- CDF sup distance,
- equal-grid Wasserstein-1 approximation,
- error in $mathbb E[X_t]$,
- error in $operatorname{Var}(X_t)$,
- error in $mathbb E[D_t]$.

The density $L^2$ error remains the primary metric because the dynamic
projection minimizes a weighted $L^2$ residual at every step.

## 8. Conditioning and numerical diagnostics

For the fixed degree-$M$ basis, the weighted projection Hessian is

$$
H
=
Phi^T WPhi.
$$

The basis and spatial grid are fixed, so $H$ is constant in time for a given
degree. F2-F records

$$
operatorname{cond}(H)
$$

once per degree.

This is important because increasing $M$ improves representation power but can
also make the highly overlapping Beta basis increasingly ill-conditioned.

F2-F additionally records

- minimum raw Euler-prediction density,
- maximum raw prediction mass error,
- maximum projected mass error,
- maximum coefficient-sum error,
- minimum coefficient,
- maximum boundary coefficient $max(c_0,c_M)$,
- number of internal steps,
- minimum and maximum internal step.

The boundary coefficient diagnostic is useful for detecting collapse toward a
boundary-localized basis component.

The experiment explicitly reports

$$
	exttt{used\_clipping=False},
qquad
	exttt{used\_renormalization=False}.
$$

## 9. Relation to the historical BM1 mixture route

The historical `bm_1_fokker_planc.solve` routine performs a grid-density
Fokker--Planck evolution with internal post-correction and projects at output
levels.

F2-F intentionally does not use that full historical evolution as its test
object. Instead, it implements the discrete projected-mixture logic explicitly
inside the BM-only experiment:

$$
	ext{mixture}
ightarrow
	ext{Euler prediction}
ightarrow
	ext{QP projection}
$$

at every internal step.

The shared basis, flux operator, active-set QP solver and model equations are
reused without modifying the shared implementation used by Water Models.

## 10. Repository changes

- `src/exe/run_model/bm_1/beta_dynamic_reference.py`:
  dynamic mixture experiment and diagnostics.
- `tests/test_bm_1_f2_beta_dynamic.py`:
  zero-operator propagation and controlled BM1 dynamic reference tests.
- `root/F2_F.md`:
  this mathematical note.

No shared Water-Model method, solver or tool implementation is changed.

## 11. Run

From the repository root:

```powershell
python -B -m unittest discover -s tests -p "test_bm_1_f2_beta_dynamic.py" -v
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.beta_dynamic_reference
```

The full experiment writes

`files/output/graphs/bm_1/bm_1_beta_dynamic_reference.png`.

## 12. Interpretation

The most useful comparison is

$$
oxed{
	ext{dynamic mixture}
quad	ext{vs}quad
	ext{best static same-degree mixture}
quad	ext{vs}quad
	ext{pure FV}.
}
$$

If the dynamic mixture closely follows the best static projection, remaining
error is mainly due to the finite Beta basis.

If the dynamic mixture is substantially worse than the static projection,
then repeated projected time evolution is introducing a significant additional
error even though the chosen basis could represent the reference density more
accurately.
