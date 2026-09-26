# F2-E: static Beta-mixture representation error

F2-E isolates the error caused by representing a probability density with the
existing finite Beta-mixture family. It does **not** evolve mixture
coefficients in time.

The validated pure finite-volume solution from F2-C is treated as the target
density at each output time. Each target snapshot is projected independently
onto several Beta-mixture spaces. This separates representation/projection
error from time-integration error.

Only BM-specific files are changed. The shared
`src/method/fokker_planc.py` implementation is used but not modified because
Water Model 1 also depends on it.

## 1. Beta basis

For degree $M$, the existing basis is

$$
ell_j(x)
=
(M+1)inom{M}{j}
x^j(1-x)^{M-j},
qquad
j=0,ldots,M.
$$

Equivalently,

$$
ell_j
=
operatorname{Beta}(j+1,M-j+1).
$$

The numerical code evaluates these functions on the finite-volume cell
centers and applies the same discrete basis normalization already used by the
existing mixture method.

A degree-$M$ approximation has the form

$$
hat p_M(x)
=
sum_{j=0}^{M}c_jell_j(x),
$$

subject to

$$
c_jge0,
qquad
sum_{j=0}^{M}c_j=1.
$$

Hence the approximation is itself a probability mixture.

## 2. Static constrained projection

At each pure-FV output time $t_k$, let $p_k$ denote the finite-volume density
on the common spatial grid.

F2-E computes

$$
oxed{
min_{cge0, mathbf 1^Tc=1}
rac12
left|
Phi c-p_k
ight|_W^2
}
$$

with

$$
W=Delta x,I.
$$

In matrix form,

$$
H=Phi^T WPhi,
qquad
q=-Phi^TWp_k.
$$

This is exactly the constrained least-squares projection already implemented
by the existing `bm_1_fokker_planc.project_density` route.

Crucially, the projection is performed **independently for every snapshot**.
There is no finite-volume prediction followed by repeated mixture projection
and no propagation of previous mixture coefficients. Therefore any measured
difference is representation/projection error, not accumulated dynamic
mixture error.

## 3. Variance floor

The $j$th degree-$M$ basis component has mean

$$
mu_j
=
rac{j+1}{M+2}
$$

and variance

$$
operatorname{Var}(Xmid j)
=
rac{mu_j(1-mu_j)}{M+3}.
$$

For a convex mixture with overall mean $m$, the law of total variance gives

$$
operatorname{Var}(X)
=
mathbb E[operatorname{Var}(Xmid j)]
+
operatorname{Var}(mu_j).
$$

This implies the lower bound

$$
oxed{
operatorname{Var}(X)
ge
rac{m(1-m)}{M+3}.
}
$$

This is a genuine representation limitation: if the target distribution is
narrower than this floor, no choice of non-negative mixture coefficients can
match its mean and variance simultaneously.

For the shared initial law

$$
X_0simoperatorname{Beta}(20,20),
$$

we have

$$
mathbb E[X_0]=rac12
$$

and

$$
operatorname{Var}(X_0)
=
rac{20cdot20}{40^2cdot41}
=
rac{0.25}{41}.
$$

For $M=20$ the centered variance floor is

$$
rac{0.25}{23},
$$

which is substantially larger than the target variance. Therefore a
degree-20 mixture cannot represent the narrow initial distribution exactly.

For $M=38$,

$$
rac{0.25}{M+3}
=
rac{0.25}{41},
$$

and the basis contains

$$
ell_{19}(x)
=
operatorname{Beta}(20,20).
$$

Thus $M=38$ is mathematically special for the chosen initial law: the
continuous target density is itself one of the degree-38 basis functions.

## 4. Degrees studied

The controlled F2-E experiment uses

$$
Min{5,10,20,30,38}.
$$

The lower degrees show the representation limitation as the basis is refined,
while $M=38$ provides a useful anchor because of the exact
$operatorname{Beta}(20,20)$ basis component at the initial time.

## 5. Error measures

For every degree and every output time, F2-E reports the discrete density
$L^2$ error

$$
|p-hat p_M|_{L^2}
approx
left(
Delta xsum_i
(p_i-hat p_{M,i})^2
ight)^{1/2}.
$$

The same cell-mass measures introduced in F2-D are reused:

$$
L^1
=
sum_i|m_i^{FV}-m_i^{mix}|,
$$

the maximum CDF difference, and the equal-grid approximation of
Wasserstein-1.

It also compares

$$
mathbb E[X_t],
qquad
operatorname{Var}(X_t),
qquad
mathbb E[D_t].
$$

Because the QP objective is a weighted density $L^2$ projection, the $L^2$
quantity is the primary direct measure of the projection problem. The other
measures show whether a small density error also preserves the stochastic
quantities relevant to BM1.

## 6. What F2-E does not test

F2-E does not assess the complete historical
`bm_1_fokker_planc.solve` evolution. That route contains finite-volume
prediction, post-correction and repeated projection.

The distinction is

$$
oxed{
	ext{F2-E: pure static representation error}
}
$$

versus the later question

$$
oxed{
	ext{dynamic mixture evolution error}.
}
$$

Only after the static approximation capacity is understood should the
additional error caused by repeated time evolution/projection be evaluated.

## 7. BM-only repository changes

- `src/exe/run_model/bm_1/beta_projection_reference.py`: static F2-E study.
- `src/diagnostics/bm_1.py`: adds density $L^2$ and the analytical
  Beta-mixture variance floor.
- `tests/test_bm_1_f2_beta_projection.py`: theory, exact initial-law
  representation and small integration tests.
- `root/F2_E.md`: this mathematical note.

No Water Model file and no shared method, solver or tool implementation is
changed.

## 8. Run

From the repository root:

```powershell
python -B -m unittest discover -s tests -p "test_bm_1_f2_beta_projection.py" -v
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.beta_projection_reference
```

The experiment writes

`files/output/graphs/bm_1/bm_1_beta_projection_reference.png`.

## 9. Next step

If F2-E shows that sufficiently rich Beta spaces represent the pure-FV
snapshots accurately, the next step is to measure the additional error caused
by the **dynamic** mixture evolution itself. That comparison should use the
same shared initial law, BM1 inputs, output times and pure-FV reference.
