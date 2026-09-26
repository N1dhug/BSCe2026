# F2-G: Beta-mixture conditioning and sensitivity

F2-F showed that the dynamic Beta-mixture approximation remains close to the
best same-degree static projection, but it also exposed very large condition
numbers for the projection Hessian at high degree.

For the default 201-cell reference grid, the reported full Hessian condition
numbers were approximately

$$
kappa(H)
approx
2.9	imes 10^{11},
qquad
3.2	imes 10^{16},
qquad
1.0	imes 10^{17}
$$

for

$$
M=20, 30, 38.
$$

F2-G determines whether these large algebraic condition numbers materially
affect the **density and BM-relevant moments**, or mainly make the mixture
coefficient vector non-unique/sensitive.

No shared Water-Model method, solver or tool implementation is changed.

## 1. Projection matrix and Hessian

On the common spatial grid, write the weighted least-squares projection as

$$
min_{cinDelta_M}
rac12|Ac-b|_2^2,
$$

where

$$
A=sqrt{Delta x},Phi^T,
qquad
b=sqrt{Delta x},p,
$$

and

$$
Delta_M
=
{c: c_jge0, mathbf 1^Tc=1}.
$$

The quadratic-program Hessian is

$$
H=A^TA.
$$

In exact arithmetic,

$$
kappa_2(H)
=
kappa_2(A)^2
$$

when $A$ has full column rank.

A large full $kappa(H)$ indicates that some combinations of coefficient
directions generate almost indistinguishable densities. However, the full
condition number may overstate the conditioning of the actual constrained
solution because many coefficients can be fixed at zero by the active-set
constraints.

## 2. Active-simplex tangent conditioning

Let

$$
F
=
{j: c_j>	au_{mathrm{active}}}
$$

denote the locally free coefficient set, with default

$$
	au_{mathrm{active}}=10^{-8}.
$$

The local equality constraint is

$$
mathbf 1_F^T d_F=0.
$$

Let the columns of $Z$ span this simplex tangent space:

$$
mathbf 1_F^TZ=0.
$$

The locally relevant curvature is then

$$
oxed{
H_{mathrm{tan}}
=
Z^T H_{FF} Z.
}
$$

F2-G reports

$$
kappa(H_{mathrm{tan}})
$$

for both the independent static F2-E projections and the F2-F dynamic mixture
coefficients at every output time.

This distinguishes

$$
	ext{full-basis ill-conditioning}
$$

from

$$
	ext{ill-conditioning in directions the constrained solution can actually move}.
$$

## 3. Feasible-start sensitivity

A convex QP should describe the same projected density regardless of the
feasible starting point used by the numerical optimizer.

For each pure-FV target density, F2-G compares the existing F2-E solution with
a second solve initialized from the uniform simplex point

$$
c_j^{(0)}=rac1{M+1}.
$$

It records

$$
|c_{mathrm{uniform}}-c_{mathrm{baseline}}|_1
$$

and

$$
|Phi c_{mathrm{uniform}}
-
Phi c_{mathrm{baseline}}|_{L^2},
$$

together with differences in mean and variance.

A large coefficient difference combined with a tiny density difference would
show that the coefficient representation is poorly identifiable while the
represented density remains robust.

## 4. Positive mass-preserving density perturbations

The second sensitivity experiment perturbs the pure-FV target itself.

Given a target density $p_i$ and a bounded centered direction $r_i$, define

$$
	ilde p_i
=
p_iexp(arepsilon r_i).
$$

The perturbed density is normalized as

$$
p_i^{(arepsilon)}
=
rac{	ilde p_i}
{sum_j	ilde p_jDelta x}.
$$

This construction ensures

$$
p_i^{(arepsilon)}ge0
$$

and

$$
sum_i p_i^{(arepsilon)}Delta x=1.
$$

The default perturbation sizes are

$$
arepsilon
in
{10^{-10},10^{-8},10^{-6}},
$$

with four deterministic random directions per output snapshot.

For each perturbed target, the QP is warm-started from the unperturbed optimal
coefficient vector.

## 5. Sensitivity measures

Let

$$
c
$$

and

$$
hat p=Phi c
$$

be the unperturbed projection, and

$$
c_arepsilon,
qquad
hat p_arepsilon=Phi c_arepsilon
$$

the perturbed projection.

F2-G records the actual input perturbation

$$
delta_{mathrm{in}}
=
|p_arepsilon-p|_{L^2},
$$

coefficient sensitivity

$$
delta_c
=
|c_arepsilon-c|_1,
$$

density sensitivity

$$
delta_p
=
|hat p_arepsilon-hat p|_{L^2},
$$

and the empirical amplification factors

$$
G_c
=
rac{delta_c}{delta_{mathrm{in}}},
qquad
G_p
=
rac{delta_p}{delta_{mathrm{in}}}.
$$

Differences in

$$
mathbb E[X]
$$

and

$$
operatorname{Var}(X)
$$

are also reported.

The distinction between $G_c$ and $G_p$ is central. A large $G_c$ does not
necessarily imply a physically important instability if several strongly
overlapping coefficient vectors represent essentially the same density.

## 6. What result would be reassuring?

The high-degree mixture is numerically useful if the experiment shows

$$
oxed{
	ext{coefficient sensitivity can be large while density/moment sensitivity remains small}.
}
$$

In that case individual $c_j(t)$ values should not be over-interpreted, but the
probability density and derived BM quantities remain meaningful.

A materially large density amplification, strong dependence on optimizer
initialization, or projection failures would instead indicate that the
high-degree approximation needs stabilization before it should be used in
later optimization work.

Possible remedies would then include regularization, basis scaling, a lower
degree, or a different/localized basis. These are not introduced in F2-G;
the purpose is first to determine whether they are actually necessary.

## 7. Repository changes

- `src/exe/run_model/bm_1/beta_conditioning_reference.py`:
  conditioning, active-simplex and perturbation-sensitivity study.
- `tests/test_bm_1_f2_beta_conditioning.py`:
  local-conditioning, positive perturbation and controlled integration tests.
- `root/F2_G.md`:
  this note.

No shared method, solver, validation or Water-Model implementation is changed.

## 8. Run

From the repository root:

```powershell
python -B -m unittest discover -s tests -p "test_bm_1_f2_beta_conditioning.py" -v
python -B -m unittest discover -s tests -v
python -B -m src.exe.run_model.bm_1.beta_conditioning_reference
```

The full experiment writes

`files/output/graphs/bm_1/bm_1_beta_conditioning_reference.png`.

## 9. Decision after F2-G

F2-G is deliberately a decision point rather than another solver extension.

If the density and moments are insensitive despite coefficient sensitivity,
the high-degree mixture can be retained and later work should focus on
density-level quantities rather than individual coefficients.

If density-level sensitivity is also large, numerical stabilization should be
addressed before the stochastic BM framework is used for optimization.
