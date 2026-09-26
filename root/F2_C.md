# F2-C: pure one-dimensional finite-volume Fokker-Planck reference

F2-C extracts a probability-conserving grid reference from the current
finite-volume + beta-mixture implementation. It does not replace the existing
mixture solver. The purpose is to obtain an independent density reference
before Monte Carlo and mixture errors are compared.

## 1. Fokker-Planck equation

For the Ito SDE

$$
dX_t = a(t,X_t)\,dt + b(X_t)\,dW_t,
$$

the density satisfies

$$
\partial_t p
=
-\partial_x(ap)
+
\frac12\partial_{xx}(b^2p).
$$

Define

$$
D(x)=\frac12 b(x)^2.
$$

The conservative form is

$$
\partial_t p + \partial_x J = 0,
\qquad
J = ap - \partial_x(Dp).
$$

For BM1,

$$
a(t,x)=\frac{D_{\mathrm{BM1}}(x,U_t,B_t)-B_t}{C},
\qquad
b(x)=\sigma_Xx(1-x).
$$

The derivative acts on the complete product Dp. F2-C therefore does not
replace the diffusion term by D p_xx.

## 2. Cell probabilities

Partition [0,1] into N finite volumes of width Delta x. The numerical unknown
is the probability mass in each cell,

$$
m_i(t)=\int_{V_i}p(x,t)\,dx.
$$

A valid discrete distribution satisfies

$$
m_i\ge0,
\qquad
\sum_i m_i=1.
$$

The solver works directly with these masses. Density values returned for
plotting are p_i approximately m_i/Delta x. An initial mass vector must already
be normalized; the reference solver does not silently renormalize it.

## 3. Conservative flux and Markov generator

At an internal face use upwind advection and a centered difference of the full
product Dp:

$$
J_{i+1/2}
\approx
a_{i+1/2}^+\frac{m_i}{\Delta x}
+
a_{i+1/2}^-\frac{m_{i+1}}{\Delta x}
-
\frac{D_{i+1}m_{i+1}-D_im_i}{(\Delta x)^2},
$$

where a^+=max(a,0) and a^-=min(a,0). This gives transition rates

$$
R_i
=
\frac{a_{i+1/2}^+}{\Delta x}
+
\frac{D_i}{(\Delta x)^2},
$$

and

$$
L_i
=
\frac{\max(-a_{i-1/2},0)}{\Delta x}
+
\frac{D_i}{(\Delta x)^2}.
$$

At the physical boundaries the outward rates are zero,

$$
L_0=0,
\qquad
R_{N-1}=0.
$$

Collecting cell masses into vector m gives

$$
\dot{\mathbf m}=Q\mathbf m,
$$

with non-negative off-diagonal transition rates and

$$
\mathbf 1^TQ=0.
$$

Thus the discretization has the structure of a continuous-time Markov
generator and conserves total probability.

The project source used for F2-C describes the same reference philosophy:
finite volumes evolve cell probabilities, the operator is interpreted as a
Markov generator, upwind advection is used and implicit Euler advances the
system. The source later presents an SDEtools-style diffusion flux using
interface diffusion values. The present implementation instead discretizes
the earlier stated conservative product flux -partial_x(Dp) directly. This
choice is explicit here rather than treating the two formulas as identical.

## 4. Zero-flux boundaries

F2-C imposes

$$
J(0,t)=J(1,t)=0.
$$

In generator form this is implemented by omitting transitions from the first
cell to the left and from the last cell to the right. Probability can move
between interior neighboring cells but cannot leave the state domain.

This is a density boundary condition. It is not the same numerical operation
as reflecting an Euler-Maruyama proposal after a sample path has crossed a
boundary.

## 5. Implicit Euler

For one internal time interval of length h, coefficients are frozen at its left
endpoint and cell masses are advanced by

$$
\mathbf m^{n+1}
=
\mathbf m^n
+
hQ_n\mathbf m^{n+1}.
$$

Equivalently,

$$
\boxed{
(I-hQ_n)\mathbf m^{n+1}=\mathbf m^n.
}
$$

The matrix is tridiagonal and is solved by the Thomas algorithm. Since
1^T Q_n=0,

$$
\mathbf 1^T\mathbf m^{n+1}
=
\mathbf 1^T\mathbf m^n.
$$

No density clipping and no mass renormalization are performed after a step.
If a materially negative cell mass appears, the reference raises an error.

## 6. Output grid and price jumps

As in F2-B, output times are separated from internal integration times.

- t_eval determines returned density snapshots.
- breakpoints inserts known input discontinuities.
- max_step limits the internal implicit-Euler step.

The generator is evaluated at the left endpoint, so a known price change must
be an integration boundary. The F2-C reference uses the same deliberately
unaligned 12.5 h price change as the F2-B EM reference.

## 7. Initial law

The old beta-mixture solver accepts a scalar init_cond and replaces the point
by a finite beta-mixture approximation. F2-C instead accepts explicit cell
probabilities.

The controlled reference uses

$$
X_0\sim\operatorname{Beta}(20,20)
$$

and computes exact initial cell probabilities from beta CDF differences,

$$
m_i(0)
=
F_{\mathrm{Beta}}(x_{i+1/2})
-
F_{\mathrm{Beta}}(x_{i-1/2}).
$$

This smooth law can later be sampled directly by Monte Carlo in F2-D, allowing
Monte Carlo and finite volume to start from the same distribution.

## 8. Verification

The focused tests check:

1. non-negative off-diagonal generator rates and zero column sums;
2. exact preservation under a zero operator;
3. stationary uniform density for zero drift and constant diffusion with
   zero-flux boundaries;
4. decreasing error under refinement for an exact Neumann heat-equation
   benchmark;
5. rejection of an unnormalized initial mass vector;
6. BM1 mass conservation, non-negativity, breakpoint handling and absence of
   clipping or renormalization.

For the heat benchmark,

$$
p(x,t)
=
1
+
A e^{-D\pi^2t}\cos(\pi x)
$$

solves p_t=D p_xx with zero normal derivative at x=0 and x=1. Exact cell
probabilities are obtained by integrating this density over each control
volume.

## 9. Repository layout

- src/solver/finite_volume.py: generic finite_volume_1d reference alongside the
  existing 2D solver.
- src/validation/solver/finite_volume.py: one-dimensional validation.
- src/solver/__init__.py and src/validation/solver/__init__.py: public API.
- src/model/bm_1.py: BM1 wrapper and moment/demand evaluation.
- src/exe/run_model/bm_1/fp_reference.py: controlled F2-C reference run.
- tests/test_bm_1_f2_fp_fv.py: focused F2-C regression tests.

The existing bm_1_fokker_planc beta-mixture implementation is intentionally
left unchanged.

## 10. Run from repository root

    python -B -m unittest discover -s tests -p "test_bm_1_f2_fp_fv.py" -v
    python -B -m unittest discover -s tests -v
    python -B -m src.exe.run_model.bm_1.fp_reference
    python -B -m src.exe.run_model.bm_1.fp_price_step

The new reference writes
files/output/graphs/bm_1/bm_1_fp_reference.png.

## 11. Next step

F2-D will initialize Monte Carlo paths from the same Beta(20,20) law and
compare Monte Carlo and finite volume on the same BM1 inputs and output times.
Only after that comparison should beta-mixture projection error be isolated.
