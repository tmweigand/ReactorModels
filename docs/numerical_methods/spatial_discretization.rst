Spatial Discretization
======================

The macroscale equations above are partial differential equations in time
:math:`t` and space :math:`x \in [0, L]`. They are discretized in space with
orthogonal collocation, which turns each PDE into a system of ODEs (or DAEs)
in time that can be solved with a stiff integrator.

Governing Idea
--------------

The solution is approximated by the Lagrange interpolating polynomial through
:math:`N` nodes :math:`x_i`,

.. math::
    :label: oc_interpolant

    u(x) \approx \sum_{j=1}^{N} u_j \, \ell_j(x),
    \qquad
    \ell_j(x) = \prod_{k \ne j} \frac{x - x_k}{x_j - x_k}

Because :math:`\ell_j(x_i) = \delta_{ij}`, the unknowns are the nodal values
:math:`u_j`. The PDE is enforced exactly at the interior nodes (the
collocation points), and the boundary conditions replace the equations at the
boundary nodes. The accuracy depends on where the nodes are placed.

Node Generation
---------------

The interior nodes are the :math:`n` roots of the Jacobi polynomial
:math:`P_n^{(\alpha,\beta)}`, which is orthogonal on :math:`[-1, 1]` with
weight :math:`(1-t)^{\alpha}(1+t)^{\beta}`. The roots are computed with the
Golub-Welsch algorithm, as the eigenvalues of the symmetric tridiagonal
matrix built from the three-term recurrence coefficients of the polynomial,
and are then mapped to :math:`[0, 1]`. Choosing the nodes this way gives
spectral convergence for smooth solutions.

The same eigen-decomposition gives the Gauss quadrature weights :math:`w_i`,
so integrals and averages over the domain are evaluated as

.. math::
    :label: oc_quadrature

    \int_0^L f \, dx \approx \sum_i w_i f_i,
    \qquad
    \bar{f} = \frac{1}{L} \sum_i w_i f_i

This is a plain integral only for the default Legendre case
(:math:`\alpha = \beta = 0`).

Boundary nodes are added to the interior roots so that boundary conditions
can be applied at :math:`x = 0` and :math:`x = L`. These nodes have zero
quadrature weight.

Differentiation Matrices
------------------------

Differentiating the interpolant and evaluating it at the nodes replaces every
spatial derivative with a matrix-vector product,

.. math::
    :label: oc_derivative_matrices

    \left. \pd{u}{x} \right|_{x_i} \approx \sum_j A_{ij} u_j,
    \qquad
    \left. \pd{^2 u}{x^2} \right|_{x_i} \approx \sum_j B_{ij} u_j

where the first derivative matrix is

.. math::
    :label: oc_first_derivative_matrix

    A_{ij} = \ell_j'(x_i) =
    \begin{cases}
    \displaystyle \frac{\prod_{k \ne i,j} (x_i - x_k)}{\prod_{k \ne j} (x_j - x_k)}
    & i \ne j \\[3ex]
    \displaystyle \sum_{k \ne i} \frac{1}{x_i - x_k} & i = j
    \end{cases}

and the second derivative matrix is :math:`\mathbf{B} = \mathbf{A}\mathbf{A}`.
The matrices are built on :math:`[0, 1]` and scaled to the domain length by
:math:`\mathbf{A} \to \mathbf{A}/L` and :math:`\mathbf{B} \to \mathbf{B}/L^2`.

Multi-Element Collocation
-------------------------

For steep fronts (high Péclet number), a single high-degree polynomial
oscillates and its differentiation matrices become ill-conditioned. In this
case the domain is divided into :math:`n_e` elements of equal width
:math:`h = 1/n_e`, each with a low-order polynomial. The local matrices are
built once, scaled by

.. math::
    :label: oc_element_scaling

    \mathbf{A}_\mathrm{loc} \to \frac{\mathbf{A}_\mathrm{loc}}{h},
    \qquad
    \mathbf{B}_\mathrm{loc} \to \frac{\mathbf{B}_\mathrm{loc}}{h^2}

and placed on the diagonal of the global matrices. Neighboring elements share
the node at their junction, so the solution is continuous and the total
number of nodes is :math:`n_e(n+1)+1`. At a junction node, the derivative
row is the average of the left and right element rows, and the quadrature
weights are summed. The model is refined by increasing :math:`n_e` until the
results stop changing.