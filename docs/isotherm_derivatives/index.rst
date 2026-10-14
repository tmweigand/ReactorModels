Isotherm Derivatives
==========================

Recall Eqn :eq:`adsorptive_mass_balance`, the conservation of mass equation for a differential column slice

.. math::
    :label:

    \gke\pd Ct + \gke v \pd Cz - \gke D \pdn 2Cz + \rho_b\pd qt = 0

and Eqn :eq:`pore_and_surface_diffusion`, the conservation of mass equation within a spherical adsorbent particle

.. math::
    :label:

    \gke_p\pd {C_p}t - \frac 1{r^2} \pd {}r \lrb{r^2\gke_pD_p\pd {C_p}r} + \rho_s\pd qt  - \frac 1{r^2} \pd {}r\lrb{r^2\rho_sD_s\pd qr} = 0

Under the assumption of local equilibrium, the chain rule is applied to obtain a single unknown time derivative variable.

.. math::
    :label:

    \pd qt = \pd qC \pd Ct

Thus, it is necessary to compute the first derivative of the isotherm equation. Single-species isotherm equations have the form

.. math::
    :label:

    q_i = f(C_i)

where the first and second derivatives return a scalar quantity. For multi-species isotherms such as the competitive Freundlich

.. math::
    :label:

    C_i = \frac {q_i}{\sLb {j=1}m q_j} \lrb{\frac {\sLb {j=1}m n_jq_j}{n_iK_i}}^{n_i}

and Langmuir

.. math::
    :label:

    q_i = \frac {q_mK_iC_i}{1 + \sLb {j=1}m K_jC_j}

the solid phase concentration of species :math:`i` depends upon the liquid phase concentration of all species in solution, and vice versa.

.. math::
    :label:

    q_i = f(C_i,\dots,C_n) \text{ or } C_i = f(q_i,\dots,C_n)

Therefore, the first derivative returns a matrix of values called a Jacobian

.. math::
    :label:

    \label{jacobian}
    \ten J =
    \begin{bmatrix}
        \pd {q_i}{C_i} & \dots & \pd {q_i}{C_n} \\
        \vdots & \ddots & \vdots\\
        \pd {q_n}{C_i} & \dots & \pd {q_n}{C_n}
    \end{bmatrix}

and the second derivative returns an array of matrices (third-order tensor)

.. math::
    :label:

    \ten H_i =
    \begin{bmatrix}
        \pdn 2{q_i}{C_i} & \dots & \pdsm {q_i}{C_i}{C_n} \\
        \vdots & \ddots & \vdots\\
        \pdsm {q_i}{C_n}{C_i} & \dots & \pdn 2{q_i}{C_n}
    \end{bmatrix}

.. math::
    :label:

    \boldsymbol{\mathcal{H}} = \lrb{\ten H_i, \dots, \ten H_n}

where :math:`n` is the number of species in the system. The Jacobian matrix is reduced to a scalar quantity for each species through an Einstein summation, and input to the conservation equation for each species, respectively. Suppose you have the vector

.. math::
    :label:

    \dot {\vec C} =
    \begin{bmatrix}
        \pd {C_i}t\\
        \vdots\\
        \pd {C_n}t
    \end{bmatrix}

which you want to contract with the Jacobian matrix

.. math::
    :label:

    \begin{bmatrix}
        \pd {q_i}{C_i} & \dots & \pd {q_i}{C_n} \\
        \vdots & \ddots & \vdots\\
        \pd {q_n}{C_i} & \dots & \pd {q_n}{C_n}
    \end{bmatrix}
    \ \vdot \ 
    \begin{bmatrix}
        \pd {C_i}t\\
        \vdots\\
        \pd {C_n}t
    \end{bmatrix}
    = 
    \begin{bmatrix}
        \pd {q_i}{C_i}\pd {C_i}t + \dots + \pd {q_i}{C_n}\pd {C_n}t\\
        \vdots\\
        \pd {q_n}{C_i}\pd {C_i}t + \dots + \pd {q_n}{C_n}\pd {C_n}t
    \end{bmatrix}
    =
    \begin{bmatrix}
        \pd {q_i}t\\
        \vdots\\
        \pd {q_n}t
    \end{bmatrix}

or in Einstein notation

.. math::
    :label:

    J_{ij} \dot C_j = \dot q_i

which can be computed in ``numpy`` using ``numpy.einsum("ij,j -> i", J, dCdt)`` where ``J`` 
is :math:`\ten J` and ``dC_dt`` is :math:`\dot {\vec C}`. The numerical solver also requires 
a Jacobian matrix for the residual equations themselves. Since the Hessian is a 3rd-order 
tensor, it also must undergo an Einstein summation to become a matrix. The following 
operation is performed for each species

.. math::
    :label:

    \begin{bmatrix}
        \pdn 2{q_i}{C_i} & \dots & \pdsm {q_i}{C_i}{C_n} \\
        \vdots & \ddots & \vdots\\
        \pdsm {q_i}{C_n}{C_i} & \dots & \pdn 2{q_i}{C_n}
    \end{bmatrix}
    \ \vdot \ 
    \begin{bmatrix}
        \pd {C_i}t\\
        \vdots\\
        \pd {C_n}t
    \end{bmatrix}
    = 
    \begin{bmatrix}
        \pdn 2{q_i}{C_i}\pd {C_i}t + \dots + \pdsm {q_i}{C_i}{C_n}\pd {C_n}t\\
        \vdots\\
        \pdsm {q_i}{C_n}{C_i}\pd {C_i}t + \dots + \pdn 2{q_i}{C_n}\pd {C_n}t
    \end{bmatrix}
    =
    \begin{bmatrix}
        \pd {\dot q_i}{C_i}\\
        \vdots\\
        \pd {\dot q_i}{C_n}
    \end{bmatrix}

then vertically stacked to obtain the matrix

.. math::
    :label:

    \begin{bmatrix}
        \pd {\dot q_i}{C_i} & \dots & \pd {\dot q_i}{C_n}\\
        \vdots & \ddots &\vdots\\
        \pd {\dot q_n}{C_i} & \dots & \pd {\dot q_n}{C_n}
    \end{bmatrix}

or in Einstein notation

.. math::
    :label:

    H_{ijk}\dot C_k = \pd {\dot q_i}{C_j}

which can be computed using ``numpy.einsum("ijk,k->ij", H, dC_dt)`` where ``H`` is 
:math:`\boldsymbol{\mathcal{H}}`. These operations are performed at every axial and 
radial node, written simply in Einstein notation as

.. math::
    :label:

    J_{ijl}\dot C_{jl} = \dot q_{il}

and

.. math::
    :label:

    H_{ijkl}\dot C_{kl} = \lrp{\pd {\dot q_i}{C_j}}_l

where :math:`l` is the node index.