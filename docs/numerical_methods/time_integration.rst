Time Integration
=======================

After the spatial discretization described in the previous section, the
macroscale equations become a system of differential-algebraic equations
(DAEs) in time. The differential equations come from the interior nodes, and
the algebraic equations come from the boundary conditions and any other
constraints. The system is written in the implicit form

.. math::
    :label: dae_system

    \mathbf{F}\lrp{t, \mathbf{y}, \dot{\mathbf{y}}} = \mathbf{0}

where :math:`\mathbf{y}` is the vector of unknowns (the nodal values of all
species in all phases) and :math:`\dot{\mathbf{y}}` is its time derivative.

The system is integrated in time with the IDA solver from the SUNDIALS
package, which is designed for DAEs in the form of Equation :eq:`dae_system`.
IDA uses variable-order, variable-step backward differentiation formulas
(BDF), in which the time derivative at step :math:`n` is approximated by a
weighted sum of the solution at previous steps,

.. math::
    :label: bdf_approximation

    \dot{\mathbf{y}}_n \approx \frac{1}{h_n} \sum_{j=0}^{q} \alpha_j \mathbf{y}_{n-j}

where :math:`h_n` is the time step and :math:`q` is the order of the method
(1 to 5). Substituting this expression into Equation :eq:`dae_system` gives a
nonlinear algebraic system at each step, which is solved with Newton's method.
The Newton iteration uses the Jacobian

.. math::
    :label: dae_jacobian

    \mathbf{J} = \pd{\mathbf{F}}{\mathbf{y}} + \frac{\alpha_0}{h_n} \pd{\mathbf{F}}{\dot{\mathbf{y}}}

Because BDF methods are implicit, they remain stable for the stiff systems
that result from combining fast adsorption or reaction with slow transport.

At each step IDA automatically adjusts the time step and the order to keep the
local error within the user-specified relative and absolute tolerances. Before
the integration starts, the initial values :math:`\mathbf{y}_0` and
:math:`\dot{\mathbf{y}}_0` must be consistent, meaning that they satisfy
Equation :eq:`dae_system` at :math:`t = 0`. IDA can compute consistent initial
conditions from the initial values of the differential variables.