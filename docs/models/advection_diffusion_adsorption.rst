Advection-Diffusion-Adsorption Model
=======================================

These equations are obtained by applying Equation :eq:`general_conservation_of_mass` to each phase with the following assumptions: (1) species :math:`i` is transported by advection and diffusion in the water phase; (2) the solid phase is stationary, so there is no advection in the solid phase; (3) surface diffusion of species :math:`i` in the solid phase is neglected; (4) no homogeneous reactions occur in either phase, so :math:`r^{iw} = r^{is} = 0`; and (5) the only interaction between the phases is the adsorption and desorption of species :math:`i` at the water-solid interface.

For the water phase, the conservation of mass equation is

.. math::
    :label: adv_diff_ads_water

    \pd{\lrp{\gke^{w}\rho^{w}\mma i w}}{t}
    + \del \vdot \lrp{\gke^{w}\rho^{w}\mma i w \vec v^{w}}
    - \del \vdot \lrp{\gke^{w}\rho^{w}D^{iw}\del\mma i w}
    - \iema M{is}{iw} = 0

and for the solid phase it is

.. math::
    :label: adv_diff_ads_solid

    \pd{\lrp{\gke^{s}\rho^{s}\mma i s}}{t}
    - \iema M{iw}{is} = 0

where :math:`\iema M{is}{iw}` is the rate of mass transfer of species
:math:`i` from the solid phase into the water phase (desorption), and
:math:`\iema M{iw}{is}` is the rate of mass transfer from the water phase
into the solid phase (adsorption). Because mass is conserved at the
interface,

.. math::
    :label: adsorption_exchange_balance

    \iema M{iw}{is} = - \iema M{is}{iw}

so the mass adsorbed onto the solid is exactly the mass removed from the
water, and the two equations are coupled through this term. A closure
relationship for :math:`\iema M{iw}{is}`, such as a linear equilibrium or a
kinetic adsorption model, is required to close the system.