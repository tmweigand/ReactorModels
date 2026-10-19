Ogata-Banks
==============

In advective-dispersive transport, species :math:`i` is assumed to be entirely in the fluid phase 
:math:`f` with no exchange of mass between phases. Thus, the final term in Eqn :eq:`two_phase` is 
dropped

.. math::
    :label:

    \pd {\lrp{\gke^{f}\rho^{f}\mma if}}t + \del \vdot \lrp{\gke^{f}\rho^{f}\mma if \vec v^\sol{f}}
    - \del \vdot \lrp{\gke^{f}\rho^{f}D^{if}\del\mma if} = 0

Writing in simplified terms

.. math::
    :label: vector_simple_form

    \gke \pd Ct + \gke \vec v^\sol{f} \del \vdot C - \gke D_L \del \vdot \del C = 0

where :math:`\gke` is the bed void fraction, :math:`C` is the species concentration in fluid phase, and :math:`D_L` is the 
axial diffusion coefficient of the species in the fluid phase. Considering a cylindrical column where concentration only 
varies in the axial direction, Eqn :eq:`vector_simple_form` can be further simplified to

.. math::
    :label:

    \gke \pd Ct + \gke v \pd Cz - \gke D_L \pdn 2Cz = 0

where :math:`v` is the interstitial velocity of the fluid phase. Dividing through by :math:`\gke` results in

.. math::
    :label:

    \pd Ct + v \pd Cz - D_L \pdn 2Cz = 0

This is known as the advection-diffusion equation from which the Ogata-Banks analytical solution for 1-D transport was 
derived :cite:`OgataBanks1961`

.. math::
    :label: ogata_banks

    \frac C{C_o} = \frac 12 \lrb{\operatorname{erfc}\lrp{\frac {z - vt}{2\sqrt{D_Lt}}} +
    \exp\lrp{\frac {vz}{D_L}}\operatorname{erfc}\lrp{\frac {z + vt}{2\sqrt{D_Lt}}}}

where :math:`C_o` is the species concentration in the fluid phase entering the column and :math:`\operatorname{erfc}` is the 
error function.
