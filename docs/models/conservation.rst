General Conservation Equations
================================

The generalized macroscale conservation of mass equation for a generic species :math:`i` in the set of species :math:`\I_s` and a generic phase :math:`\gka` in the set of phases :math:`\I_p` is given as

.. math::
    :label: general_conservation_of_mass

    \pd{\lrp{\gke^{\gka}\rho^{\gka}\mma i\gka}} t 
    + \del \vdot \lrp{\gke^{\gka}\rho^{\gka}\mma i\gka \vec v^\sol{\gka}}
    - \del \vdot \lrp{\gke^{\gka}\rho^{\gka}D^{i\gka}\del\mma i\gka} 
    - \gke^{\gka} r^{i\gka} 
    - \sum_{\gkk \in \I_{c\gka}} \iema M{i\gkk}{i\gka} 
    = 0

for all species :math:`i \in \I_s` and phases :math:`\gka \in \I_p`, where :math:`\gke^{\gka}` is the volume fraction of the :math:`\gka` phase, :math:`\rho^{\gka}` is the density of :math:`\gka` phase, :math:`\mma i\gka`  is the mass fraction of species :math:`i` in the :math:`\gka` phase, :math:`\vec v^\sol{\gka}` is the velocity of :math:`\gka` phase, :math:`D^{i\gka}` is the diffusion coefficient of species :math:`i` in :math:`\gka` phase, :math:`r^{i\gka}` is the intrinsic average production rate of species :math:`i`, and :math:`\iema M{i\qes}{i\gka}` is the mass exchange of species :math:`i` between the :math:`\gka` phase  and the :math:`\gkk` interface, where :math:`\I_{c\gka}` is the set of interfaces connected to the  :math:`\gka` phase. 

Equation :eq:`general_conservation_of_mass` is the foundation of the reactor models presented in this documentation. 


