Pore and/or Surface Diffusion
===============================

The pore and/or surface diffusion models :cite:`Friedman1984` couple the conservation of mass equations between 
the bulk fluid domain and the particle domain where species may be present in the solid phase as well as the pore 
liquid. Thus, Eqn :eq:`two_phase` is written in terms of fluid phase :math:`f` and particle phase :math:`p`

.. math::
    :label: psdm_mass_balance

    \pd {\lrp{\gke^{f}\rho^{f}\mma if}}t + \del \vdot \lrp{\gke^{f}\rho^{f}\mma if \vec v^\sol{f}}
    - \del \vdot \lrp{\gke^{f}\rho^{f}D^{if}\del\mma if} -\iema M{ip}{if} = 0

Writing Eqn :eq:`psdm_mass_balance` in simplified terms

.. math::
    :label: overall_mass_balance

    \gke \pd Ct + \gke v \pd Cz - \gke D_L \pdn 2Cz - \iema M{ip}{if} = 0

A conservation of mass equation must also be written for the particle domain. Accordingly, Eqn 
:eq:`two_phase` is written in terms of particle phase quantities

.. math::
    :label: particle_conservation

    \pd {\lrp{\gke^{p}\rho^{p}\mma ip}}t + \del \vdot \lrp{\gke^{p}\rho^{p}\mma ip \vec v^\sol{p}} - 
    \del \vdot \lrp{\gke^{p}\rho^{p}D^{ip}\del\mma ip} - \iema M{if}{ip} = 0

Since there is no advective transport of species within particles, Eqn :eq:`particle_conservation` becomes

.. math::
    :label: particle_mass_balance

    \pd {\lrp{\gke^{p}\rho^{p}\mma ip}}t - \del \vdot \lrp{\gke^{p}\rho^{p}D^{ip}\del\mma ip}
    - \iema M{if}{ip} = 0

The particle domain can be divided further into the solid and pore liquid phases. Thus the solid phase mass 
balance is given by

.. math::
    :label: particle_solid_balance

    \rho_p\pd qt - D_s \del \vdot \del \rho_pq - \iema M{l}{s} = 0

and the pore liquid phase mass balance is given by

.. math::
    :label: pore_liquid_balance

    \gke_p\pd {C_p}t - D_p \del \vdot \del \gke_pC_p - \iema M{s}{l} = 0

where :math:`C_p` is the pore liquid concentration, :math:`\gke_p` is the particle porosity, :math:`\rho_p` is the apparent 
particle density and :math:`D_s` is the surface diffusion coefficient, and :math:`D_p` is the pore diffusion coefficient. 
Assuming the species concentration only changes in the radial direction, Eqn :eq:`particle_solid_balance` may be 
written in spherical coordinates as

.. math::
    :label: surface_diffusion_model

    \rho_p\pd qt + \frac 1{r^2} \pd {}r \lrb{r^2D_s\rho_p\pd {q}r} 
    - \iema M{l}{s} = 0

and Eqn :eq:`pore_liquid_balance` becomes

.. math::
    :label: pore_diffusion_model

    \gke_p\pd {C_p}t - \frac 1{r^2} \pd {}r \lrb{r^2D_p\gke_p\pd{C_p}r} 
    - \iema M{s}{l} = 0

Under the assumption of local equilibrium, Eqn :eq:`surface_diffusion_model` and Eqn :eq:`pore_diffusion_model` can 
be added together. Since :math:`\iema M{l}{s} = - \iema M{s}{l}`, the intraparticle mass balance can be written as a 
single equation

.. math::
    :label: pore_and_surface_diffusion

    \rho_p\pd qt + \gke_p\pd {C_p}t - \frac 1{r^2} \pd {}r \lrb{r^2\lrp{D_s\rho_p\pd {q}r + D_p\gke_p\pd{C_p}r}} = 0

which is solved simultaneously with Eqn :eq:`overall_mass_balance`. A kinetic expression for transfer of species from 
the bulk fluid domain to the particle domain is needed to solve Eqn :eq:`pore_and_surface_diffusion` 
and Eqn :eq:`overall_mass_balance`. The relation for film transfer with a linear driving force is given by

.. math::
    :label: film_transfer

    - \iema M{f}{p} = \frac {A_pk_f}{Q}\lrb{C(z,t)-C_p(r=R,z,t)}

where :math:`k_f` is the film transfer coefficient and :math:`A_p` is the total surface area available for 
mass transfer. For spherical particles, :math:`A_p` is given by

.. math::
    :label:

    A_p = N_pS_p = \frac {3x}{4\pi R^3\rho_p}4\pi R^2 = \frac {3x}{R\rho_p}

where :math:`N_p` is the number of particles and :math:`S_p` is the surface area of a particle. 
Substituting

.. math::
    :label:

    \frac xQ = (1-\gke)\rho_p 

into Eqn :eq:`film_transfer`

.. math::
    :label: film_transfer_final

    - \iema M{f}{p} = \frac {3k_f(1-\gke)}{\gke R}\lrb{C(z,t)-C_p(r=R,z,t)}

gives the expression for the rate of transfer of species between the bulk fluid and particle domain. Eqn 
:eq:`film_transfer_final` is substituted into Eqn :eq:`overall_mass_balance` to obtain the final 
conservation of mass equation in the bulk fluid domain

.. math::
    :label:

    \gke \pd Ct + \gke v \pd Cz - \gke D_L \pdn 2Cz + \frac {3k_f(1-\gke)}{R}\lrb{C(z,t)-C_p(r=R,z,t)} = 0

The boundary condition for Eqn :eq:`pore_and_surface_diffusion` at the surface of the particle is given by

.. math::
    :label:

    D_s\rho_p\pd {q}r + D_p\gke_p\pd{C_p}r = k_f\lrb{C(z,t)-C_p(r=R,z,t)}

and the boundary condition at the center of the particle is given by 

.. math::
    :label:

    \rho_p\pd {q(r=0,z,t)}r + \gke_p\pd{C_p(r=0,z,t)}r = 0

Under the assumption of local equilibrium, :math:`q` and :math:`C_p` are related by the Freundlich 
isotherm equation for any given time and position

.. math::
    :label: freundlich_isotherm

    q = KC_p^{1/n} 

where :math:`K` is the capacity constant and  :math:`1/n` is the intensity constant. If the species is 
present at trace concentrations, a linear isotherm is often assumed (:math:`n=1`). Thus, Eqn 
:eq:`freundlich_isotherm` simplifies to 

.. math::
    :label:

    q = KC_p

The film transfer coefficient coefficient may be obtained using the Gnielinski correlation

.. math::
    :label:

    k_f = \frac {\lrb{1+1.5(1-\gke)}D}{d_p}\lrb{2+0.644\mathrm{Re}^{1/2}\mathrm{Sc}^{1/3}}

where :math:`d_p` is the particle diameter, :math:`D` is the liquid diffusivity of the species in water. 
The particle Reynolds number has the following expression

.. math::
    :label:

    \mathrm{Re}=\frac {\rho d_p u}{\mu}

where :math:`\rho` is the density of water and :math:`\mu` is the dynamic viscosity of water. The Schmidt 
number is given by

.. math::
    :label:

    \mathrm{Sc}=\frac {\mu}{\rho D}

For species where experimental values of :math:`D` are not available, it may be approximated 
with the Hayduk-Laudie equation

.. math::
    :label:

    D = \frac {13.26\times10^{-5}}{\mu^{1.14} V_b^{0.589}}

where :math:`V_b` is the molar volume of the species at its boiling point. The pore diffusion 
coefficient may be approximated as

.. math::
    :label:

    D_p = \frac D{\tau_p}

where :math:`\tau_p` is the particle tortuosity. The surface diffusion coefficient is given by

.. math::
    :label:

    D_s = \frac {D\gke_pC_o^{\lrp{1 - 1/n}}}{\tau_p\rho_pK} \times \text{SPDFR}

where SPDFR is the surface to pore diffusion flux ratio.