Model Development
========================

The domain of a fixed bed system contains fluid phase and solid phase. The fluid phase enters and leaves the domain through its 
boundaries (i.e. the top and bottom of a column) while the solid phase remains stationary within the domain. Fixed bed models are 
built upon the conservation of mass for a species in a phase over a differential column slice, consisting of terms for accumulation 
(first term) of species in a phase in the domain and advective transport (second term) of species in a phase through the domain

.. math::
    :label: conservation_of_mass

    \pd{\lrp{\gke^{\gka}\rho^{\gka}\mma i\gka}}t +
    \del \vdot \lrp{\gke^{\gka}\rho^{\gka}\mma i\gka \vec v^\sol{i\gka}} = 0

where :math:`\gke^{\gka}` is the volume fraction of phase :math:`\gka`, :math:`\rho^{\gka}` is the density of phase :math:`\gka`, 
:math:`\mma i\gka`  is the mass fraction of species :math:`i` in phase :math:`\gka`, and :math:`\vec v^\sol{i\gka}` 
is the velocity vector of species :math:`i` in phase :math:`\gka`.

Transport of species in a phase can occur through both advection and random (Brownian) motion of the species within the phase 
(i.e. molecular diffusion). Therefore, the species velocity is split into the bulk velocity and the deviation from the bulk 
velocity due to molecular diffusion

.. math::
    :label: species_in_phase

    \pd {\lrp{\gke^{\gka}\rho^{\gka}\mma i\gka}}t + \del \vdot \lrp{\gke^{\gka}\rho^{\gka}\mma i\gka \vec v^\sol{\gka}}
    + \del \vdot \lrp{\gke^{\gka}\rho^{\gka}\mma i\gka \vec u^\dol{i\gka}} = 0

where :math:`\vec v^\sol{\gka}` is the velocity vector of phase :math:`\gka` and :math:`\vec u^\dol{i\gka}` is the velocity of 
species :math:`i` in phase :math:`\gka` due to molecular diffusion. The velocity of species :math:`i` in phase :math:`\gka` 
due to molecular diffusion can be approximated using Fick's First Law

.. math::
    :label:

    \gke^{\gka}\rho^{\gka}\mma i\gka \vec v^\sol{\gka} \approx -\gke^{\gka}\rho^{\gka}D^{i\gka}\del\mma i\gka

where :math:`D^{i\gka}` is the diffusion coefficient of species :math:`i` in phase :math:`\gka`. Eqn :eq:`species_in_phase` 
can now be written as 

.. math::
    :label: nonadsorptive_transport

    \pd {\lrp{\gke^{\gka}\rho^{\gka}\mma i\gka}}t + \del \vdot \lrp{\gke^{\gka}\rho^{\gka}\mma i\gka \vec v^\sol{\gka}}
    - \del \vdot \lrp{\gke^{\gka}\rho^{\gka}D^{i\gka}\del\mma i\gka} = 0

When mass exchange (adsorption) occurs between the phases, another term is added to Eqn :eq:`nonadsorptive_transport` 

.. math::
    :label: two_phase

    \pd {\lrp{\gke^{\gka}\rho^{\gka}\mma i\gka}}t + \del \vdot \lrp{\gke^{\gka}\rho^{\gka}\mma i\gka \vec v^\sol{\gka}}
    - \del \vdot \lrp{\gke^{\gka}\rho^{\gka}D^{i\gka}\del\mma i\gka} - \iema M{i\qes}{i\gka} = 0

where :math:`\iema M{i\qes}{i\gka}` is the rate of transfer of species between phases. Eqn :eq:`two_phase` is the governing 
conservation of mass equation from which all transport equations are derived.

***************
Models
***************

.. toctree::
   :maxdepth: 1

   ../ogata_banks/index
   ../bohart_thomas/index
   ../numeric_models/index

**************
Nomenclature
**************

| :math:`\vec v^\sol{i\gka}`: velocity vector of species :math:`i` in phase :math:`\gka`
| :math:`\vec v^\sol{\gka}`: velocity vector of phase :math:`\gka`
| :math:`\vec u^\dol{i\gka}`: velocity of species :math:`i` in phase :math:`\gka` due to molecular diffusion
| :math:`D^{i\gka}`: diffusion coefficient of species :math:`i` in phase :math:`\gka`
| :math:`v`: interstitial velocity of the fluid phase
| :math:`C`: species concentration in fluid phase
| :math:`D_L`: axial diffusion coefficient of the species in the fluid phase
| :math:`C_o`: species concentration in the fluid phase entering the column
| :math:`\operatorname{erfc}`: error function
| :math:`\iema M{is}{if}`: rate of transfer of species between phases
| :math:`q`: species concentration in the solid phase
| :math:`u`: superficial velocity of the fluid phase (:math:`v\gke`)
| Re: Reynolds number
| Sc: Schmidt number
| :math:`D_e`: effective diffusion coefficient of species in the particle domain
| :math:`D_p`: diffusion coefficient of species in the pore liquid
| :math:`D_s`: diffusion coefficient of species in the solid phase
| :math:`q_m`: maximum concentration of species in solid phase 
| :math:`k_a`: forward reaction rate constant in Langmuir sink
| :math:`k_d`: reverse reaction rate constant in Langmuir sink
| :math:`\ce{C}`: adsorbate species
| :math:`\ce{S}`: available sorption site
| :math:`\ce{C-S}`: product of the adsorption reaction
| :math:`q_e`: solid phase concentration at equilibrium
| :math:`C_e`: fluid phase concentration at equilibrium
| :math:`t`: time
| :math:`t_V`: time in bed volumes treated
| :math:`k_{Th}`: Thomas model rate constant
| :math:`x`: mass of solid phase in the bed
| :math:`Q`: bed volume
| :math:`b`: Langmuir dissociation constant (:math:`k_a/k_d`)
| :math:`I_0`: zero-order Bessel function of the first kind
| :math:`r`: parameter in the Thomas analytical solution
| :math:`n`: parameter in the Thomas analytical solution
| :math:`T`: parameter in the Thomas analytical solution
| :math:`J`: function in the Thomas analytical solution
| :math:`k_f`: film transfer coefficient
| :math:`A_p`: total surface area available for mass transfer 
| :math:`N_p`: number of particles in the bed
| :math:`S_p`: surface area of a spherical particle
| :math:`K`: Freundlich capacity constant
| :math:`1/n`: Freundlich intensity constant
| :math:`d_p`: particle diameter
| :math:`D`: liquid diffusivity of the species in water 
| :math:`V_b`: molar volume of the species at its boiling point
| SPDFR: surface to pore diffusion flux ratio

| *Greek Letters*
| :math:`\gke^{\gka}`: volume fraction of phase :math:`\gka`
| :math:`\rho^{\gka}`: density of phase :math:`\gka` 
| :math:`\mma i\gka`: mass fraction of species :math:`i` in phase :math:`\gka`
| :math:`\gke`: bed void fraction
| :math:`\rho_p`: apparent density of the solid phase
| :math:`\rho_b`: bed density :math:`\lrp{(1-\gke)\rho_p}`
| :math:`\gke_p`: particle porosity
| :math:`\rho`: density of water 
| :math:`\mu`: dynamic viscosity of water
| :math:`\tau_p`: particle tortuosity

************
References
************

.. bibliography::