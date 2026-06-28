Model Development
========================

***************************
Conservation of Mass
***************************

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
species :math:`i` in phase :math:`\gka` due to molecular diffusion.

The velocity of species :math:`i` in phase :math:`\gka` due to molecular diffusion can be approximated using Fick's First Law

.. math::
    :label:

    \gke^{\gka}\rho^{\gka}\mma i\gka \vec v^\sol{\gka} \approx -\gke^{\gka}\rho^{\gka}D^{i\gka}\del\mma i\gka

where :math:`D^{i\gka}` is the diffusion coefficient of species :math:`i` in phase :math:`\gka`. 

Eqn :eq:`species_in_phase` can now be written as 

.. math::
    :label: nonadsorptive_transport

    \pd {\lrp{\gke^{f}\rho^{f}\mma if}}t + \del \vdot \lrp{\gke^{f}\rho^{f}\mma if \vec v^\sol{f}}
    - \del \vdot \lrp{\gke^{f}\rho^{f}D^{if}\del\mma if} = 0

where :math:`f` denotes the fluid phase.

Considering a cylindrical column where concentration only varies in the axial direction, Eqn :eq:`nonadsorptive_transport` 
can be simplified to

.. math::
    :label:

    \pd Ct + v \pd Cz - D_L \pdn 2Cz = 0


where :math:`v` is the interstitial velocity of the fluid phase, :math:`C` is the species concentration in fluid phase, :math:`D_L` is the
axial diffusion coefficient of the species in the fluid phase, and :math:`\gke` is the bed void fraction. This is known as the
advection-diffusion equation from which the Ogata-Banks analytical solution for 1-D transport was derived :cite:`OgataBanks1961`

.. math::
    :label: ogata_banks

    \frac C{C_o} = \frac 12 \lrb{\operatorname{erfc}\lrp{\frac {z - vt}{2\sqrt{D_Lt}}} +
    \exp\lrp{\frac {vz}{D_L}}\operatorname{erfc}\lrp{\frac {z + vt}{2\sqrt{D_Lt}}}}

where :math:`C_o` is the species concentration in the fluid phase entering the column and :math:`\operatorname{erfc}` is the 
error function.

When interphase  mass exchange (adsorption) occurs between the fluid and solid phases, another term is added to 
Eqn :eq:`nonadsorptive_transport` for the transfer of species from the fluid phase to the solid phase :math:`s`

.. math::
    :label: two_phase

    \pd {\lrp{\gke^{f}\rho^{f}\mma if}}t + \del \vdot \lrp{\gke^{f}\rho^{f}\mma if \vec v^\sol{f}}
    - \del \vdot \lrp{\gke^{f}\rho^{f}D^{if}\del\mma if} - \iema M{is}{if} = 0

where :math:`\iema M{is}{if}` is the rate of transfer of species between phases.

Assuming species is present in either the fluid phase or solid phase, the interphase mass transfer term can be 
written as the accumulation of species in the solid phase

.. math::
    :label:

    -\iema M{is}{if} = \pd {\lrp{\gke^s\rho^s\mma is}}t

which can be substituted into Eqn :eq:`two_phase`

.. math::
    :label: vector_mass_balance

    \pd {\lrp{\gke^{f}\rho^{f}\mma if}}t + \del \vdot \lrp{\gke^{f}\rho^{f}\mma if \vec v^\sol{f}}
    - \del \vdot \lrp{\gke^{f}\rho^{f}D^{if}\del\mma if} + \pd {\lrp{\gke^s\rho^s\mma is}}t = 0

Again, considering a cylindrical column where concentration only varies in the axial direction, Eqn :eq:`vector_mass_balance` 
simplifies to

.. math::
    :label: adsorptive_mass_balance

    \gke \pd Ct + \gke v \pd Cz - \gke D_L \pdn 2Cz + \lrp{1 - \gke} \rho_p \pd qt= 0

where :math:`q` is the species concentration in the solid phase and :math:`\rho_p` is the apparent density of the solid phase.

When operating in the Darcy Flow Regime (Re :math:`<` 10), axial diffusion is considered to be 
negligible if the product of the Reynolds and Schmidt number (Re x Sc) is greater than 200. 
Thus, Eqn :eq:`adsorptive_mass_balance` may be written as

.. math::
    :label: no_dispersion_mass_balance

    \gke \pd Ct + \gke v \pd Cz + \lrp{1 - \gke} \rho_p \pd qt= 0

Eqn :eq:`no_dispersion_mass_balance` serves as the governing conservation of mass equation in the derivation of the Bohart-Adams
:cite:`BohartAdams1920`, Thomas :cite:`Thomas1944`, and pore and/or surface diffusion :cite:`Friedman1984` models. In the derivation 
of Bohart-Adams and Thomas model analytical solutions, time derivative term is ignored due to the slow time-scale of adsorption relative 
to fluid flow. Thus, Eqn :eq:`no_dispersion_mass_balance` simplifies to the following equation after dividing through by :math:`\gke`

.. math::
    :label: no_disperion_or_time_derivative

    v \pd Cz = - \frac {\lrp{1-\gke}}{\gke}\rho_p\pd qt

Eqn :eq:`no_disperion_or_time_derivative` can be simplified further to

.. math::
    :label: BA_mass_balance

    u \pd Cz = - \rho_b\pd qt

where the superficial velocity :math:`u` is equal to :math:`v\gke`, and the bed density :math:`\rho_b` is 
equal to :math:`(1-\gke)\rho_p`.

The pore and/or surface diffusion models couple the conservation of mass equations between the bulk liquid 
domain and the particle domain where species may be present in the solid phase as well as the pore liquid. 
Thus, the interphase mass exchange term in Eqn :eq:`vector_mass_balance` can no longer be expressed in 
terms of :math:`q` alone. Assuming axial diffusion to still be negligible, the last term in Eqn :eq:`no_dispersion_mass_balance` 
is replaced by the rate of mass exchange between the bulk fluid and particle domains

.. math::
    :label: overall_mass_balance

    \gke \pd Ct + \gke v \pd Cz - \iema M{ip}{if} = 0

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

where :math:`\gke_p` is the particle porosity. Allowing accumulation of species in the solid phase and pore liquid, 
Eqn :eq:`particle_mass_balance` can be written as

.. math::
    :label: effective_diffusion_balance

    \rho_p\pd qt + \gke_p\pd {C_p}t - D_e \del \vdot \lrb{\del \lrp{\rho_pq + \gke_pC_p}} - \iema M{f}{p} = 0

where :math:`\rho_p` is the apparent particle density and :math:`D_e` is the effective diffusion coefficient 
of species in the particle phase. Assuming the species concentration only changes in the radial direction, 
Eqn :eq:`effective_diffusion_balance` may be written in spherical coordinates as

.. math::
    :label: effective_diffusion_model

    \rho_p\pd qt + \gke_p\pd {C_p}t - \frac 1{r^2} \pd {}r \lrb{r^2D_e\lrp{\rho_p\pd {q}r + \gke_p\pd{C_p}r}} 
    - \iema M{f}{p} = 0

For the pore diffusion model, intraparticle diffusion is assumed to occur primarily in the 
pore liquid. Thus, Eqn :eq:`effective_diffusion_model` becomes

.. math::
    :label: pore_diffusion_model

    \rho_p\pd qt + \gke_p\pd {C_p}t - \frac 1{r^2} \pd {}r \lrb{r^2D_p\gke_p\pd{C_p}r} 
    - \iema M{f}{p} = 0

where :math:`D_p` is the diffusion coefficient of species in the pore liquid. Alternatively, intraparticle diffusion 
may be assumed to occur primarily in the solid phase. Thus, Eqn :eq:`effective_diffusion_model` becomes

.. math::
    :label: homogenous_surface_diffusion

    \rho_p\pd qt + \gke_p\pd {C_p}t - \frac 1{r^2} \pd {}r \lrb{r^2D_s\rho_p\pd {q}r} 
    - \iema M{f}{p} = 0

where :math:`D_s` is the diffusion coefficient of species in the solid phase. Finally, intraparticle diffusion 
may be assumed to occur in both the pore liquid and solid phase. Thus, Eqn :eq:`effective_diffusion_model` becomes

.. math::
    :label: pore_and_surface_diffusion

    \rho_p\pd qt + \gke_p\pd {C_p}t - \frac 1{r^2} \pd {}r \lrb{r^2\lrp{D_s\rho_p\pd {q}r + D_p\gke_p\pd{C_p}r}} 
    - \iema M{f}{p} = 0

which is solved simultaneously with Eqn :eq:`overall_mass_balance`.

***********************
Kinetic Expressions
***********************

The Bohart-Adams and Thomas models assume second order reaction kinetics where the partial time derivative of
sorbent phase concentration may be expressed as a function of :math:`C` and :math:`q` 

.. math::
    :label:

    \pd qt = f(C,q)

The Langmuir sink kinetic relation is written as

.. math::
    :label: langmuir_sink

    \pd qt = k_a C \lrp{q_m - q} - k_dq
    
where :math:`q_m` is the maximum mass that may be adsorbed, and :math:`k_a` is the forward reaction rate constant,
and :math:`k_d` is the reverse reaction rate constant for the following sorption equation

.. math::
    :label:

    \ce{C + S <=>[$k_a$][$k_d$] C\text{-}S}

where :math:`\ce{C}` is the adsorbate species, :math:`\ce{S}` is an available sorption site, and :math:`\ce{C-S}` 
is the product of the adsorption reaction. The system is at equilibrium when the partial derivative of :math:`q` with 
respect to time is equal to 0. Adding subscript :math:`e` to :math:`C` and :math:`q` to indicate equilibrium concentrations, 
the Langmuir adsorption isotherm equation is obtained from Eqn :eq:`langmuir_sink`

.. math::
    :label: langmuir_kinetics

    0 = k_a C_e \lrp{q_m - q_e} - k_dq_e

Eqn :eq:`langmuir_kinetics` can be rearranged to

.. math::
    :label: langmuir_isotherm

    q_e = \frac {k_aq_mC_e}{k_d + k_aC_e}

The Bohart-Adams model assumes a rectangular adsorption isotherm. The rectangular isotherm equation 
is obtained from Eqn :eq:`langmuir_isotherm` by taking the limit of :math:`q_e` as :math:`k_d \to 0` 
(i.e. the forward reaction occurs until the sorbent capacity is reached)

.. math::
    :label:

    q_e = q_m

Eqn :eq:`langmuir_sink` for a rectangular adsorption isotherm simplifies to

.. math::
    :label: BA_kinetics1

    \pd qt = k C \lrp{q_m-q}

Substituting Eqn :eq:`BA_kinetics1` into Eqn :eq:`BA_mass_balance` 

.. math::
    :label: BA_kinetics2

    \pd Cz = -\rho_b \frac ku C\lrp{q_m-q}

Through a series of manipulations and applications of the boundary condition :math:`C_o = C(0,t)` 
and the initial condition :math:`q(z,0) = 0`. Eqn :eq:`BA_kinetics1` and Eqn :eq:`BA_kinetics2` 
are used to derive the Bohart-Adams analytical solution

.. math::
    :label: BA_model

    \frac C{C_o} = \frac 1{1 + \exp\lrp{\frac {\rho_bkq_mL}u -kC_ot}}

or the mathematically equivalent Thomas model

.. math::
    :label:

    \frac C{C_o} = \frac 1{1 + \exp\lrp{\frac {k_{Th}q_ex}Q -k_{Th}C_ot_V}}

where :math:`t_V` is time in bed volumes treated, :math:`k_{Th}` is the Thomas model rate constant,
:math:`x` is the mass of solid phase in the bed, and :math:`Q` is the bed volume.

The Thomas model relaxes the rectangular isotherm assumption by assuming a Langmuir sink kinetic relation.
The Langmuir dissociation constant :math:`b` is defined as :math:`k_a/k_d`, allowing Eqn :eq:`langmuir_isotherm` 
to be simplified to

.. math::
    :label: 

    q_e = \frac {q_mbC_e}{1 + bC_e}

Eqn :eq:`langmuir_sink` can now be written as

.. math::
    :label: thomas_kinetics

    \pd qt = k \lrb{C \lrp{q_m - q} - \frac qb}

Through application of the same initial and boundary conditions as in the Bohart-Adams model derivation, 
the Thomas model analytical solution is obtained

.. math::
    :label:

    \frac C {C_o} = \frac {J((n/r), nT)}{J((n/r), nT) + \lrb{1 - J(n, (nT/r))}\exp\lrb{(1 - (1/r))(n - nT)}}

with the following variables

.. math::
    r = 1 + bC_o; \quad n = \frac {\rho_pq_mkL\lrp{1 - \gke}}{\gke v}; \quad
    T = \frac {\gke\lrp{(1/b) + C_o}}{\rho_pq_m(1-\gke)} \lrp{\frac {vt}L - 1}

and function :math:`J` given by

.. math::
    J(x, y) = 1 - \ilims 0x \exp(-y - \tau)I_0(2\sqrt{y\tau})\operatorname{d}\tau   

where :math:`I_0` is the zero-order Bessel function of the first kind.

In the pore and/or surface diffusion models, a kinetic expression for transfer of species from 
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

    - \iema M{f}{p} = \frac {3k_f(1-\gke)}{R}\lrb{C(z,t)-C_p(r=R,z,t)}

gives the expression for the rate of transfer of species between the bulk fluid and particle domain. Eqn 
:eq:`film_transfer_final` is substituted into Eqn :eq:`overall_mass_balance` and Eqn :eq:`effective_diffusion_model` 
to obtain the final conservation of mass equations in the bulk fluid domain

.. math::
    :label:

    \gke \pd Ct + \gke v \pd Cz + \lrp{1 - \gke} \rho_p \pd qt + 
    \frac {3k_f(1-\gke)}{R}\lrb{C(z,t)-C_p(r=R,z,t)} = 0

and particle domain

.. math::
    :label:

    \rho_p\pd qt + \gke_p\pd {C_p}t - \frac 1{r^2} \pd {}r \lrb{r^2D_e\lrp{\rho_p\pd {q}r + \gke_p\pd{C_p}r}} 
    
    + \frac {3k_f(1-\gke)}{R}\lrb{C(z,t)-C_p(r=R,z,t)} = 0

Under the assumption of local equilibrium, :math:`q` and :math:`C_p` are related by the Freundlich 
isotherm equation for any given time and position

.. math::
    :label: freundlich_isotherm

    q = KC_p^{1/n} 

where :math:`K` is the capacity constant and  :math:`1/n` is the intensity constant. If the species is 
present at trace concentrations, a linear isotherm is often assumed (:math:`n=1`). Thus, Eqn :eq:`freundlich_isotherm` 
simplifies to 

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

******************
Vector notation
******************

.. math::
    \del f = \lrp{\pd fx, \pd fy, \pd fz}

.. math::
    \del \vdot \vec f = \pd fx + \pd fy + \pd fz

*****************
Appendix
*****************

Derivation of :eq:`BA_model` from :eq:`BA_kinetics1` & :eq:`BA_kinetics2`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applying the chain rule, these expressions may be written as

.. math::
    \pd {}t\ln\lrp{q_m-q} = - k C, \quad \pd {}z\ln\lrp{C} = -K\lrp{q_m-q}

where :math:`K = m_sk/u`

Proof

.. math::
    \pd {}t\lrp{q_m-q} = -\pd qt

.. math::
    \pd {}t\lrp{q_m-q} = - k C \lrp{q_m-q}

.. math::
    \pd {}t\ln\lrp{q_m-q} = -\frac 1{q_m-q} k C \lrp{q_m-q}

.. math::
    \pd {}t\ln\lrp{q_m-q} = -kC

Applying the boundary condition :math:`C_o = C(0,t)` and the initial condition :math:`q(z,0) = 0`

.. math::
    \pd {}t\ln\lrp{q_m-q(0,t)} = - k C_o, \quad \pd {}z\ln\lrp{C(z,0)} = -Kq_m

Integrating the general expressions to satisfy the boundary and initial condition

.. math::
    \pd {}t\ln\lrp{q_m-q} = - k C, \quad \pd {}z\ln\lrp{C} = -K\lrp{q_m-q}

.. math::
    \ln\lrp{q_m-q} = - kCt + A, \quad \ln\lrp{C} = -K\lrp{q_m-q}z + B

.. math::
    \ln\lrp{q_m-q(z,0)} = A, \quad \ln\lrp{C(0,t)} = B

.. math::
    A = \ln\lrp{q_m}, \quad B = \ln\lrp{C_o}

.. math::
    q(0,t) = q_m\lrp{1 - \exp\lrp{- k t C_o}}, \quad C(z,0) = C_o\exp\lrp{-Kq_mz}

Taking derivatives in space and time to define the initial :math:`C` throughout the column and :math:`q` at the inlet

.. math::
    \frac {\partial^2}{\partial t \partial z}\ln\lrp{q_m-q} = - k \pd Cz,\quad\frac {\partial^2}{\partial z \partial t}\ln\lrp{C} = K \pd qt

.. math::
    -\frac u{m_s}\pd Cz = \pd qt

multiplying by :math:`K`

.. math::
    -k\pd Cz = K\pd qt

Subtracting the expressions becomes

.. math::
    \frac {\partial^2}{\partial t \partial z}\ln\lrp{q_m-q} - \frac {\partial^2}{\partial z \partial t}\ln\lrp{C} = 0

.. math::
    \frac {\partial^2}{\partial t \partial z}\ln\lrp{\frac C{q_m-q}} = 0

which has the solution

.. math::
    \ln\lrp{\frac C{q_m-q}} = f(z) + g(t)

Applying boundary conditions

.. math::
    \ln\lrp{\frac {C_o}{q_m\exp\lrp{-kC_ot}}} = \ln\lrp{C_o} - \ln\lrp{q_m} + kC_ot = f(0) + g(t)

and initial conditions

.. math::
    \ln\lrp{\frac{C_o\exp\lrp{-Kq_mz}}{q_m}} = \ln\lrp{C_o} - Kq_mz - \ln\lrp{q_m} = f(z) + g(0)

Adding the two expressions and combining constants

.. math::
    kC_ot - Kq_mz = f(z) + g(t) + A

Substituting

.. math::
    \ln\lrp{\frac C{q_m-q}} = kC_ot - Kq_mz - A

.. math::
    \frac C{q_m-q} = Bexp\lrp{kC_ot - Kq_mz}

Applying initial or boundary conditions

.. math::
    \frac {C_o\exp\lrp{-Kq_mz}}{q_m} = B\exp\lrp{- Kq_mz}

.. math::
    B = \frac {C_o}{q_m}

Thus

.. math::
    C = \frac {C_o}{q_m}\lrp{q_m-q}\exp\lrp{kC_ot - Kq_mz}

Substituting into the kinetic expression

.. math::
    \pd qt = \frac {kC_o}{q_m}\lrp{q_m-q}^2\exp\lrp{kC_ot - Kq_mz}

.. math::
    \frac {\mathrm{d}q}{\lrp{q_m-q}^2} = \frac {kC_o}{q_m}\exp\lrp{kC_ot - Kq_mz}\mathrm{d}t

Integrating and applying the initial condition

.. math::
    \frac q{q_m} = 1 - \frac 1{1-\exp\lrp{-Kq_mz}+\exp\lrp{kC_ot - Kq_mz}}

.. math::
    \frac {q_m-q}{q_m} = \frac 1{1-\exp\lrp{-Kq_mz}+\exp\lrp{kC_ot - Kq_mz}}

Rearranging

.. math::
    C = \frac {C_o}{q_m}\lrp{q_m-q}\exp\lrp{kC_ot - Kq_mz}

.. math::
    \frac C{C_o} = \lrp{\frac {q_m-q}{q_m}}\exp\lrp{kC_ot - Kq_mz}

Substituting

.. math::
    \frac C{C_o} = \frac {\exp\lrp{kC_ot - Kq_mz}}{1-\exp\lrp{-Kq_mz}+\exp\lrp{kC_ot - Kq_mz}}

.. math::
    \frac C{C_o} = \frac {\exp\lrp{kC_ot - Kq_mz}}{\exp\lrp{kC_ot - Kq_mz} \lrb{\exp\lrp{Kq_mz-kC_ot}-\exp\lrp{-kC_ot}+1}}

.. math::
    \frac C{C_o} = \frac 1{\exp\lrp{Kq_mz-kC_ot}-\exp\lrp{-kC_ot}+1}

Setting :math:`z` equal to column length :math:`L` and assuming :math:`t` and :math:`L` are sufficiently large 
that the second term in the denominator is negligible, the Bohart-Adams solution is obtained

.. math::
    \frac C{C_o} = \frac 1{1 + \exp\lrp{\frac {m_skq_mL}u -kC_ot}}

The Thomas model equation expressing time in terms of bed volumes treated is

.. math::
    \frac C{C_o} = \frac 1{1 + \exp\lrp{\frac {k_{Th}q_ex}Q -k_{Th}C_ot_V}}

************
References
************

.. bibliography::