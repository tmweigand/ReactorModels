Bohart-Adams and Thomas
===========================

Models including adsorption assume species :math:`i` to be present in either the fluid phase or solid phase. 
Thus, Eqn :eq:`two_phase` is written in terms of fluid phase :math:`f` and solid phase :math:`s`

.. math::
    :label: BA_conservation_eqn

    \pd {\lrp{\gke^{f}\rho^{f}\mma if}}t + \del \vdot \lrp{\gke^{f}\rho^{f}\mma if \vec v^\sol{f}}
    - \del \vdot \lrp{\gke^{f}\rho^{f}D^{if}\del\mma if} -\iema M{is}{if} = 0

Without resolving the particle domain, the interphase mass transfer term may be written as the accumulation 
of species in the solid phase

.. math::
    :label:

    -\iema M{is}{if} = \pd {\lrp{\gke^s\rho^s\mma is}}t

which can be substituted into Eqn :eq:`BA_conservation_eqn`

.. math::
    :label: vector_mass_balance

    \pd {\lrp{\gke^{f}\rho^{f}\mma if}}t + \del \vdot \lrp{\gke^{f}\rho^{f}\mma if \vec v^\sol{f}}
    - \del \vdot \lrp{\gke^{f}\rho^{f}D^{if}\del\mma if} + \pd {\lrp{\gke^s\rho^s\mma is}}t = 0

Considering a cylindrical column where concentration only varies in the axial direction, Eqn :eq:`vector_mass_balance` 
simplifies to

.. math::
    :label: adsorptive_mass_balance

    \gke \pd Ct + \gke v \pd Cz - \gke D_L \pdn 2Cz + \lrp{1 - \gke} \rho_p \pd qt= 0

where :math:`q` is the species concentration in the solid phase, and :math:`\rho_p` is the apparent density of the 
solid phase. When operating in the Darcy Flow Regime (Re :math:`<` 10), axial diffusion is considered to be 
negligible if the product of the Reynolds and Schmidt number (Re x Sc) is greater than 200. Thus, Eqn 
:eq:`adsorptive_mass_balance` may be written as

.. math::
    :label: no_dispersion_mass_balance

    \gke \pd Ct + \gke v \pd Cz + \lrp{1 - \gke} \rho_p \pd qt= 0

Eqn :eq:`no_dispersion_mass_balance` serves as the governing conservation of mass equation for the Bohart-Adams
:cite:`BohartAdams1920` and Thomas :cite:`Thomas1944` models. In order to obtain the Bohart-Adams and Thomas model 
analytical solutions, the time derivative term is ignored due to the slow time-scale of adsorption relative 
to fluid flow. Thus, Eqn :eq:`no_dispersion_mass_balance` simplifies to the following equation after dividing 
through by :math:`\gke`

.. math::
    :label: no_disperion_or_time_derivative

    v \pd Cz = - \frac {\lrp{1-\gke}}{\gke}\rho_p\pd qt

Eqn :eq:`no_disperion_or_time_derivative` can be simplified further to

.. math::
    :label: BA_mass_balance

    u \pd Cz = - \rho_b\pd qt

where the superficial velocity :math:`u` is equal to :math:`v\gke`, and the bulk density :math:`\rho_b` is 
equal to :math:`(1-\gke)\rho_p`. Second order reaction kinetics are assumed where the partial time derivative 
of sorbent phase concentration may be expressed as a function of :math:`C` and :math:`q` 

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

    \frac C{C_o} = \frac {\exp\lrp{kC_o(t - L/v)}}{\exp\lrp{kC_o(t - L/v)} + \exp\lrp{(krho_pq_mL  / v)((1-\gke) / \gke)} - 1}

where :math:`t` is time and :math:`L` is the bed length. Though mathematically equivalent :cite:`Chu2010`, the so-called Thomas model is obtained by converting 
units of parameters in Eqn :eq:`BA_model`

.. math::
    :label:

    \frac C{C_o} = \frac {\exp\lrp{k_ThC_o(BVT - \gke)}}{\exp\lrp{k_ThC_o(BVT - \gke)} + \exp\lrp{k_Thq_ex/Q} - 1}

where :math:`BVT` is bed volumes treated, :math:`k_{Th}` is the Thomas model rate constant,
:math:`x` is the mass of solid phase in the bed, and :math:`Q` is the bed volume. 

The Thomas model relaxes the rectangular isotherm assumption by assuming a Langmuir sink kinetic relation. 
The Langmuir dissociation constant :math:`b` is defined as :math:`k_a/k_d`, allowing 
Eqn :eq:`langmuir_isotherm` to be simplified to

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