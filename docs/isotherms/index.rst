Isotherms
============

In addition to the single-species isotherms which the Bohart-Adams and Thomas
Models are built upon, multi-species isotherms may be applied to models which 
are solved numerically. Multi-species isotherms are derived using Ideal Adsorbed 
Solution Theory (IAST). Take Gibbs' equation for change in surface tension at an 
adsorption interface to the change in liquid phase concentration of the adsorbate

.. math::
    :label:

    -Ad\gks_i = Ad\pi_i = q_i RT d\lrp{\ln C_i}

where :math:`A` is the adsorption area per mass of adsorbent, :math:`d\gks_i` is the change in surface 
tension due to adsorption, :math:`d\pi_i` is the change in spreading pressure due to adsorption, and 
:math:`d\lrp{\ln C_i}` is the change in the natural log of liquid phase concentration. Spreading pressure is 
the difference between the surface tension, :math:`\gks_i` between the solid interface with pure solvent and 
the solid interface with the solution containing adsorbate (sorptive solution)

.. math::
    :label:

    d\pi_i = -d\gks_i = \gks_{\text{Pure Water/Adsorbent}} - \gks_{\text{Sorptive Solution/Adsorbent}}

This equation is convenient for constructing a multi-species isotherm because the difference in surface tension 
between two sorptive solutions may be quantified without knowing the surface tension between pure water and the 
adsorbent, a generally challenging property to measure. Assuming single-species concentrations, :math:`C^o_i` 
are in equilibrium with the spreading pressure of the mixture and the single-species concentration, :math:`C_i` 
in equilibrium with the mixture can be determined from an expression analogous to Raoult's law. IAST is 
comprised of five equations

.. math::
    :label:

    q_T = \sLb {i=1}Nq_i

.. math::
    :label:

    x_i = \frac {q_i}{q_T}

.. math::
    :label:

    C_i = x_iC^o_i

.. math::
    :label:

    \frac 1{q_T} = \sLb {i=1}N \frac {x_i}{q^o_i}

.. math::
    :label: spreading_pressure

    \frac {\pi_mA}{RT} = \frac {\pi_iA}{RT} = \ilims 0{q^o_i}\frac {d\ln C^o_i}{d \ln q^o_i}d q^o_i

where :math:`q_T` is the total surface loading, :math:`q_i` is the single-species solid phase loading, 
:math:`x_i` is the mole fraction of species :math:`i` on the adsorbent surface, :math:`C_i` concentration 
of species :math:`i` in the multi-species system, and :math:`C_i^o` is the concentration of species 
:math:`i` in the single-species system. The Freundlich isotherm equation for a single-species is given by

.. math::
    :label:

    q_i^o = K_i\lrp{C_i^o}^{1/n_i}

taking the natural log of both sides

.. math::
    :label:

    \ln q_i^o = \ln K_i + \frac 1{n_i} \ln C_i^o

and

.. math::
    :label:

    \frac {d\ln q^o_i}{d \ln C^o_i} = \frac 1{n_i}

thus Eqn :eq:`spreading_pressure` may be written as 

.. math::
    :label:

    \frac {\pi_iA}{RT} = \ilims 0{q^o_i}n_id q^o_i

evaluating the integral

.. math::
    :label:

    \frac {\pi_iA}{RT} = n_iq^o_i

since 

.. math::
    :label:

    \frac {\pi_iA}{RT} = \frac {\pi_mA}{RT}

the following expression is also true

.. math::
    :label:

    n_iq_i = n_jq_j, \quad j = 2, 3, \dots, N

    