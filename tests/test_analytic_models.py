import reactormodels
import numpy as np
import pytest


def test_clark_becomes_yoon_nelson(time=np.linspace(0, 200, 200)):
    """Clark becomes Yoon-Nelson when n = 2

    C/Co = 1 / [1 + A*exp(-r*t)]^(1 / (n - 1))

    -> C/Co = 1 / [1 + A*exp(-r*t)]

    -> C/Co = 1 / [1 + exp(B)*exp(-r*t)]

    -> C/Co = 1 / [1 + exp(B - r*t)]

    -> C/Co = 1 / [1 + exp(r*(B/r - t))]

    A = exp(B) -> B = ln(A)

    -> C/Co = 1 / [1 + exp(r*(ln(A)/r - t))]

    Yoon-Nelson:
    C/Co = 1 / (1 + exp[k_YN*(t_50 - t)])

    when k_YN = r and t_50 = ln(A)/r
    """
    r = 0.05
    A = 500
    n = 2
    k_YN = r
    t_50 = np.log(A) / r

    yoon_nelson = reactormodels.models.YoonNelson(
        t_50=t_50,
        k_YN=k_YN,
    )
    yn_solution = yoon_nelson.breakthrough_profile(time=time)

    clark = reactormodels.models.Clark(
        r=r,
        A=A,
        n=n,
    )

    clark_solution = clark.breakthrough_profile(time=time)

    assert yn_solution == pytest.approx(
        clark_solution, abs=1e-3
    ), f"Failed at t={time}: max error = {np.abs(yoon_nelson - clark).max():.3e}"


@pytest.mark.skip
def test_bohart_adams_equals_thomas(time=np.linspace(0, 200, 200)):
    """The rectangular Thomas model is equivalent to the Bohart-Adams model through unit conversion.

    Bohart-Adams:
    C/Co = 1 / [1 + exp(m_o*k_BA*q_m*L/u - k_BA*Co*t)]

    Co: inlet concentration in mg/mL
    q_m: sorbent capacity in mg/g

    m_o: sorbent loading in g/mL
    k_BA: rate constant in (mg/mL)^-1 / s
    L: bed length in cm
    u: superficial velocity in cm/s
    t: time in s

    Rectangular Thomas:
    C/Co = 1 / [1 + exp(k_Th*q_m*x/Q - k_Th*Co*BV)]sorbent loading

    k_Th: rate constant in mL/(mg-BV)
    x: sorbent mass in g
    Q: flow rate in mL/BV

    m_o*k_BA*L/u = k_Th*x/Q
    k_Th*BV = k_BA*t
    """
    diameter = 0.5
    length = 1
    flow_rate = 0.02

    column = reactormodels.Column(length=length, diameter=diameter, flow_rate=flow_rate)

    sorbent_loading = 1
    k_BA = 0.002
    sorbent_capacity = 1000
    x = length
    velocity = column.superficial_velocity()
    inlet_concentration = 100
    bed_volume = column.column_volume()
    sorbent_mass = sorbent_loading * bed_volume
    k_Th = k_BA * x / velocity
    EBCT = bed_volume / flow_rate  # use breakthrough class when added
    bed_volumes_treated = time / EBCT

    bohart_adams = reactormodels.models.BohartAdams(
        sorbent_loading=sorbent_loading,
        k_BA=k_BA,
        sorbent_capacity=sorbent_capacity,
        velocity=velocity,
        inlet_concentration=inlet_concentration,
    )

    bh_solution = bohart_adams.breakthrough_profile(time=time, x=x)

    thomas_rectangular = reactormodels.models.ThomasRectangular(
        sorbent_mass=sorbent_mass,
        k_Th=k_Th,
        sorbent_capacity=sorbent_capacity,
        bed_volume=bed_volume,
        bed_volumes_treated=bed_volumes_treated,
        inlet_concentration=inlet_concentration,
    )

    t_solution = thomas_rectangular.breakthrough_profile(time=time, x=x)

    assert bh_solution == pytest.approx(
        t_solution, abs=1e-3
    ), f"Failed at t={time}: max error = {np.abs(bohart_adams - thomas_rectangular).max():.3e}"


def thomas_limiting_form(time=np.linspace(0, 200, 200)):
    """Thomas Model with Langmuir isotherm becomes rectangular as reverse rate constant approaches zero.

    q = q_m*K*C / (1 + K*C)

    q: sorbed mass
    q_m: sorbent capacity
    C: liquid concentration
    K: Langmuir constant

    K = k_a / k_d

    k_a: forward rate constant
    K_d: reverse rate constant

    q = q_m*(k_a / k_d)*C / (1 + (k_a / k_d)*C)

    -> q = q_m*k_a*C / (k_d + k_a*C)

    k_d -> 0

    q -> q_m*k_a*C / (k_a*C) -> q_m
    """

    diameter = 0.5
    length = 1
    flow_rate = 0.02

    column = reactormodels.Column(length=length, diameter=diameter, flow_rate=flow_rate)

    apparent_density = 1
    sorbent_capacity = 1000
    x = length
    velocity = column.superficial_velocity()
    inlet_concentration = 100
    bed_volume = column.column_volume()
    sorbent_mass = apparent_density * bed_volume
    k_Th = 0.02
    EBCT = bed_volume / flow_rate  # use breakthrough class when added
    bed_volumes_treated = time / EBCT
    langmuir_constant = 1e10
    bed_void_fraction = 0.38
    interstitial_velocity = velocity / bed_void_fraction

    thomas_rectangular = reactormodels.models.ThomasRectangular(
        sorbent_mass=sorbent_mass,
        k_Th=k_Th,
        sorbent_capacity=sorbent_capacity,
        bed_volume=bed_volume,
        bed_volumes_treated=bed_volumes_treated,
        inlet_concentration=inlet_concentration,
    ).concentration_profile()

    thomas_langmuir = np.array(
        [
            reactormodels.models.ThomasLangmuir(
                langmuir_constant=langmuir_constant,
                apparent_density=apparent_density,
                inlet_concentration=inlet_concentration,
                sorbent_capacity=sorbent_capacity,
                k_Th=k_Th,
                x=x,
                bed_void_fraction=bed_void_fraction,
                interstitial_velocity=interstitial_velocity,
                time=t,
            ).concentration_profile()
            for t in time
        ]
    )

    assert thomas_langmuir == pytest.approx(
        thomas_rectangular, abs=1e-3
    ), f"Failed at t={time}: max error = {np.abs(thomas_langmuir - thomas_rectangular).max():.3e}"
