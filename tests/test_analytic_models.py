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


def test_thomas_limiting_form(time=np.linspace(0, 10, 100)):
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

    time = np.linspace(1e-10, 10, 200)
    length = 1 / np.pi
    diameter = 2
    porosity = 0.4
    bulk_density = 0.36
    particle_density = 0.6
    feed_concentrations = 1
    flow_rate = 1
    sorbent_capacity = 10
    rate_constant = 10
    K = 500000

    column = reactormodels.Column(
        length=length,
        diameter=diameter,
        porosity=porosity,
        bulk_density=bulk_density,
        media=reactormodels.Media(particle_density=particle_density),
        water=reactormodels.Water(),
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=reactormodels.Chemical(),
        feed_concentrations=feed_concentrations,
        flow_rate=flow_rate,
        time=time,
    )

    bohart_adams = reactormodels.models.BohartAdams(
        breakthrough=breakthrough,
        k_BA=rate_constant,
        sorbent_capacity=sorbent_capacity,
    )

    ba_solution = bohart_adams.breakthrough_profile(time=time, x=length)

    thomas_langmuir = reactormodels.models.ThomasLangmuir(
        breakthrough=breakthrough,
        langmuir_constant=K,
        sorbent_capacity=sorbent_capacity,
        k_Th=rate_constant,
    )

    tl_solution = thomas_langmuir.breakthrough_profile(time=time, x=length)

    assert tl_solution == pytest.approx(
        ba_solution, abs=1e-3
    ), f"Failed at t={time}: max error = {np.abs(tl_solution - ba_solution).max():.3e}"


def test_bohart_adams_equals_thomas():
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
    bulk_density = 1
    porosity = 0.38
    time = np.linspace(0, 200, 200)

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        diameter=diameter,
        bulk_density=bulk_density,
        media=reactormodels.Media(),
        water=reactormodels.Water(),
    )

    feed_concentrations = [99, 101]

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=reactormodels.Chemical(),
        feed_concentrations=feed_concentrations,
        flow_rate=flow_rate,
        time=time,
    )

    k_BA = 0.002
    sorbent_capacity = 1000
    k_Th = k_BA * length / breakthrough.superficial_velocity

    bohart_adams = reactormodels.models.BohartAdams(
        breakthrough=breakthrough,
        k_BA=k_BA,
        sorbent_capacity=sorbent_capacity,
    )

    bh_solution = bohart_adams.breakthrough_profile(time=time, x=length)

    thomas_rectangular = reactormodels.models.ThomasRectangular(
        breakthrough=breakthrough,
        k_Th=k_Th,
        sorbent_capacity=sorbent_capacity,
    )

    t_solution = thomas_rectangular.breakthrough_profile(time=time, x=length)

    assert bh_solution == pytest.approx(
        t_solution, abs=1e-3
    ), f"Failed at t={time}: max error = {np.abs(bh_solution - t_solution).max():.3e}"


# ---------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------


class MockColumn:
    porosity = 0.4

    def get_particle_density(self):
        return 1000.0

    def get_bulk_density(self):
        return 600.0

    def get_sorbent_mass(self):
        return 10.0

    def column_volume(self):
        return 2.0


class MockBreakthrough:
    def __init__(self):
        self.column = MockColumn()

    def interstitial_velocity(self):
        return 1.5

    def mean_feed_concentration(self):
        return 100.0

    def time_to_bed_volumes(self):
        return np.array([0.5, 1.0, 2.0])

    def bed_volumes_to_time(self):
        return np.array([10.0, 20.0, 30.0])


@pytest.fixture
def breakthrough():
    return MockBreakthrough()


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------


def test_base_model_not_implemented(breakthrough):
    model = reactormodels.models.analytic_models.AnalyticModels(breakthrough)

    with pytest.raises(NotImplementedError):
        model.model(1, 1)


# ---------------------------------------------------------------------
# Ogata-Banks
# ---------------------------------------------------------------------


def test_ogatabanks_scalar(breakthrough):
    model = reactormodels.models.OgataBanks(breakthrough, diffusion=0.1)

    result = model.model(1.0, 5.0)

    assert np.isfinite(result)
    assert result >= 0


def test_ogatabanks_vector(breakthrough):
    model = reactormodels.models.OgataBanks(breakthrough, diffusion=0.1)

    x = np.linspace(0, 5, 20)

    result = model.model(x, 5.0)

    assert result.shape == x.shape


# ---------------------------------------------------------------------
# Yoon-Nelson
# ---------------------------------------------------------------------


def test_yoon_nelson_midpoint():
    model = reactormodels.models.YoonNelson(k_YN=1.0, t_50=10.0)

    assert np.isclose(model.model(0, 10), 0.5)


def test_yoon_nelson_warning():
    model = reactormodels.models.YoonNelson(1.0, 10.0)

    with pytest.warns(UserWarning):
        model.breakthrough_profile(np.array([1, 2]), x=5)


def test_yoon_nelson_spatial_not_implemented():
    model = reactormodels.models.YoonNelson(1.0, 10.0)

    with pytest.raises(NotImplementedError):
        model.spatial_profile(np.array([1]), 5)


# ---------------------------------------------------------------------
# Clark
# ---------------------------------------------------------------------


def test_clark_model():
    model = reactormodels.models.Clark(r=0.2, A=3.0, n=2.5)

    result = model.model(0, 5)

    assert 0 <= result <= 1


def test_clark_warning():
    model = reactormodels.models.Clark(0.2, 3.0, 2.5)

    with pytest.warns(UserWarning):
        model.breakthrough_profile(np.array([1]), x=10)


# ---------------------------------------------------------------------
# Bohart-Adams
# ---------------------------------------------------------------------


def test_bohart_adams(breakthrough):
    model = reactormodels.models.BohartAdams(
        breakthrough,
        k_BA=0.01,
        sorbent_capacity=5,
    )

    result = model.model(1.0, 10.0)

    assert np.isfinite(result)
    assert result >= 0


# ---------------------------------------------------------------------
# Thomas Rectangular
# ---------------------------------------------------------------------


def test_thomas_rectangular(breakthrough):
    model = reactormodels.models.ThomasRectangular(
        breakthrough,
        k_Th=0.01,
        sorbent_capacity=5,
    )

    result = model.model(1, 1)

    assert np.isfinite(result).all()


def test_thomas_rectangular_warning(breakthrough):
    model = reactormodels.models.ThomasRectangular(
        breakthrough,
        k_Th=0.01,
        sorbent_capacity=5,
    )

    with pytest.warns(UserWarning):
        model.breakthrough_profile(np.array([1]), x=1)


# ---------------------------------------------------------------------
# Thomas Langmuir
# ---------------------------------------------------------------------


def test_thomas_langmuir_j_function(breakthrough):
    model = reactormodels.models.ThomasLangmuir(
        breakthrough,
        langmuir_constant=0.2,
        sorbent_capacity=5,
        k_Th=0.01,
    )

    value = model._J_function(0.5, 0.5)

    assert np.isfinite(value)


def test_thomas_langmuir_model(breakthrough):
    model = reactormodels.models.ThomasLangmuir(
        breakthrough,
        langmuir_constant=0.2,
        sorbent_capacity=5,
        k_Th=0.01,
    )

    x = np.array([1.0, 2.0, 3.0])
    t = np.array([5.0, 10.0, 15.0])

    result = model.model(x, t)

    assert result.shape == x.shape
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_ogata_banks_retardation(breakthrough):
    model = reactormodels.models.OgataBanks(breakthrough, diffusion=0.01, retardation=2)

    assert np.isfinite(model.R)
