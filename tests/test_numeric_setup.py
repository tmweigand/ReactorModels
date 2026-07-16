import reactormodels
import numpy as np
import pytest
from dataclasses import dataclass


@dataclass
class BaseParams:
    column: reactormodels.Column
    breakthrough: reactormodels.Breakthrough
    column_numerics: reactormodels.numerics.NumericsConfig
    particle_numerics: reactormodels.numerics.NumericsConfig
    isotherm: reactormodels.models.isotherm.Isotherm
    axial_diffusion: float
    pore_diffusion: float
    surface_diffusion: float
    initial_concentration: float
    k_film: float


def make_base_params():

    particle_porosity = 0.5
    particle_density = 600
    particle_diameter = 0.07

    axial_diffusion = 1e-10
    pore_diffusion = 5e-6
    surface_diffusion = 5e-9
    k_film = 0.1

    K = 100
    initial_concentration = 0

    length = 100
    diameter = 10
    porosity = 0.334
    bulk_density = 399.8

    feed_concentrations = 1
    flow_rate = 40

    t_eval = np.linspace(1e-10, 175 * 1440 * 60, 200)

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        particle_porosity=particle_porosity,
        bulk_density=bulk_density,
        particle_density=particle_density,
        diameter=diameter,
        particle_diameter=particle_diameter,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=feed_concentrations,
        flow_rate=flow_rate,
        time=t_eval,
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        column=column,
        n_interior_points=3,
        n_elements=8,
        add_inlet=True,
        resolution=reactormodels.models.DomainResolution.COLUMN,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        column=column,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
        resolution=reactormodels.models.DomainResolution.PARTICLE,
    )

    isotherm = reactormodels.models.LinearIsotherm(K=K)

    return BaseParams(
        column=column,
        breakthrough=breakthrough,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        isotherm=isotherm,
        axial_diffusion=axial_diffusion,
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
        initial_concentration=initial_concentration,
        k_film=k_film,
    )


@pytest.fixture
def b():
    return make_base_params()


def make_AD(b: BaseParams):
    return reactormodels.models.AdvectionDiffusion(
        column=b.column,
        breakthrough=b.breakthrough,
        initial_concentration=b.initial_concentration,
        diffusion=b.axial_diffusion,
        numerics=b.column_numerics,
    )


def make_ADE(
    b: BaseParams,
    k_ldf=0,
    kinetics=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
):
    return reactormodels.models.AdvectionDiffusionAdsorption(
        column=b.column,
        breakthrough=b.breakthrough,
        diffusion=b.axial_diffusion,
        initial_concentration=b.initial_concentration,
        isotherm=b.isotherm,
        numerics=b.column_numerics,
        mode=kinetics,
        k_ldf=k_ldf,
    )


def make_IP(b: BaseParams):
    return reactormodels.models.IntraparticleTransport(
        column=b.column,
        breakthrough=b.breakthrough,
        pore_diffusion=b.pore_diffusion,
        surface_diffusion=b.surface_diffusion,
        initial_concentration=b.initial_concentration,
        isotherm=b.isotherm,
        numerics=b.particle_numerics,
    )


def make_DC(b: BaseParams):
    return reactormodels.models.DomainCoupling(
        column=b.column,
        breakthrough=b.breakthrough,
        axial_diffusion=b.axial_diffusion,
        pore_diffusion=b.pore_diffusion,
        surface_diffusion=b.surface_diffusion,
        initial_concentration=b.initial_concentration,
        isotherm=b.isotherm,
        column_numerics=b.column_numerics,
        particle_numerics=b.particle_numerics,
    )


def test_n_vars(b: BaseParams):
    ade = make_ADE(b)
    assert ade._n_vars() == ade.N

    ade_ldf = make_ADE(
        b,
        k_ldf=0.1,
        kinetics=reactormodels.models.AdsorptionKinetics.LINEAR_DRIVING_FORCE,
    )
    assert ade_ldf._n_vars() == 2 * ade_ldf.N

    ade_so = make_ADE(
        b,
        k_ldf=0.1,
        kinetics=reactormodels.models.AdsorptionKinetics.SECOND_ORDER,
    )
    assert ade_so._n_vars() == 2 * ade_so.N

    ip = make_IP(b)
    assert ip._n_vars() == ip.N

    dc = make_DC(b)
    assert dc._n_vars() == dc.Nz + dc.Nr * dc.Nz


def test_split(b: BaseParams):
    ade = make_ADE(b)
    y = np.arange(2 * ade.N)

    C, q = ade._split(y)

    assert np.array_equal(C, y[: ade.N])
    assert q is None

    ip = make_IP(b)
    y_ip = np.arange(2 * ip.N)

    C_ip, q_ip = ip._split(y_ip)

    assert np.array_equal(C_ip, y_ip[: ip.N])
    assert q_ip is None

    dc = make_DC(b)
    y_dc = np.arange(dc.Nz + dc.Nz * dc.Nr)

    C_dc, Cp = dc._split(y_dc)

    assert np.array_equal(C_dc, y_dc[: dc.Nz])

    expected_Cp = y_dc[dc.Nz :].reshape(dc.Nz, dc.Nr)
    assert np.array_equal(Cp, expected_Cp)

    ade_ldf = make_ADE(
        b,
        k_ldf=0.1,
        kinetics=reactormodels.models.AdsorptionKinetics.LINEAR_DRIVING_FORCE,
    )

    y_ldf = np.arange(2 * ade_ldf.N)

    C_ldf, q_ldf = ade_ldf._split(y_ldf)

    assert ade_ldf.mode == reactormodels.models.AdsorptionKinetics.LINEAR_DRIVING_FORCE
    assert np.array_equal(C_ldf, y_ldf[: ade_ldf.N])
    assert np.array_equal(q_ldf, y_ldf[ade_ldf.N :])

    ade_so = make_ADE(
        b,
        k_ldf=0.1,
        kinetics=reactormodels.models.AdsorptionKinetics.SECOND_ORDER,
    )

    y_so = np.arange(2 * ade_so.N)

    C_so, q_so = ade_so._split(y_so)

    assert ade_so.mode == reactormodels.models.AdsorptionKinetics.SECOND_ORDER
    assert np.array_equal(C_so, y_so[: ade_so.N])
    assert np.array_equal(q_so, y_so[ade_so.N :])


def test_residual_runs(b: BaseParams):
    dc = make_DC(b)
    n = dc._n_vars()

    y = np.ones(n)
    ydot = np.zeros(n)
    result = np.zeros(n)

    dc._residual(0.0, y, ydot, result)

    assert result.shape == (n,)
    assert np.all(np.isfinite(result))

    expected = dc.inlet_bc.residual(y[0])

    assert result[0] == expected
    assert np.all(np.isfinite(result))


def test_jacobian_returns_zero(b: BaseParams):
    dc = make_DC(b)
    n = dc._n_vars()

    y = np.ones(n)
    ydot = np.zeros(n)

    jac = np.zeros((n, n))

    status = dc._jacobian(
        0.0,
        y,
        ydot,
        None,
        1.0,
        jac,
    )

    assert status == 0
    assert jac.shape == (n, n)
    assert np.all(np.isfinite(jac))
