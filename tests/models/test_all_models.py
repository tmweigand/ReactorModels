"""Finite-difference Jacobian checks for all Jacobian-enabled models."""

import inspect

import numpy as np
import pytest
import reactormodels


def _state_size(model):
    if hasattr(model, "_n_vars"):
        return model._n_vars()
    return model.N


def _build_advection_diffusion():
    breakthrough = reactormodels.fixtures.make_breakthrough(
        length=5.0,
        diameter=1.0,
        porosity=0.5,
        superficial_velocity=0.5,
        axial_diffusion=0.1,
        time=[0.0, 1.0],
    )
    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=breakthrough.column.length,
        n_interior_points=6,
        n_elements=1,
        add_inlet=True,
    )
    return reactormodels.models.AdvectionDiffusion(
        breakthrough=breakthrough,
        numerics=numerics,
    )


def _build_advection_diffusion_adsorption(
    mode=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
    k_ldf=0.0,
):
    breakthrough = reactormodels.fixtures.make_breakthrough(
        length=5.0,
        diameter=1.0,
        porosity=0.5,
        bulk_density=500.0,
        superficial_velocity=0.5,
        axial_diffusion=0.1,
        time=[0.0, 1.0],
    )
    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=breakthrough.column.length,
        n_interior_points=6,
        n_elements=1,
        add_inlet=True,
    )
    return reactormodels.models.AdvectionDiffusionAdsorption(
        breakthrough=breakthrough,
        isotherm=reactormodels.models.LinearIsotherm(K=0.5),
        numerics=numerics,
        mode=mode,
        k_ldf=k_ldf,
    )


def _build_intraparticle_transport():
    isotherm = reactormodels.models.LinearIsotherm(K=100.0)

    media = reactormodels.Media(
        particle_porosity=0.5,
        particle_diameter=0.07,
        particle_density=600.0,
    )
    column = reactormodels.Column(
        length=100.0,
        porosity=0.334,
        diameter=10.0,
        bulk_density=399.8,
        media=media,
        water=reactormodels.Water(),
    )

    chemical = reactormodels.Chemical(
        pore_diffusion=5e-6,
        surface_diffusion=5e-9,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=1.0,
        flow_rate=40.0,
        time=np.linspace(1e-10, 10.0, 5),
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
    )
    return reactormodels.models.IntraparticleTransport(
        isotherm=isotherm,
        breakthrough=breakthrough,
        numerics=numerics,
    )


def _build_psdm():
    isotherm = reactormodels.models.LinearIsotherm(K=100.0)

    media = reactormodels.Media(
        particle_porosity=0.5,
        particle_diameter=0.07,
        particle_density=600.0,
    )
    column = reactormodels.Column(
        length=100.0,
        porosity=0.334,
        diameter=10.0,
        bulk_density=399.8,
        media=media,
        water=reactormodels.Water(),
    )

    chemical = reactormodels.Chemical(
        axial_diffusion=0.0,
        pore_diffusion=5e-6,
        surface_diffusion=5e-9,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=1.0,
        flow_rate=40.0,
        time=np.linspace(1e-10, 10.0, 5),
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length,
        n_interior_points=3,
        n_elements=4,
        add_inlet=True,
    )
    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
    )
    return reactormodels.models.PSDM(
        isotherm=isotherm,
        breakthrough=breakthrough,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=0.1,
    )


MODEL_BUILDERS = {
    "AdvectionDiffusion": _build_advection_diffusion,
    "AdvectionDiffusionAdsorption": _build_advection_diffusion_adsorption,
    "IntraparticleTransport": _build_intraparticle_transport,
    "PSDM": _build_psdm,
}


def _jacobian_model_names_from_public_api():
    names = []
    for name in reactormodels.models.__all__:
        obj = getattr(reactormodels.models, name)
        if (
            inspect.isclass(obj)
            and hasattr(obj, "_jacobian")
            and hasattr(obj, "_residual")
        ):
            names.append(name)
    return set(names)


def test_all_jacobian_models_are_covered():
    discovered = _jacobian_model_names_from_public_api()
    configured = set(MODEL_BUILDERS)
    assert configured == discovered


def _assert_jacobian_matches_finite_difference(model):
    n = _state_size(model)

    rng = np.random.default_rng(7)
    y = rng.random(n) * 0.5 + 0.1
    ydot = rng.random(n)
    cj = 7.3
    t0 = 0.0

    r0 = np.zeros(n)
    model._residual(t0, y, ydot, r0)

    J_fd = np.zeros((n, n))
    eps = 1e-7
    for j in range(n):
        y_pert = y.copy()
        y_pert[j] += eps
        ydot_pert = ydot.copy()
        ydot_pert[j] += cj * eps
        r1 = np.zeros(n)
        model._residual(t0, y_pert, ydot_pert, r1)
        J_fd[:, j] = (r1 - r0) / eps

    J_analytic = np.zeros((n, n))
    model._jacobian(t0, y, ydot, None, cj, J_analytic)

    np.testing.assert_allclose(J_analytic, J_fd, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("model_name", sorted(MODEL_BUILDERS))
def test_jacobian_matches_finite_difference(model_name):
    model = MODEL_BUILDERS[model_name]()
    _assert_jacobian_matches_finite_difference(model)


@pytest.mark.parametrize(
    "mode,k_ldf",
    [
        (reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM, 0.0),
        (reactormodels.models.AdsorptionKinetics.LINEAR_DRIVING_FORCE, 0.5),
    ],
)
def test_adsorption_jacobian_matches_finite_difference_across_modes(mode, k_ldf):
    model = _build_advection_diffusion_adsorption(mode=mode, k_ldf=k_ldf)
    _assert_jacobian_matches_finite_difference(model)
