"""test_model_factory.py

The factory receives a breakthrough template and deep-copies it for each
trial. Fake model classes keep tests focused on routing and validation without
running numerical solvers.
"""

import numpy as np
import pytest

from reactormodels.parameter_estimation import fit_parameters, make_parameters
from reactormodels.models import model_factory as mf
from reactormodels.models.model_factory import make_breakthrough_model_func

TIME = np.linspace(0.0, 10.0, 5)


class FakeMedia:
    def __init__(self, porosity=0.4, particle_radius=None, particle_diameter=None):
        self.porosity = porosity
        self.particle_radius = particle_radius
        self.particle_diameter = particle_diameter


class FakeWater:
    def __init__(self, density=1000.0, temperature=298.0):
        self.density = density
        self.temperature = temperature


class FakeColumn:
    def __init__(self, media, water, length=1.0):
        self.media, self.water, self.length = media, water, length


class FakeChemical:
    def __init__(self, density=1.0, diffusivity=1e-9):  # density also belongs to water
        self.density, self.diffusivity = density, diffusivity


class FakeBreakthrough:
    def __init__(self, column, chemical, time, inlet=1.0, superficial_velocity=1.0):
        self.column, self.chemical, self.time, self.inlet = (
            column,
            chemical,
            time,
            inlet,
        )
        self.superficial_velocity = superficial_velocity


class FakeNumerics:
    def __init__(self, domain_length, n_nodes=10):
        self.domain_length, self.n_nodes = domain_length, n_nodes


class FakeIsotherm:
    def __init__(self, K=1.0):
        self.K = K


class FakeModel:
    """One NumericsConfig argument, no isotherm."""

    def __init__(self, breakthrough, numerics: FakeNumerics, dispersion=0.0):
        self.breakthrough, self.numerics, self.dispersion = (
            breakthrough,
            numerics,
            dispersion,
        )

    def solve(self):
        return self


class FakeDualModel:
    """Two NumericsConfig arguments plus an isotherm (PSDM-like)."""

    def __init__(
        self,
        breakthrough,
        column_numerics: FakeNumerics,
        particle_numerics: FakeNumerics,
        isotherm,
        mu=1.0,
    ):
        self.breakthrough = breakthrough
        self.column_numerics = column_numerics
        self.particle_numerics = particle_numerics
        self.isotherm = isotherm
        self.mu = mu

    def solve(self):
        return self


class NoNumericsModel:
    def __init__(self, breakthrough):
        self.breakthrough = breakthrough


def make_breakthrough(
    *,
    porosity=0.4,
    temperature=298.0,
    length=1.0,
    density=1.0,
    diffusivity=1e-9,
    particle_radius=None,
    inlet=1.0,
    superficial_velocity=1.0,
    time=TIME,
):
    media = FakeMedia(porosity=porosity, particle_radius=particle_radius)
    water = FakeWater(temperature=temperature)
    column = FakeColumn(media=media, water=water, length=length)
    chemical = FakeChemical(density=density, diffusivity=diffusivity)
    return FakeBreakthrough(
        column,
        chemical,
        time,
        inlet=inlet,
        superficial_velocity=superficial_velocity,
    )


@pytest.fixture(autouse=True)
def fake_numerics(monkeypatch):
    monkeypatch.setattr(mf, "NumericsConfig", FakeNumerics)


def build(model_cls, params, breakthrough=None, **kwargs):
    """Return a model_func that yields the constructed fake model."""
    kwargs.setdefault("extract_outlet", lambda model: model)
    return make_breakthrough_model_func(
        model_cls, breakthrough or make_breakthrough(), params, **kwargs
    )


# --------------------------------------------------------------------------
# 1. Routing parameters to template attributes and constructors
# --------------------------------------------------------------------------


def test_parameters_are_routed_to_breakthrough_attributes_and_components():
    breakthrough = make_breakthrough()
    params = make_parameters(
        porosity=dict(value=0.3),
        temperature=dict(value=280.0),
        length=dict(value=2.5),
        diffusivity=dict(value=2e-9),
        n_nodes=dict(value=25),
        dispersion=dict(value=0.7),
    )
    model = build(FakeModel, params, breakthrough)(params)
    bt = model.breakthrough

    assert bt.column.media.porosity == 0.3
    assert bt.column.water.temperature == 280.0
    assert bt.column.length == 2.5
    assert bt.chemical.diffusivity == 2e-9
    assert model.numerics.n_nodes == 25
    assert model.dispersion == 0.7
    np.testing.assert_array_equal(bt.time, TIME)


def test_breakthrough_attribute_can_be_fit():
    params = make_parameters(inlet=dict(value=0.5))
    model = build(FakeModel, params)(params)
    assert model.breakthrough.inlet == 0.5


def test_default_outlet_extractor_is_used_for_known_models(monkeypatch):
    monkeypatch.setitem(mf._OUTLET_EXTRACTORS, FakeModel, lambda model: "outlet")
    params = make_parameters(dispersion=dict(value=0.1))
    func = make_breakthrough_model_func(FakeModel, make_breakthrough(), params)
    assert func(params) == "outlet"


# --------------------------------------------------------------------------
# 2. Fixed constructor kwargs and call independence
# --------------------------------------------------------------------------


def test_fixed_kwargs_are_applied():
    params = make_parameters(length=dict(value=1.5))
    model = build(
        FakeModel,
        params,
        model_kwargs={"dispersion": 0.2},
        numerics_kwargs={"n_nodes": 50},
    )(params)

    assert model.breakthrough.column.length == 1.5
    assert model.dispersion == 0.2
    assert model.numerics.n_nodes == 50


def test_repeated_calls_are_independent_and_do_not_mutate_template():
    breakthrough = make_breakthrough(particle_radius=1e-3)
    params = make_parameters(particle_radius=dict(value=1e-3))
    func = build(FakeModel, params, breakthrough)

    first = func(params)
    params["particle_radius"].value = 2e-3
    second = func(params)

    assert first.breakthrough.column.media.particle_radius == 1e-3
    assert first.breakthrough.column.media.particle_diameter == 2e-3
    assert second.breakthrough.column.media.particle_radius == 2e-3
    assert second.breakthrough.column.media.particle_diameter == 4e-3
    assert breakthrough.column.media.particle_radius == 1e-3
    assert breakthrough.column.media.particle_diameter is None


def test_fit_value_overrides_template_value():
    breakthrough = make_breakthrough(length=4.0)
    params = make_parameters(length=dict(value=2.5))
    model = build(FakeModel, params, breakthrough)(params)

    assert model.breakthrough.column.length == 2.5
    assert breakthrough.column.length == 4.0


def test_constructor_fit_parameter_also_fixed_raises():
    params = make_parameters(dispersion=dict(value=0.3))
    with pytest.raises(ValueError, match="pick one"):
        build(FakeModel, params, model_kwargs={"dispersion": 0.35})


# --------------------------------------------------------------------------
# 3. Ambiguity, overrides and names controlled by the factory
# --------------------------------------------------------------------------


def test_unknown_parameter_raises():
    params = make_parameters(not_a_real_arg=dict(value=1.0))
    with pytest.raises(ValueError, match="not an attribute or argument"):
        build(FakeModel, params)


def test_ambiguous_parameter_raises():
    params = make_parameters(density=dict(value=900.0))  # Water and Chemical
    with pytest.raises(ValueError, match="ambiguous"):
        build(FakeModel, params)


def test_param_override_resolves_ambiguity():
    params = make_parameters(density=dict(value=900.0))
    model = build(
        FakeModel,
        params,
        param_overrides={"density": "water.density"},
    )(params)
    assert model.breakthrough.column.water.density == 900.0
    assert model.breakthrough.chemical.density == 1.0


def test_param_override_can_rename_the_parameter():
    params = make_parameters(rho=dict(value=3.0))
    model = build(
        FakeModel,
        params,
        param_overrides={"rho": "chemical.density"},
    )(params)
    assert model.breakthrough.chemical.density == 3.0


def test_param_override_unknown_component_raises():
    params = make_parameters(density=dict(value=1.0))
    with pytest.raises(ValueError, match="Unknown component"):
        build(FakeModel, params, param_overrides={"density": "bogus.density"})


def test_param_override_unknown_attribute_raises():
    params = make_parameters(density=dict(value=1.0))
    with pytest.raises(ValueError, match="has no 'nope' attribute or argument"):
        build(FakeModel, params, param_overrides={"density": "chemical.nope"})


@pytest.mark.parametrize("reserved", ["time", "media", "breakthrough", "domain_length"])
def test_factory_managed_names_cannot_be_fit_parameters(reserved):
    params = make_parameters(**{reserved: dict(value=1.0)})
    with pytest.raises(ValueError, match="not an attribute or argument"):
        build(FakeModel, params)


# --------------------------------------------------------------------------
# 4. NumericsConfig handling
# --------------------------------------------------------------------------


def test_single_numerics_domain_length_comes_from_column():
    params = make_parameters(length=dict(value=2.0))
    model = build(FakeModel, params)(params)
    assert model.numerics.domain_length == 2.0


def test_dual_numerics_domain_lengths_are_inferred():
    breakthrough = make_breakthrough(particle_radius=5e-4)
    params = make_parameters(length=dict(value=3.0), K=dict(value=2.0))
    model = build(
        FakeDualModel,
        params,
        breakthrough,
        isotherm_cls=FakeIsotherm,
    )(params)

    assert model.column_numerics.domain_length == 3.0
    assert model.particle_numerics.domain_length == 5e-4
    assert model.isotherm.K == 2.0


def test_particle_numerics_requires_particle_radius():
    params = make_parameters(length=dict(value=3.0))
    func = build(FakeDualModel, params, isotherm_cls=FakeIsotherm)
    with pytest.raises(ValueError, match="particle_radius"):
        func(params)


def test_dual_numerics_kwargs_are_keyed_by_argument_name():
    breakthrough = make_breakthrough(particle_radius=1e-3)
    params = make_parameters(length=dict(value=1.0))
    model = build(
        FakeDualModel,
        params,
        breakthrough,
        isotherm_cls=FakeIsotherm,
        numerics_kwargs={"column_numerics": {"n_nodes": 40}},
    )(params)

    assert model.column_numerics.n_nodes == 40
    assert model.particle_numerics.n_nodes == 10


def test_dual_numerics_unknown_key_raises():
    params = make_parameters(length=dict(value=1.0))
    with pytest.raises(ValueError, match="keyed by"):
        build(
            FakeDualModel,
            params,
            isotherm_cls=FakeIsotherm,
            numerics_kwargs={"bogus": {}},
        )


def test_shared_numerics_parameter_is_ambiguous_until_overridden():
    params = make_parameters(n_nodes=dict(value=33))
    breakthrough = make_breakthrough(particle_radius=1e-3)
    kwargs = {"isotherm_cls": FakeIsotherm}

    with pytest.raises(ValueError, match="ambiguous"):
        build(FakeDualModel, params, breakthrough, **kwargs)

    model = build(
        FakeDualModel,
        params,
        breakthrough,
        param_overrides={"n_nodes": "particle_numerics.n_nodes"},
        **kwargs,
    )(params)
    assert model.particle_numerics.n_nodes == 33
    assert model.column_numerics.n_nodes == 10


def test_model_without_numerics_argument_raises():
    params = make_parameters(length=dict(value=1.0))
    with pytest.raises(ValueError, match="no NumericsConfig"):
        build(NoNumericsModel, params)


# --------------------------------------------------------------------------
# 5. Isotherm and outlet-extractor handling
# --------------------------------------------------------------------------


def test_isotherm_required_when_model_takes_one():
    params = make_parameters(mu=dict(value=1.0))
    with pytest.raises(ValueError, match="requires isotherm_cls"):
        build(FakeDualModel, params)


def test_isotherm_rejected_when_model_takes_none():
    params = make_parameters(dispersion=dict(value=1.0))
    with pytest.raises(ValueError, match="does not take an isotherm"):
        build(FakeModel, params, isotherm_cls=FakeIsotherm)


def test_isotherm_fixed_kwargs_are_applied():
    params = make_parameters(mu=dict(value=1.0))
    model = build(
        FakeDualModel,
        params,
        breakthrough=make_breakthrough(particle_radius=1e-3),
        isotherm_cls=FakeIsotherm,
        isotherm_kwargs={"K": 7.0},
    )(params)
    assert model.isotherm.K == 7.0
    assert model.mu == 1.0


def test_missing_outlet_extractor_raises():
    params = make_parameters(dispersion=dict(value=0.1))
    with pytest.raises(ValueError, match="No default outlet extractor"):
        make_breakthrough_model_func(FakeModel, make_breakthrough(), params)


# --------------------------------------------------------------------------
# 6. End-to-end with fit_parameters
# --------------------------------------------------------------------------


def test_model_func_works_with_fit_parameters():
    """The factory output is a drop-in model_func for fit_parameters."""
    true_dispersion = 0.35
    time = np.linspace(0, 10, 40)
    rng = np.random.default_rng(0)
    y_obs = np.exp(-true_dispersion * time) + rng.normal(0, 0.005, time.size)

    params = make_parameters(dispersion=dict(value=0.1, min=0.0, max=5.0))
    breakthrough = make_breakthrough(time=time)
    model_func = make_breakthrough_model_func(
        FakeModel,
        breakthrough,
        params,
        extract_outlet=lambda model: np.exp(
            -model.dispersion * model.breakthrough.time
        ),
    )
    result = fit_parameters(model_func, params, y_data=y_obs)

    assert result.params["dispersion"].value == pytest.approx(true_dispersion, rel=0.05)
