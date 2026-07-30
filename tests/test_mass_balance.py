import reactormodels
import numpy as np


def test_mass_balance_class():

    bulk_density = 1.0
    K = 0.5
    porosity = 0.3
    velocity = 1.0
    column_length = 5.0
    inlet_concentration = 1.0
    initial_concentration = 0.0
    diameter = 1

    R = 1.0 + (bulk_density * K) / porosity
    v_eff = velocity / (porosity * R)
    t_eval = np.linspace(0.1 * column_length / v_eff, 1.5 * column_length / v_eff, 8)

    column = reactormodels.Column(
        length=column_length,
        porosity=porosity,
        bulk_density=bulk_density,
        diameter=diameter,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=inlet_concentration,
        superficial_velocity=velocity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column, n_interior_points=30, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorption(
        column=column,
        breakthrough=breakthrough,
        initial_concentration=initial_concentration,
        diffusion=0.1,
        isotherm=reactormodels.models.LinearIsotherm(K=K),
        numerics=numerics,
        mode=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
    )

    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    balances = reactormodels.postprocess.MassBalance.from_solution(
        x=x, breakthrough=breakthrough, C_history=C, q_history=q, t_eval=t_eval
    )

    for n, mb in enumerate(balances):
        assert mb.is_balanced(rel_tol=0.05), mb.summary()
