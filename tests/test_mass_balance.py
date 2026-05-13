import reactormodels
import numpy as np
import pytest


def test_mass_balance_class():
    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=30, add_inlet=True
    )
    rho_b = 1.0
    K = 0.5
    eps = 0.4
    v = 1.0
    L = 5.0
    C_in = 1.0

    model = reactormodels.models.AdvectionDiffusionAdsorption1D_two(
        column_length=L,
        velocity=v,
        dispersion=0.5,
        isotherm=reactormodels.models.LinearIsotherm(K=K),
        bulk_density=rho_b,
        porosity=eps,
        oc=oc,
        mode=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
    )

    R = 1.0 + (rho_b * K) / eps
    v_eff = v / (eps * R)
    t_eval = np.linspace(0.1 * L / v_eff, 1.5 * L / v_eff, 8)

    t_eval = [0.1, 0.3]

    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval, C_in=C_in)

    balances = reactormodels.postprocess.MassBalance.from_solution(
        x=x,
        C_history=C,
        q_history=q,
        t_eval=t_eval,
        velocity=v,
        porosity=eps,
        bulk_density=rho_b,
        C_in=C_in,
    )

    for n, mb in enumerate(balances):
        print(t_eval[n])
        assert mb.is_balanced(rel_tol=0.05), mb.summary()
