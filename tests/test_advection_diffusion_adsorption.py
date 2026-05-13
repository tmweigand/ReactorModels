import reactormodels
import numpy as np
import pytest


@pytest.mark.parametrize("K", [0.5, 2.0])
@pytest.mark.parametrize("diffusion", [0.01, 0.1])
def test_ogata_banks_linear_adsorption(diffusion, K):
    """
    With a linear isotherm, the advection-diffusion-adsorption equation
    reduces to the same form as pure advection-diffusion but with an
    effective (retarded) velocity:

        v_eff = v / (eps * R),  R = 1 + rho_b * K / eps

    So Ogata-Banks still applies as the analytical benchmark.
    """
    velocity = 1.0
    domain_length = 5.0
    porosity = 0.4
    bulk_density = 500.0  # kg/m^3
    C_in = 1.0

    isotherm = reactormodels.models.LinearIsotherm(K=K)
    R = 1.0 + (bulk_density * K) / porosity  # retardation factor
    v_eff = velocity / (porosity * R)  # retarded velocity
    D_eff = diffusion / R  # retarded dispersion

    t_eval = np.array([10.0, 100.0, 500.0, 1000.0])

    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=30, add_inlet=True
    )
    model = reactormodels.models.AdvectionDiffusionAdsorption1D(
        domain_length=domain_length,
        velocity=velocity,
        diffusion=diffusion,
        isotherm=isotherm,
        bulk_density=bulk_density,
        porosity=porosity,
        orthogonal_collocation=oc,
    )

    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval, C_in=C_in)

    for i, t in enumerate(t_eval):
        mask = x < 0.8 * domain_length
        C_analytical = reactormodels.models.ogata_banks(x[mask], t, v_eff, D_eff, C_in)
        C_numerical = C[i, mask]

        assert C_numerical == pytest.approx(C_analytical, abs=1e-1), (
            f"Failed at t={t:.1f}s (K={K}, D={diffusion}): "
            f"max error = {np.abs(C_numerical - C_analytical).max():.2e}"
        )


# @pytest.mark.parametrize("n_freundlich", [1.5, 2.0])
# def test_freundlich_adsorption_mass_balance(n_freundlich):
#     """
#     No analytical solution exists for Freundlich adsorption, so instead
#     verify global mass balance: mass in = mass stored (fluid) + mass adsorbed.
#     """
#     from model.isotherm import FreundlichIsotherm

#     velocity = 1.0
#     domain_length = 5.0
#     porosity = 0.4
#     bulk_density = 500.0
#     K_f = 1.0
#     C_in = 1.0
#     t_final = 2.0
#     t_eval = np.linspace(0.1, t_final, 20)

#     isotherm = FreundlichIsotherm(K=K_f, n=n_freundlich)
#     oc = OrthogonalCollocation(n_interior_points=30, add_inlet=True)
#     model = AdvectionDiffusionAdsorption1D(
#         domain_length=domain_length,
#         velocity=velocity,
#         diffusion=0.05,
#         isotherm=isotherm,
#         bulk_density=bulk_density,
#         porosity=porosity,
#         orthogonal_collocation=oc,
#     )

#     x, C = model.solve(t_span=(0, t_final), t_eval=t_eval, C_in=C_in)

#     for i, t in enumerate(t_eval):
#         # Mass in via advection: v * C_in * t (per unit area)
#         mass_in = velocity * C_in * t

#         # Mass in fluid phase: eps * integral(C dx)
#         mass_fluid = porosity * np.trapz(C[i], x)

#         # Mass adsorbed: rho_b * integral(q*(C) dx)
#         q = isotherm.q(C[i])
#         mass_adsorbed = bulk_density * np.trapz(q, x)

#         total_stored = mass_fluid + mass_adsorbed

#         # Allow 5% relative error (numerical dispersion + trapz error)
#         assert total_stored == pytest.approx(mass_in, rel=0.05), (
#             f"Mass balance failed at t={t:.2f}s: "
#             f"in={mass_in:.3f}, stored={total_stored:.3f}"
#         )
