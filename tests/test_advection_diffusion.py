import reactormodels
import numpy as np
import pytest


@pytest.mark.parametrize("diffusion", [0.01, 0.1])
def test_ogata_banks(diffusion):
    """
    Collocation solution must match Ogata-Banks analytical solution
    for 1D advection-diffusion with step inlet BC.
    """
    velocity = 1.0  # m/s
    domain_length = 5.0  # m
    C_in = 1.0

    t_eval = np.array([1.0, 2.0, 3.0])

    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=30, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusion1D(
        domain_length=domain_length,
        velocity=velocity,
        diffusion=diffusion,
        orthogonal_collocation=oc,
    )
    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval, C_in=C_in)

    for i, t in enumerate(t_eval):
        # Only compare interior of domain, away from outlet BC influence
        mask = x < 0.8 * domain_length
        C_analytical = reactormodels.models.ogata_banks(
            x[mask], t, velocity, diffusion, C_in
        )
        C_numerical = C[i, mask]

        assert C_numerical == pytest.approx(
            C_analytical, abs=1e-2
        ), f"Failed at t={t}: max error = {np.abs(C_numerical - C_analytical).max():.2e}"


def test_multi_element_ogata_banks():
    """High-Pe case that fails with single element should pass with multi-element."""

    v = 1.0
    D = 0.01
    L = 5.0
    C_in = 1.0
    t_eval = np.array([2.0, 4.0])

    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=3,
        n_elements=20,  # 20 elements × 4 pts = fine enough
        add_inlet=True,
    )
    model = reactormodels.models.AdvectionDiffusion1D(
        domain_length=L,
        velocity=v,
        diffusion=D,
        orthogonal_collocation=oc,
    )
    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval, C_in=C_in)

    for i, t in enumerate(t_eval):
        mask = x < 0.8 * L
        assert C[i, mask] == pytest.approx(
            reactormodels.models.ogata_banks(x[mask], t, v, D, C_in), abs=1e-2
        )


@pytest.mark.parametrize("diffusion", [0.01, 0.1])
def test_ogata_banks_ida(diffusion):
    """
    Collocation solution must match Ogata-Banks analytical solution
    for 1D advection-diffusion with step inlet BC.
    """
    velocity = 1.0  # m/s
    domain_length = 5.0  # m
    C_in = 1.0

    t_eval = np.array([1.0, 2.0, 3.0])

    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=30, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusion1DIDA(
        domain_length=domain_length,
        velocity=velocity,
        diffusion=diffusion,
        orthogonal_collocation=oc,
    )
    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval, C_in=C_in)

    for i, t in enumerate(t_eval):
        # Only compare interior of domain, away from outlet BC influence
        mask = x < 0.8 * domain_length
        C_analytical = reactormodels.models.ogata_banks(
            x[mask], t, velocity, diffusion, C_in
        )
        C_numerical = C[i, mask]

        assert C_numerical == pytest.approx(
            C_analytical, abs=1e-2
        ), f"Failed at t={t}: max error = {np.abs(C_numerical - C_analytical).max():.2e}"


def test_multi_element_ogata_banks_ida():
    """High-Pe case that fails with single element should pass with multi-element."""

    v = 1.0
    D = 0.01
    L = 5.0
    C_in = 1.0
    t_eval = np.array([2.0, 4.0])

    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=3,
        n_elements=20,  # 20 elements × 4 pts = fine enough
        add_inlet=True,
    )
    model = reactormodels.models.AdvectionDiffusion1DIDA(
        domain_length=L,
        velocity=v,
        diffusion=D,
        orthogonal_collocation=oc,
    )
    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval, C_in=C_in)

    for i, t in enumerate(t_eval):
        mask = x < 0.8 * L
        assert C[i, mask] == pytest.approx(
            reactormodels.models.ogata_banks(x[mask], t, v, D, C_in), abs=1e-2
        )
