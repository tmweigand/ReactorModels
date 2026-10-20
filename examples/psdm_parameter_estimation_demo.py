"""psdm_parameter_estimation_demo.py

Fit PSDM transport parameters (surface diffusion, pore diffusion, film
transfer coefficient) to an experimental breakthrough curve using
reactormodels.regression.fit_parameters, and report 95% confidence
intervals for each fitted parameter.

Note on bounds: give every fitted parameter a realistic `max`, not just a
`min=0`. Diffusion coefficients and rate constants are only physically
meaningful across a fairly narrow band; an unbounded upper limit lets the
optimizer wander into parameter combinations the DAE solver can't
integrate at all (mxstep exceeded), and -- especially with lmfit's default
"leastsq" method -- can produce a badly-scaled, spuriously "converged" fit
with nonsensical parameter values. fit_parameters defaults to
method="least_squares" for exactly this reason (see its docstring), but
tight bounds are still the first line of defense.

Requires scikits.odes (see the docs/README for install notes); this is a
usage demo, not something that runs in a minimal CI environment.
"""

import numpy as np

import reactormodels
from reactormodels.regression import make_parameters, fit_parameters
import matplotlib.pyplot as plt


def build_psdm(
    surface_diffusion,
    pore_diffusion,
    kf,
    time,
    particle_porosity=0.5,
    particle_radius=0.07,
):
    """Construct a fresh PSDM for one trial parameter set.

    PSDM derives several internal quantities in __init__ (e.g. discretization
    sizes from column/particle_numerics), so -- like the AnalyticModels
    family -- the simplest and safest thing to do on every fit iteration is
    to rebuild the model from scratch rather than mutate an existing
    instance in place.
    """
    isotherm = reactormodels.models.LinearIsotherm(K=100)

    media = reactormodels.Media(
        particle_porosity=particle_porosity,
        particle_radius=particle_radius,
        particle_density=600,
    )
    column = reactormodels.Column(
        length=100,
        porosity=0.334,
        diameter=10,
        bulk_density=399.8,
        media=media,
        water=reactormodels.Water(),
    )
    chemical = reactormodels.Chemical(
        axial_diffusion=0,
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
    )
    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=1.0,
        flow_rate=40,
        time=time,
        chemical=chemical,
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length, n_interior_points=8, n_elements=6
    )
    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius, n_interior_points=3, n_elements=1
    )

    return reactormodels.models.PSDM(
        isotherm=isotherm,
        breakthrough=breakthrough,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=kf,
    )


def main():
    # --- stand-in for real experimental data ---
    time_min = np.linspace(60, 20000, 40)
    time_s = time_min * 60

    true_model = build_psdm(
        surface_diffusion=5e-10, pore_diffusion=5e-6, kf=0.1, time=time_s
    )
    _, _, C_true, _ = true_model.solve()
    c_over_c0_true = C_true[:, -1]

    plt.plot(c_over_c0_true)
    plt.show()

    rng = np.random.default_rng(0)
    c_obs = np.clip(c_over_c0_true + rng.normal(0, 0.02, c_over_c0_true.size), 0, 1)

    # --- fit surface_diffusion, pore_diffusion, and kf; particle
    #     geometry stays fixed at its known/measured value ---
    def model_func(p, time=time_s):
        model = build_psdm(
            surface_diffusion=p["surface_diffusion"].value,
            pore_diffusion=p["pore_diffusion"].value,
            kf=p["kf"].value,
            time=time,
        )
        _, _, C, _ = model.solve()
        return C[:, -1]  # column-outlet C/C0

    params = make_parameters(
        surface_diffusion=dict(value=1e-10, min=1e-12, max=1e-8),
        pore_diffusion=dict(value=1e-6, min=1e-8, max=1e-4),
        kf=dict(value=0.05, min=1e-3, max=5),
    )

    result = fit_parameters(model_func, params, y_data=c_obs)
    print(result.summary())
    print("RMSE:", result.rmse())

    print("\n95% confidence intervals (linear/Wald, fast):")
    for name, (lo, best, hi) in result.ci().items():
        print(f"  {name}: {best:.4g}  [{lo:.4g}, {hi:.4g}]")

    # Likelihood-profile CIs are more trustworthy for a nonlinear DAE model
    # like PSDM, at the cost of many extra solves -- run only when needed.
    print("\n95% confidence intervals (profile likelihood, slower):")
    for name, (lo, best, hi) in result.ci(method="profile").items():
        print(f"  {name}: {best:.4g}  [{lo:.4g}, {hi:.4g}]")


if __name__ == "__main__":
    main()
