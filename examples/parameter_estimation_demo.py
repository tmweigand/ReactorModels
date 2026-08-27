"""parameter_estimation.py"""

import numpy as np

import reactormodels
from reactormodels.regression import (
    make_parameters,
    fit_parameters,
    fit_parameters_multistart,
)
import matplotlib.pyplot as plt


def build_model(axial_diffusion, superficial_velocity, time):
    """Construct a adfevtion diffusion modeel"""
    media = reactormodels.Media()
    column = reactormodels.Column(
        length=5,
        porosity=0.334,
        diameter=10,
        bulk_density=399.8,
        media=media,
        water=reactormodels.Water(),
    )
    chemical = reactormodels.Chemical(
        axial_diffusion=axial_diffusion,
    )
    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=1.0,
        superficial_velocity=superficial_velocity,
        time=time,
        chemical=chemical,
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length, n_interior_points=8, n_elements=6
    )

    return reactormodels.models.AdvectionDiffusion(
        breakthrough=breakthrough,
        numerics=column_numerics,
    )


def plot_prediction_distribution(
    result,
    model_func,
    time,
    true_solution,
    n_samples=500,
    seed=0,
    observed_data=None,
):
    """Plot prediction uncertainty from the fitted parameter covariance."""

    rng = np.random.default_rng(seed)

    fitted_names = [name for name, parameter in result.params.items() if parameter.vary]

    if not fitted_names:
        raise ValueError("No fitted parameters found.")

    best_values = np.array(
        [result.params[name].value for name in fitted_names],
        dtype=float,
    )

    covariance = result.result.covar

    if covariance is None:
        raise ValueError("The fit does not contain a covariance matrix.")

    # ---------------------------------------------------------------
    # Generate parameter samples.
    #
    # Reject samples outside the parameter bounds rather than
    # clipping them. This preserves the covariance structure better.
    # ---------------------------------------------------------------

    parameter_samples = []

    while len(parameter_samples) < n_samples:

        sample = rng.multivariate_normal(
            mean=best_values,
            cov=covariance,
        )

        valid = True

        for name, value in zip(fitted_names, sample):

            parameter = result.params[name]

            if parameter.min is not None and value < parameter.min:
                valid = False
                break

            if parameter.max is not None and value > parameter.max:
                valid = False
                break

        if valid:
            parameter_samples.append(sample)

    parameter_samples = np.asarray(parameter_samples)

    # ---------------------------------------------------------------
    # Run model for every parameter sample.
    # ---------------------------------------------------------------

    predictions = np.empty(
        (n_samples, len(time)),
        dtype=float,
    )

    for i, sample in enumerate(parameter_samples):

        sampled_params = result.params.copy()

        for name, value in zip(fitted_names, sample):
            sampled_params[name].set(value=value)

        predictions[i] = model_func(sampled_params)

    # ---------------------------------------------------------------
    # Prediction statistics.
    # ---------------------------------------------------------------

    prediction_mean = np.mean(
        predictions,
        axis=0,
    )

    prediction_lower = np.percentile(
        predictions,
        2.5,
        axis=0,
    )

    prediction_upper = np.percentile(
        predictions,
        97.5,
        axis=0,
    )

    # ---------------------------------------------------------------
    # Plot.
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    # Individual prediction curves.
    ax.plot(
        time,
        predictions.T,
        color="gray",
        alpha=0.05,
        linewidth=0.8,
    )

    # 95% prediction interval.
    ax.fill_between(
        time,
        prediction_lower,
        prediction_upper,
        color="gray",
        alpha=0.30,
        label="95% prediction interval",
    )

    # Mean fitted prediction.
    ax.plot(
        time,
        prediction_mean,
        linewidth=2,
        label="Fitted prediction",
    )

    # True solution.
    ax.plot(
        time,
        true_solution,
        linestyle="--",
        linewidth=2,
        label="True solution",
    )

    # Observed data.
    if observed_data is not None:
        ax.plot(
            time,
            observed_data,
            ".",
            label="Observed data",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$C/C_0$")
    ax.set_title("Prediction Distribution")

    ax.set_ylim(-0.05, 1.05)

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    plt.savefig("data_out/fitting_demo.png")


def main():
    # --- stand-in for real experimental data ---
    time_min = np.linspace(0.5, 2.5, 40)

    true_model = build_model(
        axial_diffusion=0.1, superficial_velocity=1.0, time=time_min
    )
    x, C_true = true_model.solve()
    c_over_c0_true = C_true[:, -1]

    rng = np.random.default_rng(0)
    c_obs = np.clip(c_over_c0_true + rng.normal(0, 0.1, c_over_c0_true.size), 0, 1)

    # --- fit surface_diffusion, pore_diffusion, and kf; particle
    #     geometry stays fixed at its known/measured value ---
    def model_func(p, time=time_min):
        model = build_model(
            axial_diffusion=p["axial_diffusion"].value,
            superficial_velocity=p["superficial_velocity"].value,
            time=time,
        )
        _, C = model.solve()
        return C[:, -1]  # column-outlet C/C0

    params = make_parameters(
        axial_diffusion=dict(value=0.25, min=0.001, max=10),
        superficial_velocity=dict(value=0.9, min=0.5, max=6),
    )

    best, all_runs = fit_parameters_multistart(
        model_func,
        params,
        y_data=c_obs,
        n_starts=15,
        log_uniform_names=("axial_diffusion",),  # spans 7 orders of magnitude
        x_scale="jac",  # also fixes the scaling mismatch
    )

    print("chisqr across starts:", sorted(round(r.result.chisqr, 6) for r in all_runs))
    result = best  # feed into the rest of your script as before

    # result = fit_parameters(model_func, params, y_data=c_obs)
    print(result.summary())
    print("RMSE:", result.rmse())

    print("\n95% confidence intervals (linear/Wald, fast):")
    for name, (lo, best, hi) in result.ci().items():
        print(f"  {name}: {best:.4g}  [{lo:.4g}, {hi:.4g}]")

    # Likelihood-profile CIs are more trustworthy for a nonlinear DAE model
    # like PSDM, at the cost of many extra solves -- run only when needed.
    print("\n95% confidence intervals (profile likelihood, slower):")
    for name, (lo, best, hi) in result.ci(method="profile", level=0.99).items():
        print(f"  {name}: {best:.4g}  [{lo:.4g}, {hi:.4g}]")

    plot_prediction_distribution(
        result=result,
        model_func=model_func,
        time=time_min,
        true_solution=c_over_c0_true,
        n_samples=500,
        observed_data=c_obs,
    )


if __name__ == "__main__":
    main()
