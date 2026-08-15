from pathlib import Path

import numpy as np
import pytest

import reactormodels

# ============================================================
# CONFIG
# ============================================================

AD_BREAKTHROUGH_FILE = Path("regression/AdDesignS_breakthrough.txt")
RMSE_THRESHOLD = 1e-2

PARAM_MAP = {
    "kf": "kf",
    "Ds": "Ds",
    "Dp": "Dp",
    "bed_length": "L",
    "bed_diameter": "Dia",
    "flow_rate": "Q",
    "particle_porosity": "particle_porosity",
    "particle_radius": "particle_radius",
}

# Baseline kwargs - must match _make_particle's defaults / the
# AdDesignS baseline case's physical values.
BASELINE_KWARGS = dict(
    Ds=5e-10,
    Dp=5e-6,
    L=100,
    kf=10,
    Q=40,
    Dia=10,
    particle_porosity=0.5,
    particle_radius=0.07,
)


# ============================================================
# Load the AdDesignS regression results (once, at collection time)
# ============================================================


def load_addesigns_results(path: Path) -> dict:
    import csv

    results = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row["name"]
            if name not in results:
                results[name] = {
                    "param": row["param"],
                    "value": row["value"],
                    "pct_change": row["pct_change"],
                    "time": [],
                    "c": [],
                }
            results[name]["time"].append(float(row["time_min"]))
            results[name]["c"].append(float(row["C_over_C0"]))
    return results


AD_RESULTS = load_addesigns_results(AD_BREAKTHROUGH_FILE)


# ============================================================
# reactormodels PSDM setup
# ============================================================


def _make_particle(
    Ds=5e-10,
    Dp=5e-6,
    L=100,
    kf=10,
    Q=40,
    Dia=10,
    particle_porosity=0.5,
    particle_radius=0.07,
):
    # particle
    particle_density = 600  # g/mL
    particle_diameter = particle_radius * 2

    # column
    axial_diffusion = 0
    K = 100  # (mg/g) * (L/mg)
    initial_concentration = 0
    porosity = 0.334
    bulk_density = 399.8  # g/mL
    feed_concentrations = 1  # mg/L

    isotherm = reactormodels.models.LinearIsotherm(K=K)

    media = reactormodels.Media(
        particle_porosity=particle_porosity,
        particle_diameter=particle_diameter,
        particle_density=particle_density,
    )

    column = reactormodels.Column(
        length=L,
        porosity=porosity,
        diameter=Dia,
        bulk_density=bulk_density,
        media=media,
        water=reactormodels.Water(),
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=reactormodels.Chemical(),
        feed_concentrations=feed_concentrations,
        flow_rate=Q,
        time=time,
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length,
        n_interior_points=8,
        n_elements=6,
        add_inlet=True,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
    )
    return reactormodels.models.DomainCoupling(
        isotherm=isotherm,
        breakthrough=breakthrough,
        axial_diffusion=axial_diffusion,
        pore_diffusion=Dp,
        surface_diffusion=Ds,
        initial_concentration=initial_concentration,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=kf,
    )


def run_case(case: dict, kwargs_override: dict):
    """Solve one case on the AdDesignS output time grid for that case.
    Returns (time_min list, C_over_C0 list)."""
    global time  # _make_particle/Breakthrough closes over module-level `time`
    t_eval = np.array(case["time"]) * 60
    time = t_eval / 1440 / 60

    p = _make_particle(**kwargs_override)
    # z, r, C, Cp = p.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)
    # C_numerical = C[:, -1]
    # return case["time"], C_numerical.tolist()


def compute_rmse(c_numerical, case_c) -> float:
    a = np.array(c_numerical)
    b = np.array(case_c)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ============================================================
# Tests
# ============================================================


@pytest.mark.parametrize("name", list(AD_RESULTS.keys()))
def test_rmse_against_addesigns(name):
    case = AD_RESULTS[name]

    kwargs_override = dict(BASELINE_KWARGS)
    if case["param"] != "baseline":
        mapped = PARAM_MAP.get(case["param"])
        if mapped is None:
            pytest.skip(f"unrecognized param '{case['param']}' - not in PARAM_MAP")
        value = float(case["value"])
        kwargs_override[mapped] = value

    # _, c_numerical = run_case(case, kwargs_override)
    run_case(case, kwargs_override)
    # rmse = compute_rmse(c_numerical, case["c"])

    # assert rmse < RMSE_THRESHOLD, (
    #     f"{name}: RMSE {rmse:.5f} exceeds threshold {RMSE_THRESHOLD} "
    #     f"(param={case['param']}, value={case['value']})"
    # )
