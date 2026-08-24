"""test_addesigns.py"""

from pathlib import Path

import csv
import numpy as np
import pytest


import reactormodels

AD_BREAKTHROUGH_FILE = Path("tests/regression/AdDesignS_breakthrough.txt")
RMSE_THRESHOLD = 1e-2

PARAM_MAP = {
    "kf": "kf",
    "surface_diffusion": "surface_diffusion",
    "pore_diffusion": "pore_diffusion",
    "bed_length": "L",
    "bed_diameter": "Dia",
    "flow_rate": "Q",
    "particle_porosity": "particle_porosity",
    "particle_radius": "particle_radius",
}


BASELINE_KWARGS = dict(
    surface_diffusion=5e-10,
    pore_diffusion=5e-6,
    L=100,
    kf=10,
    Q=40,
    diameter=10,
    particle_porosity=0.5,
    particle_radius=0.07,
)


def load_addesigns_results(path: Path) -> dict:
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


def make_particle(
    surface_diffusion=5e-10,
    pore_diffusion=5e-6,
    L=100,
    kf=10,
    Q=40,
    diameter=10,
    particle_porosity=0.5,
    particle_radius=0.07,
    time=None,
):

    isotherm = reactormodels.models.LinearIsotherm(
        K=100,  # (mg/g) * (L/mg)
    )

    media = reactormodels.Media(
        particle_porosity=particle_porosity,
        particle_radius=particle_radius,
        particle_density=600,  # g/mL
    )

    column = reactormodels.Column(
        length=L,
        porosity=0.334,
        diameter=diameter,
        bulk_density=399.8,  # g/mL
        media=media,
        water=reactormodels.Water(),
    )

    chemical = reactormodels.Chemical(
        axial_diffusion=0.0,
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=1,  # mg/L
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
    return reactormodels.models.PSDM(
        isotherm=isotherm,
        breakthrough=breakthrough,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=kf,
    )


def run_case(case: dict, kwargs_override: dict):
    """Solve one case on the AdDesignS output time grid for that case.
    Returns (time_min list, C_over_C0 list)."""
    t_eval = np.array(case["time"]) * 60

    p = make_particle(**kwargs_override, time=t_eval)
    _, _, C, _ = p.solve()
    return C[:, -1]


AD_RESULTS = load_addesigns_results(AD_BREAKTHROUGH_FILE)


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

        rmse = reactormodels.numerics.helpers.compute_rmse(
            run_case(case, kwargs_override), case["c"]
        )

        assert rmse < RMSE_THRESHOLD, (
            f"{name}: RMSE {rmse:.5f} exceeds threshold {RMSE_THRESHOLD} "
            f"(param={case['param']}, value={case['value']})"
        )
