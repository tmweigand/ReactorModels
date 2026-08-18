"""AdDesignS_comparison.py

Runs reactormodels' PSDM (`_make_particle` / DomainCoupling) for every
case already present in regression/AdDesignS_breakthrough.txt, so the
two models can be compared case-by-case.

Outputs
-------
regression/reactormodels_breakthrough.txt
    Same tab-delimited long format as AdDesignS_breakthrough.txt:
    name, param, value, pct_change, time_min, C_over_C0
    (time_min matches the AdDesignS output grid for that case)

regression/regression_out/<name>.png
    Overlay plot of ReactorModels vs AdDesignS for each case (same
    style as your original single-case script).

"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import reactormodels

# ============================================================
# CONFIG
# ============================================================

AD_BREAKTHROUGH_FILE = Path("regression/AdDesignS_breakthrough.txt")
OUT_BREAKTHROUGH_FILE = Path("regression/reactormodels_breakthrough.txt")
PLOT_DIR = Path("data_out/regression/regression_out")

# AdDesignS param name -> _make_particle kwarg name
PARAM_MAP = {
    "kf": "kf",
    "surface_diffusion": "surface_diffusion",
    "pore_diffusion": "pore_diffusion",
    "particle_porosity": "particle_porosity",
    "particle_radius": "particle_radius",
}

# Baseline kwargs - must match _make_particle's defaults / the
# AdDesignS baseline case's physical values.
BASELINE_KWARGS = dict(
    surface_diffusion=5e-10,
    pore_diffusion=5e-6,
    kf=0.1,
    particle_porosity=0.5,
    particle_radius=0.07,
)


# ============================================================
# Load the AdDesignS regression results
# ============================================================


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


# ============================================================
# reactormodels PSDM setup
# ============================================================


def _make_particle(
    surface_diffusion=5e-10,
    pore_diffusion=5e-6,
    kf=0.1,
    particle_porosity=0.5,
    particle_radius=0.07,
    time=None,
):

    isotherm = reactormodels.models.LinearIsotherm(K=100)  # (mg/g) * (L/mg)

    media = reactormodels.Media(
        particle_porosity=particle_porosity,
        particle_radius=particle_radius,
        particle_density=600,  # g/mL
    )

    column = reactormodels.Column(
        length=100,  # cm
        porosity=0.334,
        diameter=10,  # cm
        bulk_density=399.8,  # g/L
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
        flow_rate=40,  # cm3/s
        time=time,
        chemical=chemical,
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length,
        n_interior_points=8,
        n_elements=6,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=1,
    )

    return reactormodels.models.DomainCoupling(
        isotherm=isotherm,
        breakthrough=breakthrough,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=kf,
    )


# ============================================================
# Per-case run
# ============================================================


def run_case(case: dict, kwargs_override: dict):
    """Solve one case on the AdDesignS output time grid for that case.
    Returns (time_min list, C_over_C0 list)."""
    t_eval = np.array(case["time"]) * 60
    time = t_eval / 1440 / 60

    p = _make_particle(**kwargs_override, time=time)
    z, r, C, Cp = p.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)
    C_numerical = C[:, -1]
    return case["time"], C_numerical.tolist()


def make_plot(name, case, time_min, c_numerical, out_dir: Path):
    time_days = np.array(time_min) / 1440
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_days, c_numerical, linestyle="-", label="ReactorModels")
    ax.plot(time_days, case["c"], linestyle="--", label="AdDesignS", color="black")
    ax.set_title(f"PSDM Breakthrough Profile - {name}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("C/C\u2080")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=1)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{name}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def make_param_sensitivity_plot(
    param, case_names, ad_results, model_results, out_dir: Path
):
    """One figure per varied parameter: baseline + every perturbation level
    of that parameter, both models overlaid (solid = ReactorModels,
    dashed = AdDesignS), color-coded by case so a pair is easy to match."""
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(case_names)))

    for color, name in zip(colors, case_names):
        case = ad_results[name]
        mr = model_results.get(name)
        if mr is None:
            continue  # case failed to solve - skip it in the plot
        time_days = np.array(mr["time_min"]) / 1440
        if name == "baseline":
            label = "baseline"
        else:
            pct = float(case["pct_change"])
            label = f"{pct:+.1%}"
        ax.plot(
            time_days,
            mr["c_numerical"],
            color=color,
            linestyle="-",
            label=f"ReactorModels ({label})",
        )
        ax.plot(
            time_days,
            case["c"],
            color="black",
            linestyle="--",
            label=f"AdDesignS ({label})",
        )

    ax.set_title(f"Sensitivity - {param}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("C/C\u2080")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"sensitivity_{param}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ============================================================
# Main
# ============================================================


def main():
    ad_results = load_addesigns_results(AD_BREAKTHROUGH_FILE)
    print(f"Loaded {len(ad_results)} cases from {AD_BREAKTHROUGH_FILE}")

    out_rows = []
    rmse_rows = []
    model_results = {}  # name -> {"time_min": [...], "c_numerical": [...]}
    failures = []

    for name, case in ad_results.items():
        print(name)
        kwargs_override = dict(BASELINE_KWARGS)
        if case["param"] != "baseline":
            mapped = PARAM_MAP.get(case["param"])
            if mapped is None:
                print(f"  [{name}] SKIPPED - unrecognized param '{case['param']}'")
                continue
            kwargs_override[mapped] = float(case["value"])

        print(kwargs_override)

        print(
            f"  [{name}] running (param={case['param']}, "
            f"value={case['value'] or 'baseline'})..."
        )

        time_min, c_numerical = run_case(case, kwargs_override)

        model_results[name] = {"time_min": time_min, "c_numerical": c_numerical}

        for t, c in zip(time_min, c_numerical):
            out_rows.append(
                {
                    "name": name,
                    "param": case["param"],
                    "value": case["value"],
                    "pct_change": case["pct_change"],
                    "time_min": t,
                    "C_over_C0": c,
                }
            )

        rmse = reactormodels.numerics.helpers.compute_rmse(c_numerical, case["c"])
        rmse_rows.append(
            {
                "name": name,
                "param": case["param"],
                "value": case["value"],
                "pct_change": case["pct_change"],
                "rmse": rmse,
            }
        )

        make_plot(name, case, time_min, c_numerical, PLOT_DIR)

    OUT_BREAKTHROUGH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BREAKTHROUGH_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "param",
                "value",
                "pct_change",
                "time_min",
                "C_over_C0",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(out_rows)

    rmse_path = OUT_BREAKTHROUGH_FILE.parent / "rmse_summary.csv"
    with open(rmse_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "param", "value", "pct_change", "rmse"]
        )
        writer.writeheader()
        writer.writerows(rmse_rows)

    print(f"\nWrote {len(out_rows)} rows to {OUT_BREAKTHROUGH_FILE}")
    print(f"Wrote RMSE summary ({len(rmse_rows)} cases) to {rmse_path}")

    print("\nRMSE by case (ReactorModels vs AdDesignS, on C/C0):")
    for row in sorted(rmse_rows, key=lambda r: (r["param"], r["name"])):
        print(
            f"  {row['name']:<20} param={row['param']:<12} " f"rmse={row['rmse']:.5f}"
        )

    # One overlay plot per varied parameter: baseline + all its perturbations
    params_present = sorted(
        {c["param"] for c in ad_results.values() if c["param"] != "baseline"}
    )
    for param in params_present:
        case_names = [n for n, c in ad_results.items() if c["param"] == param]
        case_names.sort(key=lambda n: float(ad_results[n]["pct_change"]))
        if "baseline" in model_results:
            case_names = ["baseline"] + case_names
        make_param_sensitivity_plot(
            param, case_names, ad_results, model_results, PLOT_DIR
        )

    print(f"\nPer-case plots and per-parameter sensitivity plots written to {PLOT_DIR}")
    if failures:
        print(f"\n{len(failures)} case(s) FAILED:")
        for n, err in failures:
            print(f"  {n}: {err}")


if __name__ == "__main__":
    main()
