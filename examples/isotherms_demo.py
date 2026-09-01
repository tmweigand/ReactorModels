import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/isotherm/IEX-CM.png",
):
    """Plot single- vs multi-species isotherms."""

    C_values = np.linspace(0.01, 1000, 100)

    # freundlich
    K_f = np.array([2, 1.8])
    n = np.array([1.5, 1.2])

    # langmuir
    q_m = 50
    K_l = np.array([0.001, 0.001])

    # EPA charge equivalent
    K_ix = np.array([0.001, 0.003])
    MW = np.array([100.0, 50.0])
    z = np.array([2.0, 1.0])
    C_o = np.array([1.0, 1.0])

    q_ix = 0.5
    rho_b = 0.5

    q_values = np.linspace(0.01, 10, 100)

    # print("q =", q)
    # print("q_scale =", rho_b * z[0] / MW[0])
    # print("q_eq =", q * rho_b * z[0] / MW[0])
    # print("q_A =", q_m - np.sum(q * rho_b * z[0] / MW[0]))

    ms_ix = reactormodels.models.CompetitiveIonIsotherm(K_ix, MW, z, C_o, q_ix, rho_b)

    # print("C(q) =", ms_ix.C(q))

    # multi-capacity
    q_mc = [50, 30]

    # adsorbate complex
    K_x = 0.0001

    C = np.tile(C_values, (len(K_l), 1))

    q = np.zeros((2, 100))

    # vary species 1
    q[1, :] = np.linspace(1e-6, 0.49, 100)

    # hold species 2 constant
    q[0, :] = np.linspace(1e-6, 0.49, 100)  # np.tile(q_values, (len(K_l), 1))

    # ms_freundlich = reactormodels.models.CompetitiveFreundlichIsotherm(K_f, n)
    ms_langmuir = reactormodels.models.CompetitiveLangmuirIsotherm(K_l, q_m)
    # ms_lf = reactormodels.models.CompetitiveLangmuirFreundlichIsotherm(
    #     K_l ** (1 / n), 1 / n, q_m
    # )
    # ms_stoich = reactormodels.models.CompetitiveStoichiometricIsotherm(
    #     K_l ** (1 / n), 1 / n, q_m
    # )
    # ms_mc = reactormodels.models.MultiCapacityIsotherm(K_l, q_mc)
    # ms_adscomp = reactormodels.models.AdsorbateComplexIsotherm(K_l, q_m, K_x)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["tab:pink", "tab:green"]

    for i in range(len(K_ix)):
        color = colors[i % len(colors)]
        # ax.plot(
        #     C[i],
        #     ms_adscomp.q(C)[i],
        #     linestyle="--",
        #     linewidth=3,
        #     label=f"Adsorbate Complex: Species {i+1}",
        #     color=color,
        # )

        # freundlich = reactormodels.models.FreundlichIsotherm(K_f[i], n[i])
        # langmuir = reactormodels.models.LangmuirIsotherm(K_l[i], q_m)

        # ax.plot(
        #     freundlich.C(q[i]),
        #     q[i],
        #     linestyle="-",
        #     linewidth=3,
        #     label=f"Single-species Freundlich: Species {i+1}",
        #     color=color,
        # )
        # ax.plot(
        #     ms_freundlich.C(q)[i],
        #     q[i],
        #     linestyle="--",
        #     linewidth=3,
        #     label=f"Multi-species Freundlich: Species {i+1}",
        #     color=color,
        # )
        # ax.plot(
        #     C[i],
        #     langmuir.q(C[i]),
        #     linestyle="-",
        #     label=f"Single-species Langmuir: Species {i+1}",
        #     linewidth=3,
        #     color=color,
        # )
        # ax.plot(
        #     C[i],
        #     ms_langmuir.q(C)[i],
        #     linestyle="-",
        #     label=f"Multi-species Langmuir: Species {i+1}",
        #     linewidth=3,
        #     color=color,
        # )
        ax.plot(
            ms_ix.C(q)[i] * MW[i] / z[i],
            q[i],
            linestyle="--",
            label=f"Multi-species IEX-CM: Species {i+1}",
            linewidth=3,
            color=color,
        )

        for i in range(0, 100, 10):
            print(f"q = {q[0,i]:.6g}, C = {ms_ix.C(q)[0,i]:.6g}")
        # ax.plot(
        #     C[i],
        #     ms_lf.q(C)[i],
        #     linestyle="--",
        #     label=f"Multi-species Langmuir-Freundlich: Species {i+1}",
        #     linewidth=3,
        #     color=color,
        # )
        # ax.plot(
        #     C[i],
        #     ms_stoich.q(C)[i],
        #     linestyle=":",
        #     label=f"Multi-species Stoichiometric: Species {i+1}",
        #     linewidth=3,
        #     color=color,
        # )
        # ax.plot(
        #     C[i],
        #     ms_mc.q(C)[i],
        #     linestyle=":",
        #     label=f"Multi-capacity Langmuir: Species {i+1}",
        #     linewidth=3,
        #     color=color,
        # )

    ax.set_xlabel("C", fontsize=16)
    ax.set_ylabel("q*", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncols=1)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    run_demo()
