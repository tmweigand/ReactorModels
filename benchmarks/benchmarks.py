"""ASV benchmarks: work-precision and convergence tracking for AdvectionDiffusion."""

import numpy as np

import reactormodels
from reactormodels.numerics import NumericsConfig
from reactormodels.fixtures import make_breakthrough


class ConvergenceSuite:
    """Solve time and accuracy vs the Ogata-Banks analytical solution."""

    params = ([1, 3, 5, 7], [5, 10, 20, 40])
    param_names = ["n_interior_points", "n_elements"]

    def setup(self, n_interior_points, n_elements):
        self.t_end = 1.0
        self.breakthrough = make_breakthrough(t_end=self.t_end)
        numerics = NumericsConfig(
            domain_length=self.breakthrough.column,
            n_interior_points=n_interior_points,
            n_elements=n_elements,
            add_inlet=True,
        )
        self.model = reactormodels.models.AdvectionDiffusion(
            breakthrough=self.breakthrough, numerics=numerics
        )

    def time_solve(self, n_interior_points, n_elements):
        self.model.solve(t_span=(0.0, self.t_end), t_eval=np.array([self.t_end]))

    def track_max_error(self, n_interior_points, n_elements):
        x, concentration_history = self.model.solve(
            t_span=(0.0, self.t_end), t_eval=np.array([self.t_end])
        )
        ogata_banks = reactormodels.models.OgataBanks(
            breakthrough=self.breakthrough,
            diffusion=self.breakthrough.chemical.diffusion,
        )

        error = concentration_history[-1] - ogata_banks.spatial_profile(
            x=x, time=self.t_end
        )
        return float(np.max(np.abs(error)))

    track_max_error.unit = "concentration ratio"
