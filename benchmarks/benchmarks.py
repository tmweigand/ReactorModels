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
        self.time = np.array([1.0])
        self.breakthrough = make_breakthrough(time=self.time)
        numerics = NumericsConfig(
            domain_length=self.breakthrough.column.length,
            n_interior_points=n_interior_points,
            n_elements=n_elements,
        )
        self.model = reactormodels.models.AdvectionDiffusion(
            breakthrough=self.breakthrough, numerics=numerics
        )

    def time_solve(self, n_interior_points, n_elements):
        self.model.solve()

    def track_max_error(self, n_interior_points, n_elements):
        x, concentration_history = self.model.solve()
        ogata_banks = reactormodels.models.OgataBanks(
            breakthrough=self.breakthrough,
            diffusion=self.breakthrough.chemical.axial_diffusion,
        )

        error = concentration_history[-1] - ogata_banks.spatial_profile(
            x=x, time=self.time
        )
        return float(np.max(np.abs(error)))

    track_max_error.unit = "concentration ratio"
