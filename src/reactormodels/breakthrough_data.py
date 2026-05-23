"""breakthrough_data.py"""

import numpy as np
import math
import matplotlib.pyplot as plt
import os

class Breakthrough:
    """Experimental breakthrough parameters."""

    def __init__(
        self,
        initial_feed_concentration: float,
        midpoint_feed_concentration: float,
        compound: str,
        water_matrix: str,
        bed_volumes: np.ndarray | None = None,
        time: np.ndarray | None = None,
        effluent_concentrations: np.ndarray | None = None,
        flow_rate: float | None = None
    ):
        assert compound is not None, "No compound provided"

        if bed_volumes is not None:
            assert np.all(np.isfinite(bed_volumes)), \
                "Bed volume data contains NaN(s)"

        if time is not None:
            assert np.all(np.isfinite(time)), \
                "Time data contains NaN(s)"

        if effluent_concentrations is not None:
            assert np.all(np.isfinite(effluent_concentrations)), \
                "Effluent concentration data contains NaN(s)"
        
        self.initial_feed_concentration = initial_feed_concentration
        self.midpoint_feed_concentration = midpoint_feed_concentration
        self.compound = compound
        self.water_matrix = water_matrix
        self.bed_volumes = np.array(bed_volumes)
        self.time = np.array(time)
        self.effluent_concentrations = np.array(effluent_concentrations)
        self.flow_rate = flow_rate

    def mean_feed_concentration(self) -> float:
        """Mean feed concentration during experiment."""
        assert math.isfinite(self.initial_feed_concentration), f"Missing intial feed concentration: {initial_feed_concentration}"
        assert math.isfinite(self.midpoint_feed_concentration), f"Missing midpoint feed concentration: {midpoint_feed_concentration}"
        return (self.initial_feed_concentration + self.midpoint_feed_concentration)/2
    
    def normalize_concentration(self) -> float:
        """Normalize effluent concentration by mean feed concentration."""
        assert math.isfinite(self.mean_feed_concentration()), f"Mean feed concentration was not computed"
        assert self.effluent_concentrations is not None, "Effluent data does not exist"
        return self.effluent_concentrations / self.mean_feed_concentration()

    def empty_bed_contact_time(self, column_volume: float) -> float:
        """Calculate empty bed contact time."""
        assert self.flow_rate is not None, "No flow rate provided"
        assert column_volume is not None, "No column volume provided"
        return column_volume / self.flow_rate
    
    def time_to_bed_volumes(self, empty_bed_contact_time: float) -> np.ndarray:
        """Convert time to bed volumes."""
        assert empty_bed_contact_time is not None, "Empty bed contact time has not been computed"
        return self.time / empty_bed_contact_time
    
    def bed_volumes_to_time(self, empty_bed_contact_time: float) -> np.ndarray:
        """Convert bed volumes to time."""
        assert empty_bed_contact_time is not None, "Empty bed contact time has not been computed"
        return self.bed_volumes * empty_bed_contact_time
    
    def summary(self, column_volume: float) -> str:
        """Summarize breakthrough data."""

        summary = (
            f"Compound: {self.compound}\n"
            f"Water matrix: {self.water_matrix}\n"
            f"Mean feed concentration: {self.mean_feed_concentration():.3f}\n"
            f"Bed volumes: {self.bed_volumes}\n"
            f"Normalized concentrations: {np.array2string(self.normalize_concentration(), precision=3)}\n"
            f"Empty bed contact time: {self.empty_bed_contact_time(column_volume):.3f}\n"
            f"Flow rate: {self.flow_rate:.3f}\n"
        )

        if self.time is not None:
            summary += (
                f"Time: "
                f"{np.array2string(self.bed_volumes_to_time(self.empty_bed_contact_time(column_volume)),precision=3)}\n"
            )
            
        return summary
    
    def plot_normalized_data(self):
        """Generate and save plots of normalized data."""
        os.makedirs("./tests/normalized_plots", exist_ok=True)

        plt.figure()
        
        plt.plot(
            self.bed_volumes,
            self.normalize_concentration(),
            marker="o"
        )

        plt.xlabel("Bed Volumes")
        plt.ylabel("C/Co")
        plt.title(f"{self.compound} - {self.water_matrix}")

        plt.savefig(f'./tests/normalized_plots/{self.compound} - {self.water_matrix}.png', bbox_inches='tight')
        plt.close()
