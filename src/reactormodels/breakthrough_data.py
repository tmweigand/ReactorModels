"""breakthrough_data.py"""

import numpy as np
import math

class Breakthrough:
    """Experimental breakthrough parameters."""

    def __init__(
        self,
        initial_feed_concentration: float,
        midpoint_feed_concentration: float,
        compound: str,
        water_matrix: str,
        bed_volumes: float | None = None,
        time: float | None = None,
        effluent_concentrations: float | None = None,
        flow_rate: float | None = None
    ):
        assert compound is not None, "No compound provided"
        assert math.isfinite(bed_volumes), "Bed volume data contains NaN(s)"
        assert math.isfinite(time), "Time data contains NaN(s)"
        assert math.isfinite(effluent_concentrations), "Effluent concentration data contains NaN(s)"
        
        self.initial_feed_concentration = initial_feed_concentration
        self.midpoint_feed_concentration = midpoint_feed_concentration
        self.compound = compound
        self.water_matrix = water_matrix
        self.bed_volumes = bed_volumes
        self.time = time
        self.effluent_concentrations = effluent_concentrations
        self.flow_rate = flow_rate

    def mean_feed_concentration(self) -> float:
        """Mean feed concentration during experiment."""
        assert math.isfinite(initial_feed_concentration), f"Missing intial feed concentration: {initial_feed_concentration}"
        assert math.isfinite(midpoint_feed_concentration), f"Missing midpoint feed concentration: {midpoint_feed_concentration}"
        return (self.initial_feed_concentration + self.midpoint_feed_concentration)/2
    
    def normalize_concentration(self) -> float:
        """Normalize effluent concentration by mean feed concentration."""
        assert math.isfinite(self.mean_feed_concentration), f"Mean feed concentration was not computed"
        assert self.effluent_concentrations is not None, "Effluent data does not exist"
        return self.effluent_concentrations / self.mean_feed_concentration

    def empty_bed_contact_time(self, column_volume: float) -> float:
        """Calculate empty bed contact time."""
        assert self.flow_rate is not None, "No flow rate provided"
        assert column_volume is not None, "No column volume provided"
        return column_volume / self.flow_rate
    
    def time_to_bed_volumes(self, empty_bed_contact_time: float):
        """Convert time to bed volumes."""
        assert empty_bed_contact_time is not None, "Empty bed contact time has not been computed"
        return self.time / empty_bed_contact_time
    
    def bed_volumes_to_time(self, empty_bed_contact_time: float):
        """Convert bed volumes to time."""
        assert empty_bed_contact_time is not None, "Empty bed contact time has not been computed"
        return self.bed_volumes * empty_bed_contact_time
    
    def summary(self) -> str:
        """Summarize breakthrough data."""
        return(
            f"Compound: {self.compound}\n"
            f"Water matrix: {self.water_matrix}\n"
            f"Mean feed concentration: {self.mean_feed_concentration:.3f}\n"
            f"Bed volumes: {self.bed_volumes:.0f}\n"
            f"Normalized concentrations: {self.normalize_concentration():.3f}\n"
            f"Empty bed contact time: {self.empty_bed_contact_time():.4f}\n"
            f"Flow rate: {self.flow_rate:.3f}"
            f"Time: {self.bed_volumes_to_time():.0f}\n" if self.time is None             
        )   