from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class ExperimentalBreakthroughCurve:
    """
    Experimental breakthrough curve for one PFAS compound.

    This class is for fixed-bed adsorption or ion-exchange breakthrough data.

    Main curve:
        y = C/C0
        x = Bed Volumes or Time

    Required breakthrough_data columns:
        - Bed Volumes
        - Concentration

    Optional breakthrough_data columns:
        - Initial Concentration
        - Time
    """

    def __init__(
        self,
        pfas_name,
        breakthrough_data,
        chemical_properties=None,
        process_type="IX",
        flow_rate=None,
        empty_bed_volume=None,
        media_mass=None,
        feed_concentration=None,
        treatment_objective_fraction=0.05,
        concentration_unit="ng/L",
        flow_rate_unit="mL/min",
        empty_bed_volume_unit="mL",
        time_unit="min",
        media_mass_unit="g",
    ):
        self.pfas_name = pfas_name
        self.breakthrough_data = breakthrough_data.copy()
        self.chemical_properties = chemical_properties or {}

        self.process_type = process_type

        self.flow_rate = flow_rate
        self.empty_bed_volume = empty_bed_volume
        self.media_mass = media_mass
        self.feed_concentration = feed_concentration
        self.treatment_objective_fraction = treatment_objective_fraction

        self.concentration_unit = concentration_unit
        self.flow_rate_unit = flow_rate_unit
        self.empty_bed_volume_unit = empty_bed_volume_unit
        self.time_unit = time_unit
        self.media_mass_unit = media_mass_unit

        self._check_required_columns()


    # 1. Basic checks

    def _check_required_columns(self):
        """
        Checks that the breakthrough data has the required columns.
        """

        required_columns = ["Bed Volumes", "Concentration"]

        for column_name in required_columns:
            if column_name not in self.breakthrough_data.columns:
                raise ValueError(f"Missing required column: {column_name}")

    def _get_feed_concentration(self):
        """
        Gets feed concentration C0.

        Priority:
            1. feed_concentration passed into the class
            2. Initial Concentration column in breakthrough_data
        """

        if self.feed_concentration is not None:
            return self.feed_concentration

        if "Initial Concentration" in self.breakthrough_data.columns:
            self.feed_concentration = self.breakthrough_data[
                "Initial Concentration"
            ].iloc[0]
            return self.feed_concentration

        raise ValueError(
            "Feed concentration is missing. Provide feed_concentration "
            "or include an 'Initial Concentration' column."
        )


    # 2. Breakthrough preparation

    def prepare_breakthrough_data(self):
        """
        Prepares the normalized breakthrough data.

        Adds:
            - Feed Concentration
            - C/C0
            - Percent Removal
            - Treatment Objective Concentration
        """

        feed_concentration = self._get_feed_concentration()

        self.breakthrough_data["Feed Concentration"] = feed_concentration

        self.breakthrough_data["C/C0"] = (
            self.breakthrough_data["Concentration"] / feed_concentration
        )

        self.breakthrough_data["Percent Removal"] = (
            1.0 - self.breakthrough_data["C/C0"]
        ) * 100.0

        self.breakthrough_data["Treatment Objective Concentration"] = (
            self.treatment_objective_fraction * feed_concentration
        )

        return self.breakthrough_data

    def treatment_objective_concentration(self):
        """
        Treatment objective concentration.

        treatment objective concentration = treatment objective fraction * C0
        """

        feed_concentration = self._get_feed_concentration()
        return self.treatment_objective_fraction * feed_concentration

    # 3. EBCT, time, and bed volume calculations

    def calculate_ebct(self):
        """
        Empty-bed contact time.

        EBCT = empty bed volume / flow rate

        Example:
            empty_bed_volume = mL
            flow_rate = mL/min
            EBCT = min
        """

        if self.empty_bed_volume is None:
            raise ValueError("empty_bed_volume is required to calculate EBCT.")

        if self.flow_rate is None:
            raise ValueError("flow_rate is required to calculate EBCT.")

        return self.empty_bed_volume / self.flow_rate

    def add_time_column(self, time_column_name="Time"):
        """
        Converts bed volumes to time.

        time = bed volumes * empty bed volume / flow rate

        Equivalent:
            time = bed volumes * EBCT
        """

        if self.empty_bed_volume is None:
            raise ValueError("empty_bed_volume is required to calculate time.")

        if self.flow_rate is None:
            raise ValueError("flow_rate is required to calculate time.")

        self.breakthrough_data[time_column_name] = (
            self.breakthrough_data["Bed Volumes"]
            * self.empty_bed_volume
            / self.flow_rate
        )

        return self.breakthrough_data

    def add_bed_volumes_from_time(
        self,
        time_column_name="Time",
        output_column_name="Calculated Bed Volumes",
    ):
        """
        Converts time to bed volumes.

        bed volumes = time * flow rate / empty bed volume
        """

        if time_column_name not in self.breakthrough_data.columns:
            raise ValueError(f"Missing time column: {time_column_name}")

        if self.empty_bed_volume is None:
            raise ValueError("empty_bed_volume is required.")

        if self.flow_rate is None:
            raise ValueError("flow_rate is required.")

        self.breakthrough_data[output_column_name] = (
            self.breakthrough_data[time_column_name]
            * self.flow_rate
            / self.empty_bed_volume
        )

        return self.breakthrough_data

    # 4. Breakthrough checks

    def has_breakthrough(self):
        """
        Checks whether breakthrough occurred.

        Breakthrough condition:
            C/C0 >= treatment_objective_fraction
        """

        self.prepare_breakthrough_data()

        breakthrough_rows = self.breakthrough_data[
            self.breakthrough_data["C/C0"] >= self.treatment_objective_fraction
        ]

        return not breakthrough_rows.empty

    def get_breakthrough_point(self):
        """
        Returns the first breakthrough point.

        Returns:
            pandas Series if breakthrough occurred
            None if breakthrough was not reached
        """

        self.prepare_breakthrough_data()

        breakthrough_rows = self.breakthrough_data[
            self.breakthrough_data["C/C0"] >= self.treatment_objective_fraction
        ]

        if breakthrough_rows.empty:
            return None

        return breakthrough_rows.iloc[0]

    def breakthrough_bed_volume(self):
        """
        Returns the bed volume at breakthrough.
        """

        breakthrough_point = self.get_breakthrough_point()

        if breakthrough_point is None:
            return None

        return breakthrough_point["Bed Volumes"]

    def breakthrough_time(self):
        """
        Returns the time at breakthrough.

        If the Time column does not exist, it is calculated.
        """

        breakthrough_point = self.get_breakthrough_point()

        if breakthrough_point is None:
            return None

        if "Time" not in self.breakthrough_data.columns:
            self.add_time_column()

        breakthrough_bv = breakthrough_point["Bed Volumes"]

        matching_row = self.breakthrough_data[
            self.breakthrough_data["Bed Volumes"] == breakthrough_bv
        ]

        return matching_row["Time"].iloc[0]

    # 5. Fixed-bed performance calculations

    def treated_volume_at_breakthrough(self):
        """
        Total treated volume at breakthrough.

        treated volume = breakthrough bed volumes * empty bed volume
        """

        breakthrough_bv = self.breakthrough_bed_volume()

        if breakthrough_bv is None:
            return None

        if self.empty_bed_volume is None:
            raise ValueError("empty_bed_volume is required.")

        return breakthrough_bv * self.empty_bed_volume

    def specific_throughput_at_breakthrough(self):
        """
        Specific throughput at breakthrough.

        specific throughput = treated volume / media mass

        Example units:
            mL/g
        """

        treated_volume = self.treated_volume_at_breakthrough()

        if treated_volume is None:
            return None

        if self.media_mass is None:
            raise ValueError("media_mass is required.")

        return treated_volume / self.media_mass

    def media_usage_rate_at_breakthrough(self):
        """
        Media usage rate at breakthrough.

        media usage rate = media mass / treated volume

        Example units:
            g/mL

        For GAC:
            carbon usage rate

        For IX:
            resin usage rate
        """

        treated_volume = self.treated_volume_at_breakthrough()

        if treated_volume is None:
            return None

        if self.media_mass is None:
            raise ValueError("media_mass is required.")

        return self.media_mass / treated_volume

    # 6. Unit labels

    def concentration_label(self):
        return f"Concentration ({self.concentration_unit})"

    def feed_concentration_label(self):
        return f"Feed Concentration ({self.concentration_unit})"

    def flow_rate_label(self):
        return f"Flow Rate ({self.flow_rate_unit})"

    def empty_bed_volume_label(self):
        return f"Empty Bed Volume ({self.empty_bed_volume_unit})"

    def time_label(self):
        return f"Time ({self.time_unit})"

    def media_mass_label(self):
        return f"Media Mass ({self.media_mass_unit})"

    def specific_throughput_unit(self):
        return f"{self.empty_bed_volume_unit}/{self.media_mass_unit}"

    def media_usage_rate_unit(self):
        return f"{self.media_mass_unit}/{self.empty_bed_volume_unit}"

    # 7. One-line summary

    def one_line_summary(self):
        """
        Returns one clean summary line for one PFAS.
        """

        self.prepare_breakthrough_data()

        feed_concentration = self._get_feed_concentration()
        max_c_over_c0 = self.breakthrough_data["C/C0"].max()
        number_of_points = len(self.breakthrough_data)

        breakthrough_status = self.has_breakthrough()
        breakthrough_bv = self.breakthrough_bed_volume()

        try:
            breakthrough_time = self.breakthrough_time()
        except ValueError:
            breakthrough_time = None

        try:
            ebct = self.calculate_ebct()
        except ValueError:
            ebct = None

        try:
            treated_volume = self.treated_volume_at_breakthrough()
        except ValueError:
            treated_volume = None

        try:
            specific_throughput = self.specific_throughput_at_breakthrough()
        except ValueError:
            specific_throughput = None

        try:
            media_usage_rate = self.media_usage_rate_at_breakthrough()
        except ValueError:
            media_usage_rate = None

        molecular_weight = self.chemical_properties.get("Molecular Weight", "NA")
        density = self.chemical_properties.get("Density", "NA")
        solubility = self.chemical_properties.get("Solubility", "NA")

        return (
            f"{self.pfas_name} | "
            f"process={self.process_type} | "
            f"points={number_of_points} | "
            f"C0={feed_concentration} {self.concentration_unit} | "
            f"objective={self.treatment_objective_fraction} C/C0 "
            f"({self.treatment_objective_concentration()} {self.concentration_unit}) | "
            f"max C/C0={max_c_over_c0:.3f} | "
            f"breakthrough={breakthrough_status} | "
            f"breakthrough BV={breakthrough_bv} | "
            f"breakthrough time={breakthrough_time} {self.time_unit} | "
            f"EBCT={ebct} {self.time_unit} | "
            f"treated volume={treated_volume} {self.empty_bed_volume_unit} | "
            f"specific throughput={specific_throughput} {self.specific_throughput_unit()} | "
            f"media usage rate={media_usage_rate} {self.media_usage_rate_unit()} | "
            f"MW={molecular_weight} | "
            f"density={density} | "
            f"solubility={solubility}"
        )


    # 8. Plot only breakthrough curve

    def plot_breakthrough(
        self,
        x_axis="Bed Volumes",
        save_path=None,
        show_plot=True,
    ):
        """
        Plots only the breakthrough curve.

        y-axis:
            C/C0

        x-axis options:
            Bed Volumes
            Time
        """

        self.prepare_breakthrough_data()

        if x_axis == "Time" and "Time" not in self.breakthrough_data.columns:
            self.add_time_column()

        if x_axis not in self.breakthrough_data.columns:
            raise ValueError("x_axis must be 'Bed Volumes' or 'Time'.")

        if x_axis == "Time":
            x_label = self.time_label()
        else:
            x_label = "Bed Volumes"

        plt.figure()
        plt.plot(
            self.breakthrough_data[x_axis],
            self.breakthrough_data["C/C0"],
            marker="o",
        )

        plt.xlabel(x_label)
        plt.ylabel("C/C0")
        plt.title(f"{self.process_type} Breakthrough Curve: {self.pfas_name}")
        plt.grid(True)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    # 9. Save prepared data

    def save_prepared_data(self, output_path):
        """
        Saves prepared breakthrough data.
        """

        self.prepare_breakthrough_data()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".csv":
            self.breakthrough_data.to_csv(output_path, index=False)

        elif output_path.suffix in [".xlsx", ".xls"]:
            self.breakthrough_data.to_excel(output_path, index=False)

        else:
            raise ValueError("Output file must be .csv, .xlsx, or .xls.")