from Breakthrough_PSDM.smoke_test import empty_bed_volume
from Data_files_reader import read_model_inputs, get_column_parameter
from breakthrough_class import ExperimentalBreakthroughCurve


selected_matrix = "LGW-1"

data = read_model_inputs(selected_matrix)

compounds_data = data["compounds_data"]
column_parameters = data["column_parameters"]


flow_rate = get_column_parameter(
    column_parameters,
    "Small Column",
    "Volumetric Flow Rate",
)

bed_volume = get_column_parameter(
    column_parameters,
    "Small Column",
    "Bed Volume",
)


for pfas_name, pfas_information in compounds_data.items():

    breakthrough_data = pfas_information["breakthrough_data"]
    chemical_properties = pfas_information["pfas_properties"]

    feed_concentration = breakthrough_data["Initial Concentration"].iloc[0]

    curve = ExperimentalBreakthroughCurve(
        pfas_name=pfas_name,
        breakthrough_data=breakthrough_data,
        chemical_properties=chemical_properties,
        flow_rate=flow_rate,
        empty_bed_volume=empty_bed_volume,
        feed_concentration=feed_concentration,
    )

    print(curve.one_line_summary())

    curve.plot_breakthrough(
        x_axis="Bed Volumes",
        save_path=f"plots/{selected_matrix}_{pfas_name}_c_over_c0_vs_bv.png",
        show_plot=False,
    )

    curve.plot_breakthrough(
        x_axis="Time",
        save_path=f"plots/{selected_matrix}_{pfas_name}_c_over_c0_vs_time.png",
        show_plot=False,
    )

    curve.save_prepared_data(
        output_path=f"prepared_data/{selected_matrix}_{pfas_name}_prepared.csv"
    )