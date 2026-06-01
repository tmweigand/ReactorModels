from Data_files_reader import (
    list_breakthrough_matrices,
    read_model_inputs,
    get_column_parameter,
)

from breakthrough_class import ExperimentalBreakthroughCurve


print("\n========== STEP 1: LIST MATRICES ==========")

matrices = list_breakthrough_matrices()

for matrix in matrices:
    print("-", matrix)

if not matrices:
    raise ValueError("No breakthrough matrices found.")


print("\n========== STEP 2: SELECT MATRIX ==========")

selected_matrix = matrices[0]

print("Selected matrix:", selected_matrix)


print("\n========== STEP 3: READ MODEL INPUTS ==========")

data = read_model_inputs(selected_matrix)

compounds_data = data["compounds_data"]
column_parameters = data["column_parameters"]

print("Compounds found:")

for compound_name in compounds_data.keys():
    print("-", compound_name)

if not compounds_data:
    raise ValueError("No compounds found.")


print("\n========== STEP 4: GET COLUMN PARAMETERS ==========")

flow_rate = get_column_parameter(
    column_parameters,
    "Small Column",
    "Volumetric Flow Rate",
)

empty_bed_volume = get_column_parameter(
    column_parameters,
    "Small Column",
    "Bed Volume",
)

print("Flow rate:", flow_rate)
print("Empty bed volume:", empty_bed_volume)


print("\n========== STEP 5: CREATE FIRST CURVE ==========")

first_pfas_name = list(compounds_data.keys())[0]
first_pfas_info = compounds_data[first_pfas_name]

breakthrough_data = first_pfas_info["breakthrough_data"]
chemical_properties = first_pfas_info["pfas_properties"]

feed_concentration = breakthrough_data["Initial Concentration"].iloc[0]

curve = ExperimentalBreakthroughCurve(
    pfas_name=first_pfas_name,
    breakthrough_data=breakthrough_data,
    chemical_properties=chemical_properties,
    process_type="IX",
    flow_rate=flow_rate,
    empty_bed_volume=empty_bed_volume,
    media_mass=None,
    feed_concentration=feed_concentration,
    treatment_objective_fraction=0.05,
    concentration_unit="ng/L",
    flow_rate_unit="mL/min",
    empty_bed_volume_unit="mL",
    time_unit="min",
    media_mass_unit="g",
)

print("Curve created for:", first_pfas_name)


print("\n========== STEP 6: PREPARE C/C0 ==========")

curve.prepare_breakthrough_data()

print(curve.breakthrough_data.head().to_string())


print("\n========== STEP 7: ADD TIME ==========")

curve.add_time_column()

print(curve.breakthrough_data.head().to_string())


print("\n========== STEP 8: ONE-LINE SUMMARY ==========")

print(curve.one_line_summary())


print("\n========== STEP 9: SAVE BREAKTHROUGH PLOTS ==========")

curve.plot_breakthrough(
    x_axis="Bed Volumes",
    save_path=f"plots/{selected_matrix}_{first_pfas_name}_c_over_c0_vs_bv.png",
    show_plot=False,
)

curve.plot_breakthrough(
    x_axis="Time",
    save_path=f"plots/{selected_matrix}_{first_pfas_name}_c_over_c0_vs_time.png",
    show_plot=False,
)

print("Plots saved.")


print("\n========== STEP 10: SAVE PREPARED DATA ==========")

curve.save_prepared_data(
    output_path=f"prepared_data/{selected_matrix}_{first_pfas_name}_prepared.csv"
)

print("Prepared data saved.")


print("\n========== SMOKE TEST PASSED ==========")