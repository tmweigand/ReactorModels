"""Test Input_output.py."""

from pathlib import Path

import reactormodels

from reactormodels.IO.Input_output import load_input_file

DATA_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "src" / "reactormodels" / "IO" / "data"
)

PARAMETER_FILE = DATA_DIRECTORY / "experimental_input_parameter.xlsx"

BREAKTHROUGH_FILE = DATA_DIRECTORY / "experimental_input_breakthrough.xlsx"

ISOTHERM_FILE = DATA_DIRECTORY / "experimental_input_isotherm.xlsx"


def test_load_input_file():
    """Test loading all experimental input data."""
    data = load_input_file(
        parameter_file=PARAMETER_FILE,
        breakthrough_file=BREAKTHROUGH_FILE,
        breakthrough_sheet="effluent_concentration",
        isotherm_file=ISOTHERM_FILE,
    )

    # Properties
    assert isinstance(
        data["water"],
        reactormodels.Water,
    )

    assert isinstance(
        data["media"],
        reactormodels.Media,
    )

    assert isinstance(
        data["column"],
        reactormodels.Column,
    )

    # Chemicals
    assert data["chemicals"]

    for chemical in data["chemicals"].values():
        assert isinstance(
            chemical,
            reactormodels.Chemical,
        )

    # Breakthroughs
    assert data["breakthroughs"]

    assert data["chemicals"].keys() == data["breakthroughs"].keys()

    for breakthrough in data["breakthroughs"].values():
        assert isinstance(
            breakthrough,
            reactormodels.Breakthrough,
        )

        assert breakthrough.feed_concentrations is not None
        assert breakthrough.effluent_concentrations is not None
        assert breakthrough.bed_volumes is not None
        assert breakthrough.time is not None

    # Isotherms
    assert "isotherms" in data

    for chemical_isotherms in data["isotherms"].values():
        for isotherm in chemical_isotherms.values():
            assert isinstance(
                isotherm,
                (
                    reactormodels.models.LinearIsotherm,
                    reactormodels.models.LangmuirIsotherm,
                    reactormodels.models.FreundlichIsotherm,
                ),
            )

    # Show loaded data
    print("\n--- WATER ---")
    print(vars(data["water"]))

    print("\n--- MEDIA ---")
    print(vars(data["media"]))

    print("\n--- COLUMN ---")
    print(vars(data["column"]))

    print("\n--- CHEMICALS ---")
    for name, chemical in data["chemicals"].items():
        print(f"\n{name}")
        print(vars(chemical))

    print("\n--- BREAKTHROUGHS ---")
    for name, breakthrough in data["breakthroughs"].items():
        print(f"\n{name}")
        print(vars(breakthrough))

    print("\n--- ISOTHERMS ---")
    for name, chemical_isotherms in data["isotherms"].items():
        print(f"\n{name}")

        for model_name, isotherm in chemical_isotherms.items():
            print(
                model_name,
                vars(isotherm),
            )
