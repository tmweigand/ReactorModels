"""Test Input_output.py."""

from pathlib import Path

import reactormodels

from reactormodels.IO.Input_output import load_input_file

DATA_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "src" / "reactormodels" / "IO" / "data"
)

PARAMETER_FILE = DATA_DIRECTORY / "experimental_input_parameter.xlsx"

BREAKTHROUGH_FILE = DATA_DIRECTORY / "experimental_input_breakthrough.xlsx"


def test_load_input_file():
    """Test loading all experimental input data."""
    data = load_input_file(
        parameter_file=PARAMETER_FILE,
        breakthrough_file=BREAKTHROUGH_FILE,
        breakthrough_sheet="effluent_concentration",
    )

    assert isinstance(data["water"], reactormodels.Water)
    assert isinstance(data["media"], reactormodels.Media)
    assert isinstance(data["column"], reactormodels.Column)

    assert data["chemicals"]
    assert data["breakthroughs"]

    for chemical in data["chemicals"].values():
        assert isinstance(
            chemical,
            reactormodels.Chemical,
        )

    for breakthrough in data["breakthroughs"].values():
        assert isinstance(
            breakthrough,
            reactormodels.Breakthrough,
        )

    print("\nWater:")
    print(vars(data["water"]))

    print("\nMedia:")
    print(vars(data["media"]))

    print("\nColumn:")
    print(vars(data["column"]))

    print("\nChemicals:")
    for name, chemical in data["chemicals"].items():
        print(name, vars(chemical))

    print("\nBreakthroughs:")
    for name, breakthrough in data["breakthroughs"].items():
        print(name, vars(breakthrough))
