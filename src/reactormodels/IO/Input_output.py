"""Input_output.py."""

from pathlib import Path
import numpy as np
import openpyxl

from reactormodels.properties import Breakthrough, Chemical, Column, Media, Water


def normalize_name(name: str) -> str:
    """Normalize chemical names for matching."""
    return "".join(str(name).split()).casefold()


def _load_properties(sheet):
    """Create Water, Media, Column, and load Breakthrough parameters."""
    water_parameters = {}
    media_parameters = {}
    column_parameters = {}
    breakthrough_parameters = {}

    sections = {
        "water": water_parameters,
        "media": media_parameters,
        "column": column_parameters,
        "breakthrough": breakthrough_parameters,
    }

    current_section = None

    for parameter, value, *_ in sheet.iter_rows(values_only=True):
        if parameter is None:
            continue

        parameter_name = str(parameter).strip()
        key = parameter_name.casefold()

        if key in sections:
            current_section = sections[key]
            continue

        if key in {"parameter", "parameters"}:
            continue

        if current_section is not None:
            current_section[parameter_name] = value

    water = Water(**water_parameters)
    media = Media(**media_parameters)

    column = Column(
        water=water,
        media=media,
        **column_parameters,
    )

    return water, media, column, breakthrough_parameters


def _load_chemicals(sheet, compound_names):
    """Create Chemical objects used in the breakthrough data."""
    chemicals = {}

    normalized_compounds = {normalize_name(name) for name in compound_names}

    for column_number in range(2, sheet.max_column + 1):
        chemical_name = sheet.cell(
            row=1,
            column=column_number,
        ).value

        if chemical_name is None:
            continue

        normalized_name = normalize_name(chemical_name)

        if normalized_name not in normalized_compounds:
            continue

        chemical_parameters = {}

        for row in range(2, sheet.max_row + 1):
            parameter_name = sheet.cell(
                row=row,
                column=1,
            ).value

            if parameter_name is None:
                continue

            chemical_parameters[str(parameter_name).strip()] = sheet.cell(
                row=row,
                column=column_number,
            ).value

        chemicals[normalized_name] = Chemical(
            name=str(chemical_name).strip(),
            **chemical_parameters,
        )

    return chemicals


def _load_breakthroughs(
    workbook,
    sheet_name,
    chemicals,
    column,
    breakthrough_parameters,
):
    """Create Breakthrough objects from the selected sheet."""
    sheet = workbook[sheet_name]

    breakthroughs = {}

    for column_number in range(3, sheet.max_column + 1):
        chemical_name = sheet.cell(
            row=1,
            column=column_number,
        ).value

        if chemical_name is None:
            continue

        normalized_name = normalize_name(chemical_name)

        if normalized_name not in chemicals:
            raise ValueError(f"{chemical_name!r} is not in the chemical sheet.")

        concentrations = []
        bed_volumes = []
        time = []

        for row in range(2, sheet.max_row + 1):
            concentration = sheet.cell(
                row=row,
                column=column_number,
            ).value

            if concentration is None:
                continue

            concentrations.append(concentration)
            bed_volumes.append(sheet.cell(row=row, column=1).value)
            time.append(sheet.cell(row=row, column=2).value)

        breakthroughs[normalized_name] = Breakthrough(
            chemical=chemicals[normalized_name],
            column=column,
            feed_concentrations=np.asarray(
                concentrations,
                dtype=float,
            ),
            bed_volumes=np.asarray(
                bed_volumes,
                dtype=float,
            ),
            time=np.asarray(
                time,
                dtype=float,
            ),
            **breakthrough_parameters,
        )

    return breakthroughs


def load_input_file(
    parameter_file: str | Path,
    breakthrough_file: str | Path,
    breakthrough_sheet: str,
):
    """Load all ReactorModels inputs."""
    parameter_workbook = openpyxl.load_workbook(
        parameter_file,
        data_only=True,
    )

    breakthrough_workbook = openpyxl.load_workbook(
        breakthrough_file,
        data_only=True,
    )

    parameter_sheet = parameter_workbook.worksheets[0]
    chemical_sheet = parameter_workbook.worksheets[1]

    selected_breakthrough_sheet = breakthrough_workbook[breakthrough_sheet]

    compound_names = [
        selected_breakthrough_sheet.cell(
            row=1,
            column=column_number,
        ).value
        for column_number in range(
            3,
            selected_breakthrough_sheet.max_column + 1,
        )
        if selected_breakthrough_sheet.cell(
            row=1,
            column=column_number,
        ).value
        is not None
    ]

    (
        water,
        media,
        column,
        breakthrough_parameters,
    ) = _load_properties(parameter_sheet)

    chemicals = _load_chemicals(
        chemical_sheet,
        compound_names,
    )

    breakthroughs = _load_breakthroughs(
        workbook=breakthrough_workbook,
        sheet_name=breakthrough_sheet,
        chemicals=chemicals,
        column=column,
        breakthrough_parameters=breakthrough_parameters,
    )

    return {
        "water": water,
        "media": media,
        "column": column,
        "chemicals": chemicals,
        "breakthroughs": breakthroughs,
    }
