"""Utilities for loading input/output data files."""

from pathlib import Path

import numpy as np
import openpyxl


def load_input_output_file(raw_data_file):
    """Load a CSV, TXT, or Excel file and return the data as a NumPy array."""
    raw_data_file = Path(raw_data_file)
    ext = raw_data_file.suffix.lower()

    if ext == ".csv":
        return np.loadtxt(raw_data_file, delimiter=",", dtype=None, encoding="utf-8")

    if ext == ".txt":
        return np.loadtxt(raw_data_file, dtype=None, encoding="utf-8")

    if ext in {".xlsx", ".xlsm"}:
        workbook = openpyxl.load_workbook(raw_data_file, data_only=True)
        sheet = workbook.active

        data = []
        for row in sheet.iter_rows(values_only=True):
            if any(cell is not None for cell in row):
                data.append(list(row))

        return np.array(data, dtype=object)

    if ext == ".xls":
        raise ValueError(
            ".xls files are not supported. Save the file as .xlsx or .csv."
        )

    raise ValueError(f"Unsupported extension: {ext}")
