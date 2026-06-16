import numpy as np
from reactormodels import load_input_output_file


def test_load_csv_file(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("1,2\n3,4\n")
    result = load_input_output_file(file_path)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))


def test_load_txt_file(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("1 2\n3 4\n")
    result = load_input_output_file(file_path)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))


def test_unsupported_extension_raises(tmp_path):
    file_path = tmp_path / "data.bad"
    file_path.write_text("1 2\n")
    try:
        load_input_output_file(file_path)
    except ValueError as error:
        assert "Unsupported extension" in str(error)
    else:
        raise AssertionError("Expected ValueError")
