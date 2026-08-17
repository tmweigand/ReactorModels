"""tests/test_examples.py"""

from pathlib import Path
import subprocess
import sys

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

EXAMPLES = sorted(EXAMPLES_DIR.rglob("*.py"))


@pytest.mark.parametrize(
    "example",
    EXAMPLES,
    ids=lambda x: str(x.relative_to(EXAMPLES_DIR).with_suffix("")),
)
def test_example_runs(example):
    subprocess.run(
        [sys.executable, str(example)],
        check=True,
    )
