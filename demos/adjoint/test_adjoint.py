import pytest
from pathlib import Path
import taylor_test


@pytest.mark.parametrize("case_name", taylor_test.cases)
def test_rectangular_taylor_test(case_name):
    with open(Path(__file__).parent.resolve() / f"{case_name}.conv", "r") as f:
        minconv = float(f.read())

    assert minconv > 1.9
