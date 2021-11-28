import pytest

import sys
from pathlib import Path

import pandas as pd

cases = [
    "2D_cartesian/base_case",
    "2D_cartesian/compressible_case",
    "2D_cartesian/viscoplastic_case",
    "2D_cylindrical",
    "3D_cartesian",
    "3D_spherical",
]


def get_convergence(base):
    return pd.read_csv(base / "params.log", sep="\\s+", header=0).iloc[-1]


@pytest.mark.parametrize("benchmark", cases)
def test_benchmark(benchmark):
    b = Path(__file__).parent.resolve() / benchmark
    df = get_convergence(b)
    expected = pd.read_pickle(b / "expected.pkl")

    pd.testing.assert_series_equal(df[["u_rms", "nu_top"]], expected)


if __name__ == "__main__":
    if sys.argv[1:]:
        cases = set(cases).intersection(sys.argv[1:])

    for case in cases:
        b = Path(__file__).parent.resolve() / case
        df = get_convergence(b)[["u_rms", "nu_top"]]

        df.to_pickle(b / "expected.pkl")
