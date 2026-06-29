from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from pygta_local_extras import simulate_from_fitted_data
from pygta_local_extras.analysis import (
    simulate_from_fitted_data as analysis_simulate_from_fitted_data,
)


class _Result:
    def __init__(self, data: dict[str, xr.Dataset]) -> None:
        self.data = data


def test_simulate_from_fitted_data_is_exported() -> None:
    assert callable(simulate_from_fitted_data)
    assert callable(analysis_simulate_from_fitted_data)


def test_simulate_from_fitted_data_writes_poisson_wavelength_explicit_ascii(
    tmp_path: Path,
) -> None:
    fitted = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float),
        dims=("time", "spectral"),
        coords={"time": [0.0, 1.0, 2.0], "spectral": [500.0, 510.0]},
        name="fitted_data",
    )
    result = _Result({"sample": fitted.to_dataset()})

    simulated = simulate_from_fitted_data(
        result,
        output_dir=tmp_path,
        rng=np.random.default_rng(123),
    )

    assert list(simulated) == ["samplesim0"]
    simulated_dataset = simulated["samplesim0"]
    assert simulated_dataset["data"].dtype.kind in {"i", "u"}

    output_path = tmp_path / "samplesim0.ascii"
    assert output_path.exists()
    contents = output_path.read_text().splitlines()
    assert contents[2] == "Wavelength explicit"
    assert contents[3] == "Intervalnr 2"

    exported_rows = [list(map(float, row.split("\t"))) for row in contents[5:]]
    assert len(exported_rows) == 3
    assert [row[0] for row in exported_rows] == [0.0, 1.0, 2.0]
    assert all(float(value).is_integer() for row in exported_rows for value in row[1:])


def test_simulate_from_fitted_data_writes_normal_time_explicit_ascii(tmp_path: Path) -> None:
    fitted_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    fitted = xr.DataArray(
        fitted_values,
        dims=("time", "spectral"),
        coords={"time": [0.0, 1.0], "spectral": [500.0, 510.0, 520.0]},
        name="fitted_data",
    )
    result = _Result({"trace": fitted.to_dataset()})
    rng = np.random.default_rng(7)
    expected = fitted_values + np.random.default_rng(7).normal(0.0, 1e-3, size=fitted_values.shape)

    simulated = simulate_from_fitted_data(
        result,
        poisson=False,
        sigma=1e-3,
        output_dir=tmp_path,
        rng=rng,
    )

    np.testing.assert_allclose(simulated["tracesim1e-3"]["data"].to_numpy(), expected)

    output_path = tmp_path / "tracesim1e-3.ascii"
    assert output_path.exists()
    contents = output_path.read_text().splitlines()
    assert contents[2] == "Time explicit"
    assert contents[3] == "Intervalnr 2"

    exported_rows = [list(map(float, row.split("\t"))) for row in contents[5:]]
    assert len(exported_rows) == 3
    assert [row[0] for row in exported_rows] == [500.0, 510.0, 520.0]
    np.testing.assert_allclose(
        np.array([row[1:] for row in exported_rows]),
        expected.T,
    )
