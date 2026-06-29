"""Tests for initial concentration inspection helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr
from pygta_local_extras.analysis import initial_concentration_table
from pygta_local_extras.analysis import selected_initial_concentration_table


def _make_a_matrix_dataset(
    *,
    megacomplexes: list[tuple[str, list[str], list[float]]],
) -> xr.Dataset:
    data_vars: dict[str, xr.DataArray] = {}
    for mc_label, species, initial_concentration in megacomplexes:
        component_name = f"component_{mc_label}"
        species_name = f"species_{mc_label}"
        size = len(species)
        data_vars[f"a_matrix_{mc_label}"] = xr.DataArray(
            np.eye(size),
            dims=(component_name, species_name),
            coords={
                component_name: np.arange(size),
                species_name: species,
                f"initial_concentration_{mc_label}": (species_name, initial_concentration),
                f"lifetime_{mc_label}": (component_name, np.ones(size)),
            },
        )
    return xr.Dataset(data_vars)


def test_initial_concentration_table_collects_dataset_rows() -> None:
    result = {
        "open": _make_a_matrix_dataset(
            megacomplexes=[
                ("mc1", ["s1", "s5"], [95.0, 35.0]),
                ("mc2", ["s8"], [12.0]),
            ]
        ),
        "close": _make_a_matrix_dataset(
            megacomplexes=[
                ("mc1", ["s1", "s5"], [190.0, 70.0]),
            ]
        ),
    }

    table = initial_concentration_table(result)

    assert list(table.index) == ["open", "close"]
    assert list(table.columns) == ["s1", "s5", "s8"]
    assert table.loc["open", "s1"] == 95.0
    assert table.loc["open", "s5"] == 35.0
    assert table.loc["open", "s8"] == 12.0
    assert table.loc["close", "s1"] == 190.0
    assert table.loc["close", "s5"] == 70.0
    assert table.loc["close", "s8"] == 0.0


def test_initial_concentration_table_omits_species() -> None:
    result = {
        "open": _make_a_matrix_dataset(
            megacomplexes=[
                ("mc1", ["s1", "oscatfoo", "cscatfoo", "s5"], [95.0, 1.0, 2.0, 35.0]),
            ]
        ),
    }

    table = initial_concentration_table(result, omit=["oscatfoo", "cscatfoo"])

    assert list(table.columns) == ["s1", "s5"]
    assert table.loc["open", "s1"] == 95.0
    assert table.loc["open", "s5"] == 35.0


def test_selected_initial_concentration_table_applies_multipliers() -> None:
    result = {
        "open": _make_a_matrix_dataset(megacomplexes=[("mc1", ["s1", "s5"], [95.0, 35.0])]),
        "close": _make_a_matrix_dataset(megacomplexes=[("mc1", ["s1", "s5"], [190.0, 70.0])]),
    }

    table = selected_initial_concentration_table(
        result,
        ["s1", "s5"],
        multipliers={"s1": 1 / 95, "s5": 1 / 35},
    )

    assert list(table.columns) == ["s1", "s5"]
    assert table.loc["open", "s1"] == 1.0
    assert table.loc["open", "s5"] == 1.0
    assert table.loc["close", "s1"] == 2.0
    assert table.loc["close", "s5"] == 2.0


def test_selected_initial_concentration_table_passes_omit() -> None:
    result = {
        "open": _make_a_matrix_dataset(
            megacomplexes=[("mc1", ["s1", "oscatfoo", "s5"], [95.0, 1.0, 35.0])]
        ),
    }

    table = selected_initial_concentration_table(
        result,
        ["s1", "oscatfoo", "s5"],
        omit=["oscatfoo"],
    )

    assert list(table.columns) == ["s1", "oscatfoo", "s5"]
    assert table.loc["open", "s1"] == 95.0
    assert table.loc["open", "oscatfoo"] == 0.0
    assert table.loc["open", "s5"] == 35.0
