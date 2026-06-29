"""Helpers for inspecting initial concentrations from result a-matrixes."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import xarray as xr


def initial_concentration_table(
    result: object,
    *,
    normalize_initial_concentration: bool = False,
    omit: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create a dataset-by-species table of initial concentrations.

    The table contains one row per dataset and one column per species found in
    any ``a_matrix_*`` variable. If the same species occurs in multiple
    a-matrix blocks for one dataset, the values are summed.
    """

    result_map = _result_dataset_mapping(result)
    omitted_species = {str(species_name) for species_name in (omit or [])}
    species_order: list[str] = []
    rows: list[dict[str, float | str]] = []

    for dataset_name, result_dataset in result_map.items():
        row: dict[str, float | str] = {"dataset": dataset_name}

        for var_name in result_dataset.data_vars:
            if not var_name.startswith("a_matrix_"):
                continue

            mc_suffix = var_name.removeprefix("a_matrix_")
            species_coord = f"species_{mc_suffix}"
            initial_coord = f"initial_concentration_{mc_suffix}"
            if species_coord not in result_dataset[var_name].coords:
                continue
            if initial_coord not in result_dataset[var_name].coords:
                continue

            species = [
                str(species_name)
                for species_name in result_dataset[var_name].coords[species_coord].values.tolist()
            ]
            initial_values = result_dataset[var_name].coords[initial_coord].values.tolist()

            for species_name, initial_value in zip(species, initial_values, strict=False):
                if species_name in omitted_species:
                    continue
                value = float(initial_value)
                row[species_name] = float(row.get(species_name, 0.0)) + value
                if species_name not in species_order:
                    species_order.append(species_name)

        numeric_species = [species_name for species_name in species_order if species_name in row]
        if normalize_initial_concentration and numeric_species:
            total = sum(float(row[species_name]) for species_name in numeric_species)
            if total != 0:
                for species_name in numeric_species:
                    row[species_name] = float(row[species_name]) / total

        rows.append(row)

    if not rows:
        return pd.DataFrame(index=pd.Index([], name="dataset"))

    table = pd.DataFrame.from_records(rows).set_index("dataset")
    if species_order:
        table = table.reindex(columns=species_order, fill_value=0.0)
    return table.fillna(0.0)


def selected_initial_concentration_table(
    result: object,
    species: Sequence[str],
    *,
    multipliers: Mapping[str, float] | Sequence[float] | float | None = None,
    normalize_initial_concentration: bool = False,
    omit: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create a selected initial-concentration table, optionally scaled.

    Parameters
    ----------
    result:
        Result-like object accepted by ``result_dataset_mapping``.
    species:
        Species labels to keep as columns in the returned table.
    multipliers:
        Optional per-species scaling applied after selection. This may be a
        mapping keyed by species label, a sequence aligned with ``species``, or
        one scalar applied to every selected species.
    normalize_initial_concentration:
        Whether to normalize per-dataset totals before selection.
    """

    selected_species = [str(species_name) for species_name in species]
    table = initial_concentration_table(
        result,
        normalize_initial_concentration=normalize_initial_concentration,
        omit=omit,
    )
    if table.empty:
        return table.reindex(columns=selected_species, fill_value=0.0)

    selected_table = table.reindex(columns=selected_species, fill_value=0.0).copy()
    multiplier_map = _multiplier_mapping(selected_species, multipliers)
    for species_name, multiplier in multiplier_map.items():
        selected_table[species_name] = selected_table[species_name] * multiplier
    return selected_table


def _multiplier_mapping(
    species: Sequence[str],
    multipliers: Mapping[str, float] | Sequence[float] | float | None,
) -> dict[str, float]:
    if multipliers is None:
        return {species_name: 1.0 for species_name in species}

    if isinstance(multipliers, Mapping):
        return {
            species_name: float(multipliers.get(species_name, 1.0)) for species_name in species
        }

    if isinstance(multipliers, int | float):
        return {species_name: float(multipliers) for species_name in species}

    if len(multipliers) != len(species):
        raise ValueError("multipliers must match the length of species")

    return {
        species_name: float(multiplier)
        for species_name, multiplier in zip(species, multipliers, strict=False)
    }


def _result_dataset_mapping(result: object) -> Mapping[str, xr.Dataset]:
    if isinstance(result, xr.Dataset):
        return {"dataset": result}

    if isinstance(result, Mapping):
        return {str(key): _as_dataset(value) for key, value in result.items()}

    if isinstance(result, Sequence) and not isinstance(result, str | bytes | Path):
        return {f"dataset{index}": _as_dataset(value) for index, value in enumerate(result)}

    data_mapping = getattr(result, "data", None)
    if isinstance(data_mapping, Mapping):
        return {str(key): _as_dataset(value) for key, value in data_mapping.items()}

    raise TypeError("result must be a dataset, mapping, sequence, or expose a .data mapping")


def _as_dataset(value: object) -> xr.Dataset:
    if isinstance(value, xr.Dataset):
        return value
    raise TypeError("initial concentration helpers require xarray.Dataset inputs")
