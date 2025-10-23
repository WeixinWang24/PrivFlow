"""Built-in dummy datasets for demonstrating PrivFlow pipelines."""

from __future__ import annotations

from importlib import resources
from importlib.abc import Traversable
from typing import List, Tuple, Union

import csv

from .._deps import np


def _read_csv_matrix(path: Traversable) -> Tuple[List[List[float]], List[int]]:
    features: List[List[float]] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        _ = next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            *x_vals, y_val = row
            features.append([float(val) for val in x_vals])
            labels.append(int(y_val))
    return features, labels


def load_dummy_classification(
    as_numpy: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[List[float]], List[int]]]:
    """Return a small binary classification dataset bundled with PrivFlow."""
    data_path = resources.files(__package__).joinpath("dummy_classification.csv")
    features, labels = _read_csv_matrix(data_path)
    if as_numpy:
        return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    return features, labels

