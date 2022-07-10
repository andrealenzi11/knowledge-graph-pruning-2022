from typing import Dict, Sequence

import numpy as np
import pandas as pd
from tabulate import tabulate


def _compute_stats(scores_vector: np.ndarray,
                   decimal_precision: int) -> Dict[str, float]:
    assert scores_vector.ndim == 1
    assert isinstance(decimal_precision, int)
    return {
        "size": int(scores_vector.shape[0]),
        "mean": round(float(np.mean(a=scores_vector)), decimal_precision),
        "std": round(float(np.std(a=scores_vector, ddof=0)), decimal_precision),
        "min": round(float(np.min(a=scores_vector)), decimal_precision),
        "1q": round(float(np.percentile(a=scores_vector, q=25)), decimal_precision),
        "median": round(float(np.median(a=scores_vector)), decimal_precision),
        "3q": round(float(np.percentile(a=scores_vector, q=75)), decimal_precision),
        "max": round(float(np.max(a=scores_vector)), decimal_precision),
    }


def print_2d_statistics(scores_matrix: Sequence[np.ndarray],
                        labels: Sequence[str],
                        decimal_precision: int):
    """
    Print 2-Dimensions statistics

    :param scores_matrix: (*Sequence[np.ndarray]*) sequence of 1D numpy arrays with numerical values
    :param labels: (*Sequence[str]*) sequence of rows labels
    :param decimal_precision: (*int*) number of decimal digits to consider

    """
    assert all([scores_vector.ndim == 1 for scores_vector in scores_matrix])
    assert all([isinstance(xi, str) for xi in labels])
    assert len(labels) == len(scores_matrix)
    assert isinstance(decimal_precision, int)
    records = [_compute_stats(scores_vector=scores_vector,
                              decimal_precision=decimal_precision)
               for scores_vector in scores_matrix]
    df_stats = pd.DataFrame(
        data=records,
        index=labels,
    )
    print(tabulate(tabular_data=df_stats, headers="keys"))


def print_1d_statistics(scores_vector: np.ndarray,
                        label: str,
                        decimal_precision: int):
    """
    Print 1-Dimension statistics

    :param scores_vector: (*np.ndarray*) single 1D numpy array with numerical values
    :param label: (*str*) single row label
    :param decimal_precision: (*int*) number of decimal digits to consider

    """
    assert scores_vector.ndim == 1
    assert isinstance(label, str)
    assert isinstance(decimal_precision, int)
    df_stats = pd.DataFrame(
        data=[_compute_stats(scores_vector=scores_vector, decimal_precision=decimal_precision)],
        index=[label],
    )
    print(tabulate(tabular_data=df_stats, headers="keys"))


def get_center(scores: np.ndarray,
               use_median: bool = False) -> float:
    assert scores.ndim == 1
    assert isinstance(use_median, bool)
    if use_median:
        return float(np.median(a=scores))
    else:
        return float(np.mean(a=scores))


def find_maximum(label_values_map: Dict[str, Sequence[float]]) -> float:
    max_values = []
    for k, v in label_values_map.items():
        assert v
        max_values.append(max(v))
    return max(max_values)


def find_minimum(label_values_map: Dict[str, Sequence[float]]) -> float:
    max_values = []
    for k, v in label_values_map.items():
        assert v
        max_values.append(min(v))
    return min(max_values)
