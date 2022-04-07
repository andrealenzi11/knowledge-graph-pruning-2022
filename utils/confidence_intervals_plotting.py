from math import sqrt
from typing import Sequence, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np


def _plot_confidence_interval(x_position_index: int,
                              center: float,
                              lo: float,
                              hi: float,
                              line_color: str = "#2187bb",
                              point_color: str = "#f44336",
                              horizontal_line_width=0.25) -> Tuple[float, float, float]:
    left = x_position_index - horizontal_line_width / 2
    right = x_position_index + horizontal_line_width / 2
    plt.plot([x_position_index, x_position_index], [lo, hi], color=line_color)
    plt.plot([left, right], [lo, lo], color=line_color)
    plt.plot([left, right], [hi, hi], color=line_color)
    plt.plot(x_position_index, center, 'o', color=point_color)
    return center, lo, hi


def plot_confidence_interval_mean(x_position_index: int,
                                  values: Sequence[float],
                                  z: float = 1.96,   # 1.645: 90%  |  1.96: 95%  |  2.33: 98%  |  2.575: 99%
                                  line_color: str = "#2187bb",
                                  point_color: str = "#f44336",
                                  horizontal_line_width=0.25) -> Tuple[float, float, float]:
    mean = float(np.mean(values))
    stdev = float(np.std(values, ddof=1))
    confidence_interval = z * stdev / sqrt(len(values))
    lo = mean - confidence_interval
    hi = mean + confidence_interval
    return _plot_confidence_interval(x_position_index=x_position_index,
                                     center=mean,
                                     lo=lo,
                                     hi=hi,
                                     line_color=line_color,
                                     point_color=point_color,
                                     horizontal_line_width=horizontal_line_width)


def plot_percentile_interval_median(x_position_index: int,
                                    values: Sequence[float],
                                    percentile_min: float = 25,
                                    percentile_max: float = 75,
                                    line_color: str = "#2187bb",
                                    point_color: str = "#f44336",
                                    horizontal_line_width=0.25) -> Tuple[float, float, float]:
    median = float(np.median(values))
    lo = float(np.percentile(a=values, q=percentile_min))
    hi = float(np.percentile(a=values, q=percentile_max))
    return _plot_confidence_interval(x_position_index=x_position_index,
                                     center=median,
                                     lo=lo,
                                     hi=hi,
                                     line_color=line_color,
                                     point_color=point_color,
                                     horizontal_line_width=horizontal_line_width)


def plot_confidences_intervals(label_values_map: Dict[str, Sequence[float]],
                               title: str,
                               use_median: bool = True,
                               use_mean: bool = False,
                               percentile_min: float = 25,
                               percentile_max: float = 75,
                               z: float = 1.96,
                               line_color: str = "#2187bb",
                               point_color: str = "#f44336",
                               horizontal_line_width=0.25,
                               round_digits: int = 4):
    if use_median and use_mean:
        raise ValueError("You cannot set both 'use_median' and 'use_mean' parameters to True")
    if (not use_median) and (not use_mean):
        raise ValueError("You must set to True one parameter among 'use_median' and 'use_mean'!")
    assert use_mean != use_median
    labels = list(label_values_map.keys())
    values_sequences = list(label_values_map.values())
    num_intervals = len(labels)
    x_ticks = [int(x) for x in range(1, num_intervals + 1, 1)]
    plt.xticks(x_ticks, labels)
    plt.title(title.title().strip())
    for i, values in zip(x_ticks, values_sequences):
        if use_median:
            center, lo, hi = plot_percentile_interval_median(x_position_index=i,
                                                             values=values,
                                                             percentile_min=percentile_min,
                                                             percentile_max=percentile_max,
                                                             line_color=line_color,
                                                             point_color=point_color,
                                                             horizontal_line_width=horizontal_line_width)
        elif use_mean:
            center, lo, hi = plot_confidence_interval_mean(x_position_index=i,
                                                           values=values,
                                                           z=z,
                                                           line_color=line_color,
                                                           point_color=point_color,
                                                           horizontal_line_width=horizontal_line_width)
        else:
            raise ValueError("Invalid 'use_median' and 'use_mean' flags!")
        print(f"\t interval {i}: "
              f"lo={round(lo, round_digits)} | "
              f"center={round(center, round_digits)} | "
              f"hi={round(hi, round_digits)} ")
    plt.show()
    plt.close()


if __name__ == '__main__':
    # examples: median and Inter Quartile Range (IQR)
    plot_confidences_intervals(
        label_values_map={
            "col_A": [4, 5, 1, 7, 12, 0, 3, 8, 15, 1],
            "col_B": [3, 9, 21, 7, 20, 0, 0, 2, 1, 13],
            "col_C": [4, 20, 21, 2, 7, 9, 3, 2, 15, 18],
        },
        title="examlple",
        use_median=True,
        use_mean=False,
        percentile_min=25,
        percentile_max=75,
        line_color="#2187bb",
        point_color="#f44336",
        horizontal_line_width=0.25,
        round_digits=4
    )

    # examples: mean and confidence interval
    plot_confidences_intervals(
        label_values_map={
            "col_A": [4, 5, 1, 7, 12, 0, 3, 8, 15, 1],
            "col_B": [3, 9, 21, 7, 20, 0, 0, 2, 1, 13],
            "col_C": [4, 20, 21, 2, 7, 9, 3, 2, 15, 18],
        },
        title="examlple",
        use_median=False,
        use_mean=True,
        z=1.96,
        line_color="#2187bb",
        point_color="#f44336",
        horizontal_line_width=0.25,
        round_digits=4
    )
