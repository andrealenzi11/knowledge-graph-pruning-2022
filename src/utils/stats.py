import numpy as np


def print_statistics(scores: np.ndarray,
                     decimal_precision: int,
                     message: str = "scores"):
    print(f"{message}:",
          round(np.min(a=scores), decimal_precision),
          round(np.percentile(a=scores, q=25), decimal_precision),
          round(np.median(a=scores), decimal_precision),
          round(np.percentile(a=scores, q=75), decimal_precision),
          round(np.max(a=scores), decimal_precision),
          f"shape={scores.shape}", )


def get_center(scores: np.ndarray,
               use_median: bool) -> float:
    if use_median:
        return float(np.median(a=scores))
    else:
        return float(np.mean(a=scores))
