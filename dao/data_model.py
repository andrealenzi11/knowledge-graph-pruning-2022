from dataclasses import dataclass
from enum import Enum

import pandas as pd


class DatasetName(Enum):
    FB15K237: str = "FB15K237"
    WN18RR: str = "WN18RR"
    YAGO310: str = "YAGO310"
    COUNTRIES: str = "COUNTRIES"


class DatasetPartition(Enum):
    TRAINING: str = "training"
    VALIDATION: str = "validation"
    TESTING: str = "testing"


class NoiseLevel(Enum):
    ZERO: str = "original"
    ONE: str = "noise_1"
    FIVE: str = "noise_5"
    TEN: str = "noise_10"


@dataclass
class NoisyDataset:
    training_df: pd.DataFrame
    training_y_fake: pd.Series
    validation_df: pd.DataFrame
    validation_y_fake: pd.Series
    testing_df: pd.DataFrame
    testing_y_fake: pd.Series
