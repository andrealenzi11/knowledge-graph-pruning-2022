from dataclasses import dataclass
from enum import Enum
from typing import List

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
    training_fake_y: List[int]
    validation_df: pd.DataFrame
    validation_fake_y: List[int]
    testing_df: pd.DataFrame
    testing_fake_y: List[int]
