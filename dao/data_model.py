from enum import Enum


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
