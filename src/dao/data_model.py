from dataclasses import dataclass

import pandas as pd


@dataclass
class NoisyDataset:
    training_df: pd.DataFrame
    training_y_fake: pd.Series
    validation_df: pd.DataFrame
    validation_y_fake: pd.Series
    testing_df: pd.DataFrame
    testing_y_fake: pd.Series
