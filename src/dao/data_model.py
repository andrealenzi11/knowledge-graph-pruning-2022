from dataclasses import dataclass

import pandas as pd


@dataclass
class NoisyDataset:
    """
        Dataset with a percentage of noisy observations
    """
    training_df: pd.DataFrame
    training_y_fake: pd.Series
    validation_df: pd.DataFrame
    validation_y_fake: pd.Series
    testing_df: pd.DataFrame
    testing_y_fake: pd.Series

    def __post_init__(self):
        # training
        assert isinstance(self.training_df, pd.DataFrame)
        assert isinstance(self.training_y_fake, pd.Series)
        assert self.training_df.shape[0] == self.training_y_fake.shape[0]
        assert self.training_df.shape[1] == 3
        assert all([y == 1 or y == 0 for y in self.training_y_fake])
        # validation
        assert isinstance(self.validation_df, pd.DataFrame)
        assert isinstance(self.validation_y_fake, pd.Series)
        assert self.validation_df.shape[0] == self.validation_y_fake.shape[0]
        assert self.validation_df.shape[1] == 3
        assert all([y == 1 or y == 0 for y in self.validation_y_fake])
        # testing
        assert isinstance(self.testing_df, pd.DataFrame)
        assert isinstance(self.testing_y_fake, pd.Series)
        assert self.testing_df.shape[0] == self.testing_y_fake.shape[0]
        assert self.testing_df.shape[1] == 3
        assert all([y == 1 or y == 0 for y in self.testing_y_fake])


@dataclass
class RandomDataset:
    """
        Total Random Dataset to use as baseline
    """
    training_df: pd.DataFrame
    training_y_fake: pd.Series
    validation_df: pd.DataFrame
    validation_y_fake: pd.Series
    testing_df: pd.DataFrame
    testing_y_fake: pd.Series

    def __post_init__(self):
        # training
        assert isinstance(self.training_df, pd.DataFrame)
        assert isinstance(self.training_y_fake, pd.Series)
        assert self.training_df.shape[0] == self.training_y_fake.shape[0]
        assert self.training_df.shape[1] == 3
        assert all([y == 1 for y in self.training_y_fake])
        # validation
        assert isinstance(self.validation_df, pd.DataFrame)
        assert isinstance(self.validation_y_fake, pd.Series)
        assert self.validation_df.shape[0] == self.validation_y_fake.shape[0]
        assert self.validation_df.shape[1] == 3
        assert all([y == 1 for y in self.validation_y_fake])
        # testing
        assert isinstance(self.testing_df, pd.DataFrame)
        assert isinstance(self.testing_y_fake, pd.Series)
        assert self.testing_df.shape[0] == self.testing_y_fake.shape[0]
        assert self.testing_df.shape[1] == 3
        assert all([y == 1 for y in self.testing_y_fake])
