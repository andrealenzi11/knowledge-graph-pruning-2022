from typing import Tuple, List

import pandas as pd
from sdv.tabular import GaussianCopula

from dao.data_model import NoisyDataset


class NoiseGenerator:

    def __init__(self,
                 training_df: pd.DataFrame,
                 validation_df: pd.DataFrame,
                 testing_df: pd.DataFrame):
        self.training_df = training_df.reset_index(drop=True)
        self.validation_df = validation_df.reset_index(drop=True)
        self.testing_df = testing_df.reset_index(drop=True)
        self.model = GaussianCopula()

    def train(self):
        self.model.fit(self.training_df)

    def _generate_noise(self,
                        noise_percentage: int,
                        partition_name: str,
                        partition_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        partition_original_size = partition_df.shape[0]
        partition_sample_size = int(partition_original_size / 100 * noise_percentage)
        print(f"[noise_{noise_percentage}%] {partition_name}_sample_size: {partition_sample_size}")
        partition_anomalies_df = self.model.sample(num_rows=partition_sample_size).reset_index(drop=True)
        partition_final_df = pd.concat([partition_df, partition_anomalies_df],
                                       axis=0,
                                       ignore_index=True,
                                       verify_integrity=True).reset_index(drop=True)
        partition_fake_y = [0] * partition_original_size + [1] * partition_sample_size
        return partition_final_df, partition_fake_y

    def generate_noisy_dataset(self, noise_percentage: int) -> NoisyDataset:
        training_final_df, training_fake_y = self._generate_noise(noise_percentage=noise_percentage,
                                                                  partition_name="training",
                                                                  partition_df=self.training_df)
        validation_final_df, validation_fake_y = self._generate_noise(noise_percentage=noise_percentage,
                                                                      partition_name="validation",
                                                                      partition_df=self.validation_df)
        testing_final_df, testing_fake_y = self._generate_noise(noise_percentage=noise_percentage,
                                                                partition_name="testing",
                                                                partition_df=self.testing_df)
        return NoisyDataset(training_df=training_final_df,
                            training_fake_y=training_fake_y,
                            validation_df=validation_final_df,
                            validation_fake_y=validation_fake_y,
                            testing_df=testing_final_df,
                            testing_fake_y=testing_fake_y)
