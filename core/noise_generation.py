import math
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import pandas as pd

from config.config import HEAD, RELATION, TAIL, FAKE_FLAG, TRAINING, VALIDATION, TESTING
from dao.data_model import NoisyDataset


# from sdv.tabular import CTGAN


class NoiseGenerator(ABC):

    def __init__(self,
                 training_df: pd.DataFrame,
                 validation_df: pd.DataFrame,
                 testing_df: pd.DataFrame):
        self.training_df = training_df[[HEAD, RELATION, TAIL]].astype("str").reset_index(drop=True)  # "category"
        self.validation_df = validation_df[[HEAD, RELATION, TAIL]].astype("str").reset_index(drop=True)  # "category"
        self.testing_df = testing_df[[HEAD, RELATION, TAIL]].astype("str").reset_index(drop=True)  # "category"

    @abstractmethod
    def generate_noisy_dataset(self, noise_percentage: int) -> NoisyDataset:
        pass


# class NeuralNoiseGenerator(NoiseGenerator):
#
#     def __init__(self,
#                  training_df: pd.DataFrame,
#                  validation_df: pd.DataFrame,
#                  testing_df: pd.DataFrame,
#                  models_folder_path: str,
#                  training_sample: Optional[int] = None,
#                  batch_size: int = 500,
#                  epochs: int = 300):
#         super().__init__(training_df=training_df,
#                          validation_df=validation_df,
#                          testing_df=testing_df)
#         self.models_folder_path = models_folder_path
#         self.training_sample = training_sample
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.model = CTGAN(
#             field_names=[HEAD, RELATION, TAIL],
#             field_types={
#                 HEAD: {'type': 'categorical'},
#                 RELATION: {'type': 'categorical'},
#                 TAIL: {'type': 'categorical'},
#             },
#             verbose=True,
#             cuda=True,
#             batch_size=self.batch_size,
#             epochs=self.epochs,
#         )
#         self.is_fitted = False
#
#     def train(self):
#         print("\n\t\t - Start Fitting on Data ...")
#         if self.training_sample and (self.training_sample < self.training_df.shape[0]):
#             training_df = self.training_df.copy().sample(self.training_sample).astype("str").reset_index(drop=True)
#         else:
#             training_df = self.training_df.copy().astype("str").reset_index(drop=True)
#         self.model.fit(data=training_df)
#         self.is_fitted = True
#         print("\t\t - End fitting! \n")
#
#     @staticmethod
#     def _normalize_str(model_name: str) -> str:
#         if model_name.endswith(".pkl"):
#             return model_name
#         else:
#             return f"{model_name}.pkl"
#
#     def _get_model_path(self, model_name: str) -> str:
#         return os.path.join(self.models_folder_path, self._normalize_str(model_name=model_name))
#
#     def store_model(self, model_name: str):
#         model_path = self._get_model_path(model_name=model_name)
#         self.model.save(model_path)
#
#     def load_model(self, model_name: str):
#         model_path = self._get_model_path(model_name=model_name)
#         if not os.path.isfile(model_path):
#             raise FileNotFoundError(f"model '{model_path}' not found on File System!")
#         self.model = CTGAN.load(model_path)
#         self.is_fitted = True
#
#     def _generate_noise(self,
#                         noise_percentage: int,
#                         partition_name: str,
#                         partition_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
#         partition_original_size = partition_df.shape[0]
#         partition_sample_size = int(math.ceil(partition_original_size / 100 * noise_percentage))
#         print(f"[noise_{noise_percentage}%]  |  "
#               f"{partition_name}_sample_size: {partition_sample_size}  | "
#               f"{partition_name}_original_size: {partition_original_size}")
#         partition_anomalies_df = self.model.sample(num_rows=partition_sample_size).reset_index(drop=True)
#         partition_final_df = pd.concat([partition_df, partition_anomalies_df],
#                                        axis=0,
#                                        ignore_index=True,
#                                        verify_integrity=True).reset_index(drop=True)
#         partition_fake_y = [0] * partition_original_size + [1] * partition_sample_size
#         partition_fake_series = pd.Series(data=partition_fake_y,
#                                           dtype=int,
#                                           name=FAKE_FLAG)
#         return partition_final_df, partition_fake_series
#
#     def generate_noisy_dataset(self, noise_percentage: int) -> NoisyDataset:
#         if not self.is_fitted:
#             raise Exception("Error: the CTGAN model is not already fitted on training data!")
#         training_final_df, training_y_fake = self._generate_noise(noise_percentage=noise_percentage,
#                                                                   partition_name=TRAINING,
#                                                                   partition_df=self.training_df)
#         validation_final_df, validation_y_fake = self._generate_noise(noise_percentage=noise_percentage,
#                                                                       partition_name=VALIDATION,
#                                                                       partition_df=self.validation_df)
#         testing_final_df, testing_y_fake = self._generate_noise(noise_percentage=noise_percentage,
#                                                                 partition_name=TESTING,
#                                                                 partition_df=self.testing_df)
#         return NoisyDataset(training_df=training_final_df,
#                             training_y_fake=training_y_fake,
#                             validation_df=validation_final_df,
#                             validation_y_fake=validation_y_fake,
#                             testing_df=testing_final_df,
#                             testing_y_fake=testing_y_fake)


class DeterministicNoiseGenerator(NoiseGenerator):

    def __init__(self,
                 training_df: pd.DataFrame,
                 validation_df: pd.DataFrame,
                 testing_df: pd.DataFrame,
                 random_state_head: Optional[int] = None,
                 random_state_relation: Optional[int] = None,
                 random_state_tail: Optional[int] = None):
        super().__init__(training_df=training_df,
                         validation_df=validation_df,
                         testing_df=testing_df)

        assert random_state_head != random_state_relation
        assert random_state_head != random_state_tail
        assert random_state_relation != random_state_tail
        self.random_state_head = random_state_head
        self.random_state_relation = random_state_relation
        self.random_state_tail = random_state_tail
        self.all_df = pd.concat(objs=[self.training_df, self.validation_df, self.testing_df],
                                axis=0,
                                join="outer",
                                ignore_index=True,
                                keys=None,
                                levels=None,
                                names=None,
                                verify_integrity=True,
                                sort=False,
                                copy=True).astype("str").reset_index(drop=True)
        assert (self.all_df.shape[1] == 3) and \
            (self.all_df.shape[0] == self.training_df.shape[0] + self.validation_df.shape[0] + self.testing_df.shape[0])
        self.all_df = self.all_df.sort_values(by=[HEAD, RELATION, TAIL],
                                              axis=0,
                                              ascending=True,
                                              inplace=False,
                                              ignore_index=True).astype("str").reset_index(drop=True)
        print(f"\t all_df shape: {self.all_df.shape}")
        self.all_df = self.all_df.drop_duplicates(keep="first",
                                                  inplace=False,
                                                  ignore_index=True).astype("str").reset_index(drop=True)
        print(f"\t all_df shape after drop duplicates: {self.all_df.shape}")
        assert (self.all_df.shape[1] == 3) and \
            (self.all_df.shape[0] <= self.training_df.shape[0] + self.validation_df.shape[0] + self.testing_df.shape[0])

    def _generate_noise(self,
                        noise_percentage: int,
                        partition_name: str,
                        partition_df: pd.DataFrame,
                        sampling_with_replacement_flag: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate a nosy dataframe;

        :param noise_percentage: (*int*) percentage number [0, 100]
        :param partition_name: (*str*) "training" | "validation" | "testing"
        :param partition_df: (*pd.DataFrame*) input dataframe with triples from which generate noise
        :param sampling_with_replacement_flag: (*bool*) boolean flag that indicate the sampling strategy.
                - True ==> the same element x could be sampled more times (es. from [1,2,3,4,5] we can sample 2,2,5,5)
                - False ==> once sampled an element x, we never resample x (es. from [1,2,3,4,5] we can sample 2,3,5,1)

        :return: (noisy dataframe, boolean fake series)
        """

        # Compute anomaly df size
        partition_original_size = partition_df.shape[0]
        partition_sample_size = int(math.ceil(partition_original_size / 100 * noise_percentage))
        assert partition_sample_size < partition_original_size
        print(f"[noise_{noise_percentage}%]  |  "
              f"{partition_name}_sample_size: {partition_sample_size}  | "
              f"{partition_name}_original_size: {partition_original_size}")

        print(f"\t\t original_df: {partition_df.shape}")

        # Create anomaly df, by sampling from original df
        head_sample = self.all_df[HEAD].sample(n=partition_sample_size,
                                               replace=sampling_with_replacement_flag,
                                               random_state=self.random_state_head).values
        relation_sample = self.all_df[RELATION].sample(n=partition_sample_size,
                                                       replace=sampling_with_replacement_flag,
                                                       random_state=self.random_state_relation).values
        tail_sample = self.all_df[TAIL].sample(n=partition_sample_size,
                                               replace=sampling_with_replacement_flag,
                                               random_state=self.random_state_tail).values
        partition_anomalies_df = pd.DataFrame(data={HEAD: head_sample,
                                                    RELATION: relation_sample,
                                                    TAIL: tail_sample}).reset_index(drop=True)
        print(f"\t\t anomalies_df: {partition_anomalies_df.shape}")
        assert head_sample.shape[0] == relation_sample.shape[0]
        assert head_sample.shape[0] == tail_sample.shape[0]
        assert relation_sample.shape[0] == tail_sample.shape[0]
        assert partition_anomalies_df.shape[0] == \
               head_sample.shape[0] == relation_sample.shape[0] == tail_sample.shape[0]
        assert partition_anomalies_df.shape[1] == 3
        assert partition_anomalies_df.shape[0] < partition_df.shape[0]

        # Append anomaly df to the original df
        partition_final_df = pd.concat([partition_df, partition_anomalies_df],
                                       axis=0,
                                       ignore_index=True,
                                       verify_integrity=True).reset_index(drop=True)
        print(f"\t\t final_df: {partition_final_df.shape}")
        partition_fake_y = [0] * partition_original_size + [1] * partition_sample_size
        assert len(partition_fake_y) == partition_original_size + partition_sample_size
        assert partition_final_df.shape[0] == len(partition_fake_y)
        partition_fake_series = pd.Series(data=partition_fake_y,
                                          dtype=int,
                                          name=FAKE_FLAG)
        return partition_final_df, partition_fake_series

    def generate_noisy_dataset(self, noise_percentage: int) -> NoisyDataset:
        # training
        training_final_df, training_y_fake = self._generate_noise(noise_percentage=noise_percentage,
                                                                  partition_name=TRAINING,
                                                                  partition_df=self.training_df,
                                                                  sampling_with_replacement_flag=True)
        assert training_final_df.shape[0] == training_y_fake.shape[0]
        assert training_final_df.shape[1] == 3
        # validation
        validation_final_df, validation_y_fake = self._generate_noise(noise_percentage=noise_percentage,
                                                                      partition_name=VALIDATION,
                                                                      partition_df=self.validation_df,
                                                                      sampling_with_replacement_flag=False)
        assert validation_final_df.shape[0] == validation_y_fake.shape[0]
        assert validation_final_df.shape[1] == 3
        # testing
        testing_final_df, testing_y_fake = self._generate_noise(noise_percentage=noise_percentage,
                                                                partition_name=TESTING,
                                                                partition_df=self.testing_df,
                                                                sampling_with_replacement_flag=False)
        assert testing_final_df.shape[0] == testing_y_fake.shape[0]
        assert testing_final_df.shape[1] == 3
        # Return the obtained NoisyDataset object
        return NoisyDataset(training_df=training_final_df,
                            training_y_fake=training_y_fake,
                            validation_df=validation_final_df,
                            validation_y_fake=validation_y_fake,
                            testing_df=testing_final_df,
                            testing_y_fake=testing_y_fake)
