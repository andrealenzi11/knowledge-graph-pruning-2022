import math
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from numpy import random

from src.config.config import HEAD, RELATION, TAIL, FAKE_FLAG, TRAINING, VALIDATION, TESTING
from src.dao.data_model import NoisyDataset, RandomDataset


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
                 random_states_training: Tuple[int, int, int],
                 random_states_validation: Tuple[int, int, int],
                 random_states_testing: Tuple[int, int, int]):

        super().__init__(training_df=training_df,
                         validation_df=validation_df,
                         testing_df=testing_df)

        # Check on Random States
        assert len(random_states_training) == 3
        assert len(random_states_validation) == 3
        assert len(random_states_testing) == 3
        assert random_states_training != random_states_validation
        assert random_states_training[0] != random_states_validation[0]
        assert random_states_training[1] != random_states_validation[1]
        assert random_states_training[2] != random_states_validation[2]
        assert random_states_validation != random_states_testing
        assert random_states_validation[0] != random_states_testing[0]
        assert random_states_validation[1] != random_states_testing[1]
        assert random_states_validation[2] != random_states_testing[2]
        assert random_states_training != random_states_testing
        assert random_states_training[0] != random_states_testing[0]
        assert random_states_training[1] != random_states_testing[1]
        assert random_states_training[2] != random_states_testing[2]
        valid_random_states = set()
        for rs_triple in (
                random_states_training,
                random_states_validation,
                random_states_testing,
        ):
            for rs_value in rs_triple:
                assert isinstance(rs_value, int)
                valid_random_states.add(rs_value)
        assert len(valid_random_states) == 3 * 3

        # Fields
        self.random_states_training = random_states_training
        self.random_states_validation = random_states_validation
        self.random_states_testing = random_states_testing
        self.valid_random_states = valid_random_states
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
        assert self.all_df.shape[1] == 3
        assert self.all_df.shape[0] == \
               self.training_df.shape[0] + self.validation_df.shape[0] + self.testing_df.shape[0]
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
        assert self.all_df.shape[1] == 3
        assert self.all_df.shape[0] <= \
               self.training_df.shape[0] + self.validation_df.shape[0] + self.testing_df.shape[0]

    def _generate_noise(self,
                        noise_percentage: int,
                        partition_name: str,
                        partition_df: pd.DataFrame,
                        sampling_with_replacement_flag: bool,
                        random_state_head: int,
                        random_state_relation: int,
                        random_state_tail: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate a noisy dataframe;

        :param noise_percentage: (*int*) percentage number [0, 100]
        :param partition_name: (*str*) "training" | "validation" | "testing"
        :param partition_df: (*pd.DataFrame*) input dataframe with triples from which generate noise
        :param sampling_with_replacement_flag: (*bool*) boolean flag that indicate the sampling strategy.
                - True ==> the same element x could be sampled more times (es. from [1,2,3,4,5] we can sample 2,2,5,5)
                - False ==> once sampled an element x, we never resample x (es. from [1,2,3,4,5] we can sample 2,3,5,1)

        :return: (noisy dataframe, boolean fake series)
        """
        assert random_state_head != random_state_relation
        assert random_state_relation != random_state_tail
        assert random_state_head != random_state_relation
        print(f"\n[noise_{noise_percentage}%]  {partition_name}")
        initial_shape = partition_df.shape
        partition_df = partition_df.astype(str).drop_duplicates(keep="first").reset_index(drop=True)
        assert initial_shape == partition_df.shape

        # ===== Compute anomaly df size ==== #
        if noise_percentage == 100:
            # special case: half of positives (true triples) and half of negatives (fake synthetic triples)
            partition_original_size = partition_df.shape[0]
            partition_sample_size = partition_df.shape[0]
            assert partition_sample_size == partition_original_size
        else:
            # normal case: positives (true triples) and a limited percentage of negatives (fake synthetic triples)
            partition_original_size = partition_df.shape[0]
            partition_sample_size = int(math.ceil(partition_original_size / 100 * noise_percentage))
            assert partition_sample_size < partition_original_size

        print(f"\t\t original_df: {partition_df.shape}")

        # ===== Create anomaly df, by sampling from original df ===== #
        head_sample = self.all_df[HEAD].sample(n=partition_sample_size,
                                               replace=sampling_with_replacement_flag,
                                               random_state=random_state_head).values
        relation_sample = self.all_df[RELATION].sample(n=partition_sample_size,
                                                       replace=sampling_with_replacement_flag,
                                                       random_state=random_state_relation).values
        tail_sample = self.all_df[TAIL].sample(n=partition_sample_size,
                                               replace=sampling_with_replacement_flag,
                                               random_state=random_state_tail).values
        partition_anomalies_df = pd.DataFrame(data={HEAD: head_sample,
                                                    RELATION: relation_sample,
                                                    TAIL: tail_sample}).reset_index(drop=True).astype(str)
        print(f"\t\t anomalies_df: {partition_anomalies_df.shape}")
        assert head_sample.shape[0] == relation_sample.shape[0]
        assert head_sample.shape[0] == tail_sample.shape[0]
        assert relation_sample.shape[0] == tail_sample.shape[0]
        assert partition_anomalies_df.shape[0] == \
               head_sample.shape[0] == relation_sample.shape[0] == tail_sample.shape[0]
        assert partition_anomalies_df.shape[1] == 3
        if noise_percentage == 100:
            assert partition_anomalies_df.shape[0] == partition_df.shape[0]
        else:
            assert partition_anomalies_df.shape[0] < partition_df.shape[0]

        # ===== Manage duplicates introduced by the previous sampling ===== #
        partition_anomalies_df = partition_anomalies_df.drop_duplicates(keep="first").reset_index(drop=True)
        print(f"\t\t anomalies_df after drop duplicates: {partition_anomalies_df.shape}")
        partition_anomalies_triples = {tuple(triple)
                                       for triple in partition_anomalies_df.to_dict(orient="split")["data"]}
        partition_original_triples = {tuple(triple)
                                      for triple in partition_df.to_dict(orient="split")["data"]}
        valid_anomalies_triples = partition_anomalies_triples.difference(partition_original_triples)
        assert len(valid_anomalies_triples) <= len(partition_anomalies_triples)
        print(f"\t\t valid_anomalies_triples after drop duplicates: {len(valid_anomalies_triples)}")
        while len(valid_anomalies_triples) < partition_sample_size:
            h = str(random.choice(self.all_df[HEAD].values, size=1, replace=True)[0])
            r = str(random.choice(self.all_df[RELATION].values, size=1, replace=True)[0])
            t = str(random.choice(self.all_df[TAIL].values, size=1, replace=True)[0])
            if (h, r, t) in partition_original_triples:
                continue
            elif (h, r, t) in valid_anomalies_triples:
                continue
            else:
                valid_anomalies_triples.add((h, r, t))
        assert len(valid_anomalies_triples) == partition_sample_size
        valid_anomalies_df = pd.DataFrame(data=list(valid_anomalies_triples),
                                          columns=[HEAD, RELATION, TAIL]).reset_index(drop=True).astype(str)
        print(f"\t\t valid_anomalies_df final: {valid_anomalies_df.shape}")
        assert valid_anomalies_df.shape[0] == partition_sample_size
        assert valid_anomalies_df.shape[1] == 3

        # ===== Concatenate anomaly df to the original df (build the final df) =====
        partition_final_df = pd.concat([
            partition_df.reset_index(drop=True),
            valid_anomalies_df.reset_index(drop=True)
        ],
            axis=0,
            ignore_index=True,
            verify_integrity=True).reset_index(drop=True).astype(str)
        final_shape_1 = partition_final_df.shape
        partition_final_df = partition_final_df.drop_duplicates(keep="first").reset_index(drop=True)
        final_shape_2 = partition_final_df.shape
        print(f"\t\t final_df: {final_shape_2}")
        assert final_shape_1 == final_shape_2
        assert final_shape_2[0] == partition_original_size + partition_sample_size
        assert final_shape_2[1] == 3

        # ===== Build the y_fake vector ===== #
        partition_fake_y = [0] * partition_original_size + [1] * partition_sample_size
        assert len(partition_fake_y) == partition_original_size + partition_sample_size
        assert final_shape_2[0] == len(partition_fake_y)
        partition_fake_series = pd.Series(data=partition_fake_y,
                                          dtype=int,
                                          name=FAKE_FLAG)
        assert partition_fake_series.shape[0] == final_shape_2[0]

        # ===== Return (final_df, y_fake) ===== #
        return partition_final_df, partition_fake_series

    def generate_noisy_dataset(self, noise_percentage: int) -> NoisyDataset:
        # training
        training_final_df, training_y_fake = self._generate_noise(
            noise_percentage=noise_percentage,
            partition_name=TRAINING,
            partition_df=self.training_df,
            sampling_with_replacement_flag=True,
            random_state_head=self.random_states_training[0],
            random_state_relation=self.random_states_training[1],
            random_state_tail=self.random_states_training[2],
        )
        assert training_final_df.shape[0] == training_y_fake.shape[0]
        assert training_final_df.shape[1] == 3
        # validation
        validation_final_df, validation_y_fake = self._generate_noise(
            noise_percentage=noise_percentage,
            partition_name=VALIDATION,
            partition_df=self.validation_df,
            sampling_with_replacement_flag=False,
            random_state_head=self.random_states_validation[0],
            random_state_relation=self.random_states_validation[1],
            random_state_tail=self.random_states_validation[2],
        )
        assert validation_final_df.shape[0] == validation_y_fake.shape[0]
        assert validation_final_df.shape[1] == 3
        # testing
        testing_final_df, testing_y_fake = self._generate_noise(
            noise_percentage=noise_percentage,
            partition_name=TESTING,
            partition_df=self.testing_df,
            sampling_with_replacement_flag=False,
            random_state_head=self.random_states_testing[0],
            random_state_relation=self.random_states_testing[1],
            random_state_tail=self.random_states_testing[2],
        )
        assert testing_final_df.shape[0] == testing_y_fake.shape[0]
        assert testing_final_df.shape[1] == 3
        # Return the obtained NoisyDataset object
        return NoisyDataset(training_df=training_final_df,
                            training_y_fake=training_y_fake,
                            validation_df=validation_final_df,
                            validation_y_fake=validation_y_fake,
                            testing_df=testing_final_df,
                            testing_y_fake=testing_y_fake)

    def _generate_random(self,
                         partition_name: str,
                         partition_df: pd.DataFrame,
                         sampling_with_replacement_flag: bool,
                         random_state_head: int,
                         random_state_relation: int,
                         random_state_tail: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate a random dataframe;

        :param partition_name: (*str*) "training" | "validation" | "testing"
        :param partition_df: (*pd.DataFrame*) input dataframe with triples from which generate random dataset
        :param sampling_with_replacement_flag: (*bool*) boolean flag that indicate the sampling strategy.
                - True ==> the same element x could be sampled more times (es. from [1,2,3,4,5] we can sample 2,2,5,5)
                - False ==> once sampled an element x, we never resample x (es. from [1,2,3,4,5] we can sample 2,3,5,1)

        :return: (random dataframe, boolean fake series)
        """
        print(f"\n{partition_name}")
        assert random_state_head != random_state_relation
        assert random_state_relation != random_state_tail
        assert random_state_head != random_state_relation

        initial_shape = partition_df.shape
        partition_df = partition_df.astype(str).drop_duplicates(keep="first").reset_index(drop=True)
        assert initial_shape == partition_df.shape
        partition_original_size = partition_df.shape[0]

        # ===== Create random df, by sampling from original df ===== #
        head_sample = self.all_df[HEAD].sample(n=partition_original_size,
                                               replace=sampling_with_replacement_flag,
                                               random_state=random_state_head).values
        relation_sample = self.all_df[RELATION].sample(n=partition_original_size,
                                                       replace=sampling_with_replacement_flag,
                                                       random_state=random_state_relation).values
        tail_sample = self.all_df[TAIL].sample(n=partition_original_size,
                                               replace=sampling_with_replacement_flag,
                                               random_state=random_state_tail).values
        random_df = pd.DataFrame(data={HEAD: head_sample,
                                       RELATION: relation_sample,
                                       TAIL: tail_sample}).reset_index(drop=True).astype(str)
        print(f"\t\t {partition_name} random_df size : {random_df.shape}")
        assert head_sample.shape[0] == partition_original_size
        assert relation_sample.shape[0] == partition_original_size
        assert tail_sample.shape[0] == partition_original_size
        assert random_df.shape[0] == partition_original_size

        # ===== Manage duplicates introduced by the previous sampling ===== #
        random_df = random_df.astype(str).drop_duplicates(keep="first").reset_index(drop=True)
        print(f"\t\t {partition_name} random_df size after drop duplicates: {random_df.shape}")

        # ===== Create Random fake y vector ===== #
        random_y = pd.Series(data=[1 for _ in range(0, random_df.shape[0])],
                             dtype=int,
                             name=FAKE_FLAG)
        assert random_y.shape[0] == random_df.shape[0]

        # ===== Return (final_df, y_fake) ===== #
        return random_df, random_y

    def generate_random_dataset(self) -> RandomDataset:
        # training
        training_final_df, training_y_fake = self._generate_random(
            partition_name=TRAINING,
            partition_df=self.training_df,
            sampling_with_replacement_flag=True,
            random_state_head=self.random_states_training[0],
            random_state_relation=self.random_states_training[1],
            random_state_tail=self.random_states_training[2],
        )
        assert training_final_df.shape[0] == training_y_fake.shape[0]
        assert training_final_df.shape[1] == 3
        # validation
        validation_final_df, validation_y_fake = self._generate_random(
            partition_name=VALIDATION,
            partition_df=self.validation_df,
            sampling_with_replacement_flag=False,
            random_state_head=self.random_states_validation[0],
            random_state_relation=self.random_states_validation[1],
            random_state_tail=self.random_states_validation[2],
        )
        assert validation_final_df.shape[0] == validation_y_fake.shape[0]
        assert validation_final_df.shape[1] == 3
        # testing
        testing_final_df, testing_y_fake = self._generate_random(
            partition_name=TESTING,
            partition_df=self.testing_df,
            sampling_with_replacement_flag=False,
            random_state_head=self.random_states_testing[0],
            random_state_relation=self.random_states_testing[1],
            random_state_tail=self.random_states_testing[2],
        )
        assert testing_final_df.shape[0] == testing_y_fake.shape[0]
        assert testing_final_df.shape[1] == 3
        # Return the obtained NoisyDataset object
        return RandomDataset(training_df=training_final_df,
                             training_y_fake=training_y_fake,
                             validation_df=validation_final_df,
                             validation_y_fake=validation_y_fake,
                             testing_df=testing_final_df,
                             testing_y_fake=testing_y_fake)
