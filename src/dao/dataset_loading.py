import os
from abc import ABC
from typing import Tuple, Union, List

import pandas as pd
from pykeen import datasets

from src.config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    COUNTRIES_MODELS_FOLDER_PATH, FB15K237_MODELS_FOLDER_PATH, WN18RR_MODELS_FOLDER_PATH, \
    YAGO310_MODELS_FOLDER_PATH, CODEXSMALL_MODELS_FOLDER_PATH, NATIONS_MODELS_FOLDER_PATH, \
    COUNTRIES_TUNING_FOLDER_PATH, FB15K237_TUNING_FOLDER_PATH, WN18RR_TUNING_FOLDER_PATH, \
    YAGO310_TUNING_FOLDER_PATH, CODEXSMALL_TUNING_FOLDER_PATH, NATIONS_TUNING_FOLDER_PATH, \
    DATASETS_DIR, FAKE_FLAG, \
    TRAINING_TSV, TRAINING_Y_FAKE_TSV, \
    VALIDATION_TSV, VALIDATION_Y_FAKE_TSV, \
    TESTING_TSV, TESTING_Y_FAKE_TSV, \
    ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15, NOISE_20, NOISE_30, NOISE_100, \
    HEAD, RELATION, TAIL, TOTAL_RANDOM


class BaseDatasetLoader(ABC):

    def __init__(self, dataset_name: str):
        self.valid_datasets_names = {
            COUNTRIES,
            FB15K237,
            WN18RR,
            YAGO310,
            CODEXSMALL,
            NATIONS,
        }
        self.dataset_name = str(dataset_name).upper().strip()
        if self.dataset_name not in self.valid_datasets_names:
            raise ValueError(f"Invalid dataset name: '{str(dataset_name)}'! \n"
                             f"\t\t Specify one of the following values: \n"
                             f"\t\t {self.valid_datasets_names} \n")


class DatasetPathFactory(BaseDatasetLoader):

    def get_models_folder_path(self) -> str:
        if self.dataset_name == COUNTRIES:
            return COUNTRIES_MODELS_FOLDER_PATH
        elif self.dataset_name == FB15K237:
            return FB15K237_MODELS_FOLDER_PATH
        elif self.dataset_name == WN18RR:
            return WN18RR_MODELS_FOLDER_PATH
        elif self.dataset_name == YAGO310:
            return YAGO310_MODELS_FOLDER_PATH
        elif self.dataset_name == CODEXSMALL:
            return CODEXSMALL_MODELS_FOLDER_PATH
        elif self.dataset_name == NATIONS:
            return NATIONS_MODELS_FOLDER_PATH
        else:
            raise ValueError(f"Invalid dataset name!")

    def get_tuning_folder_path(self) -> str:
        if self.dataset_name == COUNTRIES:
            return COUNTRIES_TUNING_FOLDER_PATH
        elif self.dataset_name == FB15K237:
            return FB15K237_TUNING_FOLDER_PATH
        elif self.dataset_name == WN18RR:
            return WN18RR_TUNING_FOLDER_PATH
        elif self.dataset_name == YAGO310:
            return YAGO310_TUNING_FOLDER_PATH
        elif self.dataset_name == CODEXSMALL:
            return CODEXSMALL_TUNING_FOLDER_PATH
        elif self.dataset_name == NATIONS:
            return NATIONS_TUNING_FOLDER_PATH
        else:
            raise ValueError(f"Invalid dataset name!")


class PykeenDatasetLoader(BaseDatasetLoader):
    """
    Class that loads and returns a PyKeen DataSet
    """

    # def __init__(self, dataset_name: str):
    #     self.dataset_name = str(dataset_name).upper().strip()

    def get_pykeen_dataset(self) -> datasets.Dataset:
        # ===== 'FB15k237' dataset ===== #
        if self.dataset_name == FB15K237:
            return datasets.FB15k237(create_inverse_triples=False)

        # ===== 'WN18RR' dataset ===== #
        elif self.dataset_name == WN18RR:
            return datasets.WN18RR(create_inverse_triples=False)

        # ===== 'YAGO310' dataset ===== #
        elif self.dataset_name == YAGO310:
            return datasets.YAGO310(create_inverse_triples=False)

        # ===== 'Countries' dataset ===== #
        elif self.dataset_name == COUNTRIES:
            return datasets.Countries(create_inverse_triples=False)

        # ===== 'CoDExSmall' dataset ===== #
        elif self.dataset_name == CODEXSMALL:
            return datasets.CoDExSmall(create_inverse_triples=False)

        # ===== 'Nations' dataset ===== #
        elif self.dataset_name == NATIONS:
            return datasets.Nations(create_inverse_triples=False)

        # ===== Error ===== #
        else:
            raise ValueError(f"Invalid dataset name!")


class TsvDatasetLoader(BaseDatasetLoader):
    """
    Class that loads and returns a tsv dataset from File System
    """

    def __init__(self,
                 dataset_name: str,
                 noise_level: str):
        super().__init__(dataset_name=dataset_name)
        self.noise_level = noise_level
        self.valid_noise_levels = {
            ORIGINAL,
            TOTAL_RANDOM,
            NOISE_1,
            NOISE_5,
            NOISE_10,
            NOISE_15,
            NOISE_20,
            NOISE_30,
            NOISE_100,
        }
        if self.noise_level not in self.valid_noise_levels:
            raise ValueError(f"Invalid noise_level: '{self.noise_level}'! \n"
                             f"Specify one of the following values: {self.valid_noise_levels} \n")
        # ============ training ============ #
        self.in_path_noisy_df_training = os.path.join(DATASETS_DIR,
                                                      self.dataset_name,
                                                      self.noise_level,
                                                      TRAINING_TSV)
        assert "training" in self.in_path_noisy_df_training
        self.in_path_original_df_training = os.path.join(DATASETS_DIR,
                                                         self.dataset_name,
                                                         ORIGINAL,
                                                         TRAINING_TSV)
        assert "training" in self.in_path_original_df_training
        assert "original" in self.in_path_original_df_training
        self.in_path_y_fake_training = os.path.join(DATASETS_DIR,
                                                    self.dataset_name,
                                                    self.noise_level,
                                                    TRAINING_Y_FAKE_TSV)
        assert "training" in self.in_path_y_fake_training

        # ============ validation ============ #
        self.in_path_noisy_df_validation = os.path.join(DATASETS_DIR,
                                                        self.dataset_name,
                                                        self.noise_level,
                                                        VALIDATION_TSV)
        assert "validation" in self.in_path_noisy_df_validation
        self.in_path_original_df_validation = os.path.join(DATASETS_DIR,
                                                           self.dataset_name,
                                                           ORIGINAL,
                                                           VALIDATION_TSV)
        assert "validation" in self.in_path_original_df_validation
        assert "original" in self.in_path_original_df_validation
        self.in_path_y_fake_validation = os.path.join(DATASETS_DIR,
                                                      self.dataset_name,
                                                      self.noise_level,
                                                      VALIDATION_Y_FAKE_TSV)
        assert "validation" in self.in_path_y_fake_validation

        # ============= testing ============= #
        self.in_path_noisy_df_testing = os.path.join(DATASETS_DIR,
                                                     self.dataset_name,
                                                     self.noise_level,
                                                     TESTING_TSV)
        assert "testing" in self.in_path_noisy_df_testing
        self.in_path_original_df_testing = os.path.join(DATASETS_DIR,
                                                        self.dataset_name,
                                                        ORIGINAL,
                                                        TESTING_TSV)
        assert "testing" in self.in_path_original_df_testing
        assert "original" in self.in_path_original_df_testing
        self.in_path_y_fake_testing = os.path.join(DATASETS_DIR,
                                                   self.dataset_name,
                                                   self.noise_level,
                                                   TESTING_Y_FAKE_TSV)
        assert "testing" in self.in_path_y_fake_testing

    def get_training_validation_testing_dfs_paths(self, noisy_test_flag: bool) -> Tuple[str, str, str]:
        if noisy_test_flag:
            return self.in_path_noisy_df_training, self.in_path_noisy_df_validation, self.in_path_noisy_df_testing
        else:
            return self.in_path_noisy_df_training, self.in_path_noisy_df_validation, self.in_path_original_df_testing

    def get_training_validation_testing_dfs(self, noisy_test_flag: bool) -> Tuple[pd.DataFrame,
                                                                                  pd.DataFrame,
                                                                                  pd.DataFrame]:
        # training
        print(f"\t\t\t training_path: {self.in_path_noisy_df_training}")
        training_df = pd.read_csv(filepath_or_buffer=self.in_path_noisy_df_training,
                                  sep="\t", encoding="utf-8", names=[HEAD, RELATION, TAIL], header=None)
        # validation
        print(f"\t\t\t validation_path: {self.in_path_noisy_df_validation}")
        validation_df = pd.read_csv(filepath_or_buffer=self.in_path_noisy_df_validation,
                                    sep="\t", encoding="utf-8", names=[HEAD, RELATION, TAIL], header=None)
        # testing
        if noisy_test_flag:
            print(f"\t\t\t testing_path: {self.in_path_noisy_df_testing}")
            testing_df = pd.read_csv(filepath_or_buffer=self.in_path_noisy_df_testing,
                                     sep="\t", encoding="utf-8", names=[HEAD, RELATION, TAIL], header=None)
        else:
            print(f"\t\t\t testing_path: {self.in_path_original_df_testing}")
            testing_df = pd.read_csv(filepath_or_buffer=self.in_path_original_df_testing,
                                     sep="\t", encoding="utf-8", names=[HEAD, RELATION, TAIL], header=None)
        # return the 3 dfs
        return training_df, validation_df, testing_df

    def get_training_validation_testing_y_fakes(self) -> Union[Tuple[pd.Series, pd.Series, pd.Series], None]:
        if self.noise_level == ORIGINAL:
            return None
        training_y_fake = pd.read_csv(filepath_or_buffer=self.in_path_y_fake_training,
                                      sep="\t", encoding="utf-8", names=[FAKE_FLAG], header=None)[FAKE_FLAG]
        validation_y_fake = pd.read_csv(filepath_or_buffer=self.in_path_y_fake_validation,
                                        sep="\t", encoding="utf-8", names=[FAKE_FLAG], header=None)[FAKE_FLAG]
        testing_y_fake = pd.read_csv(filepath_or_buffer=self.in_path_y_fake_testing,
                                     sep="\t", encoding="utf-8", names=[FAKE_FLAG], header=None)[FAKE_FLAG]
        return training_y_fake, validation_y_fake, testing_y_fake

    def get_training_validation_testing_y_fakes_paths(self) -> Union[Tuple[str, str, str], None]:
        if self.noise_level == ORIGINAL:
            return None
        return self.in_path_y_fake_training, self.in_path_y_fake_validation, self.in_path_y_fake_testing


def get_data_records(kg_df: pd.DataFrame,
                     y_fake_series: pd.Series,
                     select_only_fake_flag: bool) -> List[Tuple[str, str, str]]:
    """
    Function that returns fake or true knowledge graph records

    :param kg_df: Knowledge Graph DataFrame with head, relation and tail columns
    :param y_fake_series: Series with ones that indicate fake value and zeros that indicate true values
    :param select_only_fake_flag: boolean flag that indicate if select only fake triples or not

    :return: List[Tuple[str]]
    [
        ("head_1", "relation_1", "tail_1"),
        ("head_2", "relation_2", "tail_3"),
            .           .            .
            .           .            .
            .           .            .
        ("head_N", "relation_N", "tail_N"),
    ]
    """
    assert isinstance(kg_df, pd.DataFrame)
    assert isinstance(y_fake_series, pd.Series)
    assert isinstance(select_only_fake_flag, bool)
    assert kg_df.shape[0] == y_fake_series.shape[0]
    assert kg_df.shape[1] == 3
    merged_kg_fake_df = pd.concat(objs=[kg_df.reset_index(drop=True), y_fake_series],
                                  axis=1,
                                  verify_integrity=True,
                                  ignore_index=True).reset_index(drop=True)
    assert merged_kg_fake_df.shape[0] == kg_df.shape[0]
    assert merged_kg_fake_df.shape[0] == y_fake_series.shape[0]
    assert merged_kg_fake_df.shape[1] == 4
    if select_only_fake_flag:
        value_to_select = 1  # select only fake triples
    else:
        value_to_select = 0  # select only real triples
    result_df = merged_kg_fake_df[merged_kg_fake_df.loc[:, 3] == value_to_select]  # selection on rows
    result_df = result_df.drop(3, axis=1).reset_index(drop=True)  # drop the last column
    assert result_df.shape[0] <= merged_kg_fake_df.shape[0]
    assert result_df.shape[1] == 3
    result_records = result_df.to_dict(orient="split")["data"]
    # Convert to List[Tuple[str, str, str]]
    result_records_final = []
    for my_record in result_records:
        assert len(my_record) == 3
        result_records_final.append(
            (str(my_record[0]), str(my_record[1]), str(my_record[2]))
        )
    assert len(result_records_final) == len(result_records)
    return result_records_final
