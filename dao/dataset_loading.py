import os
from typing import Tuple

import pandas as pd
from pykeen import datasets

from config import DATASETS_DIR, FAKE_FLAG, \
    TRAINING_TSV, TRAINING_Y_FAKE_TSV, \
    VALIDATION_TSV, VALIDATION_Y_FAKE_TSV, \
    TESTING_TSV, TESTING_Y_FAKE_TSV, FB15K237, WN18RR, YAGO310, COUNTRIES


class PykeenDatasetLoader:
    """
    Class that loads and returns a PyKeen DataSet
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = str(dataset_name).upper().strip()

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

        # ===== Error ===== #
        else:
            raise ValueError(F"Invalid pykeen_dataset name '{self.dataset_name}'!")


class TsvDatasetLoader:
    """
    Class that loads and returns a tsv dataset from File System
    """

    def __init__(self,
                 dataset_name: str,
                 noise_level: str):
        self.dataset_name = dataset_name
        self.noise_level = noise_level
        # training
        self.in_path_df_training = os.path.join(DATASETS_DIR, self.dataset_name, self.noise_level,
                                                TRAINING_TSV)
        self.in_path_y_fake_training = os.path.join(DATASETS_DIR, self.dataset_name, self.noise_level,
                                                    TRAINING_Y_FAKE_TSV)
        # validation
        self.in_path_df_validation = os.path.join(DATASETS_DIR, self.dataset_name, self.noise_level,
                                                  VALIDATION_TSV)
        self.in_path_y_fake_validation = os.path.join(DATASETS_DIR, self.dataset_name, self.noise_level,
                                                      VALIDATION_Y_FAKE_TSV)
        # testing
        self.in_path_df_testing = os.path.join(DATASETS_DIR, self.dataset_name, self.noise_level,
                                               TESTING_TSV)
        self.in_path_y_fake_testing = os.path.join(DATASETS_DIR, self.dataset_name, self.noise_level,
                                                   TESTING_Y_FAKE_TSV)

    def get_training_validation_testing_dfs_paths(self) -> Tuple[str, str, str]:
        return self.in_path_df_training, self.in_path_df_validation, self.in_path_df_testing

    def get_training_validation_testing_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        training_df = pd.read_csv(filepath_or_buffer=self.in_path_df_training, sep="\t", encoding="utf-8")
        validation_df = pd.read_csv(filepath_or_buffer=self.in_path_df_validation, sep="\t", encoding="utf-8")
        testing_df = pd.read_csv(filepath_or_buffer=self.in_path_df_testing, sep="\t", encoding="utf-8")
        return training_df, validation_df, testing_df

    def get_training_validation_testing_y_fakes(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        training_y_fake = pd.read_csv(filepath_or_buffer=self.in_path_y_fake_training,
                                      sep="\t", encoding="utf-8")[FAKE_FLAG]
        validation_y_fake = pd.read_csv(filepath_or_buffer=self.in_path_y_fake_validation,
                                        sep="\t", encoding="utf-8")[FAKE_FLAG]
        testing_y_fake = pd.read_csv(filepath_or_buffer=self.in_path_y_fake_testing,
                                     sep="\t", encoding="utf-8")[FAKE_FLAG]
        return training_y_fake, validation_y_fake, testing_y_fake
