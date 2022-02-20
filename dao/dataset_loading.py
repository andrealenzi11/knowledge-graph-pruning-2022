import os
from typing import Tuple

import pandas as pd
from pykeen import datasets

from config import DATASETS_DIR
from dao.data_model import DatasetName


class PykeenDatasetLoader:
    """
    Class that loads and returns a PyKeen DataSet
    """

    def __init__(self, dataset_name: DatasetName):
        self.dataset_name = dataset_name  # str(dataset_name).upper().strip()

    def get_pykeen_dataset(self) -> datasets.Dataset:
        # ===== 'FB15k237' dataset ===== #
        if self.dataset_name == DatasetName.FB15K237:
            return datasets.FB15k237(create_inverse_triples=False)

        # ===== 'WN18RR' dataset ===== #
        elif self.dataset_name == DatasetName.WN18RR:
            return datasets.WN18RR(create_inverse_triples=False)

        # ===== 'YAGO310' dataset ===== #
        elif self.dataset_name == DatasetName.YAGO310:
            return datasets.YAGO310(create_inverse_triples=False)

        # ===== 'Countries' dataset ===== #
        elif self.dataset_name == DatasetName.COUNTRIES:
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
        self.in_path_training = os.path.join(DATASETS_DIR,
                                             self.dataset_name,
                                             self.noise_level,
                                             "training.tsv")
        self.in_path_validation = os.path.join(DATASETS_DIR,
                                               self.dataset_name,
                                               self.noise_level,
                                               "validation.tsv")
        self.in_path_testing = os.path.join(DATASETS_DIR,
                                            self.dataset_name,
                                            self.noise_level,
                                            "testing.tsv")

    def get_training_validation_testing_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        training_df = pd.read_csv(filepath_or_buffer=self.in_path_training,
                                  sep="\t",
                                  encoding="utf-8")
        validation_df = pd.read_csv(filepath_or_buffer=self.in_path_validation,
                                    sep="\t",
                                    encoding="utf-8")
        testing_df = pd.read_csv(filepath_or_buffer=self.in_path_testing,
                                 sep="\t",
                                 encoding="utf-8")
        return training_df, validation_df, testing_df
