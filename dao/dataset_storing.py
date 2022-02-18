import os

import pandas as pd


class DatasetExporter:
    """
    Class that export to the File System the datasets as tsv files
    """

    def __init__(self,
                 output_folder: str,
                 training_df: pd.DataFrame,
                 validation_df: pd.DataFrame,
                 testing_df: pd.DataFrame):
        self.output_folder = output_folder
        self.training_df = training_df
        self.validation_df = validation_df
        self.testing_df = testing_df

    def _store_training(self):
        self.training_df.to_csv(path_or_buf=os.path.join(self.output_folder, "training.tsv"),
                                sep="\t", header=True, index=False, encoding="utf-8")

    def _store_validation(self):
        self.validation_df.to_csv(path_or_buf=os.path.join(self.output_folder, "validation.tsv"),
                                  sep="\t", header=True, index=False, encoding="utf-8")

    def _store_testing(self):
        self.testing_df.to_csv(path_or_buf=os.path.join(self.output_folder, "testing.tsv"),
                               sep="\t", header=True, index=False, encoding="utf-8")

    def store_to_file_system(self):
        self._store_training()
        self._store_validation()
        self._store_testing()
