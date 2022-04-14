import re
from typing import Optional

import pandas as pd
from pykeen import datasets

from src.config.config import HEAD, RELATION, TAIL


class DatasetConverter:
    """
    Class that transforms a PyKeen DataSet to three Pandas DataFrames (training, validation, testing)
    """

    def __init__(self,
                 pykeen_dataset: datasets.Dataset,
                 id_label_map1: Optional[dict] = None,
                 id_label_map2: Optional[dict] = None):
        self.pykeen_dataset = pykeen_dataset
        self.id_label_map1 = id_label_map1   # Entities Map
        self.id_label_map2 = id_label_map2   # Relations Map

    def get_training_df(self) -> pd.DataFrame:
        training_df = pd.DataFrame(data=self.pykeen_dataset.training.triples,
                                   columns=[HEAD, RELATION, TAIL])
        training_df = training_df.drop_duplicates(keep="first").reset_index(drop=True)
        return self._from_entity_ids_to_entity_labels(triples_df=training_df)

    def get_validation_df(self) -> pd.DataFrame:
        validation_df = pd.DataFrame(data=self.pykeen_dataset.validation.triples,
                                     columns=[HEAD, RELATION, TAIL])
        validation_df = validation_df.drop_duplicates(keep="first").reset_index(drop=True)
        return self._from_entity_ids_to_entity_labels(triples_df=validation_df)

    def get_testing_df(self) -> pd.DataFrame:
        testing_df = pd.DataFrame(data=self.pykeen_dataset.testing.triples,
                                  columns=[HEAD, RELATION, TAIL])
        testing_df = testing_df.drop_duplicates(keep="first").reset_index(drop=True)
        return self._from_entity_ids_to_entity_labels(triples_df=testing_df)

    def _from_entity_ids_to_entity_labels(self, triples_df: pd.DataFrame) -> pd.DataFrame:
        if self.id_label_map1 and isinstance(self.id_label_map1, dict):
            errors_cnt = 0
            records = list()
            for h, r, t in zip(triples_df[HEAD], triples_df[RELATION], triples_df[TAIL]):
                try:
                    # Get labels from entities ids
                    h_label = self._preprocess_entity(text=self.id_label_map1[str(h).lstrip("0")])
                    t_label = self._preprocess_entity(text=self.id_label_map1[str(t).lstrip("0")])
                    # Eventually, get label from relation id
                    if self.id_label_map2 and isinstance(self.id_label_map2, dict):
                        r_label = self._preprocess_entity(text=self.id_label_map2[str(r).lstrip("0")])
                    else:
                        r_label = r
                    # Append resolved triple to records list
                    records.append((h_label, r_label, t_label))
                except KeyError:
                    errors_cnt += 1
            print(f"\t #triples_with_mapping_errors: {errors_cnt}")
            return pd.DataFrame(data=records, columns=[HEAD, RELATION, TAIL]).reset_index(drop=True)
        else:
            return triples_df

    @staticmethod
    def _preprocess_entity(text: str) -> str:
        return re.sub(r"\s+", " ", text).lower().strip().replace(" ", "_").strip()
