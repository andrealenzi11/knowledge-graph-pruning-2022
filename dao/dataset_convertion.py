import re
from typing import Optional

import pandas as pd
from pykeen import datasets

from config import HEAD, RELATION, TAIL


class DatasetConverter:
    """
    Class that transforms a PyKeen DataSet to three Pandas DataFrames (training, validation, testing)
    """

    def __init__(self,
                 pykeen_dataset: datasets.Dataset,
                 id_label_map: Optional[dict] = None):
        self.pykeen_dataset = pykeen_dataset
        self.id_label_map = id_label_map

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
        if self.id_label_map and isinstance(self.id_label_map, dict):
            errors_cnt = 0
            records = list()
            for h, r, t in zip(triples_df[HEAD], triples_df[RELATION], triples_df[TAIL]):
                try:
                    h_label = self._preprocess_entity(text=self.id_label_map[str(h).lstrip("0")])
                    t_label = self._preprocess_entity(text=self.id_label_map[str(t).lstrip("0")])
                    records.append((h_label, r, t_label))
                except KeyError:
                    errors_cnt += 1
            print(f"\t #triples_with_mapping_errors: {errors_cnt}")
            return pd.DataFrame(data=records, columns=[HEAD, RELATION, TAIL]).reset_index(drop=True)
        else:
            return triples_df

    @staticmethod
    def _preprocess_entity(text: str) -> str:
        return re.sub(r"\s+", " ", text).lower().strip().replace(" ", "_")
