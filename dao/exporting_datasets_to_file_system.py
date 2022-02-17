import json
import os
import re
from typing import Optional

import nltk
import pandas as pd
from nltk.corpus import wordnet
from pykeen import datasets

from config import HEAD, RELATION, TAIL, DATASETS_DIR


class PykeenDatasetLoader:
    """
    Class that loads and returns a PyKeen DataSet
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = str(dataset_name).upper().strip()

    def get_pykeen_dataset(self) -> datasets.Dataset:
        # ===== 'FB15k237' dataset ===== #
        if self.dataset_name in ["FB15K237", "FB15K_237", "FB15K-237"]:
            return datasets.FB15k237(create_inverse_triples=False)

        # ===== 'WN18RR' dataset ===== #
        elif self.dataset_name in ["WN18RR", "WN18_RR", "WN18-RR"]:
            return datasets.WN18RR(create_inverse_triples=False)

        # ===== 'YAGO310' dataset ===== #
        elif self.dataset_name in ["YAGO310", "YAGO3_10", "YAGO3-10"]:
            return datasets.YAGO310(create_inverse_triples=False)

        # ===== 'UMLS' dataset ===== #
        elif self.dataset_name == "UMLS":
            return datasets.UMLS(create_inverse_triples=False)

        # ===== 'Countries' dataset ===== #
        elif self.dataset_name == "COUNTRIES":
            return datasets.Countries(create_inverse_triples=False)

        # ===== Error ===== #
        else:
            raise ValueError("Invalid pykeen_dataset name!")


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
                                   columns=[HEAD, RELATION, TAIL]).reset_index(drop=True)
        return self._from_entity_ids_to_entity_labels(triples_df=training_df)

    def get_validation_df(self) -> pd.DataFrame:
        validation_df = pd.DataFrame(data=self.pykeen_dataset.validation.triples,
                                     columns=[HEAD, RELATION, TAIL]).reset_index(drop=True)
        return self._from_entity_ids_to_entity_labels(triples_df=validation_df)

    def get_testing_df(self) -> pd.DataFrame:
        testing_df = pd.DataFrame(data=self.pykeen_dataset.testing.triples,
                                  columns=[HEAD, RELATION, TAIL]).reset_index(drop=True)
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


# def from_knowledge_graph_txt_to_knowledge_graph_tsv(in_file_path: str,
#                                                     out_file_path: str):
#     if not in_file_path.endswith(".txt"):
#         raise ValueError(f"Invalid in_file_path: '{in_file_path}'")
#     if not out_file_path.endswith(".tsv"):
#         raise ValueError(f"Invalid out_file_path: '{out_file_path}'")
#     with open(out_file_path, "w") as fw_out:
#         with open(in_file_path, "r") as fr_in:
#             for line in fr_in.readlines():
#                 triple = line.strip().split()
#                 if len(triple) != 3:
#                     raise ValueError(f"Invalid triple: {triple}")
#                 else:
#                     head, relation, tail = triple[0], triple[1], triple[2]
#                     fw_out.write(f"{str(head)}\t{str(relation)}\t{str(tail)}\n")


if __name__ == '__main__':

    try:
        nltk.download('omw-1.4')
    except Exception:
        print("Error in download NLTK WordNet!")

    wordnet_offset_2_wordnet_name_map = {str(s.offset()).lstrip("0"): str(s.name()) for s in wordnet.all_synsets()}
    print(f"#wordnet_offset_2_wordnet_name_map: {len(wordnet_offset_2_wordnet_name_map)}")

    with open(os.path.join(DATASETS_DIR, "FB15K237", "entity2wikidata.json"), "r") as mapping_file:
        entity_wikidata_mapping = json.load(mapping_file)
    freebase_id_2_wikidata_label_map = {k: v["label"] for k, v in entity_wikidata_mapping.items()}
    print(f"#freebase_id_2_wikidata_label_map: {len(freebase_id_2_wikidata_label_map)}")

    for current_dataset_name, current_id_label_map in [
        ("Countries", None),
        ("YAGO3-10", None),
        ("FB15K237", freebase_id_2_wikidata_label_map),
        ("WN18RR", wordnet_offset_2_wordnet_name_map),
    ]:
        print(f"\n\n\n##### {current_dataset_name} #####")
        # ===== Get Pykeen Dataset ===== #
        my_pykeen_dataset = PykeenDatasetLoader(dataset_name=current_dataset_name).get_pykeen_dataset()
        print(f" Dataset Info: {my_pykeen_dataset}")

        # ===== Conversion to DataFrames ===== #
        my_dataset_converter = DatasetConverter(pykeen_dataset=my_pykeen_dataset,
                                                id_label_map=current_id_label_map)

        # train
        my_training_df = my_dataset_converter.get_training_df()
        print(f"\t - training shape: {my_training_df.shape}")

        # valid
        my_validation_df = my_dataset_converter.get_validation_df()
        print(f"\t - validation shape: {my_validation_df.shape}")

        # test
        my_testing_df = my_dataset_converter.get_testing_df()
        print(f"\t - testing shape: {my_testing_df.shape}")

        # Overview of the DF head
        print(f"\n - Training Head:\n{my_training_df.head(10)}")

        # ===== Export to FS ===== #
        print("\n - out_folder_path:")
        my_out_folder_path = os.path.join(DATASETS_DIR, current_dataset_name, "original")
        print(my_out_folder_path)

        DatasetExporter(output_folder=my_out_folder_path,
                        training_df=my_training_df,
                        validation_df=my_validation_df,
                        testing_df=my_testing_df).store_to_file_system()
