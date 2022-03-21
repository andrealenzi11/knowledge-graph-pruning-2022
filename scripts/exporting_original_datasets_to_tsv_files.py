import json
import os

import nltk
from nltk.corpus import wordnet

from config import DATASETS_DIR, FB15K237_MAPPING_FILE, ORIGINAL, COUNTRIES, YAGO310, FB15K237, WN18RR
from dao.dataset_convertion import DatasetConverter
from dao.dataset_loading import PykeenDatasetLoader
from dao.dataset_storing import DatasetExporter

if __name__ == '__main__':

    try:
        nltk.download('omw-1.4')
    except Exception:
        print("Error in download NLTK WordNet!")

    wordnet_offset_2_wordnet_name_map = {str(s.offset()).lstrip("0"): str(s.name()) for s in wordnet.all_synsets()}
    print(f"#wordnet_offset_2_wordnet_name_map: {len(wordnet_offset_2_wordnet_name_map)}")

    with open(FB15K237_MAPPING_FILE, "r") as mapping_file:
        entity_wikidata_mapping = json.load(mapping_file)
    freebase_id_2_wikidata_label_map = {k: v["label"] for k, v in entity_wikidata_mapping.items()}
    print(f"#freebase_id_2_wikidata_label_map: {len(freebase_id_2_wikidata_label_map)}")

    for current_dataset_name, current_id_label_map in [
        (COUNTRIES, None),
        (YAGO310, None),
        (FB15K237, freebase_id_2_wikidata_label_map),
        (WN18RR, wordnet_offset_2_wordnet_name_map),
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
        my_out_folder_path = os.path.join(DATASETS_DIR, current_dataset_name, ORIGINAL)
        print(my_out_folder_path)

        DatasetExporter(output_folder=my_out_folder_path,
                        training_df=my_training_df,
                        validation_df=my_validation_df,
                        testing_df=my_testing_df).store_to_file_system()
