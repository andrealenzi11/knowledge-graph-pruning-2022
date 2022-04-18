import json
import os

import nltk
from nltk.corpus import wordnet

from src.config.config import DATASETS_DIR, ORIGINAL, COUNTRIES, YAGO310, FB15K237, WN18RR, CODEXSMALL, NATIONS, \
    FB15K237_MAPPING_FILE, CODEXSMALL_ENTITIES_MAPPING_FILE, CODEXSMALL_RELATIONS_MAPPING_FILE
from src.dao.dataset_convertion import DatasetConverter
from src.dao.dataset_loading import PykeenDatasetLoader
from src.dao.dataset_storing import DatasetExporter


if __name__ == '__main__':

    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        nltk.download('wordnet')
    except Exception:
        print("Error in download NLTK resources!")

    # WN18RR entities_id_label_map
    wordnet_offset_2_wordnet_name_map = {str(s.offset()).lstrip("0"): str(s.name()) for s in wordnet.all_synsets()}
    print(f"\n#wordnet_offset_2_wordnet_name_map: {len(wordnet_offset_2_wordnet_name_map)}")

    # FB15K237 entities_id_label_map
    with open(FB15K237_MAPPING_FILE, "r") as mapping_file:
        entity_wikidata_mapping = json.load(mapping_file)
    freebase_id_2_wikidata_label_map = {k: v["label"] for k, v in entity_wikidata_mapping.items()}
    print(f"\n#freebase_id_2_wikidata_label_map: {len(freebase_id_2_wikidata_label_map)}")

    # CODEXSMALL entities_id_label_map and relations_id_label_map
    # entity map
    with open(CODEXSMALL_ENTITIES_MAPPING_FILE, "r") as mapping_file1:
        entity_codexsmall_tmp = json.load(mapping_file1)
        entity_codexsmall_mapping_diz = {k: v["label"] for k, v in entity_codexsmall_tmp.items()}
    print(f"\n#entity_codexsmall_mapping_diz: {len(entity_codexsmall_mapping_diz)}")
    # relation map
    with open(CODEXSMALL_RELATIONS_MAPPING_FILE, "r") as mapping_file2:
        relations_codexsmall_tmp = json.load(mapping_file2)
        relations_codexsmall_mapping_diz = {k: v["label"] for k, v in relations_codexsmall_tmp.items()}
    print(f"\n#relations_codexsmall_mapping_diz: {len(relations_codexsmall_mapping_diz)}")

    for current_dataset_name, entities_id_label_map, relations_id_label_map in [
        (COUNTRIES, None, None),
        (YAGO310, None, None),
        (FB15K237, freebase_id_2_wikidata_label_map, None),
        (WN18RR, wordnet_offset_2_wordnet_name_map, None),
        (CODEXSMALL, entity_codexsmall_mapping_diz, relations_codexsmall_mapping_diz),
        (NATIONS, None, None),
    ]:
        print(f"\n\n\n##### {current_dataset_name} #####")

        if entities_id_label_map:
            print(f"\t\t - 'entities_id_label_map' size: {len(entities_id_label_map)}")
        else:
            print("\t\t - 'entities_id_label_map' is None")

        if relations_id_label_map:
            print(f"\t\t - 'relations_id_label_map' size: {len(relations_id_label_map)}")
        else:
            print("\t\t - 'relations_id_label_map' is None")

        # ===== Get Pykeen Dataset ===== #
        my_pykeen_dataset = PykeenDatasetLoader(dataset_name=current_dataset_name).get_pykeen_dataset()
        print(f" Dataset Info: {my_pykeen_dataset}")

        # ===== Conversion to DataFrames ===== #
        my_dataset_converter = DatasetConverter(pykeen_dataset=my_pykeen_dataset,
                                                id_label_map1=entities_id_label_map,
                                                id_label_map2=relations_id_label_map)

        # train
        my_training_df = my_dataset_converter.get_training_df().astype(str).reset_index(drop=True)
        my_training_df = my_training_df.drop_duplicates(keep="first").reset_index(drop=True)
        print(f"\t - training shape: {my_training_df.shape}")
        assert my_training_df.shape[1] == 3

        # valid
        my_validation_df = my_dataset_converter.get_validation_df().astype(str).reset_index(drop=True)
        my_validation_df = my_validation_df.drop_duplicates(keep="first").reset_index(drop=True)
        print(f"\t - validation shape: {my_validation_df.shape}")
        assert my_validation_df.shape[1] == 3

        # test
        my_testing_df = my_dataset_converter.get_testing_df().astype(str).reset_index(drop=True)
        my_testing_df = my_testing_df.drop_duplicates(keep="first").reset_index(drop=True)
        print(f"\t - testing shape: {my_testing_df.shape}")
        assert my_testing_df.shape[1] == 3

        # Overview of the DF head
        print(f"\n - Training Head:\n{my_training_df.head(10)}")

        # ===== Export to FS ===== #
        print("\n - out_folder_path:")
        my_out_folder_path = os.path.join(DATASETS_DIR, current_dataset_name, ORIGINAL)
        print(my_out_folder_path)
        assert ORIGINAL in my_out_folder_path
        assert current_dataset_name in my_out_folder_path

        DatasetExporter(output_folder=my_out_folder_path,
                        training_df=my_training_df,
                        validation_df=my_validation_df,
                        testing_df=my_testing_df).store_to_file_system()
