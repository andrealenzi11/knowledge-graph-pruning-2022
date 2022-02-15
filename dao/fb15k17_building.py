import json
import os
import re
from config import DATASETS_DIR


def preprocess_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower().strip().replace(" ", "_")


def convert_to_textual_triples(entity_wikidata_map_path: str,
                               in_file_path: str,
                               out_file_path: str):
    # read dictionary with mapping
    with open(entity_wikidata_map_path, "r") as mapping_file:
        entity_wikidata_mapping = json.load(mapping_file)
    valid_triples_cnt = 0
    errors_triples_cnt, errors_heads_cnt, errors_tails_cnt = 0, 0, 0
    # conversion
    with open(out_file_path, "w") as fw_out:
        with open(in_file_path, "r") as fr_in:
            for line in fr_in.readlines():
                triple = line.strip().split()
                if len(triple) != 3:
                    raise ValueError(f"Invalid triple: {triple}")
                else:
                    error_flag = False
                    head, relation, tail = triple[0], triple[1], triple[2]
                    # try to convert head
                    try:
                        textual_head = preprocess_entity(text=entity_wikidata_mapping[head]["label"])
                    except KeyError:
                        errors_heads_cnt += 1
                        error_flag = True
                    # try to convert tail
                    try:
                        textual_tail = preprocess_entity(text=entity_wikidata_mapping[tail]["label"])
                    except KeyError:
                        errors_tails_cnt += 1
                        error_flag = True
                    # manage errors
                    if error_flag:
                        errors_triples_cnt += 1
                    else:
                        fw_out.write(f"{textual_head}\t{relation}\t{textual_tail}\n")
                        valid_triples_cnt += 1

    print("\n### HITS ###")
    print(f"\t\t - valid_triples: {valid_triples_cnt}")
    print("\n### MISSES ###")
    print(f"\t\t - error_triples: {errors_triples_cnt}")
    print(f"\t\t - error_heads: {errors_heads_cnt}")
    print(f"\t\t - error_tails: {errors_tails_cnt}")


if __name__ == '__main__':

    in_train_path = os.path.join(DATASETS_DIR, "FB15k237", "train.txt")
    in_validation_path = os.path.join(DATASETS_DIR, "FB15k237", "valid.txt")
    in_test_path = os.path.join(DATASETS_DIR, "FB15k237", "test.txt")

    entity_wikidata_mapping_path = os.path.join(DATASETS_DIR, "FB15k237", "entity2wikidata.json")

    out_train_path = os.path.join(DATASETS_DIR, "FB15k237", "text", "train.tsv")
    out_validation_path = os.path.join(DATASETS_DIR, "FB15k237", "text", "validation.tsv")
    out_test_path = os.path.join(DATASETS_DIR, "FB15k237", "text", "test.tsv")

    print("\n >>> Training...")
    convert_to_textual_triples(entity_wikidata_map_path=entity_wikidata_mapping_path,
                               in_file_path=in_train_path,
                               out_file_path=out_train_path)

    print("\n >>> Validation...")
    convert_to_textual_triples(entity_wikidata_map_path=entity_wikidata_mapping_path,
                               in_file_path=in_validation_path,
                               out_file_path=out_validation_path)

    print("\n >>> Testing...")
    convert_to_textual_triples(entity_wikidata_map_path=entity_wikidata_mapping_path,
                               in_file_path=in_test_path,
                               out_file_path=out_test_path)
