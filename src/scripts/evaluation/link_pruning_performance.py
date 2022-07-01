import configparser
import os
import random

import numpy as np
import pandas as pd
import torch

from src.config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10, \
    ORIGINAL, NOISE_10, NOISE_20, NOISE_30, RESCAL, CONVE, \
    FB15K237_RESULTS_FOLDER_PATH, WN18RR_RESULTS_FOLDER_PATH, YAGO310_RESULTS_FOLDER_PATH, \
    COUNTRIES_RESULTS_FOLDER_PATH, CODEXSMALL_RESULTS_FOLDER_PATH, NATIONS_RESULTS_FOLDER_PATH, TOTAL_RANDOM
from src.core.pykeen_wrapper import get_train_test_validation, print_partitions_info, get_label_id_map, \
    get_triples_scores2, get_triples_scores3
from src.dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader
# set pandas visualization options
from src.utils.stats import get_center

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

datasets_names_results_folder_map = {
    COUNTRIES: COUNTRIES_RESULTS_FOLDER_PATH,
    WN18RR: WN18RR_RESULTS_FOLDER_PATH,
    FB15K237: FB15K237_RESULTS_FOLDER_PATH,
    YAGO310: YAGO310_RESULTS_FOLDER_PATH,
    CODEXSMALL: CODEXSMALL_RESULTS_FOLDER_PATH,
    NATIONS: NATIONS_RESULTS_FOLDER_PATH
}
for k, v in datasets_names_results_folder_map.items():
    print(f"datasets_name={k} | dataset_results_folder={v}")

all_noise_levels = {ORIGINAL, TOTAL_RANDOM, NOISE_10, NOISE_20, NOISE_30}
print(f"all_noise_levels: {all_noise_levels}")

all_metrics = {MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10}
print(f"all_metrics: {all_metrics}")

if __name__ == '__main__':

    config = configparser.ConfigParser()
    if not os.path.isfile('dataset_local.ini'):
        raise FileNotFoundError("Create your 'dataset_local.ini' file in the 'src.scripts.evaluation' package "
                                "starting from the 'dataset.ini' template!")
    config.read('dataset_local.ini')
    dataset_name = config['dataset_info']['dataset_name']
    force_saving_flag = True
    plot_confidence_flag = False
    use_median_flag = False
    force_saving = True
    n_round = 4
    selected_metrics = {
        MR,
        MRR,
        HITS_AT_1,
        HITS_AT_3,
        HITS_AT_5,
        HITS_AT_10,
    }

    dataset_models_folder_path = DatasetPathFactory(dataset_name=dataset_name).get_models_folder_path()
    assert dataset_name in dataset_models_folder_path

    dataset_results_folder_path = datasets_names_results_folder_map[dataset_name]
    assert dataset_name in dataset_results_folder_path

    print("\n> Triple Classification Evaluation - Configuration")
    print(f"\n{'*' * 80}")
    print(f"\t\t dataset_name: {dataset_name}")
    print(f"\t\t dataset_models_folder_path: {dataset_models_folder_path}")
    print(f"\t\t dataset_results_folder_path: {dataset_results_folder_path}")
    print(f"\t\t force_saving_flag: {force_saving_flag}")
    print(f"\t\t plot_confidence_flag: {plot_confidence_flag}")
    print(f"\t\t use_median_flag: {use_median_flag}")
    print(f"\t\t my_decimal_precision: {n_round}")
    print(f"{'*' * 80}\n\n")

    # ========== original dataset ========== #
    print("\n Loading original dataset...")
    datasets_loader_original = TsvDatasetLoader(dataset_name=dataset_name, noise_level=ORIGINAL)
    # paths
    training_original_path, validation_original_path, testing_original_path = \
        datasets_loader_original.get_training_validation_testing_dfs_paths(noisy_test_flag=False)
    assert "training" in training_original_path
    assert ORIGINAL in training_original_path
    assert "validation" in validation_original_path
    assert ORIGINAL in validation_original_path
    assert "testing" in testing_original_path
    assert ORIGINAL in testing_original_path
    # triples factories from original dataset
    training_original, testing_original, validation_original = \
        get_train_test_validation(training_set_path=training_original_path,
                                  test_set_path=testing_original_path,
                                  validation_set_path=validation_original_path,
                                  create_inverse_triples=False)
    # dfs
    training_original_df, validation_original_df, testing_original_df = \
        datasets_loader_original.get_training_validation_testing_dfs(noisy_test_flag=False)
    # valid_triples
    training_triples_set = {tuple(triple) for triple in training_original_df.values}
    validation_triples_set = {tuple(triple) for triple in validation_original_df.values}
    testing_triples_set = {tuple(triple) for triple in testing_original_df.values}
    valid_triples_set = training_triples_set.union(validation_triples_set).union(testing_triples_set)
    valid_entities = set()
    for vtriple in valid_triples_set:
        valid_entities.add(vtriple[0])  # add head entity
        valid_entities.add(vtriple[2])  # add tail entity
    print(f"\n#training_triples_set: {len(training_triples_set)}")
    print(f"#validation_triples_set: {len(validation_triples_set)}")
    print(f"#testing_triples_set: {len(testing_triples_set)}")
    print(f"\n#valid_triples_set: {len(valid_triples_set)}")
    print(f"#valid_entities: {len(valid_entities)}")

    # info
    print_partitions_info(training_triples=training_original,
                          training_triples_path=training_original_path,
                          validation_triples=validation_original,
                          validation_triples_path=validation_original_path,
                          testing_triples=testing_original,
                          testing_triples_path=testing_original_path)

    # ===== Iteration over noise levels ===== #
    records = {}
    selected_noise_levels = [
        TOTAL_RANDOM,
        ORIGINAL,
        NOISE_10,
        NOISE_20,
        NOISE_30,
    ]
    for noise_level in selected_noise_levels:
        print(f"\n\n#################### {noise_level} ####################\n")
        in_folder_path = os.path.join(dataset_models_folder_path, noise_level)
        assert dataset_name in in_folder_path
        assert noise_level in in_folder_path

        datasets_loader = TsvDatasetLoader(dataset_name=dataset_name, noise_level=noise_level)
        training_path, validation_path, testing_path = \
            datasets_loader.get_training_validation_testing_dfs_paths(noisy_test_flag=False)
        assert "training" in training_path
        assert noise_level in training_path
        assert "validation" in validation_path
        assert noise_level in validation_path
        assert "testing" in testing_path
        # assert noise_level in testing_path

        training, testing, validation = get_train_test_validation(training_set_path=training_path,
                                                                  test_set_path=testing_path,
                                                                  validation_set_path=validation_path,
                                                                  create_inverse_triples=False)
        print_partitions_info(training_triples=training,
                              training_triples_path=training_path,
                              validation_triples=validation,
                              validation_triples_path=validation_path,
                              testing_triples=testing,
                              testing_triples_path=testing_path)

        # ===== Iteration over KGE models ===== #
        row = {}
        for model_name in sorted(os.listdir(in_folder_path)):

            current_record = {}

            print(f"\n >>>>> model_name: {model_name}")
            in_model_dir = os.path.join(in_folder_path, model_name)
            print(f"model directory: '{in_model_dir}'")
            in_model_file = os.path.join(in_model_dir, "trained_model.pkl")
            in_entity_to_id_file = os.path.join(in_model_dir, "training_triples", "entity_to_id.tsv.gz")
            in_relation_to_id_file = os.path.join(in_model_dir, "training_triples", "relation_to_id.tsv.gz")
            assert model_name in in_model_dir

            # if model was not already trained, skip to the next iteration
            for fp in [in_model_file, in_entity_to_id_file, in_relation_to_id_file]:
                if not os.path.isfile(fp):
                    print(f"'{fp}' not present! \n")
                    continue

            # Skip Not valid models
            if model_name in [
                RESCAL,
            ]:
                continue

            # Skip ConvE for FB15K237
            if model_name == CONVE and dataset_name == FB15K237:
                continue

            # Get Label-to-Id Maps
            entities_label_id_map = get_label_id_map(gzip_training_triples_path=in_entity_to_id_file)
            relations_label_id_map = get_label_id_map(gzip_training_triples_path=in_relation_to_id_file)
            print(f"entities_label_id_map size: {len(entities_label_id_map)}")
            print(f"relations_label_id_map size: {len(relations_label_id_map)}")

            # Load model from FS
            my_pykeen_model = torch.load(in_model_file).cpu()

            # ===== Inference (computation of KGE scores) on Testing Set ====== #
            real_testing_scores = get_triples_scores2(trained_kge_model=my_pykeen_model,
                                                      triples=list(testing_triples_set),
                                                      entities_label_id_map=entities_label_id_map,
                                                      relation_label_id_map=relations_label_id_map,
                                                      debug_info=True)
            real_testing_scores_sorted = np.sort(real_testing_scores)
            real_testing_scores_center = get_center(scores=real_testing_scores,
                                                    use_median=use_median_flag)
            print(f"real_testing_scores_center: {real_testing_scores_center}")
            print(f"real_testing_scores_sorted: {real_testing_scores_sorted}")
            print(f"real_testing_scores_sorted.shape: {real_testing_scores_sorted.shape}")

            # Link Deletion Task - Evaluation algorithm
            mr_sum, mrr_sum = 0, 0
            hits_at_1_sum, hits_at_3_sum, hits_at_5_sum, hits_at_10_sum = 0, 0, 0, 0
            test_size = len(testing_triples_set)
            for h, r, t in testing_triples_set:
                fake_head_triple = None
                fake_tail_triple = None
                while True:
                    new_h = random.sample(valid_entities, 1)[0]
                    fake_head_triple = (new_h, r, t)
                    if fake_head_triple not in valid_triples_set:
                        break
                while True:
                    new_t = random.sample(valid_entities, 1)[0]
                    fake_tail_triple = (h, r, new_t)
                    if fake_tail_triple not in valid_triples_set:
                        break

                try:
                    head_tail_scores = get_triples_scores3(trained_kge_model=my_pykeen_model,
                                                           triples=[
                                                               fake_head_triple,
                                                               fake_tail_triple
                                                           ],
                                                           entities_label_id_map=entities_label_id_map,
                                                           relation_label_id_map=relations_label_id_map)
                except KeyError:
                    test_size -= 1
                    continue

                fake_h_score, fake_t_score = head_tail_scores[0], head_tail_scores[1]
                rank_head = np.searchsorted(a=real_testing_scores_sorted, v=fake_h_score, side='left') + 1
                rank_tail = np.searchsorted(a=real_testing_scores_sorted, v=fake_t_score, side='left') + 1
                rank_both = rank_head + rank_tail
                inverse_rank_both = (1.0 / rank_head) + (1.0 / rank_tail)
                mr_sum += rank_both
                mrr_sum += inverse_rank_both
                if int(rank_both / 2.0) <= 1:
                    hits_at_1_sum += 1
                if int(rank_both / 2.0) <= 3:
                    hits_at_3_sum += 1
                if int(rank_both / 2.0) <= 5:
                    hits_at_5_sum += 1
                if int(rank_both / 2.0) <= 10:
                    hits_at_10_sum += 1

            # Metrics
            print(f"new test size: {test_size}")
            double_test_size = 2 * test_size
            mr = float(mr_sum / double_test_size)
            mrr = float(mrr_sum / double_test_size)
            hits_at_1 = float(hits_at_1_sum / test_size)
            hits_at_3 = float(hits_at_3_sum / test_size)
            hits_at_5 = float(hits_at_5_sum / test_size)
            hits_at_10 = float(hits_at_10_sum / test_size)

            # Update internal current record diz
            current_record = dict()
            if noise_level == ORIGINAL:
                noise_level_k = ""
            else:
                noise_level_k = noise_level
            if MR in selected_metrics:
                current_record[f"{MR}_{noise_level_k}".rstrip("_")] = mr
            if MRR in selected_metrics:
                current_record[f"{MRR}_{noise_level_k}".rstrip("_")] = mrr
            if HITS_AT_1 in selected_metrics:
                current_record[f"{HITS_AT_1}_{noise_level_k}".rstrip("_")] = hits_at_1
            if HITS_AT_3 in selected_metrics:
                current_record[f"{HITS_AT_3}_{noise_level_k}".rstrip("_")] = hits_at_3
            if HITS_AT_5 in selected_metrics:
                current_record[f"{HITS_AT_5}_{noise_level_k}".rstrip("_")] = hits_at_5
            if HITS_AT_10 in selected_metrics:
                current_record[f"hits_at_X_{noise_level_k}".rstrip("_")] = hits_at_10

            # Update external records diz
            if model_name not in records:
                records[model_name] = current_record  # insert new record
            else:
                records[model_name] = {**records[model_name], **current_record}  # update (merge)

            print("\n")

    # Summarize and export the results after the end of iterations
    print("\n\n\n >>> Build DataFrame...")
    df_results = pd.DataFrame(data=list(records.values()),
                              index=list(records.keys())).T

    # Format for a better view the dataframe with the link prediction performance metrics
    df_results = df_results.sort_index(inplace=False, axis=0, ascending=True)
    diz_results_records = df_results.to_dict(orient="records")
    diz_results_index = list(df_results.index.values)
    step = 5
    i = 0
    new_records = []
    new_index = []
    for i_name, record in zip(diz_results_index, diz_results_records):
        if i % step == 0:
            new_index.append(f"(*)")
            new_records.append({k: np.nan for k in df_results.columns})
        new_index.append(i_name)
        new_records.append(record)
        i += 1
    df_results2 = pd.DataFrame(data=new_records, index=new_index)

    # Print the dataframe with the link prediction performance metrics
    print("\n>>> df info:")
    print(df_results2.info(memory_usage="deep"))
    print("\n>>> df overview:")
    print(df_results2)

    # Export to FS the dataframe with the link prediction performance metrics
    print("\n >>> Export DataFrame to FS...")
    out_path = os.path.join(dataset_results_folder_path, f"link_pruning_results.xlsx")
    print(f"\t out_path: {out_path}")
    assert dataset_name in out_path
    assert out_path.endswith("results.xlsx")
    assert str(out_path.split(os.path.sep)[-1]).startswith("link_pruning")
    if (os.path.isfile(out_path)) and (not force_saving):
        raise OSError(f"'{out_path}' already exists!")
    df_results2.to_excel(out_path, header=True, index=True, encoding="utf-8", engine="openpyxl")
