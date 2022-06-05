import configparser
import os

import numpy as np
import pandas as pd
import torch
from sklearn import metrics

from src.config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    ORIGINAL, NOISE_10, NOISE_20, NOISE_30, NOISE_100, \
    FB15K237_RESULTS_FOLDER_PATH, WN18RR_RESULTS_FOLDER_PATH, YAGO310_RESULTS_FOLDER_PATH, \
    COUNTRIES_RESULTS_FOLDER_PATH, CODEXSMALL_RESULTS_FOLDER_PATH, NATIONS_RESULTS_FOLDER_PATH, RESCAL, F1_MACRO, \
    F1_POS, F1_NEG, NORM_DIST, Z_STAT, TOTAL_RANDOM, CONVE
from src.core.pykeen_wrapper import get_train_test_validation, print_partitions_info, get_triples_scores2, \
    get_label_id_map
from src.dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader, get_data_records
from src.utils.distribution_plotting import draw_distribution_plot
from src.utils.stats import get_center, print_2d_statistics

# set pandas visualization options
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

all_datasets_names = {COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS}
print(f"all_datasets_names: {all_datasets_names}")

all_noise_levels = {ORIGINAL, TOTAL_RANDOM, NOISE_10, NOISE_20, NOISE_30, NOISE_100}
print(f"all_noise_levels: {all_noise_levels}")

all_metrics = {F1_MACRO, F1_POS, F1_NEG, NORM_DIST, Z_STAT}
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
        F1_MACRO,
        F1_POS,
        F1_NEG,
        NORM_DIST,
        Z_STAT,
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

    # ========== noisy 100 dataset ========== #
    print("\n Loading Noisy 100 dataset...")
    datasets_loader_100 = TsvDatasetLoader(dataset_name=dataset_name, noise_level=NOISE_100)
    # paths
    training_100_path, validation_100_path, testing_100_path = \
        datasets_loader_100.get_training_validation_testing_dfs_paths(noisy_test_flag=True)
    assert "training" in training_100_path
    assert NOISE_100 in training_100_path
    assert "validation" in validation_100_path
    assert NOISE_100 in validation_100_path
    assert "testing" in testing_100_path
    assert NOISE_100 in testing_100_path
    # dfs
    training_100_df, validation_100_df, testing_100_df = \
        datasets_loader_100.get_training_validation_testing_dfs(noisy_test_flag=True)
    print(testing_100_df.shape)
    print(testing_100_df.drop_duplicates().shape)
    # y_fakes
    training_100_y_fake, validation_100_y_fake, testing_100_y_fake = \
        datasets_loader_100.get_training_validation_testing_y_fakes()

    # fake validation records
    validation_100_fake_records = get_data_records(kg_df=validation_100_df,
                                                   y_fake_series=validation_100_y_fake,
                                                   select_only_fake_flag=True)
    print(f"\t - fake validation records size {len(validation_100_fake_records)}")
    assert len(validation_100_fake_records) == int(validation_100_df.shape[0] / 2)

    # real validation records
    validation_100_real_records = get_data_records(kg_df=validation_100_df,
                                                   y_fake_series=validation_100_y_fake,
                                                   select_only_fake_flag=False)
    print(f"\t - real validation records size {len(validation_100_real_records)}")
    assert len(validation_100_real_records) == int(validation_100_df.shape[0] / 2)

    # fake testing records
    testing_100_fake_records = get_data_records(kg_df=testing_100_df,
                                                y_fake_series=testing_100_y_fake,
                                                select_only_fake_flag=True)
    print(f"\t - fake testing records size {len(testing_100_fake_records)}")
    assert len(testing_100_fake_records) == int(testing_100_df.shape[0] / 2)
    # real testing records
    testing_100_real_records = get_data_records(kg_df=testing_100_df,
                                                y_fake_series=testing_100_y_fake,
                                                select_only_fake_flag=False)
    print(f"\t - real testing records size {len(testing_100_real_records)}")
    assert len(testing_100_real_records) == int(testing_100_df.shape[0] / 2)
    # triples factories for noise 100 dataset
    training_100, testing_100, validation_100 = get_train_test_validation(training_set_path=training_100_path,
                                                                          test_set_path=testing_100_path,
                                                                          validation_set_path=validation_100_path,
                                                                          create_inverse_triples=False)
    print_partitions_info(training_triples=training_100,
                          training_triples_path=training_100_path,
                          validation_triples=validation_100,
                          validation_triples_path=validation_100_path,
                          testing_triples=testing_100,
                          testing_triples_path=testing_100_path)

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
            datasets_loader.get_training_validation_testing_dfs_paths(noisy_test_flag=True)
        assert "training" in training_path
        assert noise_level in training_path
        assert "validation" in validation_path
        assert noise_level in validation_path
        assert "testing" in testing_path
        assert noise_level in testing_path

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

            # ===== Inference (computation of KGE scores) on Original Training Set ====== #
            training_scores_vector = get_triples_scores2(trained_kge_model=my_pykeen_model,
                                                         triples=training_original_df.to_records(index=False).tolist(),
                                                         entities_label_id_map=entities_label_id_map,
                                                         relation_label_id_map=relations_label_id_map)
            training_scores_center = get_center(scores=training_scores_vector,
                                                use_median=use_median_flag)

            # ===== Inference (computation of KGE scores) on Validation Set ====== #
            # FAKE
            fake_validation_scores = get_triples_scores2(trained_kge_model=my_pykeen_model,
                                                         triples=validation_100_fake_records,
                                                         entities_label_id_map=entities_label_id_map,
                                                         relation_label_id_map=relations_label_id_map)
            fake_validation_scores_center = get_center(scores=fake_validation_scores,
                                                       use_median=use_median_flag)
            # REAL
            real_validation_scores = get_triples_scores2(trained_kge_model=my_pykeen_model,
                                                         triples=validation_100_real_records,
                                                         entities_label_id_map=entities_label_id_map,
                                                         relation_label_id_map=relations_label_id_map)

            real_validation_scores_center = get_center(scores=real_validation_scores,
                                                       use_median=use_median_flag)
            # checks on validation scores
            if noise_level != TOTAL_RANDOM:
                assert real_validation_scores_center > fake_validation_scores_center
                assert training_scores_center > fake_validation_scores_center

            # ===== Inference (computation of KGE scores) on Testing Set ====== #
            # FAKE
            fake_testing_scores = get_triples_scores2(trained_kge_model=my_pykeen_model,
                                                      triples=testing_100_fake_records,
                                                      entities_label_id_map=entities_label_id_map,
                                                      relation_label_id_map=relations_label_id_map)
            fake_testing_scores_center = get_center(scores=fake_testing_scores,
                                                    use_median=use_median_flag)
            # REAL
            real_testing_scores = get_triples_scores2(trained_kge_model=my_pykeen_model,
                                                      triples=testing_100_real_records,
                                                      entities_label_id_map=entities_label_id_map,
                                                      relation_label_id_map=relations_label_id_map)
            real_testing_scores_center = get_center(scores=real_testing_scores,
                                                    use_median=use_median_flag)

            # print statistics
            print_2d_statistics(scores_matrix=[
                training_scores_vector,
                real_validation_scores,
                real_testing_scores,
                fake_validation_scores,
                fake_testing_scores,
            ],
                labels=[
                    "original training scores",
                    "REAL validation scores",
                    "REAL testing scores",
                    "FAKE validation scores",
                    "FAKE testing scores",
                ],
                decimal_precision=n_round)

            # check on testing scores
            if noise_level != TOTAL_RANDOM:
                assert real_testing_scores_center > fake_testing_scores_center
                assert training_scores_center > fake_testing_scores_center

            # compute classification metrics
            threshold = \
                fake_validation_scores_center + ((real_validation_scores_center - fake_validation_scores_center) / 2)
            print(f"classification threshold: {threshold}")
            if noise_level != TOTAL_RANDOM:
                assert threshold < training_scores_center
                assert threshold < real_validation_scores_center
                assert threshold < real_testing_scores_center
                assert threshold > fake_validation_scores_center
                assert threshold > fake_testing_scores_center
            y_true = [1 for _ in real_testing_scores] + [0 for _ in fake_testing_scores]
            y_pred = [1 if y >= threshold else 0 for y in real_testing_scores] + \
                     [1 if y >= threshold else 0 for y in fake_testing_scores]
            assert len(y_pred) == len(y_true)
            assert sum(y_true) == len(real_testing_scores)
            assert sum(y_pred) <= len(y_true)
            assert sum(y_pred) >= 0
            accuracy = round(metrics.accuracy_score(y_true=y_true, y_pred=y_pred), n_round)
            print("accuracy:", accuracy)
            f1_macro = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
            print("f1:", f1_macro)
            f1_pos = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1), n_round)
            print("f1_pos:", f1_pos)
            f1_neg = round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=0), n_round)
            print("f1_neg:", f1_neg)
            precision = round(metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
            print("precision:", precision)
            recall = round(metrics.recall_score(y_true=y_true, y_pred=y_pred, average="macro"), n_round)
            print("recall:", recall)

            # compute distance among the two distribution (greater is better)
            maximum = np.max(training_scores_vector)
            minimum = np.min(fake_testing_scores)
            if real_testing_scores_center > fake_testing_scores_center:
                distance = abs(real_testing_scores_center - fake_testing_scores_center) / abs(maximum - minimum)
                distance = round(distance, n_round)
                row[model_name] = distance
                print(f"distance: {distance}")
            else:
                distance = float('inf')
                row[model_name] = distance
                print("WARNING: real_testing_scores_center <= fake_testing_scores_center")

            # Compute Z-test (http://homework.uoregon.edu/pub/class/es202/ztest.html)
            # Z = (mean_1 - mean_2) / sqrt{ (std1/sqrt(N1))**2 + (std2/sqrt(N2))**2 }
            real_scores_error = (real_testing_scores.std() / (np.sqrt(real_testing_scores.shape[0]))) ** 2
            fake_scores_error = (fake_testing_scores.std() / (np.sqrt(fake_testing_scores.shape[0]))) ** 2
            Z_statistic = round(
                (real_testing_scores.mean() - fake_testing_scores.mean()) /
                np.sqrt(real_scores_error + fake_scores_error),
                2)
            print(f"Z-statistic: {round(Z_statistic, n_round)}")

            # Update current record diz
            current_record = dict()
            if noise_level == ORIGINAL:
                noise_level_k = ""
            else:
                noise_level_k = noise_level
            if F1_MACRO in selected_metrics:
                current_record[f"{F1_MACRO}_{noise_level_k}".rstrip("_")] = f1_macro
            if F1_NEG in selected_metrics:
                current_record[f"{F1_NEG}_{noise_level_k}".rstrip("_")] = f1_neg
            if F1_POS in selected_metrics:
                current_record[f"{F1_POS}_{noise_level_k}".rstrip("_")] = f1_pos
            if NORM_DIST in selected_metrics:
                current_record[f"{NORM_DIST}_{noise_level_k}".rstrip("_")] = distance
            if Z_STAT in selected_metrics:
                current_record[f"{Z_STAT}_{noise_level_k}".rstrip("_")] = Z_statistic

            # Update external records diz
            if model_name not in records:
                records[model_name] = current_record  # insert new record
            else:
                records[model_name] = {**records[model_name], **current_record}  # update (merge)

            # plot confidence intervals
            if plot_confidence_flag:
                title_fig = f"{model_name}_{noise_level}"
                out_path_fig = os.path.join(dataset_results_folder_path, f"{title_fig}.png")
                draw_distribution_plot(
                    label_values_map={
                        "real": list(real_testing_scores),
                        "threshold": threshold,
                        "fake": list(fake_testing_scores),
                    },
                    title=title_fig,
                    orient="v",
                    palette="Set3",
                    show_flag=False,
                    out_path=out_path_fig
                )

            print("\n")

    # Summarize and export the results after the end of iterations
    print("\n\n\n >>> Build DataFrame...")
    df_results = pd.DataFrame(data=list(records.values()),
                              index=list(records.keys())).T
    print(df_results)

    # Format for a better view the dataframe with the link prediction performance metrics
    df_results = df_results.sort_index(inplace=False, axis=0, ascending=True)
    diz_results_records = df_results.to_dict(orient="records")
    diz_results_index = list(df_results.index.values)
    step = len(selected_noise_levels)
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
    out_path = os.path.join(dataset_results_folder_path, f"triple_classification_results.xlsx")
    print(f"\t out_path: {out_path}")
    assert dataset_name in out_path
    assert out_path.endswith("triple_classification_results.xlsx")
    if (os.path.isfile(out_path)) and (not force_saving):
        raise OSError(f"'{out_path}' already exists!")
    df_results2.to_excel(out_path, header=True, index=True, encoding="utf-8", engine="openpyxl")
