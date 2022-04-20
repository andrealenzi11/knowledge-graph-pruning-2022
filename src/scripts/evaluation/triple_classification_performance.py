import os
from typing import List

import numpy as np
import pandas as pd
import torch
from pykeen.models.predict import predict_triples_df
from scipy.spatial.distance import jensenshannon
from sklearn import metrics

from src.config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    ORIGINAL, NOISE_10, NOISE_20, NOISE_30, NOISE_100, \
    FB15K237_RESULTS_FOLDER_PATH, WN18RR_RESULTS_FOLDER_PATH, YAGO310_RESULTS_FOLDER_PATH, \
    COUNTRIES_RESULTS_FOLDER_PATH, CODEXSMALL_RESULTS_FOLDER_PATH, NATIONS_RESULTS_FOLDER_PATH
from src.core.pykeen_wrapper import get_train_test_validation, print_partitions_info
from src.dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader
from src.utils.confidence_intervals_plotting import plot_confidences_intervals

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

all_noise_levels = {ORIGINAL, NOISE_10, NOISE_20, NOISE_30, NOISE_100}
print(f"all_noise_levels: {all_noise_levels}")


def get_data_records(kg_df: pd.DataFrame, y_fake_series: pd.Series, select_only_fake_flag: bool) -> List[List[str]]:
    assert kg_df.shape[1] == 3
    merged_kg_fake_df = pd.concat(objs=[kg_df.reset_index(drop=True), y_fake_series],
                                  axis=1,
                                  verify_integrity=True,
                                  ignore_index=True)
    assert merged_kg_fake_df.shape[0] == kg_df.shape[0] == y_fake_series.shape[0]
    assert merged_kg_fake_df.shape[1] == 4
    if select_only_fake_flag:
        fake_value = 1  # select only fake triples
    else:
        fake_value = 0  # select only real triples
    result_df = merged_kg_fake_df[merged_kg_fake_df.loc[:, 3] == fake_value]
    result_df = result_df.drop(3, axis=1).reset_index(drop=True)
    assert result_df.shape[0] <= merged_kg_fake_df.shape[0]
    assert result_df.shape[1] == 3
    result_records = result_df.to_dict(orient="split")["data"]
    assert all([len(record) == 3 for record in result_records])
    return result_records


def print_statistics(scores: np.ndarray,
                     decimal_precision: int,
                     message: str = "scores"):
    print(f"{message}:",
          round(np.min(a=scores), decimal_precision),
          round(np.percentile(a=scores, q=25), decimal_precision),
          round(np.median(a=scores), decimal_precision),
          round(np.percentile(a=scores, q=75), decimal_precision),
          round(np.max(a=scores), decimal_precision),
          f"shape={scores.shape}", )


if __name__ == '__main__':

    # Specify a Valid option: COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS
    dataset_name: str = CODEXSMALL
    force_saving_flag = True
    plot_confidence_flag = False
    use_median_flag = False
    my_decimal_precision = 4

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
    print(f"\t\t my_decimal_precision: {my_decimal_precision}")
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
    # fake testing records
    testing_100_fake_records = get_data_records(kg_df=testing_100_df,
                                                y_fake_series=testing_100_y_fake,
                                                select_only_fake_flag=True)
    print(f"\t - fake records size {len(testing_100_fake_records)}")
    assert len(testing_100_fake_records) == int(testing_100_df.shape[0] / 2)
    # real testing records
    testing_100_real_records = get_data_records(kg_df=testing_100_df,
                                                y_fake_series=testing_100_y_fake,
                                                select_only_fake_flag=False)
    print(f"\t - real records size {len(testing_100_real_records)}")
    assert len(testing_100_real_records) == int(testing_100_df.shape[0] / 2)
    # triples factories
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
    # triples factories
    training_original, testing_original, validation_original = \
        get_train_test_validation(training_set_path=training_original_path,
                                  test_set_path=testing_original_path,
                                  validation_set_path=validation_original_path,
                                  create_inverse_triples=False)
    print_partitions_info(training_triples=training_original,
                          training_triples_path=training_original_path,
                          validation_triples=validation_original,
                          validation_triples_path=validation_original_path,
                          testing_triples=testing_original,
                          testing_triples_path=testing_original_path)

    records = []
    indexes = []

    for noise_level in [
        ORIGINAL,
        NOISE_10,
        NOISE_20,
        NOISE_30,
    ]:
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
        row = {}
        for model_name in sorted(os.listdir(in_folder_path)):

            print(f"\n >>>>> model_name: {model_name}")
            in_file = os.path.join(in_folder_path, model_name, "trained_model.pkl")
            print(in_file)
            assert model_name in in_file

            # if model wa not already trained, skip to the next iteration
            if not os.path.isfile(in_file):
                print("model not present! \n")
                continue

            # Load model from FS
            my_pykeen_model = torch.load(in_file).cpu()

            # Compute KGE scores on original TRAINING set (dataset with NO noise)
            training_scores_tensor = my_pykeen_model.score_hrt(hrt_batch=training_original.mapped_triples,
                                                               mode=None)
            training_scores_vector = training_scores_tensor.cpu().detach().numpy().reshape(-1)
            if use_median_flag:
                training_scores_center = np.median(a=training_scores_vector)
            else:
                training_scores_center = np.mean(a=training_scores_vector)
            print_statistics(scores=training_scores_vector,
                             decimal_precision=my_decimal_precision,
                             message="    training scores")

            # Compute KGE scores on FAKE testing set
            fake_pred_df: pd.DataFrame = predict_triples_df(
                model=my_pykeen_model,
                triples=testing_100_fake_records,
                triples_factory=testing_100,
                batch_size=None,
                mode=None,  # "testing",
            )
            fake_scores = fake_pred_df["score"].values
            if use_median_flag:
                fake_scores_center = float(np.median(a=fake_scores))
            else:
                fake_scores_center = float(np.mean(a=fake_scores))

            # Compute KGE scores on REAL original testing set
            real_pred_df: pd.DataFrame = predict_triples_df(
                model=my_pykeen_model,
                triples=testing_100_real_records,
                triples_factory=testing_100,
                batch_size=None,
                mode=None,  # "testing",
            )
            real_scores = real_pred_df["score"].values
            if use_median_flag:
                real_scores_center = float(np.median(a=real_scores))
            else:
                real_scores_center = float(np.mean(a=real_scores))

            # check
            assert len(fake_scores) == len(real_scores)

            # Print some scores information
            print_statistics(scores=fake_scores,
                             decimal_precision=my_decimal_precision,
                             message="FAKE testing scores")
            print_statistics(scores=real_scores,
                             decimal_precision=my_decimal_precision,
                             message="REAL testing scores")

            # compute classification metrics
            assert real_scores_center > fake_scores_center
            threshold = fake_scores_center + ((real_scores_center - fake_scores_center) / 2)
            assert threshold < real_scores_center
            assert threshold > fake_scores_center
            print(threshold)
            y_true = [1 for _ in real_scores] + [0 for _ in fake_scores]
            y_pred = [1 if y >= threshold else 0 for y in real_scores] + \
                     [1 if y >= threshold else 0 for y in fake_scores]
            assert len(y_pred) == len(y_true)
            assert sum(y_true) == int(len(y_true) / 2)
            print("accuracy:",
                  round(metrics.accuracy_score(y_true=y_true, y_pred=y_pred), my_decimal_precision))
            print("f1:",
                  round(metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro"), my_decimal_precision))
            print("precision:",
                  round(metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro"), my_decimal_precision))
            print("recall:",
                  round(metrics.recall_score(y_true=y_true, y_pred=y_pred, average="macro"), my_decimal_precision))

            # compute distance among the two distribution (greater is better)
            maximum = np.max(training_scores_vector)
            minimum = np.min(fake_scores)
            if real_scores_center > fake_scores_center:
                distance = abs(real_scores_center - fake_scores_center) / abs(maximum - minimum)
                distance = round(distance, my_decimal_precision)
                row[model_name] = distance
                print(f"distance: {distance}")
            else:
                distance = float('inf')
                row[model_name] = distance
                print("WARNING: real_scores_center <= fake_scores_center")

            # Compute Jensen Shannon distance (square root of the Jensen Shannon divergence)
            js_dist = jensenshannon(real_scores, fake_scores, base=2)
            print(f"jensen shannon Distance (base 2): {round(js_dist, my_decimal_precision)}")

            # Compute Z-test (http://homework.uoregon.edu/pub/class/es202/ztest.html)
            # Z = (mean_1 - mean_2) / sqrt{ (std1/sqrt(N1))**2 + (std2/sqrt(N2))**2 }
            real_scores_error = (real_scores.std() / (np.sqrt(real_scores.shape[0]))) ** 2
            fake_scores_error = (fake_scores.std() / (np.sqrt(fake_scores.shape[0]))) ** 2
            Z_statistic = (real_scores.mean() - fake_scores.mean()) / np.sqrt(real_scores_error + fake_scores_error)
            print(f"Z-statistic: {round(Z_statistic, my_decimal_precision)}")

            # plot confidence intervals
            if plot_confidence_flag:
                use_mean_flag = not use_median_flag
                plot_confidences_intervals(
                    label_values_map={
                        "training": sorted(list(training_scores_vector)),
                        "real": sorted(list(real_scores)),
                        "fake": sorted(list(fake_scores)),
                    },
                    title=f"{model_name}",
                    use_median=use_median_flag,
                    use_mean=use_mean_flag,
                    percentile_min=0,
                    percentile_max=100,
                    z=1.645,  # 90%
                    round_digits=my_decimal_precision)

            print("\n")

        records.append(row)
        indexes.append(noise_level)

    print("\n >>> Build DataFrame...")
    df_results = pd.DataFrame(data=records, index=indexes)
    print("\n>>> df overview:")
    print(df_results)

    print("\n >>> Export DataFrame to FS...")
    out_path = os.path.join(dataset_results_folder_path, f"triple_classification_results.xlsx")
    assert dataset_name in out_path
    assert out_path.endswith("triple_classification_results.xlsx")
    print(f"\t out_path: {out_path}")
    if (os.path.isfile(out_path)) and (not force_saving_flag):
        raise OSError(f"'{out_path}' already exists!")
    df_results.to_excel(out_path, header=True, index=True, encoding="utf-8", engine="openpyxl")
