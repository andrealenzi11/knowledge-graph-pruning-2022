import os

import numpy as np
import pandas as pd
import torch
from pykeen.models.predict import predict_triples_df

from config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    ORIGINAL, NOISE_30, NOISE_10, NOISE_20, EXPERIMENT_2_DIR
from core.pykeen_wrapper import get_train_test_validation
from dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader
from utils.confidence_intervals_plotting import plot_confidences_intervals

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

all_datasets_names = {COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS}
print(f"all_datasets_names: {all_datasets_names}")

# PRECISION_NEG = "precision_negatives"
# PRECISION_POS = "precision_positives"
# RECALL_NEG = "recall_negatives"
# RECALL_POS = "recall_positives"
# F1_NEG = "f1_negatives"
# F1_POS = "f1_positives"
# F1_MACRO = "f1_macro"
# ACCURACY = "accuracy"


if __name__ == '__main__':

    # Specify a Valid option: COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS
    dataset_name: str = COUNTRIES
    force_saving_flag = True
    plot_confidence_flag = False
    use_median_flag = False

    dataset_models_folder_path = DatasetPathFactory(dataset_name=dataset_name).get_models_folder_path()

    print(f"\n{'*' * 80}")
    print("PERFORMANCE TABLE GENERATION - CONFIGURATION")
    print(f"\t\t dataset_name: {dataset_name}")
    print(f"\t\t dataset_models_folder_path: {dataset_models_folder_path}")
    print(f"{'*' * 80}\n\n")

    # ========== noisy 30 dataset ========== #
    datasets_loader_30 = TsvDatasetLoader(dataset_name=dataset_name, noise_level=NOISE_30)
    # paths
    training_30_path, validation_30_path, testing_30_path = \
        datasets_loader_30.get_training_validation_testing_dfs_paths(noisy_test_flag=True)
    # dfs
    training_30_df, validation_30_df, testing_30_df = \
        datasets_loader_30.get_training_validation_testing_dfs(noisy_test_flag=True)
    # y_fakes
    training_30_y_fake, validation_30_y_fake, testing_30_y_fake = \
        datasets_loader_30.get_training_validation_testing_y_fakes()
    # x_test
    testing_30_records = testing_30_df.to_dict(orient="split")["data"]
    # triples factories
    training_30, testing_30, validation_30 = get_train_test_validation(training_set_path=training_30_path,
                                                                       test_set_path=testing_30_path,
                                                                       validation_set_path=validation_30_path,
                                                                       create_inverse_triples=False)

    # ========== original dataset ========== #
    datasets_loader_original = TsvDatasetLoader(dataset_name=dataset_name, noise_level=ORIGINAL)
    # paths
    training_original_path, validation_original_path, testing_original_path = \
        datasets_loader_original.get_training_validation_testing_dfs_paths(noisy_test_flag=False)
    # triples factories
    training_original, testing_original, validation_original = \
        get_train_test_validation(training_set_path=training_original_path,
                                  test_set_path=testing_original_path,
                                  validation_set_path=validation_original_path,
                                  create_inverse_triples=False)

    records = []
    indexes = []

    for noise_level in [
        ORIGINAL,
        # NOISE_5,
        NOISE_10,
        # NOISE_15,
        NOISE_20,
        NOISE_30,
    ]:
        print(f"\n\n#################### {noise_level} ####################\n")
        in_folder_path = os.path.join(dataset_models_folder_path, noise_level)

        datasets_loader = TsvDatasetLoader(dataset_name=dataset_name, noise_level=noise_level)
        training_path, validation_path, testing_path = \
            datasets_loader.get_training_validation_testing_dfs_paths(noisy_test_flag=True)

        training, testing, validation = get_train_test_validation(training_set_path=training_path,
                                                                  test_set_path=testing_path,
                                                                  validation_set_path=validation_path,
                                                                  create_inverse_triples=False)
        row = {}
        for model_name in sorted(os.listdir(in_folder_path)):

            print(f"\n >>>>> model_name: {model_name}")
            in_file = os.path.join(in_folder_path, model_name, "trained_model.pkl")
            print(in_file)

            # if model wa not already trained, skip to the next iteration
            if not os.path.isfile(in_file):
                print("model not present! \n")
                continue

            # Load model from FS
            my_pykeen_model = torch.load(in_file)

            # Compute KGE scores on original training set (dataset with NO noise)
            training_scores_tensor = my_pykeen_model.score_hrt(hrt_batch=training_original.mapped_triples, mode=None)
            training_scores_vector = training_scores_tensor.cpu().detach().numpy().reshape(-1)
            if use_median_flag:
                training_scores_center = np.median(a=training_scores_vector)
            else:
                training_scores_center = np.mean(a=training_scores_vector)
            print("training scores:",
                  f"shape={training_scores_vector.shape}",
                  np.min(a=training_scores_vector),
                  np.percentile(a=training_scores_vector, q=25),
                  np.median(a=training_scores_vector),
                  np.percentile(a=training_scores_vector, q=75),
                  np.max(a=training_scores_vector))

            # Compute KGE scores on test set (dataset with 30% noise)
            fake_scores, real_scores = [], []
            predicted_scores, y_true = [], []
            for record_30, y_fake_value in zip(testing_30_records, testing_30_y_fake):
                res: pd.DataFrame = predict_triples_df(
                    model=my_pykeen_model,
                    triples=record_30,
                    triples_factory=testing_30,
                    batch_size=None,
                    mode=None,  # "testing",
                )
                score = float(res["score"][0])
                predicted_scores.append(score)
                y_fake_value = int(y_fake_value)
                y_true.append(y_fake_value)
                if y_fake_value == 1:
                    fake_scores.append(score)
                elif y_fake_value == 0:
                    real_scores.append(score)
                else:
                    raise ValueError(f"Invalid fake value '{score}'!")

            # Analyze KGE scores on test set
            # fake
            fake_scores = np.array(fake_scores)
            if use_median_flag:
                fake_scores_center = float(np.median(a=fake_scores))
            else:
                fake_scores_center = float(np.mean(a=fake_scores))
            print("fake_scores:",
                  f"shape={fake_scores.shape}",
                  np.min(a=fake_scores),
                  np.percentile(a=fake_scores, q=25),
                  np.median(a=fake_scores),
                  np.percentile(a=fake_scores, q=75),
                  np.max(a=fake_scores))

            # real
            real_scores = np.array(real_scores)
            if use_median_flag:
                real_scores_center = float(np.median(a=real_scores))
            else:
                real_scores_center = float(np.mean(a=real_scores))
            print("real_scores:",
                  f"shape={real_scores.shape}",
                  np.min(a=real_scores),
                  np.percentile(a=real_scores, q=25),
                  np.median(a=real_scores),
                  np.percentile(a=real_scores, q=75),
                  np.max(a=real_scores))

            # compute distance (greater is better)
            if real_scores_center > fake_scores_center:
                distance = round(abs(real_scores_center - fake_scores_center), 4)
                row[model_name] = distance
                print(f"distance: {distance}")
            else:
                distance = float('inf')
                row[model_name] = distance
                print("WARNING: real_scores_center <= fake_scores_center")

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
                    percentile_min=5,
                    percentile_max=95,
                    z=1.645,  # 90%
                    round_digits=4)

            print("\n")

        records.append(row)
        indexes.append(noise_level)

    print("\n >>> Build DataFrame...")
    df_results = pd.DataFrame(data=records, index=indexes)
    print("\n>>> df overview:")
    print(df_results)

    print("\n >>> Export DataFrame to FS...")
    out_path = os.path.join(EXPERIMENT_2_DIR, f"{dataset_name}_results.xlsx")
    print(f"\t out_path: {out_path}")
    if (os.path.isfile(out_path)) and (not force_saving_flag):
        raise OSError(f"'{out_path}' already exists!")
    df_results.to_excel(out_path, header=True, index=True, encoding="utf-8", engine="openpyxl")
