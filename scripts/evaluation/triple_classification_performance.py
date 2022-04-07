import os

import numpy as np
import pandas as pd
import torch
from pykeen.models.predict import predict_triples_df

from config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    ORIGINAL, NOISE_30, NOISE_10, NOISE_20
from core.pykeen_wrapper import get_train_test_validation
from dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader
from utils.confidence_intervals_plotting import plot_confidences_intervals

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

PRECISION_NEG = "precision_negatives"
PRECISION_POS = "precision_positives"
RECALL_NEG = "recall_negatives"
RECALL_POS = "recall_positives"
F1_NEG = "f1_negatives"
F1_POS = "f1_positives"
F1_MACRO = "f1_macro"
ACCURACY = "accuracy"

if __name__ == '__main__':

    force_saving = True

    # Specify a Valid option: COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS
    dataset_name: str = COUNTRIES

    selected_metrics = {
        PRECISION_NEG,
        PRECISION_POS,
        RECALL_NEG,
        RECALL_POS,
        F1_NEG,
        F1_POS,
        F1_MACRO,
        ACCURACY,
    }

    dataset_models_folder_path = DatasetPathFactory(dataset_name=dataset_name).get_models_folder_path()

    all_datasets_names = {COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS}
    all_metrics = {PRECISION_NEG, PRECISION_POS, RECALL_NEG, RECALL_POS, F1_NEG, F1_POS, F1_MACRO, ACCURACY}

    print(f"all_datasets_names: {all_datasets_names}")
    print(f"all_metrics: {all_metrics}")

    print(f"\n{'*' * 80}")
    print("PERFORMANCE TABLE GENERATION - CONFIGURATION")
    print(f"\t\t dataset_name: {dataset_name}")
    print(f"\t\t dataset_models_folder_path: {dataset_models_folder_path}")
    print(f"{'*' * 80}\n\n")

    # noisy 30
    datasets_loader_30 = TsvDatasetLoader(dataset_name=dataset_name, noise_level=NOISE_30)
    training_30_path, validation_30_path, testing_30_path = \
        datasets_loader_30.get_training_validation_testing_dfs_paths(noisy_test_flag=True)
    training_30, testing_30, validation_30 = get_train_test_validation(training_set_path=training_30_path,
                                                                       test_set_path=testing_30_path,
                                                                       validation_set_path=validation_30_path,
                                                                       create_inverse_triples=False)
    training_30_df, validation_30_df, testing_30_df = \
        datasets_loader_30.get_training_validation_testing_dfs(noisy_test_flag=True)
    training_30_y_fake, validation_30_y_fake, testing_30_y_fake = \
        datasets_loader_30.get_training_validation_testing_y_fakes()
    testing_30_records = testing_30_df.to_dict(orient="split")["data"]

    # original
    datasets_loader_original = TsvDatasetLoader(dataset_name=dataset_name, noise_level=ORIGINAL)
    training_original_path, validation_original_path, testing_original_path = \
        datasets_loader_original.get_training_validation_testing_dfs_paths(noisy_test_flag=False)
    training_original, testing_original, validation_original = \
        get_train_test_validation(training_set_path=training_original_path,
                                  test_set_path=testing_original_path,
                                  validation_set_path=validation_original_path,
                                  create_inverse_triples=False)

    records = {}
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

        for model_name in sorted(os.listdir(in_folder_path)):

            print(model_name)
            in_file = os.path.join(in_folder_path, model_name, "trained_model.pkl")
            print(in_file)

            if model_name == "RESCAL":
                continue

            # if model wa not already trained, skip to the next iteration
            if not os.path.isfile(in_file):
                print("model not present! \n")
                continue

            # Load model from FS
            my_pykeen_model = torch.load(in_file)

            # Compute KGE scores on training set
            try:
                training_scores_tensor = my_pykeen_model.score_hrt(hrt_batch=training_original.mapped_triples,
                                                                   mode="training")
            except Exception:
                training_scores_tensor = my_pykeen_model.score_hrt(hrt_batch=training_original.mapped_triples,
                                                                   mode=None)
            training_scores_vector = training_scores_tensor.cpu().detach().numpy().reshape(-1)
            training_scores_mean = np.mean(a=training_scores_vector)
            print("training scores:",
                  f"shape={training_scores_vector.shape}",
                  np.min(a=training_scores_vector),
                  np.percentile(a=training_scores_vector, q=2),
                  np.percentile(a=training_scores_vector, q=5),
                  np.percentile(a=training_scores_vector, q=10),
                  np.median(a=training_scores_vector),
                  np.percentile(a=training_scores_vector, q=90),
                  np.percentile(a=training_scores_vector, q=95),
                  np.percentile(a=training_scores_vector, q=98),
                  np.max(a=training_scores_vector))

            # Compute KGE scores on test set
            fake_scores, real_scores = [], []
            predicted_scores, y_true = [], []
            for record_30, y_fake_value in zip(testing_30_records, testing_30_y_fake):
                try:
                    res: pd.DataFrame = predict_triples_df(
                        model=my_pykeen_model,
                        triples=record_30,
                        triples_factory=testing_30,
                        batch_size=None,
                        mode=None,  # "testing",
                    )
                except Exception:
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
            fake_scores_mean = float(np.mean(a=fake_scores))
            fake_scores_min = float(np.min(a=fake_scores))
            assert training_scores_mean > fake_scores_mean
            distance_fake_from_training = abs(training_scores_mean - fake_scores_mean)
            print("fake_scores:",
                  f"shape={fake_scores.shape}",
                  np.min(a=fake_scores),
                  np.percentile(a=fake_scores, q=5),
                  np.median(a=fake_scores),
                  np.percentile(a=fake_scores, q=95),
                  np.max(a=fake_scores))

            # real
            real_scores = np.array(real_scores)
            real_scores_mean = float(np.mean(a=real_scores))
            assert training_scores_mean > real_scores_mean
            distance_real_from_training = abs(training_scores_mean - real_scores_mean)
            print("real_scores:",
                  f"shape={real_scores.shape}",
                  np.min(a=real_scores),
                  np.percentile(a=real_scores, q=5),
                  np.median(a=real_scores),
                  np.percentile(a=real_scores, q=95),
                  np.max(a=real_scores))
            # score and confidence intervals
            assert training_scores_mean > fake_scores_min
            maximum_distance = training_scores_mean - fake_scores_min
            print(distance_fake_from_training, distance_real_from_training)
            normalized_fake_dist = distance_fake_from_training * 100 / maximum_distance
            normalized_real_dist = distance_real_from_training * 100 / maximum_distance
            inverse_normalized_real_dist = 100. - normalized_real_dist
            print(normalized_fake_dist, normalized_real_dist)
            final_score = (normalized_fake_dist + inverse_normalized_real_dist) / 2.
            print(final_score)

            plot_confidences_intervals(
                label_values_map={
                    "training": sorted(list(training_scores_vector)),
                    "real": sorted(list(real_scores)),
                    "fake": sorted(list(fake_scores)),
                },
                title=f"{model_name}",
                use_median=True,
                use_mean=False,
                percentile_min=2,
                percentile_max=98,
                z=1.96,
                round_digits=4)
            print("\n")

    #         print(results_diz["metrics"])
    #         mr = round(results_diz["metrics"][strategy1][strategy2]["arithmetic_mean_rank"], 1)
    #         mrr = round(results_diz["metrics"][strategy1][strategy2]["inverse_harmonic_mean_rank"], 3)
    #         hits_at_1 = round(results_diz["metrics"][strategy1][strategy2]["hits_at_1"], 3)
    #         hits_at_3 = round(results_diz["metrics"][strategy1][strategy2]["hits_at_3"], 3)
    #         hits_at_5 = round(results_diz["metrics"][strategy1][strategy2]["hits_at_5"], 3)
    #         hits_at_10 = round(results_diz["metrics"][strategy1][strategy2]["hits_at_10"], 3)
    #
    #         current_record = dict()
    #         if MR in selected_metrics:
    #             current_record[f"{noise_level}_{MR}"] = mr
    #         if MRR in selected_metrics:
    #             current_record[f"{noise_level}_{MRR}"] = mrr
    #         if HITS_AT_1 in selected_metrics:
    #             current_record[f"{noise_level}_{HITS_AT_1}"] = hits_at_1
    #         if HITS_AT_3 in selected_metrics:
    #             current_record[f"{noise_level}_{HITS_AT_3}"] = hits_at_3
    #         if HITS_AT_5 in selected_metrics:
    #             current_record[f"{noise_level}_{HITS_AT_5}"] = hits_at_5
    #         if HITS_AT_10 in selected_metrics:
    #             current_record[f"{noise_level}_{HITS_AT_10}"] = hits_at_10
    #
    #         if model_name not in records:
    #             records[model_name] = current_record  # insert new record
    #         else:
    #             records[model_name] = {**records[model_name], **current_record}  # update (merge)
    #
    # print("\n >>> Build DataFrame...")
    # df_results = pd.DataFrame(data=list(records.values()),
    #                           index=list(records.keys())).T
    # print("\n>>> df info:")
    # print(df_results.info(memory_usage="deep"))
    # print("\n>>> df overview:")
    # print(df_results)
    #
    # print("\n >>> Export DataFrame to FS...")
    # out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{strategy1}_{strategy2}_results.xlsx")
    # print(f"\t out_path: {out_path}")
    # if (os.path.isfile(out_path)) and (not force_saving):
    #     raise OSError(f"'{out_path}' already exists!")
    # df_results.to_excel(out_path, header=True, index=True, encoding="utf-8", engine="openpyxl")
