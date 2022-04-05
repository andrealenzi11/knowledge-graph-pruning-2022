import os

import pandas as pd
import torch
from pykeen.models.predict import predict_triples_df

from config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, \
    RESULTS_DIR, ORIGINAL, NOISE_5, NOISE_10, NOISE_15, \
    MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10, \
    NOISE_20, NOISE_30, NATIONS
from core.pykeen_wrapper import get_train_test_validation
from dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader

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
    dataset_name: str = NATIONS

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

    records = {}
    for noise_level in [
        ORIGINAL,
        NOISE_5,
        NOISE_10,
        NOISE_15,
        NOISE_20,
        NOISE_30,
    ]:
        print(f"\n\n#################### {noise_level} ####################\n")
        in_folder_path = os.path.join(dataset_models_folder_path, noise_level)

        datasets_loader = TsvDatasetLoader(dataset_name=dataset_name, noise_level=noise_level)
        training_path, validation_path, testing_path = \
            datasets_loader.get_training_validation_testing_dfs_paths(noisy_test_flag=False)

        training, testing, validation = get_train_test_validation(training_set_path=training_path,
                                                                  test_set_path=testing_path,
                                                                  validation_set_path=validation_path,
                                                                  create_inverse_triples=False)

        for model_name in sorted(os.listdir(in_folder_path)):

            print(model_name, "\n")

            in_file = os.path.join(in_folder_path, model_name, "trained_model.pkl")

            if model_name in ["RESCAL", "ComplEx"]:
                continue

            # if model wa not already trained, skip to the next iteration
            if not os.path.isfile(in_file):
                continue

            # Load model from FS
            my_pykeen_model = torch.load(in_file)
            res = predict_triples_df(
                model=my_pykeen_model,
                triples=[
                    ("brazil", "embassy", "uk"),
                    ("cuba", "independence", "usa"),
                    ("indonesia", "militaryalliance", "usa"),
                    ("usa", "intergovorgs", "usa"),
                    ("jordan", "embassy", "brazil"),
                    ("uk", "relexports", "uk"),
                ],
                triples_factory=testing,
                batch_size=1,
                mode="testing",
            )
            print(res)
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
