import json
import os

import pandas as pd

from config import COUNTRIES, FB15K237, WN18RR, YAGO310, \
    COUNTRIES_MODELS_FOLDER_PATH, FB15K237_MODELS_FOLDER_PATH, WN18RR_MODELS_FOLDER_PATH, YAGO310_MODELS_FOLDER_PATH, \
    NOISE_1, NOISE_5, NOISE_10, RESULTS_DIR

if __name__ == '__main__':

    # Specify a Valid option: COUNTRIES, WN18RR, FB15K237, YAGO310
    DATASET_NAME: str = WN18RR
    STRATEGY1: str = "both"  # "both" | "head" | "tail"
    STRATEGY2: str = "realistic"  # "realistic" | "optimistic" | "pessimistic"

    if DATASET_NAME == COUNTRIES:
        DATASET_MODELS_FOLDER_PATH = COUNTRIES_MODELS_FOLDER_PATH
    elif DATASET_NAME == FB15K237:
        DATASET_MODELS_FOLDER_PATH = FB15K237_MODELS_FOLDER_PATH
    elif DATASET_NAME == WN18RR:
        DATASET_MODELS_FOLDER_PATH = WN18RR_MODELS_FOLDER_PATH
    elif DATASET_NAME == YAGO310:
        DATASET_MODELS_FOLDER_PATH = YAGO310_MODELS_FOLDER_PATH
    else:
        raise ValueError(f"Invalid dataset name: '{str(DATASET_NAME)}'! \n"
                         f"\t\t Specify one of the following values: \n"
                         f"\t\t [{COUNTRIES}, {FB15K237}, {WN18RR}, {YAGO310}] \n")

    if STRATEGY1 not in {"both", "head", "tail"}:
        raise ValueError(f"Invalid Strategy1 '{STRATEGY1}'!")
    if STRATEGY2 not in {"realistic", "optimistic", "pessimistic"}:
        raise ValueError(f"Invalid Strategy2 '{STRATEGY2}'!")

    print(f"\n{'*' * 80}")
    print(DATASET_NAME)
    print(DATASET_MODELS_FOLDER_PATH)
    print(f"{'*' * 80}\n\n")

    records = {}

    for noise_level in [
        NOISE_1,
        NOISE_5,
        NOISE_10,
    ]:
        print(f"\n\n#################### {noise_level} ####################\n")
        in_folder_path = os.path.join(DATASET_MODELS_FOLDER_PATH, noise_level)
        for model_name in sorted(os.listdir(in_folder_path)):
            in_file = os.path.join(in_folder_path, model_name, "results.json")
            if not os.path.isfile(in_file):
                continue
            with open(in_file, "r") as json_file:
                results_diz = json.load(json_file)
            mr = round(results_diz["metrics"]["arithmetic_mean_rank"][STRATEGY1][STRATEGY2], 1)
            mrr = round(results_diz["metrics"]["inverse_harmonic_mean_rank"][STRATEGY1][STRATEGY2], 3)
            hits_at_1 = round(results_diz["metrics"]["hits_at_k"][STRATEGY1][STRATEGY2]["1"], 3)
            hits_at_3 = round(results_diz["metrics"]["hits_at_k"][STRATEGY1][STRATEGY2]["3"], 3)
            hits_at_5 = round(results_diz["metrics"]["hits_at_k"][STRATEGY1][STRATEGY2]["5"], 3)
            hits_at_10 = round(results_diz["metrics"]["hits_at_k"][STRATEGY1][STRATEGY2]["10"], 3)
            current_record = {
                f"{noise_level}_MR": mr,
                f"{noise_level}_MRR": mrr,
                f"{noise_level}_hits1": hits_at_1,
                f"{noise_level}_hits3": hits_at_3,
                f"{noise_level}_hits5": hits_at_5,
                f"{noise_level}_hits10": hits_at_10,
            }
            if model_name not in records:
                records[model_name] = current_record    # insert new record
            else:
                records[model_name] = {**records[model_name], **current_record}  # update (merge)

    print("\n >>> Build DataFrame...")
    df_results = pd.DataFrame(data=list(records.values()),
                              index=list(records.keys())).T
    print("\n>>> df info:")
    print(df_results.info(memory_usage="deep"))
    print("\n>>> df overview:")
    print(df_results)

    print("\n >>> Export DataFrame to FS...")
    out_path = os.path.join(RESULTS_DIR, f"{DATASET_NAME}_{STRATEGY1}_{STRATEGY2}_results.xlsx")
    print(f"\t out_path: {out_path}")
    if os.path.isfile(out_path):
        raise OSError(f"'{out_path}' already exists!")
    df_results.to_excel(out_path, header=True, index=True, encoding="utf-8", engine="openpyxl")
