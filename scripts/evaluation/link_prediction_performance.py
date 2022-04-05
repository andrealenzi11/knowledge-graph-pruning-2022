import json
import os

import pandas as pd

from config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, \
    RESULTS_DIR, ORIGINAL, NOISE_5, NOISE_10, NOISE_15, \
    MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10, \
    BOTH_STRATEGY, HEAD_STRATEGY, TAIL_STRATEGY, \
    REALISTIC_STRATEGY, OPTIMISTIC_STRATEGY, PESSIMISTIC_STRATEGY, NOISE_20, NOISE_30, NATIONS
from dao.dataset_loading import DatasetPathFactory

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


if __name__ == '__main__':

    force_saving = True

    # Specify a Valid option: COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS
    dataset_name: str = COUNTRIES
    strategy1: str = BOTH_STRATEGY  # "both" | "head" | "tail"
    strategy2: str = REALISTIC_STRATEGY  # "realistic" | "optimistic" | "pessimistic"
    selected_metrics = {
        # MR,
        MRR,
        # HITS_AT_1,
        # HITS_AT_3,
        # HITS_AT_5,
        # HITS_AT_10,
    }

    if strategy1 not in {"both", "head", "tail"}:
        raise ValueError(f"Invalid Strategy1 '{strategy1}'!")
    if strategy2 not in {"realistic", "optimistic", "pessimistic"}:
        raise ValueError(f"Invalid Strategy2 '{strategy2}'!")

    dataset_models_folder_path = DatasetPathFactory(dataset_name=dataset_name).get_models_folder_path()

    all_datasets_names = {COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS}
    all_metrics = {MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10}
    all_strategies_1 = {BOTH_STRATEGY, HEAD_STRATEGY, TAIL_STRATEGY}
    all_strategies_2 = {REALISTIC_STRATEGY, OPTIMISTIC_STRATEGY, PESSIMISTIC_STRATEGY}
    print(f"all_datasets_names: {all_datasets_names}")
    print(f"all_metrics: {all_metrics}")
    print(f"all_strategies_1: {all_strategies_1}")
    print(f"all_strategies_2: {all_strategies_2}")

    print(f"\n{'*' * 80}")
    print("PERFORMANCE TABLE GENERATION - CONFIGURATION")
    print(f"\t\t dataset_name: {dataset_name}")
    print(f"\t\t dataset_models_folder_path: {dataset_models_folder_path}")
    print(f"\t\t strategy1: {strategy1}")
    print(f"\t\t strategy2: {strategy2}")
    print(f"{'*' * 80}\n\n")

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

        for model_name in sorted(os.listdir(in_folder_path)):
            in_file = os.path.join(in_folder_path, model_name, "results.json")
            if not os.path.isfile(in_file):
                continue
            with open(in_file, "r") as json_file:
                results_diz = json.load(json_file)

            print(results_diz["metrics"])
            mr = round(results_diz["metrics"][strategy1][strategy2]["arithmetic_mean_rank"], 1)
            mrr = round(results_diz["metrics"][strategy1][strategy2]["inverse_harmonic_mean_rank"], 3)
            hits_at_1 = round(results_diz["metrics"][strategy1][strategy2]["hits_at_1"], 3)
            hits_at_3 = round(results_diz["metrics"][strategy1][strategy2]["hits_at_3"], 3)
            hits_at_5 = round(results_diz["metrics"][strategy1][strategy2]["hits_at_5"], 3)
            hits_at_10 = round(results_diz["metrics"][strategy1][strategy2]["hits_at_10"], 3)

            current_record = dict()
            if MR in selected_metrics:
                current_record[f"{noise_level}_{MR}"] = mr
            if MRR in selected_metrics:
                current_record[f"{noise_level}_{MRR}"] = mrr
            if HITS_AT_1 in selected_metrics:
                current_record[f"{noise_level}_{HITS_AT_1}"] = hits_at_1
            if HITS_AT_3 in selected_metrics:
                current_record[f"{noise_level}_{HITS_AT_3}"] = hits_at_3
            if HITS_AT_5 in selected_metrics:
                current_record[f"{noise_level}_{HITS_AT_5}"] = hits_at_5
            if HITS_AT_10 in selected_metrics:
                current_record[f"{noise_level}_{HITS_AT_10}"] = hits_at_10

            if model_name not in records:
                records[model_name] = current_record  # insert new record
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
    out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{strategy1}_{strategy2}_results.xlsx")
    print(f"\t out_path: {out_path}")
    if (os.path.isfile(out_path)) and (not force_saving):
        raise OSError(f"'{out_path}' already exists!")
    df_results.to_excel(out_path, header=True, index=True, encoding="utf-8", engine="openpyxl")
