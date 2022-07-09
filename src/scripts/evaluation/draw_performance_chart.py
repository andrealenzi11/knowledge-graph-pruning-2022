import configparser
import os
from pprint import pprint

import pandas as pd

from src.config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10, \
    F1_MACRO, F1_POS, F1_NEG, NORM_DIST, Z_STAT, \
    ORIGINAL, NOISE_10, NOISE_20, NOISE_30, FB15K237_RESULTS_FOLDER_PATH, WN18RR_RESULTS_FOLDER_PATH, \
    YAGO310_RESULTS_FOLDER_PATH, \
    COUNTRIES_RESULTS_FOLDER_PATH, CODEXSMALL_RESULTS_FOLDER_PATH, NATIONS_RESULTS_FOLDER_PATH, TOTAL_RANDOM

# set pandas visualization options
from src.utils.linear_plotting import plot_linear_chart

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

all_metrics = {MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10, F1_MACRO, F1_POS, F1_NEG, NORM_DIST, Z_STAT}
print(f"all_metrics: {all_metrics}")

if __name__ == '__main__':

    config = configparser.ConfigParser()
    if not os.path.isfile('dataset_local.ini'):
        raise FileNotFoundError("Create your 'dataset_local.ini' file in the 'src.scripts.evaluation' package "
                                "starting from the 'dataset.ini' template!")
    config.read('dataset_local.ini')
    dataset_name = config['dataset_info']['dataset_name']
    task = config['performance']['task']
    n_round = 3
    if task == "link_pruning":
        # metric = MRR
        # prefix = "mrr"
        metric = HITS_AT_10
        prefix = "hits_at_X"
        fn = "link_pruning_results.xlsx"
    elif task == "link_prediction":
        metric = MRR
        prefix = "mrr"
        # metric = HITS_AT_10
        # prefix = "hits_at_X"
        fn = "link_prediction_both_realistic_results.xlsx"
    elif task == "triple_classification":
        # metric = F1_MACRO
        # prefix = "f1_macro"
        metric = NORM_DIST
        prefix = "norm_dist"
        fn = "triple_classification_results.xlsx"
    else:
        raise ValueError("Invalid 'task' in 'dataset_local.ini'!")

    dataset_results_folder_path = datasets_names_results_folder_map[dataset_name]
    assert dataset_name in dataset_results_folder_path

    print("\n> Drawing Chart - Configuration")
    print(f"\n{'*' * 80}")
    print(f"\t\t dataset_name: {dataset_name}")
    print(f"\t\t task: {task}")
    print(f"\t\t metric: {metric}")
    print(f"\t\t dataset_results_folder_path: {dataset_results_folder_path}")
    print(f"\t\t file_name: {fn}")
    print(f"\t\t my_decimal_precision: {n_round}")
    print(f"{'*' * 80}\n\n")

    print("\n> Read performance excel file...")
    df_performance = pd.read_excel(os.path.join(dataset_results_folder_path, fn), engine="openpyxl")
    df_performance = df_performance.rename(columns={'Unnamed: 0': 'metric'})
    print(df_performance)

    print("\n> Performance Parsing...")
    df_performance = df_performance[df_performance['metric'].str.startswith(prefix )]
    print(df_performance)
    res_diz = {}
    for model_name in list(df_performance.columns):
        res_diz[model_name] = list(df_performance[model_name].values)[0:-1]
    assert res_diz["metric"][0] == metric or (res_diz["metric"][0] == "hits_at_X" and metric == HITS_AT_10)
    assert res_diz["metric"][1].endswith(NOISE_10)
    assert res_diz["metric"][2].endswith(NOISE_20)
    assert res_diz["metric"][3].endswith(NOISE_30)
    # assert res_diz["metric"][4].endswith(TOTAL_RANDOM)
    del res_diz["metric"]
    print()
    pprint(res_diz)

    print("\n> Plotting...")
    plot_linear_chart(name_values_map=res_diz)




