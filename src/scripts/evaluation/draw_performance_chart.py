import configparser
import os
from pprint import pprint

import pandas as pd

from src.config.config import FB15K237, WN18RR, CODEXSMALL, \
    MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10, \
    F1_MACRO, F1_POS, F1_NEG, NORM_DIST, Z_STAT, \
    ORIGINAL, NOISE_10, NOISE_20, NOISE_30, \
    FB15K237_RESULTS_FOLDER_PATH, WN18RR_RESULTS_FOLDER_PATH, CODEXSMALL_RESULTS_FOLDER_PATH, \
    TOTAL_RANDOM, CONVE
from src.utils.linear_plotting import plot_linear_chart
from src.utils.stats import find_minimum, find_maximum

# set pandas visualization options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

datasets_names_results_folder_map = {
    WN18RR: WN18RR_RESULTS_FOLDER_PATH,
    FB15K237: FB15K237_RESULTS_FOLDER_PATH,
    CODEXSMALL: CODEXSMALL_RESULTS_FOLDER_PATH,
}
for k, v in datasets_names_results_folder_map.items():
    print(f"datasets_name={k} | dataset_results_folder={v}")

all_noise_levels = {ORIGINAL, TOTAL_RANDOM, NOISE_10, NOISE_20, NOISE_30}
print(f"all_noise_levels: {all_noise_levels}")

TASK_METRICS_MAP = {
    "link_prediction": (
        MR,
        MRR,
        HITS_AT_1,
        HITS_AT_3,
        HITS_AT_5,
        HITS_AT_10,
    ),
    "link_pruning": (
        MR,
        MRR,
        HITS_AT_1,
        HITS_AT_3,
        HITS_AT_5,
        HITS_AT_10,
    ),
    "triple_classification": (
        F1_MACRO,
        F1_POS,
        F1_NEG,
        NORM_DIST,
        Z_STAT,
    ),
}


if __name__ == '__main__':

    config = configparser.ConfigParser()
    if not os.path.isfile('dataset_local.ini'):
        raise FileNotFoundError("Create your 'dataset_local.ini' file in the 'src.scripts.evaluation' package "
                                "starting from the 'dataset.ini' template!")
    config.read('dataset_local.ini')
    dataset_name = config['dataset_info']['dataset_name']
    n_round = 3
    board = 0.025

    dataset_results_folder_path = datasets_names_results_folder_map[dataset_name]
    assert dataset_name in dataset_results_folder_path

    out_folder = os.path.join(dataset_results_folder_path, "charts")
    print(f"out_folder: {out_folder}")
    if not os.path.isdir(out_folder):
        os.makedirs(name=out_folder)

    for task, metrics in TASK_METRICS_MAP.items():

        if task == "link_pruning":
            fn = "link_pruning_results.xlsx"
        elif task == "link_prediction":
            fn = "link_prediction_both_realistic_results.xlsx"
        elif task == "triple_classification":
            fn = "triple_classification_results.xlsx"
        else:
            raise ValueError(f"Invalid 'task' in 'dataset_local.ini': '{task}'!")

        for metric in metrics:

            print(f"\n\n{'-' * 80}")

            # special case 1
            if metric == HITS_AT_10:
                prefix = "hits_at_X"
                top_y_function = find_maximum
                deletion_flag = False
            # special case 2
            elif metric == MR:
                prefix = MR
                top_y_function = find_minimum
                deletion_flag = True
            # default case
            else:
                prefix = metric
                top_y_function = find_maximum
                deletion_flag = False

            print(f"\n{'*' * 80}")
            print(f"\t\t dataset_name: {dataset_name}")
            print(f"\t\t task: {task}")
            print(f"\t\t metric: {metric}")
            print(f"\t\t dataset_results_folder_path: {dataset_results_folder_path}")
            print(f"\t\t file_name: {fn}")
            print(f"\t\t my_decimal_precision: {n_round}")
            print(f"{'*' * 80}\n")

            print("> Read performance excel file...")
            df_performance = pd.read_excel(os.path.join(dataset_results_folder_path, fn), engine="openpyxl")
            df_performance = df_performance.rename(columns={'Unnamed: 0': 'metric'})
            # print(df_performance)

            print("\n> Performance Parsing...")
            if deletion_flag:
                df_performance = df_performance.drop(
                    df_performance[df_performance['metric'].str.startswith("mrr")].index, axis=0)
            df_performance = df_performance[df_performance['metric'].str.startswith(prefix)]
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

            if dataset_name == FB15K237 and CONVE in res_diz:
                del res_diz[CONVE]

            print()
            pprint(res_diz)

            top_y = top_y_function(label_values_map=res_diz)
            print(f"\n - top_y: {top_y}")

            print("\n> Plotting...")
            out_file = os.path.join(out_folder, f"{dataset_name}_{task}_{metric}.png")
            print(f"out_file: {out_file}")
            plot_linear_chart(
                name_values_map=res_diz,
                title=f"Performance on {dataset_name} for {task.replace('_', ' ')} task",
                x_axis_name="noise",
                y_axis_name=metric,
                x_ticks=[0, 1, 2, 3],
                x_labels=["0%", "10%", "20%", "30%"],
                # axes_limits=(0 - board, 3 + board, 0 - 0.002, top_y + board),
                out_file_path=out_file,
                show_flag=False,
            )

            print(f"{'-' * 80} \n\n")
