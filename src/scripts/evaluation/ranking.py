import configparser
import os
from pprint import pprint
from typing import List, Tuple, Dict

import pandas as pd

from src.config.config import FB15K237, WN18RR, CODEXSMALL, \
    MR, MRR, HITS_AT_10, \
    F1_MACRO, NORM_DIST, ORIGINAL, NOISE_10, NOISE_20, NOISE_30, \
    FB15K237_RESULTS_FOLDER_PATH, WN18RR_RESULTS_FOLDER_PATH, CODEXSMALL_RESULTS_FOLDER_PATH, \
    TOTAL_RANDOM, CONVE

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

all_noise_levels = {
    ORIGINAL,
    TOTAL_RANDOM,
    NOISE_10,
    NOISE_20,
    NOISE_30,
}
print(f"all_noise_levels: {all_noise_levels}")

TASK_METRICS_MAP = {
    "link_prediction": (
        MR,
        MRR,
        # HITS_AT_1,
        # HITS_AT_3,
        # HITS_AT_5,
        HITS_AT_10,
    ),
    "link_pruning": (
        MR,
        MRR,
        # HITS_AT_1,
        # HITS_AT_3,
        # HITS_AT_5,
        HITS_AT_10,
    ),
    "triple_classification": (
        F1_MACRO,
        # F1_POS,
        # F1_NEG,
        NORM_DIST,
        # Z_STAT,
    ),
}


def get_ranking_list(labels: List[str],
                     values: List[float],
                     reverse: bool = True) -> List[Tuple[str, int, float]]:
    """
    Example:

    input:
        labels: ["a", "b", "c", "d"]
        values: [3.3, 1.1, 4.4, 2.2]
        reverse_flag: False

    output:
        [("b", 1, 1.1), ("d", 2, 2.2), ("a", 3, 3.3), ("c", 4, 4.4)]

    """
    assert len(labels) == len(values)
    sorted_values = sorted(values, reverse=reverse)
    result = []
    for index in range(len(values)):
        label = labels[index]
        value = values[index]
        position = sorted_values.index(value) + 1
        result.append((label, position, value))
    result.sort(key=lambda x: x[1])
    # return [(label, value) for (label, position, value) in result]
    return result


def get_ranking_diz(labels: List[str],
                    values: List[float],
                    reverse: bool = True) -> Dict[str, int]:
    """
    Example:

    input:
        labels: ["a", "b", "c", "d"]
        values: [3.3, 1.1, 4.4, 2.2]
        reverse_flag: False

    output:
        {"a": 3, "b": 1, "c": 4, "d": 2}
    """
    assert len(labels) == len(values)
    sorted_values = sorted(values, reverse=reverse)
    result = {}
    for index in range(len(values)):
        label = labels[index]
        value = values[index]
        position = sorted_values.index(value) + 1
        result[label] = position
    return result


if __name__ == '__main__':

    config = configparser.ConfigParser()
    if not os.path.isfile('dataset_local.ini'):
        raise FileNotFoundError("Create your 'dataset_local.ini' file in the 'src.scripts.evaluation' package "
                                "starting from the 'dataset.ini' template!")
    config.read('dataset_local.ini')
    dataset_name = config['dataset_info']['dataset_name']
    n_round = 3

    dataset_results_folder_path = datasets_names_results_folder_map[dataset_name]
    assert dataset_name in dataset_results_folder_path

    out_folder = os.path.join(dataset_results_folder_path, "ranking")
    print(f"out_folder: {out_folder}")
    if not os.path.isdir(out_folder):
        os.makedirs(name=out_folder)

    final_ranking_diz = {}

    for task, metrics in TASK_METRICS_MAP.items():

        print(f"\n\n ######################### {task} #########################")

        if task == "link_pruning":
            fn = "link_pruning_results.xlsx"
        elif task == "link_prediction":
            fn = "link_prediction_both_realistic_results.xlsx"
        elif task == "triple_classification":
            fn = "triple_classification_results.xlsx"
        else:
            raise ValueError(f"Invalid 'task' in 'dataset_local.ini': '{task}'!")

        reverse_flag = None
        task_ranking_diz = {}

        for metric in metrics:

            print(f"\n -------------------- {metric} --------------------")

            # special case 1
            if metric == HITS_AT_10:
                prefix = "hits_at_X"
                deletion_flag = False
                reverse_flag = True
            # special case 2
            elif metric == MR:
                prefix = MR
                deletion_flag = True
                reverse_flag = False
            # default case
            else:
                prefix = metric
                deletion_flag = False
                reverse_flag = True

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

            metric_df = pd.DataFrame(res_diz)
            metric_ranking_diz = {}
            for i, noise in zip(range(metric_df.shape[0]), ["0%", "10%", "20%", "30%"]):
                ranking_diz = get_ranking_diz(labels=list(metric_df.columns),
                                              values=list(metric_df.iloc[i, :].values),
                                              reverse=reverse_flag)
                metric_ranking_diz[noise] = ranking_diz

            task_ranking_diz[metric] = metric_ranking_diz

            print(f"\n --------------------------------------------- \n")

        print(">>> END of metrics loop! \n")

        pprint(task_ranking_diz)

        print(f"\n - {task}:")
        task_aggregation_diz = {
            '0%': {'AutoSF': 0, 'BoxE': 0, 'ComplEx': 0, 'ConvE': 0, 'DistMult': 0,
                   'HolE': 0, 'PairRE': 0, 'RotatE': 0, 'TransE': 0, 'TransH': 0},
            '10%': {'AutoSF': 0, 'BoxE': 0, 'ComplEx': 0, 'ConvE': 0, 'DistMult': 0,
                    'HolE': 0, 'PairRE': 0, 'RotatE': 0, 'TransE': 0, 'TransH': 0},
            '20%': {'AutoSF': 0, 'BoxE': 0, 'ComplEx': 0, 'ConvE': 0, 'DistMult': 0,
                    'HolE': 0, 'PairRE': 0, 'RotatE': 0, 'TransE': 0, 'TransH': 0},
            '30%': {'AutoSF': 0, 'BoxE': 0, 'ComplEx': 0, 'ConvE': 0, 'DistMult': 0,
                    'HolE': 0, 'PairRE': 0, 'RotatE': 0, 'TransE': 0, 'TransH': 0},
        }
        num_metrics = len(task_ranking_diz.keys())
        for metric, v1 in task_ranking_diz.items():
            for noise, v2 in v1.items():
                for model, rank in v2.items():
                    task_aggregation_diz[noise][model] += rank / num_metrics

        pprint(task_aggregation_diz)
        final_ranking_diz[task] = task_ranking_diz

        print(f"\n ######################################## \n\n")

    print(">>> END of tasks loop!")

    final_aggregation_diz = {
        '0%': {'AutoSF': 0, 'BoxE': 0, 'ComplEx': 0, 'ConvE': 0, 'DistMult': 0,
               'HolE': 0, 'PairRE': 0, 'RotatE': 0, 'TransE': 0, 'TransH': 0},
        '10%': {'AutoSF': 0, 'BoxE': 0, 'ComplEx': 0, 'ConvE': 0, 'DistMult': 0,
                'HolE': 0, 'PairRE': 0, 'RotatE': 0, 'TransE': 0, 'TransH': 0},
        '20%': {'AutoSF': 0, 'BoxE': 0, 'ComplEx': 0, 'ConvE': 0, 'DistMult': 0,
                'HolE': 0, 'PairRE': 0, 'RotatE': 0, 'TransE': 0, 'TransH': 0},
        '30%': {'AutoSF': 0, 'BoxE': 0, 'ComplEx': 0, 'ConvE': 0, 'DistMult': 0,
                'HolE': 0, 'PairRE': 0, 'RotatE': 0, 'TransE': 0, 'TransH': 0},
    }
    num_tasks = len(final_aggregation_diz.keys())
    for task, v1 in final_ranking_diz.items():
        num_metrics = len(v1.keys())
        for metric, v2 in v1.items():
            for noise, v3 in v2.items():
                for model, rank in v3.items():
                    final_aggregation_diz[noise][model] += rank / num_metrics / num_tasks

    pprint(final_aggregation_diz)

    print(f"\n### {dataset_name} ###")
    for noise, model_rank_map in final_aggregation_diz.items():
        print(f"> Noise = {noise}")
        sorted_model_rank_map = {k: v for k, v in sorted(model_rank_map.items(), key=lambda item: item[1])}
        for model, rank_score in sorted_model_rank_map.items():
            print(f"\t\t {model} : {round(rank_score, n_round)}")
