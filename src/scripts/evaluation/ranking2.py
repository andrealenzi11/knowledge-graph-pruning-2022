import os
from statistics import mean
from typing import List, Dict, Optional

import pandas as pd

from src.config.config import RESULTS_DIR, \
    FB15K237_RESULTS_FOLDER_PATH, WN18RR_RESULTS_FOLDER_PATH, CODEXSMALL_RESULTS_FOLDER_PATH

# set pandas visualization options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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


def prep_df(df: pd.DataFrame, metrics_to_select: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df = df.rename(columns={'Unnamed: 0': 'metric'})
    if metrics_to_select:
        df = df[df['metric'].isin(metrics_to_select)].reset_index(drop=True)
    print(df, end="\n\n")
    return df


def compute_ranking(df: pd.DataFrame, round_digits: int = 3) -> Dict[str, float]:
    curr_metrics = list(df["metric"].values)
    df = df.drop("metric", axis=1)
    keys = list(df.columns)
    rank_dictionaries = []
    # === compute ranking for each row of df === #
    for idx, m in zip(range(df.shape[0]), curr_metrics):
        if str(m).startswith("mr_"):
            reverse_flag = False  # low values are better
        else:
            reverse_flag = True  # high values are better
        r_diz = get_ranking_diz(labels=keys,
                                values=list(df.iloc[idx, :].values),
                                reverse=reverse_flag)
        rank_dictionaries.append(r_diz)
    # === aggregation === #
    res = dict()
    for key in keys:
        res[key] = round(mean([d[key] for d in rank_dictionaries]), round_digits)
    sorted_res = dict(sorted(res.items(), key=lambda item: item[1], reverse=False))
    print(sorted_res, end="\n\n")
    return sorted_res


if __name__ == '__main__':
    lp_fn = "link_prediction_both_realistic_results.xlsx"
    tc_fn = "triple_classification_results.xlsx"
    ld_fn = "link_pruning_results.xlsx"

    ranking_metrics = [
        "hits_at_X",
        # "hits_at_X_noise_10",
        # "hits_at_X_noise_20",
        # "hits_at_X_noise_30",
        "mr",
        # "mr_noise_10",
        # "mr_noise_20",
        # "mr_noise_30",
        "mrr",
        # "mrr_noise_10",
        # "mrr_noise_20",
        # "mrr_noise_30",
    ]

    clf_metrics = [
        "f1_macro",
        # "f1_macro_noise_10",
        # "f1_macro_noise_20",
        # "f1_macro_noise_30",
        "norm_dist",
        # "norm_dist_noise_10",
        # "norm_dist_noise_20",
        # "norm_dist_noise_30",
    ]

    # ===== CODEX SMALL ===== #
    print("\n\n >>>>>>>>>> CODEX SMALL <<<<<<<<<<", end="\n\n")
    # > Link Prediction
    df_codexs_lp = pd.read_excel(os.path.join(CODEXSMALL_RESULTS_FOLDER_PATH, lp_fn), engine="openpyxl")
    df_codexs_lp = prep_df(df=df_codexs_lp, metrics_to_select=ranking_metrics)
    # > Triple Classification
    df_codexs_tc = pd.read_excel(os.path.join(CODEXSMALL_RESULTS_FOLDER_PATH, tc_fn), engine="openpyxl")
    df_codexs_tc = prep_df(df=df_codexs_tc, metrics_to_select=clf_metrics)
    # > Link Deletion
    df_codexs_ld = pd.read_excel(os.path.join(CODEXSMALL_RESULTS_FOLDER_PATH, ld_fn), engine="openpyxl")
    df_codexs_ld = prep_df(df=df_codexs_ld, metrics_to_select=ranking_metrics)

    # ===== WN18RR ===== #
    print("\n\n >>>>>>>>>> WN18RR <<<<<<<<<<", end="\n\n")
    # > Link Prediction
    df_wn18rr_lp = pd.read_excel(os.path.join(WN18RR_RESULTS_FOLDER_PATH, lp_fn), engine="openpyxl")
    df_wn18rr_lp = prep_df(df=df_wn18rr_lp, metrics_to_select=ranking_metrics)
    # > Triple Classification
    df_wn18rr_tc = pd.read_excel(os.path.join(WN18RR_RESULTS_FOLDER_PATH, tc_fn), engine="openpyxl")
    df_wn18rr_tc = prep_df(df=df_wn18rr_tc, metrics_to_select=clf_metrics)
    # > Link Deletion
    df_wn18rr_ld = pd.read_excel(os.path.join(WN18RR_RESULTS_FOLDER_PATH, ld_fn), engine="openpyxl")
    df_wn18rr_ld = prep_df(df=df_wn18rr_ld, metrics_to_select=ranking_metrics)

    # ===== FB15K237 ===== #
    print("\n\n >>>>>>>>>> FB15K237 <<<<<<<<<<", end="\n\n")
    # > Link Prediction
    df_fb15k237_lp = pd.read_excel(os.path.join(FB15K237_RESULTS_FOLDER_PATH, lp_fn), engine="openpyxl")
    df_fb15k237_lp = prep_df(df=df_fb15k237_lp, metrics_to_select=ranking_metrics)
    # > Triple Classification
    df_fb15k237_tc = pd.read_excel(os.path.join(FB15K237_RESULTS_FOLDER_PATH, tc_fn), engine="openpyxl")
    df_fb15k237_tc = prep_df(df=df_fb15k237_tc, metrics_to_select=clf_metrics)
    # > Link Deletion
    df_fb15k237_ld = pd.read_excel(os.path.join(FB15K237_RESULTS_FOLDER_PATH, ld_fn), engine="openpyxl")
    df_fb15k237_ld = prep_df(df=df_fb15k237_ld, metrics_to_select=ranking_metrics)

    # ===== Aggregate based on task ===== #
    print("\n\n\n >>>>> Aggregation based on task...")

    print("> Link Prediction")
    lp_df = pd.concat(objs=[df_codexs_lp, df_wn18rr_lp, df_fb15k237_lp],
                      axis=0, ignore_index=True, verify_integrity=True).reset_index(drop=True)
    lp_row = compute_ranking(df=lp_df)

    print("> Triple Classification")
    tc_df = pd.concat(objs=[df_codexs_tc, df_wn18rr_tc, df_fb15k237_tc],
                      axis=0, ignore_index=True, verify_integrity=True).reset_index(drop=True)
    tc_row = compute_ranking(df=tc_df)

    print("> Link Deletion")
    ld_df = pd.concat(objs=[df_codexs_ld, df_wn18rr_ld, df_fb15k237_ld],
                      axis=0, ignore_index=True, verify_integrity=True).reset_index(drop=True)
    ld_row = compute_ranking(df=ld_df)

    aggr_df1 = pd.DataFrame(index=["link_prediction", "triple_classification", "link_deletion"],
                            data=[lp_row, tc_row, ld_row])
    aggr_df1 = aggr_df1.reindex(sorted(aggr_df1.columns), axis=1)
    aggr_df1.to_excel(os.path.join(RESULTS_DIR, "aggregation_based_on_task.xlsx"),
                      engine="openpyxl", index=True)
    print(aggr_df1)

    # ===== Aggregate based on dataset ===== #
    print("\n\n\n >>>>> Aggregation based on dataset...")

    print("> Codex Small")
    codexs_df = pd.concat(objs=[df_codexs_lp, df_codexs_tc, df_codexs_ld],
                          axis=0, ignore_index=True, verify_integrity=True).reset_index(drop=True)
    codexs_row = compute_ranking(df=codexs_df)

    print("> WN18RR")
    wn18rr_df = pd.concat(objs=[df_wn18rr_lp, df_wn18rr_tc, df_wn18rr_ld],
                          axis=0, ignore_index=True, verify_integrity=True).reset_index(drop=True)
    wn18rr_row = compute_ranking(df=wn18rr_df)

    print("> FB15K237")
    fb15k237_df = pd.concat(objs=[df_fb15k237_lp, df_fb15k237_tc, df_fb15k237_ld],
                            axis=0, ignore_index=True, verify_integrity=True).reset_index(drop=True)
    fb15k237_row = compute_ranking(df=fb15k237_df)

    aggr_df2 = pd.DataFrame(index=["CoDeX_Small", "WN18RR", "FB15k237"],
                            data=[codexs_row, wn18rr_row, fb15k237_row])
    aggr_df2 = aggr_df2.reindex(sorted(aggr_df2.columns), axis=1)
    aggr_df2.to_excel(os.path.join(RESULTS_DIR, "aggregation_based_on_dataset.xlsx"),
                      engine="openpyxl", index=True)
    print(aggr_df2)
