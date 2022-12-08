import os
from typing import Dict

import numpy as np
import pandas as pd

from src.config.config import RESULTS_DIR

# set pandas visualization options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_ranking_diz(df: pd.DataFrame) -> Dict[str, float]:
    models_ranks_diz = {k: 0 for k in list(df.columns)}
    for record in df.to_dict('records'):
        sorted_record = dict(sorted(record.items(),
                                    key=lambda item: item[1],
                                    reverse=True))
        sorted_models = list(sorted_record.keys())
        for i, m in enumerate(sorted_models):
            val = (i + 1) / len(sorted_models)
            models_ranks_diz[m] += val
    sorted_models_ranks_diz = {k: round(v, 3)
                               for k, v in sorted(models_ranks_diz.items(),
                                                  key=lambda item: item[1])}
    print(sorted_models_ranks_diz, end="\n")
    return sorted_models_ranks_diz


# ====== Ranking by dataframe ===== #
d1 = os.path.join(RESULTS_DIR, "slopes", "codex_small_slopes.csv")
d2 = os.path.join(RESULTS_DIR, "slopes", "wn18rr_slopes.csv")
d3 = os.path.join(RESULTS_DIR, "slopes", "fb15k237_slopes.csv")
out_d = os.path.join(RESULTS_DIR, "slopes", "slopes_ranking_by_dataset.xlsx")

print("\n\n >>> Ranking by dataframe... \n")
df_d1 = pd.read_csv(d1, sep=";", encoding="utf-8").astype(float)
df_d2 = pd.read_csv(d2, sep=";", encoding="utf-8").astype(float)
df_d3 = pd.read_csv(d3, sep=";", encoding="utf-8").astype(float)
records_d = list()
records_d.append(get_ranking_diz(df=df_d1))
records_d.append(get_ranking_diz(df=df_d2))
records_d.append(get_ranking_diz(df=df_d3))
print("\n DF:")
df_dataset = pd.DataFrame(data=records_d, index=["CoDEx_Small", "WN18RR", "FB15k-237"])
df_dataset = df_dataset.reindex(sorted(df_dataset.columns), axis=1)
df_dataset.to_excel(out_d, engine="openpyxl")  # to_csv(out_d, sep=";", encoding="utf-8")
print(df_dataset)
# ========================================== #


# ===== Ranking by task ===== #
t1 = os.path.join(RESULTS_DIR, "slopes", "link_prediction_slopes.csv")
t2 = os.path.join(RESULTS_DIR, "slopes", "triple_clf_slopes.csv")
t3 = os.path.join(RESULTS_DIR, "slopes", "link_deletion_slopes.csv")
out_t = os.path.join(RESULTS_DIR, "slopes", "slopes_ranking_by_task.xlsx")

print("\n\n >>> Ranking by task... \n")
df_t1 = pd.read_csv(t1, sep=";", encoding="utf-8").replace("******", np.nan).astype(float)
df_t2 = pd.read_csv(t2, sep=";", encoding="utf-8").replace("******", np.nan).astype(float)
df_t3 = pd.read_csv(t3, sep=";", encoding="utf-8").replace("******", np.nan).astype(float)
records_t = list()
records_t.append(get_ranking_diz(df=df_t1))
records_t.append(get_ranking_diz(df=df_t2))
records_t.append(get_ranking_diz(df=df_t3))
print("\n DF:")
df_task = pd.DataFrame(data=records_t, index=["Link Prediction", "Triple Classification", "Link Deletion"])
df_task = df_task.reindex(sorted(df_task.columns), axis=1)
df_task.to_excel(out_t, engine="openpyxl")  # .to_csv(out_t, sep=";", encoding="utf-8")
print(df_task)
# ========================================== #
